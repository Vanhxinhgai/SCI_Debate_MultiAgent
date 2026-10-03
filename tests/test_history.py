"""Tests cho lịch sử kiểm chứng (không gọi API).

Chạy: python -m pytest tests/test_history.py -q
"""
from pathlib import Path

import pytest

from scidebate.agents import Verdict
from scidebate.debate import DebateResult, Turn
from scidebate.history import build_record, delete_run, list_runs, load_run, save_run
from scidebate.uncertainty import ConsensusMetrics


def make_record():
    result = DebateResult(claim="Smoking causes lung cancer.")
    result.transcript = [
        Turn(1, "PRO", "Cohort studies show 15-30x risk [1].", "qwen"),
        Turn(1, "CON", "Correlation is not causation.", "nemotron", filtered_out=True, filter_reason="redundant"),
    ]
    result.verdict = Verdict("SUPPORTED", 0.9, "Strong causal evidence.", "VERDICT: SUPPORTED")
    result.num_rounds = 1
    result.elapsed_seconds = 42.0
    result.consensus = ConsensusMetrics(normalized_entropy=0.2, jsd=0.1,
                                        consensus_level="HIGH", consensus_quadrant="Strong Consensus")
    return build_record(
        result,
        original_claim="Hút thuốc gây ung thư phổi.",
        english_claim="Smoking causes lung cancer.",
        settings={"models": {"pro": "qwen", "con": "nemotron", "judge": "gpt-oss"}, "max_rounds": 1,
                  "use_rag": True, "rag_source": "arxiv", "rag_max_results": 3, "use_dar": True},
        pro_papers=[{"title": "Cohort study", "authors": "Doll", "published": "1950", "summary": "x", "pdf_link": ""}],
        con_papers=[],
        turn_translations=["Các nghiên cứu cohort cho thấy nguy cơ tăng 15-30 lần [1].", None],
        verdict_translation="Bằng chứng nhân quả mạnh.",
    )


def test_save_list_load_delete(tmp_path):
    rec = make_record()
    save_run(rec, tmp_path)

    runs = list_runs(tmp_path)
    assert [r["id"] for r in runs] == [rec["id"]]
    assert runs[0]["claim"] == "Hút thuốc gây ung thư phổi." and runs[0]["verdict"] == "SUPPORTED"

    loaded = load_run(rec["id"], tmp_path)
    assert loaded["transcript"][0]["vi"].startswith("Các nghiên cứu")
    assert loaded["transcript"][1]["filtered_out"] is True
    assert loaded["verdict"]["vi"] == "Bằng chứng nhân quả mạnh."
    assert loaded["consensus"]["consensus_quadrant"] == "Strong Consensus"

    assert delete_run(rec["id"], tmp_path) is True
    assert list_runs(tmp_path) == [] and load_run(rec["id"], tmp_path) is None


def test_corrupt_files_are_skipped(tmp_path):
    save_run(make_record(), tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(list_runs(tmp_path)) == 1
    assert load_run("broken", tmp_path) is None


def test_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        load_run("../../.env", tmp_path)
    with pytest.raises(ValueError):
        delete_run("..\\secret", tmp_path)


def test_list_is_newest_first(tmp_path):
    a, b = make_record(), make_record()
    a["created_at"], b["created_at"] = "2026-01-01T10:00:00", "2026-02-01T10:00:00"
    a["id"], b["id"] = "a", "b"
    save_run(a, tmp_path)
    save_run(b, tmp_path)
    assert [r["id"] for r in list_runs(tmp_path)] == ["b", "a"]


def test_saved_debate_view_in_app(tmp_path, monkeypatch):
    """Render the Streamlit app headlessly, open a saved debate from the sidebar, then delete it."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SCIDEBATE_HISTORY_DIR", str(tmp_path))
    rec = make_record()
    save_run(rec, tmp_path)

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not at.exception
    history_select = next(s for s in at.sidebar.selectbox if s.label == "Saved debates")
    history_select.select(rec["id"]).run()
    next(b for b in at.sidebar.button if b.label == "View").click().run()
    assert not at.exception

    page = "\n".join(m.value for m in at.markdown)
    assert "Saved debate" in page and "Hút thuốc gây ung thư phổi." in page
    assert "Các nghiên cứu cohort" in page          # turn translation
    assert "verdict-card supported" in page         # verdict card
    assert "Bằng chứng nhân quả mạnh." in page       # verdict translation
    assert "DAR FILTERED" in page                    # filtered turn kept

    history_select = next(s for s in at.sidebar.selectbox if s.label == "Saved debates")
    history_select.select(rec["id"]).run()
    next(b for b in at.sidebar.button if b.label == "Delete").click().run()
    assert not at.exception
    assert list_runs(tmp_path) == []
