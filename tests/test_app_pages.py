"""Headless UI tests (no API calls): example claims, experiment page, export.

Chạy: python -m pytest tests/test_app_pages.py -q
"""
import json
from pathlib import Path

import pytest

from scidebate.history import record_to_markdown

APP = str(Path(__file__).resolve().parent.parent / "app.py")
st_testing = pytest.importorskip("streamlit.testing.v1")


def test_example_button_fills_claim():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Refuted").click().run()
    assert at.text_area(key="claim_input").value == "Activation of PPM1D enhances p53 function."


def test_experiment_page_renders(tmp_path, monkeypatch):
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Experiment Results").run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "Experiment Results" in page


def test_record_to_markdown():
    rec = {
        "id": "x", "created_at": "2026-10-04T10:00:00",
        "claim_original": "Hút thuốc gây ung thư phổi.", "claim_english": "Smoking causes lung cancer.",
        "settings": {"models": {"pro": "a", "con": "b", "judge": "c"}, "use_rag": True, "rag_source": "scifact", "rag_max_results": 3},
        "num_rounds": 2,
        "verdict": {"verdict": "SUPPORTED", "confidence": 0.9, "justification": "Strong.", "vi": "Mạnh."},
        "consensus": {"consensus_quadrant": "Aligned NEI", "consensus_level": "LOW",
                      "normalized_entropy": 0.1, "jsd": 0.05, "calibrated_confidence": 0.7},
        "transcript": [{"round_num": 1, "speaker": "PRO", "content": "Evidence [1].", "vi": "Bằng chứng [1].", "filtered_out": False}],
        "pro_papers": [{"title": "Cohort", "source": "SciFact", "doc_id": "42"}],
        "con_papers": [],
    }
    md = record_to_markdown(rec)
    for part in ["# SciDebate", "Smoking causes lung cancer.", "**SUPPORTED** — confidence 90%",
                 "Aligned NEI", "### Round 1 — PRO", "Bằng chứng [1].", "## Evidence — PRO", "Cohort"]:
        assert part in md, part
