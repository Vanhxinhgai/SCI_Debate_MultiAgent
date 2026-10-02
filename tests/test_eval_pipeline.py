"""Offline tests cho pipeline đánh giá (không gọi API thật).

Chạy: python -m pytest tests/test_eval_pipeline.py -q
"""
import json
import sys

import pytest

from scidebate.llms import BaseLLM, CachedLLM, LLMResponse
from scidebate.tools import scifact
from experiments import metrics
from experiments.baselines import CoTBaseline, ZeroShotBaseline, _parse_verdict_from_text


class FakeLLM(BaseLLM):
    """Trả lời cố định, đếm số lần gọi."""

    def __init__(self, model="fake-model", reply=None, fail_times=0):
        super().__init__(model=model, temperature=0.0, max_tokens=100)
        self.calls = 0
        self.fail_times = fail_times
        self.reply = reply or "Argument.\nVERDICT: REFUTED\nCONFIDENCE: 0.8\nJUSTIFICATION: fake."

    def generate(self, messages, **kwargs):
        assert isinstance(messages, list), "generate() must receive a message list"
        self.calls += 1
        if self.calls <= self.fail_times:
            err = RuntimeError("rate limited")
            err.status_code = 429
            raise err
        return LLMResponse(content=f"{self.reply} #{self.calls}", model=self.model)


@pytest.fixture
def scifact_dir(tmp_path, monkeypatch):
    corpus = [
        {"doc_id": 1, "title": "Aspirin trial", "abstract": ["Aspirin did not reduce events.", "No effect was found."]},
        {"doc_id": 2, "title": "Aspirin benefit", "abstract": ["Aspirin reduced cardiovascular events."]},
        {"doc_id": 3, "title": "Unrelated", "abstract": ["We know normal cells grow."]},
    ]
    claims = [
        {"id": 10, "claim": "Aspirin reduces cardiovascular events.", "evidence": {"1": [{"sentences": [0], "label": "CONTRADICT"}]}},
        {"id": 11, "claim": "Aspirin trial outcomes are positive.", "evidence": {"2": [{"sentences": [0], "label": "SUPPORT"}]}},
        {"id": 12, "claim": "Aspirin cures baldness.", "evidence": {}},
    ]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in corpus), encoding="utf-8")
    (tmp_path / "claims_dev.jsonl").write_text("\n".join(json.dumps(c) for c in claims), encoding="utf-8")
    monkeypatch.setenv("SCIFACT_DIR", str(tmp_path))
    scifact._load_scifact_dataset_cached.cache_clear()
    return tmp_path


# ── Retrieval: không lộ nhãn ──────────────────────────────────────────────────

def test_default_retrieval_does_not_use_gold(scifact_dir):
    res = scifact.retrieve_scifact_evidence("Aspirin reduces cardiovascular events.", stance="PRO")
    assert res["retrieval_mode"] == "lexical"
    for word in ("label=", "CONTRADICT", "gold"):
        assert word not in res["context"]


def test_gold_mode_gives_same_docs_to_both_sides_without_labels(scifact_dir):
    kw = dict(use_gold_if_exact_match=True)
    pro = scifact.retrieve_scifact_evidence("Aspirin reduces cardiovascular events.", stance="PRO", **kw)
    con = scifact.retrieve_scifact_evidence("Aspirin reduces cardiovascular events.", stance="CON", **kw)
    assert [p["doc_id"] for p in pro["papers"]] == [p["doc_id"] for p in con["papers"]] == ["1"]
    assert "CONTRADICT" not in pro["context"] and "CONTRADICT" not in con["context"]


def test_stance_rescore_matches_whole_words():
    doc = {"title": "x", "abstract": ["We know normal cells."]}
    (score, _), = scifact._stance_rescore([(1.0, doc)], "CON")
    assert score == 1.0  # "no"/"not" must not match "know"/"normal"


# ── Baselines ─────────────────────────────────────────────────────────────────

def test_parse_prefers_last_verdict_line():
    text = "Doc 1 SUPPORTED the idea, doc 2 REFUTED it.\nVerdict: INCONCLUSIVE\nConfidence: 0.4"
    assert _parse_verdict_from_text(text) == ("INCONCLUSIVE", 0.4)


def test_zeroshot_baseline_runs():
    r = ZeroShotBaseline(llm=FakeLLM()).run("Some claim.")
    assert r.verdict == "REFUTED" and r.confidence == 0.8


def test_cot_baseline_runs(scifact_dir):
    r = CoTBaseline(llm=FakeLLM(), source="scifact").run("Aspirin reduces cardiovascular events.")
    assert r.verdict == "REFUTED"


# ── CachedLLM ─────────────────────────────────────────────────────────────────

def test_cache_reuses_responses_across_runs(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    inner1 = FakeLLM()
    llm1 = CachedLLM(inner1, cache_path=tmp_path / "c.sqlite")
    first = [llm1.generate(msgs).content for _ in range(2)]
    assert first[0] != first[1]  # repeated identical requests stay distinct samples

    inner2 = FakeLLM()
    llm2 = CachedLLM(inner2, cache_path=tmp_path / "c.sqlite")
    second = [llm2.generate(msgs).content for _ in range(2)]
    assert second == first and inner2.calls == 0


def test_retry_on_429():
    inner = FakeLLM(fail_times=2)
    llm = CachedLLM(inner, base_delay=0.01, max_delay=0.01)
    assert llm.generate([{"role": "user", "content": "x"}]).content
    assert llm.stats["retries"] == 2


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_mcnemar_and_bootstrap():
    assert metrics.mcnemar_exact([True] * 10, [False] * 10)["p_value"] < 0.01
    assert metrics.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    lo, hi = metrics.bootstrap_ci(["SUPPORTED"] * 5, ["SUPPORTED"] * 5)
    assert lo == hi == 1.0


# ── End-to-end eval script ────────────────────────────────────────────────────

def test_scifact_eval_end_to_end_and_resume(scifact_dir, tmp_path, monkeypatch, capsys):
    from experiments import scifact_eval

    created = []

    def fake_make(self, cfg):
        inner = FakeLLM(model=cfg["model"])
        created.append(inner)
        return CachedLLM(inner, cache_path=self.cache_path)

    monkeypatch.setattr(scifact_eval.LLMFactory, "make", fake_make)
    monkeypatch.setattr(scifact_eval, "RESULTS_DIR", tmp_path / "results")
    argv = ["scifact_eval", "--n", "0", "--rag-source", "scifact"]
    monkeypatch.setattr(sys, "argv", argv)

    scifact_eval.main()
    ckpt = tmp_path / "results" / "scifact_dev_n3_seed42_scifact.jsonl"
    lines = [json.loads(l) for l in ckpt.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 * len(scifact_eval.ALL_SYSTEMS)
    assert all(not r["error"] for r in lines), [r["error"] for r in lines if r["error"]]
    gold = {r["claim_id"]: r["ground_truth"] for r in lines}
    assert gold == {"10": "REFUTED", "11": "SUPPORTED", "12": "INCONCLUSIVE"}

    summary = json.loads((tmp_path / "results" / "scifact_dev_n3_seed42_scifact_summary.json").read_text(encoding="utf-8"))
    assert summary["n_paired_claims"] == 3
    assert "mcnemar_vs_scidebate" in summary["systems"]["homo_mad"]

    # Resume: nothing left to run, no new API calls.
    calls_before = sum(l.calls for l in created)
    scifact_eval.main()
    assert sum(l.calls for l in created) == calls_before
    assert len(ckpt.read_text(encoding="utf-8").splitlines()) == len(lines)
