"""Đánh giá retrieval trên toàn bộ SciFact dev (300 claim) — không gọi LLM.

So sánh bộ truy xuất cũ (TF-IDF + lọc theo từ khoá đầu tiên + stance boost ×(1+0.35k))
với bộ truy xuất hiện tại (BM25 + stance re-ranking có giới hạn).

Chỉ số: Recall@k = tỷ lệ claim (có evidence) mà ít nhất một tài liệu gold nằm trong top-k;
"Empty" = số claim không nhận được tài liệu nào.

Chạy: python -m experiments.retrieval_eval
Kết quả: experiments/results/retrieval_dev.json và retrieval_dev.md
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from scidebate.tools import scifact as sf

RESULTS_DIR = Path(__file__).parent / "results"
KS = (1, 3, 5, 10)


# ── Legacy retriever (reproduced from the code before the BM25 change) ────────

def _legacy_tfidf(claim: str, docs: dict) -> list[tuple[float, dict]]:
    query = Counter(sf._tokenize(claim))
    doc_terms, df, titles = sf._build_index(docs)
    n_docs = max(len(docs), 1)
    out = []
    for doc_id, counts in doc_terms.items():
        doc_len = sum(counts.values()) or 1
        score = 0.0
        for term, qtf in query.items():
            if term in counts:
                idf = math.log((n_docs + 1) / (df[term] + 0.5)) + 1.0
                score += qtf * idf * (counts[term] / doc_len) * (2.0 if term in titles[doc_id] else 1.0)
        if score > 0:
            out.append((score, docs[doc_id]))
    return sorted(out, key=lambda x: x[0], reverse=True)


def _legacy_stance(scored, stance):
    terms = sf._PRO_BOOST_TERMS if stance == "PRO" else sf._CON_BOOST_TERMS
    out = []
    for s, d in scored:
        text = f"{d.get('title', '')} {sf._abstract_text(d)}".lower()
        out.append((s * (1 + 0.35 * sum(1 for t in terms if t in text)), d))  # substring match (old bug)
    return sorted(out, key=lambda x: x[0], reverse=True)


def legacy_retrieve(claim: str, docs: dict, stance: str, k: int) -> list[str]:
    scored = _legacy_stance(_legacy_tfidf(claim, docs), stance) if stance != "NEUTRAL" else _legacy_tfidf(claim, docs)
    tokens = sf._tokenize(claim)
    if tokens:
        first = tokens[0]
        scored = [(s, d) for s, d in scored if first in sf._tokenize(f"{d.get('title', '')} {sf._abstract_text(d)}")]
    return [str(d.get("doc_id")) for _, d in scored[:k]]


def current_retrieve(claim: str, docs: dict, stance: str, k: int) -> list[str]:
    scored = sf._score_docs(claim, docs)
    if stance != "NEUTRAL":
        scored = sf._stance_rescore(scored, stance)
    return [str(d.get("doc_id")) for _, d in scored[:k]]


def evaluate(fn, claims, docs, stance) -> dict:
    kmax = max(KS)
    hits = {k: 0 for k in KS}
    empty = 0
    with_evidence = [c for c in claims if c.get("evidence")]
    for c in claims:
        ids = fn(c["claim"], docs, stance, kmax)
        if not ids:
            empty += 1
        gold = set(map(str, c.get("evidence") or {}))
        if gold:
            for k in KS:
                hits[k] += bool(set(ids[:k]) & gold)
    n = len(with_evidence)
    return {"n_claims": len(claims), "n_with_evidence": n, "empty": empty,
            **{f"recall@{k}": round(hits[k] / n, 4) for k in KS}}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = sf.load_scifact_dataset()
    docs = ds["corpus"]
    dev = [r for r in ds["claims"] if r.get("_split") == "dev"]

    rows = [
        ("Legacy TF-IDF + first-term filter", "PRO", legacy_retrieve),
        ("Legacy TF-IDF + first-term filter", "CON", legacy_retrieve),
        ("BM25 (no stance)", "NEUTRAL", current_retrieve),
        ("BM25 + stance re-ranking (ours)", "PRO", current_retrieve),
        ("BM25 + stance re-ranking (ours)", "CON", current_retrieve),
    ]
    results = []
    for name, stance, fn in rows:
        r = {"method": name, "stance": stance, **evaluate(fn, dev, docs, stance)}
        results.append(r)
        print(f"{name:<36} {stance:<8} " + " ".join(f"R@{k}={r[f'recall@{k}']:.3f}" for k in KS) + f" empty={r['empty']}")

    # How different are the PRO and CON evidence sets? (stance separation)
    diff = [len(set(current_retrieve(c["claim"], docs, "PRO", 5)) ^ set(current_retrieve(c["claim"], docs, "CON", 5))) / 2
            for c in dev]
    separation = round(sum(diff) / len(diff), 2)
    print(f"PRO vs CON top-5 differ by {separation} documents on average")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"split": "dev", "corpus_size": len(docs), "results": results, "pro_con_top5_difference": separation}
    (RESULTS_DIR / "retrieval_dev.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"SciFact dev — {len(dev)} claims ({results[0]['n_with_evidence']} with gold evidence), corpus {len(docs)} abstracts\n",
        "| Method | Stance | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Empty |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['method']} | {r['stance']} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS) + f" | {r['empty']} |")
    lines.append(f"\nPRO vs CON top-5 evidence sets differ by {separation} documents on average.")
    (RESULTS_DIR / "retrieval_dev.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {RESULTS_DIR / 'retrieval_dev.md'}")


if __name__ == "__main__":
    main()
