"""SciFact dataset loader and lexical evidence retriever.

Expected local layout:
    data/scifact/corpus.jsonl
    data/scifact/claims_train.jsonl
    data/scifact/claims_dev.jsonl
    data/scifact/claims_test.jsonl

By default the retriever uses lexical (TF-IDF) retrieval over the corpus for
every claim. Gold evidence annotations are only used when explicitly enabled
(use_gold_if_exact_match=True) — never enable this for evaluation, because gold
evidence is selected using the claim's label (label leakage).

The dataset directory can be overridden with the SCIFACT_DIR env variable.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


DEFAULT_SCIFACT_DIR = Path("data/scifact")


def _resolve_dir(data_dir: str | None) -> str:
    """Explicit arg > SCIFACT_DIR env var > data/scifact (resolved at call time)."""
    return str(data_dir or os.environ.get("SCIFACT_DIR") or DEFAULT_SCIFACT_DIR)
CLAIM_FILES = ("claims_dev.jsonl", "claims_train.jsonl", "claims_test.jsonl")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "for",
    "from", "has", "have", "in", "is", "it", "may", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "with", "without", "than",
    "into", "between", "over", "under", "after", "before", "during",
}


class SciFactDatasetError(RuntimeError):
    """Raised when SciFact files are missing or malformed."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SciFactDatasetError(f"Invalid JSON in {path} at line {line_num}: {exc}") from exc
    return rows


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    ]


def _normalize_claim(text: str) -> str:
    return " ".join(_tokenize(text))


def _abstract_text(doc: dict) -> str:
    abstract = doc.get("abstract", "")
    if isinstance(abstract, list):
        return " ".join(str(s) for s in abstract)
    return str(abstract)


def _sentence_text(doc: dict, sentence_indices: list[int] | None = None) -> str:
    abstract = doc.get("abstract", "")
    if not isinstance(abstract, list):
        return str(abstract)
    if not sentence_indices:
        return " ".join(str(s) for s in abstract)
    selected = []
    for idx in sentence_indices:
        if isinstance(idx, int) and 0 <= idx < len(abstract):
            selected.append(str(abstract[idx]))
    return " ".join(selected) if selected else " ".join(str(s) for s in abstract)


def _doc_to_paper(
    doc: dict,
    *,
    label: str | None = None,
    sentence_indices: list[int] | None = None,
    score: float | None = None,
    gold: bool = False,
) -> dict:
    doc_id = str(doc.get("doc_id", doc.get("id", "")))
    summary = _sentence_text(doc, sentence_indices)
    paper = {
        "title": doc.get("title") or f"SciFact document {doc_id}",
        "authors": "SciFact corpus",
        "published": "N/A",
        "pdf_link": "",
        "summary": summary,
        "source": "SciFact",
        "doc_id": doc_id,
        "label": label or "RETRIEVED",
        "evidence_sentences": sentence_indices or [],
        "is_gold_evidence": gold,
    }
    if score is not None:
        paper["retrieval_score"] = score
    return paper


def load_scifact_dataset(data_dir: str | None = None) -> dict:
    """Load SciFact corpus and claim files from a local directory."""
    return _load_scifact_dataset_cached(_resolve_dir(data_dir))


@lru_cache(maxsize=4)
def _load_scifact_dataset_cached(data_dir: str) -> dict:
    root = Path(data_dir)
    corpus_path = root / "corpus.jsonl"
    if not corpus_path.exists():
        raise SciFactDatasetError(
            "SciFact corpus not found. Expected data/scifact/corpus.jsonl. "
            "Place the SciFact dataset files under data/scifact/."
        )

    corpus_rows = _read_jsonl(corpus_path)
    corpus = {str(row.get("doc_id", row.get("id", ""))): row for row in corpus_rows}
    if not corpus:
        raise SciFactDatasetError(f"No documents loaded from {corpus_path}.")

    claims = []
    for filename in CLAIM_FILES:
        for row in _read_jsonl(root / filename):
            row["_split"] = filename.replace("claims_", "").replace(".jsonl", "")
            claims.append(row)

    return {
        "root": root,
        "corpus": corpus,
        "claims": claims,
    }


def find_scifact_claim(claim: str, data_dir: str | None = None) -> dict | None:
    """Find an exact normalized claim match in SciFact claim files."""
    dataset = load_scifact_dataset(data_dir)
    target = _normalize_claim(claim)
    if not target:
        return None
    for row in dataset["claims"]:
        if _normalize_claim(str(row.get("claim", ""))) == target:
            return row
    return None


_PRO_BOOST_TERMS = {
    "support", "effective", "efficacy", "positive", "benefit", "improve",
    "reduce", "protective", "confirm", "significant", "success", "demonstrate",
    "association", "correlated", "linked", "evidence", "shown", "proven",
}

_CON_BOOST_TERMS = {
    "fail", "failed", "ineffective", "negative", "adverse", "risk",
    "limitation", "controversial", "inconsistent", "lack", "contradict",
    "refute", "no", "not", "weak", "insufficient", "uncertain", "inconclusive",
    "bias", "confound", "spurious", "overestimate",
}


def _stance_rescore(scored: list[tuple[float, dict]], stance: str) -> list[tuple[float, dict]]:
    """Boost documents whose language aligns with the requested stance."""
    boost_terms = _PRO_BOOST_TERMS if stance == "PRO" else _CON_BOOST_TERMS if stance == "CON" else set()
    if not boost_terms:
        return scored
    reweighted = []
    for score, doc in scored:
        # Match whole words: substring matching made "no" hit "know"/"normal"/"not",
        # so almost every document received the CON boost.
        words = set(re.findall(r"[a-z]+", f"{doc.get('title', '')} {_abstract_text(doc)}".lower()))
        matches = len(boost_terms & words)
        reweighted.append((score * (1.0 + 0.35 * matches), doc))
    return sorted(reweighted, key=lambda item: item[0], reverse=True)


# Index cache keyed by id(corpus dict): load_scifact_dataset is lru_cached, so the
# corpus object is stable and the index is built once instead of on every query.
_INDEX_CACHE: dict[int, tuple] = {}


def _build_index(docs: dict[str, dict]) -> tuple[dict, Counter, dict]:
    key = id(docs)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    doc_terms = {}
    df = Counter()
    title_terms = {}
    for doc_id, doc in docs.items():
        text = f"{doc.get('title', '')} {_abstract_text(doc)}"
        counts = Counter(_tokenize(text))
        doc_terms[doc_id] = counts
        title_terms[doc_id] = set(_tokenize(str(doc.get("title", ""))))
        for term in counts:
            df[term] += 1

    _INDEX_CACHE[key] = (doc_terms, df, title_terms)
    return _INDEX_CACHE[key]


def _score_docs(claim: str, docs: dict[str, dict]) -> list[tuple[float, dict]]:
    query_terms = _tokenize(claim)
    if not query_terms:
        return []

    doc_terms, df, all_title_terms = _build_index(docs)

    n_docs = max(len(docs), 1)
    scores = []
    query_counts = Counter(query_terms)
    for doc_id, counts in doc_terms.items():
        score = 0.0
        doc_len = sum(counts.values()) or 1
        title_terms = all_title_terms[doc_id]
        for term, qtf in query_counts.items():
            if term not in counts:
                continue
            idf = math.log((n_docs + 1) / (df[term] + 0.5)) + 1.0
            tf = counts[term] / doc_len
            title_boost = 2.0 if term in title_terms else 1.0
            score += qtf * idf * tf * title_boost
        if score > 0:
            scores.append((score, docs[doc_id]))

    return sorted(scores, key=lambda item: item[0], reverse=True)


def _gold_evidence_from_claim(claim_row: dict, corpus: dict[str, dict], stance: str) -> list[dict]:
    # Both sides receive the same gold documents: splitting them by gold label
    # (SUPPORT → PRO, CONTRADICT → CON) would tell each agent the answer.
    evidence = claim_row.get("evidence") or {}
    wanted_labels = {"SUPPORT", "CONTRADICT"}
    papers = []

    for doc_id, entries in evidence.items():
        doc = corpus.get(str(doc_id))
        if not doc:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries or []:
            label = entry.get("label")
            if label not in wanted_labels:
                continue
            sentence_indices = entry.get("sentences") or []
            papers.append(
                _doc_to_paper(
                    doc,
                    label=label,
                    sentence_indices=sentence_indices,
                    gold=True,
                )
            )
    return papers


def retrieve_scifact_evidence(
    claim: str,
    max_results: int = 3,
    stance: str = "NEUTRAL",
    data_dir: str | None = None,
    use_gold_if_exact_match: bool = False,
) -> dict:
    """Retrieve evidence from local SciFact files.

    Returns a dict compatible with the existing arXiv RAG pipeline:
    {"context": str, "papers": list[dict], "search_query": str}
    """
    dataset = load_scifact_dataset(data_dir)
    corpus = dataset["corpus"]
    claim_row = find_scifact_claim(claim, data_dir) if use_gold_if_exact_match else None

    papers = []
    retrieval_mode = "lexical"
    if claim_row:
        retrieval_mode = "gold"
        papers = _gold_evidence_from_claim(claim_row, corpus, stance)

    if not papers:
        scored = _score_docs(claim, corpus)
        if stance != "NEUTRAL":
            scored = _stance_rescore(scored, stance)

        # Primary-term relevance filter: require papers to mention the first key
        # noun of the claim (the "X" in "X improves/causes Y").
        # This prevents aerobic-fitness / folic-acid papers from being returned
        # for a multitasking claim simply because they share words like "performance".
        claim_tokens = _tokenize(claim)
        primary_term = claim_tokens[0] if claim_tokens else ""
        if primary_term and scored:
            relevant = [
                (s, d) for s, d in scored
                if primary_term in _tokenize(
                    f"{d.get('title', '')} {_abstract_text(d)}"
                )
            ]
            # Only use filtered list if it found anything; otherwise keep empty
            # (let the caller fall through to a "no evidence" context)
            scored = relevant

        papers = [
            _doc_to_paper(doc, score=score, gold=False)
            for score, doc in scored[:max_results]
        ]
        retrieval_mode = "lexical"

    papers = papers[:max_results]

    if stance == "PRO":
        header = "### SCIFACT EVIDENCE BASE - SUPPORTING (PRO)"
    elif stance == "CON":
        header = "### SCIFACT EVIDENCE BASE - OPPOSING (CON)"
    else:
        header = "### SCIFACT EVIDENCE BASE"

    if not papers:
        context = (
            f"{header}\n"
            "No relevant SciFact evidence was found. Do not invent SciFact citations."
        )
    else:
        # Never expose gold labels (SUPPORT/CONTRADICT) or the retrieval mode to the
        # agents — both reveal the answer. Labels stay in `papers` for the UI only.
        context_lines = [
            header,
            "These are SciFact corpus documents retrieved for this claim. They may or may not be relevant.",
            "Use only directly relevant evidence. Do not fabricate citations.",
            "",
        ]
        for i, paper in enumerate(papers, 1):
            doc_id = paper.get("doc_id", "N/A")
            context_lines.extend([
                f"Document [{i}] (SciFact doc_id={doc_id}):",
                f"- Title: {paper['title']}",
                f"- Evidence Text: {paper['summary']}",
                "",
            ])
        context = "\n".join(context_lines)

    return {
        "context": context,
        "papers": papers,
        "search_query": claim,
        "source": "SciFact",
        "retrieval_mode": retrieval_mode,
        "matched_claim": claim_row,
    }
