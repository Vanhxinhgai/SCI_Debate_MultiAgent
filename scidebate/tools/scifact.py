"""SciFact dataset loader and lexical evidence retriever.

Expected local layout:
    data/scifact/corpus.jsonl
    data/scifact/claims_train.jsonl
    data/scifact/claims_dev.jsonl
    data/scifact/claims_test.jsonl

The retriever uses gold claim annotations only when the input claim exactly
matches a SciFact claim. For arbitrary user claims, it falls back to lexical
retrieval over the SciFact corpus and marks results as retrieved, not gold.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


DEFAULT_SCIFACT_DIR = Path("data/scifact")
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


@lru_cache(maxsize=4)
def load_scifact_dataset(data_dir: str = str(DEFAULT_SCIFACT_DIR)) -> dict:
    """Load SciFact corpus and claim files from a local directory."""
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


def find_scifact_claim(claim: str, data_dir: str = str(DEFAULT_SCIFACT_DIR)) -> dict | None:
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
        text = f"{doc.get('title', '')} {_abstract_text(doc)}".lower()
        matches = sum(1 for t in boost_terms if t in text)
        reweighted.append((score * (1.0 + 0.35 * matches), doc))
    return sorted(reweighted, key=lambda item: item[0], reverse=True)


def _score_docs(claim: str, docs: dict[str, dict]) -> list[tuple[float, dict]]:
    query_terms = _tokenize(claim)
    if not query_terms:
        return []

    doc_terms = {}
    df = Counter()
    for doc_id, doc in docs.items():
        text = f"{doc.get('title', '')} {_abstract_text(doc)}"
        terms = _tokenize(text)
        counts = Counter(terms)
        doc_terms[doc_id] = counts
        for term in counts:
            df[term] += 1

    n_docs = max(len(docs), 1)
    scores = []
    query_counts = Counter(query_terms)
    for doc_id, counts in doc_terms.items():
        score = 0.0
        doc_len = sum(counts.values()) or 1
        title_terms = set(_tokenize(str(docs[doc_id].get("title", ""))))
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
    evidence = claim_row.get("evidence") or {}
    wanted_labels = {"PRO": {"SUPPORT"}, "CON": {"CONTRADICT"}}.get(stance, {"SUPPORT", "CONTRADICT"})
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
    data_dir: str = str(DEFAULT_SCIFACT_DIR),
    use_gold_if_exact_match: bool = True,
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
        mode_note = (
            "These are gold SciFact evidence annotations for an exact dataset claim match."
            if retrieval_mode == "gold"
            else "These are lexically retrieved SciFact corpus documents, not gold evidence labels."
        )
        context_lines = [
            header,
            mode_note,
            "Use only directly relevant evidence. Do not fabricate citations or labels.",
            "",
        ]
        for i, paper in enumerate(papers, 1):
            label = paper.get("label", "RETRIEVED")
            doc_id = paper.get("doc_id", "N/A")
            evidence_sentences = paper.get("evidence_sentences") or []
            sentence_note = f" sentences={evidence_sentences}" if evidence_sentences else ""
            context_lines.extend([
                f"Document [{i}] (SciFact doc_id={doc_id}, label={label}{sentence_note}):",
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
