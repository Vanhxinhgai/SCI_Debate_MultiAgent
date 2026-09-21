"""Evaluation metrics for the SCI_Debate_MultiAgent system.

Metrics:
    - Accuracy
    - Macro F1 (per-class and averaged)
    - ECE (Expected Calibration Error)
    - Brier Score (multi-class)
    - Retrieval Precision (domain relevance)
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Sequence


VERDICT_LABELS = ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]
LABEL_TO_IDX = {v: i for i, v in enumerate(VERDICT_LABELS)}


# ── Accuracy ──────────────────────────────────────────────────────────────────

def verdict_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> float:
    """Fraction of predictions that exactly match ground truth."""
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


# ── Macro F1 ──────────────────────────────────────────────────────────────────

def compute_macro_f1(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> dict:
    """Compute per-class precision/recall/F1 and macro-averaged F1.

    Returns dict with keys:
        per_class: {label: {precision, recall, f1, support}}
        macro_f1: float
        macro_precision: float
        macro_recall: float
    """
    labels = sorted(set(ground_truths) | set(predictions))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    support = defaultdict(int)
    for gt in ground_truths:
        support[gt] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": support[label],
        }
        if support[label] > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

    n = len(f1s) or 1
    return {
        "per_class": per_class,
        "macro_f1": round(sum(f1s) / n, 4),
        "macro_precision": round(sum(precisions) / n, 4),
        "macro_recall": round(sum(recalls) / n, 4),
    }


# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(
    predictions: Sequence[str],
    confidences: Sequence[float],
    ground_truths: Sequence[str],
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error (lower is better, 0 = perfect calibration).

    Bins samples by confidence; in each bin measures |accuracy - avg_confidence|
    weighted by bin size.
    """
    if not predictions:
        return 0.0

    bins: list[list[tuple]] = [[] for _ in range(n_bins)]
    for pred, conf, gt in zip(predictions, confidences, ground_truths):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((pred == gt, conf))

    ece = 0.0
    n = len(predictions)
    for b in bins:
        if not b:
            continue
        acc = sum(correct for correct, _ in b) / len(b)
        avg_conf = sum(conf for _, conf in b) / len(b)
        ece += (len(b) / n) * abs(acc - avg_conf)

    return round(ece, 4)


# ── Brier Score ────────────────────────────────────────────────────────────────

def compute_brier_score(
    predictions: Sequence[str],
    confidences: Sequence[float],
    ground_truths: Sequence[str],
) -> float:
    """Multi-class Brier Score (lower is better, 0 = perfect).

    For each sample: sum over classes of (prob_class - indicator_class)^2 / n_classes.
    We treat the predicted class confidence as that class's probability and distribute
    remaining probability uniformly to other classes.
    """
    if not predictions:
        return 0.0

    n_classes = len(VERDICT_LABELS)
    total = 0.0

    for pred, conf, gt in zip(predictions, confidences, ground_truths):
        prob = {}
        remaining = 1.0 - conf
        for label in VERDICT_LABELS:
            if label == pred:
                prob[label] = conf
            else:
                prob[label] = remaining / (n_classes - 1)

        score = sum((prob[label] - (1.0 if label == gt else 0.0)) ** 2
                    for label in VERDICT_LABELS) / n_classes
        total += score

    return round(total / len(predictions), 4)


# ── Retrieval Precision ────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "neuroscience":   {"sleep", "memory", "brain", "cognitive", "neural", "neuron", "cortex"},
    "medicine":       {"diabetes", "insulin", "hiv", "antiretroviral", "vitamin", "antibiotic",
                       "clinical", "randomized", "trial", "patient", "therapy"},
    "oncology":       {"cancer", "lung", "tumor", "carcinogen", "smoking", "tobacco"},
    "public_health":  {"handwashing", "sanitation", "hygiene", "health outcome", "mortality",
                       "education", "socioeconomic"},
    "education":      {"academic", "student", "learning", "school", "gpa", "performance",
                       "multitasking", "breakfast"},
    "psychology":     {"depression", "anxiety", "mental", "social media", "adolescent", "teenager"},
}


def retrieval_precision(
    retrieved_papers: list[dict],
    claim: str,
    domain: str,
) -> float:
    """Fraction of retrieved papers whose title/abstract is relevant to the claim's domain.

    A paper is relevant if it contains ≥1 keyword from the domain set
    OR contains a keyword from the claim itself.
    """
    if not retrieved_papers:
        return 0.0

    domain_kws = _DOMAIN_KEYWORDS.get(domain, set())
    claim_words = set(claim.lower().split())

    relevant = 0
    for paper in retrieved_papers:
        text = (
            (paper.get("title") or "") + " " + (paper.get("abstract") or "")
        ).lower()

        if any(kw in text for kw in domain_kws):
            relevant += 1
        elif any(w in text for w in claim_words if len(w) > 4):
            relevant += 1

    return round(relevant / len(retrieved_papers), 4)


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def confusion_matrix_str(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
) -> str:
    """Return a simple ASCII confusion matrix."""
    labels = VERDICT_LABELS
    matrix = defaultdict(lambda: defaultdict(int))
    for pred, gt in zip(predictions, ground_truths):
        matrix[gt][pred] += 1

    col_w = 14
    header = "GT \\ Pred".ljust(col_w) + "".join(l[:12].ljust(col_w) for l in labels)
    lines = [header, "-" * (col_w * (len(labels) + 1))]
    for gt in labels:
        row = gt[:12].ljust(col_w)
        for pred in labels:
            row += str(matrix[gt][pred]).ljust(col_w)
        lines.append(row)
    return "\n".join(lines)


# ── Summary helper ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    predictions: Sequence[str],
    confidences: Sequence[float],
    ground_truths: Sequence[str],
    retrieved_papers_list: Sequence[list] = None,
    domains: Sequence[str] = None,
    claims: Sequence[str] = None,
) -> dict:
    """Compute all metrics at once and return a unified dict."""
    acc = verdict_accuracy(predictions, ground_truths)
    f1_result = compute_macro_f1(predictions, ground_truths)
    ece = compute_ece(predictions, confidences, ground_truths)
    brier = compute_brier_score(predictions, confidences, ground_truths)

    ret_precision = None
    if retrieved_papers_list and domains and claims:
        precisions = [
            retrieval_precision(papers, claim, domain)
            for papers, claim, domain in zip(retrieved_papers_list, claims, domains)
            if papers
        ]
        ret_precision = round(sum(precisions) / len(precisions), 4) if precisions else 0.0

    return {
        "accuracy": acc,
        "macro_f1": f1_result["macro_f1"],
        "macro_precision": f1_result["macro_precision"],
        "macro_recall": f1_result["macro_recall"],
        "per_class_f1": f1_result["per_class"],
        "ece": ece,
        "brier_score": brier,
        "retrieval_precision": ret_precision,
        "n_samples": len(predictions),
    }
