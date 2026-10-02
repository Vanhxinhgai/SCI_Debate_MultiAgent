"""Baseline systems for comparison against the full MAD+RAG system.

Baselines:
    A — Zero-Shot LLM: single LLM, no debate, no RAG
    B — RAG-Only:      retrieve evidence → single-pass verdict (no debate)
    B2 — CoT + RAG:    same evidence as B, step-by-step reasoning (no debate)
    C — Debate-No-RAG: multi-agent debate WITHOUT evidence retrieval
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass

from scidebate.llms.base import BaseLLM
from scidebate.tools import retrieve_evidence


VERDICT_RE = re.compile(
    r"\b(SUPPORTED|REFUTED|INCONCLUSIVE)\b", re.IGNORECASE
)
CONFIDENCE_RE = re.compile(r"confidence[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)


VERDICT_LINE_RE = re.compile(
    r"verdict\s*[:\-]\s*\**\s*(SUPPORTED|REFUTED|INCONCLUSIVE)", re.IGNORECASE
)


def _ask(llm: BaseLLM, prompt: str) -> str:
    """Single-turn call. BaseLLM.generate expects a message list and returns LLMResponse."""
    return llm.generate([{"role": "user", "content": prompt}]).content


def _parse_verdict_from_text(text: str) -> tuple[str, float]:
    """Extract (verdict, confidence) from a free-text response.

    Prefer the explicit "Verdict: X" line (the last one, since CoT reasoning may
    mention other labels first); otherwise fall back to the last bare label.
    """
    line_matches = VERDICT_LINE_RE.findall(text)
    bare_matches = VERDICT_RE.findall(text)
    if line_matches:
        verdict = line_matches[-1].upper()
    elif bare_matches:
        verdict = bare_matches[-1].upper()
    else:
        verdict = "INCONCLUSIVE"
    cm = CONFIDENCE_RE.search(text)
    confidence = float(cm.group(1)) if cm else 0.5
    confidence = max(0.0, min(1.0, confidence))
    return verdict, confidence


@dataclass
class BaselineResult:
    claim: str
    verdict: str
    confidence: float
    elapsed_seconds: float
    raw_response: str = ""
    retrieved_papers: list = None

    def __post_init__(self):
        if self.retrieved_papers is None:
            self.retrieved_papers = []


# ── Baseline A: Zero-Shot LLM ─────────────────────────────────────────────────

_ZERO_SHOT_PROMPT = """\
You are a scientific fact-checker. Evaluate the following scientific claim based on \
your knowledge of peer-reviewed research.

Claim: {claim}

Provide your verdict as exactly one of: SUPPORTED, REFUTED, or INCONCLUSIVE.
Then provide a confidence score (0.0–1.0) and a brief justification (2–3 sentences).

Format:
Verdict: <SUPPORTED|REFUTED|INCONCLUSIVE>
Confidence: <0.0–1.0>
Justification: <your reasoning>
"""


class ZeroShotBaseline:
    """Baseline A: single LLM, zero-shot verdict — no debate, no RAG."""

    name = "Baseline-A: Zero-Shot LLM"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def run(self, claim: str) -> BaselineResult:
        t0 = time.time()
        prompt = _ZERO_SHOT_PROMPT.format(claim=claim)
        raw = _ask(self.llm, prompt)
        verdict, confidence = _parse_verdict_from_text(raw)
        return BaselineResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            elapsed_seconds=round(time.time() - t0, 2),
            raw_response=raw,
        )


# ── Baseline B: RAG-Only ─────────────────────────────────────────────────────

_RAG_ONLY_PROMPT = """\
You are a scientific fact-checker. Use the retrieved evidence below to evaluate the claim.

RETRIEVED EVIDENCE:
{context}

Claim: {claim}

Based ONLY on the evidence above, provide your verdict as exactly one of: \
SUPPORTED, REFUTED, or INCONCLUSIVE.

Format:
Verdict: <SUPPORTED|REFUTED|INCONCLUSIVE>
Confidence: <0.0–1.0>
Justification: <your reasoning citing the evidence>
"""


class RAGOnlyBaseline:
    """Baseline B: retrieve evidence then single-pass verdict — no debate."""

    name = "Baseline-B: RAG-Only"

    def __init__(self, llm: BaseLLM, max_results: int = 5, source: str = "hybrid"):
        self.llm = llm
        self.max_results = max_results
        self.source = source

    def run(self, claim: str) -> BaselineResult:
        t0 = time.time()

        try:
            res = retrieve_evidence(
                claim,
                max_results=self.max_results,
                generator_llm=self.llm,
                stance="NEUTRAL",
                source=self.source,
            )
            context = res["context"]
            papers = res["papers"]
        except Exception:
            context = "No evidence retrieved."
            papers = []

        if context.strip():
            prompt = _RAG_ONLY_PROMPT.format(context=context, claim=claim)
        else:
            prompt = _ZERO_SHOT_PROMPT.format(claim=claim)

        raw = _ask(self.llm, prompt)
        verdict, confidence = _parse_verdict_from_text(raw)

        return BaselineResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            elapsed_seconds=round(time.time() - t0, 2),
            raw_response=raw,
            retrieved_papers=papers,
        )


# ── Baseline B2: Chain-of-Thought + RAG ───────────────────────────────────────

_COT_PROMPT = """You are a scientific fact-checker. Use the retrieved evidence below to evaluate the claim.

RETRIEVED EVIDENCE:
{context}

Claim: {claim}

Let's think step by step:
1. Identify what the claim asserts (population, intervention/exposure, outcome, direction).
2. For each relevant document, state whether it supports, contradicts, or does not address the claim.
3. Weigh the evidence. If no document directly addresses the claim, the verdict is INCONCLUSIVE.

After your reasoning, end with exactly these lines:
Verdict: <SUPPORTED|REFUTED|INCONCLUSIVE>
Confidence: <0.0–1.0>
"""


class CoTBaseline:
    """Baseline B2: same retrieved evidence as RAG-Only, with step-by-step reasoning."""

    name = "Baseline-B2: CoT + RAG"

    def __init__(self, llm: BaseLLM, max_results: int = 5, source: str = "hybrid"):
        self.llm = llm
        self.max_results = max_results
        self.source = source

    def run(self, claim: str) -> BaselineResult:
        t0 = time.time()
        try:
            res = retrieve_evidence(
                claim,
                max_results=self.max_results,
                generator_llm=self.llm,
                stance="NEUTRAL",
                source=self.source,
            )
            context, papers = res["context"], res["papers"]
        except Exception:
            context, papers = "No evidence retrieved.", []

        raw = _ask(self.llm, _COT_PROMPT.format(context=context, claim=claim))
        verdict, confidence = _parse_verdict_from_text(raw)
        return BaselineResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            elapsed_seconds=round(time.time() - t0, 2),
            raw_response=raw,
            retrieved_papers=papers,
        )


# ── Baseline C: Debate-No-RAG ─────────────────────────────────────────────────

class DebateNoRAGBaseline:
    """Baseline C: full multi-agent debate WITHOUT evidence retrieval."""

    name = "Baseline-C: Debate (no RAG)"

    def __init__(
        self,
        pro_llm: BaseLLM,
        con_llm: BaseLLM,
        judge_llm: BaseLLM,
        max_rounds: int = 2,
    ):
        self.pro_llm = pro_llm
        self.con_llm = con_llm
        self.judge_llm = judge_llm
        self.max_rounds = max_rounds

    def run(self, claim: str) -> BaselineResult:
        from scidebate import Debate

        t0 = time.time()
        debate = Debate(
            pro_llm=self.pro_llm,
            con_llm=self.con_llm,
            judge_llm=self.judge_llm,
            max_rounds=self.max_rounds,
            parallel_opening=False,
            compute_uncertainty=False,
            verbose=False,
            enable_rag=False,
        )
        result = debate.run(claim)
        verdict = result.verdict.verdict if result.verdict else "INCONCLUSIVE"
        confidence = result.verdict.confidence if result.verdict else 0.5

        return BaselineResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            elapsed_seconds=round(time.time() - t0, 2),
            raw_response=result.to_text(),
        )
