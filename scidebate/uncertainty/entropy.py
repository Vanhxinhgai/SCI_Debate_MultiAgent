"""Verdict Entropy — đo tính ổn định của verdict.

Chạy Judge N lần trên cùng transcript (temperature > 0),
thu phân phối verdict, tính Shannon entropy.

H = 0       → Judge luôn ra cùng verdict (rất ổn định)
H = log2(3) → Judge ra đều 3 loại (rất bất định, ~1.585 bits)

Optimization: chỉ chạy Judge (không chạy lại debate),
nên N=5 chỉ mất ~10s trên CPU (parallelized) thay vì ~50s (sequential).
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from scidebate.llms import BaseLLM


VALID_VERDICTS = {"SUPPORTED", "REFUTED", "INCONCLUSIVE"}

QUICK_VERDICT_PROMPT = """You are a scientific judge. Based on this debate transcript about the claim:

CLAIM: "{claim}"

TRANSCRIPT:
{transcript}

What is your verdict? Output ONLY one word: SUPPORTED, REFUTED, or INCONCLUSIVE."""


LOGPROBS_VERDICT_PROMPT = """You are a scientific judge. Based on this debate transcript about the claim:

CLAIM: "{claim}"

TRANSCRIPT:
{transcript}

What is your verdict? Output ONLY one character: A, B, or C.
A = SUPPORTED
B = REFUTED
C = INCONCLUSIVE

Verdict choice (A/B/C):"""


@dataclass
class EntropyResult:
    """Kết quả đo entropy."""
    entropy: float = 0.0                              # Shannon entropy (bits)
    max_entropy: float = math.log2(3)                  # log2(3) ≈ 1.585
    normalized_entropy: float = 0.0                    # entropy / max_entropy (0-1)
    verdict_counts: dict = field(default_factory=dict) # {"SUPPORTED": 3, "REFUTED": 2, ...}
    verdict_distribution: dict = field(default_factory=dict) # {"SUPPORTED": 0.6, ...}
    n_samples: int = 0
    dominant_verdict: str = "INCONCLUSIVE"


def _generate_single_verdict(
    judge_llm: BaseLLM,
    messages: list,
    max_tokens: int,
    temperature: float,
    sample_idx: int = 0,
) -> str:
    """Helper: generate 1 verdict (for parallelization).

    sample_idx: thêm vào system message để phá vỡ determinism
    (khi tất cả samples gửi cùng 1 prompt, model có thể vẫn trả về
    cùng 1 kết quả kể cả ở temperature cao)
    """
    # Inject sample index vào messages để tạo diversity
    perturbed = list(messages)  # shallow copy
    if sample_idx > 0 and perturbed:
        last = perturbed[-1]
        perturbed = perturbed[:-1] + [{
            "role": last["role"],
            "content": last["content"] + f"\n\n[Evaluation #{sample_idx + 1}]",
        }]
    response = judge_llm.generate(
        perturbed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_verdict(response.content)


def compute_verdict_entropy(
    claim: str,
    transcript_text: str,
    judge_llm: BaseLLM,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_workers: int = 4,
    use_logprobs: bool = True,
) -> EntropyResult:
    """Chạy Judge N lần (PARALLEL), tính entropy trên phân phối verdict.
    Nếu use_logprobs=True, cố gắng lấy phân phối từ token logprobs chỉ trong 1 call.

    Args:
        claim: tuyên bố khoa học
        transcript_text: toàn bộ transcript dạng string
        judge_llm: LLM dùng làm judge
        n_samples: số lần chạy Judge (nếu fallback)
        temperature: sampling temperature (> 0 để có diversity)
        max_workers: số thread parallelization (default 4)
        use_logprobs: có dùng tối ưu hóa single-call logprobs không

    Returns:
        EntropyResult
    """
    if use_logprobs:
        try:
            logprobs_prompt = LOGPROBS_VERDICT_PROMPT.format(
                claim=claim,
                transcript=transcript_text,
            )
            messages = [{"role": "user", "content": logprobs_prompt}]
            response = judge_llm.generate(
                messages,
                logprobs=True,
                top_logprobs=5,
                max_tokens=1,
                temperature=0.0,
            )
            if response.logprobs and len(response.logprobs) > 0:
                first_token_data = response.logprobs[0]
                top_lps = first_token_data.get("top_logprobs", [])
                
                # Laplace prior: -4.0 gives ~1.8% baseline (e^-4) for unseen tokens
                choice_logprobs = {"A": -4.0, "B": -4.0, "C": -4.0}

                for item in top_lps:
                    token = item["token"]
                    logprob = item["logprob"]
                    token_upper = token.strip().upper()
                    clean_token = re.sub(r'[^A-Z]', '', token_upper)

                    if clean_token == "A" or "SUPPORTED" in token_upper:
                        choice_logprobs["A"] = max(choice_logprobs["A"], logprob)
                    elif clean_token == "B" or "REFUTED" in token_upper:
                        choice_logprobs["B"] = max(choice_logprobs["B"], logprob)
                    elif clean_token == "C" or "INCONCLUSIVE" in token_upper:
                        choice_logprobs["C"] = max(choice_logprobs["C"], logprob)

                # Temperature scaling T=1.5 — moderate smoothing without flattening signal
                temp_scale = 1.5

                p_sup = math.exp(choice_logprobs["A"] / temp_scale)
                p_ref = math.exp(choice_logprobs["B"] / temp_scale)
                p_inc = math.exp(choice_logprobs["C"] / temp_scale)

                total_p = p_sup + p_ref + p_inc
                if total_p > 1e-5:
                    distribution = {
                        "SUPPORTED": p_sup / total_p,
                        "REFUTED": p_ref / total_p,
                        "INCONCLUSIVE": p_inc / total_p,
                    }
                    counts = {
                        "SUPPORTED": round(distribution["SUPPORTED"] * 10),
                        "REFUTED": round(distribution["REFUTED"] * 10),
                        "INCONCLUSIVE": round(distribution["INCONCLUSIVE"] * 10),
                    }
                    entropy = -sum(p * math.log2(p) for p in distribution.values() if p > 0)
                    max_entropy = math.log2(3)
                    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
                    dominant = max(distribution, key=distribution.get)
                    return EntropyResult(
                        entropy=entropy,
                        max_entropy=max_entropy,
                        normalized_entropy=normalized,
                        verdict_counts=counts,
                        verdict_distribution=distribution,
                        n_samples=1,
                        dominant_verdict=dominant,
                    )
        except Exception:
            pass

    prompt = QUICK_VERDICT_PROMPT.format(
        claim=claim,
        transcript=transcript_text,
    )
    messages = [{"role": "user", "content": prompt}]

    # Thu thập N verdicts (PARALLEL)
    # Mỗi sample dùng prompt khác nhau (sample_idx) để phá vỡ determinism
    verdicts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _generate_single_verdict,
                judge_llm,
                messages,
                10,
                temperature,
                i,  # sample_idx: inject vào prompt để tạo diversity
            )
            for i in range(n_samples)
        ]
        verdicts = [f.result() for f in futures]

    # Tính phân phối với Laplace smoothing (alpha = 0.1) để tránh phân cực 0% / 100% do cỡ mẫu nhỏ
    alpha = 0.1
    counts = Counter(verdicts)
    total_samples = len(verdicts)
    total_smoothed = total_samples + len(VALID_VERDICTS) * alpha
    distribution = {v: (counts.get(v, 0) + alpha) / total_smoothed for v in VALID_VERDICTS}

    # Shannon entropy: H = -Σ p·log2(p)
    entropy = 0.0
    for p in distribution.values():
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(VALID_VERDICTS))  # log2(3) ≈ 1.585
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    # Dominant verdict
    dominant = max(counts, key=counts.get) if counts else "INCONCLUSIVE"

    return EntropyResult(
        entropy=entropy,
        max_entropy=max_entropy,
        normalized_entropy=normalized,
        verdict_counts=dict(counts),
        verdict_distribution=distribution,
        n_samples=total_samples,
        dominant_verdict=dominant,
    )


def _parse_verdict(text: str) -> str:
    """Trích xuất verdict từ response ngắn."""
    text_upper = text.strip().upper()
    for v in VALID_VERDICTS:
        if v in text_upper:
            return v
    return "INCONCLUSIVE"
