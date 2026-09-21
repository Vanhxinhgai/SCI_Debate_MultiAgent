"""Disagreement via Jensen-Shannon Divergence (JSD).

Mỗi agent (Pro, Con) đọc transcript và ra verdict N lần (PARALLEL) →
thu được 2 phân phối xác suất trên {SUPPORTED, REFUTED, INCONCLUSIVE}.
JSD đo khoảng cách giữa 2 phân phối.

JSD ∈ [0, 1] (khi dùng log2):
    0.0 = Pro và Con có cùng phân phối → đồng thuận hoàn toàn
    1.0 = Pro và Con hoàn toàn trái ngược → mâu thuẫn tuyệt đối

Cơ sở toán học:
    M = (P + Q) / 2
    JSD(P, Q) = (KL(P || M) + KL(Q || M)) / 2

    Trong đó KL là Kullback-Leibler divergence:
    KL(P || M) = Σ P(x) * log2(P(x) / M(x))

Optimization: N samples của mỗi agent chạy PARALLEL (ThreadPoolExecutor).
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from scidebate.llms import BaseLLM


VALID_VERDICTS = ("SUPPORTED", "REFUTED", "INCONCLUSIVE")

AGENT_VERDICT_PROMPT = """You are the {role} in a scientific debate.

CLAIM: "{claim}"

TRANSCRIPT:
{transcript}

Based on the EVIDENCE QUALITY discussed in the transcript — not just your advocacy role — what verdict does the evidence actually support from your perspective?
If you acknowledged in the transcript that evidence is indirect, proxy, or correlational, let that reflect here.
Output ONLY one word: SUPPORTED, REFUTED, or INCONCLUSIVE."""


LOGPROBS_AGENT_VERDICT_PROMPT = """You are the {role} in a scientific debate.

CLAIM: "{claim}"

TRANSCRIPT:
{transcript}

Based on the EVIDENCE QUALITY discussed in the transcript — if the evidence is proxy or indirect, choose accordingly.
Output ONLY one character: A, B, or C.
A = SUPPORTED
B = REFUTED
C = INCONCLUSIVE

Verdict choice (A/B/C):"""

# Markers in transcript text that indicate the PRO agent acknowledged weak evidence
_INSUFFICIENCY_MARKERS = [
    "no direct evidence", "proxy evidence", "correlational evidence",
    "cannot establish", "unsubstantiated", "indirect evidence",
    "no experimental", "no causal", "only indirect", "only correlational",
    "remains unsubstantiated", "does not directly measure",
]
_UNIVERSAL_WEAKNESS_MARKERS = [
    "taper off", "capacity-bound", "threshold", "conditional", "not always",
    "only in some", "limited context", "diminish", "diminishes",
]


def _apply_distribution_sanity(
    dist: dict,
    transcript_text: str,
    role: str,
    claim_type: str = "GENERAL",
) -> dict:
    """Cap SUPPORTED probability when transcript shows PRO admitted weak evidence."""
    if role != "PRO":
        return dist

    text_lower = transcript_text.lower()
    has_insufficiency = any(m in text_lower for m in _INSUFFICIENCY_MARKERS)
    has_universal_weakness = any(m in text_lower for m in _UNIVERSAL_WEAKNESS_MARKERS)

    if not has_insufficiency and not has_universal_weakness:
        return dist

    # Universal claims with tapering/conditional findings → SUPPORTED cap = 0.30
    # Ordinary claims with only proxy/indirect evidence → SUPPORTED cap = 0.50
    cap = 0.30 if (claim_type == "UNIVERSAL_CAUSAL" or has_universal_weakness) else 0.50
    supported = dist.get("SUPPORTED", 0.0)

    if supported > cap:
        excess = supported - cap
        dist = dict(dist)
        dist["SUPPORTED"] = cap
        dist["INCONCLUSIVE"] = dist.get("INCONCLUSIVE", 0.0) + excess

    return dist


@dataclass
class AgentDistribution:
    """Phân phối verdict của 1 agent."""
    role: str                                           # "PRO" | "CON"
    distribution: dict = field(default_factory=dict)    # {"SUPPORTED": 0.6, ...}
    counts: dict = field(default_factory=dict)          # {"SUPPORTED": 3, ...}
    n_samples: int = 0


@dataclass
class DisagreementResult:
    """Kết quả đo disagreement bằng JSD."""
    jsd: float = 0.0                        # Jensen-Shannon Divergence [0, 1]
    pro_distribution: AgentDistribution = None
    con_distribution: AgentDistribution = None


def _kl_divergence(p: dict, m: dict) -> float:
    """KL(P || M) = Σ P(x) * log2(P(x) / M(x))."""
    kl = 0.0
    for x in VALID_VERDICTS:
        px = p.get(x, 0.0)
        mx = m.get(x, 0.0)
        if px > 0 and mx > 0:
            kl += px * math.log2(px / mx)
    return kl


def compute_jsd(p: dict, q: dict) -> float:
    """Tính Jensen-Shannon Divergence giữa 2 phân phối.

    Args:
        p: phân phối verdict agent 1, vd {"SUPPORTED": 0.6, "REFUTED": 0.2, ...}
        q: phân phối verdict agent 2

    Returns:
        float [0, 1]
    """
    # M = (P + Q) / 2
    m = {}
    for x in VALID_VERDICTS:
        m[x] = (p.get(x, 0.0) + q.get(x, 0.0)) / 2.0

    jsd = (_kl_divergence(p, m) + _kl_divergence(q, m)) / 2.0
    return max(0.0, min(1.0, jsd))


def _generate_single_agent_verdict(
    llm: BaseLLM,
    messages: list,
    max_tokens: int,
    temperature: float,
) -> str:
    """Helper: generate 1 agent verdict (for parallelization)."""
    response = llm.generate(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_verdict(response.content)


def _sample_agent_verdicts(
    role: str,
    claim: str,
    transcript_text: str,
    llm: BaseLLM,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_workers: int = 4,
    use_logprobs: bool = True,
) -> AgentDistribution:
    """Chạy 1 agent ra verdict N lần (PARALLEL), trả về phân phối.
    Nếu use_logprobs=True, thử dùng single-call logprobs trước.

    Args:
        role: "PRO" hoặc "CON" — agent sẽ đọc transcript từ góc nhìn này
        claim: tuyên bố khoa học
        transcript_text: toàn bộ transcript dạng string
        llm: LLM dùng cho agent này
        n_samples: số lần sample (nếu fallback)
        temperature: sampling temperature (> 0 để có diversity)
        max_workers: số thread parallelization (default 4)
        use_logprobs: có dùng tối ưu hóa single-call logprobs không
    """
    if use_logprobs:
        try:
            logprobs_prompt = LOGPROBS_AGENT_VERDICT_PROMPT.format(
                role=role,
                claim=claim,
                transcript=transcript_text,
            )
            messages = [{"role": "user", "content": logprobs_prompt}]
            response = llm.generate(
                messages,
                logprobs=True,
                top_logprobs=5,
                max_tokens=1,
                temperature=0.0,
            )
            if response.logprobs and len(response.logprobs) > 0:
                first_token_data = response.logprobs[0]
                top_lps = first_token_data.get("top_logprobs", [])
                
                # Initialize default prior logprobs for the choices (Laplace prior)
                # -4.0 represents a small but non-negligible baseline (e^-4 ≈ 1.8%)
                choice_logprobs = {"A": -4.0, "B": -4.0, "C": -4.0}
                
                for item in top_lps:
                    token = item["token"]
                    logprob = item["logprob"]
                    token_upper = token.strip().upper()
                    clean_token = re.sub(r'[^A-Z]', '', token_upper)
                    
                    # Update highest logprob found for each choice category
                    if clean_token == "A" or "SUPPORTED" in token_upper:
                        choice_logprobs["A"] = max(choice_logprobs["A"], logprob)
                    elif clean_token == "B" or "REFUTED" in token_upper:
                        choice_logprobs["B"] = max(choice_logprobs["B"], logprob)
                    elif clean_token == "C" or "INCONCLUSIVE" in token_upper:
                        choice_logprobs["C"] = max(choice_logprobs["C"], logprob)
                
                # Temperature Scaling to calibrate / smooth the distribution (T = 1.5)
                temp_scale = 1.5
                
                p_sup = math.exp(choice_logprobs["A"] / temp_scale)
                p_ref = math.exp(choice_logprobs["B"] / temp_scale)
                p_inc = math.exp(choice_logprobs["C"] / temp_scale)
                
                total_p = p_sup + p_ref + p_inc
                if total_p > 1e-5:
                    distribution = {
                        "SUPPORTED": p_sup / total_p,
                        "REFUTED": p_ref / total_p,
                        "INCONCLUSIVE": p_inc / total_p
                    }
                    
                    counts = {
                        "SUPPORTED": round(distribution["SUPPORTED"] * 10),
                        "REFUTED": round(distribution["REFUTED"] * 10),
                        "INCONCLUSIVE": round(distribution["INCONCLUSIVE"] * 10),
                    }
                    
                    return AgentDistribution(
                        role=role,
                        distribution=distribution,
                        counts=counts,
                        n_samples=1,
                    )
        except Exception:
            pass

    prompt = AGENT_VERDICT_PROMPT.format(
        role=role,
        claim=claim,
        transcript=transcript_text,
    )

    # Chạy N verdicts PARALLEL
    # Inject sample_idx vào prompt để phá vỡ determinism (temperature alone không đủ)
    verdicts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for sample_idx in range(n_samples):
            # Perturb prompt với sample index để tạo diversity
            perturbed_prompt = prompt + f"\n\n[Evaluation #{sample_idx + 1}]"
            messages = [{"role": "user", "content": perturbed_prompt}]
            
            futures.append(
                executor.submit(
                    _generate_single_agent_verdict,
                    llm,
                    messages,
                    10,
                    temperature,
                )
            )
        verdicts = [f.result() for f in futures]

    # Tính phân phối với Laplace smoothing (alpha = 0.1) để tránh phân cực 0% / 100% do cỡ mẫu nhỏ
    alpha = 0.1
    counts = Counter(verdicts)
    total_samples = len(verdicts)
    total_smoothed = total_samples + len(VALID_VERDICTS) * alpha
    distribution = {v: (counts.get(v, 0) + alpha) / total_smoothed for v in VALID_VERDICTS}

    return AgentDistribution(
        role=role,
        distribution=distribution,
        counts=dict(counts),
        n_samples=total_samples,
    )


def compute_disagreement(
    claim: str,
    transcript_text: str,
    pro_llm: BaseLLM,
    con_llm: BaseLLM,
    n_samples: int = 3,
    temperature: float = 0.7,
    max_workers: int = 4,
    use_logprobs: bool = True,
    claim_type: str = "GENERAL",
) -> DisagreementResult:
    """Đo disagreement giữa Pro và Con bằng JSD (PARALLEL).

    Mỗi agent đọc transcript từ góc nhìn riêng, ra verdict N lần (song song).
    JSD giữa 2 phân phối = mức bất đồng.

    Args:
        claim: tuyên bố khoa học
        transcript_text: toàn bộ transcript
        pro_llm: LLM dùng cho Pro sampling
        con_llm: LLM dùng cho Con sampling
        n_samples: số lần sample mỗi agent (nếu fallback)
        temperature: sampling temperature
        max_workers: số thread parallelization
        use_logprobs: có dùng tối ưu hóa single-call logprobs không
        claim_type: claim type for distribution sanity check

    Returns:
        DisagreementResult
    """
    # Pro + Con sampling PARALLEL
    with ThreadPoolExecutor(max_workers=2) as executor:
        pro_future = executor.submit(
            _sample_agent_verdicts,
            role="PRO agent who argued for the claim",
            claim=claim,
            transcript_text=transcript_text,
            llm=pro_llm,
            n_samples=n_samples,
            temperature=temperature,
            max_workers=max_workers,
            use_logprobs=use_logprobs,
        )
        con_future = executor.submit(
            _sample_agent_verdicts,
            role="CON agent who challenged the claim",
            claim=claim,
            transcript_text=transcript_text,
            llm=con_llm,
            n_samples=n_samples,
            temperature=temperature,
            max_workers=max_workers,
            use_logprobs=use_logprobs,
        )
        pro_dist = pro_future.result()
        con_dist = con_future.result()

    # Distribution sanity check: cap SUPPORTED if transcript admits weak evidence
    pro_dist.distribution = _apply_distribution_sanity(
        pro_dist.distribution, transcript_text, "PRO", claim_type
    )

    jsd = compute_jsd(pro_dist.distribution, con_dist.distribution)

    return DisagreementResult(
        jsd=jsd,
        pro_distribution=pro_dist,
        con_distribution=con_dist,
    )


def _parse_verdict(text: str) -> str:
    """Trích xuất verdict từ response ngắn."""
    text_upper = text.strip().upper()
    for v in VALID_VERDICTS:
        if v in text_upper:
            return v
    return "INCONCLUSIVE"
