"""Consensus — tổng hợp Entropy + JSD Disagreement + Confidence (PARALLEL).

Consensus Quadrant (2D: entropy × JSD):

    High JSD (Disagreement)
         |
    Q2: Genuine        Q4: Confused /
    Controversy         Insufficient
         |               Evidence
    -----+------ High Entropy →
         |
    Q1: Strong          Q3: Aligned
    Consensus           Uncertainty
         |
    Low JSD (Agreement)

Consensus Level:
    HIGH:   Q1 (low entropy + low JSD) → verdict ổn định, Pro/Con đồng thuận
    LOW:    Q2 (low entropy + high JSD) → verdict rõ nhưng tranh cãi dữ dội
    MEDIUM: Q3, Q4, hoặc borderline

Optimization: Entropy + Disagreement computation parallel (ThreadPoolExecutor).
"""
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from scidebate.llms import BaseLLM
from scidebate.uncertainty.entropy import compute_verdict_entropy, EntropyResult
from scidebate.uncertainty.disagreement import (
    compute_disagreement,
    DisagreementResult,
)
from scidebate.uncertainty.confidence import compute_calibrated_confidence


@dataclass
class ConsensusMetrics:
    """Kết quả đo lường đồng thuận khoa học."""

    # Entropy (Judge stability)
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    judge_verdict_distribution: dict = field(default_factory=dict)
    dominant_verdict: str = "INCONCLUSIVE"
    n_entropy_samples: int = 0

    # JSD Disagreement (Pro vs Con)
    jsd: float = 0.0
    pro_distribution: dict = field(default_factory=dict)
    con_distribution: dict = field(default_factory=dict)
    n_jsd_samples: int = 0

    # Confidence
    calibrated_confidence: float = 0.0
    judge_raw_confidence: float = 0.0

    # Consensus
    consensus_level: str = "MEDIUM"
    consensus_quadrant: str = ""
    explanation: str = ""

    def summary(self) -> str:
        j_dist = ", ".join(
            f"{k}={v:.0%}" for k, v in self.judge_verdict_distribution.items() if v > 0
        )
        p_dist = ", ".join(
            f"{k}={v:.0%}" for k, v in self.pro_distribution.items() if v > 0
        )
        c_dist = ", ".join(
            f"{k}={v:.0%}" for k, v in self.con_distribution.items() if v > 0
        )
        return (
            f"Consensus Level: {self.consensus_level}\n"
            f"  Quadrant: {self.consensus_quadrant}\n"
            f"  Entropy: {self.entropy:.3f} "
            f"(normalized: {self.normalized_entropy:.2f})\n"
            f"  Judge distribution: [{j_dist}] "
            f"(n={self.n_entropy_samples})\n"
            f"  JSD (Pro vs Con): {self.jsd:.3f}\n"
            f"  Pro distribution: [{p_dist}]\n"
            f"  Con distribution: [{c_dist}] "
            f"(n={self.n_jsd_samples} per agent)\n"
            f"  Calibrated confidence: {self.calibrated_confidence:.2f}\n"
            f"  Explanation: {self.explanation}"
        )


def _determine_quadrant(norm_entropy: float, jsd: float) -> str:
    low_e = norm_entropy < 0.5
    low_d = jsd < 0.4  # JSD thường thấp hơn raw disagreement score

    if low_e and low_d:
        return "Strong Consensus"
    elif low_e and not low_d:
        return "Genuine Controversy"
    elif not low_e and low_d:
        return "Aligned Uncertainty"
    else:
        return "Confused / Insufficient Evidence"


def _determine_consensus_level(quadrant: str) -> tuple[str, str]:
    if quadrant == "Strong Consensus":
        return "HIGH", (
            "Strong scientific consensus. "
            "The verdict is stable across samplings and "
            "Pro/Con verdict distributions are aligned."
        )
    elif quadrant == "Genuine Controversy":
        return "LOW", (
            "Low consensus — genuine scientific controversy. "
            "The judge's verdict is stable, but Pro and Con "
            "maintain strongly divergent positions (high JSD). "
            "This claim remains contentious in the literature."
        )
    elif quadrant == "Aligned Uncertainty":
        return "MEDIUM", (
            "Moderate consensus with verdict uncertainty. "
            "Pro and Con show similar verdict distributions, "
            "but the judge's verdict varies across samplings."
        )
    else:
        return "LOW", (
            "Low consensus — confused assessment. "
            "Both high verdict instability and high Pro/Con "
            "divergence suggest insufficient or contradictory evidence."
        )


def compute_consensus(
    claim: str,
    transcript_text: str,
    pro_llm: BaseLLM,
    con_llm: BaseLLM,
    judge_llm: BaseLLM,
    judge_raw_confidence: float = 0.5,
    n_samples: int = 5,
    uncertainty_judge_llm: BaseLLM = None,
    use_logprobs: bool = True,
) -> ConsensusMetrics:
    # Dùng model nhỏ hơn cho entropy sampling nếu được truyền vào
    # (tránh trường hợp judge mạnh như 70B quá deterministic → entropy luôn = 0)
    _entropy_judge = uncertainty_judge_llm or judge_llm
    """Tính toàn bộ consensus metrics (PARALLEL).

    Pipeline:
        1. Entropy: Judge ra verdict N lần → Shannon entropy (PARALLEL)
        2. JSD: Pro ra verdict N lần + Con ra verdict N lần → JSD (PARALLEL)
        3. Entropy + JSD computation chạy đồng thời (PARALLEL)
        4. Confidence: calibrated từ judge_confidence + entropy + JSD trend
        5. Quadrant + consensus level

    Args:
        claim: tuyên bố khoa học
        transcript_text: toàn bộ transcript dạng string
        pro_llm: LLM của Pro agent
        con_llm: LLM của Con agent
        judge_llm: LLM của Judge agent
        judge_raw_confidence: confidence từ Judge verdict ban đầu
        n_samples: số lần sample mỗi agent
        uncertainty_judge_llm: LLM của Judge cho uncertainty
        use_logprobs: có dùng tối ưu hóa single-call logprobs không

    Returns:
        ConsensusMetrics
    """
    # 1 + 2. Entropy + Disagreement PARALLEL
    with ThreadPoolExecutor(max_workers=2) as executor:
        entropy_future = executor.submit(
            compute_verdict_entropy,
            claim=claim,
            transcript_text=transcript_text,
            judge_llm=_entropy_judge,
            n_samples=n_samples,
            temperature=0.9,  # Đủ cao để model nhỏ có diversity
            use_logprobs=use_logprobs,
        )
        disagreement_future = executor.submit(
            compute_disagreement,
            claim=claim,
            transcript_text=transcript_text,
            pro_llm=pro_llm,
            con_llm=con_llm,
            n_samples=n_samples,
            use_logprobs=use_logprobs,
        )
        entropy_result: EntropyResult = entropy_future.result()
        disagreement_result: DisagreementResult = disagreement_future.result()

    # 3. Calibrated confidence
    # Dùng JSD như proxy cho disagreement_timeline (1 giá trị)
    cal_conf = compute_calibrated_confidence(
        judge_confidence=judge_raw_confidence,
        normalized_entropy=entropy_result.normalized_entropy,
        disagreement_timeline=[disagreement_result.jsd],
    )
    
    # 4. Quadrant + level
    quadrant = _determine_quadrant(
        entropy_result.normalized_entropy,
        disagreement_result.jsd,
    )
    level, explanation = _determine_consensus_level(quadrant)

    return ConsensusMetrics(
        # Entropy
        entropy=entropy_result.entropy,
        normalized_entropy=entropy_result.normalized_entropy,
        judge_verdict_distribution=entropy_result.verdict_distribution,
        dominant_verdict=entropy_result.dominant_verdict,
        n_entropy_samples=entropy_result.n_samples,
        # JSD
        jsd=disagreement_result.jsd,
        pro_distribution=disagreement_result.pro_distribution.distribution,
        con_distribution=disagreement_result.con_distribution.distribution,
        n_jsd_samples=disagreement_result.pro_distribution.n_samples,
        # Confidence
        calibrated_confidence=cal_conf,
        judge_raw_confidence=judge_raw_confidence,
        # Consensus
        consensus_level=level,
        consensus_quadrant=quadrant,
        explanation=explanation,
    )
