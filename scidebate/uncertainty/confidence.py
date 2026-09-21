"""Confidence Score — tổng hợp độ tin cậy của verdict.

Kết hợp 3 nguồn:
    1. Judge's self-reported confidence (từ verdict output)
    2. Entropy (verdict ổn định → confidence cao)
    3. Disagreement (JSD thấp → Pro/Con đồng thuận → confidence cao)
       - Multi-round: dùng TREND (JSD giảm qua rounds = debate hội tụ)
       - Single-round: dùng GIÁ TRỊ TUYỆT ĐỐI (JSD thấp → đồng thuận)

Formula:
    calibrated_confidence = w1 * judge_confidence
                          + w2 * (1 - normalized_entropy)
                          + w3 * disagreement_factor

    Trong đó:
    - w1=0.4, w2=0.35, w3=0.25 (tunable weights)
    - disagreement_factor:
        Multi-round: (jsd_round1 - jsd_lastRound + 1) / 2  → trend
        Single-round: 1 - jsd                              → absolute penalty
"""


def compute_calibrated_confidence(
    judge_confidence: float,
    normalized_entropy: float,
    disagreement_timeline: list[float],
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> float:
    """Tính calibrated confidence score.

    Args:
        judge_confidence: confidence từ Judge verdict (0-1)
        normalized_entropy: entropy / max_entropy (0-1), cao = bất ổn
        disagreement_timeline: list JSD scores qua các round.
            - 1 phần tử  → single-round, dùng absolute JSD penalty
            - 2+ phần tử → multi-round, dùng trend (giảm → tốt)

    Returns:
        float 0.0-1.0
    """
    w1, w2, w3 = weights

    # Factor 1: Judge's own confidence
    f_judge = max(0.0, min(1.0, judge_confidence))

    # Factor 2: Entropy stability (invert: low entropy → high confidence)
    f_entropy = 1.0 - max(0.0, min(1.0, normalized_entropy))

    # Factor 3: Disagreement — adaptive theo số rounds
    f_disagreement = _disagreement_factor(disagreement_timeline)

    calibrated = w1 * f_judge + w2 * f_entropy + w3 * f_disagreement
    return max(0.0, min(1.0, calibrated))


def _disagreement_factor(timeline: list[float]) -> float:
    """Tính disagreement factor từ JSD timeline.

    Single-round (1 giá trị):
        Dùng absolute JSD làm penalty:
        JSD=0.0 → factor=1.0 (Pro/Con hoàn toàn đồng thuận)
        JSD=0.5 → factor=0.5 (bất đồng trung bình)
        JSD=1.0 → factor=0.0 (Pro/Con hoàn toàn trái ngược)

    Multi-round (2+ giá trị):
        Dùng trend — debate hội tụ hay phân kỳ?
        JSD giảm mạnh → factor cao (debate đang hội tụ)
        JSD không đổi → factor trung bình (~0.5)
        JSD tăng      → factor thấp (debate phân kỳ)

    Returns:
        float 0.0-1.0
    """
    if not timeline:
        return 0.5  # không có data

    if len(timeline) == 1:
        # Single-round: absolute JSD penalty
        # JSD ∈ [0, 1] → factor = 1 - JSD
        jsd = max(0.0, min(1.0, timeline[0]))
        return 1.0 - jsd

    # Multi-round: trend (first → last)
    first = timeline[0]
    last = timeline[-1]
    delta = first - last  # positive = JSD giảm (tốt), negative = tăng (xấu)

    # Map delta (-1, 1) → factor (0, 1)
    # delta=+1.0 (giảm max) → factor=1.0
    # delta= 0.0 (không đổi) → factor=0.5
    # delta=-1.0 (tăng max)  → factor=0.0
    factor = (delta + 1.0) / 2.0
    return max(0.0, min(1.0, factor))
