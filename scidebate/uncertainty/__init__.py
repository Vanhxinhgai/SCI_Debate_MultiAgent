"""Uncertainty and consensus measurement.

3 metrics (information-theoretic framework):
    - Entropy: verdict stability (Judge samples N times)
    - JSD Disagreement: Pro vs Con verdict divergence
    - Confidence: calibrated score combining all metrics

Tổng hợp → ConsensusMetrics với consensus level + quadrant.
"""
from .entropy import compute_verdict_entropy, EntropyResult
from .disagreement import (
    compute_jsd,
    compute_disagreement,
    DisagreementResult,
    AgentDistribution,
)
from .confidence import compute_calibrated_confidence
from .consensus import compute_consensus, ConsensusMetrics

__all__ = [
    "compute_verdict_entropy",
    "EntropyResult",
    "compute_jsd",
    "compute_disagreement",
    "DisagreementResult",
    "AgentDistribution",
    "compute_calibrated_confidence",
    "compute_consensus",
    "ConsensusMetrics",
]
