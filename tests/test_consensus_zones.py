"""Tests cho bản đồ đồng thuận 6 vùng.

Chạy: python -m pytest tests/test_consensus_zones.py -q
"""
import pytest

from scidebate.uncertainty.consensus import _determine_consensus_level, _determine_quadrant

SUP = {"SUPPORTED": 0.8, "REFUTED": 0.1, "INCONCLUSIVE": 0.1}
NEI = {"SUPPORTED": 0.1, "REFUTED": 0.1, "INCONCLUSIVE": 0.8}


@pytest.mark.parametrize("entropy, jsd, dominant, pro, con, expected", [
    (0.1, 0.1, "SUPPORTED", SUP, SUP, "Strong Consensus"),
    (0.1, 0.1, "INCONCLUSIVE", SUP, SUP, "Aligned NEI"),         # Judge settles on NEI
    (0.0, 0.0, "SUPPORTED", NEI, NEI, "Aligned NEI"),            # both agents settle on NEI
    (0.1, 0.1, "SUPPORTED", NEI, SUP, "Strong Consensus"),       # only one agent NEI
    (0.1, 0.8, "SUPPORTED", SUP, NEI, "Genuine Controversy"),
    (0.9, 0.1, "SUPPORTED", SUP, SUP, "Aligned Uncertainty"),
    (0.9, 0.8, "SUPPORTED", SUP, NEI, "Confused / Insufficient Evidence"),
    (0.52, 0.1, "SUPPORTED", SUP, SUP, "Borderline"),            # near entropy threshold
    (0.1, 0.38, "SUPPORTED", SUP, SUP, "Borderline"),            # near JSD threshold
])
def test_six_zones(entropy, jsd, dominant, pro, con, expected):
    assert _determine_quadrant(entropy, jsd, dominant, pro, con) == expected


def test_every_zone_has_level_and_explanation():
    for zone in ["Strong Consensus", "Aligned NEI", "Genuine Controversy",
                 "Aligned Uncertainty", "Confused / Insufficient Evidence", "Borderline"]:
        level, explanation = _determine_consensus_level(zone)
        assert level in {"HIGH", "MEDIUM", "LOW"} and explanation
    assert _determine_consensus_level("Aligned NEI")[0] == "LOW"
    assert _determine_consensus_level("Strong Consensus")[0] == "HIGH"
