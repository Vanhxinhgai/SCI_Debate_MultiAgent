"""JudgeAgent — phán quyết cuối cùng."""
import re
from dataclasses import dataclass
from scidebate.prompts import JUDGE_SYSTEM_PROMPT
from .base import BaseAgent

@dataclass
class Verdict:
    """Kết quả phán quyết từ Judge."""
    verdict: str # "SUPPORTED" | "REFUTED" | "INCONCLUSIVE"
    confidence: float # 0.0 - 1.0
    justification: str # Giai thich ngan gon
    raw_output: str # full response tu LLM, de debug

class JudgeAgent(BaseAgent):
    """Agent tổng hợp arguments từ Pro và Con, đưa verdict."""

    name = "JUDGE"
    system_prompt = JUDGE_SYSTEM_PROMPT

    def parse_verdict(self, response_text: str) -> Verdict:
        """Parse output của Judge thành Verdict struct.

        Expected format:
            VERDICT: <SUPPORTED | REFUTED | INCONCLUSIVE>
            CONFIDENCE: <0.0 to 1.0>
            JUSTIFICATION: <text>

        Nếu LLM không tuân thủ format -> fallback: trả INCONCLUSIVE với confidence 0.0
        """

        verdict = "INCONCLUSIVE"
        confidence = 0.0
        justification = response_text.strip()

        # Match VERDICT line
        m = re.search(rf"VERDICT:\s*(SUPPORTED|REFUTED|INCONCLUSIVE)", response_text, re.IGNORECASE)

        if m:
            verdict = m.group(1).upper()
        
        # Match CONFIDENCE line
        m = re.search(rf"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", response_text, re.IGNORECASE)
        if m:
            try:
                confidence = float(m.group(1))
                confidence = max(0.0, min(1.0, confidence)) # clamp to [0.0, 1.0]
            except ValueError:
                pass
        
        # Match JUSTIFICATION line
        m = re.search(r"JUSTIFICATION:\s*(.+?)(?:\n\n|\Z)", response_text, re.DOTALL)

        if m:
            justification = m.group(1).strip()
        
        return Verdict(
            verdict=verdict,
            confidence=confidence,
            justification=justification,
            raw_output=response_text
        )
