"""ProAgent — bảo vệ claim."""
from scidebate.prompts import PRO_SYSTEM_PROMPT
from .base import BaseAgent

class ProAgent(BaseAgent):
    """Agent đứng về phía claim, đưa bằng chứng và lập luận ủng hộ."""

    name = "PRO"
    system_prompt = PRO_SYSTEM_PROMPT
