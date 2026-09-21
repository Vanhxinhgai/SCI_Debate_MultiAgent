"""ConAgent — phản biện claim."""
from scidebate.prompts import CON_SYSTEM_PROMPT
from .base import BaseAgent


class ConAgent(BaseAgent):
    """Agent đứng phía phản biện, tìm điểm yếu và bằng chứng đối lập."""

    name = "CON"
    system_prompt = CON_SYSTEM_PROMPT