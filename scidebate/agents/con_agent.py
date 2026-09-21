from scidebate.prompts import CON_SYSTEM_PROMPT
from .base import BaseAgent


class ConAgent(BaseAgent):
    name = "CON"
    system_prompt = CON_SYSTEM_PROMPT
