"""scidebate — Multi-Agent Debate for scientific fact-checking."""
from dotenv import load_dotenv

# Load .env từ thư mục project (nếu có). override=False: shell env vars có priority cao hơn .env
load_dotenv(override=True)

from .debate import Debate, DebateResult, Turn
from .uncertainty import ConsensusMetrics
from .config import load_config
from .tools import retrieve_evidence

__version__ = "0.2.0"
__all__ = ["Debate", "DebateResult", "Turn", "ConsensusMetrics", "load_config", "retrieve_evidence"]
