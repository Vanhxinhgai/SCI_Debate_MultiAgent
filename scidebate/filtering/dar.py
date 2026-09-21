"""DAR (Diversity-Aware Retention) Filtering core functions."""
from scidebate.agents.filter import FilterAgent
from scidebate.llms import BaseLLM

def apply_dar_filtering(claim: str, retained_turns: list, new_turn, filter_llm: BaseLLM) -> dict:
    """Áp dụng bộ lọc DAR cho một lượt tranh luận mới.
    
    Args:
        claim: Tuyên bố khoa học đang tranh biện
        retained_turns: Danh sách các Turn đã được giữ lại trước đó (Turn objects)
        new_turn: Lượt tranh luận mới cần đánh giá (Turn object)
        filter_llm: LLM được sử dụng để chạy Filter Agent
        
    Returns:
        dict: {"decision": "KEEP"|"DROP", "reason": str}
    """
    # Xây dựng văn bản lịch sử các lượt đã giữ lại
    history_lines = []
    for turn in retained_turns:
        history_lines.append(f"[{turn.speaker} (Round {turn.round_num})]: {turn.content}")
    history_text = "\n\n".join(history_lines) if history_lines else "No history yet (this is the first turn)."
    
    new_turn_text = f"[{new_turn.speaker} (Round {new_turn.round_num})]: {new_turn.content}"
    
    filter_agent = FilterAgent(llm=filter_llm)
    return filter_agent.evaluate_turn(claim, history_text, new_turn_text)
