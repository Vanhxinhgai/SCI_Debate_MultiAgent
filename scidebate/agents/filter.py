"""FilterAgent — Filter out redundant arguments (Diversity-Aware Retention)."""
import json
import re
from scidebate.prompts import DAR_FILTER_SYSTEM_PROMPT, dar_filter_prompt
from .base import BaseAgent

class FilterAgent(BaseAgent):
    """Filter Agent kiêm moderator để đánh giá và loại bỏ lập luận lặp lại."""

    name = "FILTER"
    system_prompt = DAR_FILTER_SYSTEM_PROMPT

    def evaluate_turn(self, claim: str, history_text: str, new_turn_text: str) -> dict:
        """Đánh giá xem turn mới có mang lại thông tin mới hay không.
        
        Trả về dict chứa:
            - decision: "KEEP" hoặc "DROP"
            - reason: giải thích của LLM
        """
        user_message = dar_filter_prompt(claim, history_text, new_turn_text)
        
        # Gọi trực tiếp LLM không lưu vào memory để tránh phình memory của filter agent
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.llm.generate(messages, max_tokens=300, temperature=0.0)
            content = response.content.strip()
            
            # Trích xuất JSON từ response
            # Đôi khi model bọc JSON trong markdown ```json ... ```
            json_str = content
            if "```json" in content:
                m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                if m:
                    json_str = m.group(1)
            elif "```" in content:
                m = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
                if m:
                    json_str = m.group(1)
                    
            res = json.loads(json_str)
            # Chuẩn hóa kết quả
            decision = res.get("decision", "KEEP").strip().upper()
            if decision not in ("KEEP", "DROP"):
                decision = "KEEP"
            
            return {
                "decision": decision,
                "reason": res.get("reason", "No reason provided.")
            }
        except Exception as e:
            # Fallback: nếu lỗi parse JSON hoặc API lỗi, mặc định là KEEP để tránh mất thông tin
            return {
                "decision": "KEEP",
                "reason": f"Fallback due to error: {e}"
            }
