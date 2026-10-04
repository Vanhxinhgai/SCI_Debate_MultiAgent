"""Hỗ trợ tiếng Việt: dịch claim đầu vào sang tiếng Anh và dịch kết quả sang tiếng Việt.

Pipeline (retrieval, prompts, SciFact corpus) hoạt động bằng tiếng Anh, nên:
    - Claim tiếng Việt  → dịch sang tiếng Anh trước khi chạy debate.
    - Kết quả (lượt nói của agent, lý giải của Judge) → dịch sang tiếng Việt để hiển thị.

Mọi hàm dịch đều không ném lỗi: nếu LLM lỗi thì trả về văn bản gốc, để debate không bị
gián đoạn chỉ vì bước dịch.
"""
import re

from scidebate.llms import BaseLLM

# Các chữ cái/dấu chỉ có trong tiếng Việt (không xuất hiện trong tiếng Anh).
_VIETNAMESE_CHARS = re.compile(
    r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
    re.IGNORECASE,
)

VERDICT_VI = {
    "SUPPORTED": "Được ủng hộ",
    "REFUTED": "Bị bác bỏ",
    "INCONCLUSIVE": "Chưa đủ cơ sở kết luận",
}

QUADRANT_VI = {
    "Strong Consensus": "Đồng thuận mạnh",
    "Genuine Controversy": "Tranh cãi thực sự",
    "Aligned Uncertainty": "Cùng bất định",
    "Confused / Insufficient Evidence": "Mơ hồ / Thiếu bằng chứng",
    "Aligned NEI": "Cùng kết luận chưa đủ bằng chứng",
    "Borderline": "Vùng biên (sát ngưỡng)",
}

LEVEL_VI = {"HIGH": "Cao", "MEDIUM": "Trung bình", "LOW": "Thấp"}

_TO_ENGLISH_PROMPT = """Translate the following Vietnamese scientific claim into one concise English scientific claim.

Rules:
- Preserve the exact meaning, including qualifiers (always, never, all, some, may, increases, reduces...).
- Do NOT add, remove, soften or strengthen anything. Do NOT answer or evaluate the claim.
- Use standard English scientific terminology.
- Output ONLY the English sentence, without quotes or explanations.

Vietnamese claim: {claim}"""

_TO_VIETNAMESE_PROMPT = """Dịch đoạn văn khoa học sau sang tiếng Việt tự nhiên, chính xác.

Quy tắc:
- Giữ nguyên ý nghĩa, không thêm bớt, không tóm tắt.
- Giữ nguyên các ký hiệu trích dẫn như [1], [2], tên tác giả, tên bài báo, số liệu.
- Giữ nguyên các nhãn SUPPORTED, REFUTED, INCONCLUSIVE, PRO, CON, JUDGE.
- Thuật ngữ chuyên ngành: dịch sang tiếng Việt và ghi thuật ngữ gốc trong ngoặc ở lần xuất hiện đầu tiên, ví dụ "thử nghiệm ngẫu nhiên có đối chứng (RCT)".
- Chỉ trả về bản dịch, không giải thích thêm.

Đoạn văn:
{text}"""


def is_vietnamese(text: str) -> bool:
    """True nếu văn bản có chữ cái/dấu đặc trưng của tiếng Việt."""
    return bool(_VIETNAMESE_CHARS.search(text or ""))


def _clean(output: str) -> str:
    text = (output or "").strip()
    # Một số model bọc kết quả trong dấu nháy hoặc thêm tiền tố "English:" / "Bản dịch:".
    text = re.sub(r"^(english( claim)?|translation|bản dịch)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"“”').strip()


def translate_claim_to_english(claim: str, llm: BaseLLM) -> tuple[str, bool]:
    """Trả về (claim tiếng Anh, có_dịch_hay_không). Claim không phải tiếng Việt giữ nguyên."""
    claim = (claim or "").strip()
    if not is_vietnamese(claim):
        return claim, False
    try:
        response = llm.generate(
            [{"role": "user", "content": _TO_ENGLISH_PROMPT.format(claim=claim)}],
            temperature=0.0,
            max_tokens=200,
        )
        english = _clean(response.content).splitlines()[0].strip() if response.content.strip() else ""
        if english and not is_vietnamese(english):
            return english, True
    except Exception as e:
        print(f"[translation] claim → English failed: {e}")
    return claim, False


def translate_to_vietnamese(text: str, llm: BaseLLM, max_tokens: int = 1500) -> str | None:
    """Dịch một đoạn kết quả sang tiếng Việt. Trả về None nếu lỗi (để UI tự bỏ qua)."""
    if not (text or "").strip():
        return None
    try:
        response = llm.generate(
            [{"role": "user", "content": _TO_VIETNAMESE_PROMPT.format(text=text)}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        translated = _clean(response.content)
        return translated or None
    except Exception as e:
        print(f"[translation] → Vietnamese failed: {e}")
        return None
