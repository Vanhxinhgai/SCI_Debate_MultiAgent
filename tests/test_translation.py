"""Offline tests cho hỗ trợ tiếng Việt (không gọi API thật).

Chạy: python -m pytest tests/test_translation.py -q
"""
from scidebate.llms import BaseLLM, LLMResponse
from scidebate.translation import is_vietnamese, translate_claim_to_english, translate_to_vietnamese


class FakeLLM(BaseLLM):
    def __init__(self, reply="", fail=False):
        super().__init__(model="fake")
        self.reply, self.fail, self.calls = reply, fail, 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("429 rate limit")
        return LLMResponse(content=self.reply)


def test_detects_vietnamese():
    assert is_vietnamese("Hút thuốc lá gây ung thư phổi.")
    assert is_vietnamese("Đường")
    assert not is_vietnamese("Smoking causes lung cancer.")
    assert not is_vietnamese("")


def test_english_claim_is_not_translated():
    llm = FakeLLM("should not be used")
    assert translate_claim_to_english("Smoking causes lung cancer.", llm) == ("Smoking causes lung cancer.", False)
    assert llm.calls == 0


def test_vietnamese_claim_is_translated_and_cleaned():
    llm = FakeLLM('English claim: "Smoking causes lung cancer."')
    assert translate_claim_to_english("Hút thuốc gây ung thư phổi.", llm) == ("Smoking causes lung cancer.", True)


def test_translation_failure_falls_back_to_original():
    claim = "Hút thuốc gây ung thư phổi."
    assert translate_claim_to_english(claim, FakeLLM(fail=True)) == (claim, False)
    # Model trả lại tiếng Việt → coi như dịch thất bại
    assert translate_claim_to_english(claim, FakeLLM(claim)) == (claim, False)
    assert translate_to_vietnamese("Some text.", FakeLLM(fail=True)) is None
    assert translate_to_vietnamese("", FakeLLM("x")) is None
