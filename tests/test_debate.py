"""Test end-to-end 1 cuộc debate.

Chạy: python tests/test_debate.py

Lưu ý: gọi LLM thật, sẽ mất 3-8 phút trên CPU với qwen2.5:3b và 2 round.
"""

from scidebate.llms.ollama_backend import OllamaLLM
from scidebate import Debate

def main():
    # Init LLM
    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=300)

    # Init Debate
    debate = Debate(llm=llm, max_rounds=2, verbose=True)

    # Test claim
    claim = "Drinking 8 glasses of water daily is necessary for health."

    result = debate.run(claim)

    # Print final result
    print("\n" + "="*20 + " Final Debate Result " + "="*20)
    print(result.summary())

    print("\n--- Transcript ({} turns) ---".format(len(result.transcript)))
    for turn in result.transcript:
        print(f"\n[R{turn.round_num} {turn.speaker}]")
        # Chỉ in 100 ký tự đầu để summary gọn
        print(turn.content[:150] + ("..." if len(turn.content) > 150 else ""))


if __name__ == "__main__":
    main()
