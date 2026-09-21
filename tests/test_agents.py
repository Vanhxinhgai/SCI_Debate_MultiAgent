"""Test các agent với Ollama thật.

Chạy: python tests/test_agents.py

Lưu ý: test này gọi LLM thật, sẽ mất 30s-2 phút trên CPU.
"""

from scidebate.llms.ollama_backend import OllamaLLM
from scidebate.agents import ProAgent, ConAgent, JudgeAgent

def test_pro_agent():
    print("\n" + "="*20 + " Testing ProAgent " + "="*20)

    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=200)
    pro_agent = ProAgent(llm=llm)
    print("Created, pro")

    response = pro_agent.respond(
        "The scientific claim: \"Drinking 8 glasses of water daily is necessary for health.\"\n"
        "Give your opening argument in 2-3 sentences."
    )

    print("\nProAgent response:")
    print(response)
    print(f"\nMemory size after: {len(pro_agent.memory)}")

def test_con_agent():
    print("\n" + "="*20 + " Testing ConAgent " + "="*20)

    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=200)
    con_agent = ConAgent(llm=llm)
    print("Created, con")

    response = con_agent.respond(
        "The scientific claim: \"Drinking 8 glasses of water daily is necessary for health.\"\n"
        "Give your opening argument in 2-3 sentences."
    )

    print("\nConAgent response:")
    print(response)
    print(f"\nMemory size after: {len(con_agent.memory)}")

def test_judge_agent():
    print("\n" + "="*20 + " Testing JudgeAgent " + "="*20)

    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=200)
    judge_agent = JudgeAgent(llm=llm)
    print("Created, judge")

    fake_transcript = """
    PRO: Studies show hydration improves cognitive function.
    CON: The 8-glasses rule has no scientific basis; individual needs vary widely.
    """

    prompt = (
        f"The scientific claim: \"Drinking 8 glasses of water daily is necessary for health.\"\n\n"
        f"Debate transcript:\n{fake_transcript}\n\n"
        f"Output STRICTLY:\n"
        f"VERDICT: <SUPPORTED | REFUTED | INCONCLUSIVE>\n"
        f"CONFIDENCE: <0.0 to 1.0>\n"
        f"JUSTIFICATION: <2-3 sentences>"
    )
    raw = judge_agent.respond(prompt)
    print("\nJudgeAgent raw response:")
    print(raw)

    verdict = judge_agent.parse_verdict(raw)
    print("\nParsed Verdict:")
    print(f"Verdict: {verdict.verdict}")
    print(f"Confidence: {verdict.confidence}")
    print(f"Justification: {verdict.justification}")

def test_reset():
    print("\n" + "=" * 60)
    print("TEST 4: Agent.reset()")
    print("=" * 60)

    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=50)
    pro = ProAgent(llm=llm)
    initial_size = len(pro.memory)
    print(f"Initial memory size (should be 1, system prompt): {initial_size}")

    pro.respond("Brief say hi.")
    print(f"After 1 turn: {len(pro.memory)} (should be 3)")

    pro.reset()
    print(f"After reset: {len(pro.memory)} (should be back to 1)")

if __name__ == "__main__":
    test_pro_agent()
    test_con_agent()
    test_judge_agent()
    test_reset()
    print("\n" + "=" * 60)
    print("All agent tests passed.")
    print("=" * 60)
