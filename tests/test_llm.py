"""Test nhanh LLM backend.

Chạy: python tests/test_llm.py
"""
from scidebate.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="qwen2.5:3b", temperature=0, max_tokens=30)
    print("LLM created:", llm)

    messages = [{"role": "user", "content": "Reply with exactly: HELLO_FROM_OLLAMA"}]
    response = llm.generate(messages)

    print("Content:", response.content)
    print(
        "Tokens (prompt/completion/total):",
        response.prompt_tokens, "/",
        response.completion_tokens, "/",
        response.total_tokens,
    )
    print("Model:", response.model)


if __name__ == "__main__":
    main()
