"""Test ChatHistoryMemory.

Chạy: python tests/test_memory.py
"""

from scidebate.memory import ChatHistoryMemory

def main():
    mem = ChatHistoryMemory()
    print("Empty memory:", mem)
    print("Length:", len(mem))

    # Add messages
    mem.add("system", "You are a helpful assistant.")
    mem.add("user", "What is the capital of France?")
    mem.add("assistant", "The capital of France is Paris.")
    
    print("Memory with messages:", mem)
    print("Length:", len(mem))

    for i, msg in enumerate(mem.get_messages()):
        print(f"Message {i}: role={msg['role']}, content={msg['content']}")

    # Clear memory
    mem.clear()
    print("After clearing memory:", mem)
    print("Length:", len(mem))

    # Test invalid role
    try:
        mem.add("invalid_role", "This should raise an error.")
        print("ERROR: should have raised ValueError")

    except ValueError as e:
        print(f"\nValidation OK: {e}")

if __name__ == "__main__":
    main()
