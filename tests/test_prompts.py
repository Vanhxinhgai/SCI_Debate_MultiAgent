"""Test prompt templates.

Chạy: python tests/test_prompts.py
"""

from scidebate.prompts import (
    PRO_SYSTEM_PROMPT,
    CON_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    pro_opening_prompt,
    con_opening_prompt,
    pro_rebuttal_prompt,
    con_rebuttal_prompt,
    judge_verdict_prompt,
)

def main():
    claim = "Vitamin C megadoses (>1000mg/day) prevent the common cold."

    print("=" * 60)
    print("SYSTEM PROMPTS")
    print("=" * 60)
    print("\n[PRO SYSTEM]")
    print(PRO_SYSTEM_PROMPT[:200], "...")
    print("\n[CON SYSTEM]")
    print(CON_SYSTEM_PROMPT[:200], "...")
    print("\n[JUDGE SYSTEM]")
    print(JUDGE_SYSTEM_PROMPT[:200], "...")

    print("\n" + "=" * 60)
    print("TURN PROMPTS (rendered with sample claim)")
    print("=" * 60)
    print("\n[PRO OPENING]")
    print(pro_opening_prompt(claim))

    print("\n[CON OPENING]")
    print(con_opening_prompt(claim, "Vitamin C boosts immune cell activity..."))

    print("\n[JUDGE VERDICT]")
    print(judge_verdict_prompt(claim, "<full transcript here>")[:300], "...")

    print("\n" + "=" * 60)
    print("All prompts loaded successfully.")

if __name__ == "__main__":
    main()
