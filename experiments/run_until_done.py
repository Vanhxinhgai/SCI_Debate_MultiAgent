"""Chạy thực nghiệm SciFact lặp lại cho tới khi xong, tự chờ khi hết quota miễn phí.

Gói miễn phí giới hạn token/request theo ngày (Groq: 200k token/model/24h trượt;
OpenRouter: 50 request/ngày). Script này chạy scifact_eval, nếu còn việc thì nghỉ
--wait phút rồi chạy lại; checkpoint + cache đảm bảo không làm lại phần đã xong.

Chạy (để máy mở, không tắt terminal):
    python -m experiments.run_until_done --n 30

Các tham số khác được chuyển nguyên cho scifact_eval.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments.scifact_eval import ALL_SYSTEMS, RESULTS_DIR, load_dev_claims, load_records

DEFAULT_ARGS = ["--max-rounds", "2", "--max-results", "3",
                "--systems", "zeroshot,ragonly,cot,homo_mad,scidebate"]


def remaining(n: int, seed: int, systems: list[str]) -> int:
    claims = load_dev_claims(n or None, seed)
    ckpt = RESULTS_DIR / f"scifact_dev_n{len(claims)}_seed{seed}_scifact.jsonl"
    done = load_records(ckpt)
    return sum((s, c["id"]) not in done for c in claims for s in systems)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait", type=int, default=30, help="Số phút nghỉ giữa các lượt khi hết quota")
    args, extra = parser.parse_known_args()
    passthrough = extra or DEFAULT_ARGS
    systems = ALL_SYSTEMS
    if "--systems" in passthrough:
        systems = passthrough[passthrough.index("--systems") + 1].split(",")

    cmd = [sys.executable, "-u", "-m", "experiments.scifact_eval",
           "--n", str(args.n), "--seed", str(args.seed), *passthrough]
    while True:
        left = remaining(args.n, args.seed, systems)
        print(f"\n[{datetime.now():%H:%M}] còn {left} lượt (claim × hệ thống).", flush=True)
        if left == 0:
            print("[DONE] Thực nghiệm đã hoàn tất. Bảng kết quả: experiments/results/*_table.md")
            return
        subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
        if remaining(args.n, args.seed, systems) == 0:
            continue
        print(f"[WAIT] Nghỉ {args.wait} phút chờ quota hồi lại... (Ctrl+C để dừng, chạy lại lệnh để tiếp tục)", flush=True)
        time.sleep(args.wait * 60)


if __name__ == "__main__":
    main()
