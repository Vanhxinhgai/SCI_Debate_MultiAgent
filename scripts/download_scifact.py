"""Tải bộ dữ liệu SciFact (Wadden et al. 2020) vào data/scifact/.

Chạy từ thư mục gốc project:
    python scripts/download_scifact.py
"""
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
TARGET = Path(__file__).resolve().parent.parent / "data" / "scifact"
FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl")


def main() -> int:
    if all((TARGET / f).exists() for f in FILES):
        print(f"SciFact already present in {TARGET}")
        return 0

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()

    TARGET.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in FILES:
                with tar.extractfile(member) as src, (TARGET / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  {name}")

    missing = [f for f in FILES if not (TARGET / f).exists()]
    if missing:
        print(f"Missing after extraction: {missing}", file=sys.stderr)
        return 1
    print(f"Done: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
