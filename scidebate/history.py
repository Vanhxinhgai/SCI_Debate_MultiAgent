"""Lịch sử kiểm chứng — lưu mỗi lần debate thành một file JSON để xem lại sau.

Mỗi bản ghi chứa đủ dữ liệu để hiển thị lại toàn bộ kết quả mà KHÔNG cần gọi API:
claim (gốc + bản dịch), cấu hình, tài liệu, transcript (kèm bản dịch tiếng Việt),
phán quyết và chỉ số đồng thuận.

Thư mục mặc định: outputs/history/ (đã nằm trong .gitignore).
Có thể đổi bằng biến môi trường SCIDEBATE_HISTORY_DIR.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

RECORD_VERSION = 1
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def history_dir(base: str | Path | None = None) -> Path:
    return Path(base or os.environ.get("SCIDEBATE_HISTORY_DIR") or "outputs/history")


def _path_for(run_id: str, base: str | Path | None = None) -> Path:
    # Chặn path traversal: id chỉ gồm chữ, số, "_" và "-".
    if not _ID_PATTERN.match(run_id or ""):
        raise ValueError(f"Invalid run id: {run_id!r}")
    return history_dir(base) / f"{run_id}.json"


def _to_plain(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def build_record(
    result,
    *,
    original_claim: str,
    english_claim: str,
    settings: dict,
    pro_papers: list[dict],
    con_papers: list[dict],
    turn_translations: list[str | None] | None = None,
    verdict_translation: str | None = None,
) -> dict:
    """Chuyển DebateResult (+ dữ liệu hiển thị) thành dict JSON-serializable."""
    turn_translations = turn_translations or []
    transcript = []
    for i, turn in enumerate(result.transcript):
        transcript.append({
            "round_num": turn.round_num,
            "speaker": turn.speaker,
            "model": turn.model,
            "content": turn.content,
            "filtered_out": turn.filtered_out,
            "filter_reason": turn.filter_reason,
            "vi": turn_translations[i] if i < len(turn_translations) else None,
        })

    verdict = None
    if result.verdict:
        v = result.verdict
        verdict = {
            "verdict": v.verdict,
            "confidence": v.confidence,
            "justification": v.justification,
            "raw_output": v.raw_output,
            "vi": verdict_translation,
        }

    now = datetime.now()
    return {
        "version": RECORD_VERSION,
        "id": f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "created_at": now.isoformat(timespec="seconds"),
        "claim_original": original_claim,
        "claim_english": english_claim,
        "settings": settings,
        "pro_papers": pro_papers or [],
        "con_papers": con_papers or [],
        "transcript": transcript,
        "verdict": verdict,
        "num_rounds": result.num_rounds,
        "elapsed_seconds": result.elapsed_seconds,
        "parallel_opening_used": result.parallel_opening_used,
        "consensus": _to_plain(result.consensus),
        "consensus_error": getattr(result, "_consensus_error", None),
    }


def save_run(record: dict, base: str | Path | None = None) -> Path:
    path = _path_for(record["id"], base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)  # ghi nguyên tử: không để lại file hỏng nếu app bị tắt giữa chừng
    return path


def load_run(run_id: str, base: str | Path | None = None) -> dict | None:
    path = _path_for(run_id, base)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_runs(base: str | Path | None = None) -> list[dict]:
    """Tóm tắt các lần chạy, mới nhất trước. Bỏ qua file hỏng."""
    root = history_dir(base)
    if not root.exists():
        return []
    runs = []
    for path in root.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append({
            "id": rec.get("id", path.stem),
            "created_at": rec.get("created_at", ""),
            "claim": rec.get("claim_original") or rec.get("claim_english", ""),
            "verdict": (rec.get("verdict") or {}).get("verdict", "—"),
        })
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def record_to_markdown(rec: dict) -> str:
    """Báo cáo Markdown của một lần debate (để tải về / đính kèm báo cáo)."""
    s = rec.get("settings") or {}
    models = s.get("models") or {}
    v = rec.get("verdict") or {}
    lines = [
        "# SciDebate — Debate Report",
        "",
        f"- **Date:** {rec.get('created_at', '').replace('T', ' ')}",
        f"- **Claim:** {rec.get('claim_original', '')}",
    ]
    if rec.get("claim_english") and rec.get("claim_english") != rec.get("claim_original"):
        lines.append(f"- **Claim (English, used for verification):** {rec['claim_english']}")
    lines += [
        f"- **Models:** PRO `{models.get('pro', '?')}` · CON `{models.get('con', '?')}` · JUDGE `{models.get('judge', '?')}`",
        f"- **Rounds:** {rec.get('num_rounds', '?')} · **Evidence:** "
        + (f"{s.get('rag_source')} ({s.get('rag_max_results')} docs/side)" if s.get("use_rag") else "disabled"),
        "",
        "## Verdict",
        "",
        f"**{v.get('verdict', '—')}** — confidence {v.get('confidence', 0):.0%}",
        "",
        v.get("justification", ""),
    ]
    if v.get("vi"):
        lines += ["", f"> **Tiếng Việt:** {v['vi']}"]

    c = rec.get("consensus")
    if c:
        lines += [
            "", "## Consensus Metrics", "",
            f"- Zone: **{c.get('consensus_quadrant')}** (level {c.get('consensus_level')})",
            f"- Normalized entropy (Judge): {c.get('normalized_entropy', 0):.3f}",
            f"- JSD (PRO vs CON): {c.get('jsd', 0):.3f}",
            f"- Calibrated confidence: {c.get('calibrated_confidence', 0):.2f}",
        ]

    lines += ["", "## Debate Transcript", ""]
    for t in rec.get("transcript") or []:
        tag = " *(filtered by DAR)*" if t.get("filtered_out") else ""
        lines += [f"### Round {t.get('round_num')} — {t.get('speaker')}{tag}", "", t.get("content", "")]
        if t.get("vi"):
            lines += ["", f"> **Tiếng Việt:** {t['vi']}"]
        lines.append("")

    for side, key in (("PRO", "pro_papers"), ("CON", "con_papers")):
        papers = rec.get(key) or []
        if papers:
            lines += [f"## Evidence — {side}", ""]
            for i, p in enumerate(papers, 1):
                link = f" — {p['pdf_link']}" if p.get("pdf_link") else ""
                lines.append(f"{i}. **{p.get('title', '')}** ({p.get('source', '')}{', ' + str(p['doc_id']) if p.get('doc_id') else ''}){link}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def delete_run(run_id: str, base: str | Path | None = None) -> bool:
    path = _path_for(run_id, base)
    if path.exists():
        path.unlink()
        return True
    return False
