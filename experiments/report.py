"""Generate evaluation report from saved results JSON.

Usage:
    python experiments/report.py                            # latest results file
    python experiments/report.py --input experiments/results/eval_20250526_120000.json
    python experiments/report.py --format latex            # LaTeX table
    python experiments/report.py --format markdown         # Markdown table (default)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM_DISPLAY = {
    "zeroshot": "Zero-Shot LLM",
    "ragonly": "RAG-Only",
    "debate_norag": "Debate (no RAG)",
    "mad": "MAD+RAG (Ours)",
}

SYSTEM_ORDER = ["zeroshot", "ragonly", "debate_norag", "mad"]


def _load_results(input_path: str = None) -> dict:
    if input_path:
        p = Path(input_path)
    else:
        files = sorted(RESULTS_DIR.glob("eval_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No result files found in {RESULTS_DIR}")
        p = files[0]
    print(f"[report] Loading: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, pct=False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.4f}"


# ── Markdown report ────────────────────────────────────────────────────────────

def _markdown_table(metrics: dict[str, dict]) -> str:
    headers = ["System", "Accuracy", "Macro F1", "Precision", "Recall", "ECE ↓", "Brier ↓"]
    rows = []
    for sys_name in SYSTEM_ORDER:
        if sys_name not in metrics:
            continue
        m = metrics[sys_name]
        display = SYSTEM_DISPLAY.get(sys_name, sys_name)
        marker = " **" if sys_name == "mad" else ""
        rows.append([
            f"{display}{marker}",
            _fmt(m.get("accuracy"), pct=True),
            _fmt(m.get("macro_f1"), pct=True),
            _fmt(m.get("macro_precision"), pct=True),
            _fmt(m.get("macro_recall"), pct=True),
            _fmt(m.get("ece")),
            _fmt(m.get("brier_score")),
        ])

    col_w = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"

    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


def _markdown_per_class(metrics: dict[str, dict]) -> str:
    lines = []
    for sys_name in SYSTEM_ORDER:
        if sys_name not in metrics:
            continue
        m = metrics[sys_name]
        display = SYSTEM_DISPLAY.get(sys_name, sys_name)
        per_class = m.get("per_class_f1", {})
        lines.append(f"\n#### {display}")
        sub_headers = ["Verdict", "Precision", "Recall", "F1", "Support"]
        rows = []
        for label in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]:
            pc = per_class.get(label, {})
            rows.append([
                label,
                _fmt(pc.get("precision"), pct=True),
                _fmt(pc.get("recall"), pct=True),
                _fmt(pc.get("f1"), pct=True),
                str(pc.get("support", 0)),
            ])
        col_w = [max(len(sub_headers[i]), max(len(r[i]) for r in rows)) for i in range(5)]
        sep = "| " + " | ".join("-" * w for w in col_w) + " |"
        header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(sub_headers)) + " |"
        lines.append(header)
        lines.append(sep)
        for row in rows:
            lines.append("| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


def _markdown_by_category(metrics: dict[str, dict]) -> str:
    lines = []
    headers = ["System", "SUPPORTED", "REFUTED", "INCONCLUSIVE"]
    rows = []
    for sys_name in SYSTEM_ORDER:
        if sys_name not in metrics:
            continue
        m = metrics[sys_name]
        by_cat = m.get("by_category", {})
        display = SYSTEM_DISPLAY.get(sys_name, sys_name)
        rows.append([
            display,
            _fmt(by_cat.get("SUPPORTED", {}).get("accuracy"), pct=True),
            _fmt(by_cat.get("REFUTED", {}).get("accuracy"), pct=True),
            _fmt(by_cat.get("INCONCLUSIVE", {}).get("accuracy"), pct=True),
        ])
    col_w = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(4)]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines.extend([header, sep])
    for row in rows:
        lines.append("| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


def _markdown_per_claim(records: list[dict]) -> str:
    lines = ["| ID | Claim (truncated) | GT | Zero-Shot | RAG-Only | Debate-noRAG | MAD+RAG |",
             "|----|-------------------|----|-----------|----------|--------------|---------|"]
    claim_ids = sorted(set(r["claim_id"] for r in records))
    sys_map_key = {"zeroshot": "Zero-Shot", "ragonly": "RAG-Only",
                   "debate_norag": "Debate-noRAG", "mad": "MAD+RAG"}

    for cid in claim_ids:
        rows_for_claim = {r["system"]: r for r in records if r["claim_id"] == cid}
        gt = next((r["ground_truth"] for r in rows_for_claim.values()), "?")
        claim_txt = next((r["claim"][:45] + "…" for r in rows_for_claim.values()), "")

        def cell(sys):
            r = rows_for_claim.get(sys)
            if not r:
                return "—"
            pred = r["predicted"]
            ok = "✓" if r.get("correct") else "✗"
            short = pred[:3]
            return f"{ok}{short}"

        lines.append(
            f"| {cid} | {claim_txt} | {gt[:3]} | {cell('zeroshot')} | {cell('ragonly')} | {cell('debate_norag')} | {cell('mad')} |"
        )
    return "\n".join(lines)


def generate_markdown(data: dict) -> str:
    metrics = data.get("system_metrics", {})
    records = data.get("records", [])
    mode = data.get("mode", "?")
    ts = data.get("timestamp", "?")
    n_claims = data.get("total_claims", len(set(r["claim_id"] for r in records)))
    systems = data.get("systems_evaluated", list(metrics.keys()))
    status = data.get("status", "complete")

    mad_acc = metrics.get("mad", {}).get("accuracy", 0)
    zs_acc = metrics.get("zeroshot", {}).get("accuracy", 0)
    gain = (mad_acc - zs_acc) * 100

    lines = [
        f"# Evaluation Report — SCI_Debate_MultiAgent",
        f"",
        f"- **Date**: {ts}",
        f"- **Mode**: {mode} ({n_claims} claims)",
        f"- **Systems**: {', '.join(SYSTEM_DISPLAY.get(s, s) for s in systems)}",
        f"- **Status**: {status}",
        f"",
        f"---",
        f"",
        f"## 1. Overall Performance",
        f"",
        _markdown_table(metrics),
        f"",
        f"> **Our system (MAD+RAG)** achieves **{mad_acc*100:.1f}% accuracy** — "
        f"a **+{gain:.1f}pp** gain over the zero-shot baseline.",
        f"",
        f"---",
        f"",
        f"## 2. Per-Category Accuracy",
        f"",
        _markdown_by_category(metrics),
        f"",
        f"---",
        f"",
        f"## 3. Per-Class F1 Breakdown",
        f"",
        _markdown_per_class(metrics),
        f"",
        f"---",
        f"",
        f"## 4. Per-Claim Results",
        f"",
        _markdown_per_claim(records),
        f"",
        f"---",
        f"",
        f"## 5. Confusion Matrix — MAD+RAG",
        f"",
        f"```",
        metrics.get("mad", {}).get("confusion_matrix", "Not available"),
        f"```",
        f"",
        f"---",
        f"",
        f"## 6. System Comparison Notes",
        f"",
        f"| Aspect | Zero-Shot | RAG-Only | Debate-no-RAG | MAD+RAG (Ours) |",
        f"|--------|-----------|----------|---------------|----------------|",
        f"| Evidence retrieval | ✗ | ✓ | ✗ | ✓ |",
        f"| Multi-agent debate | ✗ | ✗ | ✓ | ✓ |",
        f"| Universal claim handling | ✗ | ✗ | ✓ | ✓ |",
        f"| Uncertainty quantification | ✗ | ✗ | ✗ | ✓ (JSD + Entropy) |",
        f"| Domain-aware retrieval | — | Basic | — | ✓ (Group A/B filter) |",
        f"| Calibrated confidence | ✗ | ✗ | ✗ | ✓ (ECE-corrected) |",
        f"",
    ]
    return "\n".join(lines)


# ── LaTeX table ────────────────────────────────────────────────────────────────

def generate_latex(data: dict) -> str:
    metrics = data.get("system_metrics", {})
    rows = []
    for sys_name in SYSTEM_ORDER:
        if sys_name not in metrics:
            continue
        m = metrics[sys_name]
        display = SYSTEM_DISPLAY.get(sys_name, sys_name)
        bold = sys_name == "mad"
        def b(v, pct=False):
            s = _fmt(v, pct=pct)
            return f"\\textbf{{{s}}}" if bold else s
        rows.append(
            f"  {display} & {b(m.get('accuracy'),True)} & {b(m.get('macro_f1'),True)} & "
            f"{b(m.get('macro_precision'),True)} & {b(m.get('macro_recall'),True)} & "
            f"{b(m.get('ece'))} & {b(m.get('brier_score'))} \\\\"
        )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of system performance on the 20-claim test set.}",
        r"\label{tab:eval_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"System & Accuracy & Macro F1 & Precision & Recall & ECE $\downarrow$ & Brier $\downarrow$ \\",
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to eval results JSON (default: latest in experiments/results/)")
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="markdown")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file (auto-named if not specified)")
    args = parser.parse_args()

    data = _load_results(args.input)

    if args.format in ("markdown", "both"):
        md = generate_markdown(data)
        out = args.output or str(RESULTS_DIR / f"report_{data.get('timestamp','latest')}.md")
        Path(out).write_text(md, encoding="utf-8")
        print(f"[report] Markdown saved: {out}")
        print("\n" + md[:3000])

    if args.format in ("latex", "both"):
        tex = generate_latex(data)
        out_tex = (args.output or str(RESULTS_DIR / f"report_{data.get('timestamp','latest')}")) + ".tex"
        Path(out_tex).write_text(tex, encoding="utf-8")
        print(f"[report] LaTeX saved: {out_tex}")
        print("\n" + tex)


if __name__ == "__main__":
    main()
