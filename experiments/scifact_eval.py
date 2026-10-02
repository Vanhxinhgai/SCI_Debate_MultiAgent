"""Đánh giá trên SciFact dev — có checkpoint/resume, cache LLM, rate limit.

Chạy từ thư mục gốc project (cần data/scifact/corpus.jsonl và claims_dev.jsonl):

    # Chạy 100 claim (seed cố định), tất cả hệ thống, theo thứ tự claim-major:
    python -m experiments.scifact_eval --n 100

    # Hết quota giữa chừng? Chạy lại đúng lệnh trên — claim đã xong sẽ bị bỏ qua,
    # request đã gọi được lấy từ cache (không tốn quota).

    # Chỉ in lại bảng kết quả từ file checkpoint:
    python -m experiments.scifact_eval --n 100 --report-only

Hệ thống (--systems):
    zeroshot          1 LLM, không RAG, không debate
    ragonly           1 LLM + evidence (neutral retrieval)
    cot               như ragonly + Chain-of-Thought
    homo_mad          Pro/Con/Judge cùng base model, retrieval theo lập trường
    scidebate         Heter-MAD (config heter_mad), retrieval theo lập trường
    scidebate_shared  Ablation: Heter-MAD nhưng Pro/Con dùng chung neutral retrieval

So sánh cặp:  cot → homo_mad      : đóng góp của debate
              homo_mad → scidebate : đóng góp của heterogeneous models
              scidebate_shared → scidebate : đóng góp của stance-aware retrieval

Gold evidence KHÔNG được dùng: retrieval là TF-IDF trên corpus (tránh lộ nhãn).
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from collections import Counter
from pathlib import Path

from scidebate import Debate, load_config
from scidebate.llms import CachedLLM, GroqLLM, OpenRouterLLM, OllamaLLM
from scidebate.llms.cached import QuotaExhaustedError
from scidebate.tools import retrieve_evidence
from scidebate.tools.scifact import load_scifact_dataset

from experiments.baselines import CoTBaseline, RAGOnlyBaseline, ZeroShotBaseline
from experiments.metrics import (
    bootstrap_ci,
    compute_all_metrics,
    confusion_matrix_str,
    mcnemar_exact,
)

RESULTS_DIR = Path(__file__).parent / "results"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "debate_config.yaml"
ALL_SYSTEMS = ["zeroshot", "ragonly", "cot", "homo_mad", "scidebate", "scidebate_shared"]
REFERENCE_SYSTEM = "scidebate"

SCIFACT_TO_VERDICT = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}


# ── Data ──────────────────────────────────────────────────────────────────────

def _gold_verdict(claim_row: dict) -> str:
    """SciFact: không có evidence → NEI (INCONCLUSIVE); ngược lại lấy nhãn đa số."""
    labels = []
    for entries in (claim_row.get("evidence") or {}).values():
        if isinstance(entries, dict):
            entries = [entries]
        labels.extend(e.get("label") for e in entries or [])
    labels = [SCIFACT_TO_VERDICT[l] for l in labels if l in SCIFACT_TO_VERDICT]
    if not labels:
        return "INCONCLUSIVE"
    return Counter(labels).most_common(1)[0][0]


def load_dev_claims(n: int | None, seed: int) -> list[dict]:
    dataset = load_scifact_dataset()
    dev = [r for r in dataset["claims"] if r.get("_split") == "dev"]
    if not dev:
        raise SystemExit("Không tìm thấy data/scifact/claims_dev.jsonl.")
    claims = [
        {"id": str(r["id"]), "claim": r["claim"], "ground_truth": _gold_verdict(r)}
        for r in sorted(dev, key=lambda r: int(r["id"]))
    ]
    if n and n < len(claims):
        claims = random.Random(seed).sample(claims, n)
    return claims


# ── LLM factory ───────────────────────────────────────────────────────────────

class LLMFactory:
    """Tạo LLM từ config, luôn bọc CachedLLM (cache + rate limit + retry)."""

    def __init__(self, config: dict, cache_path: Path, max_tokens: int):
        self.config = config
        self.cache_path = cache_path
        self.max_tokens = max_tokens
        self.rpm = config.get("eval_settings", {}).get("rpm", {})

    def make(self, cfg: dict) -> CachedLLM:
        backend = cfg.get("backend", "Groq (cloud)")
        model = cfg["model"]
        temp = cfg.get("temperature", 0.0)
        if "Groq" in backend:
            inner, rpm = GroqLLM(model=model, temperature=temp, max_tokens=self.max_tokens), self.rpm.get("groq", 30)
        elif "OpenRouter" in backend:
            inner, rpm = OpenRouterLLM(model=model, temperature=temp, max_tokens=self.max_tokens), self.rpm.get("openrouter", 20)
        elif "Ollama" in backend:
            inner, rpm = OllamaLLM(model=model, temperature=temp, max_tokens=self.max_tokens), 0
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return CachedLLM(inner, cache_path=self.cache_path, rpm=rpm)

    def base(self) -> CachedLLM:
        """Base model cho baseline và Homo-MAD (mặc định: model của Pro trong heter_mad)."""
        heter = self.config["llm_backends"]["heter_mad"]
        cfg = self.config.get("eval_settings", {}).get("base_model") or heter["pro"]
        return self.make(cfg)

    def heter(self) -> tuple[CachedLLM, CachedLLM, CachedLLM]:
        heter = self.config["llm_backends"]["heter_mad"]
        return self.make(heter["pro"]), self.make(heter["con"]), self.make(heter["judge"])


# ── Runners ───────────────────────────────────────────────────────────────────

def _debate_kwargs(config: dict, args) -> dict:
    return dict(
        max_rounds=config.get("debate_settings", {}).get("max_rounds", 2),
        parallel_opening=False,
        compute_uncertainty=args.uncertainty,
        n_uncertainty_samples=config.get("uncertainty_settings", {}).get("n_samples", 4),
        enable_early_stopping=args.uncertainty,
        use_logprobs=config.get("uncertainty_settings", {}).get("use_logprobs", True),
        use_dar=config.get("dar_settings", {}).get("enable_dar", False),
        enable_rag=True,
        rag_max_results=args.max_results,
        rag_source=args.rag_source,
        verbose=args.verbose,
    )


def _debate_record(result) -> dict:
    v = result.verdict
    rec = {
        "predicted": v.verdict if v else "INCONCLUSIVE",
        "confidence": v.confidence if v else 0.0,
        "num_rounds": result.num_rounds,
        "n_pro_papers": len(result.pro_papers),
        "n_con_papers": len(result.con_papers),
        "pro_doc_ids": [p.get("doc_id") for p in result.pro_papers],
        "con_doc_ids": [p.get("doc_id") for p in result.con_papers],
    }
    if result.consensus:
        c = result.consensus
        rec.update({
            "quadrant": c.consensus_quadrant,
            "normalized_entropy": c.normalized_entropy,
            "jsd": c.jsd,
            "calibrated_confidence": c.calibrated_confidence,
        })
    return rec


def run_system(system: str, claim: str, factory: LLMFactory, config: dict, args) -> dict:
    if system == "zeroshot":
        r = ZeroShotBaseline(llm=factory.base()).run(claim)
        return {"predicted": r.verdict, "confidence": r.confidence}
    if system == "ragonly":
        r = RAGOnlyBaseline(llm=factory.base(), max_results=args.max_results, source=args.rag_source).run(claim)
        return {"predicted": r.verdict, "confidence": r.confidence}
    if system == "cot":
        r = CoTBaseline(llm=factory.base(), max_results=args.max_results, source=args.rag_source).run(claim)
        return {"predicted": r.verdict, "confidence": r.confidence}
    if system == "homo_mad":
        debate = Debate(pro_llm=factory.base(), con_llm=factory.base(), judge_llm=factory.base(),
                        **_debate_kwargs(config, args))
        return _debate_record(debate.run(claim))
    if system in ("scidebate", "scidebate_shared"):
        pro, con, judge = factory.heter()
        debate = Debate(pro_llm=pro, con_llm=con, judge_llm=judge, **_debate_kwargs(config, args))
        if system == "scidebate_shared":
            shared = retrieve_evidence(claim, max_results=args.max_results, stance="NEUTRAL", source=args.rag_source)
            return _debate_record(debate.run(claim, pre_retrieved_res=shared))
        return _debate_record(debate.run(claim))
    raise ValueError(f"Unknown system: {system}")


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_records(path: Path) -> dict[tuple[str, str], dict]:
    """(system, claim_id) → bản ghi thành công mới nhất."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("error"):
                done[(rec["system"], rec["claim_id"])] = rec
    return done


def append_record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(records: dict, systems: list[str], claims: list[dict]) -> dict:
    """Chỉ tính trên các claim mà MỌI hệ thống đều đã chạy xong → so sánh cặp công bằng."""
    claim_ids = [c["id"] for c in claims if all((s, c["id"]) in records for s in systems)]
    report = {"n_paired_claims": len(claim_ids), "systems": {}}
    if not claim_ids:
        return report

    gold = [records[(systems[0], cid)]["ground_truth"] for cid in claim_ids]
    report["gold_distribution"] = dict(Counter(gold))
    correctness = {}
    for s in systems:
        preds = [records[(s, cid)]["predicted"] for cid in claim_ids]
        confs = [records[(s, cid)]["confidence"] for cid in claim_ids]
        m = compute_all_metrics(preds, confs, gold)
        m["macro_f1_ci95"] = bootstrap_ci(preds, gold, "macro_f1")
        m["accuracy_ci95"] = bootstrap_ci(preds, gold, "accuracy")
        m["pred_distribution"] = dict(Counter(preds))
        m["confusion_matrix"] = confusion_matrix_str(preds, gold)
        report["systems"][s] = m
        correctness[s] = [p == g for p, g in zip(preds, gold)]

    if REFERENCE_SYSTEM in correctness:
        for s in systems:
            if s != REFERENCE_SYSTEM:
                report["systems"][s]["mcnemar_vs_scidebate"] = mcnemar_exact(
                    correctness[REFERENCE_SYSTEM], correctness[s]
                )
    return report


def print_report(report: dict) -> None:
    n = report["n_paired_claims"]
    print(f"\n{'=' * 92}\nSCIFACT DEV — {n} paired claims  gold={report.get('gold_distribution', {})}\n{'=' * 92}")
    if not n:
        print("Chưa có claim nào được tất cả hệ thống chạy xong.")
        return
    print(f"{'System':<18}{'Acc':>7}{'Macro-F1':>10}{'  95% CI (F1)':<18}{'ECE':>7}{'Brier':>8}{'  McNemar p':>12}")
    print("-" * 92)
    for s, m in report["systems"].items():
        lo, hi = m["macro_f1_ci95"]
        p = m.get("mcnemar_vs_scidebate", {}).get("p_value", "")
        print(f"{s:<18}{m['accuracy']:>7.3f}{m['macro_f1']:>10.3f}  [{lo:.3f}, {hi:.3f}]   "
              f"{m['ece']:>7.3f}{m['brier_score']:>8.3f}{str(p):>12}")
    for s, m in report["systems"].items():
        print(f"\n--- {s}  predictions={m['pred_distribution']}\n{m['confusion_matrix']}")


def markdown_table(report: dict) -> str:
    lines = [
        f"N = {report['n_paired_claims']} claims (SciFact dev, paired)\n",
        "| Hệ thống | Accuracy | Macro-F1 | 95% CI (Macro-F1) | ECE | McNemar p (vs SciDebate) |",
        "|---|---|---|---|---|---|",
    ]
    for s, m in report["systems"].items():
        lo, hi = m["macro_f1_ci95"]
        p = m.get("mcnemar_vs_scidebate", {}).get("p_value", "—")
        lines.append(f"| {s} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | [{lo:.3f}, {hi:.3f}] | {m['ece']:.3f} | {p} |")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate on SciFact dev with checkpoint/resume")
    parser.add_argument("--n", type=int, default=100, help="Số claim (0 = toàn bộ 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--rag-source", default="scifact", choices=["scifact", "hybrid", "arxiv"])
    parser.add_argument("--max-results", type=int, default=None, help="Mặc định lấy từ rag_settings")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--uncertainty", action="store_true", help="Bật consensus map (tốn thêm nhiều request)")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_results is None:
        args.max_results = config.get("rag_settings", {}).get("max_results", 5)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(ALL_SYSTEMS)
    if unknown:
        raise SystemExit(f"Hệ thống không hợp lệ: {unknown}")

    claims = load_dev_claims(args.n or None, args.seed)
    tag = f"n{len(claims)}_seed{args.seed}_{args.rag_source}"
    ckpt_path = RESULTS_DIR / f"scifact_dev_{tag}.jsonl"
    records = load_records(ckpt_path)

    if not args.report_only:
        factory = LLMFactory(config, RESULTS_DIR / "llm_cache.sqlite", args.max_tokens)
        todo = [(c, s) for c in claims for s in systems if (s, c["id"]) not in records]
        print(f"[EVAL] {len(claims)} claims × {len(systems)} systems — {len(todo)} remaining → {ckpt_path}")
        consecutive_errors = 0
        for i, (c, s) in enumerate(todo, 1):
            t0 = time.time()
            base = {"system": s, "claim_id": c["id"], "claim": c["claim"], "ground_truth": c["ground_truth"]}
            try:
                rec = {**base, **run_system(s, c["claim"], factory, config, args), "error": None}
                rec["correct"] = rec["predicted"] == c["ground_truth"]
                consecutive_errors = 0
            except QuotaExhaustedError as e:
                print(f"\n[STOP] {e}\nChạy lại cùng lệnh sau khi quota reset để tiếp tục.")
                break
            except Exception as e:
                rec = {**base, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
                consecutive_errors += 1
            rec["elapsed"] = round(time.time() - t0, 1)
            append_record(ckpt_path, rec)
            if rec.get("error"):
                print(f"[{i}/{len(todo)}] {s:<16} {c['id']:>5}  ERROR {rec['error'][:120]}")
                if consecutive_errors >= args.max_consecutive_errors:
                    print(f"\n[STOP] {consecutive_errors} lỗi liên tiếp — dừng để tránh lãng phí quota.")
                    break
            else:
                records[(s, c["id"])] = rec
                mark = "✓" if rec["correct"] else "✗"
                print(f"[{i}/{len(todo)}] {s:<16} {c['id']:>5}  {mark} pred={rec['predicted']:<12} "
                      f"gold={c['ground_truth']:<12} {rec['elapsed']:.0f}s")

    report = build_report(records, systems, claims)
    print_report(report)
    summary_path = RESULTS_DIR / f"scifact_dev_{tag}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / f"scifact_dev_{tag}_table.md").write_text(markdown_table(report), encoding="utf-8")
    print(f"\n[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
