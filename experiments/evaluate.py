"""Main evaluation runner.

Usage:
    # Quick evaluation (5 claims, ~5–8 min):
    python experiments/evaluate.py --mode quick

    # Full evaluation (20 claims, ~40–60 min):
    python experiments/evaluate.py --mode full

    # Specific system only:
    python experiments/evaluate.py --mode quick --system mad

    # Systems: mad (full), zeroshot, ragonly, debate_norag
    # Results saved to experiments/results/eval_<TIMESTAMP>.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from scidebate import Debate, load_config
from scidebate.llms import GroqLLM, OpenRouterLLM
from scidebate.tools import retrieve_evidence

from experiments.eval_claims import get_claims
from experiments.metrics import compute_all_metrics, confusion_matrix_str
from experiments.baselines import ZeroShotBaseline, RAGOnlyBaseline, DebateNoRAGBaseline

RESULTS_DIR = Path(__file__).parent / "results"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "debate_config.yaml"


# ── LLM factory (reuses debate_config.yaml) ───────────────────────────────────

def _make_llm(backend_cfg: dict, max_tokens: int = 400) -> object:
    backend = backend_cfg.get("backend", "Groq (cloud)")
    model = backend_cfg.get("model", "llama-3.3-70b-versatile")
    temp = backend_cfg.get("temperature", 0.7)

    if "Groq" in backend:
        return GroqLLM(
            model=model, temperature=temp, max_tokens=max_tokens,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    if "OpenRouter" in backend:
        return OpenRouterLLM(
            model=model, temperature=temp, max_tokens=max_tokens,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _build_llms(config: dict, max_tokens: int = 400):
    llm_cfg = config.get("llm_backends", {})
    mode = llm_cfg.get("mode", "Same model (all agents)")

    if mode == "Same model (all agents)":
        same = llm_cfg.get("same_model", {})
        llm = _make_llm(same, max_tokens)
        return llm, llm, llm
    else:
        heter = llm_cfg.get("heter_mad", {})
        pro_llm = _make_llm(heter.get("pro", {}), max_tokens)
        con_llm = _make_llm(heter.get("con", {}), max_tokens)
        judge_llm = _make_llm(heter.get("judge", {}), max_tokens)
        return pro_llm, con_llm, judge_llm


# ── Single-claim runners ───────────────────────────────────────────────────────

def _run_mad(claim_obj: dict, config: dict) -> dict:
    """Run the full MAD+RAG system on one claim."""
    pro_llm, con_llm, judge_llm = _build_llms(config, max_tokens=350)

    debate_settings = config.get("debate_settings", {})
    rag_settings = config.get("rag_settings", {})
    uncertainty_settings = config.get("uncertainty_settings", {})

    debate = Debate(
        pro_llm=pro_llm,
        con_llm=con_llm,
        judge_llm=judge_llm,
        max_rounds=debate_settings.get("max_rounds", 2),
        parallel_opening=False,
        compute_uncertainty=uncertainty_settings.get("compute_consensus", False),
        n_uncertainty_samples=uncertainty_settings.get("n_samples", 4),
        verbose=True,
        use_dar=False,
        enable_early_stopping=uncertainty_settings.get("enable_early_stopping", True),
        use_logprobs=uncertainty_settings.get("use_logprobs", True),
        enable_rag=rag_settings.get("enable_rag", True),
        rag_max_results=rag_settings.get("max_results", 5),
        rag_source=rag_settings.get("source", "hybrid"),
    )

    result = debate.run(claim_obj["claim"])

    return {
        "system": "MAD+RAG",
        "claim_id": claim_obj["id"],
        "claim": claim_obj["claim"],
        "ground_truth": claim_obj["ground_truth"],
        "predicted": result.verdict.verdict if result.verdict else "INCONCLUSIVE",
        "confidence": result.verdict.confidence if result.verdict else 0.5,
        "correct": (result.verdict.verdict == claim_obj["ground_truth"]) if result.verdict else False,
        "elapsed": result.elapsed_seconds,
        "num_rounds": result.num_rounds,
        "num_papers": len(result.retrieved_papers),
        "pro_papers": [p.get("title", "") for p in result.pro_papers],
        "con_papers": [p.get("title", "") for p in result.con_papers],
        "transcript_excerpt": result.transcript[0].content[:300] if result.transcript else "",
        "error": None,
    }


def _run_zeroshot(claim_obj: dict, config: dict) -> dict:
    _, _, judge_llm = _build_llms(config, max_tokens=350)
    baseline = ZeroShotBaseline(llm=judge_llm)
    r = baseline.run(claim_obj["claim"])
    return {
        "system": "Zero-Shot",
        "claim_id": claim_obj["id"],
        "claim": claim_obj["claim"],
        "ground_truth": claim_obj["ground_truth"],
        "predicted": r.verdict,
        "confidence": r.confidence,
        "correct": r.verdict == claim_obj["ground_truth"],
        "elapsed": r.elapsed_seconds,
        "error": None,
    }


def _run_ragonly(claim_obj: dict, config: dict) -> dict:
    _, _, judge_llm = _build_llms(config, max_tokens=350)
    rag_settings = config.get("rag_settings", {})
    baseline = RAGOnlyBaseline(
        llm=judge_llm,
        max_results=rag_settings.get("max_results", 5),
        source=rag_settings.get("source", "hybrid"),
    )
    r = baseline.run(claim_obj["claim"])
    return {
        "system": "RAG-Only",
        "claim_id": claim_obj["id"],
        "claim": claim_obj["claim"],
        "ground_truth": claim_obj["ground_truth"],
        "predicted": r.verdict,
        "confidence": r.confidence,
        "correct": r.verdict == claim_obj["ground_truth"],
        "elapsed": r.elapsed_seconds,
        "num_papers": len(r.retrieved_papers),
        "error": None,
    }


def _run_debate_norag(claim_obj: dict, config: dict) -> dict:
    pro_llm, con_llm, judge_llm = _build_llms(config, max_tokens=350)
    debate_settings = config.get("debate_settings", {})
    baseline = DebateNoRAGBaseline(
        pro_llm=pro_llm,
        con_llm=con_llm,
        judge_llm=judge_llm,
        max_rounds=debate_settings.get("max_rounds", 2),
    )
    r = baseline.run(claim_obj["claim"])
    return {
        "system": "Debate-no-RAG",
        "claim_id": claim_obj["id"],
        "claim": claim_obj["claim"],
        "ground_truth": claim_obj["ground_truth"],
        "predicted": r.verdict,
        "confidence": r.confidence,
        "correct": r.verdict == claim_obj["ground_truth"],
        "elapsed": r.elapsed_seconds,
        "error": None,
    }


SYSTEM_RUNNERS = {
    "mad": _run_mad,
    "zeroshot": _run_zeroshot,
    "ragonly": _run_ragonly,
    "debate_norag": _run_debate_norag,
}


# ── Metrics aggregation ────────────────────────────────────────────────────────

def _aggregate(records: list[dict]) -> dict:
    preds = [r["predicted"] for r in records]
    gts = [r["ground_truth"] for r in records]
    confs = [r["confidence"] for r in records]

    metrics = compute_all_metrics(preds, confs, gts)
    metrics["confusion_matrix"] = confusion_matrix_str(preds, gts)

    by_category = {}
    for cat in ["SUPPORTED", "REFUTED", "INCONCLUSIVE"]:
        subset = [r for r in records if r["ground_truth"] == cat]
        if subset:
            cat_preds = [r["predicted"] for r in subset]
            cat_correct = sum(p == cat for p in cat_preds)
            by_category[cat] = {
                "accuracy": round(cat_correct / len(subset), 4),
                "n": len(subset),
            }
    metrics["by_category"] = by_category

    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in records if r.get("difficulty") == diff]
        if subset:
            d_preds = [r["predicted"] for r in subset]
            d_gts = [r["ground_truth"] for r in subset]
            d_acc = sum(p == g for p, g in zip(d_preds, d_gts)) / len(subset)
            by_difficulty[diff] = {"accuracy": round(d_acc, 4), "n": len(subset)}
    metrics["by_difficulty"] = by_difficulty

    return metrics


# ── Main runner ────────────────────────────────────────────────────────────────

def run_evaluation(
    mode: str = "quick",
    systems: list[str] = None,
    config_path: str = str(CONFIG_PATH),
    delay_between_claims: float = 8.0,
    resume_path: str = None,
) -> dict:
    """Run the full evaluation and return results dict.

    Args:
        mode: "quick" (5 claims) or "full" (20 claims)
        systems: list from ["mad", "zeroshot", "ragonly", "debate_norag"]
        config_path: path to debate_config.yaml
        delay_between_claims: seconds to wait between API calls (rate limit protection)
        resume_path: path to a previous results JSON to continue from
    """
    if systems is None:
        systems = ["mad", "zeroshot", "ragonly", "debate_norag"]

    config = load_config(config_path)
    claims = get_claims(mode)

    completed_keys = set()
    all_records: list[dict] = []

    if resume_path and Path(resume_path).exists():
        with open(resume_path, encoding="utf-8") as f:
            prev = json.load(f)
        all_records = prev.get("records", [])
        for r in all_records:
            if not r.get("error"):
                completed_keys.add((r["system"], r["claim_id"]))
        print(f"[RESUME] Loaded {len(all_records)} prior records from {resume_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    total_claims = len(claims) * len(systems)
    done = 0

    for sys_name in systems:
        runner = SYSTEM_RUNNERS[sys_name]
        print(f"\n{'=' * 60}")
        print(f"SYSTEM: {sys_name.upper()}  ({len(claims)} claims)")
        print("=" * 60)

        for claim_obj in claims:
            key = (sys_name, claim_obj["id"])
            if key in completed_keys:
                done += 1
                print(f"  [SKIP] {claim_obj['id']} already done.")
                continue

            print(f"\n[{done+1}/{total_claims}] {sys_name} — {claim_obj['id']}: {claim_obj['claim'][:60]}...")
            try:
                record = runner(claim_obj, config)
                record["difficulty"] = claim_obj.get("difficulty", "")
                record["domain"] = claim_obj.get("domain", "")
                record["claim_type"] = claim_obj.get("claim_type", "")
                all_records.append(record)
                status = "✓ CORRECT" if record.get("correct") else "✗ WRONG"
                print(f"  {status}  pred={record['predicted']}  gt={record['ground_truth']}  conf={record['confidence']:.2f}  t={record['elapsed']:.1f}s")
            except Exception as e:
                print(f"  [ERROR] {sys_name}/{claim_obj['id']}: {e}")
                all_records.append({
                    "system": sys_name,
                    "claim_id": claim_obj["id"],
                    "claim": claim_obj["claim"],
                    "ground_truth": claim_obj["ground_truth"],
                    "predicted": "INCONCLUSIVE",
                    "confidence": 0.5,
                    "correct": False,
                    "elapsed": 0.0,
                    "difficulty": claim_obj.get("difficulty", ""),
                    "domain": claim_obj.get("domain", ""),
                    "claim_type": claim_obj.get("claim_type", ""),
                    "error": traceback.format_exc(),
                })

            done += 1

            # Save intermediate results after each claim
            _save_results(all_records, systems, output_path, mode, timestamp)

            if done < total_claims:
                time.sleep(delay_between_claims)

    # Final save with metrics
    result = _save_results(all_records, systems, output_path, mode, timestamp, final=True)
    print(f"\n[DONE] Results saved to: {output_path}")
    return result


def _save_results(
    all_records: list[dict],
    systems: list[str],
    output_path: Path,
    mode: str,
    timestamp: str,
    final: bool = False,
) -> dict:
    system_metrics = {}
    for sys_name in systems:
        sys_records = [r for r in all_records if r["system"] == sys_name and not r.get("error")]
        if sys_records:
            system_metrics[sys_name] = _aggregate(sys_records)

    result = {
        "timestamp": timestamp,
        "mode": mode,
        "systems_evaluated": systems,
        "total_claims": len(set(r["claim_id"] for r in all_records)),
        "records": all_records,
        "system_metrics": system_metrics,
        "status": "complete" if final else "in_progress",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SCI_Debate_MultiAgent system")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick=5 claims, full=20 claims")
    parser.add_argument("--system", choices=["mad", "zeroshot", "ragonly", "debate_norag", "all"],
                        default="all", help="Which system(s) to run")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="Path to debate_config.yaml")
    parser.add_argument("--delay", type=float, default=8.0,
                        help="Seconds between claims (API rate limit buffer)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous results JSON to continue from")
    args = parser.parse_args()

    systems = (
        ["mad", "zeroshot", "ragonly", "debate_norag"]
        if args.system == "all"
        else [args.system]
    )

    result = run_evaluation(
        mode=args.mode,
        systems=systems,
        config_path=args.config,
        delay_between_claims=args.delay,
        resume_path=args.resume,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'System':<20} {'Accuracy':>10} {'Macro F1':>10} {'ECE':>8} {'Brier':>8}")
    print("-" * 70)
    for sys_name, metrics in result.get("system_metrics", {}).items():
        print(
            f"{sys_name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['ece']:>8.4f} "
            f"{metrics['brier_score']:>8.4f}"
        )

    if "mad" in result.get("system_metrics", {}):
        print("\n--- MAD+RAG Confusion Matrix ---")
        print(result["system_metrics"]["mad"].get("confusion_matrix", ""))


if __name__ == "__main__":
    main()
