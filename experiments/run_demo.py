"""CLI để chạy debate từ command line.

Examples:
    python experiments/run_demo.py
    python experiments/run_demo.py --claim "..." --rounds 2
    python experiments/run_demo.py --claim "..." --uncertainty
    python experiments/run_demo.py --claim "..." --output result.json
"""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from scidebate import Debate, load_config
from scidebate.llms import OllamaLLM, GroqLLM, OpenRouterLLM


DEFAULT_CLAIM = "Drinking 8 glasses of water per day is essential for human health."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Multi-Agent Debate on a scientific claim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--claim", type=str, default=DEFAULT_CLAIM,
                        help="Scientific claim to debate")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (overrides config)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of debate rounds (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per turn (overrides config)")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Compute consensus metrics (adds ~2-3 min)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples for uncertainty (overrides config)")
    parser.add_argument("--rag", action="store_true",
                        help="Enable evidence retrieval")
    parser.add_argument("--rag-source", choices=["arxiv", "scifact", "hybrid"], default=None,
                        help="Evidence source for RAG")
    parser.add_argument("--rag-max-results", type=int, default=None,
                        help="Maximum retrieved evidence documents")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save result as JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress live logs")
    parser.add_argument("--config", type=str, default="configs/debate_config.yaml",
                        help="Path to YAML config file")
    return parser.parse_args()


def result_to_dict(result) -> dict:
    """Convert DebateResult sang dict cho JSON."""
    d = {
        "claim": result.claim,
        "num_rounds": result.num_rounds,
        "models_used": result.models_used,
        "verdict": {
            "verdict": result.verdict.verdict,
            "confidence": result.verdict.confidence,
            "justification": result.verdict.justification,
        } if result.verdict else None,
        "transcript": [
            {
                "round": t.round_num,
                "speaker": t.speaker,
                "model": t.model,
                "content": t.content,
            }
            for t in result.transcript
        ],
        "retrieved_evidence": result.retrieved_papers,
    }

    if result.consensus:
        c = result.consensus
        d["consensus"] = {
            "consensus_level": c.consensus_level,
            "consensus_quadrant": c.consensus_quadrant,
            "entropy": c.entropy,
            "normalized_entropy": c.normalized_entropy,
            "jsd": c.jsd,
            "calibrated_confidence": c.calibrated_confidence,
            "judge_verdict_distribution": c.judge_verdict_distribution,
            "pro_distribution": c.pro_distribution,
            "con_distribution": c.con_distribution,
            "explanation": c.explanation,
        }

    return d


def create_llm_instance(backend_config: dict, default_temp: float, max_tokens: int):
    """Instantiate an LLM backend according to backend_config."""
    backend = backend_config.get("backend", "Ollama (local)")
    model = backend_config.get("model", "qwen2.5:3b")
    temp = backend_config.get("temperature", default_temp)
    
    if "Ollama" in backend:
        return OllamaLLM(model=model, temperature=temp, max_tokens=max_tokens)
    elif "Groq" in backend:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to use Groq.")
        return GroqLLM(model=model, temperature=temp, max_tokens=max_tokens, api_key=api_key)
    elif "OpenRouter" in backend:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it to use OpenRouter.")
        return OpenRouterLLM(model=model, temperature=temp, max_tokens=max_tokens, api_key=api_key)
    else:
        return OllamaLLM(model=model, temperature=temp, max_tokens=max_tokens)


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    llm_backends_config = config.get("llm_backends", {})
    mode = llm_backends_config.get("mode", "Same model (all agents)")
    
    debate_settings = config.get("debate_settings", {})
    rounds = args.rounds if args.rounds is not None else debate_settings.get("max_rounds", 2)
    max_tokens = args.max_tokens if args.max_tokens is not None else debate_settings.get("max_tokens", 300)
    parallel_opening = debate_settings.get("parallel_opening", False)
    
    dar_settings = config.get("dar_settings", {})
    use_dar = dar_settings.get("enable_dar", False)
    
    uncertainty_settings = config.get("uncertainty_settings", {})
    compute_uncertainty = args.uncertainty if args.uncertainty else uncertainty_settings.get("compute_consensus", False)
    n_samples = args.n_samples if args.n_samples is not None else uncertainty_settings.get("n_samples", 5)
    enable_early_stopping = uncertainty_settings.get("enable_early_stopping", False)
    use_logprobs = uncertainty_settings.get("use_logprobs", True)
    rag_settings = config.get("rag_settings", {})
    enable_rag = args.rag or rag_settings.get("enable_rag", False)
    rag_source = args.rag_source or rag_settings.get("source", "hybrid")
    rag_max_results = args.rag_max_results if args.rag_max_results is not None else rag_settings.get("max_results", 3)

    # Instantiate LLMs
    if mode == "Same model (all agents)":
        same_model_cfg = llm_backends_config.get("same_model", {})
        model_name = args.model if args.model is not None else same_model_cfg.get("model", "qwen2.5:3b")
        temp = args.temperature if args.temperature is not None else same_model_cfg.get("temperature", 0.0)
        backend = same_model_cfg.get("backend", "Ollama (local)")
        
        backend_cfg = {"backend": backend, "model": model_name, "temperature": temp}
        llm = create_llm_instance(backend_cfg, temp, max_tokens)
        
        debate = Debate(
            llm=llm,
            max_rounds=rounds,
            parallel_opening=parallel_opening,
            compute_uncertainty=compute_uncertainty,
            n_uncertainty_samples=n_samples,
            verbose=not args.quiet,
            use_dar=use_dar,
            enable_early_stopping=enable_early_stopping,
            use_logprobs=use_logprobs,
            enable_rag=enable_rag,
            rag_max_results=rag_max_results,
            rag_source=rag_source,
        )
    else:
        # Heter-MAD
        heter_mad_cfg = llm_backends_config.get("heter_mad", {})
        pro_cfg = heter_mad_cfg.get("pro", {})
        con_cfg = heter_mad_cfg.get("con", {})
        judge_cfg = heter_mad_cfg.get("judge", {})
        
        # Override with command line options if provided
        if args.model is not None:
            pro_cfg["model"] = con_cfg["model"] = judge_cfg["model"] = args.model
        if args.temperature is not None:
            pro_cfg["temperature"] = con_cfg["temperature"] = judge_cfg["temperature"] = args.temperature
            
        pro_llm = create_llm_instance(pro_cfg, 0.6, max_tokens)
        con_llm = create_llm_instance(con_cfg, 0.8, max_tokens)
        judge_llm = create_llm_instance(judge_cfg, 0.0, max_tokens)
        
        debate = Debate(
            pro_llm=pro_llm,
            con_llm=con_llm,
            judge_llm=judge_llm,
            max_rounds=rounds,
            parallel_opening=parallel_opening,
            compute_uncertainty=compute_uncertainty,
            n_uncertainty_samples=n_samples,
            verbose=not args.quiet,
            use_dar=use_dar,
            enable_early_stopping=enable_early_stopping,
            use_logprobs=use_logprobs,
            enable_rag=enable_rag,
            rag_max_results=rag_max_results,
            rag_source=rag_source,
        )

    result = debate.run(args.claim)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(result.summary())

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_to_dict(result), f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
