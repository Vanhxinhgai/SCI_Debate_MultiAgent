# Scientific Debate Simulator

AI fact-checking system for scientific claims using Multi-Agent Debate (MAD) with RAG evidence retrieval and uncertainty quantification.

**Course Project** — Modern Topics in Computer Science, UET-VNU, 2025–2026.

---

## Overview

The system verifies scientific claims through structured adversarial debate between three LLM agents:

- **Pro Agent** — argues in support of the claim, retrieves supporting evidence
- **Con Agent** — challenges the claim, retrieves contradicting evidence
- **Judge Agent** — evaluates both sides and issues a final verdict: `SUPPORTED`, `REFUTED`, or `INCONCLUSIVE`

Core insight: *unconfident disagreement surfaces more valuable alternatives than confident agreement* — the Con agent exists to counteract Pro's confirmation bias.

---

## Architecture

```
scidebate/
├── agents/         # Pro, Con, Judge, Filter agent classes
├── llms/           # LLM backend abstraction (Groq, OpenRouter, Ollama)
├── memory/         # Debate chat history
├── tools/          # RAG pipeline: SciFact retriever + arXiv search + hybrid manager
├── filtering/      # DAR (Duplicate-Aware Retrieval) filtering
├── uncertainty/    # Shannon entropy, JSD, consensus scoring
├── prompts.py      # Prompt templates for all agents
├── debate.py       # Debate orchestrator
└── config.py       # Config loader

configs/
└── debate_config.yaml   # LLM backends, RAG, uncertainty settings

experiments/
├── eval_claims.py   # 20-claim test set (SUPPORTED / REFUTED / INCONCLUSIVE)
├── evaluate.py      # Evaluation runner — 4 systems × 20 claims
├── metrics.py       # Accuracy, Macro F1, ECE, Brier Score, Retrieval Precision
├── baselines.py     # Zero-Shot, RAG-Only, Debate-no-RAG baselines
└── report.py        # Generate Markdown / LaTeX evaluation report

data/scifact/        # SciFact benchmark dataset (not tracked, download separately)
tests/               # Unit tests
app.py               # Streamlit web UI
```

---

### Default LLM configuration (Heter-MAD)

| Agent | Backend | Model |
|-------|---------|-------|
| Pro | OpenRouter | `openai/gpt-oss-120b:free` |
| Con | Groq | `llama-3.3-70b-versatile` |
| Judge | Groq | `meta-llama/llama-4-scout-17b-16e-instruct` |

---

## Setup

### 1. Create environment

```bash
conda create -n scidebate python=3.10 -y
conda activate scidebate
pip install -e .
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Get keys: [console.groq.com](https://console.groq.com) · [openrouter.ai/keys](https://openrouter.ai/keys)

### 3. (Optional) SciFact dataset

Download from [allenai/scifact](https://github.com/allenai/scifact) and place at:

```
data/scifact/corpus.jsonl
data/scifact/claims_train.jsonl
data/scifact/claims_dev.jsonl
data/scifact/claims_test.jsonl
```

---

## Usage

### Web UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The UI has 5 tabs:

- **Debate Live** — watch the debate unfold in real time
- **Evidence Explorer** — browse retrieved papers for each side
- **Verdict Analysis** — confidence scores and calibration
- **Consensus Map** — 4-quadrant JSD/Entropy visualization
- **Settings** — configure LLM backends, rounds, RAG source

### CLI demo

```bash
# Single claim, default config
python experiments/run_demo.py --claim "Sleep deprivation impairs memory consolidation."

# With RAG + SciFact
python experiments/run_demo.py \
    --claim "Coffee reduces risk of type 2 diabetes." \
    --rag --rag-source hybrid --rounds 2

# Save result to JSON
python experiments/run_demo.py \
    --claim "Vitamin C prevents the common cold." \
    --output outputs/vitamin_c.json
```

### Evaluation

```bash
# Quick run: 5 claims × 4 systems (~20 min)
python -m experiments.evaluate --mode quick

# Full run: 20 claims × 4 systems (~3 hr)
python -m experiments.evaluate --mode full

# Single system only
python -m experiments.evaluate --mode quick --system mad

# Resume after interruption
python -m experiments.evaluate --mode full --resume experiments/results/eval_20250526_120000.json

# Generate report from latest results
python -m experiments.report --format markdown
python -m experiments.report --format latex
```

Systems evaluated: `mad` (ours), `zeroshot`, `ragonly`, `debate_norag`.

---

## Evaluation Results

Tested on 20 manually curated scientific claims (balanced: SUPPORTED / REFUTED / INCONCLUSIVE).

| System | Accuracy | Macro F1 | ECE ↓ | Brier ↓ |
|--------|----------|----------|-------|---------|
| Zero-Shot LLM | — | — | — | — |
| RAG-Only | — | — | — | — |
| Debate (no RAG) | — | — | — | — |
| **MAD+RAG (Ours)** | — | — | — | — |

*Run `python -m experiments.evaluate --mode full` to populate.*

---

## Testing

```bash
python tests/test_llm.py
python tests/test_memory.py
python tests/test_agents.py
python tests/test_debate.py
```
