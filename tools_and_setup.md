# OmniCopilot — Complete Tools, LLMs & Setup Guide

## What You Already Have

| Tool | Status |
|---|---|
| VS Code | ✅ Have it |
| GitHub Copilot (Student) | ✅ Have it |
| Laptop (16 GB RAM) | ✅ Have it |

That's a solid starting point. Here's everything else.

---

## VS Code Extensions (Install Now — All Free)

### Must-have

| Extension | Why |
|---|---|
| **C/C++** (Microsoft) | C++ IntelliSense, debugging |
| **CMake Tools** | CMake integration, build/test from VS Code |
| **clangd** | Better C++ analysis than Microsoft's extension (use one or the other) |
| **Python** (Microsoft) | Python IntelliSense, debugging |
| **Pylance** | Fast Python type checking |
| **Ruff** | Python linting + formatting (replaces pylint + black) |
| **Mypy Type Checker** | Type error highlighting |
| **vscode-proto3** | Protobuf syntax highlighting |
| **YAML** (Red Hat) | YAML syntax + validation |
| **GitLens** | Git blame, history, PR integration |
| **GitHub Copilot** | You already have this ✅ |
| **GitHub Copilot Chat** | Ask Copilot questions in sidebar |

### Nice-to-have

| Extension | Why |
|---|---|
| **Docker** | Manage containers from VS Code |
| **Jupyter** | Run notebooks in VS Code |
| **Thunder Client** | Test gRPC/REST APIs |
| **Markdown All in One** | Better markdown editing |
| **Error Lens** | Show errors inline (very useful) |
| **Todo Tree** | Find all TODOs in codebase |

---

## Development Tools (Install on Laptop)

### Required — Install now

| Tool | What it is | How to install |
|---|---|---|
| **Git** | Version control | `sudo apt install git` (likely already have it) |
| **Python 3.11** | Python runtime | `sudo apt install python3.11 python3.11-dev python3.11-venv` |
| **uv** | Fast Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **CMake** | C++ build system | `sudo apt install cmake` (need 3.22+) |
| **Ninja** | Fast C++ build tool | `sudo apt install ninja-build` |
| **g++-13** | C++ compiler | `sudo apt install g++-13` |
| **protobuf-compiler** | Proto code generation | `sudo apt install protobuf-compiler libprotobuf-dev` |
| **Make** | Task runner | Already installed on Linux |
| **Docker** | Containers (optional locally) | `sudo apt install docker.io docker-compose-v2` |

### Required — Install via uv (after repo setup)

```bash
cd omniworld
make install-dev    # Installs all Python dev dependencies
```

This gives you: pytest, mypy, ruff, pre-commit, hypothesis, etc.

### Optional — Install later when needed

| Tool | When needed | How |
|---|---|---|
| **clang-17** | If you want clang compiler + clang-tidy | `sudo apt install clang-17 clang-tidy-17` |
| **buf** | Protobuf linting | `npm install -g @bufbuild/buf` or download binary |
| **DVC** | Dataset version control (Phase 5+) | `uv pip install dvc` |

---

## LLMs — What You Need and When

### The honest truth

**You don't need ANY LLM until Phase 9 (week 21).** The LLM/RAG layer is the investigation and explanation agent — it's the last thing you build.

For Phases 1-8, there are zero LLM dependencies. Don't set anything up now.

### When you DO need LLMs (Phase 9+)

You need an LLM for two things:
1. **Investigation agent** — analyzes complex multi-signal situations
2. **Explanation agent** — explains decisions in natural language

### Option A: API-based (Simplest, recommended to start)

| Provider | Model | Cost | Best for |
|---|---|---|---|
| **Google Gemini** | Gemini 1.5 Flash | Free tier: 15 RPM, 1M tokens/day | Prototyping, light use |
| **Google Gemini** | Gemini 1.5 Pro | Free tier: 2 RPM | Complex reasoning |
| **OpenAI** | GPT-4o-mini | $0.15 / 1M input tokens | Cheap, good quality |
| **OpenAI** | GPT-4o | $2.50 / 1M input tokens | Best quality, more expensive |
| **Anthropic** | Claude Sonnet | $3 / 1M input tokens | Good reasoning |
| **Groq** | Llama 3 70B | Free tier available | Fast, free for prototyping |

**Recommendation: Start with Gemini free tier or Groq free tier. $0 cost.**

Monthly cost estimate for the investigation agent: ~$1-5 (it's not called often — only for complex situations, not every frame).

### Option B: Local LLM (Free, but needs decent hardware)

| Tool | Model | RAM needed | Quality |
|---|---|---|---|
| **Ollama** | Llama 3.1 8B (Q4) | ~6 GB RAM | Good enough for explanation |
| **Ollama** | Mistral 7B (Q4) | ~5 GB RAM | Good for tool calling |
| **Ollama** | Phi-3 Mini (Q4) | ~3 GB RAM | Fastest, OK quality |

On your 16 GB laptop: Ollama + Llama 3.1 8B quantized will run, but slowly. Fine for development and testing.

**How to install Ollama:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
# Test it:
ollama run llama3.1:8b "What is cooperative perception?"
```

### Option C: On Colab (Free GPU inference)

Run vLLM or Ollama in Colab with free T4 GPU:
```python
!pip install vllm
from vllm import LLM
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```

### What to use in code

```python
# In your investigation agent, use a simple abstraction:
# This lets you swap between API and local without changing logic

from omnicopilot.reasoning.investigation import InvestigationAgent

# Option A: API
agent = InvestigationAgent(provider="gemini", model="gemini-1.5-flash")

# Option B: Local Ollama
agent = InvestigationAgent(provider="ollama", model="llama3.1:8b")

# Option C: vLLM on Colab
agent = InvestigationAgent(provider="vllm", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
```

### Summary: LLMs

| When | What | Cost |
|---|---|---|
| Phase 1-8 | Nothing | $0 |
| Phase 9 (prototyping) | Gemini free tier or Ollama local | $0 |
| Phase 9 (production) | GPT-4o-mini or Gemini Flash | ~$1-5/month |

---

## RAG — What You Need and When

### Again: not until Phase 9 (week 21)

RAG is for the knowledge-grounded investigation agent. It retrieves relevant papers and standards when analyzing complex situations.

### Components

| Component | Tool | Cost | Why this one |
|---|---|---|---|
| **Embedding model** | `BAAI/bge-small-en-v1.5` | Free (runs locally) | Small (130 MB), fast, good quality. Runs on CPU. |
| **Vector database** | Qdrant (local mode) | Free | Runs as a local file, no server needed. Or use Docker. |
| **Document loader** | LangChain or manual PyPDF2 | Free | Load and chunk PDFs |
| **Reranker** (optional) | `BAAI/bge-reranker-base` | Free | Improves retrieval quality |

### How it works

```
PDF papers + standards
        ↓
    Chunk into paragraphs (500-1000 chars)
        ↓
    Embed with BGE model (runs on CPU)
        ↓
    Store in Qdrant (local file)
        ↓
    At query time:
        ↓
    Embed the question
        ↓
    Find top-5 similar chunks
        ↓
    Feed to LLM as context
        ↓
    LLM generates grounded answer
```

### Minimal RAG setup (Phase 9)

```python
# This is all you need — runs on your laptop, no GPU

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Embedding model (130 MB, runs on CPU)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Vector store (local file, no server)
qdrant = QdrantClient(path="data/qdrant_store")

# That's it. Index documents, search, done.
```

### What documents to collect (week 21)

| Category | Where to find | What to grab |
|---|---|---|
| Cooperative perception papers | arXiv, Google Scholar | V2VNet, Where2comm, CoBEVT, DiscoNet, OPV2V paper (~10 PDFs) |
| V2X communication | ETSI website, 3GPP | ITS standards overview documents (~5 PDFs) |
| Safety | ISO website, free summaries | ISO 26262 overview, SOTIF summary (~3 PDFs) |
| Your own docs | Your repo | Architecture doc, experiment logs |

Total: ~20-30 PDFs, ~200 MB. Nothing large.

### Summary: RAG

| When | What | Cost |
|---|---|---|
| Phase 1-8 | Nothing | $0 |
| Phase 9 | BGE embeddings + Qdrant local + 20 PDFs | $0 (all runs on CPU) |

---

## Experiment Tracking: Weights & Biases

### What it is

W&B logs your experiment results — metrics, plots, configs, model checkpoints — so you can compare runs and share results.

### Setup

1. Go to https://wandb.ai
2. Sign up with wife's .edu email → **free academic tier (unlimited)**
3. Get API key from settings page

```bash
# Login once
uv run wandb login
# Paste your API key

# Or set environment variable
export WANDB_API_KEY=your_key_here
```

### What it looks like in code

```python
import wandb

wandb.init(project="omnicopilot", name="cooperative_5agents")
wandb.log({"mAP": 0.73, "bandwidth_MB": 2.1, "occluded_detection": 0.89})
wandb.finish()
```

### When needed

From Phase 2 onwards — every experiment should log to W&B. Set it up in week 1.

### Cost

**$0** with academic email. Unlimited projects, runs, and storage.

---

## Accounts to Create (Do This Week)

| Account | URL | Cost | Why |
|---|---|---|---|
| **GitHub** (if don't have) | github.com | Free | Code hosting |
| **GitHub Student Pack** (wife) | education.github.com | Free | Copilot, credits, Pro |
| **W&B** | wandb.ai | Free (.edu) | Experiment tracking |
| **Google account** (for Colab + Drive) | google.com | Free | GPU compute + storage |
| **OPV2V access** | mobility-lab.seas.ucla.edu/opv2v | Free | Dataset |
| **Hugging Face** | huggingface.co | Free | Download models (BGE, Llama) |
| **Ollama** | ollama.com | Free | Local LLM (when needed) |

That's it. Create these accounts and you're set for the entire project.

---

## Complete Tool List by Phase

### Phase 1-2 (Weeks 1-4): Foundation

**On laptop:**
- VS Code + extensions listed above
- Git, Python 3.11, uv, CMake, Ninja, g++-13, protobuf
- Pre-commit hooks running

**Cloud:**
- GitHub repo created
- W&B project created

**Not needed yet:** Docker, LLM, RAG, Colab, any dataset beyond 1-2 scenes

---

### Phase 3-4 (Weeks 5-10): Cooperation & Communication

**On laptop:**
- Same as above
- C++ world model compiling and tested

**Cloud:**
- Google Colab (free or Pro) for running perception models
- OPV2V test split on Google Drive

**Not needed yet:** LLM, RAG, Qdrant, DAIR-V2X

---

### Phase 5 (Weeks 11-14): RL Training

**On laptop:**
- RL environment testing with small synthetic data

**Cloud:**
- Google Colab for PPO training (needs GPU)
- OPV2V validation split on Google Drive
- W&B for tracking training curves

**Not needed yet:** LLM, RAG

---

### Phase 6-8 (Weeks 15-20): Acquisition, Adversarial, Ablation

**On laptop:**
- Trust module, active acquisition (C++)
- Ablation config management

**Cloud:**
- Colab for running full ablation suite

**Not needed yet:** LLM, RAG

---

### Phase 9 (Weeks 21-22): LLM + RAG

**On laptop:**
- Ollama + Llama 3.1 8B (for development/testing)
- BGE embedding model (CPU, 130 MB)
- Qdrant local mode
- Collect ~20 PDFs

**Cloud (optional):**
- Gemini free tier API for better quality
- Or vLLM on Colab for GPU inference

---

### Phase 10 (Weeks 23-26): Demo & Release

**On laptop:**
- Streamlit demo app
- Video recording (OBS or similar)

**Cloud:**
- Final experiment runs on Colab
- Results on W&B

---

## What You Do NOT Need

| Tool | Why skip |
|---|---|
| **PyCharm / CLion** | VS Code + extensions is enough (but JetBrains is free for students if you want it) |
| **Kubernetes** | Over-engineered for this project |
| **AWS / Azure / GCP** | Colab is cheaper and simpler for your scale |
| **MLflow** | W&B is better and free for academics |
| **Elasticsearch** | Qdrant is simpler for vector search |
| **LangChain** | Mostly unnecessary complexity. LangGraph is simpler for agents. Or just write it yourself — it's not much code. |
| **Paid LLM APIs (now)** | Not needed until Phase 9, and free tiers exist |
| **CARLA (now)** | OPV2V dataset is sufficient. CARLA is a nice-to-have much later. |
| **Multiple monitors** | Nice but not required |
| **Expensive GPU** | Cloud is cheaper for your usage pattern |

---

## Total Cost Summary

### Entire project (26 weeks)

| Item | Cost |
|---|---|
| VS Code + extensions | $0 |
| GitHub (with Student Pack) | $0 |
| Copilot (Student) | $0 |
| W&B (Academic) | $0 |
| JetBrains (Student, optional) | $0 |
| Python/C++ tools | $0 |
| OPV2V dataset | $0 |
| Kaggle GPU | $0 |
| Google Colab Free | $0 |
| Ollama + local LLM | $0 |
| BGE embeddings + Qdrant | $0 |
| Gemini free tier | $0 |
| Hugging Face | $0 |
| **Subtotal (free tier only)** | **$0** |

### Optional comfort upgrades

| Item | Cost |
|---|---|
| Google Colab Pro | $10/month × 6 months = $60 |
| Google Drive 200 GB | $3/month × 6 months = $18 |
| OpenAI API (Phase 9-10 only) | ~$5-10 total |
| **Subtotal (comfort)** | **~$80-90 total** |

### Realistic budget

**If spending $0:** Totally doable. Use free tiers for everything. More session management hassle.

**If spending $10-13/month:** Colab Pro + Drive 200 GB. Comfortable. Recommended.

---

## One-Page Quick Reference

```
DEVELOP (laptop, free)
├── VS Code + Copilot + extensions
├── Git + GitHub (Student Pack)
├── Python 3.11 + uv
├── CMake + Ninja + g++-13 + protobuf
├── Pre-commit (ruff, mypy, clang-format)
└── Ollama (Phase 9 only)

TRAIN (cloud, free/$10mo)
├── Google Colab (Free or Pro)
├── Kaggle (free, backup GPU)
└── W&B (free academic)

STORE (cloud, free/$3mo)
├── Google Drive (datasets)
└── GitHub (code)

BUILD (Phase 9, free)
├── BGE embeddings (CPU, local)
├── Qdrant (local file mode)
├── Gemini API (free tier)
└── ~20 PDFs for knowledge base
```
