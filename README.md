# OmniCopilot

**Cooperative Multimodal AI Under Communication Constraints**

[![C++ CI](https://github.com/omnicopilot/omnicopilot/actions/workflows/ci-cpp.yml/badge.svg)](https://github.com/omnicopilot/omnicopilot/actions/workflows/ci-cpp.yml)
[![Python CI](https://github.com/omnicopilot/omnicopilot/actions/workflows/ci-python.yml/badge.svg)](https://github.com/omnicopilot/omnicopilot/actions/workflows/ci-python.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## What Is This?

OmniCopilot is a multi-agent AI system where distributed agents with partial observations collaboratively build an uncertainty-aware world model, intelligently decide what information to share under resource constraints, and make collectively better decisions than any individual agent.

**Core research question:**
> Can intelligent, learned communication policies outperform naive broadcasting while using significantly less bandwidth — and do they discover emergent strategies that hand-designed systems cannot replicate?

**First demonstration domain:** cooperative autonomous driving over V2X networks.

---

## Key Results

> ⚠️ **Work in progress** — results will be populated as experiments complete.

| Configuration | mAP | Bandwidth | Occluded Detection |
|---|---|---|---|
| Single Agent (baseline) | — | 0 | — |
| Naive Broadcast | — | 100% | — |
| Hand-designed Priority | — | — | — |
| **Learned Policy (ours)** | — | — | — |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON RESEARCH LAYER                      │
│  Perception │ RL Policy │ LLM/RAG │ Evaluation │ Configs     │
└──────────────────────────┬──────────────────────────────────┘
                           │ pybind11
┌──────────────────────────┴──────────────────────────────────┐
│                      C++ RUNTIME LAYER                        │
│  World Model │ Communication Engine │ Trust │ Active Acq.    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                    SIMULATION LAYER                           │
│  OPV2V Replay │ CARLA │ Network Simulator │ Scenarios        │
└─────────────────────────────────────────────────────────────┘
```

**Languages:** C++17 (runtime) + Python 3.11+ (ML/research)
**Interface contract:** Protocol Buffers

---

## Quick Start

### Prerequisites

- Ubuntu 22.04+
- NVIDIA GPU with CUDA 12+
- CMake ≥ 3.22, Ninja
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Protobuf compiler (`apt install protobuf-compiler`)

### Setup

```bash
# Clone
git clone https://github.com/omnicopilot/omnicopilot.git
cd omnicopilot

# Install Python dependencies
make install

# Build C++ core
make setup-cpp

# Generate protobuf code
make proto

# Run tests
make test

# Download dataset (OPV2V subset)
make download-data
```

### Run Experiments

```bash
# Single agent baseline
make baseline

# Cooperative with intelligent communication
make communication

# Train RL communication policy
make train-policy

# Full ablation suite
make ablation

# Launch interactive demo
make demo
```

---

## Project Structure

```
omniworld/
├── proto/                 # Protobuf definitions (THE interface contract)
├── cpp/                   # C++ runtime (world model, communication, trust)
│   ├── core/             # Core libraries
│   ├── sim/              # Simulation interfaces
│   ├── bindings/         # pybind11 Python bindings
│   └── tests/            # C++ unit tests (gtest)
├── python/omnicopilot/   # Python ML/research package
│   ├── perception/       # Detection, tracking, cooperative models
│   ├── fusion/           # Multi-source fusion, uncertainty, calibration
│   ├── rl/              # RL communication policy (env, policy, training)
│   ├── reasoning/        # LLM investigation agent, RAG
│   ├── data/            # Dataset loaders (OPV2V, DAIR-V2X)
│   ├── evaluation/      # Metrics, ablation runner
│   └── config/          # Hydra experiment configs
├── scripts/              # Experiment entry points
├── tests/                # Python tests (unit, integration, property)
├── docker/               # Dockerfiles and compose
├── deploy/               # Observability configs (Prometheus, Grafana, OTel)
├── docs/                 # Documentation
└── data/                 # Datasets and results (DVC-managed, gitignored)
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **ML/Perception** | PyTorch 2, OpenPCDet, OpenCOOD, ONNX Runtime |
| **RL** | CleanRL, Gymnasium, Ray RLlib |
| **LLM/RAG** | vLLM, LangGraph, Qdrant, BGE embeddings |
| **Runtime** | C++17, Protobuf, spdlog, pybind11 |
| **Simulation** | CARLA, OPV2V, custom network sim |
| **Experiment tracking** | Weights & Biases, Hydra, DVC |
| **Quality** | mypy strict, ruff, clang-tidy, gtest, pytest, Hypothesis |
| **Observability** | Prometheus, Grafana, OpenTelemetry |
| **CI/CD** | GitHub Actions, Docker |

---

## Experiments

### Experiment 1: Does cooperation help?
Single agent vs. cooperative on occluded scenarios.

### Experiment 2: Does intelligent communication help?
Naive broadcast vs. AI-prioritized at same bandwidth.

### Experiment 3: Does learning beat engineering?
Hand-designed prioritization vs. RL-learned policy.

### Experiment 4: Does active acquisition help?
Passive sharing vs. active information requests.

### Experiment 5: Robustness under degradation
Performance under packet loss, latency, adversarial agents.

### Experiment 6: Ablation study
Systematic removal of each component to measure contribution.

---

## Contributing

1. Fork and create a feature branch.
2. Make changes (all code must pass `make lint` and `make test`).
3. Submit a pull request with description of changes and results.

### Code Standards

- **Python:** `mypy --strict`, `ruff` lint/format, Google docstrings.
- **C++:** C++17, `clang-format` (Google style), `clang-tidy`.
- **Tests:** >80% coverage. Property-based tests for mathematical properties.
- **Configs:** All experiments reproducible from Hydra YAML + DVC data version.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use OmniCopilot in your research:

```bibtex
@software{omnicopilot2024,
  title={OmniCopilot: Cooperative Multimodal AI Under Communication Constraints},
  year={2024},
  url={https://github.com/omnicopilot/omnicopilot}
}
```
