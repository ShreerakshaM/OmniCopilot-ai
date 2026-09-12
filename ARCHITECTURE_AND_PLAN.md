# OmniCopilot — Architecture, Technology Stack & Task Breakdown

---

## Design Philosophy

This is engineered as a production-grade research system — the kind you'd see at Waymo, NVIDIA Research, or a well-funded autonomous systems startup. Not a Jupyter notebook stitched together with duct tape.

Principles:
- **Typed everywhere** — no `dict` soup, no `Any`, no guessing
- **Interface-driven** — components communicate through defined contracts, not implementation details
- **Testable** — every module has unit tests, integration tests run in CI
- **Reproducible** — any experiment can be re-run from a config file and produce identical results
- **Observable** — every decision the system makes is logged, traceable, and explainable
- **Performant where it matters** — C++ for the hot path, Python for research flexibility
- **Multi-language by design** — not "maybe C++ later" but "C++ and Python from day one with clean boundaries"

---

## Languages

### C++17 (Core Runtime)

**What it owns:**
- World model (entity storage, spatial indexing, temporal tracking)
- Message serialization / deserialization
- Network simulation engine (channel model, queue, scheduling)
- Agent runtime loop (observe → fuse → decide → communicate cycle)
- Real-time data structures (lock-free queues, ring buffers)

**Why C++ here:**
- The world model updates at every simulation tick. With 5+ agents producing 50+ observations each at 10Hz, that's 500+ fusion operations per second minimum. Python can't do this cleanly at scale.
- Message handling with bandwidth constraints requires precise timing and memory control.
- This is what autonomous driving companies (Waymo, Motional, Cruise) use for their runtime.
- Demonstrates systems engineering skill that Python-only projects cannot.

**Standard:** C++17. Use `std::variant`, `std::optional`, structured bindings, `constexpr`. No raw pointers — smart pointers and RAII throughout.

**Build system:** CMake (industry standard for C++ in robotics/autonomous systems).

### Python 3.11+ (ML, Research, Orchestration)

**What it owns:**
- All ML model training and inference (PyTorch)
- RL policy training
- Dataset loading and preprocessing
- Experiment orchestration
- LLM / RAG integration
- Evaluation and analysis
- Visualization and demo

**Why Python here:**
- ML ecosystem is Python. Fighting this is pointless.
- Research iteration speed matters — hypothesis to result in hours, not days.
- Experiment configs, hyperparameter sweeps, analysis notebooks.

**Standard:** Strict typing with `mypy` (strict mode). Dataclasses or Pydantic for all data structures. No untyped dictionaries as interfaces.

### Binding Layer: pybind11

- C++ world model and communication engine exposed to Python
- Python ML models called from C++ runtime via embedded Python or gRPC
- Clean interface: C++ handles the real-time loop, Python handles the intelligence

---

## Technology Stack (Production-Grade)

### Core Systems

| Technology | Purpose | Industry precedent |
|---|---|---|
| **C++17** | Runtime, world model, comms engine | Waymo, Tesla, NVIDIA, Cruise |
| **CMake** | C++ build system | Universal in robotics/AV |
| **pybind11** | C++ ↔ Python bindings | PyTorch itself, Open3D, most ML-meets-systems projects |
| **gRPC + Protobuf** | Inter-service communication, message format | Google, Waymo, every microservice at scale |
| **FlatBuffers** (alternative to Protobuf for hot path) | Zero-copy serialization for high-frequency messages | Google games/robotics, performance-critical paths |

### ML / Deep Learning

| Technology | Purpose | Why |
|---|---|---|
| **PyTorch 2.x** | Model training & inference | Industry standard, torch.compile for production speed |
| **PyTorch Lightning** | Training framework | Clean separation of research code from engineering boilerplate. Handles distributed training, logging, checkpointing. |
| **OpenPCDet** | 3D object detection | Best open-source 3D detection framework. Supports PointPillars, CenterPoint, PV-RCNN. |
| **OpenCOOD** | Cooperative perception baselines | V2VNet, DiscoNet, Where2comm, CoBEVT — all implemented. Saves months. |
| **ONNX Runtime** | Inference optimization | Export trained models for fast C++ inference. What production systems actually use. |
| **TensorRT** (optional) | GPU inference optimization | NVIDIA standard for deployment. 2-5x faster than raw PyTorch. |

### Reinforcement Learning

| Technology | Purpose | Why |
|---|---|---|
| **CleanRL** | RL implementation | Single-file, transparent implementations. Better for research than SB3 (easier to modify). Used by serious RL researchers. |
| **Gymnasium** | Environment API standard | Universal standard |
| **Ray RLlib** (for scale) | Distributed RL training | If you need multi-GPU / multi-node RL training. What companies use at scale. |

### Simulation & Data

| Technology | Purpose | Why |
|---|---|---|
| **CARLA 0.9.15** | High-fidelity multi-vehicle simulation | Industry standard for AV research. Full sensor simulation. |
| **OPV2V** | Cooperative perception dataset | Multi-vehicle LiDAR + camera with ground truth. Built for exactly this problem. |
| **DAIR-V2X** | Vehicle-infrastructure cooperation dataset | Different cooperation paradigm — good for generalization. |
| **nuScenes** | Large-scale single-vehicle dataset | Useful for pre-training perception models. |
| **Custom network simulator (C++)** | Realistic V2X channel model | Bandwidth, latency, loss, congestion. Not NS-3 (overkill) but not a toy either. Implements standard channel models. |

### LLM / Agentic AI

| Technology | Purpose | Why |
|---|---|---|
| **vLLM** | Local LLM serving | Production-grade inference server. PagedAttention, continuous batching. What companies deploy. |
| **Llama 3 / Mistral** (open-weight) | Base LLM for reasoning | No API dependency, reproducible, customizable |
| **OpenAI / Anthropic API** | Optional cloud LLM | For comparison or when you need frontier capability |
| **LangGraph** | Agent orchestration | More principled than LangChain for multi-step agent workflows. Graph-based. |
| **Qdrant** | Vector database for RAG | Production-grade, fast, typed API. Better than ChromaDB for anything serious. |
| **BGE / E5** embedding models | Document/query embeddings | Top-performing open embedding models. Better than basic sentence-transformers. |

### Experiment Infrastructure

| Technology | Purpose | Why |
|---|---|---|
| **Weights & Biases** | Experiment tracking, sweeps, artifacts | Industry standard. Used by OpenAI, DeepMind, most ML teams. |
| **Hydra** | Configuration management | Facebook Research standard. Composable configs, CLI overrides, sweep support. No more YAML hell. |
| **DVC** (Data Version Control) | Dataset/model versioning | Track large files (datasets, checkpoints) alongside code. Reproducibility. |
| **MLflow** (alternative to W&B) | If you want self-hosted tracking | Open-source, no vendor lock-in |

### Testing & Quality

| Technology | Purpose | Why |
|---|---|---|
| **pytest** | Python testing | Standard |
| **Google Test (gtest)** | C++ testing | Industry standard for C++ unit tests |
| **mypy** (strict) | Python static type checking | Catches bugs before runtime. Non-negotiable for production code. |
| **ruff** | Python linting + formatting | 10-100x faster than pylint + black. Single tool. |
| **clang-tidy** | C++ static analysis | Catches bugs, enforces style |
| **clang-format** | C++ formatting | Consistent style |
| **pre-commit** | Git hooks for quality gates | Run linting, typing, formatting before every commit |
| **Hypothesis** | Property-based testing (Python) | Test fusion/uncertainty logic with generated inputs. Finds edge cases. |

### Infrastructure & DevOps

| Technology | Purpose | Why |
|---|---|---|
| **Docker + Docker Compose** | Reproducible environments | Multi-container: simulation, ML, world model, demo |
| **uv** | Python package management | 10-100x faster than pip. Lockfiles. Reproducible. Created by Astral (ruff creators). |
| **GitHub Actions** | CI/CD | Tests, linting, type checking on every push |
| **Bazel** (optional, for monorepo at scale) | Build system for multi-language | What Google/Waymo use. Consider only if repo grows large. |

### Observability & Debugging

| Technology | Purpose | Why |
|---|---|---|
| **Prometheus + Grafana** | Runtime metrics (latency, throughput, queue depth) | Industry standard observability stack |
| **OpenTelemetry** | Distributed tracing (follow a message through the system) | Trace an observation from detection → prioritization → transmission → fusion → decision |
| **structlog** | Structured logging (Python) | JSON logs, machine-parseable, filterable |
| **spdlog** | High-performance logging (C++) | Fast, formatted, standard in C++ systems |

### Visualization & Demo

| Technology | Purpose | Why |
|---|---|---|
| **React + Three.js** or **Rerun.io** | 3D visualization of scenarios | Rerun is specifically built for robotics/AV visualization. Production quality. |
| **Plotly Dash** or **Streamlit** | Experiment dashboard | Interactive result exploration |
| **Open3D** | Point cloud visualization | Standard for LiDAR data |
| **FFmpeg** | Video generation for demos | Programmatic video creation from scenario replays |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              EXPERIMENT ORCHESTRATION                             │
│                    (Hydra configs → run experiments → W&B logging)                │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼─────────────────────────────────────────┐
│                              PYTHON RESEARCH LAYER                                │
│                                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ Perception       │  │ RL Policy        │  │ LLM / Agentic Reasoning        │  │
│  │                  │  │                  │  │                                 │  │
│  │ PyTorch models   │  │ CleanRL / RLlib  │  │ vLLM serving                   │  │
│  │ OpenCOOD         │  │ Policy networks  │  │ LangGraph agents               │  │
│  │ OpenPCDet        │  │ Reward shaping   │  │ Qdrant vector store            │  │
│  │ ONNX export      │  │ Training loop    │  │ Investigation / Explanation    │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────┬──────────────────┘  │
│           │                     │                            │                    │
└───────────┼─────────────────────┼────────────────────────────┼────────────────────┘
            │                     │                            │
            │              pybind11 / gRPC                     │
            │                     │                            │
┌───────────┼─────────────────────┼────────────────────────────┼────────────────────┐
│           ▼                     ▼                            ▼                    │
│                              C++ RUNTIME LAYER                                    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                         AGENT RUNTIME                                        │ │
│  │                                                                              │ │
│  │  ┌─────────────┐   ┌────────────────┐   ┌─────────────────────────────────┐ │ │
│  │  │ World Model │   │ Communication  │   │ Active Information              │ │ │
│  │  │             │   │ Engine         │   │ Acquisition                     │ │ │
│  │  │ Entities    │   │                │   │                                 │ │ │
│  │  │ Spatial idx │   │ Prioritizer    │   │ Value estimation                │ │ │
│  │  │ Temporal    │   │ Channel model  │   │ Source selection                │ │ │
│  │  │ Fusion      │   │ Serialization  │   │ Request management             │ │ │
│  │  │ Uncertainty │   │ Receiver model │   │                                 │ │ │
│  │  │ Trust       │   │ Budget mgmt    │   │                                 │ │ │
│  │  └─────────────┘   └────────────────┘   └─────────────────────────────────┘ │ │
│  │                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │                     SIMULATION INTERFACE                                     │ │
│  │                                                                              │ │
│  │  CARLA Client  │  OPV2V Replay  │  Network Simulator  │  Scenario Engine    │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              OBSERVABILITY                                        │
│                                                                                  │
│  OpenTelemetry traces  │  Prometheus metrics  │  Grafana dashboards  │  structlog │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
omniworld/
├── README.md
├── LICENSE                             # Apache 2.0
├── ARCHITECTURE_AND_PLAN.md
├── OmniCopilot_Refined.md
│
├── .github/
│   └── workflows/
│       ├── ci-cpp.yml                  # Build + test C++ (CMake + gtest)
│       ├── ci-python.yml               # Lint + type-check + test Python
│       └── integration.yml             # Full system integration tests
│
├── .pre-commit-config.yaml             # Quality gates before every commit
├── pyproject.toml                      # Python dependencies (uv)
├── uv.lock                             # Lockfile
│
│── docker/
│   ├── Dockerfile.runtime              # C++ runtime + Python bindings
│   ├── Dockerfile.ml                   # ML training environment (CUDA)
│   ├── Dockerfile.sim                  # CARLA + simulation
│   └── docker-compose.yml             # Full system (all containers)
│
│── cmake/
│   └── CMakeLists.txt                  # Top-level CMake
│
│── proto/                              # Protocol Buffer definitions (THE contract)
│   ├── observation.proto               # Agent observation message
│   ├── entity.proto                    # World model entity
│   ├── world_state.proto               # Full world state snapshot
│   ├── communication.proto             # Communication messages (request, response, broadcast)
│   └── metrics.proto                   # Metrics reporting
│
├── cpp/                                # C++ source tree
│   ├── CMakeLists.txt
│   │
│   ├── core/                           # Core libraries
│   │   ├── world_model/
│   │   │   ├── world_model.h / .cc     # Main world model class
│   │   │   ├── entity.h / .cc          # Entity representation
│   │   │   ├── spatial_index.h / .cc   # R-tree or grid-based spatial lookup
│   │   │   ├── temporal_tracker.h / .cc # Temporal state estimation
│   │   │   ├── fusion.h / .cc          # Multi-source fusion algorithms
│   │   │   └── uncertainty.h / .cc     # Confidence representation + decay
│   │   │
│   │   ├── communication/
│   │   │   ├── channel.h / .cc         # Network channel model
│   │   │   ├── prioritizer.h / .cc     # Message prioritization (hand-designed)
│   │   │   ├── receiver_model.h / .cc  # Belief model of receivers
│   │   │   ├── budget_manager.h / .cc  # Bandwidth budget allocation
│   │   │   └── serializer.h / .cc      # Protobuf/FlatBuffers serialization
│   │   │
│   │   ├── trust/
│   │   │   ├── reliability.h / .cc     # Per-agent trust scoring
│   │   │   ├── anomaly.h / .cc         # Anomaly detection
│   │   │   └── byzantine.h / .cc       # Byzantine fault tolerance
│   │   │
│   │   ├── active_acquisition/
│   │   │   ├── info_value.h / .cc      # Information value estimation
│   │   │   ├── source_selector.h / .cc # Select best agent to query
│   │   │   └── request_manager.h / .cc # Manage outstanding requests
│   │   │
│   │   └── agent/
│   │       ├── agent.h / .cc           # Agent runtime (main loop)
│   │       └── config.h               # Agent configuration
│   │
│   ├── sim/                            # Simulation interfaces
│   │   ├── carla_client.h / .cc
│   │   ├── opv2v_replay.h / .cc
│   │   ├── network_sim.h / .cc         # Network condition simulation
│   │   └── scenario_engine.h / .cc
│   │
│   ├── bindings/                       # pybind11 bindings
│   │   ├── py_world_model.cc
│   │   ├── py_communication.cc
│   │   ├── py_agent.cc
│   │   └── py_simulation.cc
│   │
│   └── tests/                          # C++ unit tests (gtest)
│       ├── test_world_model.cc
│       ├── test_fusion.cc
│       ├── test_communication.cc
│       ├── test_trust.cc
│       └── test_spatial_index.cc
│
├── python/                             # Python source tree
│   └── omnicopilot/
│       ├── __init__.py
│       ├── py.typed                    # PEP 561 typed package marker
│       │
│       ├── perception/                 # ML perception models
│       │   ├── __init__.py
│       │   ├── detector.py             # 3D object detection (wraps OpenPCDet/OpenCOOD)
│       │   ├── tracker.py              # Multi-object tracking (ByteTrack or similar)
│       │   ├── cooperative.py          # Cooperative perception models (V2VNet, Where2comm)
│       │   └── export.py              # ONNX export for production inference
│       │
│       ├── fusion/                     # Python-side fusion research
│       │   ├── __init__.py
│       │   ├── algorithms.py           # Fusion algorithm implementations (for research)
│       │   ├── uncertainty.py          # Uncertainty estimation models (MC Dropout, ensembles, evidential)
│       │   └── calibration.py          # Confidence calibration
│       │
│       ├── rl/                         # Reinforcement learning
│       │   ├── __init__.py
│       │   ├── environment.py          # Gymnasium env for communication policy
│       │   ├── reward.py               # Reward function design
│       │   ├── policy.py               # Policy network architecture
│       │   ├── train.py                # Training loop (CleanRL-style)
│       │   └── analyze.py             # Emergent strategy analysis
│       │
│       ├── reasoning/                  # LLM / agentic layer
│       │   ├── __init__.py
│       │   ├── investigation.py        # Investigation agent (LangGraph)
│       │   ├── explanation.py          # Explanation generation
│       │   ├── risk.py                 # Risk assessment agent
│       │   └── rag/
│       │       ├── __init__.py
│       │       ├── indexer.py          # Document ingestion pipeline
│       │       ├── retriever.py        # Retrieval with reranking
│       │       └── store.py            # Qdrant interface
│       │
│       ├── data/                       # Dataset loading and preprocessing
│       │   ├── __init__.py
│       │   ├── opv2v.py               # OPV2V dataset loader
│       │   ├── dair_v2x.py            # DAIR-V2X dataset loader
│       │   ├── transforms.py          # Data augmentation and preprocessing
│       │   └── schemas.py             # Data validation schemas (Pydantic)
│       │
│       ├── evaluation/                 # Metrics and evaluation
│       │   ├── __init__.py
│       │   ├── metrics.py              # mAP, precision, recall, communication efficiency
│       │   ├── ablation.py             # Ablation study runner
│       │   ├── comparative.py          # Baseline comparison framework
│       │   └── statistical.py          # Statistical significance tests
│       │
│       └── config/                     # Hydra configuration
│           ├── __init__.py
│           ├── config.py               # Typed config dataclasses
│           └── defaults/
│               ├── base.yaml
│               ├── experiment/
│               │   ├── single_agent.yaml
│               │   ├── cooperative_naive.yaml
│               │   ├── cooperative_intelligent.yaml
│               │   └── cooperative_learned.yaml
│               ├── model/
│               │   ├── pointpillars.yaml
│               │   ├── centerpoint.yaml
│               │   └── v2vnet.yaml
│               ├── communication/
│               │   ├── unlimited.yaml
│               │   ├── constrained_5mbps.yaml
│               │   └── degraded.yaml
│               └── ablation/
│                   ├── no_uncertainty.yaml
│                   ├── no_prioritization.yaml
│                   ├── no_receiver_awareness.yaml
│                   ├── no_active_acquisition.yaml
│                   └── no_trust.yaml
│
├── scripts/                            # Entry points
│   ├── train_perception.py             # Train/fine-tune perception model
│   ├── train_policy.py                 # Train RL communication policy
│   ├── run_experiment.py               # Run experiment from Hydra config
│   ├── run_ablation_suite.py           # Run all ablations
│   ├── export_model.py                 # Export to ONNX
│   ├── launch_demo.py                  # Start demo application
│   └── generate_report.py             # Generate results report
│
├── notebooks/                          # Research exploration (NOT production code)
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_agent_baseline.ipynb
│   ├── 03_cooperative_analysis.ipynb
│   ├── 04_communication_policy.ipynb
│   ├── 05_emergent_strategies.ipynb
│   └── 06_ablation_results.ipynb
│
├── tests/                              # Python tests
│   ├── conftest.py                     # Shared fixtures
│   ├── unit/
│   │   ├── test_perception.py
│   │   ├── test_fusion.py
│   │   ├── test_rl_environment.py
│   │   ├── test_reward.py
│   │   └── test_data_loading.py
│   ├── integration/
│   │   ├── test_agent_pipeline.py      # Full agent: observe → fuse → decide → communicate
│   │   ├── test_multi_agent.py         # Multiple agents interacting
│   │   └── test_experiment_runner.py
│   └── property/
│       ├── test_fusion_properties.py   # Hypothesis property-based tests
│       └── test_uncertainty_properties.py
│
├── demo/                               # Visualization / demo application
│   ├── app.py                          # Main demo (Rerun or Plotly Dash)
│   ├── components/
│   │   ├── scenario_viewer.py          # 3D scenario visualization
│   │   ├── world_model_viewer.py       # World model state display
│   │   ├── communication_viewer.py     # Message flow visualization
│   │   └── metrics_panel.py           # Live metrics
│   └── assets/
│
├── deploy/                             # Deployment configurations
│   ├── prometheus.yml                  # Metrics collection config
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── system_overview.json
│   │       └── experiment_tracking.json
│   └── otel-collector.yml             # OpenTelemetry config
│
├── docs/                               # Documentation
│   ├── setup.md                        # Development environment setup
│   ├── architecture.md                 # Detailed architecture decisions
│   ├── experiments.md                  # Experiment log
│   ├── api/                           # Auto-generated API docs (Sphinx for Python, Doxygen for C++)
│   └── paper/
│       ├── main.tex                    # Technical paper
│       ├── figures/
│       └── references.bib
│
└── data/                               # Data directory (gitignored, managed by DVC)
    ├── .dvc/
    ├── raw/                            # Raw datasets
    ├── processed/                      # Preprocessed data
    ├── models/                         # Trained model checkpoints
    └── results/                        # Experiment results
```

---

## Interface Contracts (Protobuf — the source of truth)

The `.proto` files define the contract between C++ and Python, between agents, and between components. This is non-negotiable in a multi-language system.

Example `proto/observation.proto`:
```protobuf
syntax = "proto3";
package omnicopilot;

message Observation {
  string agent_id = 1;
  string object_id = 2;
  ObjectClass object_class = 3;
  Position position = 4;
  Velocity velocity = 5;
  float confidence = 6;
  int64 timestamp_ns = 7;
  SensorType sensor = 8;
  BoundingBox3D bbox = 9;
}

enum ObjectClass {
  UNKNOWN = 0;
  VEHICLE = 1;
  PEDESTRIAN = 2;
  CYCLIST = 3;
  OBSTACLE = 4;
}

enum SensorType {
  CAMERA = 0;
  LIDAR = 1;
  RADAR = 2;
  FUSED = 3;
}

message Position {
  double x = 1;
  double y = 2;
  double z = 3;
}

message Velocity {
  double vx = 1;
  double vy = 2;
  double vz = 3;
}

message BoundingBox3D {
  Position center = 1;
  double length = 2;
  double width = 3;
  double height = 4;
  double heading = 5;
}
```

---

## Success Criteria, Novelty & Scientific Rigor

This section is the scientific backbone of the project. It defines what "success" means
in measurable terms, where the project must stop and reassess, how results are validated,
and where the actual contribution lies. Read this before the phase breakdown — the phases
exist to satisfy these criteria.

### Minimum Viable Core vs. Extensions

The project is layered so that a complete, defensible result exists early, and everything
beyond it is additive.

| Layer | Phases | Status |
|---|---|---|
| **Minimum Viable Core** | Phases 1–5 | If ONLY these are done, the project is complete and defensible: single-agent baseline, cooperative perception, intelligent communication, and a learned communication policy with a measured result. |
| **Strong Version** | Phases 6–8 | Active acquisition, adversarial robustness, ablation study. Turns a good result into a rigorous one. |
| **Specialization + Advanced** | Phases 8.5, 9, 9.4, 3.5 | Realistic V2X networking, self-improvement, LLM reasoning, multi-agent LLM team, uncertainty refinement. Depth, not load-bearing. |

Rule: never start a later layer while an earlier layer's success criteria are unmet.
Finishing Phases 1–5 well beats half-finishing all ten phases.

### Concrete Success Criteria (define targets BEFORE building)

Each core claim needs a numeric target agreed in advance, so success is unambiguous and
failure is detectable early. **Targets below are now GROUNDED in the OPV2V paper's
published numbers (from the prior-art scan), not guesses.** Reference points from OPV2V
(PointPillar, AP@IoU=0.7, default towns): single-agent (No Fusion) ≈ **0.60**, cooperative
(Intermediate Fusion) ≈ **0.82** → the field shows ~+20 AP points (~+33% relative), and
**the benefit saturates past ~4 CAVs.** Refine after measuring OUR single-agent baseline
in Phase 2, but do NOT leave them undefined.

| Claim | Metric | Target (go) | Concern threshold (investigate) |
|---|---|---|---|
| Cooperation helps | Occluded-object AP@0.7: cooperative vs. single-agent | ≥ +15 AP points (grounded in OPV2V's ~+20) | < +5 AP → premise weak on this data |
| Intelligent comm. helps | Communication volume at equal accuracy: intelligent vs. broadcast (log-scale bytes, per Where2comm convention) | ≤ 60% of broadcast volume | > 90% → prioritization adds little |
| Learning beats engineering | Learned vs. hand-designed policy (accuracy at equal comm. volume) | Learned ≥ hand-designed, ideally +5% | Learned < hand-designed → RL not justified |
| Active acquisition helps | Time-to-detection for occluded objects: active vs. passive | ≥ 20% faster | < 10% → acquisition marginal |
| Robustness | Accuracy retained at 20% packet loss | ≥ 90% of clean accuracy | < 75% → fragile |
| Realistic V2X transfer (8.5) | Learned policy advantage under realistic C-V2X vs. under idealized | Advantage retained; grows with density | Advantage vanishes → networking assumption mattered |

These become the headline numbers. Every experiment reports against them.

### Kill Gates (explicit stop-and-reassess points)

The plan must not blindly march forward assuming success. At each gate, if the concern
threshold is hit, STOP the forward march and diagnose before spending further weeks.

- **Gate A — end of Phase 3 (cooperation).** If cooperative perception does not clear the
  "concern threshold" for occluded-object improvement, the core premise is weak on OPV2V.
  Do NOT proceed to communication work. Diagnose: wrong scenes? coordinate-frame bug?
  detector too strong already? Consider switching scenes/dataset (V2XSet, DAIR-V2X) or
  focusing on harder occlusion cases.
- **Gate B — end of Phase 4 (intelligent communication).** If intelligent prioritization
  gives no bandwidth advantage over broadcast, reconsider before investing in RL (Phase 5).
- **Gate C — mid Phase 5 (RL convergence).** If PPO has not shown stable learning within
  the compute budget, invoke the RL fallback (see Phase 5 contingency) rather than burning
  compute.

A gate being hit is INFORMATION, not failure. It redirects effort early.

### Novelty Positioning (internal reference — informs the paper's "contribution" framing)

The individual building blocks are published and mature — cooperative perception
(V2VNet, DiscoNet, OPV2V, V2X-ViT, Where2comm, CoBEVT), V2X communication, LLM/agentic
orchestration, uncertainty in perception. **None of these individually is a contribution
of this project, and none should be claimed as such.**

The contribution of this project is **Tier 3: a novel combination + rigorous analysis
under an underexplored set of constraints.** Specifically, the defensible gaps — **as
confirmed by the prior-art scan (`docs/prior_art.md`) of V2VNet, OPV2V, DiscoNet, V2X-ViT,
CoBEVT, and Where2comm**:

> IMPORTANT CORRECTION from the prior-art scan: "communicating only what matters" and
> "receiver requests missing info" are NOT novel — **Where2comm (NeurIPS 2022) already
> does both** (spatial confidence maps + request map + multi-round). Do NOT claim these
> as contributions. The surviving, genuinely underexplored gaps are narrower and sharper:

1. **Learned communication under REALISTIC network constraints.** Every paper scanned
   (including Where2comm and V2X-ViT) assumes an idealized channel, or at most Gaussian
   pose/time noise — NONE model the actual V2X channel (CSMA/CA contention, congestion
   control, distance-based path loss, density-dependent throughput collapse). Training the
   "what/when/to-whom to communicate" decision as an RL policy and evaluating it under
   realistic V2X mechanics, showing whether the advantage transfers and scales with vehicle
   density — this is the STRONGEST and clearest gap. It is Developer A's networking
   specialization (Phase 8.5 + Phase 5).
2. **RL-learned policy vs. supervised differentiable selection.** Where2comm's selection is
   a supervised spatial mask over feature maps. A reinforcement-learned policy optimizing a
   long-horizon communication reward (with delayed effects, congestion feedback) is a
   different, underexplored formulation.
3. **Object/world-model-level cooperation with explicit trust + adversarial resilience.**
   All six papers assume honest agents and operate at the feature level. Trust-weighted
   fusion + Byzantine resilience at the object level (Phase 7) is open.
4. **LLM/agentic reasoning layer feeding structured events back into cooperation** (Phase
   9) — orthogonal to all six; none touch high-level reasoning.
5. **The integrated system + ablation** — all of the above evaluated together under
   realistic constraints.

**Honest limitation to state plainly:** feature-level fusion (Where2comm, V2X-ViT) will
likely beat our object-level world model on raw detection mAP. Our value is
interpretability + trust + realistic-comms + reasoning, NOT beating them on mAP. Consider
adopting a feature-fusion backbone (via OpenCOOD) and layering our contributions on top,
rather than competing on raw accuracy. Evaluate against Where2comm on
communication-efficiency-under-realistic-channel and robustness, not raw mAP.

**Key principle:** novelty is proven by RESULTS, not by the plan. The exact contribution
statement is finalized only after the literature review (Task 1.5) and the first results.
The sharpest single-sentence framing to aim for:

> "Communication as a learned, receiver-aware decision evaluated under realistic V2X
> network constraints — transferring from idealized training to realistic C-V2X, with the
> benefit growing under congestion."

### Evaluation Protocol (how results are validated — non-negotiable for credibility)

- **Multiple seeds.** All RL and stochastic results reported over ≥ 5 random seeds. Report
  mean ± std (or 95% CI), never a single run.
- **Paired comparison.** All conditions (single/naive/intelligent/learned) evaluated on the
  IDENTICAL set of test scenes, so comparisons are paired.
- **Statistical test.** Paired bootstrap or paired t-test; report p-values for headline
  claims; treat p < 0.05 as significant. State the test used.
- **Confidence intervals** on every headline number.
- **Ablation isolation.** Change one component at a time (Phase 8); never confound.
- **Held-out data.** Tune on validation split; report final numbers on test split only,
  once.

### Reproducibility Artifact

- Every headline figure/number must be regenerable from: pinned data version (DVC) +
  exact config (Hydra) + fixed seed + logged run (W&B run ID).
- Provide a single entry point (e.g. `scripts/reproduce.py --result <name>`) that
  regenerates each key result. A cloned repo must reproduce the core numbers with one
  command. "Trust me" is not acceptable.

### Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Coordinate-frame / pose errors corrupt fusion | High | Critical | Explicit transform task (3.0); validate on GT poses before fusion | A |
| R2 | Cooperation shows little gain on OPV2V | Medium | Critical | Kill Gate A; harder occlusion scenes; V2XSet/DAIR-V2X fallback | Both |
| R3 | RL policy fails to converge | Medium | High | Phase 5 fallback: imitation learning / contextual bandit | B |
| R4 | Time/clock skew across agents misaligns observations | Medium | High | State sync assumption or model skew (Task 3.0) | A |
| R5 | Idea overlaps a published method | Medium | Medium | Prior-art scan (Task 1.5) BEFORE building | B |
| R6 | Demo scene doesn't exist in dataset | Low | Medium | Verify the blind-intersection scene in Phase 1 (Task 1.5) | Both |
| R7 | Scope creep from advanced modules | Medium | Medium | Minimum-viable-core rule; extensions gated behind core | Both |
| R8 | Covariance numerical drift in hand-rolled KF | Low | Medium | Eigen adoption in Task 3.5d | A |
| R9 | Object-level fusion loses to feature fusion on raw mAP | High | Medium | Two-track design (below): feature-fusion backbone for accuracy, object layer for reasoning | Both |

### Corrections From the Prior-Art Scan (design changes)

After reading V2VNet, OPV2V, DiscoNet, V2X-ViT, CoBEVT, and Where2comm (see
`docs/prior_art.md`), the following plan corrections apply. These override earlier framing.

**C1 — Two-track fusion (accuracy vs. reasoning) + USE OPENCOOD FOR DATA LOADING.**
All six SOTA papers fuse at the FEATURE level (share neural feature maps); OPV2V's own
numbers show late/object-level fusion is weaker (AP@0.7 0.78 vs. 0.82 for intermediate).
Therefore:
- Do NOT try to win raw detection mAP with our object-level `FusionEngine`.
- **Adopt a feature-fusion backbone from OpenCOOD** (e.g. AttFuse/V2VNet/Where2comm) as
  the perception+detection layer (Phase 2-3).
- Keep our object-level world model / trust / existence-confidence / comms-policy /
  LLM layer LAYERED ON TOP — its value is interpretability, trust, adversarial resilience,
  realistic-comms evaluation, and reasoning, NOT raw mAP. Evaluate our contributions on
  THOSE axes, not on beating Where2comm's detection accuracy.

**C1b — FIRM DECISION: use OpenCOOD's data loading + coordinate transforms; do NOT
hand-roll them.** Verified empirically: hand-rolling the OPV2V coordinate transform
produced a 70m error (see below). OPV2V ground-truth `location` is in the CARLA GLOBAL
frame, and cross-agent projection MUST use OpenCOOD's convention:
```
# opencood/utils/transformation_utils.py
x_to_world(pose)        # [x,y,z,roll,yaw,pitch] (DEGREES) -> 4x4 local->world
x1_to_x2(x1, x2):       # transform agent x1's frame -> agent x2's frame
    return inv(x_to_world(x2)) @ x_to_world(x1)
```
Key facts learned from the real data + OpenCOOD:
- GT `location` is GLOBAL (CARLA map) coords, NOT ego frame. Our earlier assumption was
  wrong; multiplying by pose again double-transformed → the 70m error.
- Cross-agent transform ALWAYS inverts the destination pose: `inv(x2_to_world) @ x1_to_world`.
  We never inverted → part of the error.
- Angles are in DEGREES (OpenCOOD calls `np.radians` internally).
- `extent` is HALF-dimensions (double for full L/W/H).
- Object dict keys are per-agent; `ass_id` (often -1) would be the cross-agent identity —
  do NOT assume matching dict keys mean the same physical car.
- YAML uses embedded numpy objects → must use `yaml.unsafe_load`, not `safe_load`.

**Action:** Task 1.3's `opv2v.py` becomes a THIN WRAPPER around OpenCOOD's `BaseDataset` +
`opencood.utils.transformation_utils`, NOT a from-scratch parser. Reuse their
`x_to_world` / `x1_to_x2` for ALL coordinate math. This eliminates the coordinate-bug
class entirely and is the concrete realization of correction C1. Task 3.0 (coordinate
transforms) correspondingly shrinks to "adopt and wrap OpenCOOD's transforms," not
"implement transforms."

**C2 — Shared BEV coordinate frame.** Where2comm avoids per-observation coordinate
transforms by projecting all agents into a common BEV/global frame up front. Adopt the
same: work in a shared BEV frame. Task 3.0 still handles pose error, but the transform
mechanics are simpler than originally feared.

**C3 — Pose error & time delay are SOLVED techniques — reuse, don't reinvent.** V2VNet
(Δt warping), DiscoNet (pose-error regression), and V2X-ViT (delay-aware positional
encoding; noise levels σxyz∈[0,0.5]m, σheading∈[0,1°], delay=100ms) already handle these.
Task 3.0 reuses these standard approaches and uses V2X-ViT's noise levels as the
robustness test settings. Do NOT claim pose/time handling as a contribution.

**C4 — Communication metric convention.** Report communication as log-scale volume
(bytes) per the Where2comm convention, so results are directly comparable to the field.
Update `CommunicationMetrics` accordingly.

**C5 — Agent count.** OPV2V has 2-7 CAVs (mean 2.89) and benefit saturates past ~4.
Standardize experiments on 2-5 agents; run the density-scaling experiment (Phase 8.5) up
to 7. Align all configs (ScenarioEngine default, experiment YAMLs) to this range.

**C6 — Add When2com to the prior-art reading (Task 1.5).** When2com (CVPR 2020) is a
handshake mechanism deciding WHEN and WITH WHOM to communicate — very close to our
"who to communicate with." Read it before claiming any novelty there.

**C7 — Baseline set for comparison.** Our headline comparisons must include Where2comm as
the strongest baseline, compared on OUR axis (communication efficiency under a REALISTIC
channel + robustness), not on raw mAP where feature fusion wins.

---

## Task Breakdown — Phase by Phase

### Phase 1: Foundation (Weeks 1-2)

#### Task 1.1 — Repository & Build System [Developer A]
- Initialize git repo with proper .gitignore
- Set up CMake for C++ (with gtest, protobuf, pybind11 as dependencies)
- Set up pyproject.toml with uv
- Configure pre-commit hooks (ruff, mypy, clang-format, clang-tidy)
- Set up GitHub Actions CI for both C++ and Python
- Create Docker files (runtime, ml, sim)
- Set up Hydra config structure

#### Task 1.2 — Proto Definitions & Core Types [Developer A]
- Write all .proto files (observation, entity, world_state, communication)
- Generate C++ and Python code from protos
- Define C++ core types (Entity, WorldState interface)
- Define Python Pydantic schemas for data validation
- Write unit tests for serialization round-trip

#### Task 1.3 — Data Pipeline [Developer B]
- Download OPV2V dataset
- Implement `opv2v.py` dataset loader with proper typing
- Validate data against Pydantic schemas
- Set up DVC for dataset versioning
- Write `01_data_exploration.ipynb`
- Set up W&B project, log first dataset statistics

#### Task 1.4 — Development Environment Documentation [Both]
- Write `docs/setup.md` — how to build C++, install Python deps, run tests
- Verify Docker build works end-to-end
- Ensure `make build && make test` works from clean checkout

#### Task 1.5 — Prior-Art Scan & OpenCOOD Study [Developer B lead, Both review]

**This is engineering diligence, NOT an academic literature review.** No citations, no
related-work prose, no comprehensive survey. The goal is purely practical: reuse what
exists, avoid known landmines, confirm the chosen angle isn't already solved, and verify
the demo scene exists. Budget ~2 days, do it in parallel with setup — before heavy
building.

**Read 3-4 papers, SKIMMING most sections.** For each, answer only three questions in
`docs/prior_art.md` (half a page per paper is plenty):
1. What can I REUSE? (data format, baseline numbers, evaluation metrics, code)
2. What do they ASSUME that I'm relaxing? (usually: idealized/unlimited communication)
3. What did they try that DIDN'T work, or what pitfalls do they warn about?

Priority reading order:
- **OPV2V (Xu et al., ICRA 2022)** — MUST read; you use their data. Focus on the dataset
  structure, coordinate frames, and reported single-agent vs. cooperative baselines.
- **Where2comm (Hu et al., NeurIPS 2022)** — closest neighbor; they study "what to
  communicate" via spatial confidence maps. Read to know exactly what's DONE so you build
  the part that isn't (the realistic-network-constraints + learned-policy angle).
- **V2X-ViT (Xu et al., ECCV 2022)** — read for the early/mid/late fusion tradeoff and
  how mid-level feature fusion works. Decides which fusion level you target.
- **CoBEVT (Xu et al., CoRL 2022)** — optional; BEV cooperative fusion, skim only.

Supplementary (skim / reference, not full reads):
- **When2com (Liu et al., CVPR 2020)** — handshake mechanism for WHEN and WITH WHOM to
  communicate; closest prior work to our "who to communicate with." Read before claiming
  any novelty on collaborator selection. (Correction C6.)
- **V2VNet (Wang et al., ECCV 2020)** — the original "share intermediate features"
  cooperative perception paper; read the intro/method to understand the lineage.
- **DiscoNet (Li et al., NeurIPS 2021)** — knowledge-distillation-based cooperative
  perception; optional.
- A cooperative-perception survey (e.g. "Collaborative Perception" surveys on arXiv,
  2023-2024) — use as a map to find anything above that's relevant, not a full read.

**Most valuable of all — study OpenCOOD directly** (https://github.com/DerrickXuNu/OpenCOOD):
- It implements V2VNet, DiscoNet, Where2comm, CoBEVT, V2X-ViT and loads OPV2V/V2XSet.
- Read the README + data loaders + one model's forward pass. This tells you what is
  ALREADY built (so you don't rebuild it) and how the data actually flows — worth more
  than any single paper for practical purposes.

**Also confirm:**
- The target demo scenario (blind intersection: an occluded pedestrian visible to one
  agent but not another) actually EXISTS in the OPV2V test split. Note the scene ID(s).
  If not, pick the closest available scene NOW, not in Phase 10. (Risk R6.)
- The angle we intend to occupy (learned, receiver-aware communication under REALISTIC
  V2X constraints) is not already fully covered by the papers above. Confirm the gap.

**Deliverables:** `docs/prior_art.md` (reuse / assumptions / pitfalls per paper + a
one-paragraph "our angle vs. what exists" note) + confirmed demo scene ID(s) + a short
list of OpenCOOD components we will reuse rather than reimplement.

---

### Phase 2: Single Agent Baseline (Weeks 3-4)

#### Task 2.1 — Perception Model Integration [Developer B]
- Integrate pre-trained 3D detector (PointPillars or CenterPoint via OpenPCDet)
- Wrap in typed `detector.py` interface
- Add uncertainty estimation (MC Dropout or ensemble)
- Export baseline model to ONNX
- Evaluate single-agent mAP on OPV2V — establish baseline numbers
- Log to W&B with full config

#### Task 2.2 — Agent Runtime (C++) [Developer A]
- Implement basic `Agent` class in C++ (observe → process → store)
- Implement `WorldModel` class (add observations, query entities)
- Implement `SpatialIndex` (R-tree or grid for efficient neighbor queries)
- Expose to Python via pybind11
- Write gtest unit tests for world model operations

#### Task 2.3 — Single Agent Pipeline [Both]
- Wire together: dataset → perception → world model → evaluation
- Implement `metrics.py` (mAP, precision, recall vs. ground truth)
- Create `single_agent.yaml` Hydra config
- Run and record baseline: "Single agent achieves X mAP"
- Identify failure cases (occluded objects) for cooperative experiments

---

### Phase 3: Basic Cooperative Perception (Weeks 5-8)

#### Task 3.0 — Coordinate Frames & Time Synchronization [Developer A] (PREREQUISITE — do first)

**This is the silent-failure landmine of cooperative perception. If this is wrong,
nothing downstream works — entities won't associate, fusion produces garbage. Do it
before any multi-agent fusion.** (Risks R1, R4.)

- **Coordinate transformation.** Each vehicle observes in its OWN local frame. Before
  fusing Agent A's and Agent B's observations, transform BOTH into a common frame
  (world frame, or a chosen ego frame) using each agent's pose matrix:
  `world_position = Pose_agent · local_position` (4×4 homogeneous transform).
  - Implement and unit-test the transform (round-trip: local → world → local must be
    identity within tolerance).
  - Validate on OPV2V ground-truth poses FIRST (clean poses), before touching fusion.
- **Pose error handling.** Real poses have GPS/localization drift (sub-meter to several
  meters). Add a task to (a) quantify pose error in the dataset, and (b) test how much
  pose error the fusion tolerates before association breaks. This is a known failure mode
  and a legitimate robustness result.
- **Time synchronization.** Observations carry timestamps from different agents. State the
  assumption explicitly:
  - Baseline: assume synchronized clocks (document this assumption).
  - Realistic (optional, ties to networking specialization): model clock skew and
    latency, and Kalman-predict all observations to a common reference time before
    fusion (the temporal tracker already supports predict-to-time).
- **Deliverable:** a tested coordinate-transform utility + a documented time-alignment
  strategy, both verified on OPV2V before Task 3.2 fusion begins.

#### Task 3.1 — Multi-Agent Simulation Engine [Developer A]
- Implement `ScenarioEngine` (C++) — manage multiple agents in a shared scene
- Implement time-step simulation (synchronous ticks)
- Implement `Channel` (C++) — initially unlimited (pass all messages)
- Expose multi-agent scenario to Python for experiment orchestration
- Integration test: 3 agents running simultaneously, exchanging observations

#### Task 3.2 — Fusion Engine [Developer B]
- Implement observation association (Hungarian algorithm / IoU matching)
- Implement confidence-weighted fusion (Bayesian update, Dempster-Shafer, or learned)
- Implement conflict resolution (contradictory observations)
- Implement in Python first, profile, port critical path to C++ if needed
- Property-based tests (Hypothesis): fusion properties (commutativity, monotonicity)

#### Task 3.3 — Uncertainty & Temporal Tracking [Developer A → C++]
- Implement confidence decay (exponential, configurable half-life)
- Implement `TemporalTracker` — Kalman filter or similar for state estimation
- Implement trajectory prediction (constant velocity, then learned)
- Implement stale-entity removal
- gtest: verify temporal consistency, decay behavior

#### Task 3.4 — Cooperative Evaluation [Both]
- Run: single agent vs. 3-agent naive cooperation (share everything)
- Measure: mAP improvement, occluded object detection rate
- Statistical significance test (paired t-test or bootstrap)
- Create comparison plots, log to W&B
- **MILESTONE: "Cooperation improves occluded detection by X%"**

#### Task 3.5 — Uncertainty Model Refinement [Developer A] (do AFTER 3.4 milestone)

**Design principle: "Kalman for WHERE, noisy-OR for WHETHER."** The system tracks two
distinct kinds of uncertainty and must not conflate them:

| Quantity | Question | Owner |
|---|---|---|
| **State uncertainty** | GIVEN it exists, how precisely do we know WHERE it is? | Kalman covariance `P` / innovation `S` |
| **Existence confidence** | Does the object REALLY exist, or is it a false positive? | Noisy-OR over independent trust-weighted sources |

Rationale: Kalman covariance can shrink happily around a *ghost* — feeding it repeated
false positives makes it "precise" about an object that isn't there. So covariance cannot
answer existence. Conversely, noisy-OR says nothing about localization precision. Each
tool must do only what it is good at.

**3.5a — Let the Kalman filter own ALL position/velocity fusion**
- Current `FuseObservation` updates position via a crude confidence-weighted running
  average AND the Kalman `Update` also fuses position → redundant, and the running
  average is the less-principled of the two.
- Refactor so `TemporalTracker` (Kalman) is the single source of truth for
  position/velocity/heading. `FuseObservation` should then only handle: existence
  confidence (noisy-OR), class voting, evidence provenance, and source counting.
- Remove the redundant running-average position update.

**3.5b — Mahalanobis-gated association (uses the Kalman `S` we already compute)**
- Current association uses plain Euclidean distance with a fixed 5m threshold.
- Upgrade to Mahalanobis distance:
  `d² = (obs − predicted)ᵀ · S⁻¹ · (obs − predicted)`
  where `S` is the innovation covariance already produced by the Kalman `Update`.
- Effect: when we are UNCERTAIN about an entity (large covariance) we allow matches from
  farther away; when we are PRECISE we require closer matches. This is what production
  trackers (SORT, JPDA, MHT) do and typically reduces ID switches.
- Keep a distance ceiling as a fallback gate for numerical safety.

**3.5c — Keep noisy-OR for existence (do NOT replace with covariance)**
- Existence confidence continues to drive lifecycle (Tentative→Confirmed), staleness,
  safety-critical flagging, and communication priority. These MUST stay probabilistic.
- Document the framing in the paper:
  *"We separate existence confidence (noisy-OR over independent sources) from state
  uncertainty (Kalman covariance), following the IPDA framework, with Mahalanobis-gated
  association."* — signals theoretical awareness to reviewers.

**Deliberately deferred / future work (do NOT do now):**
- Full IPDA (Integrated Probabilistic Data Association) — a rigorous separate
  track-existence recursion. Our noisy-OR is a simplified version; upgrade only if an
  experiment or reviewer demands it. Note it in the paper's future-work section.
- PHD / JPDA multi-hypothesis trackers — out of scope.

**3.5d — Adopt Eigen for the covariance / Mahalanobis math [Developer A]**

Until now the Kalman filter is hand-rolled (fixed 4-state constant-velocity, no external
dependency) — the right call for Phase 1-3: fast in the 10Hz hot loop, dependency-light,
and fully inspectable. Task 3.5 is the natural point to introduce **Eigen** (header-only
C++ linear algebra, industry standard in robotics/AV).

Why adopt it here specifically:
- Task 3.5b needs matrix inversion of the innovation covariance `S` and the quadratic
  form `(z − ẑ)ᵀ S⁻¹ (z − ẑ)`. Eigen makes this clean and correct as the state grows.
- **Numerical stability:** the current hand-rolled covariance update uses the simple
  form, which can lose symmetry / positive-definiteness over many iterations. Eigen's
  `LLT`/`LDLT` decompositions (and Joseph-form updates) guard against covariance drift.
- Header-only → no linking hassle; add via CMake `FetchContent` like gtest/spdlog.

Scope guidance:
- Keep the hand-rolled filter for the plain predict/update if it stays fast enough —
  Eigen is primarily for the NEW math (Mahalanobis gating, cleaner covariance ops).
- Do NOT pull in OpenCV's `cv::KalmanFilter` (heavy, ties core to OpenCV).
- If a NON-LINEAR motion model is ever needed (constant-turn-rate, bicycle model),
  that's the point to consider an EKF/UKF library (e.g. mherb/kalman) — future work,
  not this task.
- **Python side:** use **FilterPy** for any Kalman prototyping in notebooks (research
  only, not the hot loop). It is the standard, well-documented Python KF/EKF/UKF library.

**Why AFTER the 3.4 milestone (not before):** With no real data yet, tuning a
sophisticated uncertainty model optimizes for the wrong thing. Ship "cooperation helps"
on the simple model first, observe where the simple model actually fails on OPV2V, THEN
apply these targeted refinements. Avoids premature sophistication — the project's #1 risk.

- **MILESTONE: "Mahalanobis-gated association + Kalman-owned state fusion (Eigen-backed
  for numerical stability) reduce ID switches and localization error vs. the
  Euclidean/running-average baseline, while existence confidence remains a clean
  probabilistic signal for the communication layer."**

---

### Phase 4: Intelligent Communication (Weeks 9-10)

#### Task 4.1 — Network Channel Model [Developer A → C++]
- Implement realistic channel model:
  - Configurable bandwidth (messages/sec, bytes/sec)
  - Latency model (fixed + jitter)
  - Packet loss (Bernoulli or Gilbert-Elliott for burst loss)
  - Queue with configurable depth and drop policy
- Expose network parameters as runtime-configurable (for RL environment later)
- Test: verify message delivery under various conditions

#### Task 4.2 — Hand-Designed Prioritization [Developer A → C++]
- Implement `Prioritizer`:
  - Score: safety_relevance × confidence × novelty × time_sensitivity × receiver_benefit
  - Configurable weights per factor
  - Top-K selection within bandwidth budget
- Compare: unlimited vs. prioritized at 50%, 25%, 10% bandwidth

#### Task 4.3 — Receiver Belief Model [Developer B]
- Implement `ReceiverModel`:
  - Maintain estimated belief state of each other agent
  - Compute novelty: KL divergence between sender's observation and receiver's likely belief
  - Suppress messages where expected information gain < threshold
- Compare: naive priority vs. receiver-aware priority

#### Task 4.4 — Communication Evaluation [Both]
- Accuracy-bandwidth Pareto curve
- Show: intelligent communication achieves same accuracy at Y% less bandwidth
- Evaluate under degraded conditions (high loss, high latency)
- **MILESTONE: "Same perception accuracy at 40% less bandwidth"**

---

### Phase 5: Learned Communication Policy — RL (Weeks 11-14)

#### Task 5.1 — Gymnasium Environment [Developer A]
- Implement `CommunicationEnv(gymnasium.Env)`:
  - Observation space: agent's detections + estimated receiver states + network state
  - Action space: per-observation transmit probability (continuous) or top-K selection (discrete)
  - Step: simulate transmission through channel model, update receivers' world models
  - Reward: computed by reward function
  - Info: detailed metrics for analysis
- Verify environment with random policy (sanity check)

#### Task 5.2 — Reward Engineering [Developer B]
- Implement reward function:
  - Primary: Δ mAP at receivers (before vs. after message received)
  - Secondary: uncertainty reduction on safety-critical entities
  - Cost: bandwidth penalty (proportional to bytes sent)
  - Shaping: bonus for novel observations that change receiver's predictions
- Ablation: test different reward formulations, pick best

#### Task 5.3 — Policy Training [Developer B]
- Implement policy network (MLP or attention-based)
- Train with PPO (CleanRL implementation — transparent, debuggable)
- Hyperparameter sweep via W&B Sweeps
- Learning curve: show communication efficiency improving over episodes
- Train to convergence, save best checkpoint

#### Task 5.3b — RL Convergence Contingency [Developer B] (Kill Gate C)
RL is finicky and may not converge in the compute budget. This is planned for, not feared.
If PPO shows no stable learning by mid-Phase-5, fall back in this order:
1. **Imitation learning** — train the policy to mimic the hand-designed prioritizer
   (Phase 4), then fine-tune with RL. Gives a working policy immediately and a warm start.
2. **Contextual bandit** formulation — treat each communication decision as a one-step
   bandit (no long-horizon credit assignment). Much easier to train; often sufficient
   since communication reward is fairly immediate.
3. **Simpler action space** — top-K selection (discrete) instead of per-observation
   continuous scores, reducing the learning problem's dimensionality.
A working fallback policy that beats broadcast is a valid result. "We tried RL, it didn't
converge, so imitation learning" is honest and still demonstrates the pipeline. Do not
burn the entire compute budget chasing PPO convergence.

#### Task 5.4 — Comparison & Emergent Analysis [Both]
- Compare: hand-designed vs. learned policy across all metrics
- Record ALL communication decisions from learned policy over 1000 episodes
- Statistical analysis: where does learned policy disagree with baseline?
- Categorize emergent strategies
- Write analysis notebook
- **MILESTONE: "Learned policy outperforms hand-designed by Z% / discovers strategy X"**

---

### Phase 6: Active Information Acquisition (Weeks 15-16)

#### Task 6.1 — Information Value Estimator [Developer B]
- Implement `information_value.py`:
  - For each uncertain entity, estimate expected information gain from each agent
  - Consider: agent's position, sensor type, line-of-sight, historical reliability
  - Use mutual information or expected uncertainty reduction
- Compare: which estimation method works best?

#### Task 6.2 — Request-Response Protocol [Developer A → C++]
- Implement in `RequestManager`:
  - Typed request message (query about specific location/entity)
  - Budget: limited requests per timestep
  - Timeout handling
  - Response prioritization at receiving agent
- Integrate with channel model (requests consume bandwidth)
- **Expose `SubmitRequest()` as an OPEN input** so any caller can drive acquisition,
  not just the routine world-model logic. Two intended callers:
  1. The world model itself (routine, automatic — "this entity is uncertain, query it").
  2. LATER, the Phase 9 LLM reasoning layer (Task 9.4) — when the investigation
     agent(s) conclude "we should look at (X,Y)", they call this same `SubmitRequest`.
  This keeps Phase 6 fast C++ with NO LLM inside it, while letting the deliberative
  reasoning layer *drive* acquisition through the interface. Build the hook now so the
  Phase 9 connection needs zero changes to Phase 6 code later.
  (Design principle: "Phase 6 = the hands, Phase 9 = the brain; the brain reaches the
  hands through this interface, never by moving into them.")

#### Task 6.3 — Active vs. Passive Evaluation [Both]
- Compare: passive sharing only vs. passive + active requests
- Measure: time-to-detection for occluded objects, uncertainty reduction rate
- **MILESTONE: "Active acquisition detects hidden objects X% faster"**

---

### Phase 7: Adversarial Robustness (Weeks 17-18)

#### Task 7.1 — Trust Module [Developer A → C++]
- Implement `Reliability`:
  - Per-agent trust (EMA of accuracy vs. consensus)
  - Contextual trust (per-condition reliability)
  - Trust-weighted fusion integration
- Implement `Byzantine`: majority voting for critical observations

#### Task 7.2 — Anomaly Detection [Developer B]
- Physical plausibility detector (impossible kinematics)
- Statistical deviation detector (sudden accuracy change)
- Consensus deviation detector
- Combine detectors → trust adjustment signal

#### Task 7.3 — Adversarial Experiments [Both]
- Inject 1, 2, 3 adversarial agents (random, strategic, coordinated)
- Measure: world model accuracy with/without trust module
- Find breaking point: how many adversarial agents before system fails?
- **MILESTONE: "System maintains 90% accuracy with 20% adversarial agents"**

---

### Phase 8: Ablation Study (Weeks 19-20)

#### Task 8.1 — Ablation Framework [Developer A]
- `ablation.py`: load base config, systematically disable each component
- Run full evaluation suite for each ablation
- Structured output: JSON results per ablation

#### Task 8.2 — Run & Analyze [Both]
- Run all ablations (8 configurations)
- Statistical analysis: which component contributes most?
- Interaction effects: does removing X affect the value of Y?
- Create ablation results table and bar chart
- **MILESTONE: "Component X contributes Y% of total improvement"**

---

### Phase 8.5: Realistic V2X Networking Layer (Weeks 20-21) [Networking Specialization]

**Rationale.** Phases 1-8 use a fast *statistical* channel model (latency + jitter +
Bernoulli/Gilbert-Elliott loss) — the right choice for RL training because it is fast
enough to run millions of steps. This phase adds a higher-fidelity, protocol-aware V2X
channel that models the actual mechanics of vehicular radio communication. It is the
systems/networking counterpart to the ML work, and the primary contribution of
Developer A (networking background).

**Key idea.** Keep the fast model for training. Add a realistic model for *evaluation and
robustness*. Then show that a policy trained on the fast model still performs well under
realistic V2X constraints — a transfer/robustness result that strengthens the paper.

#### Task 8.5.1 — RF Propagation & Path Loss [Developer A]
- Implement `RealisticV2XChannel` alongside the existing `Channel` (same interface).
- Add distance-based path loss (log-distance model):
  `PL(d) = PL(d0) + 10 · n · log10(d/d0) + Xσ`
  where `n` is the path-loss exponent (≈2 free space, higher in urban), `Xσ` is
  shadow-fading noise.
- Map received signal power → packet success probability (SNR → PER curve).
- Result: loss rate now DEPENDS on distance between agents, not a fixed constant.

#### Task 8.5.2 — Medium Access Control (CSMA/CA) [Developer A]
- Implement carrier-sense multiple access with collision avoidance.
- When multiple agents transmit in the same slot within range → collision → both lost.
- Model backoff and contention windows.
- Half-duplex constraint: an agent cannot transmit and receive simultaneously.
- Result: throughput degrades as agent density increases (the "broadcast storm" problem).

#### Task 8.5.3 — C-V2X Mode 4 Sidelink (optional, advanced) [Developer A]
- Model 3GPP PC5 sidelink resource allocation.
- Semi-Persistent Scheduling (SPS): agents reserve time-frequency resources.
- Sensing-based resource selection and reselection.
- Compare against 802.11p/DSRC (CSMA/CA) behavior.

#### Task 8.5.4 — Decentralized Congestion Control (DCC) [Developer A]
- Implement ETSI DCC: as channel busy-ratio rises, agents must throttle their
  transmission rate.
- This couples the networking layer to the communication policy — the policy now
  operates under a dynamically shrinking bandwidth budget during congestion.
- Result: intelligent prioritization matters MORE under congestion (the core thesis).

#### Task 8.5.5 — Message Fragmentation & Aggregation [Developer A]
- Model MTU limits: large payloads split into multiple packets.
- Each packet independently subject to loss → partial message delivery.
- Reassembly logic; drop message if any packet is lost (or support partial decode).

#### Task 8.5.6 — Robustness / Transfer Evaluation [Both]
- Take the RL policy trained on the fast statistical channel (Phase 5).
- Evaluate it on the `RealisticV2XChannel` WITHOUT retraining.
- Compare: hand-designed vs. learned policy under realistic V2X.
- Sweep agent density (3, 5, 10, 20 vehicles) to show contention effects.
- **MILESTONE: "Policy trained on the simplified channel retains X% of its
  advantage under realistic C-V2X constraints, and its benefit GROWS with agent
  density due to congestion."**

**Optional — NS-3 cross-validation:** For one or two scenarios, cross-check the custom
`RealisticV2XChannel` against NS-3's V2X module (full packet-level simulation) to
validate that the custom model produces realistic loss/latency distributions. NS-3 is too
slow for training but valuable as ground truth for the channel model.

**New files this phase adds:**
```
cpp/core/communication/realistic_channel.h / .cc   # Protocol-aware channel
cpp/core/communication/propagation.h / .cc         # Path loss / RF model
cpp/core/communication/mac.h / .cc                  # CSMA/CA, contention
cpp/core/communication/congestion_control.h / .cc   # DCC
python/omnicopilot/config/defaults/communication/realistic_cv2x.yaml
python/omnicopilot/config/defaults/communication/realistic_dsrc.yaml
```

**Why this is a differentiator.** Most cooperative-perception papers use a simplified
channel (like our Phase 1-8 model) and hand-wave the networking. By modeling actual V2X
mechanics — path loss, contention, congestion control — this project demonstrates
genuine networking-engineering depth and produces a stronger, more defensible robustness
story. This is Developer A's systems specialization made concrete.

---

### Phase 9: Self-Improvement & LLM Layer (Weeks 21-22)

#### Task 9.1 — Learning Curves & Adaptation [Developer B]
- Communication policy improvement over training
- Trust calibration accuracy over interaction count
- Contextual reliability emergence
- Spatial prior development
- Plot all curves — show the system gets smarter

#### Task 9.2 — LLM Investigation Agent [Developer A]
- Implement with LangGraph:
  - Input: world model anomaly (multiple unusual observations)
  - Tools: query world model, retrieve from RAG, request observation
  - Output: structured `InvestigationResult` (hypothesis, confidence, recommended
    actions, explanation, sources consulted)
- Connect to vLLM serving local Llama/Mistral (or Qwen 2.5) — or Gemini/DeepSeek API
- Triggered ONLY for complex/unusual situations — never in the real-time 10Hz loop

#### Task 9.2b — LLM Result Feedback Loop [Developer A]

The LLM's output must not be a dead end. Its result is consumed in three ways, in
increasing order of coupling. **Governing principle: the LLM's conclusion is treated as
evidence with a confidence, NEVER as a command.** It re-enters the same probabilistic
machinery that handles sensor observations, so a confident/corroborated conclusion has
influence while a shaky guess is down-weighted, and the system degrades gracefully if the
LLM is wrong.

**9.2b-i — Output as human-facing explanation (SAFE, do first)**
- `InvestigationResult.explanation` and `hypothesis` surface to:
  - The demo UI (Phase 10) as a reasoning narrative
  - Structured logs (structlog) for post-hoc analysis
- No automatic effect on driving or fusion. Advisory only.

**9.2b-ii — Structured output fed back as a WorldEvent (POWERFUL, do next)**
- Convert the LLM's structured conclusion into a `WorldEvent`
  (the proto already defines `WorldEvent` with `HAZARD_DETECTED` and a
  `reporting_agent` field — the design anticipated this).
  ```
  LLM: "black-ice hazard at (X,Y), confidence 0.87"
    → WorldEvent{ type=HAZARD_DETECTED, location=(X,Y),
                  confidence=0.87, reporting_agent="llm_investigation" }
    → injected into WorldModel via pybind11 (Python → C++)
  ```
- The injected event is confidence-weighted (NOT ground truth). It then influences:
  - **Prioritizer**: hazard events near approaching vehicles become high-priority
    to communicate.
  - **Active acquisition**: `SourceSelector` asks agents near (X,Y) to corroborate
    the LLM's hypothesis.
- The LLM effectively becomes "just another agent" reporting events — subject to the
  same trust/confidence handling as any unreliable source.

**9.2b-iii — Conclusions as a learning signal (STRETCH, only if time)**
- Trust nudges: if the LLM concludes an agent's reports were consistent with a
  verified hazard, nudge that agent's trust score.
- Spatial priors: if the LLM concludes "this intersection is hazard-prone," store a
  spatial prior that lowers the uncertainty threshold for alerts in that area
  (ties into Task 9.1 spatial prior development).
- **Explicitly OUT OF SCOPE:** using the LLM as an RL reward/critic signal (RLAIF).
  This is a research rabbit hole and a safety risk — do not attempt within this project.

- **MILESTONE: "The LLM investigation of a multi-signal anomaly produces a hazard
  event that the system then prioritizes for communication and corroborates via
  active acquisition — demonstrating a closed reasoning loop where high-level
  inference influences low-level behavior, without ever bypassing the confidence-based
  safety machinery."**

#### Task 9.3 — RAG Knowledge Base [Developer B]
- Ingest: cooperative perception papers, V2X standards, safety guidelines
- Embed with BGE-large
- Store in Qdrant
- Retrieval + reranking pipeline
- Integrate with investigation agent

#### Task 9.4 — Multi-Agent LLM Investigation Team (STRETCH) [Developer A]

**Upgrade path, not a rewrite.** Task 9.2 builds a SINGLE LLM investigation agent. This
task splits that single agent into a *team* of specialized LLM agents that collaborate on
complex anomalies. Because the single and multi-agent versions share the SAME input
(world-model anomaly) and SAME output (`InvestigationResult`), this is a drop-in internal
upgrade — nothing else in the system changes.

**Two levels of multi-agent intelligence (the framing).** The project then demonstrates
multi-agent AI at two distinct layers:
- **Physical layer (real-time):** cooperating vehicles + the MARL communication policy
  (Phases 3-5) — fast, deterministic multi-agent systems.
- **Reasoning layer (deliberative):** a team of LLM agents investigating anomalies
  (this task) — slow, language-based multi-agent AI.
Fast multi-agent systems handle perception; deliberative multi-agent LLMs handle
high-level reasoning. Two paradigms, each used where it fits.

**Agent roles (orchestrated via LangGraph — already in Developer A's stack):**
```
Trigger: world model surfaces a complex multi-signal anomaly
    ↓
  Evidence Agent      → "What observations support / contradict a hazard?"
    ↓
  Historical Agent    → RAG query: "Have we seen this pattern before?"
    ↓
  Hypothesis Agent    → "Most likely explanation: black ice at (X,Y)"
    ↓
  Verification Agent  → "Is the hypothesis consistent with ALL evidence?
                         Flag contradictions."
    ↓
  Risk Agent          → "Severity + which/when vehicles are affected"
    ↓
  Decision Agent      → structured InvestigationResult
    ↓
  → feeds back as a WorldEvent (Task 9.2b), confidence-weighted, advisory only
  → MAY recommend active-acquisition targets ("query agent B about (X,Y)")
    which the Phase 6 acquisition mechanism then executes (LLM recommends,
    fast C++ executes — the LLM never runs in the acquisition hot path)
```

**Hard constraints (non-negotiable):**
- Lives ONLY in the non-real-time reasoning layer. NEVER in the 10Hz perception /
  fusion / communication loop. LLM latency (100ms–1s) makes real-time use impossible
  and unsafe.
- Output is always confidence-weighted evidence (a `WorldEvent`), never a command.
- Triggered rarely — only on complex anomalies simple rules cannot resolve.

**Why marked STRETCH (do only if core is done):**
- It does NOT strengthen the project's core thesis ("intelligent communication beats
  naive broadcast"). It is an additive capability, not a load-bearing one.
- It risks scope creep — the #1 project risk. Build the single agent (9.2) first; only
  split into a team if Phases 1-9 core work is complete and time remains.
- Evaluation overhead: proving the multi-agent team beats a single agent is itself an
  extra experiment.

**Where multi-agent LLMs must NOT go (explicitly wrong):**
- Perception (Phase 2), fusion/world-model (Phase 3), communication policy (Phases 4-5):
  all real-time — LLM agents here would break latency and safety. This is precisely the
  "AI-wrapper anti-pattern" the project deliberately avoids.
- Active acquisition (Phase 6): the routine "which agent to query" decision stays fast
  C++ math. The LLM team may only *recommend* targets from the Phase 9 layer; it does not
  run inside Phase 6.

**Technical significance:** demonstrates agentic AI (multi-agent LLM orchestration, tool
use, RAG-grounded reasoning) cleanly separated from safety-critical real-time control —
a sophisticated, defensible architecture keeping deliberative reasoning out of the
deterministic loop.

- **MILESTONE (if attempted): "A team of specialized LLM agents collaboratively
  investigates a multi-signal anomaly, produces a grounded hypothesis with verification,
  and feeds a confidence-weighted hazard event back into the world model — measurably
  more accurate or better-explained than the single-agent investigation baseline."**

---

### Phase 10: Demo, Write-up & Release (Weeks 23-26)

#### Task 10.1 — Interactive Demo [Developer A]
- Build with Rerun.io (or Plotly Dash):
  - 3D bird's-eye scenario view
  - Per-agent observation overlay
  - Communication flow visualization (who sent what to whom)
  - World model state (entities, confidence, predictions)
  - Network condition controls (sliders)
  - Toggle: cooperation on/off, learned vs. hand-designed
  - Live metrics panel

#### Task 10.2 — Demo Video & Presentation [Both]
- 5-minute video: problem → approach → demo → results
- Technical presentation (15-20 slides)
- Record narrated walkthrough

#### Task 10.3 — Technical Paper [Both]
- 8-10 pages, conference format (IEEE or NeurIPS workshop style)
- Sections: introduction, related work, method, experiments, results, ablation, conclusion
- Target: arxiv preprint + submit to CoRL / ICRA / NeurIPS MARL workshop

#### Task 10.4 — Open Source Release [Both]
- Code cleanup, full docstrings (Google style)
- API documentation (Sphinx + Doxygen)
- Comprehensive README
- Setup guide (tested on fresh machine)
- Example configs for reproducing all experiments
- Release on GitHub

---

## Developer A vs. Developer B — Final Summary

### Developer A (You)

**Languages:** C++17, Python, Protobuf
**Frameworks:** CMake, pybind11, gRPC, Gymnasium, LangGraph
**Domains:** World model, communication engine, network simulation, **realistic V2X networking layer (path loss, CSMA/CA, C-V2X sidelink, congestion control)**, RL environment, trust, active acquisition protocol, observability, demo UI
**Tools:** Docker, CI/CD, Prometheus/Grafana, OpenTelemetry, Rerun.io, **NS-3 (optional, for channel-model validation)**
**Specialization:** The realistic V2X networking layer (Phase 8.5) is Developer A's signature contribution — turning a networking background into a concrete, defensible systems result.

### Developer B (Your wife)

**Languages:** Python, Protobuf (consumer)
**Frameworks:** PyTorch, PyTorch Lightning, OpenPCDet/OpenCOOD, CleanRL, sentence-transformers, Qdrant
**Domains:** Perception models, fusion algorithms, uncertainty estimation, RL policy training, reward design, emergent analysis, RAG, evaluation
**Tools:** W&B, Jupyter, Hydra, DVC, Hypothesis

---

## Quality Standards (Non-Negotiable)

| Standard | Enforcement |
|---|---|
| All Python code passes `mypy --strict` | CI blocks merge on type errors |
| All Python code passes `ruff` lint | Pre-commit hook |
| All C++ code passes `clang-tidy` | CI blocks on warnings |
| All C++ code formatted by `clang-format` | Pre-commit hook |
| Unit test coverage > 80% | Measured in CI |
| Every experiment reproducible from config | Hydra config + DVC data + W&B run ID |
| Every PR reviewed by the other developer | GitHub branch protection |
| Proto files are the source of truth for interfaces | Generated code, never hand-edited |
| No `Any` type in Python | mypy strict catches this |
| No raw pointers in C++ | clang-tidy check |

---

## Hardware Requirements

### Development (each developer)
- **GPU:** NVIDIA RTX 3080+ (12GB+ VRAM)
- **RAM:** 32GB minimum, 64GB recommended
- **Storage:** 500GB SSD (datasets are large)
- **OS:** Ubuntu 22.04 or later

### Training (shared or cloud)
- **GPU:** NVIDIA A100 40GB or RTX 4090 24GB
- **For:** RL policy training, perception model fine-tuning
- **Cloud options:** Lambda Labs ($1.10/hr A100), Vast.ai, RunPod

### CARLA (when needed)
- **GPU:** RTX 3080+ (CARLA is GPU-hungry)
- **RAM:** 32GB+
- **Note:** Run on dedicated machine or powerful cloud instance

---

## What to Do This Week

**Developer A:**
1. `git init` the repository
2. Set up CMake skeleton (compiles empty main, links gtest)
3. Write `proto/observation.proto` and `proto/entity.proto`
4. Generate C++ and Python from protos (verify builds)
5. Set up `pyproject.toml` with uv, install core deps
6. Configure pre-commit hooks
7. Push with CI working (C++ builds, Python lints)

**Developer B:**
1. Download OPV2V dataset (or subset)
2. Write `python/omnicopilot/data/opv2v.py` — typed dataset loader
3. Write `01_data_exploration.ipynb` — understand the data format
4. Set up W&B project, log dataset statistics
5. Research OpenCOOD: what models are available, what's the API?

**Both:**
- Agree on the proto definitions (this is your contract)
- Set up shared development environment (Docker)
- Write `docs/setup.md`
