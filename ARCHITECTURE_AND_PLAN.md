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
  - Output: structured hypothesis with confidence and evidence
- Connect to vLLM serving local Llama/Mistral

#### Task 9.3 — RAG Knowledge Base [Developer B]
- Ingest: cooperative perception papers, V2X standards, safety guidelines
- Embed with BGE-large
- Store in Qdrant
- Retrieval + reranking pipeline
- Integrate with investigation agent

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
**Domains:** World model, communication engine, network simulation, RL environment, trust, active acquisition protocol, observability, demo UI
**Tools:** Docker, CI/CD, Prometheus/Grafana, OpenTelemetry, Rerun.io

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
