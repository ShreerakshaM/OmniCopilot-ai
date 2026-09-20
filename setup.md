# Development Environment Setup

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 24.04 |
| GPU | NVIDIA RTX 3060 (12GB) | RTX 4090 (24GB) |
| RAM | 32 GB | 64 GB |
| Storage | 200 GB SSD | 500 GB NVMe |
| CUDA | 12.0+ | 12.4+ |
| Python | 3.11 | 3.11 |
| CMake | 3.22+ | 3.28+ |

---

## Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    g++-13 \
    clang-17 \
    clang-format-17 \
    clang-tidy-17 \
    protobuf-compiler \
    libprotobuf-dev \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    curl \
    docker.io \
    docker-compose-v2
```

---

## Step 2: Install uv (Python Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # Or restart shell.
```

---

## Step 3: Clone and Install

```bash
git clone https://github.com/omnicopilot/omnicopilot.git
cd omnicopilot

# Install all Python dependencies (creates .venv automatically)
make install

# Verify Python setup
uv run python -c "import omnicopilot; print(omnicopilot.__version__)"
```

---

## Step 4: Build C++ Core

```bash
make setup-cpp
```

This will:
1. Configure CMake with Ninja generator.
2. Fetch dependencies (gtest, pybind11, spdlog) via FetchContent.
3. Generate protobuf C++ sources.
4. Build the core library, simulation library, and Python bindings.
5. Run C++ unit tests.

### Troubleshooting C++ build

```bash
# If protobuf version mismatch:
sudo apt install --reinstall protobuf-compiler libprotobuf-dev

# If pybind11 not found:
pip install pybind11[global]

# For verbose build output:
cmake --build build --parallel $(nproc) --verbose
```

---

## Step 5: Generate Protobuf Python Code

```bash
make proto-python
```

---

## Step 6: Verify Everything Works

```bash
# Run all tests
make test

# Check linting passes
make lint

# Run a quick experiment (will fail with NotImplementedError until implemented)
# make baseline
```

---

## Step 7: Download Dataset (OPV2V)

```bash
# Download OPV2V (or a subset for development)
make download-data
```

OPV2V dataset: https://mobility-lab.seas.ucla.edu/opv2v/

For initial development, download only the test split (~10GB).

---

## Step 8: IDE Setup

### VS Code (recommended extensions)

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.mypy-type-checker",
        "charliermarsh.ruff",
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "zxh404.vscode-proto3",
        "eamodio.gitlens"
    ]
}
```

### PyCharm / CLion

- Mark `python/` as Sources Root.
- Set Python interpreter to `.venv/bin/python`.
- Set CMake build directory to `build/`.
- Enable mypy plugin.

---

## Step 9: Pre-commit Hooks

```bash
# Already installed by `make install`, but to manually set up:
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Run on all files (first time):
uv run pre-commit run --all-files
```

---

## Step 10: Docker (Optional)

For isolated, reproducible environments:

```bash
# Build all images
make docker-build

# Start ML training environment (with GPU)
make docker-ml

# Start observability stack (Prometheus + Grafana)
make docker-observability
```

Access:
- Jupyter Lab: http://localhost:8888
- Grafana: http://localhost:3000 (admin / omnicopilot)
- Prometheus: http://localhost:9090

---

## Step 11: Weights & Biases Setup

```bash
# Login to W&B (free account)
uv run wandb login

# Or set the API key as environment variable:
export WANDB_API_KEY=your_key_here
```

---

## Common Commands Reference

```bash
make help          # Show all available commands
make install       # Install Python deps
make setup-cpp     # Build C++ core
make proto         # Generate all protobuf code
make build         # Build everything
make test          # Run all tests (C++ + Python)
make lint          # Run all linters
make format        # Auto-format all code
make demo          # Launch interactive demo
make docker-build  # Build Docker images
make clean         # Remove build artifacts
```

---

## Troubleshooting

### CUDA not found

```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# If not installed, follow: https://developer.nvidia.com/cuda-downloads
```

### protoc version too old

```bash
# Need protobuf >= 3.21
protoc --version

# If too old, install from source or use:
# https://github.com/protocolbuffers/protobuf/releases
```

### Permission denied on Docker

```bash
sudo usermod -aG docker $USER
newgrp docker
```
