# Copyright 2024 OmniCopilot Authors
# SPDX-License-Identifier: Apache-2.0
#
# ML Training image: Full CUDA environment with PyTorch, RL, and perception models.
# Based on NVIDIA's PyTorch container for optimal GPU performance.

FROM nvcr.io/nvidia/pytorch:24.02-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /workspace

# Copy project files
COPY pyproject.toml ./
COPY python/ python/

# Install all Python dependencies (including ML, RL, notebooks)
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install -e ".[all]"

# Copy remaining project files
COPY proto/ proto/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY configs/ configs/ 2>/dev/null || true

ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/.venv"

# Generate Python protobuf files
RUN python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=python/omnicopilot \
    --pyi_out=python/omnicopilot \
    proto/*.proto || true

# Default: start Jupyter for interactive research
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
