# Copyright 2024 OmniCopilot Authors
# SPDX-License-Identifier: Apache-2.0
#
# OmniCopilot — Development task runner
# Run `make help` to see available targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─── Variables ───────────────────────────────────────────────────────────────

PYTHON := python3
UV := uv
CMAKE := cmake
BUILD_DIR := build
PROTO_DIR := proto
PROTO_OUT_CPP := $(BUILD_DIR)/generated/proto
PROTO_OUT_PY := python/omnicopilot/generated

# ─── Help ────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo "OmniCopilot — Cooperative Multimodal AI"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all Python dependencies (dev + all extras)
	$(UV) sync --extra all
	$(UV) run pre-commit install

.PHONY: install-dev
install-dev: ## Install Python dev dependencies only
	$(UV) sync --extra dev
	$(UV) run pre-commit install

.PHONY: setup-cpp
setup-cpp: ## Configure and build C++ components
	$(CMAKE) -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DOMNICOPILOT_BUILD_TESTS=ON \
		-DOMNICOPILOT_BUILD_BINDINGS=ON
	$(CMAKE) --build $(BUILD_DIR) --parallel $$(nproc)

# ─── Proto Generation ────────────────────────────────────────────────────────

.PHONY: proto
proto: proto-cpp proto-python ## Generate all protobuf code

.PHONY: proto-cpp
proto-cpp: ## Generate C++ protobuf sources (via CMake build)
	@echo "C++ protos generated during cmake build"
	@test -d $(BUILD_DIR) || $(MAKE) setup-cpp

.PHONY: proto-python
proto-python: ## Generate Python protobuf sources
	@mkdir -p $(PROTO_OUT_PY)
	$(PYTHON) -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(PROTO_OUT_PY) \
		--pyi_out=$(PROTO_OUT_PY) \
		--grpc_python_out=$(PROTO_OUT_PY) \
		$(PROTO_DIR)/*.proto
	@touch $(PROTO_OUT_PY)/__init__.py

# ─── Build ───────────────────────────────────────────────────────────────────

.PHONY: build
build: setup-cpp proto-python ## Build everything (C++ + proto)

.PHONY: build-release
build-release: ## Build C++ in Release mode
	$(CMAKE) -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DOMNICOPILOT_BUILD_TESTS=OFF \
		-DOMNICOPILOT_BUILD_BINDINGS=ON
	$(CMAKE) --build $(BUILD_DIR) --parallel $$(nproc)

# ─── Test ────────────────────────────────────────────────────────────────────

.PHONY: test
test: test-cpp test-python ## Run all tests

.PHONY: test-cpp
test-cpp: setup-cpp ## Run C++ unit tests
	cd $(BUILD_DIR) && ctest --output-on-failure --parallel $$(nproc)

.PHONY: test-python
test-python: ## Run Python unit tests
	$(UV) run pytest tests/unit/ -x --timeout=120

.PHONY: test-integration
test-integration: ## Run integration tests
	$(UV) run pytest tests/integration/ -x --timeout=300

.PHONY: test-property
test-property: ## Run property-based tests
	$(UV) run pytest tests/property/ -x --timeout=300

.PHONY: test-all
test-all: test-cpp ## Run all tests with coverage
	$(UV) run pytest tests/ \
		--cov=python/omnicopilot \
		--cov-report=html:htmlcov \
		--cov-report=term-missing \
		--timeout=300

# ─── Lint & Format ───────────────────────────────────────────────────────────

.PHONY: lint
lint: lint-python lint-cpp lint-proto ## Run all linters

.PHONY: lint-python
lint-python: ## Lint Python code
	$(UV) run ruff check python/ tests/ scripts/
	$(UV) run mypy python/omnicopilot/

.PHONY: lint-cpp
lint-cpp: ## Check C++ formatting
	@find cpp/ -name '*.h' -o -name '*.cc' | \
		xargs clang-format --style=file --dry-run --Werror

.PHONY: lint-proto
lint-proto: ## Lint protobuf files
	buf lint

.PHONY: format
format: ## Format all code (Python + C++)
	$(UV) run ruff format python/ tests/ scripts/
	$(UV) run ruff check --fix python/ tests/ scripts/
	@find cpp/ -name '*.h' -o -name '*.cc' | xargs clang-format --style=file -i

# ─── Experiments ─────────────────────────────────────────────────────────────

.PHONY: baseline
baseline: ## Run single-agent baseline experiment
	$(UV) run python scripts/run_experiment.py experiment=single_agent

.PHONY: cooperative
cooperative: ## Run cooperative perception experiment
	$(UV) run python scripts/run_experiment.py experiment=cooperative_naive

.PHONY: communication
communication: ## Run intelligent communication experiment
	$(UV) run python scripts/run_experiment.py experiment=cooperative_intelligent

.PHONY: train-policy
train-policy: ## Train RL communication policy
	$(UV) run python scripts/train_policy.py

.PHONY: ablation
ablation: ## Run full ablation suite
	$(UV) run python scripts/run_ablation_suite.py

.PHONY: evaluate
evaluate: ## Run full evaluation pipeline
	$(UV) run python scripts/run_experiment.py --multirun \
		experiment=single_agent,cooperative_naive,cooperative_intelligent,cooperative_learned

# ─── Demo ────────────────────────────────────────────────────────────────────

.PHONY: demo
demo: ## Launch interactive demo application
	$(UV) run python demo/app.py

.PHONY: demo-video
demo-video: ## Generate demo scenario video
	$(UV) run python scripts/generate_report.py --video

# ─── Data ────────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: ## Download OPV2V dataset (subset)
	@echo "Downloading OPV2V dataset..."
	@mkdir -p data/raw/opv2v
	@echo "TODO: Add download script for OPV2V"

# ─── Docker ──────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

.PHONY: docker-runtime
docker-runtime: ## Start runtime container
	docker compose -f docker/docker-compose.yml up runtime

.PHONY: docker-ml
docker-ml: ## Start ML training container (GPU)
	docker compose -f docker/docker-compose.yml up ml

.PHONY: docker-observability
docker-observability: ## Start observability stack (Prometheus + Grafana)
	docker compose -f docker/docker-compose.yml --profile observability up

.PHONY: docker-down
docker-down: ## Stop all containers
	docker compose -f docker/docker-compose.yml --profile observability down

# ─── Cleanup ─────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BUILD_DIR)
	rm -rf htmlcov .coverage coverage.xml
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## Remove build artifacts + data + models
	@echo "WARNING: This will remove downloaded data and trained models."
	@echo "Press Ctrl+C to cancel, or wait 5 seconds..."
	@sleep 5
	rm -rf data/processed data/models data/results
