"""Model export utilities — PyTorch to ONNX for production inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: dict[str, torch.Tensor],
    output_path: Path,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model in eval mode.
        dummy_input: Example input tensors for tracing.
        output_path: Where to save the .onnx file.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axis configuration for variable batch sizes.

    Returns:
        Path to the exported ONNX model.
    """
    # TODO: Implement export with proper input/output naming.
    raise NotImplementedError
