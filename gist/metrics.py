"""Losses and device-side metrics for benchmark tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from gist.datasets import Benchmark

HIGHER_IS_BETTER = {"accuracy": True, "micro_f1": True, "mse": False, "rel_l2": False}
DEFAULT_SELECTION_METRIC = {"multiclass": "accuracy", "multilabel": "micro_f1", "regression": "mse"}


def _is_better(metric: str, score: float, best_score: float | None) -> bool:
    return best_score is None or (score > best_score if HIGHER_IS_BETTER[metric] else score < best_score)


def task_loss(task: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the loss function for a benchmark task.

    Args:
        task: One of ``multiclass``, ``multilabel``, or ``regression``.

    Returns:
        A loss function accepting model outputs and targets.
    """
    if task == "multiclass":
        return F.cross_entropy
    if task == "multilabel":
        return lambda outputs, targets: F.binary_cross_entropy_with_logits(outputs, targets.float())
    if task == "regression":
        return lambda outputs, targets: F.mse_loss(outputs.reshape(-1), targets.float().reshape(-1))
    raise ValueError(task)


def compute_metrics(
    benchmark: Benchmark,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    graph_index: torch.Tensor,
) -> dict[str, float]:
    """Compute task metrics from evaluated nodes while keeping reductions on device.

    Args:
        benchmark: Benchmark task and target normalization statistics.
        outputs: Concatenated model outputs.
        targets: Concatenated targets.
        graph_index: Mesh index for each target, used by relative L2.

    Returns:
        Metric names and values as Python floats.
    """
    if outputs.shape[0] == 0:
        reason = "empty evaluation"
        raise ValueError(reason)
    if benchmark.task == "multiclass":
        accuracy = (outputs.argmax(dim=-1) == targets.reshape(-1)).float().mean()
        return {"accuracy": accuracy.item()}
    if benchmark.task == "multilabel":
        predicted = outputs > 0
        actual = targets.bool()
        true_positive = (predicted & actual).sum()
        positive_predictions = predicted.sum()
        positive_targets = actual.sum()
        micro_f1 = (2 * true_positive).float() / (positive_predictions + positive_targets).clamp_min(1)
        return {"micro_f1": micro_f1.item()}
    if benchmark.task != "regression":
        raise ValueError(benchmark.task)

    prediction = outputs.float().reshape(-1)
    target = targets.float().reshape(-1)
    error = prediction - target
    squared_error = error.square()
    mse = squared_error.mean()
    variance = (target - target.mean()).square().sum()
    r2 = 1 - squared_error.sum() / variance.clamp_min(1e-12)

    physical_error = error * benchmark.target_std
    physical_target = target * benchmark.target_std + benchmark.target_mean
    meshes, mesh_index = torch.unique(graph_index.reshape(-1), sorted=True, return_inverse=True)
    mesh_error = torch.zeros(meshes.numel(), dtype=prediction.dtype, device=prediction.device)
    mesh_target = torch.zeros_like(mesh_error)
    mesh_error.index_add_(0, mesh_index, physical_error.square())
    mesh_target.index_add_(0, mesh_index, physical_target.square())
    rel_l2 = (mesh_error.sqrt() / mesh_target.sqrt().clamp_min(1e-12)).mean()

    values = torch.stack((mse, rel_l2, r2)).tolist()
    return dict(zip(("mse", "rel_l2", "r2"), values, strict=True))
