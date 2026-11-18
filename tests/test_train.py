"""Training state restoration on a small transductive graph."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from gist import datasets, train


def test_best_epoch_weights_are_restored(monkeypatch: pytest.MonkeyPatch) -> None:

    nodes = torch.arange(8)
    data = Data(
        x=torch.randn(8, 4),
        edge_index=torch.stack((nodes, (nodes + 1) % 8)),
        y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        graph_embed=torch.randn(8, 8),
        train_mask=nodes < 4,
        val_mask=(nodes >= 4) & (nodes < 6),
        test_mask=nodes >= 6,
    )
    benchmark = datasets.Benchmark(
        name="synthetic",
        task="multiclass",
        splits={"train": [data], "val": [data], "test": [data]},
        num_features=4,
        num_targets=2,
        transductive=True,
    )
    models: list[torch.nn.Module] = []
    snapshots: list[dict[str, torch.Tensor]] = []
    original_model = train.GIST
    original_step = torch.optim.AdamW.step

    def build_model(*args: Any, **kwargs: Any) -> torch.nn.Module:
        model = original_model(*args, **kwargs)
        models.append(model)
        return model

    def capture_step(optimizer: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        result = original_step(optimizer, *args, **kwargs)
        snapshots.append({key: value.clone() for key, value in models[-1].state_dict().items()})
        return result

    scores = iter((0.9, 0.6, 0.3, 0.2))
    monkeypatch.setattr(train, "GIST", build_model)
    monkeypatch.setattr(torch.optim.AdamW, "step", capture_step)
    monkeypatch.setattr(train, "compute_metrics", lambda *_args: {"accuracy": next(scores)})
    monkeypatch.setattr(train, "_load", lambda *_args: benchmark)
    result = train.run(
        {
            "dataset": "synthetic",
            "seeds": [0],
            "embedding": {"embed_dim": 8, "deg": 2},
            "model": {"hidden_dim": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0},
            "train": {"epochs": 3, "eval_freq": 1, "lr": 0.01, "compile": False},
        }
    )

    assert len(snapshots) == 3
    assert any(not torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
    assert all(torch.equal(value, snapshots[0][key]) for key, value in models[0].state_dict().items())
    assert result["per_seed"][0]["selected_epoch"] == 1
    assert result["per_seed"][0]["val"] == {"accuracy": 0.9}
    json.dumps(result, allow_nan=False)


def test_refit_view_adds_validation() -> None:
    nodes = torch.arange(8)
    data = Data(train_mask=nodes < 4, val_mask=(nodes >= 4) & (nodes < 6))
    transductive = datasets.Benchmark("synthetic", "multiclass", {"train": [data], "val": [data]}, 4, 2, True)
    assert transductive.training_view(include_val=True)[0].train_mask.sum() == 6
    assert data.train_mask.sum() == 4
    graphs = [Data(), Data(), Data()]
    inductive = datasets.Benchmark("synthetic", "regression", {"train": graphs[:2], "val": graphs[2:]}, 3, 1, False)
    assert len(inductive.training_view(include_val=True)) == 3


def _graphs() -> list[Data]:
    generator = torch.Generator().manual_seed(0)
    graphs = []
    for num_nodes in (5, 7, 6):
        nodes = torch.arange(num_nodes)
        graphs.append(
            Data(
                x=torch.randn(num_nodes, 3, generator=generator),
                edge_index=torch.stack((nodes, (nodes + 1) % num_nodes)),
                y=torch.randn(num_nodes, generator=generator),
                graph_embed=torch.randn(num_nodes, 8, generator=generator),
            )
        )
    return graphs


def _sharded_evaluation(rank: int, world_size: int, port: int, path: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    benchmark = datasets.Benchmark("synthetic", "regression", {"test": _graphs()}, 3, 1, False, -2.0, 3.0)
    torch.manual_seed(0)
    model = train.GIST(input_dim=3, output_dim=1, hidden_dim=8, num_layers=1, num_heads=2)
    metrics = train._evaluate(model, benchmark, "test", torch.device("cpu"), 0)
    if rank == 0:
        torch.save(metrics, path)
    torch.distributed.destroy_process_group()


def test_evaluation_split_across_processes_matches_one_process(tmp_path) -> None:
    single, sharded = str(tmp_path / "single.pt"), str(tmp_path / "sharded.pt")
    torch.multiprocessing.spawn(_sharded_evaluation, args=(1, 29511, single), nprocs=1)
    torch.multiprocessing.spawn(_sharded_evaluation, args=(2, 29512, sharded), nprocs=2)
    expected, actual = torch.load(single), torch.load(sharded)
    # Nodes sum in a different order across processes, which moves float32 metrics by a few units in the last place.
    assert actual == pytest.approx(expected, rel=1e-5)
