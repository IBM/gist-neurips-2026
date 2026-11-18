"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from gist.config import load_config


def test_precedence_and_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """dataset: cora
seeds: [3, 5]
embedding:
  embed_dim: 128
model:
  hidden_dim: 64
  dropout: 0.1
train:
  lr: 0.01
  weight_decay: 1e-5
  batch_size: 4
""",
        encoding="utf-8",
    )
    config = load_config(path, ["train.lr=2e-2", "model.num_layers=3", "embedding.embed_dim=256"])
    assert config["train"]["lr"] == 0.02
    assert config["train"]["weight_decay"] == 1e-5
    assert config["train"]["batch_size"] == 4
    assert config["train"]["grad_clip"] == 0.1
    assert config["seeds"] == [3, 5]
    assert config["embedding"]["embed_dim"] == 256
    assert config["model"]["num_layers"] == 3
    assert config["model"]["dropout"] == 0.1

    path.write_text("dataset: cora\ntrain:\n  lrr: 0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.lrr"):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    ["model.branch_dim=3", "model.branch_dim=0", "model.spectral_threshold=null", "model.spectral_threshold=.nan"],
)
def test_invalid_model_values_are_rejected(tmp_path: Path, override: str) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("dataset: cora\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, [override])
