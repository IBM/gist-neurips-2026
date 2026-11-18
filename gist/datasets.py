"""Benchmark datasets and their training and evaluation splits."""

import json
import os
from collections.abc import Callable, Sized
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.datasets import PPI, Amazon, Planetoid
from torch_geometric.utils import to_undirected

from gist.data import EmbeddedGraphDataset

SUPPORTED = (
    "cora",
    "citeseer",
    "pubmed",
    "photo",
    "ogbn-arxiv",
    "ppi",
    "airfrans",
    "shapenet-car",
    "drivaernet",
)
Task = Literal["multiclass", "multilabel", "regression"]


class _SingleGraphDataset(Dataset[Data]):
    """Hold one graph while preserving its node masks."""

    def __init__(self, graph: Data) -> None:
        self.graph = graph

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError(index)
        return self.graph


class _TransformDataset(Dataset[Data]):
    """Apply a graph transform when an item is accessed."""

    def __init__(self, dataset: Dataset[Data], transform: Callable[[Data], Data]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Data:
        return self.transform(self.dataset[index])


@dataclass
class Benchmark:
    """A benchmark with graph splits and target metadata.

    Args:
        name: Dataset name.
        task: Prediction task.
        splits: Graph datasets indexed by split name.
        num_features: Number of input features per node.
        num_targets: Number of classes, labels, or regression targets.
        transductive: Whether splits use masks on one shared graph.
        target_mean: Training target mean for regression.
        target_std: Training target standard deviation for regression.
        seeded_split: Whether the split is drawn at random from ``split_seed``.
    """

    name: str
    task: Task
    splits: dict[str, Dataset[Data]]
    num_features: int
    num_targets: int
    transductive: bool
    target_mean: float = 0.0
    target_std: float = 1.0
    seeded_split: bool = False

    def training_view(self, include_val: bool) -> Dataset[Data]:
        """Return graphs for training, optionally including validation data.

        Args:
            include_val: Include validation graphs or nodes.

        Returns:
            Dataset with the training graph view.
        """
        train = self.splits["train"]
        if not include_val:
            return train
        if self.transductive:
            graph = train[0].clone()
            graph.train_mask = graph.train_mask | graph.val_mask
            return _SingleGraphDataset(graph)
        return ConcatDataset([train, self.splits["val"]])


def _standardize(
    graph: Data,
    mean: float,
    std: float,
    input_stats: tuple[Tensor, Tensor] | None = None,
    inlet_columns: tuple[int, int] | None = None,
) -> Data:
    """Standardize regression targets, and node features when their statistics are given.

    With ``inlet_columns``, the inlet velocity in those raw feature columns sets a per-node output scale and offset that
    map a predicted pressure coefficient, pressure over 0.5 * |inlet velocity|^2, to standardized pressure.
    """
    graph.y = (graph.y.float() - mean).div_(std)
    if inlet_columns is not None:
        dynamic_pressure = graph.x[:, list(inlet_columns)].float().square().sum(dim=1, keepdim=True).mul_(0.5)
        graph.output_scale = dynamic_pressure / std
        graph.output_offset = torch.full_like(dynamic_pressure, -mean / std)
    if input_stats is not None:
        input_mean, input_std = input_stats
        graph.x = (graph.x - input_mean).div_(input_std)
    return graph


def _cached_stats(path: Path, dataset: Dataset[Data], key: str) -> tuple[Tensor, Tensor]:
    """Return the per-column node mean and standard deviation of a dataset, computed once and cached as JSON."""
    if not path.is_file():
        from gist.mesh import node_stats

        # Written aside and renamed, so concurrent processes never read a partial file.
        partial_path = path.with_name(f"{path.name}.{os.getpid()}")
        partial_path.write_text(json.dumps([values.squeeze().tolist() for values in node_stats(dataset, key)]))
        os.replace(partial_path, path)
    mean, std = torch.tensor(json.loads(path.read_text()), dtype=torch.float32)
    return mean, std


def _set_node_masks(graph: Data, indices: dict[str, Tensor]) -> None:
    """Attach node masks from index tensors."""
    for split, selected in indices.items():
        mask = torch.zeros(int(graph.num_nodes), dtype=torch.bool)
        mask[selected] = True
        setattr(graph, f"{split}_mask", mask)


def load_dataset(
    name: str,
    embed_dim: int,
    deg: int,
    root: str = "data",
    split_seed: int = 0,
    generate_embeddings: bool = False,
) -> Benchmark:
    """Load a benchmark and attach its cached graph embeddings.

    Args:
        name: Benchmark name.
        embed_dim: Number of embedding coordinates.
        deg: Number of diffusion powers.
        root: Parent directory for benchmark data.
        split_seed: Seed for random node splits (Photo).
        generate_embeddings: Generate missing embedding cache files.

    Returns:
        Benchmark with full resolution evaluation splits.
    """
    name = name.lower()
    if name not in SUPPORTED:
        message = f"Unknown dataset {name!r}. Supported datasets: {', '.join(SUPPORTED)}"
        raise ValueError(message)

    root_path = Path(root).expanduser()
    raw_splits: dict[str, Dataset[Data]]
    num_targets: int
    target_mean = 0.0
    target_std = 1.0
    target_transform: Callable[[Data], Data] | None = None
    if name in {"cora", "citeseer", "pubmed", "photo", "ogbn-arxiv"}:
        display_names = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "photo": "Photo"}
        data_root = root_path / display_names.get(name, name)
        if name in {"cora", "citeseer", "pubmed"}:
            source = Planetoid(root=str(data_root), name=display_names[name], split="public")
            raw: Dataset[Data] = source
        elif name == "photo":
            source = Amazon(root=str(data_root), name="Photo")
            graph = source[0]
            permutation = torch.randperm(int(graph.num_nodes), generator=torch.Generator().manual_seed(split_seed))
            train_end = int(0.6 * int(graph.num_nodes))
            val_end = train_end + int(0.2 * int(graph.num_nodes))
            _set_node_masks(
                graph,
                {
                    "train": permutation[:train_end],
                    "val": permutation[train_end:val_end],
                    "test": permutation[val_end:],
                },
            )
            raw = _SingleGraphDataset(graph)
        else:
            try:
                from ogb.nodeproppred import PygNodePropPredDataset
                from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
                from torch_geometric.data.storage import GlobalStorage
            except ImportError as exc:
                raise ImportError("ogbn-arxiv needs the ogb package: pip install 'gist[ogb]'") from exc
            with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                source = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
                split_indices = source.get_idx_split()
            graph = source[0]
            # Citation edges point from citing to cited paper; use them in both directions.
            graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
            _set_node_masks(
                graph,
                {
                    "train": torch.as_tensor(split_indices["train"]),
                    "val": torch.as_tensor(split_indices["valid"]),
                    "test": torch.as_tensor(split_indices["test"]),
                },
            )
            graph.y = graph.y.view(-1)
            raw = _SingleGraphDataset(graph)
        num_features = int(source.num_features)
        num_targets = int(source.num_classes)
        raw_splits = {"all": raw}
        task: Task = "multiclass"
        transductive = True
    elif name == "ppi":
        data_root = root_path / "PPI"
        raw_splits = {split: PPI(root=str(data_root), split=split) for split in ("train", "val", "test")}
        num_features = int(raw_splits["train"][0].x.shape[-1])
        num_targets = 121
        task = "multilabel"
        transductive = False
    else:
        data_root = root_path / name
        try:
            from gist.mesh import AIRFRANS_INLET_COLUMNS, DrivAerNetDataset, ShapeNetCarDataset, airfrans
        except ImportError as exc:
            raise ImportError("Mesh datasets need pyvista: pip install 'gist[mesh]'") from exc

        if name == "drivaernet":
            factory: Callable[..., Dataset[Data]] = DrivAerNetDataset
            split_names: tuple[str, ...] = ("train", "val", "test")
        elif name == "shapenet-car":
            factory = ShapeNetCarDataset
            split_names = ("train", "test")
        else:
            factory = airfrans
            split_names = ("train", "test")
        raw_splits = {split: factory(root=str(data_root), split=split) for split in split_names}
        if name in {"airfrans", "shapenet-car"}:
            # AirfRANS and ShapeNet-Car ship no validation split: the last tenth of the training samples is held out.
            train = raw_splits["train"]
            held_out = len(cast(Sized, train)) - len(cast(Sized, train)) // 10
            raw_splits["val"] = Subset(train, range(held_out, len(cast(Sized, train))))
            raw_splits["train"] = Subset(train, range(held_out))

        # Reading every training mesh is slow, so the statistics are computed once per data root.
        target_mean, target_std = (
            float(value) for value in _cached_stats(data_root / "target_stats.json", raw_splits["train"], "y")
        )
        target_std = max(target_std, 1e-12)
        input_stats: tuple[Tensor, Tensor] | None = None
        if name == "airfrans":
            # AirfRANS inputs mix inlet velocities of tens of m/s with unit normals, so they are standardized too.
            input_mean, input_std = _cached_stats(data_root / "input_stats.json", raw_splits["train"], "x")
            input_stats = input_mean, input_std.clamp_min(1e-12)
        num_features = int(raw_splits["train"][0].x.shape[-1])
        num_targets = 1
        task = "regression"
        transductive = False
        target_transform = partial(
            _standardize,
            mean=target_mean,
            std=target_std,
            input_stats=input_stats,
            inlet_columns=AIRFRANS_INLET_COLUMNS if name == "airfrans" else None,
        )

    embedded_splits: dict[str, Dataset[Data]] = {}
    for split, raw_dataset in raw_splits.items():
        embedded = EmbeddedGraphDataset(
            raw_dataset,
            name=name,
            root=data_root,
            split=split,
            embed_dim=embed_dim,
            deg=deg,
            generate_embeddings=generate_embeddings,
        )
        embedded_splits[split] = _TransformDataset(embedded, target_transform) if target_transform else embedded
    if transductive:
        embedded_splits = dict.fromkeys(("train", "val", "test"), embedded_splits["all"])

    return Benchmark(
        name=name,
        task=task,
        splits=embedded_splits,
        num_features=num_features,
        num_targets=num_targets,
        transductive=transductive,
        target_mean=target_mean,
        target_std=target_std,
        seeded_split=name == "photo",
    )
