"""Disk-backed datasets for surface meshes and airfoil simulations."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected

from gist.data import save_atomic
from gist.logs import log_progress

_MISSING_MESH_EXTRA = "Reading surface meshes requires the gist[mesh] extra (pyvista)"
_EMPTY_DATASET = "Cannot compute node statistics for an empty dataset"
# Bounding box of the DrivAerNet cars in metres, per axis.
_DRIVAERNET_MIN = torch.tensor([-1.16, -1.20, 0.0])
_DRIVAERNET_MAX = torch.tensor([4.21, 1.19, 1.77])
# Inlet velocity among AirfRANS node features, after the 2D position that _AirfRANSDataset.get prepends.
AIRFRANS_INLET_COLUMNS = (2, 3)


def _face_edges(faces: np.ndarray, triangulate: bool) -> Tensor:
    """Return both directions of each unique mesh face edge."""
    edges: set[tuple[int, int]] = set()
    offset = 0
    while offset < len(faces):
        width = int(faces[offset])
        vertices = [int(vertex) for vertex in faces[offset + 1 : offset + 1 + width]]
        offset += width + 1
        if width < 3:
            continue
        polygons = (
            [[vertices[0], vertices[j], vertices[j + 1]] for j in range(1, width - 1)] if triangulate else [vertices]
        )
        for polygon in polygons:
            for source, target in zip(polygon, polygon[1:] + polygon[:1], strict=True):
                if source != target:
                    edges.add((min(source, target), max(source, target)))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
    return torch.cat((forward, forward.flip(0)), dim=1)


def _mesh_data(path: Path, pressure_field: str, triangulate: bool) -> Data:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(_MISSING_MESH_EXTRA) from exc

    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        # ShapeNet-Car stores its quad surface as an unstructured grid.
        mesh = mesh.extract_surface()
    point_data = mesh.point_data
    normal_field = next((key for key in ("Normals", "normals", "PointNormals") if key in point_data), None)
    normals = (
        point_data[normal_field]
        if normal_field is not None
        else mesh.compute_normals(point_normals=True, cell_normals=False).point_data["Normals"]
    )
    pos = torch.as_tensor(np.asarray(mesh.points).copy(), dtype=torch.float32)
    normal_tensor = torch.as_tensor(np.asarray(normals).copy(), dtype=torch.float32)
    pressure = torch.as_tensor(np.asarray(point_data[pressure_field]).copy(), dtype=torch.float32).reshape(-1)
    edge_index = _face_edges(np.asarray(mesh.faces), triangulate)
    return Data(x=torch.cat((pos, normal_tensor), dim=1), pos=pos, edge_index=edge_index, y=pressure)


def _processed_split_dir(root: str | Path, variant: str, split: str) -> Path:
    """Return the processed graph directory for a dataset split."""
    return Path(root) / "processed" / variant / split


class _DiskDataset(Dataset):
    """Load one processed graph per file."""

    def __init__(self, root: str | Path, split: str, variant: str, file_names: list[str]) -> None:
        self.split = split
        self.variant = variant
        self._file_names = file_names
        super().__init__(root=str(root), log=False)

    @property
    def processed_dir(self) -> str:
        return str(_processed_split_dir(self.root, self.variant, self.split))

    @property
    def processed_file_names(self) -> list[str]:
        return self._file_names

    def _process(self) -> None:
        if self._file_names and all(Path(path).is_file() for path in self.processed_paths):
            return
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.process()

    def len(self) -> int:
        return len(self._file_names)

    def get(self, idx: int) -> Data:
        return cast(Data, torch.load(str(Path(self.processed_dir) / self._file_names[idx]), weights_only=False))


class _MeshDataset(_DiskDataset):
    """Process VTK meshes into separate graph files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        variant: str,
        mesh_files: list[Path],
        file_names: list[str],
        pressure_field: str,
        triangulate: bool,
    ) -> None:
        self._mesh_files = mesh_files
        self._pressure_field = pressure_field
        self._triangulate = triangulate
        super().__init__(root, split, variant, file_names)

    def process(self) -> None:
        pending = [
            (source, destination)
            for source, destination in zip(self._mesh_files, self.processed_paths, strict=True)
            if not Path(destination).is_file()
        ]
        for source, destination in log_progress(pending, len(pending), f"{self.variant} {self.split} meshes"):
            if not source.is_file():
                raise FileNotFoundError(source)
            save_atomic(_mesh_data(source, self._pressure_field, self._triangulate), Path(destination))


class DrivAerNetDataset(_MeshDataset):
    """Read DrivAerNet surface pressure meshes from split design IDs.

    Positions are mapped from a fixed bounding box into ``[-1, 1]`` per axis.

    Args:
        root: Directory containing ``SurfacePressureVTK`` and split files.
        split: One of ``train``, ``val``, or ``test``.
    """

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        root_path = Path(root).expanduser()
        mesh_dir = root_path / "SurfacePressureVTK"
        split_file = root_path / "train_val_test_splits" / f"{split}_design_ids.txt"
        if not mesh_dir.is_dir():
            raise FileNotFoundError(mesh_dir)
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        super().__init__(
            root_path,
            split,
            "drivaernet",
            [mesh_dir / f"{design_id}.vtk" for design_id in ids],
            [f"{design_id}.pt" for design_id in ids],
            "p",
            False,
        )

    def get(self, idx: int) -> Data:
        """Return a processed mesh with positions scaled into ``[-1, 1]``."""
        data = super().get(idx)
        data.x[:, :3] = (data.x[:, :3] - _DRIVAERNET_MIN).mul_(2).div_(_DRIVAERNET_MAX - _DRIVAERNET_MIN).sub_(1)
        data.pos = data.x[:, :3].clone()
        return data


class ShapeNetCarDataset(_MeshDataset):
    """Read quad surface meshes from ShapeNet-Car parameter folders.

    Args:
        root: Directory containing ``param0`` through ``param8``.
        split: ``train`` for ``param1`` through ``param8`` or ``test`` for ``param0``.

    Sample folders without ``quadpress_smpl.vtk`` are skipped; the published archive has a few.
    """

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(split)
        root_path = Path(root).expanduser()
        indices = range(1, 9) if split == "train" else (0,)
        mesh_files: list[Path] = []
        file_names: list[str] = []
        for index in indices:
            param_dir = root_path / f"param{index}"
            if not param_dir.is_dir():
                raise FileNotFoundError(param_dir)
            for sample_dir in sorted(path for path in param_dir.iterdir() if path.is_dir()):
                mesh_file = sample_dir / "quadpress_smpl.vtk"
                if not mesh_file.is_file():
                    continue
                mesh_files.append(mesh_file)
                file_names.append(f"{param_dir.name}_{sample_dir.name}.pt")
        super().__init__(root_path, split, "shapenet_car", mesh_files, file_names, "point_scalars", True)


def convert_airfrans_sample(sample: Data) -> Data:
    """Keep the airfoil surface nodes, link each to its eight nearest, and select pressure.

    Args:
        sample: AirfRANS graph with positions, targets, and surface flags.

    Returns:
        A surface graph with scalar pressure targets and cached spatial edges.
    """
    from scipy.spatial import cKDTree

    surface = sample.surf.to(torch.bool)
    surface_pos = sample.pos[surface]
    pos = surface_pos.detach().cpu().numpy()
    num_nodes = int(pos.shape[0])
    if num_nodes < 2:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        neighbors = cKDTree(pos).query(pos, k=min(9, num_nodes))[1]
        rows = np.arange(num_nodes)[:, None]
        nonself = neighbors != rows
        selected = nonself & (np.cumsum(nonself, axis=1) <= 8)
        sources = np.broadcast_to(rows, neighbors.shape)[selected]
        targets = neighbors[selected]
        edge_index = to_undirected(torch.from_numpy(np.stack((sources, targets))).long(), num_nodes=num_nodes)
    return Data(x=sample.x[surface], pos=surface_pos, y=sample.y[surface, 2].float(), edge_index=edge_index)


def _airfrans_source(root: str | Path, split: str) -> Dataset:
    from torch_geometric.datasets import AirfRANS

    return AirfRANS(str(root), task="full", train=split == "train")


class _AirfRANSDataset(_DiskDataset):
    """Cache converted AirfRANS samples as separate graph files."""

    def __init__(self, root: str | Path, split: str) -> None:
        root_path = Path(root).expanduser()
        count_file = _processed_split_dir(root_path, "airfrans", split) / "count.txt"
        self._source: Dataset | None = None
        if count_file.is_file():
            count = int(count_file.read_text(encoding="utf-8"))
        else:
            self._source = _airfrans_source(root_path, split)
            count = len(self._source)
        super().__init__(root_path, split, "airfrans", [f"{index}.pt" for index in range(count)])
        self._source = None

    def process(self) -> None:
        source = self._source if self._source is not None else _airfrans_source(self.root, self.split)
        for index, destination in enumerate(self.processed_paths):
            if not Path(destination).is_file():
                save_atomic(convert_airfrans_sample(source[index]), Path(destination))
        (Path(self.processed_dir) / "count.txt").write_text(str(len(self)), encoding="utf-8")

    def get(self, idx: int) -> Data:
        """Prepend positions to the features of a processed AirfRANS graph."""
        data = super().get(idx)
        data.x = torch.cat((data.pos, data.x), dim=1)
        return data


def airfrans(root: str | Path, split: str) -> Dataset:
    """Load AirfRANS graphs with cached pressure and spatial edges.

    Node features are the 2D position followed by the five AirfRANS inputs (inlet velocity, distance to the
    airfoil, and surface normal).

    Args:
        root: Directory used by the PyG AirfRANS dataset.
        split: Training split when ``train``, or the test split otherwise.

    Returns:
        A disk-backed dataset of converted AirfRANS graphs.
    """
    return _AirfRANSDataset(root, split)


def node_stats(dataset: Dataset, key: str) -> tuple[Tensor, Tensor]:
    """Compute the population mean and standard deviation of a node attribute over all graphs, per column.

    Args:
        dataset: Graphs with a node attribute ``key`` of shape ``[N]`` or ``[N, C]``.
        key: Attribute name, such as ``x`` or ``y``.

    Returns:
        Mean and standard deviation, each of shape ``[C]`` (``[1]`` for a scalar attribute).
    """
    total = torch.zeros((), dtype=torch.float64)
    total_squared = torch.zeros((), dtype=torch.float64)
    count = 0
    for sample in dataset:
        values = sample[key].to(torch.float64)
        values = values.reshape(values.shape[0], -1)
        total = total + values.sum(dim=0)
        total_squared = total_squared + values.square().sum(dim=0)
        count += values.shape[0]
    if count == 0:
        raise ValueError(_EMPTY_DATASET)
    mean = total / count
    std = (total_squared / count - mean.square()).clamp_min(0.0).sqrt()
    return mean, std
