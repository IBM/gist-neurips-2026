"""Small surface-mesh and airfoil graph fixtures."""

import math
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch
from torch_geometric.data import Data, Dataset

from gist.datasets import _standardize
from gist.mesh import (
    AIRFRANS_INLET_COLUMNS,
    DrivAerNetDataset,
    ShapeNetCarDataset,
    convert_airfrans_sample,
    node_stats,
)


@pytest.mark.parametrize(
    ("source", "feature_width", "position_width", "edge_count"),
    [
        ("drivaernet", 6, 3, 24),
        ("shapenet_car", 6, 3, 32),
        ("airfrans", 5, 2, 20),
    ],
)
def test_mesh_graphs(
    tmp_path: Path,
    source: str,
    feature_width: int,
    position_width: int,
    edge_count: int,
) -> None:
    if source == "airfrans":
        positions = torch.tensor([(i, j) for i in range(3) for j in range(3)], dtype=torch.float32)
        targets = torch.zeros((9, 4), dtype=torch.float32)
        targets[:, 2] = torch.arange(9, dtype=torch.float32)
        # Views into a larger split, as PyG's in-memory AirfRANS returns them.
        raw = Data(
            x=torch.arange(90, dtype=torch.float32).reshape(18, 5)[:9],
            pos=torch.cat((positions, positions))[:9],
            y=torch.cat((targets, targets))[:9],
            surf=(torch.arange(18) % 2 == 0)[:9],
        )
        item = convert_airfrans_sample(raw)
        assert torch.equal(item.y, targets[raw.surf, 2])
        loaded = item.clone()
        loaded.x = torch.cat((loaded.pos, loaded.x), dim=1)
        scaled = _standardize(loaded, mean=1.0, std=2.0, inlet_columns=AIRFRANS_INLET_COLUMNS)
        pressure_coefficient = item.y / item.x[:, :2].square().sum(dim=1).mul(0.5)
        prediction = scaled.output_scale.squeeze(1) * pressure_coefficient + scaled.output_offset.squeeze(1)
        assert torch.allclose(prediction, scaled.y)
        assert all(value.untyped_storage().nbytes() == value.nbytes for _, value in item)
    else:
        mesh = pv.Plane(i_resolution=2, j_resolution=2)
        if source == "drivaernet":
            del mesh.point_data["Normals"]
            mesh.point_data["p"] = np.arange(9, dtype=np.float32)
            mesh_dir = tmp_path / "SurfacePressureVTK"
            mesh_dir.mkdir()
            mesh.save(mesh_dir / "car.vtk")
            split_dir = tmp_path / "train_val_test_splits"
            split_dir.mkdir()
            (split_dir / "train_design_ids.txt").write_text("car\n", encoding="utf-8")
            dataset: Dataset = DrivAerNetDataset(tmp_path, "train")
        else:
            mesh.point_data["point_scalars"] = np.arange(9, dtype=np.float32)
            for index in range(1, 9):
                (tmp_path / f"param{index}").mkdir()
            mesh_dir = tmp_path / "param1" / "car"
            mesh_dir.mkdir()
            mesh.save(mesh_dir / "quadpress_smpl.vtk")
            (tmp_path / "param2" / "no_mesh").mkdir()
            dataset = ShapeNetCarDataset(tmp_path, "train")
        assert len(dataset) == 1
        processed_dir = tmp_path / "processed" / ("drivaernet" if source == "drivaernet" else "shapenet_car") / "train"
        assert len(list(processed_dir.glob("*.pt"))) == 1
        item = dataset[0]
        if source == "drivaernet":
            assert torch.all(item.pos[:, 2] == -1)
            mean, std = node_stats(dataset, "y")
            assert float(mean) == pytest.approx(4.0)
            assert float(std) == pytest.approx(math.sqrt(20 / 3))

    num_nodes = 5 if source == "airfrans" else 9
    assert item.x.shape == (num_nodes, feature_width)
    assert item.pos.shape == (num_nodes, position_width)
    assert item.y.shape == (num_nodes,)
    assert item.x.dtype == item.pos.dtype == item.y.dtype == torch.float32
    if source != "airfrans":
        assert torch.equal(item.x[:, :position_width], item.pos)
    assert item.edge_index.shape == (2, edge_count)
    assert item.edge_index.dtype == torch.long
    edges = set(map(tuple, item.edge_index.T.tolist()))
    assert len(edges) == edge_count
    assert all(0 <= source_node < num_nodes and 0 <= target_node < num_nodes for source_node, target_node in edges)
    assert all(source_node != target_node and (target_node, source_node) in edges for source_node, target_node in edges)
