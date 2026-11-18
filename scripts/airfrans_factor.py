"""Compute the factor between AirfRANS surface-pressure MSE standardized over surface nodes and over all nodes.

Pools the mean and standard deviation of pressure over the first training meshes, once over all mesh nodes (the
convention of Transolver and the AirfRANS paper) and once over surface nodes (this code's training). An MSE on
surface-standardized pressure times (surface std / all-node std) ** 2 is the MSE on all-node-standardized pressure.
"""

import argparse
import json

import torch
from torch_geometric.datasets import AirfRANS


def _pooled(values: list[torch.Tensor]) -> tuple[float, float, int]:
    """Return the mean, population standard deviation and count of all values together."""
    count = sum(value.numel() for value in values)
    mean = sum(value.double().sum().item() for value in values) / count
    variance = sum((value.double() - mean).square_().sum().item() for value in values) / count
    return mean, variance**0.5, count


def main() -> None:
    """Print the pooled statistics and the factor for the first 720 and all 800 training meshes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/airfrans", help="AirfRANS directory, as used by training")
    args = parser.parse_args()
    dataset = AirfRANS(args.root, task="full", train=True)
    every, surface = [], []
    for sample in dataset:
        every.append(sample.y[:, 2].clone())
        surface.append(sample.y[sample.surf.bool(), 2])
    for meshes in (720, len(dataset)):
        all_mean, all_std, all_nodes = _pooled(every[:meshes])
        surface_mean, surface_std, surface_nodes = _pooled(surface[:meshes])
        print(
            json.dumps(
                {
                    "meshes": meshes,
                    "all_mean": all_mean,
                    "all_std": all_std,
                    "all_nodes": all_nodes,
                    "surface_mean": surface_mean,
                    "surface_std": surface_std,
                    "surface_nodes": surface_nodes,
                    "factor": (surface_std / all_std) ** 2,
                }
            )
        )


if __name__ == "__main__":
    main()
