"""Behavioral checks for the GIST model."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from gist.model import GIST


@pytest.mark.parametrize("compile_blocks", [False, True])
def test_graph_isolation_and_node_permutation(compile_blocks: bool) -> None:
    graph_a = Data(
        x=torch.tensor([[0.3, 1.2, -0.4], [1.1, -0.6, 0.2], [-0.2, 0.7, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        graph_embed=torch.tensor([[0.8, -0.2, 0.4, 0.9], [0.5, 0.7, -0.1, 0.3], [1.1, 0.2, 0.6, -0.4]]),
    )
    graph_b = Data(
        x=torch.tensor([[1.4, -0.1, 0.8], [-0.5, 0.9, 0.1], [0.6, 0.4, -0.7]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        graph_embed=torch.tensor([[0.2, 1.0, 0.3, -0.5], [0.6, -0.3, 0.9, 0.4], [-0.1, 0.7, 0.5, 0.8]]),
    )
    torch.manual_seed(0)
    model = GIST(3, 2, hidden_dim=4, num_layers=1, num_heads=2, mlp_ratio=2, dropout=0.2).eval()
    if compile_blocks:
        for block in model.blocks:
            block.compile()

    with torch.no_grad():
        combined = model(Batch.from_data_list([graph_a, graph_b]))
        for graph, output in [(graph_a, combined[:3]), (graph_b, combined[3:])]:
            torch.testing.assert_close(output, model(Batch.from_data_list([graph])), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(output, model(graph), atol=1e-5, rtol=1e-5)

        permutation = torch.tensor([2, 0, 1])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        permuted = Data(
            x=graph_a.x[permutation],
            edge_index=inverse[graph_a.edge_index],
            graph_embed=graph_a.graph_embed[permutation],
        )
        torch.testing.assert_close(model(permuted), combined[:3][permutation], atol=1e-5, rtol=1e-5)


def test_branch_widths_and_trained_threshold() -> None:
    graph = Data(
        x=torch.randn(6, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        graph_embed=torch.randn(6, 4),
    )
    model = GIST(3, 2, hidden_dim=8, num_layers=2, num_heads=2, branch_dim=4, spectral_threshold=1.5)
    model(graph).sum().backward()
    threshold = model.blocks[0].cross_attn_feat.attn.threshold
    assert threshold.tolist() == [1.5, 1.5]
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.blocks[0].proj1.in_features == 4
    assert model.blocks[0].proj2.in_features == 4
