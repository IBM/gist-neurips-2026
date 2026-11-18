"""FastRP node embeddings with sparse graph diffusion."""

import math

import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from torch import Tensor
from torch_geometric.utils import degree, to_scipy_sparse_matrix


def _subtract_component_means(values: Tensor, edge_index: Tensor) -> Tensor:
    """Center ``values`` in place within each connected component."""
    matrix = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=values.shape[0])
    count, component = connected_components(matrix, directed=False)
    labels = torch.as_tensor(component, device=values.device, dtype=torch.long)
    # Summed in float64 so that identical rows cancel exactly instead of normalizing rounding residue to full norm.
    sums = torch.zeros(count, values.shape[1], device=values.device, dtype=torch.float64)
    sums.index_add_(0, labels, values.double())
    return values.sub_((sums / labels.bincount(minlength=count).unsqueeze(1)).to(values.dtype)[labels])


def fastrp(
    edge_index: Tensor,
    num_nodes: int,
    embed_dim: int,
    deg: int,
    seed: int,
) -> Tensor:
    """Embed nodes by diffusing a very sparse random projection, centered per connected component.

    Args:
        edge_index: Directed edges of shape ``[2, num_edges]``.
        num_nodes: Number of nodes, including isolated nodes.
        embed_dim: Number of embedding coordinates.
        deg: Number of transition matrix powers to sum.
        seed: Seed for the local random generator.

    Returns:
        Float32 embeddings of shape ``[num_nodes, embed_dim]``.
    """
    if num_nodes < 0 or embed_dim < 1 or deg < 1:
        message = "num_nodes must be nonnegative, and embed_dim and deg must be positive"
        raise ValueError(message)

    device = edge_index.device
    edges = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    adjacency = torch.sparse_coo_tensor(
        edges,
        torch.ones(edges.shape[1], device=device, dtype=torch.float32),
        (num_nodes, num_nodes),
        check_invariants=True,
    ).coalesce()
    adjacency.values().clamp_(max=1.0)

    rows = adjacency.indices()[0]
    degrees = degree(rows, num_nodes=num_nodes, dtype=torch.float32)
    inverse_degrees = torch.where(degrees > 0, degrees.reciprocal(), 0.0)
    transition = torch.sparse_coo_tensor(
        adjacency.indices(),
        adjacency.values() * inverse_degrees[rows],
        adjacency.shape,
        is_coalesced=True,
        check_invariants=False,
    )

    # Drawn on the CPU so a seed gives the same projection on every device.
    generator = torch.Generator().manual_seed(seed)

    sparsity = math.sqrt(num_nodes)
    nonzeros_per_column = int(num_nodes / sparsity) if num_nodes else 0
    projection_rows = torch.cat(
        [
            torch.randperm(num_nodes, device="cpu", generator=generator)[:nonzeros_per_column].clone()
            for _ in range(embed_dim)
        ]
    ).to(device)
    projection_cols = torch.arange(embed_dim, device=device).repeat_interleave(nonzeros_per_column)
    signs = (2 * torch.randint(0, 2, (projection_rows.numel(),), device="cpu", generator=generator) - 1).to(device)
    inverse_sqrt_degrees = torch.where(degrees > 0, degrees.rsqrt(), 0.0)
    projection = torch.sparse_coo_tensor(
        torch.stack((projection_rows, projection_cols)),
        signs.to(torch.float32) * math.sqrt(sparsity) * inverse_sqrt_degrees[projection_rows],
        (num_nodes, embed_dim),
        check_invariants=False,
    ).coalesce()

    current = torch.sparse.mm(transition, projection).to_dense()
    total = current
    for _ in range(1, deg):
        current = torch.sparse.mm(transition, current)
        total += current
    total = _subtract_component_means(total, edge_index)
    return F.normalize(total, p=2, dim=1) * math.sqrt(embed_dim)
