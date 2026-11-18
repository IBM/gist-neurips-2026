"""Gauge-Invariant Spectral Transformer model."""

from typing import cast

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphConv

from gist.attention import (
    GaugeEquivariantSpectralAttentionBlock,
    GaugeInvariantSpectralAttentionBlock,
    LinearAttentionBlock,
)


class GISTBlock(nn.Module):
    """Combine feature, neighborhood, and spectral attention pathways.

    Args:
        dim: Node feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        branch_dim: Width of the feature self-attention and graph convolution pathways.
        spectral_threshold: Initial threshold of the spectral attention kernel.
        update_embedding: Whether the block updates the spectral embeddings for the next block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        branch_dim: int,
        spectral_threshold: float,
        update_embedding: bool,
    ) -> None:
        super().__init__()
        self.feature_in = nn.Linear(dim, branch_dim) if branch_dim != dim else nn.Identity()
        self.local_in = nn.Linear(dim, branch_dim) if branch_dim != dim else nn.Identity()
        self.self_attn_blocks = nn.ModuleList(
            [LinearAttentionBlock(width, num_heads, mlp_ratio) for width in (branch_dim, branch_dim, dim)]
        )
        self.gconv = GraphConv(branch_dim, branch_dim, aggr="mean")
        self.cross_attn_feat = GaugeInvariantSpectralAttentionBlock(dim, num_heads, mlp_ratio, spectral_threshold)
        self.cross_attn_graph = GaugeEquivariantSpectralAttentionBlock(dim, num_heads) if update_embedding else None
        self.proj1 = nn.Linear(branch_dim, dim)
        self.proj2 = nn.Linear(branch_dim, dim)
        self.proj3 = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        graph_embed: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all three branches; refine spectral embeddings only when another block follows."""
        x1 = self.self_attn_blocks[0](self.feature_in(x), batch, num_graphs)
        x2 = self.self_attn_blocks[1](self.gconv(self.local_in(x), edge_index), batch, num_graphs)
        x3 = self.cross_attn_feat(x, graph_embed, batch, num_graphs)
        x3 = self.self_attn_blocks[2](x3, batch, num_graphs)
        if self.cross_attn_graph is not None:
            graph_embed = self.cross_attn_graph(graph_embed, x, batch, num_graphs)
        x_out = self.proj1(x1) + self.proj2(x2) + self.proj3(x3)
        return x_out, graph_embed


class GIST(nn.Module):
    """GIST (Gauge-Invariant Spectral Transformer) for graph node predictions.

    Args:
        input_dim: Input node feature dimension.
        output_dim: Output node feature dimension.
        hidden_dim: Hidden node feature dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        dropout: Dropout probability after the input layer and each block.
        branch_dim: Width of each block's feature self-attention and graph convolution pathways; ``hidden_dim`` when
            ``None``.
        spectral_threshold: Initial threshold of each block's spectral attention kernel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.5,
        branch_dim: int | None = None,
        spectral_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                GISTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio,
                    hidden_dim if branch_dim is None else branch_dim,
                    spectral_threshold,
                    update_embedding=layer < num_layers - 1,
                )
                for layer in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Predict one output vector per node.

        Args:
            data: Graph data with ``x``, ``edge_index``, and ``graph_embed``.

        Returns:
            Tensor of shape ``[N, output_dim]``.
        """
        batch: torch.Tensor | None = getattr(data, "batch", None)
        num_graphs = 1
        if batch is not None:
            num_graphs = int(data.num_graphs) if isinstance(data, Batch) else int(batch.max()) + 1
        x = self.dropout(self.input_proj(data.x))
        graph_embed = data.graph_embed.to(data.x.dtype)
        for block in self.blocks:
            x, graph_embed = block(x, graph_embed, data.edge_index, batch, num_graphs)
            x = self.dropout(x)
        return cast(torch.Tensor, self.head(self.norm(x)))
