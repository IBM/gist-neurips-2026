"""Linear attention and spectral attention blocks for GIST."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch: torch.Tensor | None,
    num_graphs: int,
) -> torch.Tensor:
    """Apply linear attention independently within each graph.

    Args:
        q: Nonnegative queries of shape ``[N, heads, key_dim]``.
        k: Nonnegative keys with the same shape as ``q``.
        v: Values of shape ``[N, heads, value_dim]``.
        batch: Graph index for each node, or ``None`` for one graph.
        num_graphs: Number of graphs in the batch.

    Returns:
        Attended values of shape ``[N, heads, value_dim]``.
    """
    indicator = None if batch is None or num_graphs == 1 else F.one_hot(batch, num_classes=num_graphs)
    if indicator is None:
        numerator = torch.einsum("nhk,hkv->nhv", q, torch.einsum("nhk,nhv->hkv", k, v))
    else:
        node_index = torch.arange(q.shape[0], device=q.device)
        summary = torch.einsum("nb,nhk,nhv->bhkv", indicator.to(k.dtype), k, v)
        numerator = torch.einsum("nhk,bhkv->nbhv", q, summary)[node_index, batch]
    # The normalizer is a sum over all nodes of a graph, so it is kept in float32 even under autocast.
    with torch.autocast(q.device.type, enabled=False):
        q32, k32 = q.float(), k.float()
        if indicator is None:
            denominator = torch.einsum("nhk,hk->nh", q32, k32.sum(dim=0))
        else:
            key_sum = torch.einsum("nb,nhk->bhk", indicator.float(), k32)
            denominator = torch.einsum("nhk,bhk->nbh", q32, key_sum)[node_index, batch]
        return (numerator.float() / (denominator.unsqueeze(-1) + 1e-6)).to(v.dtype)


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch: torch.Tensor | None,
    num_graphs: int,
    num_heads: int,
    q_norm: nn.Module | None = None,
    k_norm: nn.Module | None = None,
) -> torch.Tensor:
    q = rearrange(q, "n (h d) -> n h d", h=num_heads)
    k = rearrange(k, "n (h d) -> n h d", h=num_heads)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)
    q = F.relu(q)
    k = F.relu(k)
    v = rearrange(v, "n (h d) -> n h d", h=num_heads)
    return rearrange(linear_attention(q, k, v, batch, num_graphs), "n h d -> n (h d)")


class MLP(nn.Module):
    """Gated feed-forward layer with a SiLU activation.

    Args:
        dim: Input and output dimension.
        mlp_ratio: Hidden dimension divided by ``dim``.
    """

    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, mlp_ratio * dim)
        self.w2 = nn.Linear(mlp_ratio * dim, dim)
        self.w3 = nn.Linear(dim, mlp_ratio * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transformation."""
        return cast(torch.Tensor, self.w2(F.silu(self.w1(x)) * self.w3(x)))


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkvw = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = nn.RMSNorm(dim // num_heads)
        self.k_norm = nn.RMSNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None, num_graphs: int) -> torch.Tensor:
        q, k, v = self.qkvw(x).chunk(3, dim=-1)
        return cast(
            torch.Tensor, self.proj(_attend(q, k, v, batch, num_graphs, self.num_heads, self.q_norm, self.k_norm))
        )


class LinearAttentionBlock(nn.Module):
    """Self-attention on node features followed by a feed-forward layer.

    Args:
        dim: Node feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.attn = _SelfAttention(dim, num_heads)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None, num_graphs: int) -> torch.Tensor:
        """Update the node features within each graph."""
        x = self.w(x)
        x = x + self.attn(x, batch, num_graphs)
        return cast(torch.Tensor, x + self.ffn(self.ffn_norm(x)))


class _GaugeInvariantAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, threshold: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.threshold = nn.Parameter(torch.full((num_heads,), float(threshold)))
        self.vw = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        graph_embed: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        keys = graph_embed - self.threshold.repeat_interleave(graph_embed.shape[-1] // self.num_heads)
        attended = _attend(
            keys,
            keys,
            self.vw(x),
            batch,
            num_graphs,
            self.num_heads,
        )
        return cast(torch.Tensor, self.proj(attended))


class GaugeInvariantSpectralAttentionBlock(nn.Module):
    """Gauge-Invariant Spectral Self-Attention for node features.

    Spectral embeddings, shifted by a learned per-head threshold, provide the
    queries and keys; node features provide the values.

    Args:
        dim: Node feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        threshold: Initial threshold subtracted from the embeddings before the attention kernel's relu.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, threshold: float) -> None:
        super().__init__()
        self.attn = _GaugeInvariantAttention(dim, num_heads, threshold)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = MLP(dim, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        graph_embed: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        """Update node features from spectral attention within each graph."""
        x = x + self.attn(graph_embed, x, batch, num_graphs)
        return cast(torch.Tensor, x + self.ffn(self.ffn_norm(x)))


class _GaugeEquivariantAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, 2 * dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        graph_embed: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        q, k = self.qk(x).chunk(2, dim=-1)
        return _attend(q, k, graph_embed, batch, num_graphs, self.num_heads)


class GaugeEquivariantSpectralAttentionBlock(nn.Module):
    """Gauge-Equivariant Spectral Self-Attention for spectral embeddings.

    Projected node features provide queries and keys; spectral embeddings
    provide the values.

    Args:
        dim: Node feature dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = _GaugeEquivariantAttention(dim, num_heads)

    def forward(
        self,
        graph_embed: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        """Update spectral embeddings from node features within each graph."""
        return cast(torch.Tensor, graph_embed + self.attn(x, graph_embed, batch, num_graphs))
