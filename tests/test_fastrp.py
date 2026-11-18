import math

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from gist.data import EmbeddedGraphDataset
from gist.fastrp import fastrp


def test_fastrp_matches_dense_diffusion_and_is_reproducible() -> None:
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 4, 5, 6, 7], [1, 1, 0, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    num_nodes, embed_dim, deg, seed = 10, 7, 3, 23

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, target in edge_index.T.tolist():
        adjacency[source, target] = 1
        adjacency[target, source] = 1
    degrees = adjacency.sum(dim=1)
    transition = adjacency / degrees.clamp_min(1).unsqueeze(1)

    generator = torch.Generator().manual_seed(seed)
    sparsity = math.sqrt(num_nodes)
    count = int(num_nodes / sparsity)
    rows = torch.stack([torch.randperm(num_nodes, generator=generator)[:count] for _ in range(embed_dim)])
    signs = 2 * torch.randint(0, 2, (rows.numel(),), generator=generator) - 1
    projection = torch.zeros((num_nodes, embed_dim), dtype=torch.float32)
    columns = torch.arange(embed_dim).repeat_interleave(count)
    projection[rows.flatten(), columns] = signs.float() * math.sqrt(sparsity)
    projection *= torch.where(degrees > 0, degrees.rsqrt(), 0.0).unsqueeze(1)

    current = transition @ projection
    total = current.clone()
    for _ in range(1, deg):
        current = transition @ current
        total += current
    total[:9] -= total[:9].mean(dim=0, keepdim=True)
    expected = F.normalize(total, p=2, dim=1) * math.sqrt(embed_dim)

    actual = fastrp(edge_index, num_nodes, embed_dim, deg, seed)
    assert actual.shape == (num_nodes, embed_dim)
    assert actual.dtype == torch.float32
    assert actual.device == edge_index.device
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(actual, fastrp(edge_index, num_nodes, embed_dim, deg, seed))
    assert torch.count_nonzero(actual[9]) == 0


def test_embedding_cache_stores_and_loads_float16(tmp_path) -> None:
    graph = Data(x=torch.zeros(4, 1), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), num_nodes=4)
    dataset = EmbeddedGraphDataset(
        [graph], embed_dim=8, deg=2, generate_embeddings=True, split="all", name="toy", root=tmp_path
    )
    loaded = dataset[0].graph_embed
    assert loaded.dtype == torch.float16
    torch.testing.assert_close(loaded.float(), fastrp(graph.edge_index, 4, 8, 2, seed=0), atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fastrp_gives_the_same_embeddings_on_cpu_and_gpu() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    cpu = fastrp(edge_index, 5, 8, 3, seed=7)
    torch.testing.assert_close(fastrp(edge_index.cuda(), 5, 8, 3, seed=7).cpu(), cpu)
