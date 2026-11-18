# AGENTS.md

GIST (Gauge-Invariant Spectral Transformer) is a scalable neural operator for graphs and irregular meshes. It restricts attention to pairwise inner products of approximate spectral node embeddings (FastRP random projections), which estimate a graph kernel invariant to the choice of eigenvector basis, at end-to-end O(N) cost and without eigendecomposition. In this code, the embeddings pass through a learned per-head threshold and a ReLU before those inner products. The README explains the model, each benchmark's commands, data, storage and compute, and the format of the result files.

## Using GIST in another project

- Install from GitHub: `pip install git+https://github.com/IBM/gist-neurips-2026`. The package imports as `gist`; the unrelated `gist` on PyPI is not this code.
- GIST predicts one vector per node (classification or regression); it has no graph-level output. It suits training across many graphs or meshes as well as single large graphs.
- `fastrp(edge_index, num_nodes, embed_dim, deg, seed)` computes a graph's embeddings once, before training; recompute them when the graph changes. Store them as `data.graph_embed`.
- `GIST(input_dim, output_dim, hidden_dim=..., num_layers=...)` takes a PyG `Data`, or a `Batch` of several graphs, with `x`, `edge_index` (both directions) and `graph_embed`, and returns a `[num_nodes, output_dim]` tensor. Attention stays within each graph of a batch.
- Starting points from the configs: `embed_dim` 256 and `deg` 64 on meshes, 512 and 128 to 256 on citation graphs; `hidden_dim` 256 with 2 or 3 layers. The README section "Your own graph" has a complete training example.

## Layout

- `gist/model.py`: `GIST` and `GISTBlock`. `gist/attention.py`: linear attention, spectral attention.
- `gist/fastrp.py`: `fastrp`, the node embeddings. `gist/data.py`: their on-disk cache. `gist/datasets.py`, `gist/mesh.py`: benchmark loading, splits and standardization.
- `gist/train.py`: the training command (`python -m gist.train CONFIG`). `gist/config.py`: defaults and validation. `gist/metrics.py`: evaluation metrics.
- `configs/<benchmark>.yaml`: one config per benchmark; settings it leaves out take the defaults in `gist/config.py`. Its `# Expected` line gives the test results of this code.
- `scripts/generate_embeddings.py`: downloads a benchmark and caches its embeddings.

## Checks

```bash
make install   # uv environment with all extras and dev tools
make test      # pytest, CPU only; one GPU test is skipped without a GPU
make check     # lock file, ruff lint and format, mypy
```

Before running a benchmark, check the install with the two-epoch Cora run in the README (under a minute on a CPU).

## Running benchmarks

- Generate the embeddings first, with the `embedding` values of the config: `python scripts/generate_embeddings.py <dataset> --embed_dim <embed_dim> --deg <deg>`. Training loads them from `data/<dataset directory>/graph_embeds/` (for Cora, `data/Cora/graph_embeds/`) and stops with an error when they are missing.
- Override settings with `--set section.key=value`, for example `--set train.epochs=2`. On a CPU add `--set train.compile=false`: compiling blocks for a CPU takes minutes.
- Costs differ by orders of magnitude: the graph benchmarks take minutes on one GPU, AirfRANS and ShapeNet-Car about an hour per seed, and DrivAerNet about 3 hours on four GPUs with 1 TB of data. Each README section lists the storage, GPU memory and time.
- Only the inductive benchmarks (PPI and the meshes) run on several GPUs: `torchrun --nproc_per_node=N -m gist.train CONFIG`, with N at most the number of graphs in the smallest split (2 for PPI).
- GPU training is not bitwise reproducible; compare a rerun with the mean and standard deviation of the `# Expected` line, not its exact digits.

## Changing the code

- Cache folders are named by dataset, split, `embed_dim` and `deg` only: after changing what `fastrp` returns, delete the existing `graph_embeds/` folders, or training silently loads the old embeddings.
- A default changed in `gist/config.py` applies to every benchmark whose config leaves that setting out; check `configs/` before changing one.
- Keep `make test` and `make check` passing.
