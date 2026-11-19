# GIST: Gauge-Invariant Spectral Transformers

[Paper (NeurIPS 2026)](https://neurips.cc/virtual/2026/poster/152787) · [arXiv:2603.16849](https://arxiv.org/abs/2603.16849) · [Project page](https://ibm.github.io/gist-neurips-2026/)

GIST is a graph transformer with end-to-end linear cost in the number of nodes. Spectral node embeddings come from FastRP random projections, and attention uses them only through their inner products, which estimate a graph kernel that does not depend on the arbitrary choice of eigenvector basis (sign flips, rotations within eigenspaces, solver details). Every GIST block runs three branches in parallel (linear self-attention on node features, a graph convolution followed by linear self-attention, and gauge-invariant spectral attention); only blocks followed by another refine the embeddings for the next block with gauge-equivariant attention. This repository contains the model and the code to train it on the node-classification and mesh-regression benchmarks of the paper.

## Install

Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock` (install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`):

```bash
git clone https://github.com/IBM/gist-neurips-2026.git
cd gist-neurips-2026
uv sync                       # node-classification benchmarks
uv sync --all-extras          # or: also meshes (pyvista), ogbn-arxiv (ogb) and TensorBoard
source .venv/bin/activate     # or prefix the python and torchrun commands below with `uv run`
```

With uv, skip the `pip install` lines in the benchmark sections below: `uv sync --all-extras` covers them.

With pip, which resolves the versions afresh:

```bash
pip install -e .            # node-classification benchmarks
pip install -e ".[mesh]"    # AirfRANS, ShapeNet-Car, DrivAerNet (pyvista)
pip install -e ".[ogb]"     # ogbn-arxiv
pip install -e ".[tensorboard]"  # training curves in TensorBoard
```

`make install` runs `uv sync --all-extras`; `make test` and `make check` run the tests and the lint and type checks.

## Usage

```python
from gist import GIST, load_dataset

bench = load_dataset("cora", embed_dim=512, deg=128, generate_embeddings=True)
data = bench.splits["train"][0]
model = GIST(input_dim=bench.num_features, output_dim=bench.num_targets)
logits = model(data)  # [num_nodes, num_classes]
```

`load_dataset` attaches the FastRP embeddings (`data.graph_embed`) from an on-disk cache, stored and loaded as float16; `GIST` converts them to the dtype of `data.x`. Precompute them once per dataset and embedding size, with the `embedding` values of the benchmark's config:

```bash
python scripts/generate_embeddings.py cora --embed_dim 512 --deg 128
```

### Your own graph

`GIST` takes a PyG `Data` with node features `x`, edges listed in both directions and FastRP embeddings in `graph_embed`:

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gist import GIST, fastrp

x = torch.randn(1000, 16)  # node features
y = torch.randint(3, (1000,))  # node labels
edge_index = to_undirected(torch.randint(1000, (2, 5000)), num_nodes=1000)
data = Data(x=x, edge_index=edge_index, y=y)
data.graph_embed = fastrp(edge_index, data.num_nodes, embed_dim=256, deg=64, seed=0)

model = GIST(input_dim=16, output_dim=3, num_layers=2, branch_dim=128)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(100):
    loss = F.cross_entropy(model(data), data.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

For a dataset of several graphs, compute the embeddings of each graph separately and batch the graphs with PyG's `DataLoader`; attention stays within each graph. The configs in `configs/` give tested settings for `embed_dim`, `deg` and the model: 256 and 64 on meshes, 512 and 128 to 256 on citation graphs.

## Reproducing the results

Each benchmark has one config in `configs/` holding its hyperparameters (multi-seed run).

```bash
python -m gist.train configs/cora.yaml                  # writes results/cora_<timestamp>.json
python -m gist.train configs/ppi.yaml --set train.epochs=50
```

Each run writes one JSON file with `config` (the settings after defaults and `--set`), `per_seed` (for each seed: `seed`, `selected_epoch` and the `val` and `test` metrics) and `mean` and `std` (population standard deviation over seeds) of the `val` and `test` metrics. Metrics are fractions: `accuracy` for single-label classification, `micro_f1` for PPI, and `mse`, `rel_l2` and `r2` for meshes. The `# Expected` line at the top of each config gives the test results of this code, to compare with `mean.test` (its percentages are the fractions × 100).

To check an install end to end in under a minute on a CPU (download, embeddings, training, refit, test), train Cora for two epochs with one seed:

```bash
python scripts/generate_embeddings.py cora --embed_dim 512 --deg 128
python -m gist.train configs/cora.yaml --set 'seeds=[0]' --set train.epochs=2 --set train.compile=false --output results/smoke.json
```

The run passes when it ends with `Results written to results/smoke.json`; two epochs are not expected to reach the accuracy of the full run.

Training logs one line per epoch (loss, learning rate, validation metrics, time). With `--set tensorboard_dir=runs`, it also writes the curves of each seed to `runs/<dataset>_<timestamp>/seed<seed>` for TensorBoard.

Inductive benchmarks train on several GPUs with `torchrun --nproc_per_node=4 -m gist.train configs/drivaernet.yaml`: each GPU takes its share of the graphs, so a run uses at most as many GPUs as its smallest split has graphs (2 for PPI), and `train.lr`, the rate for one graph per step, is multiplied by the number of graphs per step (GPUs × `batch_size`).

Node-classification datasets are downloaded automatically into `data/`. Photo has no official split: each seed draws a random 60/20/20 train/validation/test split.

### Replicated results

These benchmarks have been rerun with this code; each entry gives the full procedure behind its number.

Node classification, test results of this code as reported in the paper (mean ± standard deviation over 6 seeds):

| Benchmark | Config | Metric (%) | Test |
|---|---|---|---|
| Cora | `configs/cora.yaml` | accuracy | 84.25 ± 0.65 |
| CiteSeer | `configs/citeseer.yaml` | accuracy | 73.85 ± 0.63 |
| PubMed | `configs/pubmed.yaml` | accuracy | 82.60 ± 0.37 |
| Photo | `configs/photo.yaml` | accuracy | 95.65 ± 0.46 |
| ogbn-arxiv | `configs/ogbn-arxiv.yaml` | accuracy | 72.31 ± 0.29 |
| PPI | `configs/ppi.yaml` | micro-F1 | 99.533 ± 0.014 |

#### Cora

```bash
pip install -e .
python scripts/generate_embeddings.py cora --embed_dim 512 --deg 128  # downloads Cora into data/Cora, caches the FastRP embeddings
python -m gist.train configs/cora.yaml --output results/cora.json
```

- Storage: 20 MB in `data/Cora`.
- Compute: one GPU with 1 GB of memory; about 1 minute for the 6 seeds on an H200.
- Protocol: the public Planetoid split (140 train, 500 validation, 1000 test nodes). Each seed selects its epoch on the validation nodes, retrains from scratch on the training and validation nodes for that many epochs (`final_training`), and scores the test nodes.

#### CiteSeer

```bash
pip install -e .
python scripts/generate_embeddings.py citeseer --embed_dim 512 --deg 128  # downloads CiteSeer into data/CiteSeer, caches the FastRP embeddings
python -m gist.train configs/citeseer.yaml --output results/citeseer.json
```

- Storage: 52 MB in `data/CiteSeer`.
- Compute: one GPU with 3 GB of memory; about 1 minute for the 6 seeds on an H200.
- Protocol: the public Planetoid split (120 train, 500 validation, 1000 test nodes). Each seed selects its epoch on the validation nodes, retrains from scratch on the training and validation nodes for that many epochs (`final_training`), and scores the test nodes.

#### PubMed

```bash
pip install -e .
python scripts/generate_embeddings.py pubmed --embed_dim 512 --deg 256  # downloads PubMed into data/PubMed, caches the FastRP embeddings
python -m gist.train configs/pubmed.yaml --output results/pubmed.json
```

- Storage: 67 MB in `data/PubMed`.
- Compute: one GPU with 3 GB of memory; about 5 minutes for the 6 seeds on an H200.
- Protocol: the public Planetoid split (60 train, 500 validation, 1000 test nodes). Each seed selects its epoch on the validation nodes, retrains from scratch on the training and validation nodes for that many epochs (`final_training`), and scores the test nodes.

#### Photo

```bash
pip install -e .
python scripts/generate_embeddings.py photo --embed_dim 512 --deg 256  # downloads Amazon Photo into data/Photo, caches the FastRP embeddings
python -m gist.train configs/photo.yaml --output results/photo.json
```

- Storage: 50 MB in `data/Photo`.
- Compute: one GPU with 2 GB of memory; about 3 minutes for the 6 seeds on an H200.
- Protocol: Photo has no standard split; the nodes are split at random into 60% train, 20% validation and 20% test, a new split for each seed. Each seed selects its epoch on the validation nodes, retrains from scratch on the training and validation nodes for that many epochs (`final_training`), and scores the test nodes.

#### ogbn-arxiv

```bash
pip install -e ".[ogb]"
python scripts/generate_embeddings.py ogbn-arxiv --embed_dim 512 --deg 256  # downloads ogbn-arxiv into data/ogbn-arxiv, caches the FastRP embeddings
python -m gist.train configs/ogbn-arxiv.yaml --output results/ogbn-arxiv.json
```

- Storage: 350 MB in `data/ogbn-arxiv`.
- Compute: one GPU with 15 GB of memory; about 3 minutes per seed on an H200, 20 minutes for the 6 seeds.
- Protocol: the official OGB split by publication year (90,941 train, 29,799 validation, 48,603 test papers), with citation edges in both directions. Each seed selects its epoch on the validation nodes, retrains from scratch on the training and validation nodes for that many epochs (`final_training`), and scores the test nodes.

#### PPI

```bash
pip install -e .
python scripts/generate_embeddings.py ppi --embed_dim 256 --deg 64  # downloads PPI into data/PPI, caches the FastRP embeddings
python -m gist.train configs/ppi.yaml --output results/ppi.json
```

- Storage: 220 MB in `data/PPI` (131 MB download, 62 MB processed graphs, 29 MB embeddings).
- Compute: one GPU with 3 GB of memory; 7 to 10 minutes per seed on an H200, about 50 minutes for the 6 seeds.
- Protocol: each seed selects its epoch on the validation graphs, retrains from scratch on the training and validation graphs for that many epochs (`final_training`), and scores the test graphs.

#### AirfRANS

Transolver: surface-pressure MSE 0.0142. This code: 0.0017 ± 0.0002 (mean ± standard deviation over 3 seeds), relative L2 2.20 ± 0.05%. Like Transolver and the AirfRANS paper, this MSE standardizes pressure with the mean and standard deviation over all training nodes, volume included; training standardizes it over surface nodes, where its MSE is 0.00060 ± 0.00007, and the squared ratio of the two standard deviations, 2.77, converts one into the other (`python scripts/airfrans_factor.py` computes it).

```bash
pip install -e ".[mesh]"
python scripts/generate_embeddings.py airfrans --embed_dim 256 --deg 64  # downloads AirfRANS into data/airfrans, caches the FastRP embeddings
python -m gist.train configs/airfrans.yaml --output results/airfrans.json
```

- Storage: 16 GB in `data/airfrans` (505 MB embeddings).
- Compute: one GPU with 1 GB of memory; about 51 minutes per seed on an H200, 2.6 hours for the 3 seeds.
- Protocol: each seed trains for 300 epochs, selects its epoch on the validation airfoils, and scores the test airfoils, without retraining.

#### ShapeNet-Car

Transolver: relative L2 7.45%. This code: 7.02 ± 0.05%, MSE 0.0088 ± 0.0003 (mean ± standard deviation over 3 seeds).

```bash
pip install -e ".[mesh]"
python scripts/generate_embeddings.py shapenet-car --embed_dim 256 --deg 64  # expects the meshes under data/shapenet-car, caches the FastRP embeddings
python -m gist.train configs/shapenet-car.yaml --output results/shapenet-car.json
```

- Storage: 2.2 GB in `data/shapenet-car` (1.6 GB embeddings).
- Compute: one GPU with 1 GB of memory; about 45 minutes per seed on an H200, 2.2 hours for the 3 seeds.
- Protocol: 711 training, 78 validation and 100 test cars. Each seed trains for 200 epochs, selects its epoch on validation relative L2, and scores the test cars, without retraining.

#### DrivAerNet

This code, on the full meshes in training and evaluation: MSE 0.0400, relative L2 14.46% per car and 20.0% pooled over all nodes on standardized pressure (seed 42). The pooled value is √(MSE / 0.9965), where 0.9965 is the mean squared standardized pressure over the test nodes.

```bash
pip install -e ".[mesh]"
python scripts/generate_embeddings.py drivaernet --embed_dim 256 --deg 64  # expects the meshes under data/drivaernet, caches the FastRP embeddings on every visible GPU
torchrun --nproc_per_node=4 -m gist.train configs/drivaernet.yaml --output results/drivaernet.json
```

- Storage: about 1 TB in `data/drivaernet` (95 GB meshes, 152 GB processed graphs, 769 GB embeddings); keep it on a fast local disk during training.
- Compute: four GPUs with 33 GB of memory each; 3.4 hours on four H100s.
- Protocol: trains for up to 100 epochs, validates every 2 epochs, stops after 20 validations without improvement, selects its epoch on validation MSE, and scores the test cars, without retraining.

### Mesh data

AirfRANS is downloaded automatically. The other two come from their authors:

- ShapeNet-Car: [mlcfd_data.zip](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip) (2 GB, Umetani and Bickel, 2018); move the folders `param0` to `param8` of its `mlcfd_data/training_data/` into `data/shapenet-car/`.
- DrivAerNet: the release of 4,000 cars; its [page](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1) describes `SurfacePressureVTK` and `train_val_test_splits` but gives no download link; ask the authors for them.

The expected layout:

```
data/drivaernet/SurfacePressureVTK/*.vtk
data/drivaernet/train_val_test_splits/{train,val,test}_design_ids.txt
data/shapenet-car/param{0..8}/<sample>/quadpress_smpl.vtk
data/airfrans/                              # downloaded by torch_geometric.datasets.AirfRANS
```

DrivAerNet positions are scaled into [-1, 1] per axis with a fixed bounding box. Targets are surface pressure, standardized with the training split's mean and standard deviation. MSE is reported on standardized pressure over all evaluated nodes; relative L2 is computed per mesh in physical units and averaged over meshes. AirfRANS (task `full`) holds out the last tenth of its 800 training airfoils for validation; only its airfoil surface nodes are kept, each linked to its eight nearest surface nodes; node features are the 2D position and the five AirfRANS inputs, and inputs and pressure are standardized with training statistics. The model predicts the pressure coefficient, pressure over half the squared inlet speed, which is converted to standardized pressure before the loss and the metrics. ShapeNet-Car (fold `param0` as test, 100 cars) holds out the last tenth of its 789 training cars for validation; sample folders without a pressure mesh are skipped.

### Dataset terms

This repository ships no data. Each benchmark is downloaded from its authors and comes with its own terms, separate from the Apache-2.0 licence of this code; users are responsible for following them.

| Dataset | Source | Terms |
|---|---|---|
| Cora, CiteSeer, PubMed | Planetoid splits (Yang et al., 2016), via PyG | no licence stated; cite the authors |
| Amazon Photo | Shchur et al. (2018), via PyG | no licence stated; cite the authors |
| ogbn-arxiv | Open Graph Benchmark | ODC-BY |
| PPI | Hamilton et al. (2017), via PyG | no licence stated; cite the authors |
| AirfRANS | Bonnet et al. (2022) | ODbL 1.0 |
| DrivAerNet | Elrefaie et al. (2024) | CC BY-NC 4.0, non-commercial use only |
| ShapeNet-Car | Umetani and Bickel (2018), car shapes from ShapeNet | ShapeNet terms of use, non-commercial research and education only |

## Citation

```bibtex
@inproceedings{rigotti2026gist,
  title     = {{GIST}: Gauge-Invariant Spectral Transformers for Scalable Graph Neural Operators},
  author    = {Rigotti, Mattia and Thumiger, Nicholas and Frick, Thomas},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```

## License

Apache License 2.0, see `LICENSE`.
