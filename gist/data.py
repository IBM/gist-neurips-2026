"""Datasets with cached graph embeddings."""

import copy
import io
import logging
import multiprocessing as mp
import os
import shlex
from collections.abc import Iterator, Sized
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gist.fastrp import fastrp
from gist.logs import log_progress

logger = logging.getLogger(__name__)


def _edge_payload(data: Data) -> bytes:
    """Serialize edges as bytes for transfer to worker processes."""
    buffer = io.BytesIO()
    torch.save((data.edge_index.cpu(), int(data.num_nodes)), buffer)
    return buffer.getvalue()


def save_atomic(obj: object, path: Path) -> None:
    """Save ``obj`` to ``path`` with ``torch.save`` through a per-process temporary file renamed into place."""
    # An interrupted run then leaves no truncated file, and concurrent runs filling the same cache do not collide.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(obj, partial)
    partial.replace(path)


def _write_embedding(task: tuple[bytes, Path, int, int, int], device: str) -> None:
    """Compute one graph embedding on ``device`` and save it to its cache path."""
    payload, path, embed_dim, deg, seed = task
    edge_index, num_nodes = cast(tuple[Tensor, int], torch.load(io.BytesIO(payload), weights_only=True))
    embedding = fastrp(edge_index.to(device), num_nodes, embed_dim, deg, seed=seed)
    save_atomic(embedding.half().cpu(), path)


def _write_embeddings(tasks: Iterator[tuple[bytes, Path, int, int, int]], count: int) -> Iterator[None]:
    """Write embeddings, yielding once per finished one, one at a time per GPU or on four CPU workers."""
    gpus = torch.cuda.device_count()
    devices = [f"cuda:{index}" for index in range(gpus)] if gpus else ["cpu"] * min(4, count)
    if count > 1 and len(devices) > 1:
        # One process per device for the whole run, so that no process holds memory on two GPUs.
        with ExitStack() as stack:
            context = mp.get_context("spawn")
            executors = [stack.enter_context(ProcessPoolExecutor(1, mp_context=context)) for _ in devices]
            while batch := list(islice(tasks, len(devices))):
                futures = [
                    executor.submit(_write_embedding, task, device)
                    for executor, task, device in zip(executors, batch, devices, strict=False)
                ]
                for future in futures:
                    future.result()
                    yield None
    else:
        for task in tasks:
            _write_embedding(task, devices[0])
            yield None


class EmbeddedGraphDataset(Dataset[Data]):
    """Attach cached FastRP embeddings to graphs in a dataset.

    Args:
        dataset: Dataset yielding PyG graph data.
        embed_dim: Number of embedding coordinates.
        deg: Number of diffusion powers.
        generate_embeddings: Generate missing cache files when true.
        split: Split name used in the cache key.
        name: Dataset name used in the cache key.
        root: Directory where graph embeddings are cached.
    """

    def __init__(
        self,
        dataset: Dataset[Data],
        embed_dim: int,
        deg: int,
        generate_embeddings: bool,
        split: str,
        name: str,
        root: str | Path,
    ) -> None:
        self.dataset = dataset
        self.cache_dir = Path(root) / "graph_embeds" / f"{name}_{split}_dim{embed_dim}_deg{deg}"
        self.cache_paths = [self.cache_dir / f"embed_{index:06d}.pt" for index in range(len(cast(Sized, dataset)))]
        missing = [index for index, path in enumerate(self.cache_paths) if not path.is_file()]
        if missing and not generate_embeddings:
            command = f"python scripts/generate_embeddings.py {name} --embed_dim {embed_dim} --deg {deg} --root {shlex.quote(str(Path(root).parent))}"
            message = f"Missing {len(missing)} cached graph embeddings. Run {command}, or pass generate_embeddings=True"
            raise ValueError(message)
        if missing:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Generating %d graph embeddings in %s", len(missing), self.cache_dir)
            tasks = (
                (_edge_payload(dataset[index]), self.cache_paths[index], embed_dim, deg, index) for index in missing
            )
            for _ in log_progress(_write_embeddings(tasks, len(missing)), len(missing), f"{name} {split} embeddings"):
                pass

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Data:
        graph = copy.copy(self.dataset[index])
        graph.graph_embed = torch.load(self.cache_paths[index], map_location="cpu", weights_only=True, mmap=True)
        return graph
