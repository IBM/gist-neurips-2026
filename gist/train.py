"""Train GIST on node and mesh benchmarks."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sized
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gist.config import load_config, resolve_config
from gist.datasets import Benchmark, load_dataset
from gist.logs import configure_logging, format_duration
from gist.metrics import DEFAULT_SELECTION_METRIC, _is_better, compute_metrics, task_loss
from gist.model import GIST

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def _new_model(benchmark: Benchmark, config: Mapping[str, Any], device: torch.device) -> GIST:
    model = GIST(
        input_dim=benchmark.num_features,
        output_dim=benchmark.num_targets,
        **config["model"],
    ).to(device)
    if config["train"]["compile"]:
        for block in model.blocks:
            block.compile()
    return model


def _world() -> tuple[int, int]:
    """Return this process's rank and the number of processes."""
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _init_distributed() -> torch.device:
    """Join the process group when launched by torchrun with several processes, and return this process's device."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        atexit.register(dist.destroy_process_group)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    return device


def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate the rows of ``tensor`` from every process, in rank order."""
    size = torch.tensor([tensor.shape[0]], device=tensor.device)
    sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    counts = [int(count) for count in sizes]
    padded = tensor.new_zeros((max(counts), *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(gathered, padded)
    return torch.cat([rows[:count] for rows, count in zip(gathered, counts, strict=True)])


def _training_data(
    benchmark: Benchmark, config: Mapping[str, Any], device: torch.device, include_val: bool
) -> DataLoader | tuple[Data, ...]:
    dataset = benchmark.training_view(include_val=include_val)
    if benchmark.transductive:
        return (dataset[0].to(device),)
    rank, world_size = _world()
    # Seeded from the run seed so that every process shuffles alike.
    sampler: DistributedSampler[Data] | None = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, world_size, rank, seed=torch.initial_seed() % 2**63)
    workers = config["train"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    return loader


def _scheduler(optimizer: AdamW, config: Mapping[str, Any], steps_per_epoch: int) -> LambdaLR:
    """Warm up per step, then decay linearly over ``train.epochs``."""
    warmup_steps = int(config["train"]["warmup_epochs"]) * steps_per_epoch
    total_steps = int(config["train"]["epochs"]) * steps_per_epoch

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return 1.0 - min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))

    return LambdaLR(optimizer, lr_lambda=factor)


def _predict(model: Callable[[Data], torch.Tensor], data: Data) -> torch.Tensor:
    """Run the model, then apply the per-node output scale and offset a benchmark may attach to its graphs."""
    output = model(data)
    if hasattr(data, "output_scale"):
        output = output * data.output_scale + data.output_offset
    return output


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _evaluate(
    model: GIST, benchmark: Benchmark, split: str, device: torch.device, num_workers: int
) -> dict[str, float]:
    """Evaluate one split, applying its masks when present; several processes each take every n-th graph."""
    model.eval()
    dataset = benchmark.splits[split]
    rank, world_size = _world()
    graph_numbers = range(rank, len(cast(Sized, dataset)), world_size)
    batches: Iterable[Data]
    if benchmark.transductive:
        batches = (dataset[0],)
    else:
        batches = DataLoader(
            Subset(dataset, graph_numbers), batch_size=1, num_workers=num_workers, pin_memory=device.type == "cuda"
        )
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    graph_indices: list[torch.Tensor] = []
    with torch.no_grad():
        for graph_number, batch in zip(graph_numbers, batches, strict=True):
            data = batch.to(device)
            with _autocast(device):
                output = _predict(model, data)
            mask = getattr(data, f"{split}_mask") if benchmark.transductive else None
            if mask is not None:
                output = output[mask]
                target = data.y[mask]
            else:
                target = data.y
            outputs.append(output)
            targets.append(target)
            graph_indices.append(torch.full((target.shape[0],), graph_number, device=device, dtype=torch.long))
    gathered = [torch.cat(values) for values in (outputs, targets, graph_indices)]
    del outputs, targets, graph_indices
    if world_size > 1:
        gathered = [_all_gather(values) for values in gathered]
    return compute_metrics(benchmark, *gathered)


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{name} {value:.4f}" for name, value in metrics.items())


def _describe_splits(benchmark: Benchmark) -> str:
    if benchmark.transductive:
        graph = benchmark.splits["train"][0]
        splits = [split for split in benchmark.splits if hasattr(graph, f"{split}_mask")]
        counts = "/".join(str(int(getattr(graph, f"{split}_mask").count_nonzero())) for split in splits)
        return f"one graph of {graph.num_nodes} nodes, {counts} {'/'.join(splits)} nodes"
    sizes = "/".join(str(len(cast(Sized, dataset))) for dataset in benchmark.splits.values())
    return f"{sizes} {'/'.join(benchmark.splits)} graphs"


def _summary_writer(root: Path | None, seed: int) -> AbstractContextManager[SummaryWriter | None]:
    """Return a TensorBoard writer for one seed under ``root``, or no writer without a root."""
    if root is None:
        return nullcontext()
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(str(root / f"seed{seed}"))


def _write_scalars(writer: SummaryWriter, namespace: str, metrics: Mapping[str, float], step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{namespace}/{name}", value, step)


def _log_epoch(
    phase: str,
    epoch: int,
    epochs: int,
    stats: Mapping[str, float],
    val_metrics: Mapping[str, float],
    note: str,
    writer: SummaryWriter | None,
) -> None:
    """Log one epoch as a line and, with a writer, as TensorBoard scalars."""
    parts = [
        f"{phase} epoch {epoch:>{len(str(epochs))}}/{epochs}",
        f"loss {stats['loss']:.4f}",
        f"lr {stats['lr']:.2e}",
    ]
    if val_metrics:
        parts.append(f"val {_format_metrics(val_metrics)}")
    if note:
        parts.append(note)
    parts.append(format_duration(stats["seconds"]))
    if "memory_gib" in stats:
        parts.append(f"{stats['memory_gib']:.1f} GiB")
    logger.info("  ".join(parts))
    if writer is not None:
        _write_scalars(writer, phase, stats, epoch)
        _write_scalars(writer, "val", val_metrics, epoch)


def _fit(
    benchmark: Benchmark,
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    include_val: bool,
    writer: SummaryWriter | None,
) -> tuple[GIST, dict[str, float] | None, int]:
    """Fit a model for ``epochs`` epochs and restore its best validation weights unless refitting.

    The learning-rate schedule always spans ``train.epochs``, so a shorter run follows its first part.
    ``train.lr`` is the rate for one graph per step: on inductive benchmarks it is multiplied by the number of
    graphs per step across processes.
    """
    model = _new_model(benchmark, config, device)
    _, world_size = _world()
    train_model = DistributedDataParallel(model) if world_size > 1 else model
    train_config = config["train"]
    metric = train_config["selection_metric"]
    graphs_per_step = 1 if benchmark.transductive else world_size * train_config["batch_size"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_config["lr"] * graphs_per_step,
        weight_decay=train_config["weight_decay"],
        fused=device.type == "cuda",
    )
    data_iter = _training_data(benchmark, config, device, include_val)
    scheduler = _scheduler(optimizer, config, len(data_iter))
    loss_fn = task_loss(benchmark.task)
    best_weights: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    selected_epoch = epochs
    stale_evaluations = 0
    patience = train_config["patience"]
    phase = "refit" if include_val else "train"
    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "%s: %d epochs of %d steps, %d graphs per step at lr %.2e, %.2fM parameters",
        phase,
        epochs,
        len(data_iter),
        graphs_per_step,
        train_config["lr"] * graphs_per_step,
        parameters / 1e6,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sampler = getattr(data_iter, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        for data in data_iter:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = _predict(train_model, data)
                if benchmark.transductive:
                    mask = data.train_mask
                    loss = loss_fn(output[mask], data.y[mask])
                else:
                    loss = loss_fn(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()

        if world_size > 1:
            dist.all_reduce(loss_sum)
        stats = {"loss": loss_sum.item() / (len(data_iter) * world_size), "lr": float(scheduler.get_last_lr()[0])}
        val_metrics: dict[str, float] = {}
        note = ""
        if not include_val and (epoch % train_config["eval_freq"] == 0 or epoch == epochs):
            val_metrics = _evaluate(model, benchmark, "val", device, train_config["num_workers"])
            score = val_metrics[metric]
            if not math.isfinite(score):
                raise ValueError(f"validation {metric} is {score} at epoch {epoch}")
            best_score = best_metrics[metric] if best_metrics is not None else None
            # Ties go to the later epoch: on a flat validation curve the earliest best often refits too short.
            if score == best_score or _is_better(metric, score, best_score):
                best_metrics = val_metrics
                best_weights = {key: value.clone() for key, value in model.state_dict().items()}
                selected_epoch = epoch
                stale_evaluations = 0
                note = "best"
            else:
                stale_evaluations += 1
                note = f"no improvement {stale_evaluations}" + (f"/{patience}" if patience is not None else "")
        stats["seconds"] = time.perf_counter() - start
        if device.type == "cuda":
            stats["memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        _log_epoch(phase, epoch, epochs, stats, val_metrics, note, writer)
        if patience is not None and stale_evaluations >= patience and stale_evaluations > 0:
            logger.info("Stopping early: no val %s improvement in %d evaluations", metric, stale_evaluations)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_metrics, selected_epoch


def _summarize(per_seed: list[dict[str, Any]], split: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-metric mean and population std."""
    metrics = [entry[split] for entry in per_seed]
    names = metrics[0]
    mean = {name: sum(values[name] for values in metrics) / len(metrics) for name in names}
    std = {
        name: math.sqrt(sum((values[name] - mean[name]) ** 2 for values in metrics) / len(metrics)) for name in names
    }
    return mean, std


def _load(config: Mapping[str, Any], split_seed: int) -> Benchmark:
    return load_dataset(
        name=config["dataset"],
        root=config["root"],
        embed_dim=config["embedding"]["embed_dim"],
        deg=config["embedding"]["deg"],
        split_seed=split_seed,
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train one model per seed and summarize validation and test metrics.

    Args:
        config: Experiment settings, with defaults filled when omitted.

    Returns:
        JSON-serializable per-seed metrics and aggregate statistics.
    """
    resolved = resolve_config(config)
    benchmark = _load(resolved, resolved["seeds"][0])
    metric = resolved["train"]["selection_metric"] or DEFAULT_SELECTION_METRIC[benchmark.task]
    allowed = {
        "multiclass": {"accuracy"},
        "multilabel": {"micro_f1"},
        "regression": {"mse", "rel_l2"},
    }
    if metric not in allowed[benchmark.task]:
        raise ValueError(metric)
    resolved["train"]["selection_metric"] = metric
    device = _init_distributed()
    rank, world_size = _world()
    if world_size > 1 and (
        benchmark.transductive or any(len(cast(Sized, dataset)) < world_size for dataset in benchmark.splits.values())
    ):
        reason = "several processes need an inductive benchmark with at least one graph per process in every split"
        raise ValueError(reason)
    if rank > 0:
        logging.getLogger().setLevel(logging.WARNING)
    train_config = resolved["train"]
    logger.info(
        "%s, %s: %s; %d inputs, %d outputs per node",
        benchmark.name,
        benchmark.task,
        _describe_splits(benchmark),
        benchmark.num_features,
        benchmark.num_targets,
    )
    logger.info(
        "%d x %s, %s%s; select on val %s every %d epochs, patience %s",
        world_size,
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "bf16" if device.type == "cuda" else "fp32",
        ", compiled" if train_config["compile"] else "",
        metric,
        train_config["eval_freq"],
        train_config["patience"],
    )
    tensorboard_root = None
    if resolved["tensorboard_dir"] is not None and rank == 0:
        tensorboard_root = Path(resolved["tensorboard_dir"]) / f"{benchmark.name}_{datetime.now():%Y%m%d_%H%M%S}"
        logger.info("TensorBoard logs in %s", tensorboard_root)
    per_seed: list[dict[str, Any]] = []
    for number, seed in enumerate(resolved["seeds"], 1):
        logger.info("Seed %d (%d of %d)", seed, number, len(resolved["seeds"]))
        start = time.perf_counter()
        if benchmark.seeded_split and number > 1:
            benchmark = _load(resolved, seed)
        with _summary_writer(tensorboard_root, seed) as writer:
            torch.manual_seed(seed)
            model, val_metrics, selected_epoch = _fit(
                benchmark, resolved, device, train_config["epochs"], include_val=False, writer=writer
            )
            if train_config["final_training"]:
                torch.manual_seed(seed)
                del model
                model, _, _ = _fit(benchmark, resolved, device, selected_epoch, include_val=True, writer=writer)
            test_metrics = _evaluate(model, benchmark, "test", device, train_config["num_workers"])
            if writer is not None:
                _write_scalars(writer, "test", test_metrics, selected_epoch)
        logger.info(
            "Seed %d: selected epoch %d, test %s, %s",
            seed,
            selected_epoch,
            _format_metrics(test_metrics),
            format_duration(time.perf_counter() - start),
        )
        per_seed.append({"seed": seed, "selected_epoch": selected_epoch, "val": val_metrics, "test": test_metrics})
        del model
    val_mean, val_std = _summarize(per_seed, "val")
    test_mean, test_std = _summarize(per_seed, "test")
    return {
        "config": resolved,
        "per_seed": per_seed,
        "mean": {"val": val_mean, "test": test_mean},
        "std": {"val": val_std, "test": test_std},
    }


def _write_result(result: Mapping[str, Any], output: str | None) -> Path:
    """Return the result path, writing the JSON only on rank zero."""
    if output is None:
        config = result["config"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{config['dataset']}_{timestamp}.json"
        path = Path(config["output_dir"]) / label
    else:
        path = Path(output)
    if _world()[0] == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _log_summary(result: Mapping[str, Any]) -> None:
    logger.info("%-12s %20s %20s", "Metric", "Validation mean ± std", "Test mean ± std")
    val_mean, val_std = result["mean"]["val"], result["std"]["val"]
    test_mean, test_std = result["mean"]["test"], result["std"]["test"]
    for metric in test_mean:
        validation = f"{val_mean[metric]:.4f} ± {val_std[metric]:.4f}"
        test = f"{test_mean[metric]:.4f} ± {test_std[metric]:.4f}"
        logger.info("%-12s %20s %20s", metric, validation, test)


def main() -> None:
    """Run a configured experiment and write its metrics as JSON."""
    parser = argparse.ArgumentParser(description="Train and evaluate GIST")
    parser.add_argument("config", help="YAML configuration path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    result = run(config)
    _log_summary(result)
    logger.info("Results written to %s", _write_result(result, args.output))


if __name__ == "__main__":
    configure_logging()
    main()
