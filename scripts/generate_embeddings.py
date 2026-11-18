"""Generate cached graph embeddings for a benchmark."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gist.datasets import load_dataset
from gist.logs import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate FastRP embeddings for every graph split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Benchmark name")
    parser.add_argument("--root", default="data", help="Parent directory for benchmark data")
    parser.add_argument("--embed_dim", type=int, required=True, help="Embedding dimension")
    parser.add_argument("--deg", type=int, required=True, help="Number of diffusion powers")
    args = parser.parse_args()

    configure_logging()
    benchmark = load_dataset(
        args.name,
        root=args.root,
        embed_dim=args.embed_dim,
        deg=args.deg,
        generate_embeddings=True,
    )
    for split, dataset in benchmark.splits.items():
        logger.info("%s: %d graphs", split, len(dataset))


if __name__ == "__main__":
    main()
