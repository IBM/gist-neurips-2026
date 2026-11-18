"""Log format and progress reporting."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Log INFO messages to stdout with the time of day."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)


def format_duration(seconds: float) -> str:
    """Return a duration in seconds, minutes, or hours."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def log_progress(items: Iterable[T], total: int, label: str) -> Iterator[T]:
    """Yield items, logging the count, time per item, and time left at about every tenth of the total."""
    start = time.perf_counter()
    step = max(1, total // 10)
    for done, item in enumerate(items, 1):
        yield item
        if done % step == 0 or done == total:
            each = (time.perf_counter() - start) / done
            logger.info(
                "%s: %d/%d, %.2f s each, %s left", label, done, total, each, format_duration(each * (total - done))
            )
