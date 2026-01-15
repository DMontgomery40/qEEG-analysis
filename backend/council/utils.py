from __future__ import annotations

import asyncio
import os
import random
from typing import Any


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


async def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter; attempt starts at 0.
    base = min(8.0, 0.5 * (2**attempt))
    jitter = random.random() * 0.2
    await asyncio.sleep(base + jitter)

