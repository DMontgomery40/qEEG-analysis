from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

OnEvent = Callable[[dict[str, Any]], Awaitable[None]] | None


@dataclass(frozen=True)
class StageDef:
    num: int
    name: str
    kind: str
    content_type: str
    ext: str


@dataclass(frozen=True)
class PageImage:
    page: int
    base64_png: str
    label: str | None = None

