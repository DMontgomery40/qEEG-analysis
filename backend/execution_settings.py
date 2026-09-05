"""Task-local, immutable effective environment for opt-in owned execution."""

from contextvars import ContextVar
import os
from typing import Mapping

settings: ContextVar[Mapping[str, str] | None] = ContextVar(
    "qeeg_execution_settings", default=None
)


def execution_getenv(name: str, default=None):
    frozen = settings.get()
    if frozen is not None and (
        name.startswith("QEEG_")
        or name
        in {
            "OPENAI_REASONING_EFFORT",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_APP_TITLE",
        }
    ):
        return frozen.get(name, default)
    return os.getenv(name, default)
