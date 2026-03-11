from __future__ import annotations

import contextlib
import logging
import os
import uuid
from typing import Any, Iterator

import structlog
from structlog.contextvars import (
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    get_contextvars,
    merge_contextvars,
)


def _level_name() -> str:
    return (os.getenv("QEEG_LOG_LEVEL", "INFO") or "INFO").upper()


def configure_logging() -> structlog.stdlib.BoundLogger:
    if getattr(configure_logging, "_configured", False):
        return structlog.get_logger("backend")

    level_name = _level_name()
    level = getattr(logging, level_name, logging.INFO)
    backend_logger = logging.getLogger("backend")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    backend_logger.handlers.clear()
    backend_logger.addHandler(handler)
    backend_logger.setLevel(level)
    backend_logger.propagate = False

    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    configure_logging._configured = True  # type: ignore[attr-defined]
    return structlog.get_logger("backend")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex


def current_log_context() -> dict[str, Any]:
    return dict(get_contextvars())


@contextlib.contextmanager
def log_context(**values: Any) -> Iterator[dict[str, Any]]:
    cleaned = {key: value for key, value in values.items() if value is not None}
    with bound_contextvars(**cleaned):
        yield current_log_context()


def reset_log_context() -> None:
    clear_contextvars()


def bind_log_context(**values: Any) -> None:
    cleaned = {key: value for key, value in values.items() if value is not None}
    if cleaned:
        bind_contextvars(**cleaned)
