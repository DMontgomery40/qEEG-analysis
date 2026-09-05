"""Coordinated release switch: only original confirmed intent uses E6."""

import os


def shared_execution_enabled():
    return os.getenv("QEEG_CLINIC_SHARED_EXECUTION", "0").lower().strip() in (
        "1",
        "true",
        "yes",
        "on",
    )
