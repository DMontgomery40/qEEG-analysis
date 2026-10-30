"""Replace producer outputs without modifying any retained shared file body."""

from contextlib import contextmanager
import os
from pathlib import Path
import uuid
from typing import Iterator


@contextmanager
def atomic_destination(destination: Path) -> Iterator[Path]:
    """Stage a complete file beside its destination, then replace the name.

    Do not resolve the destination: it may be a compatibility symlink to an
    immutable body shared by other registered files.
    """
    destination = Path(destination)
    # mkstemp would create the output owner-only (0600); keep the umask default
    # a direct write produced. Never copy the destination's mode: it may be a
    # link to a read-only shared body.
    pending = destination.parent / f".pending-{uuid.uuid4().hex}"
    os.close(os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
    try:
        yield pending
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
