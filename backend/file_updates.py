"""Replace producer outputs without modifying any retained shared file body."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def atomic_destination(destination: Path) -> Iterator[Path]:
    """Stage a complete file beside its destination, then replace the name.

    Do not resolve the destination: it may be a compatibility symlink to an
    immutable body shared by other registered files.
    """
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=destination.parent)
    os.close(fd)
    pending = Path(name)
    try:
        yield pending
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
