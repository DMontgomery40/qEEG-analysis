"""Keep standalone tools from bypassing admission in the installed clinic tree."""
from pathlib import Path


def refuse_live_portal_write(*paths: Path) -> None:
    # This is the installed clinic tree, not DATA_DIR: scratch portal trees remain
    # writable, and changing a tool's DATA_DIR must not bypass the live guard.
    live_root = (Path.home() / "qEEG-analysis/data/portal_patients").resolve()
    for path in paths:
        path = Path(path).expanduser()
        # Atomic replacement changes the directory entry, even when an existing
        # symlink points outside the live tree. Check both entry and target.
        entry = path.parent.resolve() / path.name
        if entry.is_relative_to(live_root) or path.resolve().is_relative_to(live_root):
            raise SystemExit(
                "Direct writes to the live portal are disabled; use the engine's "
                "catalogue admission path (/api/clinic/internal/artifacts) "
                "or choose an output directory outside the live portal."
            )
