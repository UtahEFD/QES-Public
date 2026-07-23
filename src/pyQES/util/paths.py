"""Filesystem helpers for pyQES run orchestration."""

from __future__ import annotations

import tempfile
from pathlib import Path

__all__ = ["resolve_work_dir", "resolve_path"]


def resolve_work_dir(work_dir: str | Path | None) -> Path:
    """Return an existing working directory, creating a temp one if needed."""
    if work_dir is None:
        return Path(tempfile.mkdtemp(prefix="pyqes_"))
    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve ``path`` to an absolute path, optionally relative to ``base``."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None:
        return (Path(base) / p).resolve()
    return p.resolve()
