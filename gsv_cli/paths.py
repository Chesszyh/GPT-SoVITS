from __future__ import annotations

from pathlib import Path


def project_path(root: str, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path
