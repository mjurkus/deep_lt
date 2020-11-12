import os
from pathlib import Path


def to_absolute_path(path: str) -> str:
    path = Path(path)

    if path.is_absolute():
        return path

    base = Path(os.getcwd())
    return str(base / path)
