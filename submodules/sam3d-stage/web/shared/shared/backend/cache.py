from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

Scalar = Union[bytes, str, int, float]


def compute_sha(parts: dict[str, Scalar]) -> str:
    """SHA-256 over a canonicalized dict.

    bytes -> raw; str/int/float -> utf-8 of repr.
    Sort keys for determinism. Return hex digest, truncated to 16 chars.
    """
    h = hashlib.sha256()
    for key in sorted(parts.keys()):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        value = parts[key]
        if isinstance(value, bytes):
            h.update(value)
        else:
            h.update(repr(value).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


class JobDirs:
    """Convenience for a cache layout: ``<base>/<sha>/{input,output}/``."""

    def __init__(self, base_dir: Path, sha: str) -> None:
        self._root = Path(base_dir) / sha

    @property
    def root(self) -> Path:
        return self._root

    @property
    def input_dir(self) -> Path:
        return self._root / "input"

    @property
    def output_dir(self) -> Path:
        return self._root / "output"

    def exists(self) -> bool:
        """True iff the output directory exists and is non-empty."""
        out = self.output_dir
        return out.is_dir() and any(out.iterdir())

    def ensure_input(self) -> Path:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        return self.input_dir

    def ensure_output(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
