"""Collision backend protocol and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class CollisionBackend(Protocol):
    def load_model(
        self,
        *,
        object_id: str,
        part_to_obj_path: dict[str, Path],
        vhacd_cache_root: Path,
        coacd_run_params: dict,
        vhacd_cache_metadata: dict,
    ) -> None:
        ...

    def set_pose(self, part_name: str, rotation: np.ndarray, translation: np.ndarray) -> None:
        ...

    def reset_to_identity(self) -> None:
        ...

    def overlap(self, moving_parts: list[str], static_parts: list[str]) -> bool:
        ...

    def clear(self) -> None:
        ...


def make_backend(spike_result: Path | None = None) -> CollisionBackend:
    """V1 local default: use FCL unless a later Isaac slice overrides this."""
    from ._fcl_backend import FclBackend

    return FclBackend()
