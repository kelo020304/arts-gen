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
        coordinate_transform: str | None = None,
    ) -> None:
        ...

    def set_pose(self, part_name: str, rotation: np.ndarray, translation: np.ndarray) -> None:
        ...

    def reset_to_identity(self) -> None:
        ...

    def overlap(self, moving_parts: list[str], static_parts: list[str]) -> bool:
        ...

    def overlapping_pairs(
        self,
        moving_parts: list[str],
        static_parts: list[str],
    ) -> list[tuple[str, str]]:
        ...

    def load_exact_meshes(
        self,
        *,
        part_to_obj_path: dict[str, Path],
        coordinate_transform: str | None = None,
    ) -> None:
        ...

    def exact_overlapping_pairs(
        self,
        moving_parts: list[str],
        static_parts: list[str],
    ) -> list[tuple[str, str]]:
        ...

    def clear(self) -> None:
        ...


def make_backend(spike_result: Path | None = None) -> CollisionBackend:
    """V1 local default: use FCL unless a later Isaac slice overrides this."""
    from ._fcl_backend import FclBackend

    return FclBackend()
