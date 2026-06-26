"""V1 KinematicSolver constants and dataclass configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

V1_CONDA_PYTHON: Path = Path("/home/mi/anaconda3/envs/env-isaacsim/bin/python")
V1_PINNED_COACD_VERSION: str = "1.0.9"

V1_TEN_IDS: list[str] = [
    "ra_007", "ra_017", "ra_027", "ra_037", "ra_047",
    "ra_057", "ra_067", "ra_077", "ra_087", "ra_097",
]

V1_VHACD_CACHE_METADATA: Mapping[str, str] = {
    "backend": "coacd",
    "version": V1_PINNED_COACD_VERSION,
}

V1_COACD_RUN_PARAMS: Mapping[str, object] = {
    "threshold": 0.05,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "mcts_nodes": 20,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 256,
    "extrude": False,
    "extrude_margin": 0.01,
    "apx_mode": "ch",
    "seed": 0,
    "max_convex_hull": -1,
    "real_metric": True,
}


@dataclass(frozen=True)
class V1DatasetRoots:
    converter_output_root: Path = Path("data/RealAppliance-4view-0515-baked")
    source_root: Path = Path("data/RealAppliance")

    def aligned_usd_for(self, object_id: str) -> Path:
        source_id = object_id.removeprefix("ra_")
        return self.source_root / f"source/model/{source_id}/Aligned.usd"


@dataclass(frozen=True)
class SearchConfig:
    prismatic_step_m: float = 0.01
    revolute_step_rad: float = math.radians(2.0)
    initial_high_prismatic_m: float = 0.5
    initial_high_revolute_rad: float = math.pi
    allow_initial_penetration: bool = False
    viz_stride: int = 5


@dataclass(frozen=True)
class CollisionConstraintConfig:
    allow_initial_penetration: bool = False


@dataclass(frozen=True)
class ComparisonConfig:
    success_rel_err_threshold: float = 0.10
