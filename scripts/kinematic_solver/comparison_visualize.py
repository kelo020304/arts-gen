"""Comparison-side per-direction predicted and reference PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import CollisionBackend
from .manual_transform import apply_joint_transform_world_baked
from .render import render_backend_frame


def _render_at_q(
    *,
    backend: CollisionBackend,
    joint: dict,
    q_signed: float,
    out_path: Path,
) -> None:
    direction = 1 if q_signed >= 0 else -1
    rotation, translation = apply_joint_transform_world_baked(
        joint_type=joint["type"],
        direction=direction,
        q_abs=abs(float(q_signed)),
        axis_world=np.asarray(joint["axis_world"], dtype=np.float64),
        origin_world=np.asarray(joint["origin_world"], dtype=np.float64),
    )
    for part in joint["moving_parts"]:
        backend.set_pose(part, rotation, translation)
    render_backend_frame(backend, out_path)


def write_per_direction_overlays(
    *,
    backend: CollisionBackend,
    joint: dict,
    prediction: dict,
    gt: dict,
    out_dir: Path,
) -> list[Path]:
    """Write separate predicted/reference PNGs for each successful direction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for side, status_key, pred_key, gt_key in [
        ("upper", "status_upper", "predicted_upper", "upper"),
        ("lower", "status_lower", "predicted_lower", "lower"),
    ]:
        if prediction.get(status_key) != "ok":
            continue
        pred_path = out_dir / f"pred_{side}.png"
        gt_path = out_dir / f"gt_{side}.png"
        _render_at_q(
            backend=backend,
            joint=joint,
            q_signed=float(prediction[pred_key]),
            out_path=pred_path,
        )
        _render_at_q(
            backend=backend,
            joint=joint,
            q_signed=float(gt[gt_key]),
            out_path=gt_path,
        )
        written.extend([pred_path, gt_path])
    return written
