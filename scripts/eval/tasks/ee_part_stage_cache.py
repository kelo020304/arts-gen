#!/usr/bin/env python3
"""Cache contract for the ee-eval promptable part stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PART_STAGE_SIGNATURE_NAME = "part_stage_signature.json"
PART_STAGE_SIGNATURE_SCHEMA = "arts-gen.ee.part-stage.v1"


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_part_stage_signature(
    *,
    part_seg_ckpt: str | Path,
    ss_latent_path: str | Path,
    voxel_path: str | Path,
    expected_parts: int,
    joint_candidate_mode: str,
    joint_refine: bool,
    joint_refine_iters: int,
    joint_refine_pairwise: float,
    joint_refine_margin: float,
    joint_refine_margin_quantile: float,
    joint_refine_neighborhood: int,
    joint_refine_min_vote_gain: float,
    joint_refine_preserve_small_classes: int,
    joint_save_logits: bool,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "schema": PART_STAGE_SIGNATURE_SCHEMA,
        "backend": "promptable_seg",
        "decode_backend": "trellis",
        "part_seg_ckpt": _file_identity(part_seg_ckpt),
        "inputs": {
            "ss_latent": _file_identity(ss_latent_path),
            "whole_voxel": _file_identity(voxel_path),
        },
        "expected_parts": int(expected_parts),
        "seed": None if seed is None else int(seed),
        "joint_partition": {
            "candidate_mode": str(joint_candidate_mode),
            "refine": bool(joint_refine),
            "refine_iters": int(joint_refine_iters),
            "refine_pairwise": float(joint_refine_pairwise),
            "refine_margin": float(joint_refine_margin),
            "refine_margin_quantile": float(joint_refine_margin_quantile),
            "refine_neighborhood": int(joint_refine_neighborhood),
            "refine_min_vote_gain": float(joint_refine_min_vote_gain),
            "refine_preserve_small_classes": int(joint_refine_preserve_small_classes),
            "save_logits": bool(joint_save_logits),
        },
    }


def part_stage_cache_status(
    parts_dir: str | Path,
    *,
    expected_parts: int,
    expected_signature: dict[str, Any],
) -> tuple[bool, str]:
    output_ok, output_reason = part_stage_outputs_status(
        parts_dir,
        expected_parts=expected_parts,
        require_joint_logits=bool(
            (expected_signature.get("joint_partition") or {}).get("save_logits", False)
        ),
    )
    if not output_ok:
        return False, output_reason

    parts_dir = Path(parts_dir)
    signature_path = parts_dir / PART_STAGE_SIGNATURE_NAME
    if not signature_path.is_file():
        return False, f"missing {PART_STAGE_SIGNATURE_NAME}"
    try:
        actual_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid {PART_STAGE_SIGNATURE_NAME}: {exc}"
    if actual_signature != expected_signature:
        return False, "part stage signature mismatch"
    return True, "signature and outputs match"


def part_stage_outputs_status(
    parts_dir: str | Path,
    *,
    expected_parts: int,
    require_joint_logits: bool,
) -> tuple[bool, str]:
    parts_dir = Path(parts_dir)
    if int(expected_parts) <= 0:
        return False, "expected_parts must be positive"
    missing = [
        path.name
        for path in (parts_dir / f"part_{index:02d}_voxel.npz" for index in range(int(expected_parts)))
        if not path.is_file()
    ]
    if missing:
        return False, f"missing part voxel outputs: {missing[:5]}"
    if bool(require_joint_logits) and not (parts_dir / "joint_partition.npz").is_file():
        return False, "joint_partition.npz was requested but is missing"
    return True, "required part outputs exist"


def clear_part_stage_outputs(parts_dir: str | Path) -> list[str]:
    parts_dir = Path(parts_dir)
    removed: list[str] = []
    if not parts_dir.is_dir():
        return removed
    for pattern in ("part_*_latent.npy", "part_*_meta.json", "part_*_voxel.npz"):
        for path in parts_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(path.name)
    for name in ("joint_partition.npz", PART_STAGE_SIGNATURE_NAME):
        path = parts_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed.append(path.name)
    return sorted(removed)
