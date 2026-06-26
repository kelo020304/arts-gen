"""Candidate range visualization for estimate_limit.py runs."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from post_process.kinematic_solver.utils.backend import make_backend
from post_process.kinematic_solver.utils.config import (
    V1_COACD_RUN_PARAMS,
    V1_VHACD_CACHE_METADATA,
)
from post_process.kinematic_solver.utils.visualize import visualize_one_joint

from .schemas import EstimateContext, LimitEstimate


def write_estimate_viewers(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
    *,
    converter_output_root: Path,
    out_dir: Path,
    spike_result: Path | None = None,
    viz_steps: int = 16,
) -> dict:
    estimates_by_name = {
        estimate.joint_name: estimate
        for estimate in estimates
        if estimate.joint_name in ctx.joints
    }
    if not estimates_by_name:
        return {"object_viewers": []}

    _copy_viewer_vendor(out_dir)
    backend = make_backend(spike_result)
    backend.load_model(
        object_id=ctx.object_id,
        part_to_obj_path=_part_to_obj_paths(converter_output_root, ctx.object_id),
        vhacd_cache_root=converter_output_root / f"raw/vhacd/{ctx.object_id}",
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
    )
    viewers = []
    try:
        for joint_name, estimate in sorted(estimates_by_name.items()):
            joint = dict(ctx.joints[joint_name])
            joint.setdefault("object_id", ctx.object_id)
            joint.setdefault("joint_name", joint_name)
            if estimate.axis_world is not None:
                joint["axis_world"] = [float(value) for value in estimate.axis_world]
            prediction = _prediction_payload(ctx, joint_name, joint, estimate)
            trace = _trace_payload(ctx, joint_name, estimate, viz_steps=viz_steps)
            viz_dir = out_dir / ctx.object_id / "agent_viz" / joint_name
            backend.reset_to_identity()
            visualize_one_joint(
                backend=backend,
                joint=joint,
                prediction=prediction,
                trace=trace,
                out_dir=viz_dir,
                viz_stride=1,
                render_frames=False,
            )
            viewers.append({
                "joint_name": joint_name,
                "href": f"{ctx.object_id}/agent_viz/{joint_name}/step_viewer.html",
                "lower": float(estimate.lower),
                "upper": float(estimate.upper),
                **({"axis_label": estimate.axis_label} if estimate.axis_label else {}),
            })
    finally:
        backend.clear()
    return {"object_viewers": viewers}


def _part_to_obj_paths(converter_output_root: Path, object_id: str) -> dict[str, Path]:
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        path.stem: path
        for path in sorted(obj_dir.glob("*.obj"))
        if path.stem == "body" or path.stem.startswith("part_")
    }


def _prediction_payload(
    ctx: EstimateContext,
    joint_name: str,
    joint: dict,
    estimate: LimitEstimate,
) -> dict:
    payload = {
        "object_id": ctx.object_id,
        "joint_name": joint_name,
        "type": joint.get("type"),
        "canonical_unit": joint.get("canonical_unit"),
        "predicted_lower": float(estimate.lower),
        "predicted_upper": float(estimate.upper),
        "predicted_axis_world": (
            [float(value) for value in estimate.axis_world]
            if estimate.axis_world is not None
            else None
        ),
        "predicted_axis_label": estimate.axis_label,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }
    estimate_payload = asdict(estimate)
    payload["reason"] = estimate_payload.get("reason", "")
    payload["confidence"] = estimate_payload.get("confidence")
    return payload


def _trace_payload(
    ctx: EstimateContext,
    joint_name: str,
    estimate: LimitEstimate,
    *,
    viz_steps: int,
) -> dict:
    samples = [
        {"q": q, "valid": True}
        for q in _range_samples(float(estimate.lower), float(estimate.upper), viz_steps)
    ]
    return {
        "object_id": ctx.object_id,
        "joint_name": joint_name,
        "trace_upper": samples,
        "trace_lower": [],
    }


def _range_samples(lower: float, upper: float, viz_steps: int) -> list[float]:
    steps = max(int(viz_steps), 2)
    if lower == upper:
        return [lower]
    return [
        lower + (upper - lower) * idx / (steps - 1)
        for idx in range(steps)
    ]


def _copy_viewer_vendor(out_dir: Path) -> None:
    vendor_src = (
        Path(__file__).resolve().parents[3]
        / "submodules/dataset_toolkits/vendor/three/0.160.0/classic/three.min.js"
    )
    if not vendor_src.is_file():
        return
    vendor_dst = out_dir / "vendor/three.min.js"
    vendor_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor_src, vendor_dst)
