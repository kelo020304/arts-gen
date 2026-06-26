"""Validation dispatch and Isaac/FCL fallback checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CollisionConstraintConfig
from .errors import InvalidValidationContextError, SchemaMismatchError


@dataclass
class ValidationContext:
    prediction: dict
    vlm_oracle_model: dict
    joint_name: str
    object_id: str
    usd_path: Path
    predicted_usd_path: Path | None
    part_to_obj_path: dict[str, Path]
    vhacd_cache_root: Path
    coacd_run_params: dict
    vhacd_cache_metadata: dict
    stage_metadata: dict
    thresholds: dict = field(default_factory=lambda: {
        "force_N": 50.0,
        "torque_Nm": 100.0,
        "reach_rel": 0.05,
    })


def _isaac_runtime_available() -> bool:
    try:
        return importlib.util.find_spec("omni.isaac.kit") is not None
    except (ImportError, ValueError):
        return False


def _geometry_sanity_validate(ctx: ValidationContext) -> dict:
    """Fallback validation: check predicted endpoints remain collision-free."""
    from ._fcl_backend import FclBackend
    from .constraints import CollisionConstraint
    from .joint_evaluator import JointEvaluator

    joint = ctx.vlm_oracle_model["joints"][ctx.joint_name]
    backend = FclBackend()
    backend.load_model(
        object_id=ctx.object_id,
        part_to_obj_path=ctx.part_to_obj_path,
        vhacd_cache_root=ctx.vhacd_cache_root,
        coacd_run_params=ctx.coacd_run_params,
        vhacd_cache_metadata=ctx.vhacd_cache_metadata,
    )
    evaluator = JointEvaluator(
        joint=joint,
        constraints=[
            CollisionConstraint(
                list(joint["moving_parts"]),
                list(joint["static_parts"]),
                backend=backend,
                config=CollisionConstraintConfig(allow_initial_penetration=True),
            )
        ],
        backend=backend,
    )
    evaluator.calibrate_at_zero()
    lower_ok = bool(evaluator(float(ctx.prediction["predicted_lower"])))
    upper_ok = bool(evaluator(float(ctx.prediction["predicted_upper"])))
    backend.clear()
    return {
        "object_id": ctx.object_id,
        "joint_name": ctx.joint_name,
        "validation_status": "skipped_backend_unavailable",
        "reason": "omni.isaac.kit unavailable; used geometry endpoint sanity",
        "predicted_lower": ctx.prediction["predicted_lower"],
        "predicted_upper": ctx.prediction["predicted_upper"],
        "geometry_overlap_at_lower": not lower_ok,
        "geometry_overlap_at_upper": not upper_ok,
        "sanity_passed": lower_ok and upper_ok,
    }


def _articulation_pd_validate(ctx: ValidationContext) -> dict:
    """Run a headless Isaac articulation sweep across the predicted interval."""
    from omni.isaac.kit import SimulationApp

    sim_app = SimulationApp({"headless": True})
    try:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        add_reference_to_stage(
            usd_path=str(ctx.predicted_usd_path),
            prim_path=f"/World/{ctx.object_id}",
        )
        articulation = Articulation(prim_path=f"/World/{ctx.object_id}")
        world.scene.add(articulation)
        world.reset()
        articulation.initialize()

        joint_prim_path = ctx.stage_metadata["joint_prim_paths"][ctx.joint_name]
        joint_idx = None
        for idx, name in enumerate(getattr(articulation, "dof_names", [])):
            if name in joint_prim_path:
                joint_idx = idx
                break
        if joint_idx is None:
            raise SchemaMismatchError(
                f"{ctx.object_id}/{ctx.joint_name}: no articulation DOF matches "
                f"joint prim path {joint_prim_path!r}"
            )

        q_lower = float(ctx.prediction["predicted_lower"])
        q_upper = float(ctx.prediction["predicted_upper"])
        max_torque = 0.0
        reached_q = q_lower
        for q_value in np.linspace(q_lower, q_upper, 50):
            action = ArticulationAction(
                joint_positions=np.array([q_value], dtype=float),
                joint_indices=np.array([joint_idx], dtype=int),
            )
            articulation.apply_action(action)
            for _ in range(5):
                world.step(render=False)
            efforts = np.asarray(articulation.get_applied_joint_efforts(), dtype=float)
            if efforts.size:
                max_torque = max(max_torque, float(abs(efforts[joint_idx])))
            positions = np.asarray(articulation.get_joint_positions(), dtype=float)
            if positions.size:
                reached_q = float(positions[joint_idx])

        reach_error = abs(reached_q - q_upper) / max(abs(q_upper - q_lower), 1e-6)
        failed = []
        if max_torque > float(ctx.thresholds["torque_Nm"]):
            failed.append("torque_Nm")
        if reach_error > float(ctx.thresholds["reach_rel"]):
            failed.append("reach_rel")
        result = {
            "object_id": ctx.object_id,
            "joint_name": ctx.joint_name,
            "validation_status": "failed" if failed else "passed",
            "predicted_lower": q_lower,
            "predicted_upper": q_upper,
            "max_contact_force_N": 0.0,
            "max_torque_Nm": max_torque,
            "reach_error_rel": reach_error,
            "thresholds": dict(ctx.thresholds),
        }
        if failed:
            result["failed_thresholds"] = failed
        return result
    finally:
        sim_app.close()


def validate_joint(ctx: ValidationContext) -> dict:
    prediction = ctx.prediction
    if prediction["status"] != "ok":
        if ctx.predicted_usd_path is not None:
            raise InvalidValidationContextError(
                "non-ok prediction must not have predicted_usd_path"
            )
        return {
            "object_id": ctx.object_id,
            "joint_name": ctx.joint_name,
            "validation_status": "skipped_non_ok",
            "reason": f"prediction status={prediction['status']}",
        }
    if ctx.predicted_usd_path is None:
        raise InvalidValidationContextError(
            "ok prediction requires predicted_usd_path before validation"
        )
    if not _isaac_runtime_available():
        return _geometry_sanity_validate(ctx)
    return _articulation_pd_validate(ctx)
