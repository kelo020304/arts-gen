"""Run the GT-free kinematic agent on decoded body/part OBJ geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .sdk import (
    KinematicAgentConfig,
    delivery_joint_payload,
    export_decoded_mesh_obj,
    infer_kinematics,
    load_mesh_points,
    write_kinematic_bundle_mjcf,
    write_kinematic_bundle_usda,
)


def _semantic_type_hint(label: str) -> str | None:
    """Use predicted/user-visible part semantics, never dataset joint fields."""
    normalized = label.lower().replace("-", "_")
    if "grind_amount_dial" in normalized:
        return "prismatic"
    if any(token in normalized for token in (
        "glass_lid", "pitcher_lid", "jar_lid", "dual_opening_lid", "removable_vent_cover",
        "玻璃盖", "盖子",
    )):
        return "prismatic"
    if any(token in normalized for token in (
        "top_cover", "front_cover", "top_lid", "spin_dryer_lid", "外盖", "上盖",
    )):
        return "revolute"
    if any(token in normalized for token in ("drawer", "slider", "tray", "shelf", "抽屉", "拉篮", "滑轨", "托盘", "炸桶")):
        return "prismatic"
    if any(token in normalized for token in ("door", "knob", "dial", "handle", "hinge", "门", "旋钮", "把手", "铰链")):
        return "revolute"
    return None


def _semantic_part_category(label: str) -> str | None:
    normalized = label.lower().replace("-", "_")
    if any(token in normalized for token in ("drawer", "slider", "tray", "shelf", "抽屉", "拉篮", "滑轨", "托盘", "炸桶")):
        return "drawer"
    if any(token in normalized for token in ("knob", "dial", "旋钮")):
        return "knob"
    if any(token in normalized for token in ("door", "门")):
        return "door"
    if any(token in normalized for token in ("lid", "cover", "盖")):
        return "lid"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-obj", type=Path, required=True)
    parser.add_argument("--moving-obj", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--object-name", default="decoded_object")
    parser.add_argument("--joint-name", default="joint_0")
    parser.add_argument("--joint-type-hint", choices=("prismatic", "revolute"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--max-iterations", type=int, default=7)
    args = parser.parse_args()

    joint_type_hint = args.joint_type_hint or _semantic_type_hint(args.moving_obj.stem)
    result = infer_kinematics(
        load_mesh_points(args.moving_obj),
        load_mesh_points(args.body_obj),
        config=KinematicAgentConfig(max_iterations=args.max_iterations),
        joint_type_hint=joint_type_hint,
        part_category=_semantic_part_category(args.moving_obj.stem),
        dataset_profile=args.dataset_id,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.out_dir / "assets"
    body_obj = export_decoded_mesh_obj(args.body_obj, assets_dir / "body.obj")
    moving_obj = export_decoded_mesh_obj(args.moving_obj, assets_dir / "moving_part.obj")
    force_prismatic_local_z = str(args.dataset_id or "").lower() == "realappliance"
    apply_root_correction = force_prismatic_local_z
    delivery = delivery_joint_payload(result.candidate, force_prismatic_local_z)
    (args.out_dir / "kinematic_result.json").write_text(
        json.dumps({
            "format": "arts_gen_kin_agent_v11",
            "input_contract": (
                "decoded SLat meshes plus optional calibrated multi-state 2D boxes/cameras; "
                "no GT mesh, joint annotations, joint transforms, or source USD joint fields"
            ),
            "dataset_id": args.dataset_id,
            "iterations": result.iterations,
            "candidate": asdict(result.candidate),
            "delivery_candidate": delivery,
            "trace": result.trace,
            "evaluation_scope": {
                "runtime_gt_free": list(result.candidate.signals),
                "benchmark_only": [
                    "joint_type_accuracy",
                    "axis_angular_error_deg",
                    "origin_line_distance",
                    "range_endpoint_error",
                ],
            },
        }, indent=2),
        encoding="utf-8",
    )
    parts = [{
        "body_name": "moving_part", "joint_name": args.joint_name,
        "mesh": moving_obj, "source_mesh": args.moving_obj.resolve(),
        "candidate": result.candidate,
    }]
    write_kinematic_bundle_mjcf(
        args.out_dir / "object.xml",
        object_name=args.object_name,
        body_mesh=body_obj,
        parts=parts,
        force_prismatic_local_z=force_prismatic_local_z,
        apply_root_correction=apply_root_correction,
    )
    write_kinematic_bundle_usda(
        args.out_dir / "object.usda",
        object_name=args.object_name,
        body_mesh=args.body_obj.resolve(),
        parts=parts,
        force_prismatic_local_z=force_prismatic_local_z,
        apply_root_correction=apply_root_correction,
    )


if __name__ == "__main__":
    main()
