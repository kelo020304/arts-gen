"""Run-scoped MJCF preview assets for the post_process viewer."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .coordinate_frame import context_uses_canonical_frame, copy_obj_as_canonical
from .schemas import EstimateContext, LimitEstimate


def write_iteration_mjcf_preview(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
    *,
    converter_output_root: Path,
    run_dir: Path,
    iteration: int,
    motion_search: list[dict[str, Any]] | None = None,
    joint_states: dict[str, Any] | None = None,
    manual_sliders: bool = False,
    preview_kind: str = "candidate",
) -> dict[str, Any]:
    """Write an MJCF asset for one agent iteration and return its manifest."""
    run_dir = Path(run_dir)
    asset_root = run_dir / "object_assets"
    asset_name = _asset_name(ctx.object_id, run_dir, iteration)
    asset_dir = asset_root / asset_name
    mjcf_dir = asset_dir / "mjcf"
    assets_dir = mjcf_dir / "assets"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    part_to_obj = _copy_preview_meshes(
        ctx,
        estimates,
        converter_output_root=Path(converter_output_root),
        assets_dir=assets_dir,
    )
    xml_path = mjcf_dir / f"{asset_name}.xml"
    xml_path.write_text(
        _build_mjcf_xml(ctx, estimates, part_to_obj=part_to_obj),
        encoding="utf-8",
    )
    manifest = _generate_manifest(asset_name, asset_root)
    playback = {
        "mode": "manual_sliders" if manual_sliders else "sequential_full_range",
        "seconds_per_joint": 1.8,
    }
    preview = {
        "iteration": int(iteration),
        "object_id": ctx.object_id,
        "run_dir": str(run_dir),
        "asset_root": str(asset_root),
        "asset_name": asset_name,
        "asset_dir": str(asset_dir),
        "xml_path": str(xml_path),
        "preview_kind": preview_kind,
        "manifest": manifest,
        "playback": playback,
        "motion_search": list(motion_search or []),
        "joint_states": dict(joint_states or {}),
        "joint_types": _joint_types(ctx),
        "estimates": [asdict(estimate) for estimate in estimates],
    }
    _update_frontend_state(run_dir, preview)
    return preview


def write_rest_mjcf_preview(
    ctx: EstimateContext,
    *,
    converter_output_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Write the initial rest-pose asset before any candidate limits are known."""
    rest_estimates = [
        LimitEstimate(
            joint_name=joint_name,
            lower=0.0,
            upper=0.0,
            axis_world=joint.get("axis_world"),
            axis_label=_axis_label(joint.get("axis_world")),
            confidence=None,
            reason="Initial rest-pose preview before agent limit estimation.",
        )
        for joint_name, joint in sorted(ctx.joints.items())
    ]
    run_dir = Path(run_dir)
    asset_root = run_dir / "object_assets"
    asset_name = _asset_name(ctx.object_id, run_dir, 0)
    asset_dir = asset_root / asset_name
    mjcf_dir = asset_dir / "mjcf"
    assets_dir = mjcf_dir / "assets"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    part_to_obj = _copy_preview_meshes(
        ctx,
        rest_estimates,
        converter_output_root=Path(converter_output_root),
        assets_dir=assets_dir,
        include_all_meshes=True,
    )
    xml_path = mjcf_dir / f"{asset_name}.xml"
    xml_path.write_text(
        _build_mjcf_xml(
            ctx,
            rest_estimates,
            part_to_obj=part_to_obj,
            include_joints=False,
        ),
        encoding="utf-8",
    )
    manifest = _generate_manifest(asset_name, asset_root)
    preview = {
        "iteration": 0,
        "object_id": ctx.object_id,
        "run_dir": str(run_dir),
        "asset_root": str(asset_root),
        "asset_name": asset_name,
        "asset_dir": str(asset_dir),
        "xml_path": str(xml_path),
        "manifest": manifest,
    }
    preview["preview_kind"] = "rest"
    preview["playback"] = {
        "mode": "rest_pose",
        "seconds_per_joint": 1.8,
    }
    preview["estimates"] = []
    preview["joint_types"] = _joint_types(ctx)
    _update_frontend_state(Path(run_dir), preview)
    return preview


def _asset_name(object_id: str, run_dir: Path, iteration: int) -> str:
    digest = hashlib.sha1(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_object = re.sub(r"[^A-Za-z0-9_]+", "_", object_id).strip("_")
    return f"ks_{safe_object}_{digest}_iter_{int(iteration):03d}"


def _copy_preview_meshes(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
    *,
    converter_output_root: Path,
    assets_dir: Path,
    include_all_meshes: bool = False,
) -> dict[str, str]:
    obj_dir = converter_output_root / f"raw/partseg/{ctx.object_id}/objs"
    required_parts = {
        path.stem
        for path in obj_dir.glob("*.obj")
    } if include_all_meshes else {"body"}
    for estimate in estimates:
        joint = ctx.joints.get(estimate.joint_name, {})
        required_parts.update(str(part) for part in joint.get("moving_parts", []))
    part_to_file: dict[str, str] = {}
    for part in sorted(required_parts):
        source = obj_dir / f"{part}.obj"
        if not source.is_file():
            continue
        target = assets_dir / source.name
        if context_uses_canonical_frame(ctx):
            copy_obj_as_canonical(source, target)
        else:
            shutil.copyfile(source, target)
        part_to_file[part] = f"assets/{source.name}"
    return part_to_file


def _build_mjcf_xml(
    ctx: EstimateContext,
    estimates: list[LimitEstimate],
    *,
    part_to_obj: dict[str, str],
    include_joints: bool = True,
) -> str:
    root = ET.Element("mujoco", {"model": ctx.object_id})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": ".",
        },
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        {
            "type": "mesh",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.72 0.76 0.80 1",
        },
    )
    asset = ET.SubElement(root, "asset")
    for part, mesh_file in sorted(part_to_obj.items()):
        ET.SubElement(asset, "mesh", {"name": f"{part}_mesh", "file": mesh_file})

    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "body", "pos": "0 0 0"})
    if "body" in part_to_obj:
        ET.SubElement(
            body,
            "geom",
            {
                "name": "body_visual",
                "mesh": "body_mesh",
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    handled_parts = {"body"}
    for estimate in estimates:
        if estimate.joint_name not in ctx.joints:
            continue
        joint = ctx.joints[estimate.joint_name]
        moving_parts = list(joint.get("moving_parts") or [estimate.joint_name])
        axis = estimate.axis_world or joint.get("axis_world") or [0.0, 0.0, 1.0]
        origin = joint.get("origin_world") or [0.0, 0.0, 0.0]
        joint_type = "hinge" if joint.get("type") == "revolute" else "slide"
        for idx, part in enumerate(moving_parts):
            if part not in part_to_obj:
                continue
            handled_parts.add(str(part))
            part_body_name = str(part)
            part_body = ET.SubElement(
                body,
                "body",
                {"name": part_body_name, "pos": "0 0 0"},
            )
            if include_joints and idx == 0:
                joint_range = _mjcf_valid_range(float(estimate.lower), float(estimate.upper))
                ET.SubElement(
                    part_body,
                    "joint",
                    {
                        "name": estimate.joint_name,
                        "type": joint_type,
                        "pos": _float_list(origin),
                        "axis": _float_list(axis),
                        "range": _float_list(joint_range),
                        "limited": "true",
                    },
                )
            ET.SubElement(
                part_body,
                "geom",
                {
                    "name": f"{part}_visual",
                    "mesh": f"{part}_mesh",
                    "group": "2",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )

    for part in sorted(set(part_to_obj) - handled_parts):
        static_body = ET.SubElement(body, "body", {"name": str(part), "pos": "0 0 0"})
        ET.SubElement(
            static_body,
            "geom",
            {
                "name": f"{part}_visual",
                "mesh": f"{part}_mesh",
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    _indent_xml(root)
    return ET.tostring(root, encoding="unicode")


def _float_list(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _mjcf_valid_range(lower: float, upper: float) -> list[float]:
    if lower < upper:
        return [lower, upper]
    return [lower, lower + 1e-6]


def _axis_label(axis: Any) -> str | None:
    if not axis or len(axis) != 3:
        return None
    values = [float(value) for value in axis]
    idx = max(range(3), key=lambda item: abs(values[item]))
    names = ("X", "Y", "Z")
    return ("+" if values[idx] >= 0.0 else "-") + names[idx]


def _joint_types(ctx: EstimateContext) -> dict[str, str]:
    return {
        str(name): str(joint.get("type") or "")
        for name, joint in sorted(ctx.joints.items())
        if isinstance(joint, dict)
    }


def _generate_manifest(asset_name: str, asset_root: Path) -> dict[str, Any]:
    try:
        from post_process.object_post_process.mjcf_parser import generate_manifest
    except ModuleNotFoundError:
        from object_post_process.mjcf_parser import generate_manifest

    return generate_manifest(asset_name, Path(asset_root))


def _update_frontend_state(run_dir: Path, preview: dict[str, Any]) -> None:
    state_path = run_dir / "frontend_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}
    iterations = [
        item
        for item in state.get("iterations", [])
        if isinstance(item, dict) and item.get("iteration") != preview["iteration"]
    ]
    compact_preview = {
        key: value
        for key, value in preview.items()
        if key not in {"manifest"}
    }
    iterations.append(compact_preview)
    iterations.sort(key=lambda item: int(item.get("iteration", 0)))
    state.update(
        {
            "object_id": preview["object_id"],
            "latest_iteration": preview["iteration"],
            "latest_preview": compact_preview,
            "iterations": iterations,
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            _indent_xml(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent
