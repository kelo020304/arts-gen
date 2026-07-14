#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_COMPILE = Path("/home/mi/mujoco-3.2.7/bin/compile")
DRAWER_RANGE_MAX = 0.381732268


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _asset_paths(xml_path: Path, root: ET.Element) -> list[Path]:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
    texturedir = compiler.get("texturedir", ".") if compiler is not None else "."
    out: list[Path] = []
    for mesh in root.findall("./asset/mesh"):
        file_value = mesh.get("file")
        if file_value:
            out.append((xml_path.parent / meshdir / file_value).resolve())
    for texture in root.findall("./asset/texture"):
        file_value = texture.get("file")
        if file_value:
            out.append((xml_path.parent / texturedir / file_value).resolve())
    return out


def _static_xml_checks(xml_path: Path) -> dict[str, Any]:
    text = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assets = _asset_paths(xml_path, root)
    missing_assets = [str(path) for path in assets if not path.is_file()]
    floor_count = 0
    for geom in root.findall(".//geom"):
        if geom.get("name") == "floor" or geom.get("type") == "plane" or geom.get("material") == "blue_checker":
            floor_count += 1
    shell_count = text.count('inertia="shell"')
    object_body = root.find("./worldbody/body[@name='object']")
    drawer_body = root.find(".//body[@name='part_00_u70b8_u7bee_0']")
    if drawer_body is None:
        drawer_body = next(
            (
                body
                for body in root.findall(".//body")
                if any(joint.get("type") == "slide" for joint in body.findall("joint"))
            ),
            None,
        )
    glass_parent_is_drawer = False
    glass_parent_name = None
    glass_geom_count = 0
    for body in root.findall(".//body"):
        if body.find("geom[@name='drawer_glass_visual']") is not None:
            glass_geom_count += 1
            glass_parent_name = body.get("name")
            glass_parent_is_drawer = body is drawer_body
    glass_mesh_count = len(root.findall("./asset/mesh[@name='drawer_glass_mesh']"))
    joints = root.findall(".//joint")
    axes = [joint.get("axis", "") for joint in joints]
    slide_joints = [joint for joint in joints if joint.get("type") == "slide"]
    compiler = root.find("compiler")
    visual_geoms = root.findall(".//geom[@group='0']")
    collision_geoms = [geom for geom in root.findall(".//geom") if geom.get("group") in {"3", "4", "5"}]
    moving_bodies = [body for body in root.findall(".//body") if body.find("joint") is not None]
    inertia_ready_bodies = []
    for body in moving_bodies:
        has_inertial = body.find("inertial") is not None
        has_inertia_geom = any(geom.get("group") in {"3", "4", "5"} for geom in body.findall("geom"))
        if has_inertial or has_inertia_geom:
            inertia_ready_bodies.append(body.get("name"))
    return {
        "floor_count": int(floor_count),
        "shell_count": int(shell_count),
        "missing_assets": missing_assets,
        "object_quat": None if object_body is None else object_body.get("quat"),
        "xml_joint_count": int(len(joints)),
        "xml_joint_axes": axes,
        "slide_joint_count": len(slide_joints),
        "slide_joint_axes": [joint.get("axis", "") for joint in slide_joints],
        "slide_joint_ranges": [joint.get("range", "") for joint in slide_joints],
        "compiler_balanceinertia": None if compiler is None else compiler.get("balanceinertia"),
        "compiler_inertiagrouprange": None if compiler is None else compiler.get("inertiagrouprange"),
        "visual_group0_count": len(visual_geoms),
        "collision_group3_5_count": len(collision_geoms),
        "moving_body_count": len(moving_bodies),
        "inertia_ready_body_names": inertia_ready_bodies,
        "glass_mesh_count": int(glass_mesh_count),
        "glass_geom_count": int(glass_geom_count),
        "glass_parent_name": glass_parent_name,
        "glass_parent_is_drawer": bool(glass_parent_is_drawer),
        "has_blue_checker": "blue_checker" in text,
    }


def _run_compile(xml_path: Path, compile_path: Path) -> dict[str, Any]:
    if not compile_path.is_file():
        return {"available": False, "path": str(compile_path), "ok": None, "stdout": "", "stderr": ""}
    with tempfile.TemporaryDirectory(prefix="mujoco_compile_") as tmp:
        out_xml = Path(tmp) / "compiled.xml"
        proc = subprocess.run(
            [str(compile_path), str(xml_path), str(out_xml)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        return {
            "available": True,
            "path": str(compile_path),
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "compiled_xml": str(out_xml) if out_xml.is_file() else None,
        }


def _import_mujoco() -> Any:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco  # noqa: PLC0415

    return mujoco


def _name(mujoco: Any, model: Any, obj_type: int, idx: int) -> str:
    value = mujoco.mj_id2name(model, obj_type, int(idx))
    return "" if value is None else str(value)


def _render_compare(
    *,
    mujoco: Any,
    model: Any,
    out_png: Path,
    joint_poses: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = np.asarray(model.stat.center, dtype=np.float64)
    camera.distance = max(0.25, float(model.stat.extent) * 2.2)
    camera.azimuth = 180.0
    camera.elevation = -8.0

    frames = []
    poses = [{"joint_name": "rest", "qpos_index": None, "qpos": 0.0}] + joint_poses
    for pose in poses:
        data = mujoco.MjData(model)
        if pose["qpos_index"] is not None:
            data.qpos[int(pose["qpos_index"])] = float(pose["qpos"])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append((pose, renderer.render().copy()))
    renderer.close()

    canvas = Image.new("RGB", (width * len(frames), height + 34), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (pose, pixels) in enumerate(frames):
        canvas.paste(Image.fromarray(pixels), (width * index, 34))
        label = "rest" if pose["qpos_index"] is None else f"{pose['joint_name']} q={pose['qpos']:.5g}"
        draw.text((width * index + 8, 8), label, fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    per_joint = []
    for pose, pixels in frames[1:]:
        diff = np.mean(np.abs(frames[0][1].astype(np.float32) - pixels.astype(np.float32)))
        per_joint.append({
            "joint_name": pose["joint_name"],
            "qpos_index": int(pose["qpos_index"]),
            "qpos": float(pose["qpos"]),
            "mean_abs_pixel_diff": float(diff),
        })
    pixel_std = float(np.std(frames[0][1]))
    return {
        "ok": pixel_std > 1.0,
        "path": str(out_png),
        "mean_abs_pixel_diff": max((row["mean_abs_pixel_diff"] for row in per_joint), default=0.0),
        "per_joint": per_joint,
        "rest_pixel_std": pixel_std,
    }


def _mujoco_checks(
    *,
    xml_path: Path,
    out_png: Path,
    qpos_max: float,
    render_width: int,
    render_height: int,
) -> dict[str, Any]:
    mujoco = _import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data0 = mujoco.MjData(model)
    mujoco.mj_forward(model, data0)
    jnt_axis = np.asarray(model.jnt_axis, dtype=np.float64).reshape(model.njnt, 3)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
    slide_ids = [index for index in range(model.njnt) if int(model.jnt_type[index]) == slide_type]
    slide_checks = []
    joint_motion_checks = []
    joint_poses = []
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    for joint_id in range(model.njnt):
        qpos_index = int(model.jnt_qposadr[joint_id])
        lower, upper = [float(value) for value in model.jnt_range[joint_id]]
        target = upper if abs(upper) >= abs(lower) else lower
        if int(model.jnt_type[joint_id]) == slide_type:
            target = min(float(qpos_max), target) if target > 0.0 else max(-float(qpos_max), target)
        moved = mujoco.MjData(model)
        moved.qpos[qpos_index] = target
        mujoco.mj_forward(model, moved)
        body_id = int(model.jnt_bodyid[joint_id])
        position_delta = np.asarray(moved.xpos[body_id] - data0.xpos[body_id], dtype=np.float64)
        rest_rotation = np.asarray(data0.xmat[body_id], dtype=np.float64).reshape(3, 3)
        moved_rotation = np.asarray(moved.xmat[body_id], dtype=np.float64).reshape(3, 3)
        relative_rotation = rest_rotation.T @ moved_rotation
        rotation_angle = float(np.arccos(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)))
        joint_type = "hinge" if int(model.jnt_type[joint_id]) == hinge_type else "slide"
        motion_magnitude = rotation_angle if joint_type == "hinge" else float(np.linalg.norm(position_delta))
        joint_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_motion_checks.append({
            "joint_id": joint_id,
            "joint_name": joint_name,
            "joint_type": joint_type,
            "body_name": _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "qpos_index": qpos_index,
            "qpos": target,
            "range": [lower, upper],
            "position_delta": position_delta.tolist(),
            "rotation_angle_rad": rotation_angle,
            "motion_magnitude": motion_magnitude,
            "motion_ok": bool(abs(target) > 1e-8 and motion_magnitude > 1e-5),
        })
        joint_poses.append({"joint_name": joint_name, "qpos_index": qpos_index, "qpos": target})
    for joint_id in slide_ids:
        qpos_index = int(model.jnt_qposadr[joint_id])
        lower, upper = [float(value) for value in model.jnt_range[joint_id]]
        qpos = min(float(qpos_max), upper) if upper > 0.0 else max(-float(qpos_max), lower)
        moved = mujoco.MjData(model)
        moved.qpos[qpos_index] = qpos
        mujoco.mj_forward(model, moved)
        body_id = int(model.jnt_bodyid[joint_id])
        delta = np.asarray(moved.xpos[body_id] - data0.xpos[body_id], dtype=np.float64)
        slide_checks.append({
            "joint_id": joint_id,
            "joint_name": _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
            "body_name": _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            "qpos_index": qpos_index,
            "qpos": qpos,
            "axis": jnt_axis[joint_id].tolist(),
            "range": [lower, upper],
            "delta": delta.tolist(),
            "local_z_ok": bool(np.allclose(jnt_axis[joint_id], [0, 0, 1])),
            "positive_range_ok": bool(lower >= -1e-9 and upper > lower),
            "forward_negative_y_ok": bool(qpos > 0.0 and delta[1] < -0.5 * qpos),
        })

    geom_names = [_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, idx) for idx in range(model.ngeom)]
    glass_geom_ids = [idx for idx, name in enumerate(geom_names) if name == "drawer_glass_visual"]
    drawer_geom_ids = [
        idx
        for idx, name in enumerate(geom_names)
        if name.startswith("part_") and name.endswith("_visual") and name != "drawer_glass_visual"
    ]
    glass_same_body = False
    if glass_geom_ids and drawer_geom_ids:
        glass_body = int(model.geom_bodyid[glass_geom_ids[0]])
        drawer_bodies = {int(model.geom_bodyid[idx]) for idx in drawer_geom_ids}
        glass_same_body = glass_body in drawer_bodies

    render = {"ok": False, "path": str(out_png)}
    try:
        render = _render_compare(
            mujoco=mujoco,
            model=model,
            out_png=out_png,
            joint_poses=joint_poses,
            width=render_width,
            height=render_height,
        )
    except Exception as exc:  # rendering backend can be unavailable on headless machines
        render = {"ok": False, "path": str(out_png), "error": f"{type(exc).__name__}: {exc}"}

    return {
        "mujoco_version": str(getattr(mujoco, "__version__", "")),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "ntex": int(model.ntex),
        "jnt_axis": jnt_axis.tolist(),
        "slide_joint_checks": slide_checks,
        "joint_motion_checks": joint_motion_checks,
        "joint_names": [_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)],
        "actuator_names": [_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)],
        "geom_names": geom_names,
        "joint_body_masses": [float(model.body_mass[int(model.jnt_bodyid[index])]) for index in range(model.njnt)],
        "has_drawer_glass": bool(glass_geom_ids),
        "glass_same_body_as_drawer": bool(glass_same_body),
        "render": render,
    }


def _overall_ok(report: dict[str, Any], expect_nq: int, expect_nv: int, expect_nu: int, expect_njnt: int) -> bool:
    xml = report["xml"]
    mj = report.get("mujoco") or {}
    compile_report = report.get("compile") or {}
    slide_checks = mj.get("slide_joint_checks") or []
    slides_ok = all(
        item.get("local_z_ok") and item.get("positive_range_ok") and item.get("forward_negative_y_ok")
        for item in slide_checks
    )
    masses_ok = all(float(value) > 0.0 for value in (mj.get("joint_body_masses") or []))
    motion_checks = mj.get("joint_motion_checks")
    motions_ok = True if motion_checks is None else all(bool(item.get("motion_ok")) for item in motion_checks)
    compile_ok = True if not compile_report.get("available") else bool(compile_report.get("ok"))
    has_glass = int(xml.get("glass_geom_count") or 0) > 0 or bool(mj.get("has_drawer_glass"))
    glass_ok = True if not has_glass else bool(mj.get("glass_same_body_as_drawer"))
    return bool(
        compile_ok
        and xml.get("floor_count") == 0
        and xml.get("shell_count") == 0
        and not xml.get("missing_assets")
        and xml.get("object_quat") == "0.707106781 0.707106781 0 0"
        and xml.get("compiler_balanceinertia") == "true"
        and xml.get("compiler_inertiagrouprange") == "3 5"
        and xml.get("visual_group0_count", 0) >= 1
        and xml.get("collision_group3_5_count", 0) >= xml.get("moving_body_count", 0)
        and len(xml.get("inertia_ready_body_names") or []) == xml.get("moving_body_count", 0)
        and mj.get("nq") == expect_nq
        and mj.get("nv") == expect_nv
        and mj.get("nu") == expect_nu
        and mj.get("njnt") == expect_njnt
        and slides_ok
        and masses_ok
        and motions_ok
        and glass_ok
        and bool((mj.get("render") or {}).get("ok"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R-D MuJoCo XML assets for drawer RealAppliance exports.")
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--compile", type=Path, default=DEFAULT_COMPILE)
    parser.add_argument("--qpos-max", type=float, default=DRAWER_RANGE_MAX)
    parser.add_argument("--expect-nq", type=int, default=1)
    parser.add_argument("--expect-nv", type=int, default=1)
    parser.add_argument("--expect-nu", type=int, default=1)
    parser.add_argument("--expect-njnt", type=int, default=1)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    out_json = args.out_json or (xml_path.parent / "mujoco_validation.json")
    out_png = args.out_png or (xml_path.parent / "mujoco_qpos_compare.png")
    report: dict[str, Any] = {
        "xml_path": str(xml_path),
        "xml": _static_xml_checks(xml_path),
        "compile": _run_compile(xml_path, args.compile),
    }
    try:
        report["mujoco"] = _mujoco_checks(
            xml_path=xml_path,
            out_png=out_png,
            qpos_max=float(args.qpos_max),
            render_width=int(args.render_width),
            render_height=int(args.render_height),
        )
    except Exception as exc:
        report["mujoco"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["ok"] = _overall_ok(
        report,
        expect_nq=int(args.expect_nq),
        expect_nv=int(args.expect_nv),
        expect_nu=int(args.expect_nu),
        expect_njnt=int(args.expect_njnt),
    )
    _write_json(out_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
