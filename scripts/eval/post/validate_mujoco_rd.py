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
    return {
        "floor_count": int(floor_count),
        "shell_count": int(shell_count),
        "missing_assets": missing_assets,
        "object_quat": None if object_body is None else object_body.get("quat"),
        "xml_joint_count": int(len(joints)),
        "xml_joint_axes": axes,
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
    qpos_max: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = np.asarray([0.0, -0.15, 0.0], dtype=np.float64)
    camera.distance = 2.0
    camera.azimuth = 180.0
    camera.elevation = -8.0

    frames = []
    for qpos in (0.0, qpos_max):
        data = mujoco.MjData(model)
        if model.nq:
            data.qpos[0] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append((qpos, renderer.render().copy()))
    renderer.close()

    left = Image.fromarray(frames[0][1])
    right = Image.fromarray(frames[1][1])
    canvas = Image.new("RGB", (width * 2, height + 34), (255, 255, 255))
    canvas.paste(left, (0, 34))
    canvas.paste(right, (width, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"qpos={frames[0][0]:.6g}", fill=(0, 0, 0))
    draw.text((width + 8, 8), f"qpos={frames[1][0]:.6g}", fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    diff = np.mean(np.abs(frames[0][1].astype(np.float32) - frames[1][1].astype(np.float32)))
    return {"ok": True, "path": str(out_png), "mean_abs_pixel_diff": float(diff)}


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
    data1 = mujoco.MjData(model)
    if model.nq:
        data1.qpos[0] = qpos_max
    mujoco.mj_forward(model, data1)

    jnt_axis = np.asarray(model.jnt_axis, dtype=np.float64).reshape(model.njnt, 3)
    joint_body = int(model.jnt_bodyid[0]) if model.njnt else -1
    drawer_delta = None
    drawer_forward_ok = None
    if joint_body >= 0:
        delta = np.asarray(data1.xpos[joint_body] - data0.xpos[joint_body], dtype=np.float64)
        drawer_delta = [float(v) for v in delta]
        drawer_forward_ok = bool(delta[1] < -0.5 * qpos_max)

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
            qpos_max=qpos_max,
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
        "joint_names": [_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)],
        "actuator_names": [_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)],
        "geom_names": geom_names,
        "drawer_body_name": "" if joint_body < 0 else _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, joint_body),
        "drawer_delta_qpos_max": drawer_delta,
        "drawer_forward_negative_y_ok": drawer_forward_ok,
        "has_drawer_glass": bool(glass_geom_ids),
        "glass_same_body_as_drawer": bool(glass_same_body),
        "render": render,
    }


def _overall_ok(report: dict[str, Any], expect_nq: int, expect_nv: int, expect_nu: int, expect_njnt: int) -> bool:
    xml = report["xml"]
    mj = report.get("mujoco") or {}
    compile_report = report.get("compile") or {}
    axis = np.asarray(mj.get("jnt_axis") or [], dtype=np.float64)
    axis_ok = bool(axis.shape[0] == expect_njnt and (expect_njnt == 0 or np.allclose(axis[0], [0, 0, 1])))
    compile_ok = True if not compile_report.get("available") else bool(compile_report.get("ok"))
    has_glass = int(xml.get("glass_geom_count") or 0) > 0 or bool(mj.get("has_drawer_glass"))
    glass_ok = True if not has_glass else bool(mj.get("glass_same_body_as_drawer"))
    return bool(
        compile_ok
        and xml.get("floor_count") == 0
        and xml.get("shell_count") == 0
        and not xml.get("missing_assets")
        and xml.get("object_quat") == "0.707106781 0.707106781 0 0"
        and mj.get("nq") == expect_nq
        and mj.get("nv") == expect_nv
        and mj.get("nu") == expect_nu
        and mj.get("njnt") == expect_njnt
        and axis_ok
        and bool(mj.get("drawer_forward_negative_y_ok"))
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
