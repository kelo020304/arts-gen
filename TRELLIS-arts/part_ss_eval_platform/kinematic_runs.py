from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import infer_runs


DEFAULT_KIN_OUTPUT_ROOT = "/root/code/arts-gen/kin_test/eval_platform_runs"
DEFAULT_TEST_DATA_ROOT = "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-single-image-0512"


@dataclass(frozen=True)
class PreparedKinematicInput:
    run_dir: Path
    converter_output_root: Path
    context_json: Path
    initial_joints_json: Path
    object_id: str
    source_run_dir: Path
    target_part_map: dict[str, str]
    joint_count: int


def kinematic_output_root() -> Path:
    return Path(os.environ.get("PART_SS_PLATFORM_KIN_OUTPUT_ROOT") or DEFAULT_KIN_OUTPUT_ROOT)


def test_data_root() -> Path:
    return Path(os.environ.get("PART_SS_PLATFORM_KIN_TEST_DATA_ROOT") or DEFAULT_TEST_DATA_ROOT)


def list_runs(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = Path(root) if root else kinematic_output_root()
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "kinematic_meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        out.append(_run_overview(run_dir, meta))
    out.sort(key=lambda item: (item.get("mtime", 0.0), item.get("run_id", "")), reverse=True)
    return out


def get_run(run_id: str, root: str | Path | None = None) -> dict[str, Any]:
    rd = safe_kin_run_dir(run_id, root=root)
    meta_path = rd / "kinematic_meta.json"
    if not meta_path.is_file():
        raise KeyError(run_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _run_overview(rd, meta)


def safe_kin_run_dir(run_id: str, root: str | Path | None = None) -> Path:
    base = (Path(root) if root else kinematic_output_root()).resolve()
    rd = (base / str(run_id)).resolve()
    if not (str(rd) + os.sep).startswith(str(base) + os.sep):
        raise ValueError(f"kinematic run 路径越界：{run_id}")
    return rd


def safe_kin_artifact(run_id: str, rel: str, root: str | Path | None = None) -> Path:
    rd = safe_kin_run_dir(run_id, root=root)
    target = (rd / str(rel)).resolve()
    if not (str(target) + os.sep).startswith(str(rd) + os.sep):
        raise ValueError(f"kinematic artifact 路径越界：{rel}")
    return target


def build_run_id(object_id: str, source_run_id: str, angle_idx: int | str) -> str:
    safe_source = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(source_run_id))
    return f"{object_id}-{int(angle_idx)}-{safe_source}"


def prepare_kinematic_input(
    *,
    source_root: str | Path,
    object_id: str,
    source_run_id: str,
    angle_idx: int | str = 0,
    out_root: str | Path | None = None,
    test_root: str | Path | None = None,
    overwrite: bool = False,
) -> PreparedKinematicInput:
    """Materialize a kinematic-solver input from stage3 part assets.

    Stage3 here means the infer ``slat`` stage: ``<run>/parts/body.glb`` and
    ``<run>/parts/part_*.glb``. The temporary converter root mimics the old
    RealAppliance layout enough for MJCF preview:

    ``raw/partseg/<object_id>/objs/{body,part_00,...}.obj``.

    VLM initial guesses are currently seeded from dataset GT in
    ``part_info/<object_id>/part_info.json``.
    """
    object_id = str(object_id)
    angle_idx_int = int(angle_idx)
    out_base = Path(out_root) if out_root else kinematic_output_root()
    rd = out_base / build_run_id(object_id, source_run_id, angle_idx_int)
    if rd.exists() and overwrite:
        _clear_generated_run(rd)
    rd.mkdir(parents=True, exist_ok=True)

    source_rd = infer_runs._run_dir(source_root, object_id, source_run_id, angle_idx=angle_idx_int)
    if not source_rd.is_dir():
        raise KeyError(f"source run 不存在：{object_id}/{source_run_id}")
    parts_dir = source_rd / "parts"
    if not parts_dir.is_dir():
        raise FileNotFoundError(f"stage3 parts 目录不存在（先跑 slat）：{parts_dir}")

    data_root = Path(test_root) if test_root else test_data_root()
    part_info = _load_part_info(data_root, object_id)
    labels = infer_runs.part_labels(source_root, object_id, source_run_id, angle_idx=angle_idx_int)
    stem_to_label = {
        str(item.get("stem")): str(item.get("label"))
        for item in labels.get("components", [])
        if item.get("stem") and item.get("label")
    }

    converter_root = rd / "converter_output"
    obj_dir = converter_root / "raw" / "partseg" / object_id / "objs"
    obj_dir.mkdir(parents=True, exist_ok=True)
    target_part_map = _export_stage3_objs(parts_dir, obj_dir, stem_to_label, part_info)
    ctx, initial = _build_context_and_initial(object_id, part_info, target_part_map, obj_dir)
    context_json = rd / "context.json"
    initial_json = rd / "vlm_initial_from_gt.json"
    context_json.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    initial_json.write_text(json.dumps(initial, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "object_id": object_id,
        "angle_idx": angle_idx_int,
        "source_root": str(source_root),
        "source_run_id": str(source_run_id),
        "source_run_dir": str(source_rd),
        "test_data_root": str(data_root),
        "converter_output_root": str(converter_root),
        "context_json": str(context_json),
        "initial_joints_json": str(initial_json),
        "target_part_map": target_part_map,
        "joint_count": len(ctx["joints"]),
        "status": "prepared",
    }
    (rd / "kinematic_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedKinematicInput(
        run_dir=rd,
        converter_output_root=converter_root,
        context_json=context_json,
        initial_joints_json=initial_json,
        object_id=object_id,
        source_run_dir=source_rd,
        target_part_map=target_part_map,
        joint_count=len(ctx["joints"]),
    )


def update_run_meta(run_dir: Path, **updates: Any) -> None:
    meta_path = Path(run_dir) / "kinematic_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_generated_run(run_dir: Path) -> None:
    for name in (
        "candidate_report.json",
        "predictions.jsonl",
        "frontend_state.json",
        "agent_events.jsonl",
        "agent_subprocess.log",
        "context.json",
        "vlm_initial_from_gt.json",
    ):
        p = run_dir / name
        if p.exists():
            p.unlink()
    for name in ("converter_output", "object_assets"):
        p = run_dir / name
        if p.exists():
            shutil.rmtree(p)


def _load_part_info(data_root: Path, object_id: str) -> dict[str, Any]:
    path = Path(data_root) / "part_info" / str(object_id) / "part_info.json"
    if not path.is_file():
        raise FileNotFoundError(f"测试数据缺 part_info：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("object_id")) != str(object_id):
        raise ValueError(f"part_info object_id mismatch: {path}")
    return payload


def _export_stage3_objs(
    parts_dir: Path,
    obj_dir: Path,
    stem_to_label: dict[str, str],
    part_info: dict[str, Any],
) -> dict[str, str]:
    """Export stage3 glb meshes to kinematic part names.

    The kinematic solver expects a static body plus one moving part per joint.
    Stage3 only contains target parts plus ``body.glb``. If a target part is GT
    fixed, we merge it into ``body.obj``. If it is movable, it becomes
    ``part_XX.obj`` and is mapped back to the GT part name.
    """
    import trimesh

    info_by_name = dict(part_info.get("parts") or {})
    movable_sources: list[tuple[str, Path, str]] = []
    body_meshes = []

    body_glb = parts_dir / "body.glb"
    if body_glb.is_file():
        body_meshes.append(_load_mesh(body_glb, trimesh=trimesh))

    for glb in sorted(parts_dir.glob("part_*.glb")):
        label = stem_to_label.get(glb.stem, glb.stem)
        gt = info_by_name.get(label)
        if not isinstance(gt, dict):
            # Unknown labels are still geometry context, but they cannot define a
            # solver joint.
            body_meshes.append(_load_mesh(glb, trimesh=trimesh))
            continue
        if _is_movable_joint(gt.get("joint_type")):
            movable_sources.append((glb.stem, glb, label))
        else:
            body_meshes.append(_load_mesh(glb, trimesh=trimesh))

    if not body_meshes:
        overall = parts_dir / "overall.glb"
        if overall.is_file():
            body_meshes.append(_load_mesh(overall, trimesh=trimesh))
    if not body_meshes:
        raise FileNotFoundError(f"stage3 缺 body/overall glb，无法构造 body.obj：{parts_dir}")
    _concat_meshes(body_meshes, trimesh=trimesh).export(obj_dir / "body.obj")

    target_map: dict[str, str] = {}
    for idx, (_stem, glb, label) in enumerate(movable_sources):
        kin_name = f"part_{idx:02d}"
        mesh = _load_mesh(glb, trimesh=trimesh)
        mesh.export(obj_dir / f"{kin_name}.obj")
        target_map[kin_name] = label
    if not target_map:
        raise ValueError(
            "stage3 run 没有可动 target part；检查 part_flow target_part_names 与 "
            "测试数据 part_info 是否匹配。"
        )
    return target_map


def _load_mesh(path: Path, *, trimesh):
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if getattr(geom, "vertices", None) is not None and len(geom.vertices) > 0
        ]
        if not meshes:
            raise ValueError(f"mesh 为空：{path}")
        return _concat_meshes(meshes, trimesh=trimesh)
    return loaded


def _concat_meshes(meshes, *, trimesh):
    meshes = [mesh for mesh in meshes if mesh is not None and len(getattr(mesh, "vertices", [])) > 0]
    if not meshes:
        raise ValueError("empty mesh list")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _build_context_and_initial(
    object_id: str,
    part_info: dict[str, Any],
    target_part_map: dict[str, str],
    obj_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info_by_name = dict(part_info.get("parts") or {})
    available = ["body", *sorted(target_part_map)]
    centers = {name: _obj_vertex_center(obj_dir / f"{name}.obj") for name in available}
    joints: dict[str, Any] = {}
    initial: dict[str, Any] = {}
    for kin_name, gt_name in sorted(target_part_map.items()):
        gt = info_by_name.get(gt_name)
        if not isinstance(gt, dict):
            continue
        joint_type = _joint_type_name(gt.get("joint_type"))
        if joint_type is None:
            continue
        params = gt.get("joint_params") or []
        axis = _unit3(params[:3], f"{gt_name}.joint_params[:3]")
        origin = _vec3(params[3:6], default=centers.get(kin_name, [0.0, 0.0, 0.0]))
        lower, upper = _limits_for_joint(joint_type, params)
        static = [part for part in available if part != kin_name]
        joints[kin_name] = {
            "object_id": object_id,
            "joint_name": kin_name,
            "joint_path": f"/World/{kin_name}/{kin_name}",
            "type": joint_type,
            "canonical_unit": "meters" if joint_type == "prismatic" else "radians",
            "axis_world": axis,
            "origin_world": origin,
            "moving_parts": [kin_name],
            "static_parts": static,
            "body0_path": "/World/body",
            "child_body_path": f"/World/{kin_name}",
            "body0_link_name": "body",
            "source": "physx_part_info_gt",
            "gt_part_name": gt_name,
            "gt_joint_group_id": str(gt.get("joint_group_id") or ""),
        }
        # load_vlm_initial_context expects prismatic limits in millimeters and
        # revolute limits in degrees. This file intentionally uses GT as a VLM
        # stand-in until real VLM outputs are wired.
        initial_limit = (
            [lower * 1000.0, upper * 1000.0]
            if joint_type == "prismatic"
            else [math.degrees(lower), math.degrees(upper)]
        )
        initial[kin_name] = {
            "type": joint_type,
            "axis": axis,
            "origin_world": origin,
            "limit": initial_limit,
            "parent": "body",
            "moving_parts": [kin_name],
            "static_parts": static,
            "gt_part_name": gt_name,
        }
    evidence = {
        "__available_parts__": available,
        "__part_centers__": centers,
        "__target_part_map__": target_part_map,
        "__source__": "stage3_parts_plus_physx_part_info_gt",
    }
    for kin_name, gt_name in target_part_map.items():
        evidence[kin_name] = {"labels": [gt_name], "gt_part_name": gt_name}
    return (
        {"object_id": object_id, "joints": joints, "evidence": evidence},
        {"object_id": object_id, "initial_joints": initial, "source": "dataset_gt_as_vlm"},
    )


def _is_movable_joint(value: Any) -> bool:
    return str(value) in {"B", "C"}


def _joint_type_name(value: Any) -> str | None:
    raw = str(value)
    if raw == "B":
        return "prismatic"
    if raw == "C":
        return "revolute"
    return None


def _limits_for_joint(joint_type: str, params: list[Any]) -> tuple[float, float]:
    if len(params) >= 8:
        lower = float(params[6])
        upper = float(params[7])
    else:
        lower = 0.0
        upper = 0.1 if joint_type == "prismatic" else math.pi / 2.0
    if joint_type == "revolute":
        lower *= math.pi
        upper *= math.pi
    return (min(lower, upper), max(lower, upper))


def _unit3(raw: Any, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise ValueError(f"{label} must contain at least 3 numbers")
    vals = [float(raw[i]) for i in range(3)]
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return [v / norm for v in vals]


def _vec3(raw: Any, *, default: list[float]) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return [float(v) for v in default]
    return [float(raw[i]) for i in range(3)]


def _obj_vertex_center(path: Path) -> list[float]:
    vertices = []
    if not path.is_file():
        return [0.0, 0.0, 0.0]
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    if not vertices:
        return [0.0, 0.0, 0.0]
    return [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]


def _run_overview(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    report = run_dir / "candidate_report.json"
    preds = run_dir / "predictions.jsonl"
    frontend = run_dir / "frontend_state.json"
    status = str(meta.get("status") or "")
    if report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            status = "done" if payload.get("passed") else "failed"
        except (OSError, json.JSONDecodeError):
            status = status or "done"
    elif (run_dir / "agent_subprocess.log").is_file() and not status:
        status = "running"
    return {
        "run_id": run_dir.name,
        "path": str(run_dir.resolve()),
        "mtime": run_dir.stat().st_mtime,
        "object_id": meta.get("object_id"),
        "angle_idx": meta.get("angle_idx"),
        "source_root": meta.get("source_root"),
        "source_run_id": meta.get("source_run_id"),
        "source_run_dir": meta.get("source_run_dir"),
        "joint_count": meta.get("joint_count", 0),
        "status": status or "prepared",
        "has_report": report.is_file(),
        "has_predictions": preds.is_file(),
        "has_frontend_state": frontend.is_file(),
        "artifacts": {
            "candidate_report.json": report.is_file(),
            "predictions.jsonl": preds.is_file(),
            "frontend_state.json": frontend.is_file(),
            "vlm_initial_from_gt.json": (run_dir / "vlm_initial_from_gt.json").is_file(),
            "context.json": (run_dir / "context.json").is_file(),
        },
    }
