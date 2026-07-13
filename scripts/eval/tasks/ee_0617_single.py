#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SS_FLOW_FUSION_MODE", "concat")

import inference  # noqa: E402
from inference_pipeline.part_prompt_seg_stage import (  # noqa: E402
    _dense_occ_from_voxel_npz,
    _load_part_masks2d,
    _load_prompt_seg_model,
    _mask_morphology,
)
from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SS_DECODER_CKPT,
    _command_for_sample,
    _find_dataset_sample,
    _load_datasets,
    _run_dir_for_sample,
    _sample_data_config_path,
)
from part_ss_eval_platform.eval_real_0615 import _execute, _load_coords, render_preview_voxel  # noqa: E402
from scripts.eval.tasks.ee_part_stage_cache import (  # noqa: E402
    PART_STAGE_SIGNATURE_NAME,
    build_part_stage_signature,
    clear_part_stage_outputs,
    part_stage_cache_status,
    part_stage_outputs_status,
)
from scripts.eval.post.joint_boundary_diagnostics import run_joint_boundary_diagnostics  # noqa: E402
from scripts.eval.post.part_connected_components import (  # noqa: E402
    filter_part_connected_components as _filter_part_connected_components,
    unique_coords as _unique_coords,
)


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402
from trellis.renderers.mesh_renderer import MeshRenderer  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-128ee")
DEFAULT_SPLIT_JSON = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)
DEFAULT_OBJECT_ID = "05a035c3347645b8a7ceb6d65f825ac3"
DEFAULT_DATASET_ID = "phyx-verse"
DEFAULT_ANGLE = 0
DEFAULT_PART_SEG_CKPT = Path(
    "/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg_full_S_0618-1/ckpts/step_100000.pt"
)
DEFAULT_SS_FLOW_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
)
DEFAULT_GAUSSIAN_DECODER = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)

GS_PRESET = {
    "name": "scaleq997_abs020_scale0.75_opx1.5",
    "max_scale_quantile": 0.997,
    "max_scale_abs": 0.020,
    "scale_mult": 0.75,
    "opacity_mult": 1.5,
    "kernel_size": 0.05,
}


def _safe_name(value: str, max_len: int = 80) -> str:
    chars: list[str] = []
    for ch in str(value):
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
        else:
            chars.append(f"_u{ord(ch):04x}_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return (out or "part")[:max_len]


def _usd_identifier(value: str, fallback: str = "mesh", max_len: int = 80) -> str:
    chars = []
    for ch in str(value):
        if ch == "_" or ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            chars.append(ch)
        else:
            chars.append("_")
    out = "_".join(part for part in "".join(chars).split("_") if part)
    if not out:
        out = fallback
    if out[0].isdigit():
        out = f"{fallback}_{out}"
    return out[:max_len]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        child = None
        for child in elem:
            _indent_xml(child, level + 1)
        if child is not None and (not child.tail or not child.tail.strip()):
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _xml_float_list(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


MUJOCO_RD_OBJECT_QUAT = "0.707106781 0.707106781 0 0"
MUJOCO_RD_DRAWER_AXIS = "0 0 1"
MUJOCO_RD_DRAWER_RANGE_MAX = 0.381732268
MUJOCO_RD_DRAWER_MASS = 0.05
MUJOCO_RD_DRAWER_DIAGINERTIA = "0.001 0.001 0.001"


def _xml_precise_float_list(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _part_component_label(part_idx: int, part: dict[str, Any]) -> str:
    return f"part_{int(part_idx):02d}_{_safe_name(str(part['part_name']))}"


def _mujoco_motion_meta(part_idx: int, part: dict[str, Any]) -> dict[str, Any]:
    target_part = part.get("target_part") if isinstance(part.get("target_part"), dict) else {}
    motion = target_part.get("motion") if isinstance(target_part.get("motion"), dict) else {}
    joint = str(target_part.get("joint") or part.get("joint") or motion.get("motion_type") or "").lower()
    joint_type = str(target_part.get("joint_type") or part.get("joint_type") or "").upper()
    is_drawer_slide = joint == "prismatic" or joint_type == "B"
    return {
        "part_index": int(part_idx),
        "part_name": str(part.get("part_name") or target_part.get("name") or ""),
        "semantic_type": str(target_part.get("type") or target_part.get("item_name") or ""),
        "source_joint": joint,
        "source_joint_type": joint_type,
        "source_motion": motion,
        "is_drawer_slide": bool(is_drawer_slide),
        "mjcf_joint_axis": [0.0, 0.0, 1.0] if is_drawer_slide else None,
        "mjcf_range": [0.0, MUJOCO_RD_DRAWER_RANGE_MAX] if is_drawer_slide else None,
        "rule": (
            "R-D MuJoCo export: RealAppliance prismatic/B joints become local +Z slide joints; "
            "raw/GT meshes are not used"
            if is_drawer_slide
            else "fixed visual part"
        ),
    }


def _mesh_bbox_center(vertices: np.ndarray) -> tuple[float, float, float]:
    if hasattr(vertices, "detach"):
        points = vertices.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        points = np.asarray(vertices, dtype=np.float64)
    if points.size == 0:
        return (0.0, 0.0, 0.0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    return (float(center[0]), float(center[1]), float(center[2]))


def _obj_bbox_center(path: Path) -> tuple[float, float, float]:
    lo = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    hi = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    count = 0
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("v "):
            continue
        fields = raw.split()
        if len(fields) < 4:
            continue
        point = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
        lo = np.minimum(lo, point)
        hi = np.maximum(hi, point)
        count += 1
    if count == 0:
        return (0.0, 0.0, 0.0)
    center = (lo + hi) * 0.5
    return (float(center[0]), float(center[1]), float(center[2]))


def _read_textured_obj(path: Path) -> dict[str, Any]:
    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[tuple[list[int], list[int | None]]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[0] == "vt" and len(fields) >= 3:
            texcoords.append((float(fields[1]), float(fields[2])))
        elif fields[0] == "f" and len(fields) >= 4:
            v_idx: list[int] = []
            vt_idx: list[int | None] = []
            for token in fields[1:]:
                parts = token.split("/")
                vi = int(parts[0])
                v_idx.append(vi - 1 if vi > 0 else len(vertices) + vi)
                if len(parts) >= 2 and parts[1]:
                    ti = int(parts[1])
                    vt_idx.append(ti - 1 if ti > 0 else len(texcoords) + ti)
                else:
                    vt_idx.append(None)
            faces.append((v_idx, vt_idx))
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "texcoords": np.asarray(texcoords, dtype=np.float64),
        "faces": faces,
    }


def _connected_face_components(
    faces: list[tuple[list[int], list[int | None]]],
    face_indices: np.ndarray,
) -> list[list[int]]:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if face_indices.size == 0:
        return []
    parent = np.arange(face_indices.size, dtype=np.int64)
    rank = np.zeros(face_indices.size, dtype=np.int8)

    def find(x: int) -> int:
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    first_face_for_vertex: dict[int, int] = {}
    for local_idx, face_idx in enumerate(face_indices.tolist()):
        v_idx, _vt_idx = faces[int(face_idx)]
        for vertex_idx in v_idx:
            previous = first_face_for_vertex.get(int(vertex_idx))
            if previous is None:
                first_face_for_vertex[int(vertex_idx)] = local_idx
            else:
                union(local_idx, previous)

    comps: dict[int, list[int]] = {}
    for local_idx, face_idx in enumerate(face_indices.tolist()):
        comps.setdefault(find(local_idx), []).append(int(face_idx))
    return sorted(comps.values(), key=len, reverse=True)


def _sample_obj_face_texture_colors(
    mesh: dict[str, Any],
    texture_path: Path,
) -> np.ndarray:
    faces = mesh["faces"]
    texcoords = np.asarray(mesh["texcoords"], dtype=np.float64)
    if texcoords.size == 0 or not Path(texture_path).is_file():
        return np.zeros((len(faces), 3), dtype=np.float32)
    texture = np.asarray(Image.open(texture_path).convert("RGB"), dtype=np.float32) / 255.0
    height, width = texture.shape[:2]
    colors = np.zeros((len(faces), 3), dtype=np.float32)
    for face_idx, (_v_idx, vt_idx) in enumerate(faces):
        valid = [texcoords[int(idx)] for idx in vt_idx if idx is not None and 0 <= int(idx) < len(texcoords)]
        if not valid:
            continue
        uv = np.asarray(valid, dtype=np.float64).mean(axis=0)
        u = float(np.clip(uv[0], 0.0, 1.0))
        v = float(np.clip(uv[1], 0.0, 1.0))
        x = int(round(u * (width - 1)))
        y = int(round((1.0 - v) * (height - 1)))
        colors[face_idx] = texture[y, x, :3]
    return colors


def _write_obj_subset(
    *,
    path: Path,
    mesh: dict[str, Any],
    face_indices: list[int] | np.ndarray,
    mtllib: str | None,
    material: str | None,
    keep_uv: bool,
) -> dict[str, int]:
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    texcoords = np.asarray(mesh["texcoords"], dtype=np.float64)
    faces = mesh["faces"]
    face_indices = [int(idx) for idx in face_indices]
    vertex_map: dict[int, int] = {}
    texcoord_map: dict[int, int] = {}
    used_vertices: list[int] = []
    used_texcoords: list[int] = []

    for face_idx in face_indices:
        v_idx, vt_idx = faces[face_idx]
        for idx in v_idx:
            if int(idx) not in vertex_map:
                vertex_map[int(idx)] = len(used_vertices) + 1
                used_vertices.append(int(idx))
        if keep_uv:
            for idx in vt_idx:
                if idx is not None and int(idx) not in texcoord_map:
                    texcoord_map[int(idx)] = len(used_texcoords) + 1
                    used_texcoords.append(int(idx))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# R-D MuJoCo postprocess OBJ subset; vertex coordinates are not rotated or rebaked\n")
        if mtllib:
            handle.write(f"mtllib {mtllib}\n")
        if material:
            handle.write(f"usemtl {material}\n")
        for idx in used_vertices:
            x, y, z = vertices[idx]
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        if keep_uv:
            for idx in used_texcoords:
                u, v = texcoords[idx]
                handle.write(f"vt {u:.8f} {v:.8f}\n")
        for face_idx in face_indices:
            v_idx, vt_idx = faces[face_idx]
            tokens = []
            for old_v, old_vt in zip(v_idx, vt_idx):
                new_v = vertex_map[int(old_v)]
                if keep_uv and old_vt is not None and int(old_vt) in texcoord_map:
                    tokens.append(f"{new_v}/{texcoord_map[int(old_vt)]}")
                else:
                    tokens.append(str(new_v))
            handle.write("f " + " ".join(tokens) + "\n")
    return {"vertices": int(len(used_vertices)), "faces": int(len(face_indices))}


def _extract_drawer_glass_from_body_mesh(
    *,
    mujoco_assets_dir: Path,
    body_item: dict[str, Any],
    drawer_item: dict[str, Any],
) -> dict[str, Any] | None:
    texture_file = body_item.get("texture_file")
    if not texture_file:
        return None
    body_mesh_path = Path(str(body_item["mesh_path"]))
    texture_path = mujoco_assets_dir / str(texture_file).removeprefix("assets/")
    if not body_mesh_path.is_file() or not texture_path.is_file():
        return None

    mesh = _read_textured_obj(body_mesh_path)
    faces = mesh["faces"]
    if not faces:
        return None
    colors = _sample_obj_face_texture_colors(mesh, texture_path)
    red = colors[:, 0]
    green = colors[:, 1]
    blue = colors[:, 2]
    brightness = colors.mean(axis=1)
    blue_glass_face = (
        (blue > 0.38)
        & (green > 0.30)
        & (brightness > 0.30)
        & (blue > red + 0.08)
        & (green > red + 0.02)
    )
    if int(blue_glass_face.sum()) < 64:
        return None

    selected_faces: list[int] = []
    full_components = _connected_face_components(faces, np.arange(len(faces), dtype=np.int64))
    component_reports: list[dict[str, Any]] = []
    for comp in full_components:
        comp_idx = np.asarray(comp, dtype=np.int64)
        if comp_idx.size < 64:
            continue
        frac = float(blue_glass_face[comp_idx].mean())
        mean = colors[comp_idx].mean(axis=0)
        report = {
            "faces": int(comp_idx.size),
            "blue_face_fraction": frac,
            "mean_rgb": [float(mean[0]), float(mean[1]), float(mean[2])],
        }
        component_reports.append(report)
        if frac >= 0.20 and mean[2] > mean[0] + 0.06 and mean[1] > mean[0] + 0.01:
            selected_faces.extend(comp)

    selection_mode = "full_connected_component"
    if not selected_faces:
        candidates = np.flatnonzero(blue_glass_face)
        candidate_components = _connected_face_components(faces, candidates)
        candidate_components = [comp for comp in candidate_components if len(comp) >= 64]
        if not candidate_components:
            return None
        selected_faces = list(candidate_components[0])
        selection_mode = "blue_texture_connected_component"

    selected = np.asarray(sorted(set(selected_faces)), dtype=np.int64)
    if selected.size == 0 or selected.size >= len(faces):
        return None
    keep = np.setdiff1d(np.arange(len(faces), dtype=np.int64), selected, assume_unique=True)
    body_counts = _write_obj_subset(
        path=body_mesh_path,
        mesh=mesh,
        face_indices=keep,
        mtllib="material.mtl",
        material="material_0",
        keep_uv=True,
    )
    glass_dir = mujoco_assets_dir / "drawer_glass"
    glass_mesh_path = glass_dir / "mesh.obj"
    glass_counts = _write_obj_subset(
        path=glass_mesh_path,
        mesh=mesh,
        face_indices=selected,
        mtllib=None,
        material=None,
        keep_uv=False,
    )
    body_item["vertices"] = int(body_counts["vertices"])
    body_item["faces"] = int(body_counts["faces"])
    body_item["glass_removed_faces"] = int(selected.size)
    drawer_label = str(drawer_item["label"])
    return {
        "role": "drawer_glass",
        "label": "drawer_glass",
        "parent_label": drawer_label,
        "mesh_file": "assets/drawer_glass/mesh.obj",
        "mesh_path": str(glass_mesh_path.resolve()),
        "texture_file": None,
        "material_file": None,
        "vertices": int(glass_counts["vertices"]),
        "faces": int(glass_counts["faces"]),
        "appearance_source": "light-blue connected component cut from predicted body texture; no GT mesh",
        "selection_mode": selection_mode,
        "source_body_mesh": str(body_mesh_path.resolve()),
        "source_texture": str(texture_path.resolve()),
        "selected_faces": int(selected.size),
        "component_reports": component_reports[:8],
    }


def _apply_mujoco_rd_postprocess(
    *,
    mujoco_assets_dir: Path,
    mesh_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body_item = next((item for item in mesh_items if str(item.get("role")) == "body"), None)
    drawer_items = [
        item
        for item in mesh_items
        if str(item.get("role")) == "part" and bool((item.get("motion") or {}).get("is_drawer_slide", False))
    ]
    glass_item = None
    if body_item is not None and drawer_items:
        glass_item = _extract_drawer_glass_from_body_mesh(
            mujoco_assets_dir=mujoco_assets_dir,
            body_item=body_item,
            drawer_item=drawer_items[0],
        )
    out_items = list(mesh_items)
    if glass_item is not None:
        out_items.append(glass_item)
    report = {
        "enabled": True,
        "name": "mujoco_r_d_root_rotation_drawer_slide",
        "object_quat": MUJOCO_RD_OBJECT_QUAT,
        "drawer_axis": [0.0, 0.0, 1.0],
        "drawer_range": [0.0, MUJOCO_RD_DRAWER_RANGE_MAX],
        "drawer_count": int(len(drawer_items)),
        "drawer_labels": [str(item.get("label")) for item in drawer_items],
        "glass_extracted": glass_item is not None,
        "glass_item": glass_item,
        "no_gt_mesh": True,
        "rule": (
            "Do not rotate/bake OBJ vertices. Put Z-forward correction on root object body quat; "
            "prismatic RealAppliance parts get local +Z slide joints, actuator, and home keyframe; "
            "light-blue drawer glass is cut from predicted body mesh and parented under the drawer body."
        ),
    }
    return out_items, report


def _write_static_mujoco_xml(
    *,
    out_xml: Path,
    model_name: str,
    mesh_items: list[dict[str, Any]],
) -> Path:
    has_textures = any(item.get("texture_file") for item in mesh_items)
    root = ET.Element("mujoco", {"model": _safe_name(model_name, 120)})
    compiler = {
        "angle": "radian",
        "meshdir": ".",
        "texturedir": ".",
        "balanceinertia": "true",
        "inertiagrouprange": "3 5",
    }
    ET.SubElement(root, "compiler", compiler)
    ET.SubElement(root, "statistic", {"center": "0 0 0", "extent": "1.6", "meansize": "0.05"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.81"})
    default = ET.SubElement(root, "default")
    default_geom = {
        "type": "mesh",
        "group": "0",
        "contype": "0",
        "conaffinity": "0",
    }
    if not has_textures:
        default_geom["rgba"] = "0.72 0.76 0.80 1"
    ET.SubElement(default, "geom", default_geom)
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "top", "pos": "0 0 3", "dir": "0 0 -1"})
    object_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "object", "pos": "0 0 0", "quat": MUJOCO_RD_OBJECT_QUAT},
    )

    part_colors = [
        (0.80, 0.16, 0.18, 1.0),
        (0.13, 0.47, 0.70, 1.0),
        (0.17, 0.63, 0.17, 1.0),
        (0.58, 0.40, 0.74, 1.0),
        (1.00, 0.50, 0.05, 1.0),
        (0.55, 0.34, 0.29, 1.0),
        (0.89, 0.47, 0.76, 1.0),
        (0.50, 0.50, 0.50, 1.0),
    ]
    asset_bindings: dict[int, tuple[str, str | None]] = {}
    for item in mesh_items:
        label = _safe_name(str(item["label"]), 80)
        mesh_name = f"{label}_mesh"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(item["mesh_file"])})
        material_name = None
        if item.get("texture_file"):
            texture_name = f"{label}_texture"
            material_name = f"{label}_material"
            ET.SubElement(
                asset,
                "texture",
                {
                    "name": texture_name,
                    "type": "2d",
                    "file": str(item["texture_file"]),
                },
            )
            ET.SubElement(
                asset,
                "material",
                {
                    "name": material_name,
                    "texture": texture_name,
                    "rgba": "1 1 1 1",
                    "specular": "0",
                    "shininess": "0",
                    "reflectance": "0.08",
                },
            )
        elif str(item.get("role")) == "drawer_glass":
            material_name = f"{label}_material"
            ET.SubElement(
                asset,
                "material",
                {
                    "name": material_name,
                    "rgba": "0.55 0.82 1 0.45",
                    "specular": "0.2",
                    "shininess": "0.1",
                    "reflectance": "0.02",
                },
            )
        asset_bindings[id(item)] = (mesh_name, material_name)

    part_idx = 0
    part_bodies: dict[str, ET.Element] = {}
    actuator_specs: list[tuple[str, str]] = []
    qpos_home: list[str] = []
    for item in mesh_items:
        role = str(item.get("role") or "part")
        if role == "drawer_glass":
            continue
        label = _safe_name(str(item["label"]), 80)
        mesh_name, material_name = asset_bindings[id(item)]
        if role == "body":
            body = object_body
            geom_name = "body_visual"
            rgba = (0.72, 0.76, 0.80, 0.35)
        else:
            body = ET.SubElement(object_body, "body", {"name": label, "pos": "0 0 0"})
            part_bodies[str(item["label"])] = body
            motion = item.get("motion") or {}
            if bool(motion.get("is_drawer_slide", False)):
                center = item.get("bbox_center") or (0.0, 0.0, 0.0)
                center_text = _xml_precise_float_list(center)
                ET.SubElement(
                    body,
                    "inertial",
                    {
                        "pos": center_text,
                        "mass": f"{MUJOCO_RD_DRAWER_MASS:.6g}",
                        "diaginertia": MUJOCO_RD_DRAWER_DIAGINERTIA,
                    },
                )
                part_index = int(motion.get("part_index", part_idx))
                joint_name = f"j_{part_index:02d}_00"
                actuator_name = f"a_{part_index:02d}_00"
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": joint_name,
                        "type": "slide",
                        "pos": center_text,
                        "axis": MUJOCO_RD_DRAWER_AXIS,
                        "range": f"0 {MUJOCO_RD_DRAWER_RANGE_MAX:.9g}",
                        "limited": "true",
                        "damping": "5",
                    },
                )
                actuator_specs.append((actuator_name, joint_name))
                qpos_home.append("0")
            geom_name = f"{label}_visual"
            rgba = part_colors[part_idx % len(part_colors)]
            part_idx += 1
        geom = {"name": geom_name, "mesh": mesh_name}
        if material_name is not None:
            geom["material"] = material_name
        else:
            geom["rgba"] = _xml_float_list(rgba)
        ET.SubElement(body, "geom", geom)

    for item in mesh_items:
        if str(item.get("role")) != "drawer_glass":
            continue
        parent = part_bodies.get(str(item.get("parent_label")))
        if parent is None:
            continue
        label = _safe_name(str(item["label"]), 80)
        mesh_name, material_name = asset_bindings[id(item)]
        geom = {"name": f"{label}_visual", "mesh": mesh_name}
        if material_name is not None:
            geom["material"] = material_name
        else:
            geom["rgba"] = "0.55 0.82 1 0.45"
        ET.SubElement(parent, "geom", geom)

    if actuator_specs:
        actuator = ET.SubElement(root, "actuator")
        for actuator_name, joint_name in actuator_specs:
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": actuator_name,
                    "joint": joint_name,
                    "ctrlrange": f"0 {MUJOCO_RD_DRAWER_RANGE_MAX:.9g}",
                    "ctrllimited": "true",
                    "kp": "5",
                },
            )
        keyframe = ET.SubElement(root, "keyframe")
        ET.SubElement(keyframe, "key", {"name": "home", "qpos": " ".join(qpos_home)})

    _indent_xml(root)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out_xml


def _write_hidden_color_fill_mujoco_xml(
    *,
    summary: dict[str, Any],
    report: dict[str, Any],
    out_xml: Path,
    model_name: str,
) -> Path:
    after_by_component = {
        str(row["component"]): Path(str(row["after_mesh"])).resolve()
        for row in report.get("metrics") or []
        if row.get("component") and row.get("after_mesh")
    }
    mesh_items: list[dict[str, Any]] = []

    def _rel_mesh(path: Path) -> str:
        return os.path.relpath(str(path), start=str(out_xml.parent.resolve()))

    body = summary.get("mujoco_body_mesh") or {}
    body_path = after_by_component.get("body_without_parts")
    if body and body_path is not None and body_path.is_file():
        item = dict(body)
        item["mesh_file"] = _rel_mesh(body_path)
        item["mesh_path"] = str(body_path)
        item["texture_file"] = None
        item["material_file"] = None
        item["appearance_source"] = "decoded_mesh_vertex_colors_obj_hidden_color_filled"
        mesh_items.append(item)

    for part in summary.get("mujoco_part_meshes") or []:
        label = str(part.get("label") or "")
        part_path = after_by_component.get(label)
        if not label or part_path is None or not part_path.is_file():
            continue
        item = dict(part)
        item["mesh_file"] = _rel_mesh(part_path)
        item["mesh_path"] = str(part_path)
        item["texture_file"] = None
        item["material_file"] = None
        item["appearance_source"] = "decoded_mesh_vertex_colors_obj_hidden_color_filled"
        mesh_items.append(item)

    if not mesh_items:
        raise ValueError(f"{out_xml}: hidden color fill did not produce any MuJoCo mesh items")
    return _write_static_mujoco_xml(out_xml=out_xml, model_name=model_name, mesh_items=mesh_items)


def _save_textured_mujoco_mesh(
    *,
    gaussian: Any,
    mesh: Any,
    asset_dir: Path,
    stem: str,
    texture_size: int,
    render_resolution: int,
    nviews: int,
    mode: str,
) -> dict[str, Any]:
    """Bake decoded Gaussian appearance onto decoded mesh UV texture for MuJoCo."""
    import trimesh  # noqa: PLC0415
    from trellis.utils import postprocessing_utils, render_utils  # noqa: PLC0415

    vertices_t = getattr(mesh, "vertices", None)
    faces_t = getattr(mesh, "faces", None)
    if vertices_t is None or faces_t is None:
        raise TypeError(f"{stem}: decoded mesh lacks vertices/faces")
    vertices = vertices_t.detach().float().cpu().numpy() if torch.is_tensor(vertices_t) else np.asarray(vertices_t)
    faces = faces_t.detach().long().cpu().numpy() if torch.is_tensor(faces_t) else np.asarray(faces_t)
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"{stem}: decoded mesh is empty")

    vertices_before = int(vertices.shape[0])
    faces_before = int(faces.shape[0])
    simplify_ratio = 0.75 if faces_before > 200_000 else 0.0
    if simplify_ratio > 0:
        vertices, faces = postprocessing_utils.postprocess_mesh(
            vertices,
            faces.astype(np.int32, copy=False),
            simplify=True,
            simplify_ratio=float(simplify_ratio),
            fill_holes=False,
            verbose=False,
        )
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces.astype(np.int32, copy=False))
    size = int(texture_size)
    resolution = int(render_resolution)
    view_count = int(nviews)
    alpha_threshold = 0.12
    yaws: list[float] = []
    pitchs: list[float] = []
    for idx in range(view_count):
        yaw, pitch = render_utils.sphere_hammersley_sequence(idx, view_count)
        yaws.append(float(yaw))
        pitchs.append(float(pitch))
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 2, 40)
    renderer_black = _make_gaussian_renderer(resolution)
    renderer_white = _make_gaussian_renderer(resolution)
    renderer_black.rendering_options.bg_color = (0, 0, 0)
    renderer_white.rendering_options.bg_color = (1, 1, 1)
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    alpha_fracs: list[float] = []
    alpha_means: list[float] = []
    with torch.no_grad():
        for extrinsic, intrinsic in zip(extrinsics, intrinsics):
            black = renderer_black.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
            white = renderer_white.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
            black_np = black.permute(1, 2, 0).numpy()
            white_np = white.permute(1, 2, 0).numpy()
            alpha = np.clip(1.0 - (white_np - black_np).mean(axis=-1), 0.0, 1.0)
            color = black_np / np.maximum(alpha[..., None], 1e-4)
            mask = alpha > alpha_threshold
            alpha_fracs.append(float(mask.mean()))
            alpha_means.append(float(alpha[mask].mean()) if np.any(mask) else 0.0)
            observations.append(np.clip(color * 255.0, 0, 255).astype(np.uint8))
            masks.append(mask)

    texture = postprocessing_utils.bake_texture(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        np.asarray(uvs, dtype=np.float32),
        observations,
        masks,
        [extrinsics[i].detach().cpu().numpy() for i in range(len(extrinsics))],
        [intrinsics[i].detach().cpu().numpy() for i in range(len(intrinsics))],
        texture_size=size,
        mode=str(mode),
        lambda_tv=0.01,
        verbose=False,
    )

    vertices_y_up = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    material = trimesh.visual.material.SimpleMaterial(image=Image.fromarray(texture))
    textured = trimesh.Trimesh(
        vertices=np.asarray(vertices_y_up),
        faces=np.asarray(faces),
        visual=trimesh.visual.TextureVisuals(uv=np.asarray(uvs), material=material),
        process=False,
    )
    mesh_dir = asset_dir / stem
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / "mesh.obj"
    textured.export(obj_path)
    mtl_path = mesh_dir / "material.mtl"
    texture_path = mesh_dir / "material_0.png"
    if not mtl_path.is_file() or not texture_path.is_file():
        raise FileNotFoundError(f"{stem}: textured OBJ export did not create material.mtl/material_0.png")
    return {
        "mesh": f"{stem}/mesh.obj",
        "material": f"{stem}/material.mtl",
        "texture": f"{stem}/material_0.png",
        "bake_stats": {
            "source": "decoded_gaussian_multiview_render",
            "backend": "TRELLIS postprocessing_utils.bake_texture",
            "texture_size": int(size),
            "render_resolution": int(resolution),
            "nviews": int(view_count),
            "mode": str(mode),
            "alpha_threshold": float(alpha_threshold),
            "alpha_mask_fraction_minmax": [float(min(alpha_fracs)), float(max(alpha_fracs))],
            "alpha_visible_mean_minmax": [float(min(alpha_means)), float(max(alpha_means))],
            "texture_mean": [float(x) for x in texture.reshape(-1, 3).mean(axis=0).tolist()],
            "texture_std": [float(x) for x in texture.reshape(-1, 3).std(axis=0).tolist()],
            "mesh_vertices_before": int(vertices_before),
            "mesh_faces_before": int(faces_before),
            "mesh_simplify_target_reduction": float(simplify_ratio),
            "uv_vertices": int(np.asarray(vertices).shape[0]),
            "uv_faces": int(np.asarray(faces).shape[0]),
        },
    }


def _mesh_vertex_colors(mesh: Any, vertex_count: int, *, label: str) -> np.ndarray:
    attrs_t = getattr(mesh, "vertex_attrs", None)
    if attrs_t is None:
        raise TypeError(f"{label}: decoded mesh has no vertex_attrs for MuJoCo mesh-color texture export")
    colors = attrs_t.detach().float().cpu().numpy() if torch.is_tensor(attrs_t) else np.asarray(attrs_t)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[0] != int(vertex_count) or colors.shape[1] < 3:
        raise ValueError(f"{label}: vertex_attrs shape {colors.shape} does not match vertices={vertex_count}")
    colors = colors[:, :3]
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _fill_texture_holes(texture: np.ndarray, filled: np.ndarray, fallback_color: np.ndarray) -> np.ndarray:
    if bool(filled.all()):
        return texture
    if not bool(filled.any()):
        texture[...] = np.asarray(fallback_color, dtype=np.float32)[None, None, :]
        return texture
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

        nearest = distance_transform_edt(~filled, return_distances=False, return_indices=True)
        texture[~filled] = texture[nearest[0][~filled], nearest[1][~filled]]
        return texture
    except Exception:
        # Small fallback when scipy is unavailable: iterative nearest-neighbor dilation.
        out = texture.copy()
        mask = filled.copy()
        for _ in range(64):
            if bool(mask.all()):
                break
            new_out = out.copy()
            new_mask = mask.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                src_y = slice(max(0, -dy), mask.shape[0] - max(0, dy))
                src_x = slice(max(0, -dx), mask.shape[1] - max(0, dx))
                dst_y = slice(max(0, dy), mask.shape[0] - max(0, -dy))
                dst_x = slice(max(0, dx), mask.shape[1] - max(0, -dx))
                take = (~new_mask[dst_y, dst_x]) & mask[src_y, src_x]
                dst_block = new_out[dst_y, dst_x]
                dst_block[take] = out[src_y, src_x][take]
                new_out[dst_y, dst_x] = dst_block
                new_mask[dst_y, dst_x] |= mask[src_y, src_x]
            out, mask = new_out, new_mask
        out[~mask] = np.asarray(fallback_color, dtype=np.float32)[None, :]
        return out


def _rasterize_vertex_color_texture(
    *,
    uvs: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    texture_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = int(texture_size)
    if size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}")
    uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if uvs.shape[0] != colors.shape[0]:
        raise ValueError(f"uv/color vertex count mismatch: uvs={uvs.shape}, colors={colors.shape}")

    accum = np.zeros((size, size, 3), dtype=np.float32)
    weights = np.zeros((size, size), dtype=np.float32)
    pix = np.stack(
        [
            np.clip(uvs[:, 0], 0.0, 1.0) * float(size - 1),
            (1.0 - np.clip(uvs[:, 1], 0.0, 1.0)) * float(size - 1),
        ],
        axis=1,
    )
    rasterized_faces = 0
    rasterized_pixels = 0
    for face in faces:
        pts = pix[face]
        cols = colors[face]
        min_x = max(0, int(np.floor(float(pts[:, 0].min()))) - 1)
        max_x = min(size - 1, int(np.ceil(float(pts[:, 0].max()))) + 1)
        min_y = max(0, int(np.floor(float(pts[:, 1].min()))) - 1)
        max_y = min(size - 1, int(np.ceil(float(pts[:, 1].max()))) + 1)
        if max_x < min_x or max_y < min_y:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        p = np.stack([xx, yy], axis=-1)
        a, b, c = pts.astype(np.float32)
        v0 = b - a
        v1 = c - a
        v2 = p - a
        denom = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(denom) < 1.0e-8:
            continue
        inv = 1.0 / denom
        w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) * inv
        w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) * inv
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1.0e-4) & (w1 >= -1.0e-4) & (w2 >= -1.0e-4)
        if not bool(inside.any()):
            continue
        color = w0[..., None] * cols[0] + w1[..., None] * cols[1] + w2[..., None] * cols[2]
        yi, xi = np.nonzero(inside)
        target_y = yi + min_y
        target_x = xi + min_x
        accum[target_y, target_x] += color[inside]
        weights[target_y, target_x] += 1.0
        rasterized_faces += 1
        rasterized_pixels += int(inside.sum())

    filled = weights > 0
    texture = np.zeros_like(accum)
    texture[filled] = accum[filled] / weights[filled, None]
    fallback = colors.mean(axis=0) if colors.size else np.array([0.72, 0.76, 0.80], dtype=np.float32)
    texture = _fill_texture_holes(texture, filled, fallback)
    texture_u8 = np.clip(texture * 255.0, 0, 255).round().astype(np.uint8)
    return texture_u8, {
        "texture_size": int(size),
        "rasterized_faces": int(rasterized_faces),
        "rasterized_pixels": int(rasterized_pixels),
        "filled_fraction_before_hole_fill": float(filled.mean()),
        "texture_mean": [float(x) for x in texture.reshape(-1, 3).mean(axis=0).tolist()],
        "texture_std": [float(x) for x in texture.reshape(-1, 3).std(axis=0).tolist()],
    }


def _save_vertex_color_textured_mujoco_mesh(
    *,
    mesh: Any,
    asset_dir: Path,
    stem: str,
    texture_size: int,
) -> dict[str, Any]:
    """Convert decoded mesh.vertex_attrs into a regular textured OBJ for MuJoCo."""
    vertices_t = getattr(mesh, "vertices", None)
    faces_t = getattr(mesh, "faces", None)
    if vertices_t is None or faces_t is None:
        raise TypeError(f"{stem}: decoded mesh lacks vertices/faces")
    vertices = vertices_t.detach().float().cpu().numpy() if torch.is_tensor(vertices_t) else np.asarray(vertices_t)
    faces = faces_t.detach().long().cpu().numpy() if torch.is_tensor(faces_t) else np.asarray(faces_t)
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"{stem}: decoded mesh is empty")
    colors = _mesh_vertex_colors(mesh, int(vertices.shape[0]), label=stem)

    face_count = int(faces.shape[0])
    cell_size = 3
    min_size = int(np.ceil(np.sqrt(float(face_count))) * cell_size)
    size = 1
    while size < max(int(texture_size), min_size):
        size *= 2
    cols = max(1, size // cell_size)
    rows = int(np.ceil(face_count / float(cols)))
    if rows * cell_size > size:
        size *= 2
        cols = max(1, size // cell_size)
        rows = int(np.ceil(face_count / float(cols)))

    face_colors = np.clip(colors[faces].mean(axis=1), 0.0, 1.0)
    fallback = np.clip(colors.mean(axis=0), 0.0, 1.0)
    texture = np.broadcast_to((fallback * 255.0).round().astype(np.uint8), (size, size, 3)).copy()
    for idx, color in enumerate(face_colors):
        row = idx // cols
        col = idx - row * cols
        y0 = row * cell_size
        x0 = col * cell_size
        texture[y0 : y0 + cell_size, x0 : x0 + cell_size] = np.clip(color * 255.0, 0, 255).round().astype(np.uint8)

    expanded_vertices = vertices[faces.reshape(-1)]
    vertices_y_up = expanded_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    uvs = np.empty((face_count * 3, 2), dtype=np.float32)
    inv = 1.0 / float(size)
    for idx in range(face_count):
        row = idx // cols
        col = idx - row * cols
        x0 = float(col * cell_size)
        y0 = float(row * cell_size)
        base = idx * 3
        uvs[base + 0] = ((x0 + 0.5) * inv, 1.0 - (y0 + 0.5) * inv)
        uvs[base + 1] = ((x0 + cell_size - 0.5) * inv, 1.0 - (y0 + 0.5) * inv)
        uvs[base + 2] = ((x0 + 0.5) * inv, 1.0 - (y0 + cell_size - 0.5) * inv)

    mesh_dir = asset_dir / stem
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / "mesh.obj"
    mtl_path = mesh_dir / "material.mtl"
    texture_path = mesh_dir / "material_0.png"
    Image.fromarray(texture).save(texture_path)
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl material_0",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "Ns 1.000000",
                "d 1.000000",
                "illum 2",
                "map_Kd material_0.png",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with obj_path.open("w", encoding="utf-8") as f:
        f.write("mtllib material.mtl\n")
        f.write("usemtl material_0\n")
        for x, y, z in vertices_y_up:
            f.write(f"v {float(x):.8g} {float(y):.8g} {float(z):.8g}\n")
        for u, v in uvs:
            f.write(f"vt {float(u):.8g} {float(v):.8g}\n")
        for idx in range(face_count):
            a = idx * 3 + 1
            f.write(f"f {a}/{a} {a + 1}/{a + 1} {a + 2}/{a + 2}\n")
    if not mtl_path.is_file() or not texture_path.is_file():
        raise FileNotFoundError(f"{stem}: textured OBJ export did not create material.mtl/material_0.png")
    return {
        "mesh": f"{stem}/mesh.obj",
        "material": f"{stem}/material.mtl",
        "texture": f"{stem}/material_0.png",
        "bake_stats": {
            "source": "decoded_mesh_vertex_attrs",
            "backend": "per_face_uv_atlas_vertex_attr_texture",
            "requested_texture_size": int(texture_size),
            "texture_size": int(size),
            "atlas_cell_size": int(cell_size),
            "atlas_cols": int(cols),
            "atlas_rows": int(rows),
            "mesh_vertices_before": int(vertices.shape[0]),
            "mesh_faces_before": int(faces.shape[0]),
            "uv_vertices": int(vertices_y_up.shape[0]),
            "uv_faces": int(face_count),
            "texture_mean": [float(x) for x in (texture.reshape(-1, 3).mean(axis=0) / 255.0).tolist()],
            "texture_std": [float(x) for x in (texture.reshape(-1, 3).std(axis=0) / 255.0).tolist()],
        },
    }


def _usd_array(items: list[str], *, indent: str, per_line: int = 1) -> str:
    if not items:
        return "[]"
    lines = ["["]
    for idx in range(0, len(items), int(per_line)):
        lines.append(f"{indent}{', '.join(items[idx:idx + int(per_line)])},")
    lines.append("]")
    return "\n".join(lines)


def _usd_points(vertices: np.ndarray) -> list[str]:
    return [f"({float(x):.8g}, {float(y):.8g}, {float(z):.8g})" for x, y, z in vertices]


def _usd_colors(colors: np.ndarray) -> list[str]:
    return [f"({float(r):.6g}, {float(g):.6g}, {float(b):.6g})" for r, g, b in colors]


def _usd_indices(faces: np.ndarray) -> list[str]:
    return [str(int(value)) for value in faces.reshape(-1)]


def _mesh_to_usd_payload(mesh: Any, *, label: str, role: str) -> dict[str, Any]:
    vertices_t = getattr(mesh, "vertices", None)
    faces_t = getattr(mesh, "faces", None)
    attrs_t = getattr(mesh, "vertex_attrs", None)
    if vertices_t is None or faces_t is None:
        raise TypeError(f"{label}: decoded mesh lacks vertices/faces")
    if attrs_t is None:
        raise TypeError(f"{label}: decoded mesh lacks vertex_attrs for USD displayColor")
    vertices = vertices_t.detach().float().cpu().numpy() if torch.is_tensor(vertices_t) else np.asarray(vertices_t)
    faces = faces_t.detach().long().cpu().numpy() if torch.is_tensor(faces_t) else np.asarray(faces_t)
    colors = attrs_t.detach().float().cpu().numpy() if torch.is_tensor(attrs_t) else np.asarray(attrs_t)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[0] != vertices.shape[0] or colors.shape[1] < 3:
        raise ValueError(f"{label}: vertex_attrs shape {colors.shape} does not match vertices {vertices.shape}")
    colors = colors[:, :3]
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    vertices_y_up = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    return {
        "label": str(label),
        "role": str(role),
        "prim_name": _usd_identifier(str(label), "mesh", 96),
        "points": vertices_y_up,
        "faces": faces,
        "colors": colors,
    }


def _write_decoded_mesh_usda(*, out_usda: Path, model_name: str, mesh_items: list[dict[str, Any]]) -> Path:
    out_usda.parent.mkdir(parents=True, exist_ok=True)
    with out_usda.open("w", encoding="utf-8") as f:
        f.write("#usda 1.0\n")
        f.write("(\n")
        root = _usd_identifier(model_name, "Model", 96)
        f.write(f'    defaultPrim = "{root}"\n')
        f.write('    upAxis = "Y"\n')
        f.write("    metersPerUnit = 1\n")
        f.write(")\n\n")
        f.write(f'def Xform "{root}"\n')
        f.write("{\n")
        f.write('    def Xform "Geometry"\n')
        f.write("    {\n")
        used_prims: set[str] = set()
        for item in mesh_items:
            base_prim = _usd_identifier(str(item["prim_name"]), "mesh", 96)
            prim = base_prim
            suffix = 1
            while prim in used_prims:
                suffix += 1
                prim = _usd_identifier(f"{base_prim}_{suffix}", "mesh", 96)
            used_prims.add(prim)
            item["prim_name"] = prim
            points = np.asarray(item["points"], dtype=np.float32)
            faces = np.asarray(item["faces"], dtype=np.int64).reshape(-1, 3)
            colors = np.asarray(item["colors"], dtype=np.float32)
            f.write(f'        def Mesh "{prim}"\n')
            f.write("        {\n")
            f.write("            uniform bool doubleSided = 1\n")
            f.write("            int[] faceVertexCounts = ")
            f.write(_usd_array(["3"] * int(faces.shape[0]), indent="                ", per_line=24))
            f.write("\n")
            f.write("            int[] faceVertexIndices = ")
            f.write(_usd_array(_usd_indices(faces), indent="                ", per_line=18))
            f.write("\n")
            f.write("            point3f[] points = ")
            f.write(_usd_array(_usd_points(points), indent="                ", per_line=1))
            f.write("\n")
            f.write("            color3f[] primvars:displayColor = ")
            f.write(_usd_array(_usd_colors(colors), indent="                ", per_line=1))
            f.write("\n")
            f.write('            uniform token primvars:displayColor:interpolation = "vertex"\n')
            f.write('            rel material:binding = </Materials/DecodedVertexColor>\n')
            f.write("        }\n")
        f.write("    }\n")
        f.write("}\n\n")
        f.write('def Scope "Materials"\n')
        f.write("{\n")
        f.write('    def Material "DecodedVertexColor"\n')
        f.write("    {\n")
        f.write('        token outputs:surface.connect = </Materials/DecodedVertexColor/PreviewSurface.outputs:surface>\n')
        f.write('        def Shader "PrimvarReader"\n')
        f.write("        {\n")
        f.write('            uniform token info:id = "UsdPrimvarReader_float3"\n')
        f.write('            string inputs:varname = "displayColor"\n')
        f.write("            color3f outputs:result\n")
        f.write("        }\n")
        f.write('        def Shader "PreviewSurface"\n')
        f.write("        {\n")
        f.write('            uniform token info:id = "UsdPreviewSurface"\n')
        f.write('            color3f inputs:diffuseColor.connect = </Materials/DecodedVertexColor/PrimvarReader.outputs:result>\n')
        f.write("            float inputs:roughness = 0.55\n")
        f.write("            token outputs:surface\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("}\n")
    return out_usda


def _find_sample(ds: Any, object_id: str, angle: int, dataset_id: str) -> SimpleNamespace:
    for row in ds.samples:
        if str(row["obj_id"]) == object_id and int(row["angle_idx"]) == int(angle):
            return SimpleNamespace(
                split=str(row.get("split", "held")),
                dataset_id=dataset_id,
                obj_id=object_id,
                angle_idx=int(angle),
                data_root=str(row.get("_eval_data_root") or ds.data_root),
                manifest_path=str(row.get("_eval_manifest_path") or ds.manifest_path),
            )
    raise KeyError(f"{dataset_id}::{object_id} angle={angle} not found")


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _seed_all(seed: int | None) -> None:
    if seed is None:
        return
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i % (2**32))
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def load_camera_matrices(camera_path: Path, view_indices: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    payload = json.loads(_require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 12:
        raise ValueError(f"{camera_path}: frames must have length 12")
    if view_indices is None:
        selected = list(range(len(frames)))
    else:
        selected = list(view_indices)
        if not selected:
            raise ValueError("view_indices must be non-empty")
        bad = [idx for idx in selected if idx < 0 or idx >= len(frames)]
        if bad:
            raise ValueError(f"{camera_path}: view_indices out of range [0,{len(frames)}): {bad}")
    extrinsics = []
    intrinsics = []
    for idx in selected:
        frame = frames[idx]
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        if tuple(c2w.shape) != (4, 4):
            raise ValueError(f"{camera_path}: frame {idx} transform_matrix must be 4x4")
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        extrinsics.append(torch.inverse(c2w))
        fov = torch.tensor(float(frame["camera_angle_x"]), dtype=torch.float32, device="cuda")
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return torch.stack(extrinsics), torch.stack(intrinsics)


def _ensure_ss_and_part(args: argparse.Namespace, ds: Any, sample: SimpleNamespace, ds_sample: dict[str, Any]) -> Path:
    run_dir = _run_dir_for_sample(args.out_dir, sample)
    progress_path = args.out_dir / "progress_single.jsonl"
    expected_parts = len(ds_sample["parts"])
    ss_done = (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file()
    local_args = argparse.Namespace(**vars(args))
    local_args.data_config = str(_sample_data_config_path(args.out_dir, sample, ds))

    _seed_all(args.seed)
    if ss_done and not args.force_stage:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "ss", "status": "skipped", "run_dir": str(run_dir)}) + "\n")
    else:
        spec = _command_for_sample(args.out_dir, sample, local_args, "ss", ds)
        rec = _execute(
            spec,
            gpu=str(args.gpu),
            progress_path=progress_path,
            label=f"0617-128ee/ss/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}",
        )
        rec["status"] = "done"

    if not (run_dir / "ss_latent.npy").is_file() or not (run_dir / "voxel.npz").is_file():
        raise FileNotFoundError(f"missing SS outputs before part stage: {run_dir}")
    part_signature = build_part_stage_signature(
        part_seg_ckpt=args.part_seg_ckpt,
        ss_latent_path=run_dir / "ss_latent.npy",
        voxel_path=run_dir / "voxel.npz",
        expected_parts=expected_parts,
        joint_candidate_mode=args.part_joint_candidate_mode,
        joint_refine=bool(args.part_joint_refine),
        joint_refine_iters=int(args.part_joint_refine_iters),
        joint_refine_pairwise=float(args.part_joint_refine_pairwise),
        joint_refine_margin=float(args.part_joint_refine_margin),
        joint_refine_margin_quantile=float(args.part_joint_refine_margin_quantile),
        joint_refine_neighborhood=int(args.part_joint_refine_neighborhood),
        joint_refine_min_vote_gain=float(args.part_joint_refine_min_vote_gain),
        joint_refine_preserve_small_classes=int(args.part_joint_refine_preserve_small_classes),
        joint_save_logits=bool(args.part_joint_save_logits),
        seed=args.seed,
    )
    parts_dir = run_dir / "parts"
    part_done, cache_reason = part_stage_cache_status(
        parts_dir,
        expected_parts=expected_parts,
        expected_signature=part_signature,
    )
    _seed_all(args.seed)
    if part_done and not args.force_stage:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "stage": "part",
                        "status": "skipped",
                        "run_dir": str(run_dir),
                        "cache_reason": cache_reason,
                    }
                )
                + "\n"
            )
    else:
        removed = clear_part_stage_outputs(parts_dir)
        print(
            f"[0617-128ee] rerun part stage cache_reason={cache_reason!r} removed={removed}",
            flush=True,
        )
        spec = _command_for_sample(args.out_dir, sample, local_args, "part", ds)
        rec = _execute(
            spec,
            gpu=str(args.gpu),
            progress_path=progress_path,
            label=f"0617-128ee/part/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}",
        )
        rec["status"] = "done"
        output_ok, output_reason = part_stage_outputs_status(
            parts_dir,
            expected_parts=expected_parts,
            require_joint_logits=bool(args.part_joint_save_logits),
        )
        if not output_ok:
            raise RuntimeError(f"part stage outputs failed cache contract: {output_reason}")
        _write_json(parts_dir / PART_STAGE_SIGNATURE_NAME, part_signature)
    if not (run_dir / "voxel.npz").is_file():
        raise FileNotFoundError(f"missing whole voxel after ss stage: {run_dir / 'voxel.npz'}")
    return run_dir


def _rgba_view_image(data_root: Path, object_id: str, angle: int, view_idx: int) -> Image.Image:
    rgb_path = data_root / "renders" / object_id / f"angle_{int(angle)}" / "rgb" / f"view_{int(view_idx)}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"SLat input RGB view not found: {rgb_path}")
    image = Image.open(rgb_path)
    if image.mode == "RGBA" or "A" in image.getbands():
        return image.convert("RGBA")

    mask_candidates = [
        data_root / "renders" / object_id / f"angle_{int(angle)}" / "mask" / f"mask_{int(view_idx)}.npy",
        data_root / "renders" / object_id / f"angle_{int(angle)}" / "mask" / f"mask_{int(view_idx)}.png",
    ]
    mask_path = next((path for path in mask_candidates if path.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(f"SLat input view has no alpha and mask is missing for view {view_idx}")
    if mask_path.suffix == ".npy":
        mask = np.asarray(np.load(mask_path))
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        alpha = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    else:
        alpha = Image.open(mask_path).convert("L")
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.NEAREST)
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _load_slat_cond_tokens_for_views(
    ds: Any,
    sample: dict[str, Any],
    view_indices: list[int],
    *,
    token_source: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    data_root = Path(ds.data_root)
    object_id = str(sample["obj_id"])
    angle = int(sample["angle_idx"])
    picked_views = [int(v) for v in view_indices]
    if not picked_views:
        raise ValueError("slat view list must be non-empty")

    if token_source == "live":
        images = [_rgba_view_image(data_root, object_id, angle, view_idx) for view_idx in picked_views]
        picked = inference._images_to_tokens(images).detach().float().cpu()
        return picked, {
            "token_source": "live_official_trellis_rgba",
            "preprocess": "TRELLIS RGBA alpha crop + black premultiply + 518 resize + DINO x_prenorm layer_norm",
            "view_indices": picked_views,
            "picked_token_shape": list(picked.shape),
            "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
            "image_paths": [
                str((data_root / "renders" / object_id / f"angle_{angle}" / "rgb" / f"view_{view_idx}.png").resolve())
                for view_idx in picked_views
            ],
        }
    if token_source != "cache":
        raise ValueError(f"unsupported slat token source: {token_source!r}")

    token_candidates = [
        data_root / ds.recon_subdir / "dinov2_tokens" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_prenorm" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_official_prenorm1374" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
    ]
    token_path = next((path for path in token_candidates if path.is_file()), token_candidates[0])
    if not token_path.is_file():
        raise FileNotFoundError(f"TRELLIS SLat DINO tokens not found: {token_path}")
    with np.load(token_path, allow_pickle=False) as data:
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError(f"{token_path} expected [V,T,1024], got {tokens.shape}")
    if max(picked_views) >= tokens.shape[0] or min(picked_views) < 0:
        raise ValueError(f"{token_path} has {tokens.shape[0]} views, cannot select {picked_views}")
    picked = torch.from_numpy(np.ascontiguousarray(tokens[picked_views])).float()
    picked = torch.nn.functional.layer_norm(picked, picked.shape[-1:])
    return picked, {
        "token_source": "cache",
        "token_path": str(token_path.resolve()),
        "available_token_shape": list(tokens.shape),
        "view_indices": picked_views,
        "picked_token_shape": list(picked.shape),
        "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
    }


def _sparse_subset_from_coords(slat: SparseTensor, coords: np.ndarray, label: str) -> tuple[SparseTensor, int]:
    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords_np.size == 0:
        raise ValueError(f"{label}: empty part coords")
    spatial = slat.coords[:, 1:].detach().long().cpu().numpy()
    resolution = int(max(int(spatial.max(initial=0)) + 1, int(coords_np.max(initial=0)) + 1, 64))
    slat_keys = spatial[:, 0] * resolution * resolution + spatial[:, 1] * resolution + spatial[:, 2]
    part_keys = coords_np[:, 0] * resolution * resolution + coords_np[:, 1] * resolution + coords_np[:, 2]
    part_set = set(int(x) for x in part_keys.tolist())
    keep_np = np.fromiter((int(k) in part_set for k in slat_keys.tolist()), dtype=bool, count=len(slat_keys))
    matched = int(keep_np.sum())
    if matched == 0:
        raise ValueError(f"{label}: no part coords matched whole SLat coords ({len(coords_np)} requested)")
    keep = torch.from_numpy(keep_np).to(device=slat.feats.device)
    return SparseTensor(feats=slat.feats[keep].contiguous(), coords=slat.coords[keep].contiguous()), matched


def _dilate_coords(coords: np.ndarray, radius: int, *, resolution: int = 64) -> np.ndarray:
    radius = int(radius)
    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if radius <= 0 or coords_np.size == 0:
        return coords_np.astype(np.int32, copy=False)
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if max(abs(dx), abs(dy), abs(dz)) <= radius
        ],
        dtype=np.int64,
    )
    expanded = coords_np[:, None, :] + offsets[None, :, :]
    expanded = expanded.reshape(-1, 3)
    keep = np.all((expanded >= 0) & (expanded < int(resolution)), axis=1)
    return np.unique(expanded[keep], axis=0).astype(np.int32, copy=False)


def _residual_body_coords(whole_coords: np.ndarray, part_coords: list[np.ndarray]) -> np.ndarray:
    whole = np.asarray(whole_coords, dtype=np.int64).reshape(-1, 3)
    if whole.size == 0 or not part_coords:
        return whole.astype(np.int32, copy=False)
    stacked_parts = [
        np.asarray(coords, dtype=np.int64).reshape(-1, 3)
        for coords in part_coords
        if np.asarray(coords).size
    ]
    if not stacked_parts:
        return whole.astype(np.int32, copy=False)
    parts = np.unique(np.concatenate(stacked_parts, axis=0), axis=0)
    resolution = int(max(int(whole.max(initial=0)) + 1, int(parts.max(initial=0)) + 1, 64))
    whole_keys = whole[:, 0] * resolution * resolution + whole[:, 1] * resolution + whole[:, 2]
    part_keys = parts[:, 0] * resolution * resolution + parts[:, 1] * resolution + parts[:, 2]
    keep = ~np.isin(whole_keys, part_keys)
    return whole[keep].astype(np.int32, copy=False)


def _write_part_voxel_npz(path: Path, coords: np.ndarray, *, part_index: int, part_name: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=_unique_coords(coords),
        resolution=np.int32(64),
        coord_frame="canonical_grid",
        source=str(source),
        part_index=np.int32(part_index),
        target_part_name=str(part_name),
    )


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return float(np.log(value / (1.0 - value)))


def _dense_part_logits_from_output(out: dict[str, Any], *, fill_value: float = -20.0) -> np.ndarray:
    dense = np.full((64, 64, 64), float(fill_value), dtype=np.float32)
    logits = out["voxel_logits"][0].float().view(-1)
    pad_mask = out["voxel_pad_mask"][0].bool().view(-1)
    coords = out["voxel_coords"][0].long().view(-1, 3)
    valid_len = min(coords.shape[0], logits.shape[0], pad_mask.shape[0])
    keep = ~pad_mask[:valid_len]
    if bool(keep.any()):
        picked_coords = coords[:valid_len][keep].detach().cpu().numpy().astype(np.int64)
        picked_logits = logits[:valid_len][keep].detach().cpu().numpy().astype(np.float32)
        valid = np.all((picked_coords >= 0) & (picked_coords < 64), axis=1)
        picked_coords = picked_coords[valid]
        picked_logits = picked_logits[valid]
        dense[picked_coords[:, 0], picked_coords[:, 1], picked_coords[:, 2]] = picked_logits
    return dense


def _shift_labels_and_mask(
    labels: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
    dz: int,
) -> tuple[np.ndarray, np.ndarray]:
    shifted_labels = np.zeros_like(labels, dtype=labels.dtype)
    shifted_mask = np.zeros_like(mask, dtype=bool)
    src = [
        slice(max(0, -int(dx)), labels.shape[0] - max(0, int(dx))),
        slice(max(0, -int(dy)), labels.shape[1] - max(0, int(dy))),
        slice(max(0, -int(dz)), labels.shape[2] - max(0, int(dz))),
    ]
    dst = [
        slice(max(0, int(dx)), labels.shape[0] - max(0, -int(dx))),
        slice(max(0, int(dy)), labels.shape[1] - max(0, -int(dy))),
        slice(max(0, int(dz)), labels.shape[2] - max(0, -int(dz))),
    ]
    shifted_labels[tuple(dst)] = labels[tuple(src)]
    shifted_mask[tuple(dst)] = mask[tuple(src)]
    return shifted_labels, shifted_mask


def _smooth_t0_boundary_labels(
    labels: np.ndarray,
    scores: np.ndarray,
    whole_occ: np.ndarray,
    *,
    margin_threshold: float,
    smooth_iters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int16).copy()
    whole_occ = np.asarray(whole_occ, dtype=bool)
    if scores.shape[0] < 2 or int(smooth_iters) <= 0:
        return labels, {
            "enabled": False,
            "ambiguous_voxels": 0,
            "changed_voxels": 0,
            "iters": 0,
            "margin_threshold": float(margin_threshold),
        }

    top2 = np.partition(scores, kth=-2, axis=0)[-2:]
    margin = top2[1] - top2[0]
    ambiguous = whole_occ & (margin < float(margin_threshold))
    total_changed = 0
    iters_run = 0
    for _ in range(int(smooth_iters)):
        iters_run += 1
        counts = np.zeros((int(scores.shape[0]),) + labels.shape, dtype=np.uint16)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    shifted_labels, shifted_mask = _shift_labels_and_mask(labels, whole_occ, dx, dy, dz)
                    for label_idx in range(int(scores.shape[0])):
                        counts[label_idx] += (shifted_mask & (shifted_labels == label_idx)).astype(np.uint16)
        best = counts.argmax(axis=0).astype(np.int16, copy=False)
        best_count = counts.max(axis=0)
        grid = np.indices(labels.shape)
        current_count = counts[labels, grid[0], grid[1], grid[2]]
        update = ambiguous & (best_count > current_count)
        changed = int((update & (best != labels)).sum())
        if changed <= 0:
            break
        labels[update] = best[update]
        total_changed += changed

    labels[~whole_occ] = 0
    return labels, {
        "enabled": True,
        "ambiguous_voxels": int(ambiguous.sum()),
        "changed_voxels": int(total_changed),
        "iters": int(iters_run),
        "margin_threshold": float(margin_threshold),
    }


@torch.no_grad()
def _run_part_t0_filter(
    *,
    args: argparse.Namespace,
    ds: Any,
    ds_sample: dict[str, Any],
    run_dir: Path,
) -> tuple[list[np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    z_global = torch.from_numpy(np.load(run_dir / "ss_latent.npy")).float().unsqueeze(0).cuda()
    if tuple(z_global.shape) != (1, 8, 16, 16, 16):
        raise ValueError(f"ss_latent.npy shape {tuple(z_global.shape)} is invalid for T0")
    model, _empty_code, ckpt_args = _load_prompt_seg_model(str(args.part_seg_ckpt.resolve()))
    route = str(ckpt_args.get("route", "latent"))
    if route != "voxel":
        raise ValueError(f"--part-t0-filter requires route=voxel part-seg ckpt, got {route}")
    if bool(ckpt_args.get("joint_seg", False)):
        raise ValueError(
            "--part-t0-filter uses the legacy independent voxel head and must not override a joint-seg checkpoint; "
            "use --part-joint-refine for joint logits"
        )
    full_occ_t = _dense_occ_from_voxel_npz(run_dir / "voxel.npz", device=z_global.device)
    whole_occ = full_occ_t[0, 0].detach().cpu().numpy().astype(bool)
    part_names = [str(part["part_name"]) for part in ds_sample["parts"]]
    logits_by_part: list[np.ndarray] = []
    visible_prompt_views: list[int] = []
    print(f"[0617-128ee] part T0 start parts={len(part_names)}", flush=True)
    for part_index, part in enumerate(ds_sample["parts"]):
        masks2d = _load_part_masks2d(ds, ds_sample, part).unsqueeze(0).cuda()
        visible_views = int((masks2d.flatten(2).sum(dim=2) > 0).sum().item())
        if visible_views <= 0:
            raise ValueError(f"T0 part {part['part_name']} has no visible prompt views")
        out_cell = model(
            z_global,
            masks2d,
            candidate_cells=torch.ones((1, 16, 16, 16), dtype=torch.float32, device=z_global.device),
            full_occ=full_occ_t,
        )
        pred_m = (out_cell["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
        out_voxel = model(
            z_global,
            masks2d,
            candidate_cells=_mask_morphology(pred_m, "dilate"),
            full_occ=full_occ_t,
        )
        dense_logits = _dense_part_logits_from_output(out_voxel)
        dense_logits[~whole_occ] = -20.0
        logits_by_part.append(dense_logits)
        visible_prompt_views.append(visible_views)
        print(
            f"[0617-128ee] part T0 logits {part_index:02d} {part['part_name']} visible_views={visible_views}",
            flush=True,
        )

    if logits_by_part:
        part_scores = np.stack(logits_by_part, axis=0).astype(np.float32, copy=False)
    else:
        part_scores = np.zeros((0, 64, 64, 64), dtype=np.float32)
    body_score = np.full((1, 64, 64, 64), _logit(float(args.part_t0_part_threshold)), dtype=np.float32)
    scores = np.concatenate([body_score, part_scores], axis=0)
    labels = scores.argmax(axis=0).astype(np.int16, copy=False)
    labels[~whole_occ] = 0
    pre_smooth_counts = [int((labels == (idx + 1)).sum()) for idx in range(len(part_names))]
    labels, smooth_record = _smooth_t0_boundary_labels(
        labels,
        scores,
        whole_occ,
        margin_threshold=float(args.part_t0_margin_threshold),
        smooth_iters=int(args.part_t0_smooth_iters),
    )
    pre_cc_counts = [int((labels == (idx + 1)).sum()) for idx in range(len(part_names))]
    part_coord_sets: list[np.ndarray] = []
    cc_filter_records: list[dict[str, Any]] = []
    for part_index, part_name in enumerate(part_names):
        coords = np.argwhere(labels == (part_index + 1)).astype(np.int32, copy=False)
        if not bool(args.part_t0_disable_cc):
            coords, cc_record = _filter_part_connected_components(
                coords,
                part_index=int(part_index),
                part_name=part_name,
                min_component_voxels=int(args.part_cc_min_component_voxels),
                min_component_fraction=float(args.part_cc_min_component_fraction),
                max_component_distance=int(args.part_cc_max_component_distance),
                max_large_component_distance=args.part_cc_max_large_component_distance,
            )
            cc_filter_records.append(cc_record)
        part_coord_sets.append(coords)
        source = "pred_t0_joint_argmax"
        if bool(smooth_record.get("enabled", False)):
            source += "_smooth"
        if not bool(args.part_t0_disable_cc):
            source += "_cc_filtered"
        _write_part_voxel_npz(
            run_dir / "parts" / f"part_{part_index:02d}_voxel.npz",
            coords,
            part_index=int(part_index),
            part_name=part_name,
            source=source,
        )
    t0_record = {
        "enabled": True,
        "rule": (
            "joint argmax over default body logit plus independent part logits; "
            "optionally smooth top-2 logit-margin band with 26-neighborhood majority pass; "
            "optionally reassign small remote part connected components to body"
        ),
        "part_threshold": float(args.part_t0_part_threshold),
        "body_default_logit": float(_logit(float(args.part_t0_part_threshold))),
        "margin_threshold": float(args.part_t0_margin_threshold),
        "smooth_iters": int(args.part_t0_smooth_iters),
        "part_names": part_names,
        "visible_prompt_views": visible_prompt_views,
        "pre_smooth_part_voxels": pre_smooth_counts,
        "pre_cc_part_voxels": pre_cc_counts,
        "post_cc_part_voxels": [int(coords.shape[0]) for coords in part_coord_sets],
        "smooth": smooth_record,
        "cc": {
            "enabled": not bool(args.part_t0_disable_cc),
            "parts_with_removed_components": int(
                sum(1 for item in cc_filter_records if int(item.get("reassigned_to_body_voxels", 0)) > 0)
            ),
            "reassigned_to_body_voxels": int(
                sum(int(item.get("reassigned_to_body_voxels", 0)) for item in cc_filter_records)
            ),
        },
        "ckpt_route": route,
        "ckpt_joint_seg": bool(ckpt_args.get("joint_seg", False)),
    }
    _write_json(run_dir / "part_t0_filter_meta.json", t0_record)
    print(
        "[0617-128ee] part T0 done "
        f"smooth_changed={smooth_record['changed_voxels']} "
        f"cc_enabled={t0_record['cc']['enabled']} "
        f"cc_reassigned={t0_record['cc']['reassigned_to_body_voxels']}",
        flush=True,
    )
    _load_prompt_seg_model.cache_clear()
    del z_global, model, full_occ_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return part_coord_sets, t0_record, cc_filter_records


def _new_like_gaussian(gaussian: Any) -> Any:
    out = type(gaussian)(**gaussian.init_params)
    out.active_sh_degree = gaussian.active_sh_degree
    return out


def _subset_gaussian(gaussian: Any, keep: torch.Tensor) -> Any:
    out = _new_like_gaussian(gaussian)
    keep = keep.to(device=gaussian.get_xyz.device, dtype=torch.bool)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach()[keep].clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach()[keep].clone()
    return out


def _adjust_gaussian(gaussian: Any) -> Any:
    out = _new_like_gaussian(gaussian)
    for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(gaussian, name).detach().clone())
    out._features_rest = None if gaussian._features_rest is None else gaussian._features_rest.detach().clone()
    scale_mult = float(GS_PRESET["scale_mult"])
    opacity_mult = float(GS_PRESET["opacity_mult"])
    if scale_mult != 1.0:
        scaling = torch.clamp(out.get_scaling * scale_mult, min=out.mininum_kernel_size + 1e-7)
        out.from_scaling(scaling)
    if opacity_mult != 1.0:
        opacity = torch.clamp(out.get_opacity * opacity_mult, 1e-5, 0.995)
        out.from_opacity(opacity)
    return out


def _apply_gs_preset(gaussian: Any) -> tuple[Any, dict[str, Any]]:
    scale_max = gaussian.get_scaling.detach().max(dim=1).values
    quantile_limit = torch.quantile(scale_max, float(GS_PRESET["max_scale_quantile"]))
    abs_limit = scale_max.new_tensor(float(GS_PRESET["max_scale_abs"]))
    limit = torch.minimum(quantile_limit, abs_limit)
    keep = scale_max <= limit
    adjusted = _adjust_gaussian(_subset_gaussian(gaussian, keep))
    return adjusted, {
        **GS_PRESET,
        "scale_quantile_limit": float(quantile_limit.detach().cpu().item()),
        "scale_abs_limit": float(abs_limit.detach().cpu().item()),
        "scale_limit_used": float(limit.detach().cpu().item()),
        "gaussians_before": int(gaussian.get_xyz.shape[0]),
        "gaussians_after": int(adjusted.get_xyz.shape[0]),
        "removed": int((~keep).sum().detach().cpu().item()),
    }


def _make_gaussian_renderer(resolution: int) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (1, 1, 1)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(GS_PRESET["kernel_size"])
    renderer.pipe.scale_modifier = 1.0
    return renderer


def _make_mesh_renderer(resolution: int) -> MeshRenderer:
    return MeshRenderer({"resolution": int(resolution), "near": 0.1, "far": 10.0, "ssaa": 1})


@torch.no_grad()
def _render_gaussian(gaussian: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, resolution: int) -> Image.Image:
    color = _make_gaussian_renderer(resolution).render(gaussian, extrinsic, intrinsic)["color"]
    arr = (color.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def _render_mesh(mesh: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, resolution: int) -> Image.Image:
    renderer = _make_mesh_renderer(resolution)
    if getattr(mesh, "vertex_attrs", None) is None:
        ret = renderer.render(mesh, extrinsic, intrinsic, return_types=["normal", "mask"])
        color = ret["normal"].detach().float().cpu().clamp(0, 1)
    else:
        ret = renderer.render(mesh, extrinsic, intrinsic, return_types=["color", "mask"])
        color = ret["color"].detach().float().cpu().clamp(0, 1)
    mask = ret["mask"].detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    color = color * mask + torch.full_like(color, 0.94) * (1.0 - mask)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _tile(image: Image.Image, label: str, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 32), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 32), fill=(0, 0, 0))
    draw.text((7, 9), label[:64], fill=(255, 255, 255))
    tile.paste(image, ((size - image.width) // 2, 32 + (size - image.height) // 2))
    return tile


def _error_image(message: str, resolution: int) -> Image.Image:
    size = max(128, int(resolution))
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(200, 60, 60), width=3)
    text = str(message)[:160]
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 28 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    y = max(12, size // 2 - 10 * len(lines))
    for line in lines[:8]:
        draw.text((12, y), line, fill=(120, 0, 0))
        y += 22
    return image


def _panel(tiles: list[tuple[str, Image.Image]], out_png: Path, *, tile_size: int, cols: int) -> None:
    cols = max(1, min(int(cols), len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + 32)), (255, 255, 255))
    for idx, (label, image) in enumerate(tiles):
        canvas.paste(_tile(image, label, tile_size), ((idx % cols) * tile_size, (idx // cols) * (tile_size + 32)))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _labeled_fit(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    body_h = max(1, int(height) - 30)
    image.thumbnail((int(width), body_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, int(width), 30), fill=(0, 0, 0))
    draw.text((8, 9), label[:96], fill=(255, 255, 255))
    tile.paste(image, ((int(width) - image.width) // 2, 30 + (body_h - image.height) // 2))
    return tile


def _input_views_panel(ds: Any, ds_sample: dict[str, Any]) -> Image.Image:
    view_indices = [int(v) for v in ds_sample.get("view_indices", [])]
    image_paths = list(ds_sample.get("image_paths", []))
    tiles: list[Image.Image] = []
    for idx, rel_path in enumerate(image_paths[:4]):
        path = Path(ds.data_root) / str(rel_path)
        if not path.is_file():
            continue
        view_id = view_indices[idx] if idx < len(view_indices) else idx
        tiles.append(_labeled_fit(Image.open(path), f"input view {view_id}", 360, 220))
    if not tiles:
        return _labeled_fit(Image.new("RGB", (720, 440), (245, 245, 245)), "input views missing", 720, 440)
    while len(tiles) < 4:
        tiles.append(_labeled_fit(Image.new("RGB", (360, 220), (245, 245, 245)), "missing", 360, 220))
    canvas = Image.new("RGB", (720, 440), (255, 255, 255))
    for idx, tile in enumerate(tiles[:4]):
        canvas.paste(tile, ((idx % 2) * 360, (idx // 2) * 220))
    return canvas


def _write_diagnostic_panel(
    *,
    ds: Any,
    ds_sample: dict[str, Any],
    whole_coords: np.ndarray,
    part_items: list[tuple[str, np.ndarray]],
    gaussian_png: Path,
    mesh_png: Path,
    joint_boundary_png: Path | None,
    out_png: Path,
    object_id: str,
    angle: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="0617_128ee_diag_") as tmp:
        tmp_dir = Path(tmp)
        ss_voxel_png = tmp_dir / "ss_decode_voxel.png"
        partseg_voxel_png = tmp_dir / "partseg_voxel.png"
        render_preview_voxel(whole_coords, [], ss_voxel_png, object_id, int(angle))
        render_preview_voxel(whole_coords, part_items, partseg_voxel_png, object_id, int(angle))
        input_panel = _input_views_panel(ds, ds_sample)
        ss_img = Image.open(ss_voxel_png).copy()
        partseg_img = Image.open(partseg_voxel_png).copy()
    gaussian_img = Image.open(gaussian_png).copy()
    mesh_img = Image.open(mesh_png).copy()
    boundary_img = Image.open(joint_boundary_png).copy() if joint_boundary_png is not None else None

    canvas_height = 1440 if boundary_img is not None else 940
    canvas = Image.new("RGB", (1600, canvas_height), (255, 255, 255))
    top = [
        _labeled_fit(input_panel, "input 4 views", 760, 440),
        _labeled_fit(ss_img, "SS decode voxel", 410, 440),
        _labeled_fit(partseg_img, "PartSeg voxel", 410, 440),
    ]
    x = 0
    for tile in top:
        canvas.paste(tile, (x, 0))
        x += tile.width + 10
    canvas.paste(_labeled_fit(gaussian_img, "Gaussian overall + body + parts", 790, 500), (0, 440))
    canvas.paste(_labeled_fit(mesh_img, "Mesh overall + body + parts", 790, 500), (810, 440))
    if boundary_img is not None:
        canvas.paste(
            _labeled_fit(boundary_img, "Joint boundary quantitative diagnostics", 1600, 500),
            (0, 940),
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def _prefix(dataset_id: str, object_id: str, angle: int) -> str:
    return f"{dataset_id}__{_safe_name(object_id)}__angle_{int(angle):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0617-128ee single-object EE smoke from input tokens, SS concat, partseg, SLat.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    parser.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_PART_SEG_CKPT)
    parser.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_SS_FLOW_CKPT)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    parser.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    parser.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument(
        "--slat-seed",
        type=int,
        default=None,
        help="SLat flow seed. When omitted, inherits --seed if provided, otherwise uses 42.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed SS flow, promptable part segmentation setup, and SLat flow. If set, overrides --slat-seed unless --slat-seed is explicitly changed.",
    )
    parser.add_argument(
        "--part-decode-dilation",
        type=int,
        default=0,
        help="Experimental: dilate part/body voxel coords before slicing whole SLat for per-component decode. Default 0 preserves existing behavior.",
    )
    parser.add_argument(
        "--part-cc-filter",
        action="store_true",
        help="Post-process each predicted part voxel mask by reassigning small remote connected components to body.",
    )
    parser.add_argument("--part-cc-min-component-voxels", type=int, default=32)
    parser.add_argument("--part-cc-min-component-fraction", type=float, default=0.05)
    parser.add_argument("--part-cc-max-component-distance", type=int, default=2)
    parser.add_argument(
        "--part-cc-max-large-component-distance",
        type=int,
        default=None,
        help="Optional hard bbox-gap limit for non-main components kept only because they are large.",
    )
    parser.add_argument("--part-joint-candidate-mode", choices=("proposal", "full_occ"), default="proposal")
    parser.add_argument("--part-joint-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part-joint-refine-iters", type=int, default=1)
    parser.add_argument("--part-joint-refine-pairwise", type=float, default=3.0)
    parser.add_argument("--part-joint-refine-margin", type=float, default=0.0)
    parser.add_argument("--part-joint-refine-margin-quantile", type=float, default=0.01)
    parser.add_argument("--part-joint-refine-neighborhood", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--part-joint-refine-min-vote-gain", type=float, default=0.0)
    parser.add_argument("--part-joint-refine-preserve-small-classes", type=int, default=32)
    parser.add_argument("--part-joint-save-logits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--part-t0-filter",
        action="store_true",
        help=(
            "T0 zero-training boundary postprocess: joint argmax over default body + part logits, "
            "one 26-neighbor smoothing pass in the competition band, then connected-component island filtering."
        ),
    )
    parser.add_argument("--part-t0-part-threshold", type=float, default=0.5)
    parser.add_argument("--part-t0-margin-threshold", type=float, default=0.35)
    parser.add_argument("--part-t0-smooth-iters", type=int, default=1)
    parser.add_argument(
        "--part-t0-disable-cc",
        action="store_true",
        help="For T0 ablations: keep joint argmax output without connected-component island filtering.",
    )
    parser.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="SLat flow condition source. live uses accepted TRELLIS RGBA preprocessing from input renders; cache is diagnostic.",
    )
    parser.add_argument(
        "--slat-view-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override SLat appearance condition views. Default uses the sample's manifest view_indices.",
    )
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=240)
    parser.add_argument("--panel-cols", type=int, default=4)
    parser.add_argument(
        "--export-mujoco",
        action="store_true",
        help="Export per-part OBJ meshes and a static no-joint MJCF XML for MuJoCo visualization.",
    )
    parser.add_argument(
        "--export-usd",
        action="store_true",
        help="Export decoded body/part meshes as a USDA scene with vertex-color primvars for Isaac Sim.",
    )
    parser.add_argument(
        "--mujoco-textured-assets",
        action="store_true",
        help="Bake decoded appearance into standard OBJ/MTL/PNG assets and bind them from MJCF.",
    )
    parser.add_argument(
        "--mujoco-appearance-source",
        choices=("obj-vertex-color", "mesh-vertex-texture", "gaussian-texture"),
        default="obj-vertex-color",
        help=(
            "Appearance source for --mujoco-textured-assets. mesh-vertex-texture matches the diagnostic "
            "mesh renderer by baking mesh.vertex_attrs to UV texture; gaussian-texture bakes decoded "
            "Gaussian renders; obj-vertex-color preserves the legacy raw OBJ vertex-color export."
        ),
    )
    parser.add_argument("--mujoco-texture-size", type=int, default=512)
    parser.add_argument("--mujoco-texture-render-resolution", type=int, default=512)
    parser.add_argument("--mujoco-texture-nviews", type=int, default=30)
    parser.add_argument("--mujoco-texture-mode", choices=("fast", "opt"), default="fast")
    parser.add_argument(
        "--fill-hidden-vertex-colors",
        action="store_true",
        help="Optional Track2 fallback: recolor hidden dark component OBJ vertices from nearest visible same-component vertices. Default off.",
    )
    parser.add_argument("--hidden-color-fill-out-dir", type=Path, default=None)
    parser.add_argument("--hidden-color-fill-dark-threshold", type=float, default=0.18)
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SS_FLOW_FUSION_MODE"] = "concat"
    if args.slat_seed is None:
        args.slat_seed = int(args.seed) if args.seed is not None else 42
    _seed_all(args.seed)
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for attr, label in (
        ("data_config", "data config"),
        ("split_json", "split json"),
        ("part_seg_ckpt", "part segmentation checkpoint"),
        ("ss_flow_ckpt", "tre-ss-concat-0616-1 SS-flow ckpt"),
        ("ss_decoder_ckpt", "SS decoder ckpt"),
        ("slat_flow_ckpt", "SLat flow ckpt"),
        ("slat_mesh_decoder_ckpt", "SLat mesh decoder ckpt"),
        ("slat_gaussian_decoder_ckpt", "SLat gaussian decoder ckpt"),
    ):
        setattr(args, attr, _require_file(Path(getattr(args, attr)), label))
    print(f"[0617-128ee] seed={args.seed} slat_seed={args.slat_seed}", flush=True)
    print(f"[0617-128ee] part-seg ckpt={args.part_seg_ckpt.resolve()}", flush=True)
    print(f"[0617-128ee] SS-flow ckpt={args.ss_flow_ckpt.resolve()}", flush=True)

    datasets, dataset_meta = _load_datasets(args)
    if args.dataset_id not in datasets:
        raise KeyError(f"dataset_id={args.dataset_id!r} not found; available={sorted(datasets)}")
    ds = datasets[args.dataset_id]
    sample = _find_sample(ds, args.object_id, int(args.angle), args.dataset_id)
    ds_sample = _find_dataset_sample(ds, sample)
    started = time.time()
    run_dir = _ensure_ss_and_part(args, ds, sample, ds_sample)
    _seed_all(args.seed)

    prefix = _prefix(args.dataset_id, args.object_id, int(args.angle))
    joint_partition_path = run_dir / "parts" / "joint_partition.npz"
    joint_boundary_png = args.out_dir / f"{prefix}__joint_boundary.png"
    joint_boundary_json = args.out_dir / f"{prefix}__joint_boundary.json"
    joint_boundary_diagnostics: dict[str, Any] | None = None
    joint_boundary_error: str | None = None
    if joint_partition_path.is_file():
        try:
            diagnostic_sample = dict(ds_sample)
            diagnostic_sample.setdefault("_eval_dataset_id", str(args.dataset_id))
            joint_boundary_diagnostics = run_joint_boundary_diagnostics(
                joint_partition_path,
                data_root=Path(ds.data_root),
                ds_sample=diagnostic_sample,
                output_png=joint_boundary_png,
                output_json=joint_boundary_json,
                whole_pred_path=run_dir / "voxel.npz",
                title=f"{args.dataset_id}::{args.object_id} angle={int(args.angle)}",
            )
            print(
                f"[0617-128ee] joint boundary diagnostics -> {joint_boundary_png}",
                flush=True,
            )
        except Exception as exc:
            joint_boundary_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[0617-128ee] WARNING joint boundary diagnostics failed: {joint_boundary_error}",
                flush=True,
            )
    else:
        joint_boundary_error = f"joint partition not found: {joint_partition_path}"

    slat_view_indices = (
        [int(v) for v in args.slat_view_indices]
        if args.slat_view_indices is not None
        else [int(v) for v in ds_sample.get("view_indices", [])]
    )
    if not slat_view_indices:
        raise ValueError(f"{args.dataset_id}::{args.object_id} angle={int(args.angle)} has no manifest view_indices")
    cond_tokens, slat_cond_meta = _load_slat_cond_tokens_for_views(
        ds,
        ds_sample,
        slat_view_indices,
        token_source=str(args.slat_token_source),
    )
    whole_coords = _load_coords(run_dir / "voxel.npz")
    whole_coords_t = torch.from_numpy(np.ascontiguousarray(whole_coords.astype(np.int64, copy=False))).long()
    part_t0_coord_sets: list[np.ndarray] | None = None
    part_t0_filter_record: dict[str, Any] = {"enabled": False}
    t0_cc_filter_records: list[dict[str, Any]] = []
    if args.part_t0_filter:
        part_t0_coord_sets, part_t0_filter_record, t0_cc_filter_records = _run_part_t0_filter(
            args=args,
            ds=ds,
            ds_sample=ds_sample,
            run_dir=run_dir,
        )
    print(
        f"[0617-128ee] SLat flow ONCE {args.dataset_id}::{args.object_id} "
        f"angle={int(args.angle)} coords={whole_coords.shape[0]} views={slat_cond_meta['view_indices']}",
        flush=True,
    )
    overall_slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        whole_coords_t,
        str(args.slat_flow_ckpt.resolve()),
        num_steps=int(args.slat_steps),
        seed=int(args.slat_seed),
    )

    extrinsics, intrinsics = load_camera_matrices(
        Path(ds.data_root) / "renders" / sample.obj_id / f"angle_{int(sample.angle_idx)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    extrinsic, intrinsic = extrinsics[0], intrinsics[0]

    components: list[tuple[str, np.ndarray, SparseTensor | None, str, int, str | None]] = [
        ("overall", whole_coords, overall_slat, "whole_slat_flow_once", int(whole_coords.shape[0]), None),
    ]
    part_items: list[tuple[str, np.ndarray]] = []
    part_coord_sets: list[np.ndarray] = []
    part_motion_by_label: dict[str, dict[str, Any]] = {}
    cc_filter_records: list[dict[str, Any]] = list(t0_cc_filter_records)
    decode_dilation = max(0, int(args.part_decode_dilation))
    for part_idx, part in enumerate(ds_sample["parts"]):
        if part_t0_coord_sets is not None:
            coords = part_t0_coord_sets[part_idx]
        else:
            coords = _load_coords(run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz")
        if args.part_cc_filter and part_t0_coord_sets is None:
            coords, cc_record = _filter_part_connected_components(
                coords,
                part_index=int(part_idx),
                part_name=str(part["part_name"]),
                min_component_voxels=int(args.part_cc_min_component_voxels),
                min_component_fraction=float(args.part_cc_min_component_fraction),
                max_component_distance=int(args.part_cc_max_component_distance),
                max_large_component_distance=args.part_cc_max_large_component_distance,
            )
            cc_filter_records.append(cc_record)
            _write_part_voxel_npz(
                run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz",
                coords,
                part_index=int(part_idx),
                part_name=str(part["part_name"]),
                source="pred_cc_filtered",
            )
        part_coord_sets.append(coords)
        part_items.append((str(part["part_name"]), coords))
        label = _part_component_label(part_idx, part)
        part_motion_by_label[label] = _mujoco_motion_meta(part_idx, part)
        decode_coords = _dilate_coords(coords, decode_dilation) if decode_dilation else coords
        try:
            part_slat, matched = _sparse_subset_from_coords(overall_slat, decode_coords, label)
            source = "subset_from_whole_slat_by_coords"
            if decode_dilation:
                source = f"{source}_dilated_r{decode_dilation}"
            components.append((label, coords, part_slat, source, matched, None))
        except ValueError as exc:
            components.append((label, coords, None, "subset_from_whole_slat_by_coords_failed", 0, str(exc)))
    body_coords = _residual_body_coords(whole_coords, part_coord_sets)
    part_items_with_body = [("body_without_parts", body_coords), *part_items]
    body_decode_coords = _dilate_coords(body_coords, decode_dilation) if decode_dilation else body_coords
    try:
        body_slat, body_matched = _sparse_subset_from_coords(overall_slat, body_decode_coords, "body_without_parts")
        body_source = "subset_from_whole_slat_after_part_union_subtract"
        if decode_dilation:
            body_source = f"{body_source}_dilated_r{decode_dilation}"
        body_component = (
            "body_without_parts",
            body_coords,
            body_slat,
            body_source,
            body_matched,
            None,
        )
    except ValueError as exc:
        body_component = (
            "body_without_parts",
            body_coords,
            None,
            "subset_from_whole_slat_after_part_union_subtract_failed",
            0,
            str(exc),
        )
    components.insert(1, body_component)

    gauss_tiles: list[tuple[str, Image.Image]] = []
    mesh_tiles: list[tuple[str, Image.Image]] = []
    records: list[dict[str, Any]] = []
    mujoco_dir = args.out_dir / f"{prefix}__mujoco"
    mujoco_assets_dir = mujoco_dir / "assets"
    mujoco_overall_mesh: dict[str, Any] | None = None
    mujoco_body_mesh: dict[str, Any] | None = None
    mujoco_meshes: list[dict[str, Any]] = []
    mujoco_part_meshes: list[dict[str, Any]] = []
    usd_meshes: list[dict[str, Any]] = []
    for label, coords, slat, slat_source, matched, subset_error in components:
        t0 = time.time()
        print(f"[0617-128ee] decode+render {label} coords={len(coords)} matched={matched}", flush=True)
        if slat is None:
            error_text = subset_error or "missing part SLat"
            gauss_tiles.append((label, _error_image(error_text, int(args.resolution))))
            mesh_tiles.append((label, _error_image(error_text, int(args.resolution))))
            records.append(
                {
                    "label": label,
                    "coords": int(len(coords)),
                    "matched_coords": int(matched),
                    "slat_source": slat_source,
                    "slat_subset_error": error_text,
                    "mesh_vertices": 0,
                    "mesh_faces": 0,
                    "mesh_has_vertex_attrs": False,
                    "mesh_error": error_text,
                    "gaussian_error": error_text,
                    "gs_preset": None,
                    "seconds": round(time.time() - t0, 3),
                }
            )
            continue
        decoded = inference.decode_slat_assets(
            slat,
            gaussian_decoder_ckpt=str(args.slat_gaussian_decoder_ckpt.resolve()),
            mesh_decoder_ckpt=str(args.slat_mesh_decoder_ckpt.resolve()),
            slat_is_normalized=True,
        )
        gaussian = decoded.get("gaussian")
        mesh = decoded.get("mesh")
        if gaussian is None:
            raise RuntimeError(f"{label}: gaussian decoder returned None")
        gaussian, gs_stats = _apply_gs_preset(gaussian)
        gauss_tiles.append((label, _render_gaussian(gaussian, extrinsic, intrinsic, int(args.resolution))))
        mesh_error = None
        mesh_vertices = 0
        mesh_faces = 0
        mesh_has_vertex_attrs = False
        if mesh is None or not getattr(mesh, "success", True):
            mesh_error = "mesh decoder failed"
            mesh_tiles.append((label, _error_image(mesh_error, int(args.resolution))))
        else:
            mesh_vertices = int(mesh.vertices.shape[0])
            mesh_faces = int(mesh.faces.shape[0])
            mesh_has_vertex_attrs = bool(getattr(mesh, "vertex_attrs", None) is not None)
            mesh_tiles.append((label, _render_mesh(mesh, extrinsic, intrinsic, int(args.resolution))))
            role = "body" if label == "body_without_parts" else "part"
            if label == "overall":
                role = "overall"
            if bool(args.export_usd) and role != "overall":
                usd_meshes.append(_mesh_to_usd_payload(mesh, label="body" if role == "body" else label, role=role))
            if args.export_mujoco:
                stem = _safe_name(label, 96)
                mujoco_appearance_source = str(args.mujoco_appearance_source)
                use_textured_asset = bool(args.mujoco_textured_assets) and role != "overall"
                if use_textured_asset:
                    if mujoco_appearance_source == "gaussian-texture":
                        assets = _save_textured_mujoco_mesh(
                            gaussian=gaussian,
                            mesh=mesh,
                            asset_dir=mujoco_assets_dir,
                            stem=stem,
                            texture_size=int(args.mujoco_texture_size),
                            render_resolution=int(args.mujoco_texture_render_resolution),
                            nviews=int(args.mujoco_texture_nviews),
                            mode=str(args.mujoco_texture_mode),
                        )
                        appearance_source = "decoded_gaussian_baked_texture"
                    elif mujoco_appearance_source == "mesh-vertex-texture":
                        assets = _save_vertex_color_textured_mujoco_mesh(
                            mesh=mesh,
                            asset_dir=mujoco_assets_dir,
                            stem=stem,
                            texture_size=int(args.mujoco_texture_size),
                        )
                        appearance_source = "decoded_mesh_vertex_attrs_baked_texture"
                    else:
                        raise ValueError(
                            "--mujoco-textured-assets requires --mujoco-appearance-source "
                            "mesh-vertex-texture or gaussian-texture"
                        )
                    mesh_file = f"assets/{assets['mesh']}"
                    texture_file = f"assets/{assets['texture']}"
                    material_file = f"assets/{assets['material']}"
                else:
                    obj_name = f"{stem}.obj"
                    assets = save_decoded_slat_assets({"mesh": mesh}, mujoco_assets_dir, mesh_name=obj_name)
                    mesh_file = f"assets/{assets['mesh']}"
                    texture_file = None
                    material_file = None
                    appearance_source = "decoded_mesh_vertex_colors_obj"
                mesh_item = {
                    "role": role,
                    "label": "body" if role == "body" else label,
                    "mesh_file": mesh_file,
                    "mesh_path": str((mujoco_assets_dir / assets["mesh"]).resolve()),
                    "texture_file": texture_file,
                    "material_file": material_file,
                    "bake_stats": assets.get("bake_stats"),
                    "coords": int(len(coords)),
                    "matched_coords": int(matched),
                    "vertices": int(mesh_vertices),
                    "faces": int(mesh_faces),
                    "bbox_center": _obj_bbox_center(mujoco_assets_dir / assets["mesh"]),
                    "appearance_source": appearance_source,
                }
                if role == "part":
                    mesh_item["motion"] = part_motion_by_label.get(label, {})
                if role == "overall":
                    mujoco_overall_mesh = mesh_item
                else:
                    mujoco_meshes.append(mesh_item)
                if role == "body":
                    mujoco_body_mesh = mesh_item
                elif role == "part":
                    mujoco_part_meshes.append(mesh_item)
        records.append(
            {
                "label": label,
                "coords": int(len(coords)),
                "matched_coords": int(matched),
                "slat_source": slat_source,
                "slat_subset_error": subset_error,
                "mesh_vertices": mesh_vertices,
                "mesh_faces": mesh_faces,
                "mesh_has_vertex_attrs": mesh_has_vertex_attrs,
                "mesh_error": mesh_error,
                "gaussian_error": None,
                "gs_preset": gs_stats,
                "seconds": round(time.time() - t0, 3),
            }
        )
        del decoded, gaussian, mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gaussian_png = args.out_dir / f"{prefix}__gaussian.png"
    mesh_png = args.out_dir / f"{prefix}__mesh.png"
    diagnostic_png = args.out_dir / f"{prefix}__diagnostic.png"
    summary_path = args.out_dir / f"{prefix}__summary.json"
    mujoco_xml = None
    mujoco_rd_report: dict[str, Any] | None = None
    usd_scene = None
    if args.export_mujoco and mujoco_meshes:
        mujoco_meshes, mujoco_rd_report = _apply_mujoco_rd_postprocess(
            mujoco_assets_dir=mujoco_assets_dir,
            mesh_items=mujoco_meshes,
        )
        mujoco_part_meshes = [item for item in mujoco_meshes if str(item.get("role")) == "part"]
        mujoco_body_mesh = next((item for item in mujoco_meshes if str(item.get("role")) == "body"), mujoco_body_mesh)
        mujoco_xml = _write_static_mujoco_xml(
            out_xml=mujoco_dir / f"{prefix}.xml",
            model_name=prefix,
            mesh_items=mujoco_meshes,
        )
    if args.export_usd and usd_meshes:
        usd_scene = _write_decoded_mesh_usda(
            out_usda=args.out_dir / f"{prefix}__usd" / f"{prefix}.usda",
            model_name=prefix,
            mesh_items=usd_meshes,
        )
    if not gaussian_png.is_file() or args.force_export:
        _panel(gauss_tiles, gaussian_png, tile_size=int(args.tile_size), cols=int(args.panel_cols))
    if not mesh_png.is_file() or args.force_export:
        _panel(mesh_tiles, mesh_png, tile_size=int(args.tile_size), cols=int(args.panel_cols))
    if not diagnostic_png.is_file() or args.force_export:
        _write_diagnostic_panel(
            ds=ds,
            ds_sample=ds_sample,
            whole_coords=whole_coords,
            part_items=part_items_with_body,
            gaussian_png=gaussian_png,
            mesh_png=mesh_png,
            joint_boundary_png=joint_boundary_png if joint_boundary_png.is_file() else None,
            out_png=diagnostic_png,
            object_id=args.object_id,
            angle=int(args.angle),
        )

    summary = {
        "status": "done",
        "dataset_id": args.dataset_id,
        "object_id": args.object_id,
        "angle": int(args.angle),
        "out_dir": str(args.out_dir),
        "run_dir": str(run_dir.resolve()),
        "gaussian_png": str(gaussian_png.resolve()),
        "mesh_png": str(mesh_png.resolve()),
        "diagnostic_png": str(diagnostic_png.resolve()),
        "joint_boundary_png": str(joint_boundary_png.resolve()) if joint_boundary_png.is_file() else None,
        "joint_boundary_json": str(joint_boundary_json.resolve()) if joint_boundary_json.is_file() else None,
        "joint_boundary_diagnostics": (
            joint_boundary_diagnostics
            if joint_boundary_diagnostics is not None
            else {"status": "error", "error": joint_boundary_error}
        ),
        "mujoco_xml": None if mujoco_xml is None else str(mujoco_xml.resolve()),
        "mujoco_assets_dir": None if mujoco_xml is None else str(mujoco_assets_dir.resolve()),
        "mujoco_overall_mesh": mujoco_overall_mesh,
        "mujoco_body_mesh": mujoco_body_mesh,
        "mujoco_meshes": mujoco_meshes,
        "mujoco_part_meshes": mujoco_part_meshes,
        "mujoco_rd_postprocess": mujoco_rd_report,
        "usd_scene": None if usd_scene is None else str(usd_scene.resolve()),
        "usd_meshes": [
            {
                "role": str(item["role"]),
                "label": str(item["label"]),
                "prim_name": str(item["prim_name"]),
                "vertices": int(np.asarray(item["points"]).shape[0]),
                "faces": int(np.asarray(item["faces"]).shape[0]),
                "appearance_source": "decoded_mesh.vertex_attrs -> USD primvars:displayColor",
            }
            for item in usd_meshes
        ],
        "mujoco_textured_assets": {
            "enabled": bool(args.mujoco_textured_assets),
            "source": (
                "decoded mesh.vertex_attrs baked to OBJ UV texture"
                if args.mujoco_textured_assets and args.mujoco_appearance_source == "mesh-vertex-texture"
                else "decoded Gaussian appearance baked to OBJ UV texture"
                if args.mujoco_textured_assets and args.mujoco_appearance_source == "gaussian-texture"
                else "decoded mesh OBJ vertex colors"
                if args.mujoco_appearance_source == "obj-vertex-color"
                else None
            ),
            "appearance_source": str(args.mujoco_appearance_source),
            "texture_size": int(args.mujoco_texture_size),
            "render_resolution": int(args.mujoco_texture_render_resolution),
            "nviews": int(args.mujoco_texture_nviews),
            "mode": str(args.mujoco_texture_mode),
        },
        "body_without_parts": {
            "coords": int(body_coords.shape[0]),
            "source": "whole_coords minus union(part_coords)",
        },
        "component_count": len(records),
        "components": records,
        "seed": None if args.seed is None else int(args.seed),
        "slat_seed": int(args.slat_seed),
        "part_decode_dilation": int(args.part_decode_dilation),
        "part_cc_filter": {
            "enabled": bool(args.part_cc_filter or (args.part_t0_filter and not args.part_t0_disable_cc)),
            "min_component_voxels": int(args.part_cc_min_component_voxels),
            "min_component_fraction": float(args.part_cc_min_component_fraction),
            "max_component_distance": int(args.part_cc_max_component_distance),
            "max_large_component_distance": args.part_cc_max_large_component_distance,
            "parts_with_removed_components": int(
                sum(1 for item in cc_filter_records if int(item.get("reassigned_to_body_voxels", 0)) > 0)
            ),
            "reassigned_to_body_voxels": int(
                sum(int(item.get("reassigned_to_body_voxels", 0)) for item in cc_filter_records)
            ),
            "records": cc_filter_records,
            "rule": (
                "keep largest connected component plus nearby components; large components remain eligible unless "
                "an optional hard maximum bbox distance is configured; removed components become body residual voxels"
            ),
        },
        "part_t0_filter": part_t0_filter_record,
        "part_joint_partition": {
            "candidate_mode": str(args.part_joint_candidate_mode),
            "refine": bool(args.part_joint_refine),
            "refine_iters": int(args.part_joint_refine_iters),
            "refine_pairwise": float(args.part_joint_refine_pairwise),
            "refine_margin": float(args.part_joint_refine_margin),
            "refine_margin_quantile": float(args.part_joint_refine_margin_quantile),
            "refine_neighborhood": int(args.part_joint_refine_neighborhood),
            "refine_min_vote_gain": float(args.part_joint_refine_min_vote_gain),
            "refine_preserve_small_classes": int(args.part_joint_refine_preserve_small_classes),
            "save_logits": bool(args.part_joint_save_logits),
        },
        "ss_stage": {
            "source": "input 4-view DINO tokens used by SS stage, not part_synthesis_slat",
            "fusion_mode": "concat",
            "ckpt": str(args.ss_flow_ckpt.resolve()),
            "seed": None if args.seed is None else int(args.seed),
        },
        "part_stage": {
            "backend": "promptable_seg",
            "ckpt": str(args.part_seg_ckpt.resolve()),
            "seed": None if args.seed is None else int(args.seed),
        },
        "slat_stage": {
            "flow_calls": 1,
            "part_rule": "whole SLat flow once, then slice by body_without_parts and part voxel coords",
            "ckpt": str(args.slat_flow_ckpt.resolve()),
            "seed": int(args.slat_seed),
            "condition": slat_cond_meta,
        },
        "gs_preset": GS_PRESET,
        "datasets": dataset_meta,
        "seconds": round(time.time() - started, 3),
    }
    _write_json(summary_path, summary)
    if args.fill_hidden_vertex_colors:
        if not args.export_mujoco:
            raise ValueError("--fill-hidden-vertex-colors requires --export-mujoco so component OBJ meshes exist")
        fill_out_dir = (
            Path(args.hidden_color_fill_out_dir).resolve()
            if args.hidden_color_fill_out_dir is not None
            else args.out_dir / f"{prefix}__hidden_color_fill"
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/eval/post/fill_hidden_vertex_colors.py"),
            "--summary",
            str(summary_path.resolve()),
            "--out-dir",
            str(fill_out_dir),
            "--resolution",
            str(int(args.resolution)),
            "--render-view",
            str(int(args.render_view)),
            "--dark-threshold",
            str(float(args.hidden_color_fill_dark_threshold)),
        ]
        print(f"[0617-128ee] hidden color fill -> {fill_out_dir}", flush=True)
        subprocess.run(command, check=True)
        hidden_report_path = fill_out_dir / prefix / "report.json"
        hidden_color_fill_mujoco_xml = None
        if hidden_report_path.is_file():
            hidden_report = json.loads(hidden_report_path.read_text(encoding="utf-8"))
            hidden_color_fill_mujoco_xml = _write_hidden_color_fill_mujoco_xml(
                summary=summary,
                report=hidden_report,
                out_xml=fill_out_dir / prefix / "mujoco_color_filled" / f"{prefix}.xml",
                model_name=f"{prefix}__hidden_color_filled",
            )
        summary["hidden_vertex_color_fill"] = {
            "enabled": True,
            "out_dir": str(fill_out_dir.resolve()),
            "report": str((fill_out_dir / "report.md").resolve()),
            "aggregate_report": str((fill_out_dir / "aggregate_report.json").resolve()),
            "object_report": str(hidden_report_path.resolve()),
            "mujoco_xml": None if hidden_color_fill_mujoco_xml is None else str(hidden_color_fill_mujoco_xml.resolve()),
            "dark_threshold": float(args.hidden_color_fill_dark_threshold),
            "command": command,
        }
        _write_json(summary_path, summary)
    print(f"[0617-128ee] gaussian -> {gaussian_png}", flush=True)
    print(f"[0617-128ee] mesh -> {mesh_png}", flush=True)
    if mujoco_xml is not None:
        print(f"[0617-128ee] mujoco -> {mujoco_xml}", flush=True)
    print(f"[0617-128ee] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
