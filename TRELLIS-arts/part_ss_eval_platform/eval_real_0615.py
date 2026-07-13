from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")


def _setup_trellis_imports() -> None:
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines", "renderers"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules.setdefault(f"trellis.{sp}", mod)


_setup_trellis_imports()

from part_ss_eval_platform.eval_0615 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_PART_SEG_CKPT,
    DEFAULT_SPLIT_JSON,
    DEFAULT_SS_DECODER_CKPT,
    DEFAULT_SS_FLOW_CKPT,
    build_selection,
    load_data_config,
    _dataset_for,
    part_bucket,
)
from part_ss_eval_platform.infer_jobs import InferJobRequest, build_infer_command  # noqa: E402
import inference as trellis_inference  # noqa: E402


OUT_DIR = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0615-1")
DEFAULT_SLAT_FLOW_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
DEFAULT_SLAT_MESH_DECODER_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
FOCUS_PART_KEYWORDS = ("button", "switch", "knob", "handle", "key")
BODY_COLOR = (0.48, 0.48, 0.48, 0.30)
BASE_BODY_COLOR = (0.48, 0.48, 0.48, 0.88)


def _build_part_palette(n: int = 128) -> list[tuple[float, float, float, float]]:
    golden = 0.381966011
    sv_variants = [(0.85, 0.88), (0.62, 0.96), (0.95, 0.68)]
    palette: list[tuple[float, float, float, float]] = []
    for idx in range(n):
        hue = (idx * golden) % 1.0
        sat, val = sv_variants[idx % len(sv_variants)]
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((r, g, b, 0.98))
    return palette


PART_COLORS = _build_part_palette(128)


class VramSampler:
    def __init__(self, gpu: str, interval: float = 0.5) -> None:
        self.gpu = str(gpu).split(",")[0]
        self.interval = float(interval)
        self.max_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self.max_mib = max(self.max_mib, value)
            self._stop.wait(self.interval)

    def _query(self) -> int | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", self.gpu],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip()))
            except ValueError:
                pass
        return max(values) if values else None


def _run_dir(out_dir: Path, split: str, obj_id: str, angle: int) -> Path:
    return out_dir / "_platform_runs" / split / f"{obj_id}-{int(angle)}" / "real-B"


def _command(out_dir: Path, sample, args: argparse.Namespace, stage: str):
    req = InferJobRequest(
        stage=stage,
        object_id=sample.obj_id,
        root=str(out_dir / "_platform_runs" / sample.split),
        run_id="real-B",
        mode="B",
        view="four",
        data_config=str(args.data_config),
        angle_idx=int(sample.angle_idx),
        part_seg_ckpt=str(args.part_seg_ckpt),
        part_joint_candidate_mode=str(getattr(args, "part_joint_candidate_mode", "proposal")),
        part_joint_refine=bool(getattr(args, "part_joint_refine", False)),
        part_joint_refine_iters=int(getattr(args, "part_joint_refine_iters", 1)),
        part_joint_refine_pairwise=float(getattr(args, "part_joint_refine_pairwise", 3.0)),
        part_joint_refine_margin=float(getattr(args, "part_joint_refine_margin", 0.0)),
        part_joint_refine_margin_quantile=float(getattr(args, "part_joint_refine_margin_quantile", 0.01)),
        part_joint_refine_neighborhood=int(getattr(args, "part_joint_refine_neighborhood", 6)),
        part_joint_refine_min_vote_gain=float(getattr(args, "part_joint_refine_min_vote_gain", 0.0)),
        part_joint_refine_preserve_small_classes=int(
            getattr(args, "part_joint_refine_preserve_small_classes", 32)
        ),
        part_joint_save_logits=bool(getattr(args, "part_joint_save_logits", False)),
        ss_flow_ckpt=str(args.ss_flow_ckpt),
        ss_decoder_ckpt=str(args.ss_decoder_ckpt),
        part_backend="promptable_seg",
        decode_backend="trellis",
        gpu_ids=str(args.gpu),
        seed=getattr(args, "seed", None),
        overwrite=True,
    )
    return build_infer_command(req, repo_root=REPO_ROOT)


def _execute(spec, *, gpu: str, progress_path: Path, label: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(spec.env or {})
    Path(spec.run_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log, VramSampler(gpu) as sampler:
        log.write(f"\n[eval_real_0615] {label} cmd={' '.join(spec.args)}\n")
        log.flush()
        proc = subprocess.Popen(spec.args, cwd=spec.cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        code = proc.wait()
    rec = {
        "stage": label,
        "returncode": int(code),
        "seconds": round(time.time() - started, 3),
        "peak_vram_mib": int(sampler.max_mib),
        "log_path": str(log_path),
    }
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if code != 0:
        raise RuntimeError(f"{label} failed code={code} log={log_path}")
    return rec


def _load_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["coords"], dtype=np.int64).reshape(-1, 3)


def _valid_unique(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size == 0:
        return coords
    valid = np.all((coords >= 0) & (coords < resolution), axis=1)
    coords = coords[valid]
    return np.unique(coords, axis=0)


def _is_focus(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in FOCUS_PART_KEYWORDS)


def _is_base(name: str) -> bool:
    low = name.lower()
    return low.startswith("base_") or "base" in low or "body" in low


def _encode(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    coords = _valid_unique(coords, resolution)
    if len(coords) == 0:
        return np.empty((0,), dtype=np.int64)
    return coords[:, 0] * resolution * resolution + coords[:, 1] * resolution + coords[:, 2]


def _decode(encoded: np.ndarray, resolution: int = 64) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.int64)
    if encoded.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    x = encoded // (resolution * resolution)
    rem = encoded % (resolution * resolution)
    y = rem // resolution
    z = rem % resolution
    return np.column_stack([x, y, z]).astype(np.int64)


def _coord_label_map(
    whole_coords: np.ndarray,
    part_items: list[tuple[str, np.ndarray]],
    *,
    resolution: int = 64,
) -> tuple[dict[tuple[int, int, int], int], dict[int, tuple[float, float, float, float]], list[tuple[str, tuple[float, float, float, float]]]]:
    whole_coords = _valid_unique(whole_coords, resolution)
    indexed = [(idx + 1, name, _valid_unique(coords, resolution)) for idx, (name, coords) in enumerate(part_items)]
    indexed = [(label, name, coords) for label, name, coords in indexed if len(coords) > 0]
    encoded_parts = [_encode(coords, resolution) for _label, _name, coords in indexed if len(coords) > 0]
    part_union = _decode(np.unique(np.concatenate(encoded_parts)), resolution) if encoded_parts else np.empty((0, 3), dtype=np.int64)
    body_context_coords = _subtract(whole_coords, part_union, resolution)

    color_by_label: dict[int, tuple[float, float, float, float]] = {1: BODY_COLOR}
    label_by_coord: dict[tuple[int, int, int], int] = {
        (int(x), int(y), int(z)): 1
        for x, y, z in body_context_coords
    }

    focus_encoded = [_encode(coords, resolution) for _label, name, coords in indexed if _is_focus(name)]
    focus_union = _decode(np.unique(np.concatenate(focus_encoded)), resolution) if focus_encoded else np.empty((0, 3), dtype=np.int64)

    def order(item):
        label, name, coords = item
        layer = 0 if _is_base(name) else 2 if _is_focus(name) else 1
        return (layer, -len(coords), label)

    legend_items: list[tuple[str, tuple[float, float, float, float]]] = [("whole/body", BODY_COLOR)]
    for label, name, coords in sorted(indexed, key=order):
        if _is_base(name):
            coords = _subtract(coords, focus_union, resolution)
            if len(coords) == 0:
                continue
            color = tuple(float(x) for x in BASE_BODY_COLOR)
        else:
            color = tuple(float(x) for x in PART_COLORS[(label - 1) % len(PART_COLORS)])
        render_label = int(label) + 1
        color_by_label[render_label] = color
        for x, y, z in coords:
            label_by_coord[(int(x), int(y), int(z))] = render_label
        if len(legend_items) < 22:
            legend_items.append((name[:28], color))
    return label_by_coord, color_by_label, legend_items


def _occupied_bbox(*coord_sets: np.ndarray, resolution: int = 64) -> tuple[np.ndarray, np.ndarray]:
    valid_sets = [_valid_unique(coords, resolution) for coords in coord_sets if len(coords) > 0]
    valid_sets = [coords for coords in valid_sets if len(coords) > 0]
    if not valid_sets:
        return np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64)
    merged = np.concatenate(valid_sets, axis=0)
    lo = np.maximum(merged.min(axis=0) - 1, 0).astype(np.int64)
    hi = np.minimum(merged.max(axis=0) + 2, resolution).astype(np.int64)
    return lo, hi


def _crop_for_voxels(
    filled: np.ndarray,
    colors: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sx = slice(int(lo[0]), int(hi[0]))
    sy = slice(int(lo[1]), int(hi[1]))
    sz = slice(int(lo[2]), int(hi[2]))
    cropped_filled = filled[sx, sy, sz]
    cropped_colors = colors[sx, sy, sz]
    x, y, z = np.indices(np.asarray(cropped_filled.shape) + 1)
    x = x + int(lo[0])
    y = y + int(lo[1])
    z = z + int(lo[2])
    return x, y, z, cropped_filled, cropped_colors


def _subtract(coords: np.ndarray, remove: np.ndarray, resolution: int = 64) -> np.ndarray:
    if len(coords) == 0 or len(remove) == 0:
        return coords
    keep = ~np.isin(_encode(coords, resolution), _encode(remove, resolution))
    return coords[keep]


def _setup_axes(ax, obj_id: str, angle: int) -> None:
    ax.set_title(f"{obj_id} angle_{angle} - real inference voxel", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-45)


def _face_vertices(x: int, y: int, z: int, direction: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    dx, dy, dz = direction
    if dx == -1:
        return [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)]
    if dx == 1:
        return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
    if dy == -1:
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
    if dy == 1:
        return [(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)]
    if dz == -1:
        return [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)]
    return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]


def _surface_faces(
    labels: np.ndarray,
    color_by_label: dict[int, tuple[float, float, float, float]],
    *,
    resolution: int = 64,
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[float, float, float, float]]]:
    faces: list[list[tuple[int, int, int]]] = []
    facecolors: list[tuple[float, float, float, float]] = []
    coords = np.argwhere(labels > 0)
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for x, y, z in coords:
        label = int(labels[x, y, z])
        for direction in directions:
            nx, ny, nz = int(x + direction[0]), int(y + direction[1]), int(z + direction[2])
            if 0 <= nx < resolution and 0 <= ny < resolution and 0 <= nz < resolution and int(labels[nx, ny, nz]) == label:
                continue
            faces.append(_face_vertices(int(x), int(y), int(z), direction))
            facecolors.append(color_by_label[label])
    return faces, facecolors


def render_preview_voxel(
    whole_coords: np.ndarray,
    part_items: list[tuple[str, np.ndarray]],
    out_path: Path,
    obj_id: str,
    angle: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path.parent.mkdir(parents=True, exist_ok=True)
    whole_coords = _valid_unique(whole_coords)
    label_volume = np.zeros((64, 64, 64), dtype=np.int16)
    label_by_coord, color_by_label, legend_items = _coord_label_map(whole_coords, part_items)
    for (x, y, z), label in label_by_coord.items():
        label_volume[x, y, z] = label
    legend = [
        Patch(facecolor=color[:3], edgecolor="none", alpha=color[3], label=name)
        for name, color in legend_items
    ]

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, obj_id, angle)
    faces, colors = _surface_faces(label_volume, color_by_label)
    if faces:
        collection = Poly3DCollection(
            faces,
            facecolors=colors,
            edgecolors=(0.04, 0.04, 0.04, 0.24),
            linewidths=0.08,
        )
        collection.set_zsort("average")
        ax.add_collection3d(collection)
    if legend:
        ax.legend(handles=legend, loc="upper left", fontsize=7, framealpha=0.82)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.94)
    fig.savefig(out_path)
    plt.close(fig)


def _result_png_paths(out_dir: Path, sample, duplicate_counts: dict[tuple[str, str], int]) -> tuple[Path, Path]:
    obj_dir = out_dir / sample.split / sample.obj_id
    if duplicate_counts[(sample.split, sample.obj_id)] > 1:
        stem = f"result_angle_{int(sample.angle_idx)}"
    else:
        stem = "result"
    return obj_dir / f"{stem}_voxel.png", obj_dir / f"{stem}_mesh.png"


def _rgba_view_image(data_root: Path, object_id: str, angle_idx: int, view_idx: int):
    from PIL import Image

    rgb_path = data_root / "renders" / object_id / f"angle_{int(angle_idx)}" / "rgb" / f"view_{int(view_idx)}.png"
    if not rgb_path.is_file():
        raise FileNotFoundError(f"SLat input RGB view not found: {rgb_path}")
    image = Image.open(rgb_path)
    if image.mode == "RGBA" or "A" in image.getbands():
        return image.convert("RGBA")
    mask_candidates = [
        data_root / "renders" / object_id / f"angle_{int(angle_idx)}" / "mask" / f"mask_{int(view_idx)}.npy",
        data_root / "renders" / object_id / f"angle_{int(angle_idx)}" / "mask" / f"mask_{int(view_idx)}.png",
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


def _load_slat_cond_tokens(ds, sample: dict[str, Any], token_source: str = "live") -> tuple[torch.Tensor, dict[str, Any]]:
    data_root = Path(ds.data_root)
    view_indices = [int(v) for v in sample["view_indices"]]
    if len(view_indices) != 4:
        raise ValueError(f"{sample['obj_id']} angle={sample['angle_idx']} expected 4 view indices, got {view_indices}")
    if token_source == "live":
        images = [
            _rgba_view_image(data_root, str(sample["obj_id"]), int(sample["angle_idx"]), view_idx)
            for view_idx in view_indices
        ]
        picked = trellis_inference._images_to_tokens(images).detach().float().cpu()
        return picked.reshape(-1, picked.shape[-1]), {
            "token_source": "live_official_trellis_rgba",
            "preprocess": "TRELLIS RGBA alpha crop + black premultiply + 518 resize + DINO x_prenorm layer_norm",
            "view_indices": view_indices,
            "picked_token_shape": list(picked.shape),
            "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
        }
    if token_source != "cache":
        raise ValueError(f"unsupported SLat token source: {token_source!r}")
    token_candidates = [
        data_root / ds.recon_subdir / "dinov2_tokens" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_prenorm" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_official_prenorm1374" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
    ]
    token_path = next((path for path in token_candidates if path.is_file()), token_candidates[0])
    if not token_path.is_file():
        raise FileNotFoundError(f"TRELLIS SLat DINO tokens not found: {token_path}")
    with np.load(token_path, allow_pickle=False) as data:
        if "tokens" not in data.files:
            raise KeyError(f"{token_path} expected key 'tokens', got {data.files}")
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError(f"{token_path} expected [V,T,1024], got {tokens.shape}")
    if max(view_indices) >= tokens.shape[0] or min(view_indices) < 0:
        raise ValueError(f"{token_path} has {tokens.shape[0]} views, cannot select {view_indices}")
    picked = torch.from_numpy(np.ascontiguousarray(tokens[view_indices])).float()
    picked = torch.nn.functional.layer_norm(picked, picked.shape[-1:])
    return picked.reshape(-1, picked.shape[-1]), {
        "token_source": "cache",
        "token_path": str(token_path),
        "view_indices": view_indices,
        "picked_token_shape": list(picked.shape),
        "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
    }


def _mesh_numpy(mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise TypeError(f"decoded mesh lacks vertices/faces: {type(mesh).__name__}")
    if torch.is_tensor(vertices):
        vertices = vertices.detach().float().cpu().numpy()
    else:
        vertices = np.asarray(vertices, dtype=np.float32)
    if torch.is_tensor(faces):
        faces = faces.detach().long().cpu().numpy()
    else:
        faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _vertex_voxel_labels(
    vertices: np.ndarray,
    label_by_coord: dict[tuple[int, int, int], int],
    *,
    resolution: int = 64,
) -> np.ndarray:
    if vertices.size == 0:
        return np.empty((0,), dtype=np.int16)
    coords = np.floor((np.asarray(vertices, dtype=np.float32) + 0.5) * resolution).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    labels = np.zeros((coords.shape[0],), dtype=np.int16)
    offsets = [
        (0, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ]
    for idx, (x, y, z) in enumerate(coords):
        for dx, dy, dz in offsets:
            key = (
                int(np.clip(x + dx, 0, resolution - 1)),
                int(np.clip(y + dy, 0, resolution - 1)),
                int(np.clip(z + dz, 0, resolution - 1)),
            )
            label = label_by_coord.get(key)
            if label is not None:
                labels[idx] = int(label)
                break
    labels[labels == 0] = 1
    return labels


def _mesh_vertex_colors(
    vertices: np.ndarray,
    label_by_coord: dict[tuple[int, int, int], int],
    color_by_label: dict[int, tuple[float, float, float, float]],
) -> np.ndarray:
    labels = _vertex_voxel_labels(vertices, label_by_coord)
    colors = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    for label in np.unique(labels):
        color = color_by_label.get(int(label), BODY_COLOR)
        colors[labels == label] = np.asarray(color[:3], dtype=np.float64)
    return colors


def _camera_vectors(azimuth_deg: float, elevation_deg: float, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import math

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array(
        [
            distance * math.cos(el) * math.cos(az),
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
        ],
        dtype=np.float64,
    )
    center = np.zeros(3, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return eye, center, up


def _render_mesh_open3d(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path, resolution: int) -> None:
    import open3d as o3d
    from PIL import Image

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = float(np.max(bbox.get_extent()))
    if extent <= 0:
        raise ValueError("invalid zero-size mesh bounds")
    mesh.translate(-center, relative=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(int(resolution), int(resolution))
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.LightingProfile.MED_SHADOWS,
        (0.35, -0.45, -0.82),
    )
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.base_roughness = 0.78
    material.base_metallic = 0.0
    scene.add_geometry("mesh", mesh, material)

    distance = extent * 2.2
    eye, target, up = _camera_vectors(315.0, 24.0, distance)
    scene.camera.look_at(target, eye, up)
    scene.camera.set_projection(
        35.0,
        1.0,
        max(0.001, distance - extent * 1.5),
        distance + extent * 1.5,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    image = renderer.render_to_image()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image)).save(out_path)
    del renderer


def _compose_mesh_comparison(
    colored_path: Path,
    plain_path: Path,
    out_path: Path,
    *,
    title_left: str = "mask colors",
    title_right: str = "plain mesh",
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def _font(size: int) -> ImageFont.ImageFont:
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ):
            path = Path(candidate)
            if path.is_file():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    left = Image.open(colored_path).convert("RGB")
    right = Image.open(plain_path).convert("RGB")
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((round(left.width * h / left.height), h), Image.Resampling.LANCZOS)
    if right.height != h:
        right = right.resize((round(right.width * h / right.height), h), Image.Resampling.LANCZOS)
    header = max(34, int(h * 0.045))
    gap = max(8, int(h * 0.012))
    canvas = Image.new("RGB", (left.width + right.width + gap, h + header), (255, 255, 255))
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))
    draw = ImageDraw.Draw(canvas)
    font = _font(max(16, int(header * 0.52)))
    draw.text((12, max(4, int(header * 0.18))), title_left, fill=(0, 0, 0), font=font)
    draw.text((left.width + gap + 12, max(4, int(header * 0.18))), title_right, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _render_mesh_matplotlib(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path, obj_id: str, angle: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{obj_id} angle_{angle} - TRELLIS SLat mesh", fontsize=11)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-45)
    if len(vertices) > 0 and len(faces) > 0:
        tri = vertices[faces]
        facecolors = colors[faces].mean(axis=1)
        collection = Poly3DCollection(tri, facecolors=facecolors, edgecolors=(0.05, 0.05, 0.05, 0.08), linewidths=0.02)
        collection.set_zsort("average")
        ax.add_collection3d(collection)
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        center = (lo + hi) / 2.0
        span = max(float(np.max(hi - lo)), 1e-3)
        ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        ax.set_zlim(center[2] - span / 2.0, center[2] + span / 2.0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_trellis_slat_mesh(
    ds,
    ds_sample: dict[str, Any],
    whole_coords: np.ndarray,
    part_items: list[tuple[str, np.ndarray]],
    out_path: Path,
    obj_id: str,
    angle: int,
    args: argparse.Namespace,
    *,
    progress_path: Path,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    whole_coords = _valid_unique(whole_coords)
    if len(whole_coords) == 0:
        raise ValueError(f"{obj_id} angle={angle}: empty whole voxel coords for TRELLIS SLat mesh")
    cond, cond_meta = _load_slat_cond_tokens(ds, ds_sample, token_source=str(getattr(args, "slat_token_source", "live")))
    label_by_coord, color_by_label, _legend_items = _coord_label_map(whole_coords, part_items)
    started = time.time()
    with VramSampler(str(args.gpu)) as sampler:
        trellis_inference._load_slat_vae_decoder.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        slat = trellis_inference.run_slat_flow_from_tokens(
            cond,
            torch.from_numpy(whole_coords).long(),
            str(args.slat_flow_ckpt),
            num_steps=int(args.slat_steps),
            seed=int(args.slat_seed),
        )
        trellis_inference._load_slat_flow.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        decoded = trellis_inference.decode_slat_assets(
            slat,
            mesh_decoder_ckpt=str(args.slat_mesh_decoder_ckpt),
            slat_is_normalized=True,
        )
        trellis_inference._load_slat_vae_decoder.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mesh = decoded.get("mesh")
        if mesh is None:
            raise RuntimeError(f"{obj_id} angle={angle}: TRELLIS mesh decoder returned None")
        if not getattr(mesh, "success", True):
            raise RuntimeError(f"{obj_id} angle={angle}: TRELLIS mesh decoder success=False")
        vertices, faces = _mesh_numpy(mesh)
        colors = _mesh_vertex_colors(vertices, label_by_coord, color_by_label)
        plain_colors = np.full_like(colors, 0.78, dtype=np.float64)
        colored_tmp = out_path.with_name(f".{out_path.stem}.mask_color.tmp.png")
        plain_tmp = out_path.with_name(f".{out_path.stem}.plain.tmp.png")
        try:
            _render_mesh_open3d(vertices, faces, colors, colored_tmp, int(args.mesh_render_resolution))
            _render_mesh_open3d(vertices, faces, plain_colors, plain_tmp, int(args.mesh_render_resolution))
            _compose_mesh_comparison(colored_tmp, plain_tmp, out_path)
            renderer = "open3d"
        except Exception as exc:
            _render_mesh_matplotlib(vertices, faces, colors, colored_tmp, obj_id, angle)
            _render_mesh_matplotlib(vertices, faces, plain_colors, plain_tmp, obj_id, angle)
            _compose_mesh_comparison(colored_tmp, plain_tmp, out_path)
            renderer = f"matplotlib_fallback:{type(exc).__name__}:{exc}"
        finally:
            colored_tmp.unlink(missing_ok=True)
            plain_tmp.unlink(missing_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec = {
        "stage": f"B/trellis_slat_mesh/{obj_id}/{angle}",
        "seconds": round(time.time() - started, 3),
        "peak_vram_mib": int(sampler.max_mib),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "renderer": renderer,
        "slat_condition": cond_meta,
        "out_path": str(out_path),
    }
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _summarize(rows: list[dict[str, Any]], peak_vram: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["bucket"])].append(row)
        groups[(row["split"], "all")].append(row)
    out = []
    for split in ("train", "held"):
        for bucket in ("tiny", "small", "medium", "large", "button", "all"):
            group = groups.get((split, bucket), [])
            if not group:
                continue
            out.append({
                "split": split,
                "bucket": bucket,
                "n": len(group),
                "mean_IoU": float(np.mean([float(r["IoU"]) for r in group])),
                "success@IoU0.5": float(np.mean([int(r["hit@0.5"]) for r in group])),
                "peak_vram_mib": int(peak_vram),
            })
    return out


def _coords_iou(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = _valid_unique(pred)
    gt = _valid_unique(gt)
    pred_set = set(map(tuple, pred.tolist()))
    gt_set = set(map(tuple, gt.tolist()))
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return {
        "IoU": float(inter / union) if union else 1.0,
        "pred_voxels": int(len(pred_set)),
        "raw_voxels": int(len(gt_set)),
    }


def collect_real_metrics(ds, selected, out_dir: Path, peak_vram: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in selected:
        run_dir = _run_dir(out_dir, sample.split, sample.obj_id, int(sample.angle_idx))
        _, ds_sample = next((i, s) for i, s in enumerate(ds.samples) if str(s["obj_id"]) == sample.obj_id and int(s["angle_idx"]) == int(sample.angle_idx))
        for part_idx, part in enumerate(ds_sample["parts"]):
            part_name = str(part["part_name"])
            pred_path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
            pred = _load_coords(pred_path) if pred_path.is_file() else np.empty((0, 3), dtype=np.int64)
            raw = ds._load_raw_ind_coords(ds_sample, part).numpy().astype(np.int64)
            metric = _coords_iou(pred, raw)
            iou = float(metric["IoU"])
            raw_count = int(metric["raw_voxels"])
            rows.append({
                "split": sample.split,
                "obj_id": sample.obj_id,
                "angle": int(sample.angle_idx),
                "part_name": part_name,
                "bucket": part_bucket(part_name, part, raw_count),
                "raw_voxels": raw_count,
                "pred_voxels": int(metric["pred_voxels"]),
                "IoU": iou,
                "hit@0.5": int(iou >= 0.5),
                "peak_vram_mib": int(peak_vram),
            })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-inference-only 0615 eval with preview-style voxel PNGs.")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--gpu", default="0")
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    p.add_argument("--part-seg-ckpt", default=str(DEFAULT_PART_SEG_CKPT))
    p.add_argument("--ss-flow-ckpt", default=str(DEFAULT_SS_FLOW_CKPT))
    p.add_argument("--ss-decoder-ckpt", default=str(DEFAULT_SS_DECODER_CKPT))
    p.add_argument("--slat-flow-ckpt", default=str(DEFAULT_SLAT_FLOW_CKPT))
    p.add_argument("--slat-mesh-decoder-ckpt", default=str(DEFAULT_SLAT_MESH_DECODER_CKPT))
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--slat-seed", type=int, default=42)
    p.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="SLat flow condition source. live is the accepted TRELLIS RGBA preprocessing path.",
    )
    p.add_argument("--mesh-render-resolution", type=int, default=768)
    p.add_argument("--per-split", type=int, default=64)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--keep-runs", action="store_true")
    return p.parse_args()


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.slat_flow_ckpt = _require_file(Path(args.slat_flow_ckpt), "TRELLIS SLat flow ckpt")
    _require_file(args.slat_flow_ckpt.with_suffix(".json"), "TRELLIS SLat flow config")
    args.slat_mesh_decoder_ckpt = _require_file(Path(args.slat_mesh_decoder_ckpt), "TRELLIS SLat mesh decoder ckpt")
    _require_file(args.slat_mesh_decoder_ckpt.with_suffix(".json"), "TRELLIS SLat mesh decoder config")
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dc = load_data_config(Path(args.data_config))
    ds = _dataset_for("four", dc)
    selected, selection_manifest = build_selection(ds, Path(args.split_json), per_split=int(args.per_split))
    if int(args.limit_samples) > 0:
        selected = selected[: int(args.limit_samples)]
        selection_manifest["limit_samples"] = int(args.limit_samples)
    (out_dir / "selection.json").write_text(json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in selected:
        duplicate_counts[(sample.split, sample.obj_id)] += 1
    progress_path = out_dir / "progress.jsonl"
    records: list[dict[str, Any]] = []
    peak_vram = 0
    for idx, sample in enumerate(selected, 1):
        print(f"[eval_real_0615] {idx}/{len(selected)} {sample.split} {sample.obj_id} angle={sample.angle_idx} bucket={sample.bucket}", flush=True)
        for stage in ("ss", "part"):
            spec = _command(out_dir, sample, args, stage)
            rec = _execute(spec, gpu=args.gpu, progress_path=progress_path, label=f"B/{stage}/{sample.split}/{sample.obj_id}/{sample.angle_idx}")
            records.append(rec)
            peak_vram = max(peak_vram, int(rec["peak_vram_mib"]))
        run_dir = _run_dir(out_dir, sample.split, sample.obj_id, int(sample.angle_idx))
        part_paths = sorted((run_dir / "parts").glob("part_*_voxel.npz"))
        part_items = []
        _, ds_sample = next((i, s) for i, s in enumerate(ds.samples) if str(s["obj_id"]) == sample.obj_id and int(s["angle_idx"]) == int(sample.angle_idx))
        for part_idx, part in enumerate(ds_sample["parts"]):
            path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
            if path.is_file():
                part_items.append((str(part["part_name"]), _load_coords(path)))
        whole_path = run_dir / "voxel.npz"
        whole_coords = _load_coords(whole_path) if whole_path.is_file() else np.empty((0, 3), dtype=np.int64)
        voxel_path, mesh_path = _result_png_paths(out_dir, sample, duplicate_counts)
        render_preview_voxel(
            whole_coords,
            part_items,
            voxel_path,
            sample.obj_id,
            int(sample.angle_idx),
        )
        try:
            mesh_rec = render_trellis_slat_mesh(
                ds,
                ds_sample,
                whole_coords,
                part_items,
                mesh_path,
                sample.obj_id,
                int(sample.angle_idx),
                args,
                progress_path=progress_path,
            )
            records.append(mesh_rec)
            peak_vram = max(peak_vram, int(mesh_rec["peak_vram_mib"]))
        except Exception as exc:
            mesh_rec = {
                "stage": f"B/trellis_slat_mesh/{sample.split}/{sample.obj_id}/{sample.angle_idx}",
                "returncode": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "out_path": str(mesh_path),
            }
            records.append(mesh_rec)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(mesh_rec, ensure_ascii=False) + "\n")
            print(f"[eval_real_0615] mesh failed {sample.split} {sample.obj_id} angle={sample.angle_idx}: {type(exc).__name__}: {exc}", flush=True)

    metrics = collect_real_metrics(ds, selected, out_dir, peak_vram)
    summary = _summarize(metrics, peak_vram)
    _write_csv(out_dir / "metrics.csv", metrics, ["split", "obj_id", "angle", "part_name", "bucket", "raw_voxels", "pred_voxels", "IoU", "hit@0.5", "peak_vram_mib"])
    _write_csv(out_dir / "metrics_summary.csv", summary, ["split", "bucket", "n", "mean_IoU", "success@IoU0.5", "peak_vram_mib"])
    (out_dir / "peak_vram.txt").write_text(f"{peak_vram}\n", encoding="utf-8")
    (out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.keep_runs:
        shutil.rmtree(out_dir / "_platform_runs", ignore_errors=True)
    for name in ("selection.json", "progress.jsonl", "records.json", "peak_vram.txt"):
        (out_dir / name).unlink(missing_ok=True)
    print(f"[eval_real_0615] done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
