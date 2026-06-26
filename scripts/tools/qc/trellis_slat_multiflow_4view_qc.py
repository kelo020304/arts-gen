#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from scripts.tools.render.render_glb_open3d_preview import render as render_open3d  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.pipelines.samplers import FlowEulerCfgSampler  # noqa: E402
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DATA_ROOT = Path("/mnt/robot-data-lab/arts-gen-data/data/PhysX-Mobility-full-4view-0511")
MANIFEST = Path(
    "/mnt/robot-data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/"
    "PhysX-Mobility-full-4view-0511/manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
)
SLAT_FLOW_CKPT = PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
MESH_DECODER_CKPT = PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/trellis_slat_multiflow_4view_qc")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    require_file(path, "manifest")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            row["_line_no"] = line_no
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def select_cases(rows: list[dict[str, Any]], cases: list[str] | None, count: int) -> list[dict[str, Any]]:
    if cases:
        wanted = set()
        for raw in cases:
            parts = raw.split(":")
            if len(parts) != 2:
                raise ValueError(f"--case must be object_id:angle_idx, got {raw!r}")
            wanted.add((parts[0], int(parts[1])))
        selected = [
            row for row in rows
            if (str(row.get("object_id", "")), int(row.get("angle_idx", row.get("angle", -1)))) in wanted
        ]
        found = {
            (str(row.get("object_id", "")), int(row.get("angle_idx", row.get("angle", -1))))
            for row in selected
        }
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"requested cases missing from manifest: {missing}")
        return selected

    selected = []
    seen = set()
    for row in rows:
        object_id = str(row.get("object_id", ""))
        angle_idx = int(row.get("angle_idx", row.get("angle", -1)))
        key = (object_id, angle_idx)
        if object_id and key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"manifest only yielded {len(selected)} unique cases; need {count}")
    return selected


def load_coords(path: Path) -> np.ndarray:
    coords = np.load(require_file(path, "surface coords"))
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords must be [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: coords is empty")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    lo = int(coords.min())
    hi = int(coords.max())
    if lo < 0 or hi >= 64:
        raise ValueError(f"{path}: coords out of [0,64), min={lo} max={hi}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def load_tokens(path: Path) -> np.ndarray:
    with np.load(require_file(path, "prenorm DINO token cache")) as data:
        if set(data.files) != {"tokens"}:
            raise ValueError(f"{path}: expected only tokens key, got {sorted(data.files)}")
        tokens = data["tokens"]
    if tokens.shape != (12, 1374, 1024):
        raise ValueError(f"{path}: tokens shape must be (12,1374,1024), got {tokens.shape}")
    if tokens.dtype != np.float32:
        raise ValueError(f"{path}: tokens dtype must be float32, got {tokens.dtype}")
    if not np.isfinite(tokens).all():
        raise ValueError(f"{path}: tokens contain NaN/Inf")
    return np.ascontiguousarray(tokens)


def tokens_for_manifest_views(data_root: Path, object_id: str, angle_idx: int, view_indices: list[int]) -> torch.Tensor:
    if len(view_indices) != 4:
        raise ValueError(f"{object_id} angle_{angle_idx}: expected 4 view_indices, got {view_indices}")
    if any(v < 0 or v >= 12 for v in view_indices):
        raise ValueError(f"{object_id} angle_{angle_idx}: view_indices out of [0,12): {view_indices}")
    token_path = data_root / "reconstruction/dinov2_tokens_prenorm" / object_id / f"angle_{angle_idx}" / "tokens.npz"
    tokens = load_tokens(token_path)
    picked = tokens[np.asarray(view_indices, dtype=np.int64)]
    return torch.from_numpy(np.ascontiguousarray(picked)).float()


def surface_path_for_row(data_root: Path, row: dict[str, Any]) -> Path:
    object_id = str(row["object_id"])
    angle_idx = int(row.get("angle_idx", row.get("angle")))
    paths = row.get("paths")
    rel = None
    if isinstance(paths, dict):
        rel = paths.get("overall_surface")
    if rel:
        path = Path(rel)
        return path if path.is_absolute() else data_root / path
    return data_root / "reconstruction/voxel_expanded" / object_id / f"angle_{angle_idx}" / "64/surface.npy"


def rgb_panel_paths(row: dict[str, Any], data_root: Path) -> list[Path]:
    paths = row.get("image_paths")
    if not isinstance(paths, list) or len(paths) != 4:
        object_id = str(row["object_id"])
        angle_idx = int(row.get("angle_idx", row.get("angle")))
        view_indices = [int(v) for v in row.get("view_indices", [])]
        paths = [
            data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb" / f"view_{view_idx}.png"
            for view_idx in view_indices
        ]
    out = []
    for path in paths:
        path = Path(path)
        out.append(path if path.is_absolute() else data_root / path)
    return [require_file(path, "input view image") for path in out]


def make_noise(coords_np: np.ndarray, seed: int) -> SparseTensor:
    coords = torch.from_numpy(coords_np).to(device="cuda", dtype=torch.int32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    sp_coords = torch.cat([batch, coords], dim=1)
    feats = inference._make_slat_initial_feats(
        coords.shape[0],
        8,
        device=coords.device,
        dtype=torch.float32,
        seed=int(seed),
    )
    return SparseTensor(coords=sp_coords, feats=feats)


def sample_multiflow_from_view_tokens(
    view_tokens: torch.Tensor,
    coords_np: np.ndarray,
    ckpt_path: Path,
    *,
    steps: int,
    seed: int,
    cfg_strength: float,
) -> SparseTensor:
    if view_tokens.shape != (4, 1374, 1024):
        raise ValueError(f"view_tokens must be [4,1374,1024], got {tuple(view_tokens.shape)}")
    model = inference._load_slat_flow(str(ckpt_path.resolve()))
    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    sample = make_noise(coords_np, seed)
    cond_views = view_tokens.to(device="cuda", dtype=torch.float32)
    neg_cond = torch.zeros((1, cond_views.shape[1], cond_views.shape[2]), device="cuda", dtype=torch.float32)
    t_seq = np.linspace(1, 0, int(steps) + 1)
    with torch.no_grad():
        for step_idx, (t, t_prev) in enumerate(zip(t_seq[:-1], t_seq[1:]), start=1):
            preds = []
            for view_idx in range(cond_views.shape[0]):
                pred = FlowEulerSampler._inference_model(
                    sampler,
                    model,
                    sample,
                    float(t),
                    cond=cond_views[view_idx:view_idx + 1],
                )
                preds.append(pred)
            pred = sum(preds) / len(preds)
            neg_pred = FlowEulerSampler._inference_model(
                sampler,
                model,
                sample,
                float(t),
                cond=neg_cond,
            )
            pred_v = (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred
            sample = sample - float(t - t_prev) * pred_v
            print(f"[multiflow-step] {step_idx}/{steps}", flush=True)
    return sample


def save_latent_npz(path: Path, slat: SparseTensor) -> None:
    coords = slat.coords.detach().cpu().numpy().astype(np.int32, copy=False)
    feats = slat.feats.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"slat coords must be [N,4], got {coords.shape}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise ValueError(f"slat feats must be [N,8], got {feats.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=coords[:, 1:], feats=feats)


def decode_mesh_and_render(
    slat: SparseTensor,
    *,
    label: str,
    out_dir: Path,
    mesh_decoder_ckpt: Path,
    render_resolution: int,
) -> dict[str, Any]:
    decoded = inference.decode_slat_assets(
        slat,
        mesh_decoder_ckpt=str(mesh_decoder_ckpt.resolve()),
        slat_is_normalized=True,
    )
    mesh = decoded.get("mesh")
    if mesh is None:
        raise RuntimeError(f"{label}: mesh decoder returned None")
    if not getattr(mesh, "success", True):
        raise RuntimeError(f"{label}: mesh decoder success=False")
    assets = save_decoded_slat_assets(decoded, out_dir, mesh_name=f"{label}.glb")
    glb = out_dir / assets["mesh"]
    renders = render_open3d(
        glb,
        out_dir=out_dir / "open3d",
        views={"iso": (315.0, 24.0), "front": (270.0, 8.0), "side": (0.0, 8.0)},
        resolution=int(render_resolution),
        use_vertex_colors=True,
    )
    return {
        "label": label,
        "glb": str(glb.resolve()),
        "renders": [str(path.resolve()) for path in renders],
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
    }


def make_case_panel(case_dir: Path, input_paths: list[Path], records: list[dict[str, Any]]) -> Path:
    images: list[tuple[str, Image.Image]] = []
    for idx, path in enumerate(input_paths):
        images.append((f"view_{idx}", Image.open(path).convert("RGB")))
    for record in records:
        iso = next((Path(path) for path in record["renders"] if path.endswith("_iso.png")), None)
        if iso is None:
            raise RuntimeError(f"{record['label']}: missing iso render")
        images.append((record["label"], Image.open(iso).convert("RGB")))

    thumb = 320
    label_h = 28
    pad = 8
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * thumb + (cols + 1) * pad, rows * (thumb + label_h) + (rows + 1) * pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(images):
        tile = Image.new("RGB", (thumb, thumb), (30, 30, 30))
        resized = image.copy()
        resized.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        tile.paste(resized, ((thumb - resized.width) // 2, (thumb - resized.height) // 2))
        x = pad + (idx % cols) * (thumb + pad)
        y = pad + (idx // cols) * (thumb + label_h + pad)
        draw.text((x + 6, y + 6), label, fill=(20, 20, 20))
        canvas.paste(tile, (x, y + label_h))
    panel_path = case_dir / "panel_inputs_concat_vs_multiflow.png"
    canvas.save(panel_path)
    return panel_path


def run_case(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    object_id = str(row["object_id"])
    angle_idx = int(row.get("angle_idx", row.get("angle")))
    view_indices = [int(v) for v in row.get("view_indices", [])]
    case_name = f"{object_id}_angle_{angle_idx}"
    case_dir = args.out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    surface_path = surface_path_for_row(args.data_root, row)
    coords_np = load_coords(surface_path)
    view_tokens = tokens_for_manifest_views(args.data_root, object_id, angle_idx, view_indices)
    input_paths = rgb_panel_paths(row, args.data_root)

    print(
        f"[case] {case_name} views={view_indices} coords={coords_np.shape[0]} "
        f"steps={args.steps} seed={args.seed}",
        flush=True,
    )

    concat = inference.run_slat_flow_from_tokens(
        view_tokens.reshape(-1, view_tokens.shape[-1]),
        torch.from_numpy(coords_np).long(),
        str(args.slat_flow_ckpt.resolve()),
        num_steps=int(args.steps),
        seed=int(args.seed),
    )
    save_latent_npz(case_dir / "concat_4v_latent.npz", concat)
    concat_record = decode_mesh_and_render(
        concat,
        label="concat_4v",
        out_dir=case_dir / "concat_4v",
        mesh_decoder_ckpt=args.mesh_decoder_ckpt,
        render_resolution=args.render_resolution,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multiflow = sample_multiflow_from_view_tokens(
        view_tokens,
        coords_np,
        args.slat_flow_ckpt,
        steps=int(args.steps),
        seed=int(args.seed),
        cfg_strength=float(args.cfg_strength),
    )
    save_latent_npz(case_dir / "multiflow_4v_latent.npz", multiflow)
    multiflow_record = decode_mesh_and_render(
        multiflow,
        label="multiflow_4v",
        out_dir=case_dir / "multiflow_4v",
        mesh_decoder_ckpt=args.mesh_decoder_ckpt,
        render_resolution=args.render_resolution,
    )
    panel = make_case_panel(case_dir, input_paths, [concat_record, multiflow_record])
    return {
        "case": case_name,
        "object_id": object_id,
        "angle_idx": int(angle_idx),
        "view_indices": view_indices,
        "surface_path": str(surface_path.resolve()),
        "surface_voxels": int(coords_np.shape[0]),
        "token_contract": "dinov2_tokens_prenorm/tokens.npz, manifest-selected 4 views, shape [4,1374,1024]",
        "concat_4v": concat_record,
        "multiflow_4v": multiflow_record,
        "panel": str(panel.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=SLAT_FLOW_CKPT)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, default=MESH_DECODER_CKPT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--case", action="append", help="object_id:angle_idx")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--render-resolution", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = require_dir(args.data_root, "data root")
    args.manifest = require_file(args.manifest, "manifest")
    args.slat_flow_ckpt = require_file(args.slat_flow_ckpt, "TRELLIS SLat flow ckpt")
    require_file(args.slat_flow_ckpt.with_suffix(".json"), "TRELLIS SLat flow config")
    args.mesh_decoder_ckpt = require_file(args.mesh_decoder_ckpt, "TRELLIS mesh decoder ckpt")
    require_file(args.mesh_decoder_ckpt.with_suffix(".json"), "TRELLIS mesh decoder config")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(args.manifest)
    selected = select_cases(rows, args.case, int(args.count))
    report = {
        "out_dir": str(args.out_dir),
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "slat_flow_ckpt": str(args.slat_flow_ckpt),
        "mesh_decoder_ckpt": str(args.mesh_decoder_ckpt),
        "mode": "concat_4v versus multiflow_4v per-step averaged view predictions",
        "cases": [],
    }
    total = len(selected)
    for idx, row in enumerate(selected, start=1):
        case_record = run_case(row, args)
        report["cases"].append(case_record)
        print(f"[slat-multiflow] finished {idx}/{total} case={case_record['case']}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "case_count": len(report["cases"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
