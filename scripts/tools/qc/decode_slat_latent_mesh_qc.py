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
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_case(raw: str) -> tuple[str, int]:
    parts = raw.split(":")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"--case must be object_id:angle_idx, got {raw!r}")
    return parts[0], int(parts[1])


def load_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(require_file(path, "SLat latent")) as data:
        if set(data.files) != {"coords", "feats"}:
            raise ValueError(f"{path}: expected keys ['coords','feats'], got {sorted(data.files)}")
        coords = data["coords"]
        feats = data["feats"]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: coords must be [N,3], got {coords.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: empty coords")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,64), min={coords.min()} max={coords.max()}")
    if feats.ndim != 2 or feats.shape[1] != 8:
        raise ValueError(f"{path}: feats must be [N,8], got {feats.shape}")
    if feats.dtype != np.float32:
        raise ValueError(f"{path}: feats dtype must be float32, got {feats.dtype}")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(f"{path}: coords rows {coords.shape[0]} != feats rows {feats.shape[0]}")
    if not np.isfinite(feats).all():
        raise ValueError(f"{path}: feats contain NaN/Inf")
    return np.ascontiguousarray(coords.astype(np.int32, copy=False)), np.ascontiguousarray(feats)


def make_sparse(coords_np: np.ndarray, feats_np: np.ndarray) -> SparseTensor:
    coords = torch.from_numpy(coords_np).to(device="cuda", dtype=torch.int32)
    feats = torch.from_numpy(feats_np).to(device="cuda", dtype=torch.float32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def decode_one(
    latent_path: Path,
    *,
    label: str,
    mesh_decoder_ckpt: Path,
    out_root: Path,
    render_resolution: int,
) -> dict[str, Any]:
    coords_np, feats_np = load_latent(latent_path)
    slat = make_sparse(coords_np, feats_np)
    decoded = inference.decode_slat_assets(
        slat,
        mesh_decoder_ckpt=str(mesh_decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    mesh = decoded.get("mesh")
    if mesh is None:
        raise RuntimeError(f"{label}: mesh decoder returned None")
    if not getattr(mesh, "success", True):
        raise RuntimeError(f"{label}: mesh decoder success=False")
    case_dir = out_root / label
    assets = save_decoded_slat_assets(decoded, case_dir, mesh_name=f"{label}.glb")
    glb = case_dir / assets["mesh"]
    renders = render_open3d(
        glb,
        out_dir=case_dir / "open3d",
        views={"iso": (315.0, 24.0), "front": (270.0, 8.0), "side": (0.0, 8.0)},
        resolution=render_resolution,
        use_vertex_colors=True,
    )
    return {
        "label": label,
        "latent": str(latent_path.resolve()),
        "coords_shape": list(coords_np.shape),
        "feats_shape": list(feats_np.shape),
        "feats_mean": float(feats_np.mean()),
        "feats_std": float(feats_np.std()),
        "glb": str(glb.resolve()),
        "open3d_renders": [str(path.resolve()) for path in renders],
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
    }


def make_panel(records: list[dict[str, Any]], out_path: Path) -> None:
    iso_paths = []
    for record in records:
        iso = next((Path(path) for path in record["open3d_renders"] if path.endswith("_iso.png")), None)
        if iso is None:
            raise RuntimeError(f"{record['label']}: missing iso render")
        iso_paths.append((record["label"], iso))
    images = [(label, Image.open(path).convert("RGB")) for label, path in iso_paths]
    if not images:
        raise RuntimeError("panel needs at least one image")
    cell_w, cell_h = images[0][1].size
    label_h = 28
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * (cell_h + label_h)
        draw.rectangle((x, y, x + cell_w, y + label_h), fill=(255, 255, 255))
        draw.text((x + 8, y + 7), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_case(data_root: Path, object_id: str, angle_idx: int, args: argparse.Namespace) -> dict[str, Any]:
    inst = f"{object_id}_angle_{angle_idx}"
    inst_root = require_dir(data_root / "part_synthesis_slat" / object_id[:2] / inst, "SLat instance root")
    out_root = args.out_dir / inst
    records = []
    overall = inst_root / "overall" / "latent.npz"
    records.append(
        decode_one(
            overall,
            label="overall",
            mesh_decoder_ckpt=args.mesh_decoder_ckpt,
            out_root=out_root,
            render_resolution=args.render_resolution,
        )
    )
    part_paths = sorted(path for path in inst_root.glob("*/latent.npz") if path.parent.name != "overall")
    if not part_paths:
        raise RuntimeError(f"{inst_root}: no part latent.npz files found")
    for part_path in part_paths:
        records.append(
            decode_one(
                part_path,
                label=part_path.parent.name,
                mesh_decoder_ckpt=args.mesh_decoder_ckpt,
                out_root=out_root,
                render_resolution=args.render_resolution,
            )
        )
    panel_path = out_root / "panel_iso.png"
    make_panel(records, panel_path)
    return {
        "object_id": object_id,
        "angle_idx": int(angle_idx),
        "instance": inst,
        "instance_root": str(inst_root.resolve()),
        "records": records,
        "panel": str(panel_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="object_id:angle_idx")
    parser.add_argument("--render-resolution", type=int, default=768)
    args = parser.parse_args()
    data_root = require_dir(args.data_root, "data root")
    args.mesh_decoder_ckpt = require_file(args.mesh_decoder_ckpt, "TRELLIS mesh decoder ckpt")
    require_file(args.mesh_decoder_ckpt.with_suffix(".json"), "TRELLIS mesh decoder config")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "data_root": str(data_root.resolve()),
        "mesh_decoder_ckpt": str(args.mesh_decoder_ckpt.resolve()),
        "cases": [],
    }
    for raw_case in args.case:
        object_id, angle_idx = parse_case(raw_case)
        report["cases"].append(run_case(data_root, object_id, angle_idx, args))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "case_count": len(report["cases"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
