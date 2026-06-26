#!/usr/bin/env python3
"""Encode a full object voxel with TRELLIS SLat encoder and decode mesh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


EXPECTED_TOKEN_SHAPE = (12, 1370, 1024)
EXPECTED_PATCH_GRID = 37
EXPECTED_FEATURE_DIM = 1024


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


def load_surface(data_root: Path, object_id: str, angle_idx: int) -> np.ndarray:
    path = require_file(
        data_root / "reconstruction" / "voxel_expanded" / object_id / f"angle_{angle_idx}" / "64" / "surface.npy",
        "surface voxel",
    )
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: empty coords")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,64): min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def load_camera_matrices(camera_path: Path, view_indices: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
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


def load_patchtokens(token_path: Path, view_indices: list[int] | None = None) -> torch.Tensor:
    with np.load(require_file(token_path, "DINOv2 tokens")) as data:
        if set(data.files) != {"tokens"}:
            raise ValueError(f"{token_path}: keys must be exactly ['tokens'], got {data.files}")
        tokens = data["tokens"]
    if tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        raise ValueError(f"{token_path}: expected {EXPECTED_TOKEN_SHAPE}, got {tokens.shape}")
    if view_indices is not None:
        selected = np.asarray(view_indices, dtype=np.int64)
        if selected.ndim != 1 or selected.shape[0] == 0:
            raise ValueError("view_indices must be a non-empty 1D list")
        if int(selected.min()) < 0 or int(selected.max()) >= tokens.shape[0]:
            raise ValueError(f"{token_path}: view_indices out of range for tokens shape {tokens.shape}: {view_indices}")
        tokens = tokens[selected]
    patch = torch.from_numpy(tokens[:, 1:, :].astype(np.float32, copy=False)).cuda()
    return patch.permute(0, 2, 1).reshape(tokens.shape[0], EXPECTED_FEATURE_DIM, EXPECTED_PATCH_GRID, EXPECTED_PATCH_GRID)


def load_manifest_view_indices(manifest_path: Path, object_id: str, angle_idx: int) -> list[int]:
    manifest_path = require_file(manifest_path, "manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("object_id")) != str(object_id):
                continue
            if int(row.get("angle_idx", -1)) != int(angle_idx):
                continue
            view_indices = row.get("view_indices")
            if not isinstance(view_indices, list) or len(view_indices) != 4:
                raise ValueError(f"{manifest_path}:{line_no}: expected 4 view_indices, got {view_indices!r}")
            if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in view_indices):
                raise ValueError(f"{manifest_path}:{line_no}: view_indices must be integers, got {view_indices!r}")
            return view_indices
    raise ValueError(f"{manifest_path}: no row for object_id={object_id} angle_idx={angle_idx}")


def project_features(coords_np: np.ndarray, patchtokens: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    import utils3d

    coords = torch.as_tensor(coords_np, dtype=torch.float32, device="cuda")
    positions = (coords + 0.5) / 64.0 - 0.5
    uv = utils3d.torch.project_cv(positions, extrinsics, intrinsics)[0] * 2 - 1
    sampled = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode="bilinear",
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1)
    return sampled.mean(dim=0).float()


def load_encoder(ckpt: Path):
    from trellis.models.structured_latent_vae.encoder import SLatEncoder

    ckpt = require_file(ckpt, "TRELLIS SLat encoder ckpt")
    cfg_path = require_file(ckpt.with_suffix(".json"), "TRELLIS SLat encoder config")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SLatEncoder":
        raise ValueError(f"{cfg_path}: expected name SLatEncoder, got {cfg.get('name')!r}")
    model = SLatEncoder(**cfg["args"]).cuda().eval()
    model.load_state_dict(load_file(str(ckpt), device="cuda"), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[ckpt] encoder={ckpt}", flush=True)
    return model


def make_sparse(coords_np: np.ndarray, feats: torch.Tensor):
    from trellis.modules.sparse import SparseTensor

    coords = torch.as_tensor(coords_np.astype(np.int32, copy=False), dtype=torch.int32, device=feats.device)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def run(args: argparse.Namespace) -> None:
    data_root = require_dir(args.data_root, "DATA_ROOT")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, label in (
        (args.encoder_ckpt, "encoder ckpt"),
        (args.mesh_decoder_ckpt, "mesh decoder ckpt"),
    ):
        require_file(path, label)
        require_file(path.with_suffix(".json"), f"{label} config")

    coords = load_surface(data_root, args.object_id, args.angle_idx)
    token_path = data_root / "reconstruction" / "dinov2_tokens" / args.object_id / f"angle_{args.angle_idx}" / "tokens.npz"
    camera_path = data_root / "renders" / args.object_id / f"angle_{args.angle_idx}" / "camera_transforms.json"
    if args.manifest is not None and args.view_indices is not None:
        raise ValueError("pass only one of --manifest or --view-indices")
    view_indices = args.view_indices
    if args.manifest is not None:
        view_indices = load_manifest_view_indices(args.manifest, args.object_id, args.angle_idx)
    patchtokens = load_patchtokens(token_path, view_indices)
    extrinsics, intrinsics = load_camera_matrices(camera_path, view_indices)
    if patchtokens.shape[0] != extrinsics.shape[0]:
        raise RuntimeError(f"view count mismatch: patchtokens={patchtokens.shape[0]} cameras={extrinsics.shape[0]}")
    encoder = load_encoder(args.encoder_ckpt)
    feats = project_features(coords, patchtokens, extrinsics, intrinsics)
    sparse = make_sparse(coords, feats)
    with torch.no_grad():
        slat = encoder(sparse, sample_posterior=False)
    if not torch.isfinite(slat.feats).all():
        raise RuntimeError("encoder returned NaN/Inf SLat feats")
    decoded = inference.decode_slat_assets(
        slat,
        mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    record = save_decoded_slat_assets(decoded, out_dir, mesh_name="complete.glb")
    mesh = decoded.get("mesh")
    if mesh is None:
        raise RuntimeError("mesh decoder returned None")
    if not getattr(mesh, "success", True):
        raise RuntimeError("mesh decoder returned success=False")
    report = {
        "object_id": args.object_id,
        "angle_idx": int(args.angle_idx),
        "data_root": str(data_root),
        "encoder_ckpt": str(args.encoder_ckpt.resolve()),
        "mesh_decoder_ckpt": str(args.mesh_decoder_ckpt.resolve()),
        "surface_voxels": int(coords.shape[0]),
        "view_indices": view_indices if view_indices is not None else list(range(12)),
        "num_projected_views": int(patchtokens.shape[0]),
        "slat_rows": int(slat.feats.shape[0]),
        "slat_feat_range": [float(slat.feats.min().item()), float(slat.feats.max().item())],
        "mesh_success": bool(getattr(mesh, "success", True)),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "assets": record,
        "output_glb": str((out_dir / "complete.glb").resolve()),
        "token_path": str(token_path.resolve()),
        "camera_path": str(camera_path.resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--angle-idx", type=int, required=True)
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--view-indices", type=int, nargs="+")
    parser.add_argument("--manifest", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
