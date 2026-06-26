#!/usr/bin/env python3
"""SAM3D SLat encoder -> decoder round-trip on fixed voxel coords.

This diagnostic is the SAM3D counterpart of trellis_full_voxel_mesh_roundtrip.py:
project existing DINO tokens onto GT surface coords, encode a raw SAM3D SLat,
then decode it with the SAM3D mesh/gaussian decoders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate


REPO = Path(__file__).resolve().parents[2]
SAM3D_ROOT = REPO / "submodules" / "sam3d-stage" / "submodules" / "sam-3d-objects"
GLUE = REPO / "submodules" / "sam3d-stage" / "infer_glue"
TRELLIS_ROOT = REPO / "TRELLIS-arts"
for item in (str(SAM3D_ROOT), str(GLUE), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import slat_stage  # noqa: E402
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


EXPECTED_TOKEN_SHAPE = (12, 1370, 1024)
PATCH_GRID = 37
FEATURE_DIM = 1024


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
        raise ValueError(f"{path}: expected [N,3], got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: empty coords")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(f"{path}: coords dtype must be integer, got {coords.dtype}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"{path}: coords out of [0,64), min={coords.min()} max={coords.max()}")
    return np.ascontiguousarray(coords.astype(np.int32, copy=False))


def load_manifest_view_indices(manifest_path: Path, object_id: str, angle_idx: int) -> list[int]:
    manifest_path = require_file(manifest_path, "manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("object_id")) == str(object_id) and int(row.get("angle_idx", -1)) == int(angle_idx):
                view_indices = row.get("view_indices")
                if not isinstance(view_indices, list) or len(view_indices) != 4:
                    raise ValueError(f"{manifest_path}:{line_no}: expected 4 view_indices, got {view_indices!r}")
                if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in view_indices):
                    raise ValueError(f"{manifest_path}:{line_no}: view_indices must be integer list")
                return view_indices
    raise ValueError(f"{manifest_path}: no row for object_id={object_id} angle_idx={angle_idx}")


def load_camera_matrices(camera_path: Path, view_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    import utils3d

    payload = json.loads(require_file(camera_path, "camera_transforms").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 12:
        raise ValueError(f"{camera_path}: expected 12 frames")
    bad = [idx for idx in view_indices if idx < 0 or idx >= len(frames)]
    if bad:
        raise ValueError(f"{camera_path}: view_indices out of range: {bad}")
    extrinsics = []
    intrinsics = []
    for idx in view_indices:
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


def load_xnorm_patchtokens(token_path: Path, view_indices: list[int]) -> torch.Tensor:
    with np.load(require_file(token_path, "DINOv2 tokens")) as data:
        if set(data.files) != {"tokens"}:
            raise ValueError(f"{token_path}: keys must be exactly ['tokens'], got {data.files}")
        tokens = data["tokens"]
    if tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
        raise ValueError(f"{token_path}: expected {EXPECTED_TOKEN_SHAPE}, got {tokens.shape}")
    selected = tokens[np.asarray(view_indices, dtype=np.int64)]
    patch = torch.from_numpy(selected[:, 1:, :].astype(np.float32, copy=False)).cuda()
    return patch.permute(0, 2, 1).reshape(len(view_indices), FEATURE_DIM, PATCH_GRID, PATCH_GRID)


def project_features(coords_np: np.ndarray, patchtokens: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
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


def load_slat_encoder(config_path: Path, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    config = OmegaConf.load(require_file(config_path, "SLat encoder config"))
    encoder = instantiate(config)
    state = torch.load(require_file(ckpt_path, "SLat encoder ckpt"), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{ckpt_path}: checkpoint did not load as state dict")
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    print(
        "[contract] input=sp.SparseTensor(coords int32 [N,4], feats float32 [N,1024]); "
        "forward=encoder(x, sample_posterior=False); output=raw SparseTensor feats [N,8]",
        flush=True,
    )
    return encoder


def make_sparse(coords_np: np.ndarray, feats: torch.Tensor) -> sp.SparseTensor:
    coords = torch.as_tensor(coords_np, dtype=torch.int32, device=feats.device)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return sp.SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def run(args: argparse.Namespace) -> None:
    data_root = require_dir(args.data_root, "DATA_ROOT")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest is not None and args.view_indices is not None:
        raise ValueError("pass only one of --manifest or --view-indices")
    view_indices = list(args.view_indices) if args.view_indices is not None else None
    if args.manifest is not None:
        view_indices = load_manifest_view_indices(args.manifest, args.object_id, args.angle_idx)
    if view_indices is None:
        view_indices = list(range(12))

    coords = load_surface(data_root, args.object_id, args.angle_idx)
    token_path = data_root / "reconstruction" / "dinov2_tokens" / args.object_id / f"angle_{args.angle_idx}" / "tokens.npz"
    camera_path = data_root / "renders" / args.object_id / f"angle_{args.angle_idx}" / "camera_transforms.json"
    patchtokens = load_xnorm_patchtokens(token_path, view_indices)
    extrinsics, intrinsics = load_camera_matrices(camera_path, view_indices)
    feats = project_features(coords, patchtokens, extrinsics, intrinsics)

    device = torch.device(args.device)
    encoder = load_slat_encoder(args.encoder_config, args.encoder_ckpt, device)
    sparse = make_sparse(coords, feats.to(device))
    with torch.no_grad():
        slat = encoder(sparse, sample_posterior=False)
    if not torch.isfinite(slat.feats).all():
        raise RuntimeError("encoder returned NaN/Inf SLat feats")

    np.savez_compressed(
        out_dir / "latent.npz",
        coords=slat.coords.detach().cpu().numpy().astype(np.int32),
        feats=slat.feats.detach().float().cpu().numpy().astype(np.float32),
        normalized=np.array(False),
        view_indices=np.asarray(view_indices, dtype=np.int32),
    )

    pipe = slat_stage._build_pipeline(args.pipeline_config, args.device, load_mesh_decoder=True)
    try:
        decoded = pipe.decode_slat(slat, ["mesh", "gaussian"])
        mesh = decoded["mesh"][0] if isinstance(decoded.get("mesh"), list) else decoded.get("mesh")
        gaussian = decoded["gaussian"][0] if isinstance(decoded.get("gaussian"), list) else decoded.get("gaussian")
        record = save_decoded_slat_assets(
            {"mesh": mesh, "gaussian": gaussian},
            out_dir,
            mesh_name="complete.glb",
            gaussian_name="complete.ply",
        )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "object_id": args.object_id,
        "angle_idx": int(args.angle_idx),
        "view_indices": view_indices,
        "feature_mode": "existing_tokens_x_norm_patchtokens",
        "surface_voxels": int(coords.shape[0]),
        "slat_rows": int(slat.feats.shape[0]),
        "slat_feat_range": [float(slat.feats.min().item()), float(slat.feats.max().item())],
        "assets": record,
        "latent": str((out_dir / "latent.npz").resolve()),
        "token_path": str(token_path.resolve()),
        "camera_path": str(camera_path.resolve()),
        "encoder_config": str(args.encoder_config.resolve()),
        "encoder_ckpt": str(args.encoder_ckpt.resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--angle-idx", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--view-indices", type=int, nargs="+")
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--encoder-ckpt", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
