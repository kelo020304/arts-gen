#!/usr/bin/env python
"""SAM 3D Objects per-part SS-latent decode glue script.

Decodes the per-part SS latents emitted by the Part SS-Latent Flow stage
(``part_flow_stage``) back into per-part voxel coords, using SAM 3D Objects'
``ss_decoder`` (the SAME decoder the upstream pipeline uses to turn an SS latent
into a sparse-structure occupancy grid).

Why this exists
---------------
``part_flow_stage`` writes one SS latent per part as
``parts/part_NN_latent.npy`` of shape ``(8, 16, 16, 16)`` (plus a sidecar
``parts/part_NN_meta.json`` carrying ``part_index`` / ``target_part_name``).
This script feeds each latent through ``pipe.models["ss_decoder"]`` and persists
the resulting per-part voxel grid as ``parts/part_NN_voxel.npz`` following the
canonical voxel.npz contract (see ``inference_pipeline.voxel_io`` / Phase 2A and
``inference_pipeline/part_flow_stage.save_part_voxels``).

The decode mirrors ``InferencePipelinePointMap.sample_sparse_structure``:
    ss = ss_decoder(zt)                                   # zt = [1,8,16,16,16]
    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()  # [N,4] = (b,x,y,z)
    xyz = coords[:, 1:].cpu().numpy().astype('int32')      # [N,3]

NOTE: the part-flow latent is ALREADY laid out as ``(8, 16, 16, 16)``, so it is
fed to ``ss_decoder`` directly as ``[1, 8, 16, 16, 16]``. We do NOT apply the
``[bs, 4096, 8] -> permute -> view`` reshape that the upstream code uses on the
SS *generator's* raw output (that reshape only applies to the generator output,
not to an already-decoded z layout).

Environment
-----------
This MUST run inside the SAM 3D Objects venv (CUDA 12.1 / cu121). It imports
``sam3d_objects`` (the heavy pipeline) which is NOT installable in the
arts-reconstruction ``arts-gen`` env (cu118). ``--help`` and ``py_compile`` stay
fast / importable in arts-gen because every heavy import is deferred into a
function body.

Usage
-----
    python decode_ss_glue.py --parts-dir <dir> --config <pipeline.yaml> \
        [--device cuda:0]

``<dir>`` is the ``parts/`` directory (or a run dir containing it -- we accept
either) holding ``part_NN_latent.npy`` + ``part_NN_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# part_(\d+)_latent.npy -- the per-part latent emitted by part_flow_stage.
_LATENT_RE = re.compile(r"^part_(\d+)_latent\.npy$")


def _resolve_parts_dir(parts_dir: Path) -> Path:
    """Accept either the parts/ dir itself or a run dir containing parts/."""
    parts_dir = Path(parts_dir)
    if not parts_dir.is_dir():
        raise NotADirectoryError(f"--parts-dir not a directory: {parts_dir}")
    # If a run dir was passed, descend into its parts/ subdir.
    if (parts_dir / "parts").is_dir() and parts_dir.name != "parts":
        return parts_dir / "parts"
    return parts_dir


def _discover_parts(parts_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return [(index, latent_path, meta_path), ...] sorted by index.

    Each latent must have a sibling ``part_NN_meta.json`` (written by
    part_flow_stage). Missing meta is a hard error -- no silent fallback.
    """
    found: list[tuple[int, Path, Path]] = []
    for latent_path in parts_dir.iterdir():
        m = _LATENT_RE.match(latent_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        meta_path = parts_dir / f"part_{m.group(1)}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"latent {latent_path.name} has no sidecar meta: {meta_path.name}"
            )
        found.append((idx, latent_path, meta_path))
    if not found:
        raise FileNotFoundError(
            f"no part_NN_latent.npy files found in {parts_dir} "
            f"(expected names like part_00_latent.npy)"
        )
    found.sort(key=lambda t: t[0])
    return found


def _load_meta(meta_path: Path) -> tuple[int, str]:
    """Read {part_index:int, target_part_name:str} from the sidecar meta.json."""
    with meta_path.open() as f:
        meta = json.load(f)
    if "part_index" not in meta or "target_part_name" not in meta:
        raise KeyError(
            f"meta {meta_path} missing part_index/target_part_name "
            f"(got keys {sorted(meta)})"
        )
    return int(meta["part_index"]), str(meta["target_part_name"])


def _build_pipeline(config_path: Path, device: str):
    """Construct the SAM 3D Objects pipeline exactly like ss_stage._build_pipeline.

    Heavy imports happen INSIDE this function so ``--help`` stays fast and does
    not pull in torch / sam3d_objects.
    """
    # The public sam3d_objects release lacks a `sam3d_objects.init` module that
    # the top-level package tries to import; this env flag short-circuits it.
    # MUST be set BEFORE importing sam3d_objects.
    import os

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    from omegaconf import OmegaConf
    import hydra

    import sam3d_objects  # noqa: F401 -- triggers package init side-effects

    config_path = Path(config_path)
    config = OmegaConf.load(str(config_path))
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    config.device = device

    # instantiate(config) builds every sub-model declared in pipeline.yaml,
    # same as the SS wrapper. We only use models["ss_decoder"].
    pipe = hydra.utils.instantiate(config)
    return pipe


def _build_ss_decoder(config_path: Path, device: str, ss_decoder_ckpt: Path | None = None):
    """Build ONLY the SS decoder, instead of instantiating the whole pipeline.

    The full ``InferencePipelinePointMap`` also builds the SS/SLat generators and
    the GS/mesh SLat decoders, which import kaolin / pytorch3d / gsplat. For the
    SS decode we only need ``ss_decoder`` — a dense-conv SS-VAE decoder
    (``SparseStructureDecoderTdfyWrapper``: torch only, no kaolin/pytorch3d/gsplat).
    Building it standalone lets the SS decode run on a venv WITHOUT those heavy
    compiled deps. Mirrors ``InferencePipeline._load_model``: read the model's own
    yaml, ``instantiate(config)``, then load the ckpt with ``strict=False``.
    Paths come from pipeline.yaml's ``ss_decoder_config_path`` /
    ``ss_decoder_ckpt_path``, resolved against the pipeline.yaml dir (workspace).
    ``ss_decoder_ckpt`` can override the ckpt path so the eval platform's
    selected decoder is the one actually used by this subprocess.
    """
    import os

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    # Force the no-extra-dep attention backend: the SS decoder's middle blocks use
    # the tdfy attention module, whose default is already "sdpa" (pure torch); pin
    # it so an inherited ATTN_BACKEND=flash_attn/xformers can't trigger importing a
    # compiled attention kernel this minimal venv doesn't have.
    os.environ["ATTN_BACKEND"] = "sdpa"

    import torch
    from omegaconf import OmegaConf
    import hydra
    import sam3d_objects  # noqa: F401 -- package init side-effects (init skipped)

    config_path = Path(config_path)
    workspace_dir = config_path.parent
    pipe_cfg = OmegaConf.load(str(config_path))
    ss_cfg_path = workspace_dir / str(pipe_cfg["ss_decoder_config_path"])
    ss_ckpt_path = (
        Path(ss_decoder_ckpt)
        if ss_decoder_ckpt is not None
        else workspace_dir / str(pipe_cfg["ss_decoder_ckpt_path"])
    )
    if not ss_cfg_path.is_file():
        raise FileNotFoundError(f"ss_decoder config not found: {ss_cfg_path}")
    if not ss_ckpt_path.is_file():
        raise FileNotFoundError(f"ss_decoder ckpt not found: {ss_ckpt_path}")

    ss_decoder = hydra.utils.instantiate(OmegaConf.load(str(ss_cfg_path)))
    if str(ss_ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(str(ss_ckpt_path))
    else:
        state_dict = torch.load(str(ss_ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = ss_decoder.load_state_dict(state_dict, strict=False)
    print(f"[INFO] ss_decoder loaded (strict=False): "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return ss_decoder.to(torch.device(device)).eval()


def _decode_one(ss_decoder, latent_path: Path, device) -> np.ndarray:
    """Decode a single (8,16,16,16) SS latent into [N,3] int32 voxel coords.

    Mirrors InferencePipelinePointMap.sample_sparse_structure's decode step
    (threshold > 0, index [0,2,3,4]) but feeds the latent DIRECTLY -- the
    part-flow latent is already [8,16,16,16], no permute/view reshape.
    """
    import torch

    z = np.load(latent_path)
    if z.shape != (8, 16, 16, 16):
        raise ValueError(
            f"{latent_path.name}: expected SS latent shape (8,16,16,16), "
            f"got {tuple(z.shape)}"
        )
    zt = torch.from_numpy(z).float().to(device).unsqueeze(0)  # [1,8,16,16,16]
    with torch.no_grad():
        ss = ss_decoder(zt)
    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()  # [N,4]=(b,x,y,z)
    xyz = coords[:, 1:].detach().cpu().numpy().astype("int32")  # [N,3]
    return xyz


def _save_voxel(
    out_path: Path,
    xyz: np.ndarray,
    *,
    part_index: int,
    target_part_name: str,
    resolution: int = 64,
) -> None:
    """Write part_NN_voxel.npz per the canonical voxel.npz contract.

    Matches inference_pipeline.voxel_io / Phase 2A and
    inference_pipeline/part_flow_stage.save_part_voxels.
    """
    c = np.asarray(xyz).astype(np.int32).reshape(-1, 3)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= int(resolution)):
        raise ValueError(
            f"voxel coords 越界 [0,{resolution}): "
            f"min={int(c.min())} max={int(c.max())} ({out_path.name})"
        )
    np.savez_compressed(
        out_path,
        coords=c,
        resolution=np.int32(resolution),
        coord_frame="canonical_grid",
        source="pred",
        part_index=np.int32(part_index),
        target_part_name=str(target_part_name),
    )


def decode_parts(
    *,
    parts_dir: Path,
    config_path: Path,
    device: str,
    ss_decoder_ckpt: Path | None = None,
) -> dict:
    """Decode every per-part SS latent in ``parts_dir`` to part_NN_voxel.npz.

    Returns a small summary dict for logging.
    """
    import torch

    parts_dir = _resolve_parts_dir(parts_dir)
    parts = _discover_parts(parts_dir)
    print(f"[INFO] discovered {len(parts)} part latents in {parts_dir}")

    print(
        f"[INFO] building SS decoder from config={config_path} on device={device} "
        f"ckpt_override={ss_decoder_ckpt or '<pipeline.yaml>'}"
    )
    ss_decoder = _build_ss_decoder(config_path, device, ss_decoder_ckpt=ss_decoder_ckpt)
    device_t = torch.device(device)

    written: list[str] = []
    total_voxels = 0
    try:
        for idx, latent_path, meta_path in parts:
            part_index, target_part_name = _load_meta(meta_path)
            xyz = _decode_one(ss_decoder, latent_path, device_t)
            n_vox = int(xyz.shape[0])
            if n_vox == 0:
                raise ValueError(
                    f"ss_decoder produced 0 voxels for {latent_path.name} "
                    f"(part_index={part_index}, name='{target_part_name}'); "
                    f"decode failed"
                )
            out_path = parts_dir / f"part_{idx:02d}_voxel.npz"
            _save_voxel(
                out_path,
                xyz,
                part_index=part_index,
                target_part_name=target_part_name,
            )
            total_voxels += n_vox
            written.append(out_path.name)
            print(
                f"[INFO] part {idx:02d} (part_index={part_index}, "
                f"name='{target_part_name}'): {n_vox} voxels -> {out_path.name}"
            )
    finally:
        # Free VRAM like the wrapper's unload().
        del ss_decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "num_parts": len(written),
        "total_voxels": total_voxels,
        "parts_dir": str(parts_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decode-ss-glue",
        description=(
            "Decode per-part SS latents (parts/part_NN_latent.npy) into per-part "
            "voxels (parts/part_NN_voxel.npz) using SAM 3D Objects' ss_decoder. "
            "Run inside the sam3d venv (cu121)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        required=True,
        dest="parts_dir",
        help="parts/ directory (or a run dir containing it) with part_NN_latent.npy + part_NN_meta.json.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to SAM 3D pipeline.yaml."
    )
    parser.add_argument(
        "--ss-decoder-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional SS decoder checkpoint override. When omitted, the path in "
            "pipeline.yaml is used."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    args = parser.parse_args()

    # Fail loudly on missing inputs (no silent fallback).
    if not args.parts_dir.exists():
        raise FileNotFoundError(f"--parts-dir not found: {args.parts_dir}")
    if not args.config.exists():
        raise FileNotFoundError(f"--config not found: {args.config}")
    if args.ss_decoder_ckpt is not None and not args.ss_decoder_ckpt.exists():
        raise FileNotFoundError(f"--ss-decoder-ckpt not found: {args.ss_decoder_ckpt}")

    summary = decode_parts(
        parts_dir=args.parts_dir,
        config_path=args.config,
        device=args.device,
        ss_decoder_ckpt=args.ss_decoder_ckpt,
    )
    print(
        f"[INFO] DONE: decoded {summary['num_parts']} parts, "
        f"{summary['total_voxels']} voxels total -> {summary['parts_dir']}"
    )


if __name__ == "__main__":
    main()
