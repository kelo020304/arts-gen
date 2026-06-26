#!/usr/bin/env python3
"""Vanilla TRELLIS Image -> 3D inference (fast asset reconstruction).

Wraps ``submodules/TRELLIS.1`` (the upstream microsoft/TRELLIS reference
implementation). Given 1..N RGB images of an object, it runs the standard
TRELLIS pipeline:

    image(s) -> SS Flow DiT  -> SS latent -> SS decoder  -> sparse coords
                                                         -> SLat Flow DiT -> SLat
                                                         -> {SLat Gaussian decoder, SLat Mesh decoder}

and writes any subset of:
    gaussians.ply   (Gaussian splat points)
    mesh.obj        (raw extracted mesh)
    mesh.glb        (glTF binary, mesh textured from Gaussian appearance — needs both)
    preview.mp4     (turntable rotation of gaussian | mesh side-by-side)

to ``--output_dir``.

Memory note (16GB GPU):
    Decoding gaussian + mesh in one ``pipeline.run()`` call OOMs because the
    FlexiCubes mesh extractor peaks ~4GB while the gaussian output stays
    resident. We split decoding into per-format passes with
    ``torch.cuda.empty_cache()`` between them so a 4090 / 16GB card can do
    both. ``radiance_field`` is opt-in (extra ~3GB) on a 24GB+ card.

Defaults assume the local mirror at ``pretrained/TRELLIS-image-large/``;
pass ``--model microsoft/TRELLIS-image-large`` to fall back to HuggingFace.

Usage::

    conda activate arts-gen
    python scripts/inference/quick_reconstruct.py \\
        --images path/to/img_front.png path/to/img_side.png \\
        --output_dir outputs/quick_recon
"""
from __future__ import annotations

# Sparse-conv backend toggles must be set before importing trellis.
# TORCH_HOME steers torch.hub at the vendored DINOv2 cache so the pipeline's
# ``image_cond_model`` (DINOv2) loads from ``pretrained/torch_hub/`` instead
# of fetching from the network.
import gc
import os
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("SPCONV_ALGO", "native")
# TRELLIS' sparse-attention module only accepts 'flash_attn' or 'xformers'.
# arts-gen ships xformers but not flash-attn, so default to xformers.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("TORCH_HOME", str(_PROJECT_ROOT / "pretrained" / "torch_hub"))
# Reduces fragmentation OOMs when running gaussian+mesh decoders sequentially.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRELLIS_DIR = PROJECT_ROOT / "submodules" / "TRELLIS.1"
sys.path.insert(0, str(TRELLIS_DIR))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_images(paths: Iterable[str]) -> List[Image.Image]:
    images = []
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"image not found: {p}")
        images.append(Image.open(p))
    return images


def _free_cuda() -> None:
    """Aggressively reclaim CUDA memory between decode passes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _save_gaussian(gs, out_path: Path) -> bool:
    if gs is None:
        return False
    save_ply = getattr(gs, "save_ply", None)
    if callable(save_ply):
        save_ply(str(out_path))
        return True
    print(f"[warn] gaussian has no save_ply() — skipping {out_path}")
    return False


def _save_mesh_obj(mesh, out_path: Path) -> bool:
    """Save raw extracted mesh as .obj (geometry only, no texture)."""
    if mesh is None or not getattr(mesh, "success", True):
        print(f"[warn] mesh extraction reported success=False — skipping {out_path}")
        return False
    verts = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if verts is None or faces is None:
        print(f"[warn] mesh has no .vertices/.faces — skipping {out_path}")
        return False
    import trimesh

    if torch.is_tensor(verts):
        verts = verts.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(out_path))
    return True


def _save_textured_glb(gaussian, mesh, out_path: Path,
                       simplify: float = 0.95, texture_size: int = 1024) -> bool:
    """Use TRELLIS' to_glb (Gaussian appearance baked onto mesh) to write .glb."""
    if gaussian is None or mesh is None or not getattr(mesh, "success", True):
        print("[warn] need both gaussian + (successful) mesh for GLB — skipping")
        return False
    try:
        from trellis.utils import postprocessing_utils
        glb = postprocessing_utils.to_glb(
            gaussian, mesh,
            simplify=simplify, texture_size=texture_size, verbose=False,
        )
        glb.export(str(out_path))
        return True
    except ImportError as exc:
        print(f"[skip glb] missing dep: {exc!r} (need nvdiffrast)")
        return False
    except Exception as exc:  # noqa: BLE001 — surface but don't kill the run
        print(f"[skip glb] {type(exc).__name__}: {exc}")
        return False


def _maybe_render_video(gaussian, mesh, out_path: Path) -> None:
    try:
        import numpy as np
        import imageio
        from trellis.utils import render_utils
    except ImportError as exc:
        print(f"[skip video] missing dep: {exc!r}")
        return

    frames_gs = None
    frames_mesh = None
    if gaussian is not None:
        try:
            frames_gs = render_utils.render_video(gaussian)["color"]
        except Exception as exc:
            print(f"[skip gaussian render] {type(exc).__name__}: {exc}")
    if mesh is not None and getattr(mesh, "success", True):
        try:
            frames_mesh = render_utils.render_video(mesh)["normal"]
        except Exception as exc:
            print(f"[skip mesh render] {type(exc).__name__}: {exc}")

    if frames_gs is not None and frames_mesh is not None:
        frames = [np.concatenate([fg, fm], axis=1) for fg, fm in zip(frames_gs, frames_mesh)]
    else:
        frames = frames_gs if frames_gs is not None else frames_mesh
    if frames:
        imageio.mimsave(str(out_path), frames, fps=30)
        print(f"[saved] {out_path}")


# --------------------------------------------------------------------------- #
# Pipeline (mirrors TrellisImageTo3DPipeline.run_multi_image but exposes slat)#
# --------------------------------------------------------------------------- #


def _offload_to_cpu(pipeline: TrellisImageTo3DPipeline, names: List[str]) -> None:
    """Move named submodels to CPU to free GPU memory for the decode phase."""
    for n in names:
        m = pipeline.models.get(n)
        if m is not None and hasattr(m, "to"):
            m.to("cpu")
    _free_cuda()


def _sample_slat(pipeline: TrellisImageTo3DPipeline,
                 images: List[Image.Image],
                 *,
                 seed: int,
                 ss_steps: int, ss_cfg: float,
                 slat_steps: int, slat_cfg: float,
                 mode: str = "stochastic"):
    """Run preprocessing + SS sampling + SLat sampling once. Returns ``slat``."""
    images = [pipeline.preprocess_image(img) for img in images]
    cond = pipeline.get_cond(images)
    cond["neg_cond"] = cond["neg_cond"][:1]
    torch.manual_seed(seed)
    ss_params = {"steps": ss_steps, "cfg_strength": ss_cfg}
    slat_params = {"steps": slat_steps, "cfg_strength": slat_cfg}

    if len(images) > 1:
        with pipeline.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode,
        ):
            coords = pipeline.sample_sparse_structure(cond, num_samples=1,
                                                      sampler_params=ss_params)
        with pipeline.inject_sampler_multi_image(
            "slat_sampler", len(images), slat_steps, mode=mode,
        ):
            slat = pipeline.sample_slat(cond, coords, sampler_params=slat_params)
    else:
        coords = pipeline.sample_sparse_structure(cond, num_samples=1,
                                                  sampler_params=ss_params)
        slat = pipeline.sample_slat(cond, coords, sampler_params=slat_params)
    return slat


def _decode_one(pipeline: TrellisImageTo3DPipeline, slat, fmt: str):
    """Call a single decoder with no-grad context. Returns the rep."""
    with torch.no_grad():
        if fmt == "gaussian":
            out = pipeline.models["slat_decoder_gs"](slat)
        elif fmt == "mesh":
            out = pipeline.models["slat_decoder_mesh"](slat)
        elif fmt == "radiance_field":
            out = pipeline.models["slat_decoder_rf"](slat)
        else:
            raise ValueError(f"unknown format: {fmt}")
    # decoder returns a list (batch dim 1)
    return out[0] if hasattr(out, "__len__") and len(out) > 0 else out


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Vanilla TRELLIS Image -> 3D inference (1..N input images).",
    )
    ap.add_argument("--images", required=True, nargs="+",
                    help="paths to one or more RGB(A) images of the same object")
    ap.add_argument("--output_dir", required=True,
                    help="directory to write gaussians.ply / mesh.obj / mesh.glb / preview.mp4")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "pretrained" / "TRELLIS-image-large"),
                    help="local model dir (must contain pipeline.json + ckpts/) "
                         "or HF repo id like 'microsoft/TRELLIS-image-large'")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ss_steps", type=int, default=12,
                    help="sparse-structure flow sampling steps (default 12)")
    ap.add_argument("--slat_steps", type=int, default=12,
                    help="SLat flow sampling steps")
    ap.add_argument("--ss_cfg", type=float, default=7.5)
    ap.add_argument("--slat_cfg", type=float, default=3.0)
    ap.add_argument("--formats", default="gaussian,mesh",
                    help="comma-separated subset of {mesh, gaussian, radiance_field}. "
                         "GLB needs both 'gaussian' and 'mesh'.")
    ap.add_argument("--simplify", type=float, default=0.95,
                    help="GLB face-removal ratio (default 0.95 = aggressive simplify)")
    ap.add_argument("--texture_size", type=int, default=1024,
                    help="GLB baked-texture resolution (default 1024)")
    ap.add_argument("--no_glb", action="store_true",
                    help="skip the textured .glb export (still writes .ply / .obj)")
    ap.add_argument("--no_video", action="store_true",
                    help="skip turntable preview MP4")
    args = ap.parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    invalid = [f for f in formats if f not in {"mesh", "gaussian", "radiance_field"}]
    if invalid:
        ap.error(f"--formats: unsupported {invalid}; choose from mesh/gaussian/radiance_field")

    images = _load_images(args.images)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] pipeline from {args.model}")
    t0 = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print(f"[load] done in {time.time() - t0:.1f}s")

    print(f"[sample] {len(images)} image(s)  ss_steps={args.ss_steps}  slat_steps={args.slat_steps}")
    t0 = time.time()
    slat = _sample_slat(
        pipeline, images,
        seed=args.seed,
        ss_steps=args.ss_steps, ss_cfg=args.ss_cfg,
        slat_steps=args.slat_steps, slat_cfg=args.slat_cfg,
    )
    print(f"[sample] done in {time.time() - t0:.1f}s")
    # Sampling models are no longer needed during the decode phase. Offload
    # them to CPU so the (memory-hungry) FlexiCubes mesh extractor and the
    # nvdiffrast GLB texture bake don't have to share VRAM with them. Frees
    # ~3-4GB on a TRELLIS-image-large setup.
    _offload_to_cpu(pipeline, [
        "sparse_structure_flow_model",
        "slat_flow_model",
        "image_cond_model",
        "sparse_structure_decoder",
    ])

    # Decode each requested format separately so the FlexiCubes mesh extractor
    # doesn't have to share peak memory with a resident gaussian splat tensor.
    # Also: only the active decoder lives on GPU; the others stay on CPU until
    # their turn (saves ~500MB-1GB when both gaussian and mesh are requested).
    decoder_names = {
        "gaussian": "slat_decoder_gs",
        "mesh": "slat_decoder_mesh",
        "radiance_field": "slat_decoder_rf",
    }
    other_decoders = [decoder_names[f] for f in {"gaussian", "mesh", "radiance_field"}
                      if f not in formats and decoder_names[f] in pipeline.models]
    _offload_to_cpu(pipeline, other_decoders)

    results: dict = {}
    for fmt in formats:
        print(f"[decode {fmt}]")
        # Bring this decoder back to GPU; push others off.
        active = decoder_names[fmt]
        for f, n in decoder_names.items():
            if n in pipeline.models:
                pipeline.models[n].to("cuda" if f == fmt else "cpu")
        _free_cuda()

        t0 = time.time()
        results[fmt] = _decode_one(pipeline, slat, fmt)
        print(f"[decode {fmt}] done in {time.time() - t0:.1f}s")
        _free_cuda()

    # Save geometry/appearance.
    gaussian = results.get("gaussian")
    mesh = results.get("mesh")

    if gaussian is not None:
        if _save_gaussian(gaussian, out_dir / "gaussians.ply"):
            print(f"[saved] {out_dir / 'gaussians.ply'}")

    if mesh is not None:
        if _save_mesh_obj(mesh, out_dir / "mesh.obj"):
            print(f"[saved] {out_dir / 'mesh.obj'}")

    if (not args.no_glb) and gaussian is not None and mesh is not None:
        print("[glb] baking gaussian appearance onto mesh -> mesh.glb")
        t0 = time.time()
        ok = _save_textured_glb(
            gaussian, mesh, out_dir / "mesh.glb",
            simplify=args.simplify, texture_size=args.texture_size,
        )
        if ok:
            print(f"[saved] {out_dir / 'mesh.glb'} ({time.time() - t0:.1f}s)")

    if not args.no_video:
        print("[video] rendering turntable preview")
        _free_cuda()
        _maybe_render_video(gaussian, mesh, out_dir / "preview.mp4")

    print(f"[done] -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
