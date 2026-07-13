#!/usr/bin/env python3
"""SAM 3D Objects SLat stage, run for body + each part.

This glue script mirrors `generate_texture/texture/pipeline.py`
(`TexturePipeline.__init__` + `__call__`) but instead of decoding a single
object voxel, it decodes the body context plus the per-part voxels produced by
the upstream part-flow stage and writes one (glb + ply) pair per component.

Pipeline shape (identical to TexturePipeline):
  1. Set os.environ['LIDRA_SKIP_INIT']='true' BEFORE importing sam3d_objects.
  2. OmegaConf.load(config); rendering_engine='pytorch3d'; compile_model=False;
     workspace_dir=config_path.parent; device=device; instantiate(config).
     We load the mesh decoder (load_mesh_decoder=True) and skip the heavy gs_4
     decoder (load_gs4_decoder=False) by nulling its config/ckpt paths.
  3. Build body coords from <run>/voxel.npz minus union(part_NN_voxel.npz),
     then for body and each <parts-dir>/part_NN_voxel.npz:
       coords -> (N,4) int32 with a zero batch column, on device
       rgba   = image[...,:3] + (mask>0)*255 alpha
       slat_input = pipe.preprocess_image(rgba, pipe.slat_preprocessor)
       torch.manual_seed(seed)
       slat = pipe.sample_slat(slat_input, coords, inference_steps=None,
                               use_distillation=False)
       outputs = pipe.decode_slat(slat, formats)
       save gaussian (.ply) + mesh (.glb) via the shared TRELLIS writer.

Failure-exposing by design: a malformed part voxel file (wrong shape / dtype /
out-of-range coords / empty) raises immediately. No silent fallbacks.

NOTE: this box cannot run sam3d (cu118 vs sam3d's cu121); the script is written
to match the confirmed sam3d signatures in
`sam3d_objects/pipeline/inference_pipeline.py` (sample_slat / decode_slat /
preprocess_image) and is intended to run in the sam3d env only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Resolve sibling stage packages and the TRELLIS-arts shared writer up front so
# that the heavy sam3d import (deferred to run()) is the only torch-touching
# import. `--help` must stay fast and must NOT pull in torch / sam3d_objects.
_THIS_DIR = Path(__file__).resolve().parent
# .../submodules/sam3d-stage/infer_glue/slat_stage.py
#                  ^_SAM3D_STAGE_ROOT
_SAM3D_STAGE_ROOT = _THIS_DIR.parent
# arts-reconstruction repo root: sam3d-stage lives under submodules/.
# .../arts-reconstruction/submodules/sam3d-stage
_ARTS_REPO_ROOT = _SAM3D_STAGE_ROOT.parent.parent
_TRELLIS_ARTS_ROOT = _ARTS_REPO_ROOT / "TRELLIS-arts"

DEFAULT_OFFLINE_MOGE_MODEL = Path(
    "/robot/data-lab/jzh/art-gen/weights/hub/models--Ruicheng--moge-vitl/"
    "snapshots/979e84da9415762c30e6c0cf8dc0962896c793df/model.pt"
)

# part_NN_voxel.npz, NN = zero-padded part index (e.g. part_00_voxel.npz).
_PART_VOXEL_RE = re.compile(r"^part_(\d+)_voxel\.npz$")
_VOXEL_RESOLUTION = 64
_BODY_STEM = "body"


def _iter_moge_model_candidates(config_path: Path):
    """Yield local MoGe checkpoint candidates near the pipeline weights."""
    weights_dir = Path(config_path).parent
    seen: set[Path] = set()

    def emit(path: Path):
        path = Path(path)
        if path not in seen:
            seen.add(path)
            yield path

    for path in (
        weights_dir / "moge-vitl" / "model.pt",
        weights_dir / "moge" / "model.pt",
        DEFAULT_OFFLINE_MOGE_MODEL,
    ):
        yield from emit(path)

    for hub_root in (
        weights_dir / "hub",
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        if hub_root is None:
            continue
        snapshot_root = hub_root / "models--Ruicheng--moge-vitl" / "snapshots"
        if snapshot_root.is_dir():
            for path in sorted(snapshot_root.glob("*/model.pt")):
                yield from emit(path)


def _resolve_moge_model_path(config_path: Path) -> Path | None:
    """Return a local MoGe checkpoint path, honoring SAM3D_MOGE_MODEL_PATH."""
    env_path = os.environ.get("SAM3D_MOGE_MODEL_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SAM3D_MOGE_MODEL_PATH points to a missing file: {path}"
            )
        return path

    for path in _iter_moge_model_candidates(config_path):
        if path.is_file():
            return path
    return None


def _configure_offline_moge(config, config_path: Path) -> None:
    """Prevent MoGe from hitting Hugging Face in offline evaluation runs."""
    depth_model = config.get("depth_model")
    if depth_model is None:
        return
    model = depth_model.get("model") if hasattr(depth_model, "get") else None
    if model is None or not hasattr(model, "get"):
        return

    pretrained = model.get("pretrained_model_name_or_path")
    if not pretrained or Path(str(pretrained)).exists():
        return

    moge_model_path = _resolve_moge_model_path(config_path)
    if moge_model_path is not None:
        model.pretrained_model_name_or_path = str(moge_model_path)
        print(f"[INFO] using offline MoGe checkpoint: {moge_model_path}")
        return

    model.local_files_only = True
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print(
        "[WARN] local MoGe checkpoint not found; forcing Hugging Face "
        "local_files_only for depth_model.model"
    )


def _load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as uint8 [H,W,3] (mirrors texture/cli.py)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _load_mask(path: Path) -> np.ndarray:
    """Load a binary mask as bool [H,W] (mirrors texture/cli.py)."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # RGBA -> alpha; otherwise collapse channels with max.
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    if arr.dtype == bool:
        return arr
    return arr > 127


def _merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB + binary mask -> uint8 RGBA [H,W,4] (mirrors the stage pipelines)."""
    mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    return np.concatenate([image[..., :3], mask_u8[..., None]], axis=-1)


def _discover_part_voxels(parts_dir: Path) -> list[tuple[int, Path]]:
    """Return [(part_index, path), ...] for part_NN_voxel.npz files, sorted by index.

    Fails loudly if the directory holds no matching files.
    """
    parts_dir = Path(parts_dir)
    if not parts_dir.is_dir():
        raise FileNotFoundError(f"--parts-dir not a directory: {parts_dir}")

    found: list[tuple[int, Path]] = []
    for child in sorted(parts_dir.iterdir()):
        m = _PART_VOXEL_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))

    if not found:
        raise FileNotFoundError(
            f"no part_NN_voxel.npz files found in {parts_dir} "
            f"(expected files like part_00_voxel.npz from the part-flow stage)"
        )
    found.sort(key=lambda t: t[0])
    return found


def _validate_coords_array(source: Path | str, coords: np.ndarray) -> np.ndarray:
    """Validate coords:int[N,3] in [0,63] and return int32 numpy coords."""
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"{source}: coords must have shape (N, 3); got {coords.shape}"
        )
    if coords.shape[0] == 0:
        raise ValueError(f"{source}: coords is empty (N=0)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError(
            f"{source}: coords dtype must be integer; got {coords.dtype}"
        )

    lo, hi = int(coords.min()), int(coords.max())
    if lo < 0 or hi >= _VOXEL_RESOLUTION:
        raise ValueError(
            f"{source}: coords out of [0, {_VOXEL_RESOLUTION - 1}] range: "
            f"min={lo}, max={hi}"
        )

    return np.ascontiguousarray(coords.astype(np.int32, copy=False))


def _load_voxel_coords_np(voxel_path: Path) -> np.ndarray:
    """Read coords:int32[N,3] in [0,63] from a voxel npz file."""
    with np.load(voxel_path) as data:
        if "coords" not in data:
            raise ValueError(
                f"{voxel_path}: missing 'coords' array "
                f"(keys present: {list(data.keys())})"
            )
        coords = data["coords"]
    return _validate_coords_array(voxel_path, coords)


def _coords_np_to_torch(coords: np.ndarray):
    """Convert validated numpy coords [N,3] to torch int32 [N,4] with batch=0.

    The returned tensor has a zero batch column prepended: [batch=0, x, y, z].
    torch is imported here (not module-level) so that --help stays cheap.
    """
    import torch

    coords = _validate_coords_array("coords", coords)
    n = coords.shape[0]
    batch_col = np.zeros((n, 1), dtype=np.int64)
    full = np.concatenate([batch_col, coords.astype(np.int64)], axis=1).astype(np.int32)
    return torch.from_numpy(full)


def _load_part_coords(voxel_path: Path):
    """Read a voxel npz and return a torch (N,4) int32 tensor."""
    return _coords_np_to_torch(_load_voxel_coords_np(voxel_path))


def _coord_keys(coords: np.ndarray) -> np.ndarray:
    """Encode 0..63 xyz coords into stable integer keys for set subtraction."""
    coords = _validate_coords_array("coords", coords).astype(np.int64, copy=False)
    return (
        coords[:, 0] * (_VOXEL_RESOLUTION * _VOXEL_RESOLUTION)
        + coords[:, 1] * _VOXEL_RESOLUTION
        + coords[:, 2]
    )


def _build_body_coords(parts_dir: Path, part_voxels: list[tuple[int, Path]]) -> tuple[Path, np.ndarray]:
    """Return body coords = whole SS voxel minus all decoded part voxels."""
    whole_voxel_path = Path(parts_dir).parent / "voxel.npz"
    if not whole_voxel_path.is_file():
        raise FileNotFoundError(
            f"body decode requested but whole-object voxel is missing: {whole_voxel_path} "
            "(pass --no-body to decode only part_NN_voxel.npz files)"
        )

    whole = _load_voxel_coords_np(whole_voxel_path)
    part_chunks = [_load_voxel_coords_np(path) for _, path in part_voxels]
    if part_chunks:
        part_coords = np.concatenate(part_chunks, axis=0)
        part_keys = np.unique(_coord_keys(part_coords))
        keep = ~np.isin(_coord_keys(whole), part_keys)
        body = whole[keep]
    else:
        body = whole

    if body.shape[0] == 0:
        raise ValueError(
            f"{whole_voxel_path}: body coords empty after subtracting "
            f"{len(part_voxels)} part voxel file(s)"
        )
    return whole_voxel_path, np.ascontiguousarray(body.astype(np.int32, copy=False))


def _resolve_save_fn():
    """Return a callable matching `save_decoded_slat_assets`'s contract.

    Prefer the shared TRELLIS writer
    (`trellis.utils.arts.slat_asset_writer.save_decoded_slat_assets`) which has
    the failure-exposing checks (gaussian must expose save_ply; mesh must have
    vertices/faces and success != False). If TRELLIS-arts isn't importable in
    this environment, fall back to an inline writer with the same checks.
    """
    if str(_TRELLIS_ARTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_TRELLIS_ARTS_ROOT))
    try:
        from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets

        print(f"[INFO] using shared writer: {_TRELLIS_ARTS_ROOT}/trellis/utils/arts/slat_asset_writer.py")
        return save_decoded_slat_assets
    except Exception as exc:  # ImportError or transitive import failure
        print(f"[INFO] shared TRELLIS writer unavailable ({exc!r}); using inline writer")
        return _inline_save_decoded_slat_assets


def _inline_save_decoded_slat_assets(
    decoded: dict,
    asset_dir: Path,
    *,
    mesh_name: str = "mesh.glb",
    gaussian_name: str = "gaussians.ply",
) -> dict[str, str]:
    """Inline fallback mirroring save_decoded_slat_assets (same failure checks)."""
    import torch
    import trimesh

    def to_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def vertices_y_up(value):
        vertices = np.asarray(to_numpy(value), dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"decoded mesh vertices must have shape (N, 3); got {vertices.shape}"
            )
        return vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

    def vertex_colors(mesh_obj, vertex_count: int):
        attrs = getattr(mesh_obj, "vertex_attrs", None)
        if attrs is None:
            return None
        colors = to_numpy(attrs)
        if colors.ndim != 2 or colors.shape[0] != vertex_count or colors.shape[1] < 3:
            raise ValueError(
                "decoded mesh vertex_attrs must have shape (num_vertices, >=3); "
                f"got {colors.shape}, vertices={vertex_count}"
            )
        colors = np.asarray(colors[:, :3], dtype=np.float32)
        if colors.size == 0:
            return None
        if colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        return np.concatenate([colors, alpha], axis=1)

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, str] = {}

    gaussian = decoded.get("gaussian")
    if gaussian is not None:
        save_ply = getattr(gaussian, "save_ply", None)
        if not callable(save_ply):
            raise TypeError(
                f"gaussian does not expose save_ply(): {type(gaussian).__name__}"
            )
        save_ply(str(asset_dir / gaussian_name))
        record["gaussian"] = gaussian_name

    mesh = decoded.get("mesh")
    if mesh is not None:
        if not getattr(mesh, "success", True):
            raise ValueError(f"decoded mesh success=False: {asset_dir}")
        vertices = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if vertices is None or faces is None:
            raise TypeError(
                f"decoded mesh missing vertices/faces: {type(mesh).__name__}"
            )
        vertices = vertices_y_up(vertices)
        faces = to_numpy(faces)
        tri_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices), faces=np.asarray(faces), process=False
        )
        colors = vertex_colors(mesh, len(tri_mesh.vertices))
        if colors is not None:
            tri_mesh.visual.vertex_colors = colors
        tri_mesh.export(str(asset_dir / mesh_name))
        record["mesh"] = mesh_name

    return record


def _build_pipeline(config_path: Path, device: str, *, load_mesh_decoder: bool):
    """Construct the sam3d InferencePipelinePointMap, mirroring TexturePipeline.__init__.

    Returns the instantiated pipeline object. The heavy sam3d / torch imports
    happen here, NOT at module load, so --help stays fast.
    """
    config_path = Path(config_path)

    # Must be set BEFORE importing sam3d_objects (mirrors the stage pipelines and
    # notebook/inference.py:6). The public sam3d_objects release lacks a
    # `sam3d_objects.init` module that the top-level package tries to import;
    # this env flag short-circuits that path.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    # spconv 2.2.x MaskImplicitGemm can raise SIGFPE on the part-level sparse
    # coords used here. Native is slower but stable on the evaluation H20 nodes.
    os.environ.setdefault("SPCONV_ALGO", "native")

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    import sam3d_objects  # noqa: F401 -- triggers package init side-effects
    from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: F401
        InferencePipelinePointMap,
    )

    config = OmegaConf.load(str(config_path))
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    config.device = device

    # We never decode gaussian_4 here, so avoid loading slat_decoder_gs_4.ckpt
    # (~163MB) by nulling its config/ckpt paths (mirrors TexturePipeline with
    # load_gs4_decoder=False).
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    if not load_mesh_decoder:
        config.slat_decoder_mesh_config_path = None
        config.slat_decoder_mesh_ckpt_path = None
    _configure_offline_moge(config, config_path)

    pipe = instantiate(config)

    if load_mesh_decoder and "slat_decoder_mesh" not in getattr(pipe, "models", {}):
        # decode_slat reads self.models["slat_decoder_mesh"]; surface the gap now
        # instead of mid-loop with a cryptic KeyError.
        raise RuntimeError(
            "mesh decoder requested but 'slat_decoder_mesh' not present in the "
            "instantiated pipeline's models; check pipeline.yaml"
        )
    return pipe


def run(args: argparse.Namespace) -> str:
    """Run the SAM3D SLat stage. Returns a 3-line summary string."""
    import torch

    formats = tuple(args.formats)
    want_mesh = "mesh" in formats
    want_gaussian = "gaussian" in formats

    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image HxW {image.shape[:2]}"
        )
    rgba = _merge_mask_to_rgba(image, mask)

    seed_base = int(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    components: list[dict[str, object]] = []
    if args.whole_voxel is not None:
        whole_coords_np = _load_voxel_coords_np(args.whole_voxel)
        print(
            f"[INFO] overall ({args.whole_stem}): {whole_coords_np.shape[0]} voxels "
            f"from {args.whole_voxel}"
        )
        components.append(
            {
                "label": "overall",
                "seed": seed_base,
                "stem": args.whole_stem,
                "path": args.whole_voxel,
                "coords_np": whole_coords_np,
            }
        )
    else:
        part_voxels = _discover_part_voxels(args.parts_dir)
        print(f"[INFO] discovered {len(part_voxels)} part voxel file(s) in {args.parts_dir}")

        if args.include_body:
            body_source, body_coords_np = _build_body_coords(args.parts_dir, part_voxels)
            body_voxel_path = out_dir / f"{_BODY_STEM}_voxel.npz"
            np.savez_compressed(body_voxel_path, coords=body_coords_np)
            print(
                f"[INFO] body ({_BODY_STEM}): {body_coords_np.shape[0]} voxels from "
                f"{body_source.name} minus {len(part_voxels)} part voxel file(s); "
                f"wrote {body_voxel_path.name}"
            )
            components.append(
                {
                    "label": "body",
                    "seed": seed_base,
                    "stem": _BODY_STEM,
                    "path": body_voxel_path,
                    "coords_np": body_coords_np,
                }
            )

        for part_index, voxel_path in part_voxels:
            stem = voxel_path.name[: -len("_voxel.npz")]  # e.g. "part_00"
            components.append(
                {
                    "label": f"part {part_index}",
                    "seed": seed_base + part_index + 1,
                    "stem": stem,
                    "path": voxel_path,
                    "coords_np": None,
                }
            )

    save_fn = _resolve_save_fn()

    device = torch.device(args.device)
    pipe = _build_pipeline(args.config, args.device, load_mesh_decoder=want_mesh)

    written_components = 0
    written_files: list[str] = []
    processed_stems: list[str] = []
    try:
        with device:
            # preprocess_image is independent of the per-part coords, but we keep
            # it inside the loop body's reach; computing it once is fine since the
            # rgba/image+mask are shared across all parts of this object.
            slat_input_dict = pipe.preprocess_image(rgba, pipe.slat_preprocessor)

            for component in components:
                stem = str(component["stem"])
                label = str(component["label"])
                voxel_path = Path(component["path"])
                seed = int(component["seed"])
                print(f"[INFO] {label} ({stem}): loading {voxel_path.name}")

                coords_np = component.get("coords_np")
                coords = (
                    _coords_np_to_torch(coords_np)
                    if coords_np is not None
                    else _load_part_coords(voxel_path)
                )
                coords_dev = coords.to(device)
                print(
                    f"[INFO] {label}: {coords.shape[0]} voxels, "
                    f"seed={seed}"
                )

                # Offset the seed per component so each body/part gets an
                # independent RNG draw (mirrors TexturePipeline's decoupling).
                torch.manual_seed(seed)
                slat = pipe.sample_slat(
                    slat_input_dict,
                    coords_dev,
                    inference_steps=None,
                    use_distillation=False,
                )

                outputs = pipe.decode_slat(slat, list(formats))
                # sam3d returns a batch list per format; bs=1 here -> take [0].
                gs = outputs["gaussian"][0] if "gaussian" in outputs else None
                mesh = outputs["mesh"][0] if "mesh" in outputs else None

                record = save_fn(
                    {"gaussian": gs, "mesh": mesh},
                    out_dir,
                    mesh_name=f"{stem}.glb",
                    gaussian_name=f"{stem}.ply",
                )
                for fmt, fname in record.items():
                    written_files.append(fname)
                    print(f"[INFO] {label}: wrote {fmt} -> {out_dir / fname}")
                processed_stems.append(stem)
                written_components += 1
    finally:
        # Mirror the stage pipelines' unload(): free the (large) pipeline + VRAM.
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fmt_desc = ", ".join(
        f for f, want in (("gaussian", want_gaussian), ("mesh", want_mesh)) if want
    )
    return (
        f"SLat stage: processed {written_components}/{len(components)} component(s).\n"
        f"Formats: {fmt_desc}; wrote {len(written_files)} file(s) to {out_dir}.\n"
        f"Component stems: {', '.join(processed_stems)}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slat_stage",
        description=(
            "SAM 3D Objects SLat stage, run BODY + PER PART: build body coords "
            "from <run>/voxel.npz minus part_NN_voxel.npz, then sample/decode "
            "each component and write <out>/body.glb plus <out>/part_NN.glb "
            "(and .ply when gaussian is requested)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        help=(
            "Directory containing part_NN_voxel.npz (each: coords int32[N,3] in [0,63]). "
            "Required unless --whole-voxel is provided."
        ),
    )
    parser.add_argument(
        "--whole-voxel",
        type=Path,
        default=None,
        help=(
            "Decode exactly this whole-object voxel.npz as one SAM3D component, "
            "using --image/--mask for SLat conditioning. Skips per-part/body logic."
        ),
    )
    parser.add_argument(
        "--whole-stem",
        type=str,
        default="overall",
        help="Output stem used with --whole-voxel, e.g. overall -> overall.glb / overall.ply.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="RGB image shared across all parts (same one used upstream).",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Binary mask shared across all parts (same one used upstream).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to sam3d pipeline.yaml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-part part_NN.glb / part_NN.ply (created).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["gaussian", "mesh"],
        default=["gaussian", "mesh"],
        help="Decode formats to write per part.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed base; body seed = seed, per-part seed = seed + part_index + 1.",
    )
    parser.add_argument(
        "--include-body",
        dest="include_body",
        action="store_true",
        default=True,
        help="Decode body = whole voxel minus all part voxels before decoding parts.",
    )
    parser.add_argument(
        "--no-body",
        dest="include_body",
        action="store_false",
        help="Decode only part_NN_voxel.npz files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for the sam3d pipeline.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate inputs up front (failure-exposing, before the heavy import).
    if args.whole_voxel is None:
        if args.parts_dir is None:
            raise ValueError("--parts-dir is required unless --whole-voxel is provided")
        if not args.parts_dir.is_dir():
            raise FileNotFoundError(f"--parts-dir not a directory: {args.parts_dir}")
    else:
        if not args.whole_voxel.is_file():
            raise FileNotFoundError(f"--whole-voxel not found: {args.whole_voxel}")
        if not args.whole_stem:
            raise ValueError("--whole-stem cannot be empty")
    if not args.image.exists():
        raise FileNotFoundError(f"--image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"--mask not found: {args.mask}")
    if not args.config.exists():
        raise FileNotFoundError(f"--config not found: {args.config}")

    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
