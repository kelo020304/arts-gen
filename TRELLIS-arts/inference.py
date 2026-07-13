#!/usr/bin/env python3
"""Pure-function inference API for the arts pipeline.

No GT comparison, no metrics; pipeline/0*_*.py are thin orchestrators that
import the public functions defined here and persist outputs to disk.

Public API:
    run_ss_flow(images, ckpt_path, num_steps=20, cfg_strength=7.5)
                                                             -> Tensor [8,16,16,16] CPU
    run_ss_flow_from_tokens(tokens, ckpt_path, ...)          -> Tensor [8,16,16,16] CPU
    decode_ss(z_s, decoder_ckpt_path, threshold=0.0)        -> LongTensor [N,3] CPU
    run_part_ss_latent_flow(z_global, cond_tokens, ...)     -> Dict[str, Dict]
    run_slat_flow(images, coords, ckpt_path, num_steps=25)  -> SparseTensor
    run_slat_flow_from_tokens(cond_tokens, coords, ...)     -> SparseTensor
    decode_slat(slat, decoder_ckpt_path, formats=...)       -> Dict[str, Any]
    decode_slat_assets(slat, gs_ckpt=..., mesh_ckpt=...)    -> Dict[str, Any]

All ckpt loaders are wrapped with functools.lru_cache so that successive
invocations within the same process re-use the loaded modules at zero cost.

Sources (per D-26..D-30):
    - run_ss_flow / decode_ss     <- scripts/eval/stage2/infer.py:215-328
    - run_part_ss_latent_flow     <- part_ss_latent_flow RF
    - run_slat_flow               <- scripts/eval/stage4/infer.py:256-408
    - decode_slat                 <- scripts/train/stage4/render_utils.py:35-150
                                     (now trellis.utils.arts.slat_render_utils)
"""
from __future__ import annotations

import functools
import json
import math
import os
import sys
import types
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# PROJECT_ROOT (D-13: 1 level up; inference.py lives in TRELLIS-arts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRELLIS_PATH = os.path.join(PROJECT_ROOT, "TRELLIS-arts")
if TRELLIS_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_PATH)

# ---------------------------------------------------------------------------
# Minimal-deps trellis registration (mirrors train_arts.py).
# Avoids triggering trellis/__init__.py which eagerly pulls
# pipelines -> rembg -> torchvision and renderers -> nvdiffrast.
# decode_slat needs the renderer subpackage on demand, so we register a
# package shell here and let the actual heavy modules load lazily on first
# attribute access.
# ---------------------------------------------------------------------------
_pkg = types.ModuleType("trellis")
_pkg.__path__ = [os.path.join(TRELLIS_PATH, "trellis")]
_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _pkg)
# Stub heavy subpackages whose __init__ pulls big deps we don't want at import:
#   pipelines -> rembg -> torchvision (image-to-3d demo deps)
#   renderers -> nvdiffrast (compiled CUDA, optional)
# Their submodules can still be imported via __path__.
for _sp in ("models", "modules", "trainers", "utils", "datasets",
            "pipelines", "renderers"):
    _m = types.ModuleType(f"trellis.{_sp}")
    _m.__path__ = [os.path.join(TRELLIS_PATH, "trellis", _sp)]
    _m.__package__ = f"trellis.{_sp}"
    sys.modules.setdefault(f"trellis.{_sp}", _m)
# representations: NOT stubbed — its __init__.py is light (4 internal imports)
# and decode_slat needs `from ..representations import Gaussian, MeshExtractResult`.
# Stubbing it breaks SLat decoding (same fix as train_arts.py).

_LOCAL_TORCH_HOME = os.path.join(PROJECT_ROOT, "pretrained", "torch_hub")
os.environ.setdefault(
    "TORCH_HOME",
    _LOCAL_TORCH_HOME if os.path.isdir(_LOCAL_TORCH_HOME)
    else os.path.join(PROJECT_ROOT, "submodules", "TRELLIS.1"),
)
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded model defaults (mirror scripts/eval/stage{2,4}/infer.py)
# ---------------------------------------------------------------------------
_SS_FLOW_DEFAULT_ARGS = dict(
    resolution=16,
    in_channels=8,
    model_channels=1024,
    cond_channels=1024,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    num_head_channels=64,
    mlp_ratio=4,
    patch_size=1,
    pe_mode="ape",
    use_fp16=True,
    use_checkpoint=False,
    share_mod=False,
    qk_rms_norm=True,
    qk_rms_norm_cross=False,
)

_SLAT_FLOW_DEFAULT_ARGS = dict(
    resolution=64,
    in_channels=8,
    model_channels=1024,
    cond_channels=1024,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    num_head_channels=64,
    mlp_ratio=4,
    patch_size=2,
    num_io_res_blocks=2,
    io_block_channels=[128],
    pe_mode="ape",
    use_fp16=True,
    use_checkpoint=False,
    qk_rms_norm=True,
    qk_rms_norm_cross=False,
)

# SLat feature normalization (configs/arts/slat_flow_art/mv_4view.yaml).
_DEFAULT_SLAT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,   1.2039363384246826,
], dtype=torch.float32)

_DEFAULT_SLAT_STD = torch.tensor([
    2.377650737762451,  2.386378288269043,  2.124418020248413,
    2.1748552322387695, 2.663944721221924,  2.371192216873169,
    2.6217446327209473, 2.684523105621338,
], dtype=torch.float32)

# DINOv2-L/14-reg constants (scripts/data_process/encode_dinov2_mobility.py).
_DINOV2_RESOLUTION = 518
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_CFG_STRENGTH = 3.0
_SS_FLOW_CFG_STRENGTH = 7.5
_SS_FLOW_SEED = 20260610
_SS_FLOW_MODEL_CONFIG = "/robot/data-lab/jzh/art-gen/weights/ss_flow_img_dit_L_16l8_fp16.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve(path: str) -> str:
    """Return ``path`` made absolute (relative to PROJECT_ROOT if needed)."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _load_state_dict(ckpt_abs: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    ckpt_abs = _resolve_weight_pointer(ckpt_abs)
    if ckpt_abs.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(ckpt_abs)
    return torch.load(ckpt_abs, map_location=device, weights_only=True)


def _resolve_weight_pointer(path: str) -> str:
    """Resolve local text pointer files used to avoid duplicating big weights."""
    if not os.path.isfile(path) or os.path.getsize(path) > 4096:
        return path
    try:
        with open(path, "rb") as f:
            payload = f.read(4096)
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return path
    if not text or "\n" in text or "\r" in text:
        return path
    if not (text.endswith(".safetensors") or text.endswith(".pt") or text.endswith(".pth")):
        return path
    resolved = text if os.path.isabs(text) else os.path.normpath(os.path.join(os.path.dirname(path), text))
    return resolved if os.path.isfile(resolved) else path


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


@functools.lru_cache(maxsize=1)
def _load_dinov2() -> torch.nn.Module:
    """Load DINOv2-L/14-reg from vendored files; never require network."""
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    shared_torch_home = os.path.join(
        "/robot/data-lab/jzh/art-gen/third-party-weights",
        "trellis",
        "shared",
        "torch_hub",
    )
    candidate_repos = [
        os.path.join(PROJECT_ROOT, "pretrained", "torch_hub", "hub", "facebookresearch_dinov2_main"),
        os.path.join(PROJECT_ROOT, "pretrained", "dinov2"),
        os.path.join(torch_home, "hub", "facebookresearch_dinov2_main"),
        os.path.join(shared_torch_home, "hub", "facebookresearch_dinov2_main"),
    ]
    dinov2_repo = next(
        (repo for repo in candidate_repos if os.path.isfile(os.path.join(repo, "hubconf.py"))),
        "",
    )
    if dinov2_repo:
        if dinov2_repo not in sys.path:
            sys.path.insert(0, dinov2_repo)
        model = torch.hub.load(dinov2_repo, "dinov2_vitl14_reg", source="local", pretrained=False)
    else:
        candidate_package_roots = [
            os.environ.get("DINOV2_PACKAGE_ROOT", ""),
            os.path.join(PROJECT_ROOT, "sam3d_cu118_src_deps", "MoGe", "moge", "model"),
            os.path.join(PROJECT_ROOT, "sam3d_cu118_deps", "MoGe", "moge", "model"),
        ]
        dinov2_package_root = next(
            (
                root for root in candidate_package_roots
                if root and os.path.isfile(os.path.join(root, "dinov2", "hub", "backbones.py"))
            ),
            "",
        )
        if not dinov2_package_root:
            raise FileNotFoundError(
                "DINOv2 local repo/package not found; expected one of: "
                + ", ".join(candidate_repos + candidate_package_roots)
            )
        if dinov2_package_root not in sys.path:
            sys.path.insert(0, dinov2_package_root)
        from dinov2.hub.backbones import dinov2_vitl14_reg
        model = dinov2_vitl14_reg(pretrained=False)

    candidate_weights = [
        os.environ.get("DINOV2_WEIGHTS", ""),
        os.path.join(PROJECT_ROOT, "pretrained", "torch_hub", "checkpoints", "dinov2_vitl14_reg4_pretrain.pth"),
        os.path.join(torch_home, "hub", "checkpoints", "dinov2_vitl14_reg4_pretrain.pth"),
        os.path.join(torch_home, "checkpoints", "dinov2_vitl14_reg4_pretrain.pth"),
        os.path.join(shared_torch_home, "hub", "checkpoints", "dinov2_vitl14_reg4_pretrain.pth"),
        os.path.join(shared_torch_home, "checkpoints", "dinov2_vitl14_reg4_pretrain.pth"),
    ]
    weights_path = next((path for path in candidate_weights if os.path.isfile(path)), "")
    if not weights_path:
        raise FileNotFoundError(
            "DINOv2 local weights not found; expected one of: "
            + ", ".join(candidate_weights)
        )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().cuda()


def _preprocess_trellis_image(input_image: Image.Image) -> Image.Image:
    """Official TRELLIS RGBA preprocess: crop foreground and premultiply on black."""
    if input_image.mode != "RGBA":
        raise ValueError(
            "_images_to_tokens now expects RGBA images with alpha. "
            f"Got mode={input_image.mode}; pass an explicit foreground mask upstream."
        )
    image = input_image.convert("RGBA")
    rgba = np.asarray(image)
    alpha = rgba[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.shape[0] == 0:
        raise ValueError("_images_to_tokens received an RGBA image with empty alpha foreground")
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0)
    center = ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)
    size = int(max(int(x1) - int(x0), int(y1) - int(y0)) * 1.2)
    if size <= 0:
        raise ValueError(f"invalid foreground bbox for TRELLIS preprocess: {(int(x0), int(y0), int(x1), int(y1))}")
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    image = image.crop(bbox)
    image = image.resize((_DINOV2_RESOLUTION, _DINOV2_RESOLUTION), Image.Resampling.LANCZOS)
    rgba = np.asarray(image).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
    return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def _images_to_tokens(images: List[Image.Image]) -> torch.Tensor:
    """Encode PIL RGBA images into official TRELLIS DINOv2 tokens [V,1374,1024]."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
    ])

    tensors = [transform(_preprocess_trellis_image(img)) for img in images]
    batch = torch.stack(tensors).cuda()  # [V,3,518,518]

    model = _load_dinov2()
    with torch.no_grad():
        feats = model(batch, is_training=True)
        if "x_prenorm" not in feats:
            raise KeyError(f"DINOv2 output missing x_prenorm; keys={sorted(feats.keys())}")
        tokens = F.layer_norm(feats["x_prenorm"], feats["x_prenorm"].shape[-1:]).float()
    expected_shape = (len(images), 1374, 1024)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(f"DINOv2 token shape {tuple(tokens.shape)} != expected {expected_shape}")
    return tokens


# ---------------------------------------------------------------------------
# Cached ckpt loaders (lru_cache => same ckpt_path served at zero cost on reuse)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _load_ss_flow(ckpt_path: str) -> torch.nn.Module:
    """Load SparseStructureFlowModel for the 0611 multiflow checkpoint."""
    from trellis.models.sparse_structure_flow import SparseStructureFlowModel

    model_config = os.environ.get("SS_FLOW_MODEL_CONFIG", _SS_FLOW_MODEL_CONFIG)
    if os.path.isfile(model_config):
        cfg = _load_json(model_config)
        model_args = dict(cfg.get("args", cfg.get("models", {}).get("denoiser", {}).get("args", {})))
    else:
        model_args = dict(_SS_FLOW_DEFAULT_ARGS)
    ckpt_abs = _resolve(ckpt_path)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"SS Flow ckpt not found: {ckpt_abs}")
    sd = _load_state_dict(ckpt_abs)
    model_args["use_camera_pose"] = "view_pose_proj.weight" in sd
    model_args["use_view_id_embedding"] = "view_id_embedding.weight" in sd
    if "view_id_embedding.weight" in sd:
        model_args["num_view_embeddings"] = int(sd["view_id_embedding.weight"].shape[0])
    else:
        model_args["num_view_embeddings"] = 4
    model = SparseStructureFlowModel(**model_args).cuda()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SS Flow ckpt incompatible: missing={missing[:20]} "
            f"unexpected={unexpected[:20]}"
        )
    print(f"[inference._load_ss_flow] loaded {ckpt_abs}: "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@functools.lru_cache(maxsize=2)
def _load_ss_decoder(ckpt_path: str) -> torch.nn.Module:
    """Load SparseStructureDecoder.

    Mirrors trellis.models.from_pretrained: load json sibling for ``args``,
    instantiate, then load .safetensors weights. We avoid
    ``trellis.models.from_pretrained`` directly because that helper triggers
    the eager ``trellis/__init__.py`` import (CLAUDE.md "minimal deps").
    """
    import json
    from trellis.models.sparse_structure_vae import SparseStructureDecoder

    ckpt_abs = _resolve(ckpt_path)
    base, ext = os.path.splitext(ckpt_abs)
    config_path = base + ".json"
    weights_path = ckpt_abs if ext == ".safetensors" else base + ".safetensors"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"SS decoder config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SS decoder weights not found: {weights_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    decoder = SparseStructureDecoder(**config["args"]).cuda()
    sd = _load_state_dict(weights_path)
    decoder.load_state_dict(sd, strict=False)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    print(f"[inference._load_ss_decoder] loaded {weights_path}")
    return decoder


@functools.lru_cache(maxsize=4)
def _load_part_ss_latent_flow(ckpt_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load PartSSLatentFlowModel from a trainer checkpoint."""
    from trellis.models.part_flow import PartSSLatentFlowModel

    ckpt_abs = _resolve(ckpt_path)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"Part SS latent flow ckpt not found: {ckpt_abs}")

    ckpt = torch.load(ckpt_abs, map_location="cuda", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), dict):
        raise RuntimeError(
            "Part SS latent flow checkpoint is missing ckpt['config']['model']; "
            "refusing to instantiate with implicit defaults. "
            f"Checkpoint: {ckpt_abs}"
        )
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg = model_cfg.get("args", model_cfg)
    model = PartSSLatentFlowModel(**model_cfg).cuda()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Part SS latent flow checkpoint weights do not exactly match "
            "PartSSLatentFlowModel. Refusing silent missing/unexpected "
            f"weight fallback for checkpoint: {ckpt_abs}"
        ) from exc
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[inference._load_part_ss_latent_flow] loaded {ckpt_abs}")
    return model, cfg


@functools.lru_cache(maxsize=4)
def _load_slat_flow(ckpt_path: str) -> torch.nn.Module:
    """Load ElasticSLatFlowModel.

    Source: scripts/eval/stage4/infer.py:145-166 (build_model).
    """
    from trellis.models.structured_latent_flow import ElasticSLatFlowModel

    model = ElasticSLatFlowModel(**_SLAT_FLOW_DEFAULT_ARGS).cuda()
    ckpt_abs = _resolve(ckpt_path)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"SLat Flow ckpt not found: {ckpt_abs}")
    sd = _load_state_dict(ckpt_abs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[inference._load_slat_flow] loaded {ckpt_abs}: "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@functools.lru_cache(maxsize=2)
def _load_slat_decoder(ckpt_path: str) -> torch.nn.Module:
    """Load SLatGaussianDecoder.

    Delegates to ``trellis.utils.arts.slat_render_utils.load_slat_decoder``
    which already implements the json+safetensors load (D-09 / D-30).
    """
    from trellis.utils.arts.slat_render_utils import load_slat_decoder
    ckpt_abs = _resolve_weight_pointer(_resolve(ckpt_path))
    # load_slat_decoder takes a basename without extension.
    basename = ckpt_abs[:-len(".safetensors")] if ckpt_abs.endswith(".safetensors") else ckpt_abs
    return load_slat_decoder(basename)


@functools.lru_cache(maxsize=4)
def _load_slat_vae_decoder(ckpt_path: str) -> torch.nn.Module:
    """Load a frozen SLat VAE decoder from a local json+safetensors pair."""
    import json
    from safetensors.torch import load_file

    ckpt_abs = _resolve_weight_pointer(_resolve(ckpt_path))
    base, ext = os.path.splitext(ckpt_abs)
    config_path = base + ".json"
    weights_path = ckpt_abs if ext == ".safetensors" else base + ".safetensors"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"SLat decoder config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SLat decoder weights not found: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    name = str(config.get("name", ""))
    if name == "SLatGaussianDecoder":
        from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder
        decoder_cls = SLatGaussianDecoder
    elif name == "SLatMeshDecoder":
        from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
        decoder_cls = SLatMeshDecoder
    else:
        raise ValueError(
            f"Unsupported SLat VAE decoder {name!r} in {config_path}; "
            "expected one of ['SLatGaussianDecoder', 'SLatMeshDecoder']"
        )
    decoder = decoder_cls(**config["args"]).cuda().eval()
    decoder.load_state_dict(load_file(weights_path), strict=True)
    for p in decoder.parameters():
        p.requires_grad_(False)
    print(f"[inference._load_slat_vae_decoder] loaded {weights_path}")
    return decoder


@functools.lru_cache(maxsize=1)
def _ensure_sparse_tensor_init() -> None:
    """Trigger TRELLIS SparseTensor lazy backend init before loaded tensors call replace()."""
    from trellis.modules.sparse.basic import SparseTensor

    _dummy_coords = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    _dummy_feats = torch.zeros((1, 1), dtype=torch.float32, device="cuda")
    SparseTensor(coords=_dummy_coords, feats=_dummy_feats)


def _make_slat_initial_feats(
    num_voxels: int,
    channels: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is None:
        return torch.randn(num_voxels, channels, device=device, dtype=dtype)

    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randn(num_voxels, channels, generator=generator, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_ss_flow_from_tokens(
    cond_tokens: torch.Tensor | np.ndarray,
    ckpt_path: str,
    num_steps: int = 20,
    cfg_strength: float = _SS_FLOW_CFG_STRENGTH,
    seed: int = _SS_FLOW_SEED,
    fusion_mode: str | None = None,
) -> torch.Tensor:
    """Precomputed 4-view DINO tokens -> SS latent.

    ``fusion_mode="multidiffusion"`` matches the 0611 SS-flow rule: run the
    same latent through every single-view condition and average velocities.
    ``fusion_mode="concat"`` matches the 0616 concat checkpoint: pass four
    physical views as [1, 4, 1374, 1024] so the model can add view-id
    embeddings before its internal concat.
    """
    if isinstance(cond_tokens, np.ndarray):
        tokens = torch.from_numpy(np.ascontiguousarray(cond_tokens))
    else:
        tokens = cond_tokens
    tokens = tokens.to(device="cuda", dtype=torch.float32)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (1374, 1024):
        raise ValueError(f"SS-flow tokens must be [V,1374,1024], got {tuple(tokens.shape)}")
    V, T, D = tokens.shape
    if V != 4:
        raise ValueError(f"SS-flow expects 4 views, got {V}")
    mode = str(fusion_mode or os.environ.get("SS_FLOW_FUSION_MODE") or "multidiffusion").lower()
    aliases = {
        "multi": "multidiffusion",
        "multiflow": "multidiffusion",
        "avg": "multidiffusion",
        "average": "multidiffusion",
        "concat4": "concat",
        "concat_view": "concat",
        "concat_views": "concat",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"multidiffusion", "concat"}:
        raise ValueError(f"unsupported SS-flow fusion_mode={fusion_mode!r}")

    model = _load_ss_flow(ckpt_path)
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    sample = torch.randn(
        1,
        model.in_channels,
        model.resolution,
        model.resolution,
        model.resolution,
        generator=generator,
        device=tokens.device,
        dtype=torch.float32,
    )
    if mode == "concat":
        cond = tokens.unsqueeze(0).contiguous()
        neg_cond = torch.zeros_like(cond)
    else:
        cond = tokens.contiguous()
        neg_cond = torch.zeros(1, T, D, device=tokens.device, dtype=tokens.dtype)
    t_seq = np.linspace(1.0, 0.0, int(num_steps) + 1)

    with torch.no_grad():
        for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
            t_model = torch.tensor([1000.0 * float(t)], device=tokens.device, dtype=torch.float32)
            if mode == "concat":
                pred = model(sample, t_model, cond)
            else:
                preds = [model(sample, t_model, cond[i : i + 1]) for i in range(V)]
                pred = torch.stack(preds, dim=0).mean(dim=0)
            neg_pred = model(sample, t_model, neg_cond)
            pred_v = (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred
            sample = sample - (float(t) - float(t_prev)) * pred_v
    return sample[0].detach().float().cpu()  # [8,16,16,16]


def run_ss_flow(images: List[Image.Image], ckpt_path: str,
                num_steps: int = 20, cfg_strength: float = _SS_FLOW_CFG_STRENGTH,
                cam_pose: torch.Tensor | None = None,
                seed: int = _SS_FLOW_SEED) -> torch.Tensor:
    """Multi-view RGB -> SS latent.

    Args:
        images: list of PIL.Image (length 1..N, typical N=4 multi-view).
        ckpt_path: path to SS Flow DiT checkpoint (.safetensors).
        num_steps: flow-matching sampling steps.
        cfg_strength: classifier-free guidance strength.
        cam_pose: ignored for 0611 multiflow checkpoints; kept for API
            compatibility.

    Returns:
        Tensor [8, 16, 16, 16] fp32 on CPU.

    Source: 0611 multiflow inference rule: each Euler step runs the same
    current latent through every single-view condition independently, averages
    the predicted velocity, then applies CFG with a zero single-view token.
    """
    if not images:
        raise ValueError("run_ss_flow requires at least 1 image")

    tokens = _images_to_tokens(images)               # [V,1374,1024]
    if cam_pose is not None:
        # The 0611 multiflow checkpoints were trained without pose conditioning.
        # Keep the argument accepted so existing callers do not need branching.
        cam_pose = None

    return run_ss_flow_from_tokens(
        tokens,
        ckpt_path,
        num_steps=num_steps,
        cfg_strength=cfg_strength,
        seed=seed,
    )


def decode_ss(z_s: torch.Tensor, decoder_ckpt_path: str,
              threshold: float = 0.0) -> torch.Tensor:
    """SS latent [8,16,16,16] -> sparse occupancy coords.

    Args:
        z_s: SS latent Tensor [8,16,16,16] (output of run_ss_flow).
        decoder_ckpt_path: SparseStructureDecoder ckpt (basename or .safetensors).
        threshold: occupancy binarization threshold; voxels are kept where
            ``decoder.forward(z_s)[0,0] > threshold``.

    Returns:
        LongTensor [N, 3] (xyz int coords in 64^3 grid) on CPU.

    Source: 70% wrapping of SparseStructureDecoder.forward + nonzero.
    """
    if z_s.dim() != 4:
        raise ValueError(f"z_s must have 4 dims [C,H,W,D], got {tuple(z_s.shape)}")

    decoder = _load_ss_decoder(decoder_ckpt_path)
    z = z_s.unsqueeze(0).cuda()                      # [1,8,16,16,16]
    if next(decoder.parameters()).dtype == torch.float16:
        z = z.half()
    with torch.no_grad():
        logits = decoder(z)                          # [1, out_channels, 64, 64, 64]
    occ = logits[0, 0].float() > float(threshold)    # [64,64,64] bool
    coords = torch.nonzero(occ, as_tuple=False).long().cpu()  # [N,3]
    return coords


@functools.lru_cache(maxsize=2)
def _load_ss_encoder(ckpt_path: str) -> torch.nn.Module:
    """Load the TRELLIS SparseStructureEncoder (json sibling + .safetensors).

    Mirrors _load_ss_decoder; this is the SAME encoder pipeline/04→08 used to make
    the training z_global, so re-encoding a voxel with it lands in the part-flow
    checkpoint's SS latent space.
    """
    import json
    from trellis.models.sparse_structure_vae import SparseStructureEncoder

    ckpt_abs = _resolve(ckpt_path)
    base, ext = os.path.splitext(ckpt_abs)
    config_path = base + ".json"
    weights_path = ckpt_abs if ext == ".safetensors" else base + ".safetensors"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"SS encoder config not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SS encoder weights not found: {weights_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    encoder = SparseStructureEncoder(**config["args"]).cuda()
    encoder.load_state_dict(_load_state_dict(weights_path), strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[inference._load_ss_encoder] loaded {weights_path}")
    return encoder


def encode_ss(coords: torch.Tensor, encoder_ckpt_path: str,
              resolution: int = 64) -> torch.Tensor:
    """Surface voxel coords [N,3] -> SS latent z_global [8,16,16,16] (TRELLIS space).

    Mirrors pipeline/08_encode_ss_latents_per_part.py: build a [1,1,R,R,R] occupancy
    grid, run the encoder with ``sample_posterior=False`` and take the latent ``mean``.
    Used by mode-B to re-encode the (sam3d) surface voxel into the z_global the
    TRELLIS part-flow checkpoint expects, instead of feeding a sam3d-space latent.
    """
    import numpy as np

    c = np.asarray(coords).astype(np.int64).reshape(-1, 3)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= int(resolution)):
        raise ValueError(f"voxel coords 越界 [0,{resolution}): min={int(c.min())} max={int(c.max())}")
    encoder = _load_ss_encoder(encoder_ckpt_path)
    ss = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32, device="cuda")
    ct = torch.as_tensor(c, dtype=torch.long, device="cuda")
    ss[:, :, ct[:, 0], ct[:, 1], ct[:, 2]] = 1.0
    if next(encoder.parameters()).dtype == torch.float16:
        ss = ss.half()
    with torch.no_grad():
        latent = encoder(ss, sample_posterior=False)
    mean = latent.mean if (hasattr(latent, "mean") and not torch.is_tensor(latent)) else latent
    z_global = mean[0].detach().float().cpu()        # [8,16,16,16]
    if tuple(z_global.shape) != (8, resolution // 4, resolution // 4, resolution // 4):
        raise ValueError(f"encode_ss z_global 形状异常 {tuple(z_global.shape)}")
    return z_global


def run_part_ss_latent_flow(
    z_global: torch.Tensor,
    cond_tokens: torch.Tensor,
    ckpt_path: str,
    *,
    target_slots: List[int],
    mask_token_labels: torch.Tensor,
    target_part_names: List[str],
    ss_decoder_ckpt: str,
    part_token_weights: torch.Tensor | None = None,
    num_steps: int | None = None,
    decode_threshold: float = 0.0,
    decode: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate joint target-part SS latents and decoded voxel coords.

    This is a single-sample API: callers loop over objects externally.

    When ``decode=False`` the (TRELLIS) SS decoder is skipped entirely:
    ``part_latents``/``target_slots`` are still populated but ``part_coords``
    stays empty, and ``ss_decoder_ckpt`` may be "" since it is unused.
    """
    from trellis.trainers.arts.part_ss_latent_flow_losses import (
        build_part_ss_sampler_kwargs,
        sample_part_ss_latent,
    )

    if z_global.dim() != 4:
        raise ValueError(f"z_global must be [C,R,R,R], got {tuple(z_global.shape)}")
    if cond_tokens.dim() != 2:
        raise ValueError(f"cond_tokens must be single-sample [V*T,D], got {tuple(cond_tokens.shape)}")
    if mask_token_labels is None:
        raise ValueError("mask_token_labels is required and must be a real [V*T] long tensor")
    if mask_token_labels.dim() != 1:
        raise ValueError(f"mask_token_labels must be single-sample [V*T], got {tuple(mask_token_labels.shape)}")
    if mask_token_labels.dtype != torch.long:
        raise TypeError(f"mask_token_labels dtype must be torch.long, got {mask_token_labels.dtype}")
    if mask_token_labels.shape[0] != cond_tokens.shape[0]:
        raise ValueError(
            f"mask_token_labels length {mask_token_labels.shape[0]} does not match "
            f"cond token count {cond_tokens.shape[0]}"
        )
    if len(target_slots) != len(target_part_names):
        raise ValueError(
            f"target_slots length {len(target_slots)} must match "
            f"target_part_names length {len(target_part_names)}"
        )
    if not target_slots:
        raise ValueError("target_slots must contain at least one target part")
    if part_token_weights is None:
        for slot in target_slots:
            if not bool((mask_token_labels == int(slot)).any()):
                raise ValueError(f"target_slot={int(slot)} has zero 2D mask token coverage")
    else:
        if part_token_weights.dim() != 2:
            raise ValueError(
                f"part_token_weights must be [K,V*T], got {tuple(part_token_weights.shape)}"
            )
        if part_token_weights.shape != (len(target_slots), cond_tokens.shape[0]):
            raise ValueError(
                f"part_token_weights shape {tuple(part_token_weights.shape)} does not match "
                f"({len(target_slots)}, {cond_tokens.shape[0]})"
            )
        missing = [
            f"{target_part_names[idx]}(slot={int(slot)})"
            for idx, slot in enumerate(target_slots)
            if float(part_token_weights[idx].sum().item()) <= 0.0
        ]
        if missing:
            raise ValueError(
                "target parts have zero soft 2D mask token coverage: "
                + ", ".join(missing)
            )

    model, cfg = _load_part_ss_latent_flow(ckpt_path)
    flow_cfg = dict(cfg.get("flow", {})) if isinstance(cfg, dict) else {}
    if num_steps is None:
        num_steps = int(flow_cfg.get("num_steps", 20))
    noise_scale = float(flow_cfg.get("noise_scale", 1.0))
    latent_scale = float(flow_cfg.get("latent_scale", 1.0))
    if decode:
        if not ss_decoder_ckpt:
            raise ValueError("ss_decoder_ckpt is required when decode=True")
        # Preflight decoder files before sampling; decode_ss reuses this cached module.
        _load_ss_decoder(ss_decoder_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_global_b = z_global.unsqueeze(0).float().to(device)
    cond_b = cond_tokens.unsqueeze(0).float().to(device)
    labels_b = mask_token_labels.unsqueeze(0).long().to(device)
    part_valid = torch.ones((1, len(target_slots)), dtype=torch.bool, device=z_global_b.device)
    target_slots_t = torch.tensor([target_slots], dtype=torch.long, device=z_global_b.device)
    # Soft mask-overlap pooling weights. The checkpoint was trained with these (the
    # dataset emits item['part_token_weights'] under use_mask_overlap_pooling); the
    # eval/trainer always forward them. Dropping them silently falls back to the hard
    # min_fg=3 mask-vote pooling/role branch — a train/inference mismatch that makes
    # the part prediction degenerate. None keeps the legacy hard path for non-pooling
    # checkpoints. Single-sample -> unsqueeze to batch=1.
    part_token_weights_b = None
    if part_token_weights is not None:
        part_token_weights_b = part_token_weights.unsqueeze(0).float().to(z_global_b.device)

    result: Dict[str, Dict[str, np.ndarray]] = {
        "part_latents": {},
        "part_coords": {},
        "target_slots": {},
    }
    with torch.no_grad():
        part_latents = sample_part_ss_latent(
            model,
            z_global=z_global_b,
            cond=cond_b,
            mask_token_labels=labels_b,
            part_valid=part_valid,
            target_slots=target_slots_t,
            part_token_weights=part_token_weights_b,
            num_steps=int(num_steps),
            noise_scale=noise_scale,
            latent_scale=latent_scale,
            **build_part_ss_sampler_kwargs(model, flow_cfg),
        )[0].detach().float().cpu()
    for part_idx, (part_name, slot) in enumerate(zip(target_part_names, target_slots)):
        part_latent = part_latents[part_idx]
        result["part_latents"][str(part_name)] = part_latent.numpy().astype(np.float32)
        result["target_slots"][str(part_name)] = int(slot)
        if decode:
            coords = decode_ss(part_latent, ss_decoder_ckpt, threshold=decode_threshold)
            result["part_coords"][str(part_name)] = coords.numpy().astype(np.int64)
    return result


def run_slat_flow(images: List[Image.Image], coords: torch.Tensor,
                  ckpt_path: str, num_steps: int = 25, seed: int | None = None):
    """Multi-view RGB + sparse coords -> SLat (SparseTensor).

    Args:
        images: list of PIL.Image multi-view inputs.
        coords: LongTensor [N,3] from decode_ss.
        ckpt_path: SLat Flow DiT checkpoint.
        num_steps: flow-matching sampling steps.
        seed: optional local RNG seed for deterministic initial SLat noise.

    Returns:
        SparseTensor (trellis.modules.sparse.SparseTensor) with normalized feats.

    Source: scripts/eval/stage4/infer.py:256-408 (95% reuse, PSNR/SSIM stripped, D-29).
    """
    if not images:
        raise ValueError("run_slat_flow requires at least 1 image")

    tokens = _images_to_tokens(images)              # [V,1374,1024]
    V, T, D = tokens.shape
    return run_slat_flow_from_tokens(tokens.reshape(V * T, D), coords, ckpt_path, num_steps=num_steps, seed=seed)


def run_slat_flow_from_tokens(
    cond_tokens: torch.Tensor,
    coords: torch.Tensor,
    ckpt_path: str,
    num_steps: int = 25,
    seed: int | None = None,
):
    """Pre-encoded DINOv2 tokens + sparse coords -> normalized SLat.

    Args:
        cond_tokens: single-sample DINOv2 tokens, either [V*T,D] or [V,T,D].
        coords: LongTensor [N,3] sparse occupancy coords in the 64^3 grid.
        ckpt_path: SLat Flow DiT checkpoint.
        num_steps: flow-matching sampling steps.
        seed: optional local RNG seed for deterministic initial SLat noise.

    Returns:
        SparseTensor with normalized SLat feats, matching ``run_slat_flow``.
    """
    from trellis.modules.sparse import SparseTensor
    from trellis.pipelines.samplers import FlowEulerCfgSampler

    if cond_tokens.dim() == 3:
        cond_tokens = cond_tokens.reshape(-1, cond_tokens.shape[-1])
    if cond_tokens.dim() != 2:
        raise ValueError(f"cond_tokens must be [V*T,D] or [V,T,D], got {tuple(cond_tokens.shape)}")
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {tuple(coords.shape)}")
    if coords.shape[0] == 0:
        raise ValueError("coords must contain at least one voxel for SLat Flow sampling")

    cond = cond_tokens.unsqueeze(0).float().cuda()  # [1, V*T, 1024]
    neg_cond = torch.zeros_like(cond)

    # Build SparseTensor with a batch column prepended to coords.
    coords_int = coords.int().cuda()                # [N,3]
    N = coords_int.shape[0]
    batch_col = torch.zeros(N, 1, dtype=torch.int32, device=coords_int.device)
    sp_coords = torch.cat([batch_col, coords_int], dim=1)  # [N,4]
    feats = _make_slat_initial_feats(N, 8, device=coords_int.device, dtype=torch.float32, seed=seed)
    noise = SparseTensor(feats=feats, coords=sp_coords)

    model = _load_slat_flow(ckpt_path)
    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    with torch.no_grad():
        result = sampler.sample(
            model, noise=noise, cond=cond, neg_cond=neg_cond,
            steps=num_steps, cfg_strength=_CFG_STRENGTH, verbose=False,
        )
    return result.samples


def _slat_part_seed(base_seed: int, dataset_index: int, part_index: int) -> int:
    return (int(base_seed) + int(dataset_index) * 1_000_003 + int(part_index) * 9_176) % (2 ** 63 - 1)


def run_slat_flow_per_part(
    cond_tokens: torch.Tensor,
    part_coords: Dict[str, torch.Tensor],
    ckpt_path: str,
    *,
    num_steps: int = 25,
    base_seed: int | None = None,
    dataset_index: int = 0,
) -> Dict[str, Any]:
    """对每个 part 的 voxel 坐标独立跑 SLat flow（tokens 版，复用 DINO tokens，避免逐 part 重跑 DINOv2）。

    Args:
        cond_tokens: single-sample DINOv2 tokens, either [V*T,D] or [V,T,D]; shared across parts.
        part_coords: {part_name: LongTensor [N,3]} sparse occupancy coords per part.
        ckpt_path: SLat Flow DiT checkpoint, shared across parts.
        num_steps: flow-matching sampling steps, forwarded per part.
        base_seed: optional base seed; per-part seeds are derived deterministically.
                   ``None`` forwards ``seed=None`` to every part.
        dataset_index: sample index feeding the deterministic per-part seed formula.

    Returns:
        {part_name: SparseTensor}, key order preserving ``part_coords``.

    种子按 (base_seed, dataset_index, part_index) 确定性派生，与
    export_part_ss_latent_flow_examples.py 的 _slat_part_seed 公式一致。
    """
    out: Dict[str, Any] = {}
    for part_index, (name, coords) in enumerate(part_coords.items()):
        seed = None if base_seed is None else _slat_part_seed(int(base_seed), int(dataset_index), part_index)
        out[name] = run_slat_flow_from_tokens(cond_tokens, coords, ckpt_path, num_steps=num_steps, seed=seed)
    return out


def decode_slat(slat, decoder_ckpt_path: str,
                formats: List[str] = ("mesh", "gaussian")) -> Dict[str, Any]:
    """SLat -> {'mesh': trimesh.Trimesh, 'gaussian': trellis.representations.Gaussian, ...}.

    Source: scripts/train/stage4/render_utils.py:35-150 (85% reuse, D-30):
      load_slat_decoder + un_normalize_slat + decoder forward + format selection.

    Notes:
        Currently the underlying ``SLatGaussianDecoder`` (from the canonical
        TRELLIS pretrained ckpts) only emits Gaussian primitives. ``mesh``
        format will be returned as ``None`` when the loaded decoder does not
        support it; pipeline/03_final_decode.py persists whatever is present.
    """
    from trellis.utils.arts.slat_render_utils import un_normalize_slat
    # torch.load(slat.pt) can bypass SparseTensor.__init__; warm the lazy
    # backend once before un_normalize_slat calls SparseTensor.replace().
    _ensure_sparse_tensor_init()

    decoder = _load_slat_decoder(decoder_ckpt_path)
    mean = _DEFAULT_SLAT_MEAN.cuda()
    std = _DEFAULT_SLAT_STD.cuda()
    slat_raw = un_normalize_slat(slat, mean, std)

    with torch.no_grad():
        out = decoder(slat_raw)   # list-like, len == batch (1 here)

    sample = out[0] if hasattr(out, "__len__") else out
    result: Dict[str, Any] = {}
    if "gaussian" in formats:
        # SLatGaussianDecoder returns a Gaussian representation directly.
        result["gaussian"] = sample
    if "mesh" in formats:
        # Mesh extraction would require a SLatMeshDecoder; return None when
        # the loaded decoder does not provide a mesh interface.
        mesh = getattr(sample, "to_mesh", None)
        result["mesh"] = mesh() if callable(mesh) else None
    return result


def decode_slat_assets(
    slat,
    *,
    gaussian_decoder_ckpt: str | None = None,
    mesh_decoder_ckpt: str | None = None,
    slat_is_normalized: bool = True,
) -> Dict[str, Any]:
    """Decode one SLat with separate Gaussian and Mesh VAE decoder ckpts.

    ``run_slat_flow`` and ``run_slat_flow_from_tokens`` return normalized
    features, so the default path un-normalizes before decoding. A caller that
    passes raw SLat encoder output can set ``slat_is_normalized=False``.
    """
    from trellis.utils.arts.slat_render_utils import un_normalize_slat

    if gaussian_decoder_ckpt is None and mesh_decoder_ckpt is None:
        raise ValueError("decode_slat_assets requires at least one decoder ckpt")

    _ensure_sparse_tensor_init()

    slat_raw = slat
    if slat_is_normalized:
        mean = _DEFAULT_SLAT_MEAN.cuda()
        std = _DEFAULT_SLAT_STD.cuda()
        slat_raw = un_normalize_slat(slat, mean, std)

    result: Dict[str, Any] = {}

    def _first_sample(out):
        return out[0] if hasattr(out, "__len__") else out

    if gaussian_decoder_ckpt is not None:
        decoder = _load_slat_vae_decoder(gaussian_decoder_ckpt)
        with torch.no_grad():
            result["gaussian"] = _first_sample(decoder(slat_raw))
    if mesh_decoder_ckpt is not None:
        decoder = _load_slat_vae_decoder(mesh_decoder_ckpt)
        with torch.no_grad():
            result["mesh"] = _first_sample(decoder(slat_raw))
    return result
