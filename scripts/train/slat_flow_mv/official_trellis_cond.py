#!/usr/bin/env python3
"""Official TRELLIS RGBA image-conditioning helpers for Track2.

This module intentionally reuses ``TRELLIS-arts/inference.py`` for the actual
preprocess and DINOv2 encode path so Track2 stays aligned with ee-eval's
``live_official_trellis_rgba`` token source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import inference  # noqa: E402


def load_mask_alpha(mask_path: str | Path, *, size: tuple[int, int]) -> Image.Image:
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"reference mask missing: {path}")
    if path.suffix == ".npy":
        mask = np.asarray(np.load(path))
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"{path}: expected 2-D mask after squeeze/max, got {mask.shape}")
        alpha = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    else:
        alpha = Image.open(path).convert("L")
    if alpha.size != size:
        alpha = alpha.resize(size, Image.Resampling.NEAREST)
    return alpha


def load_rgba_views(reference_rgb: Sequence[str | Path], reference_masks: Sequence[str | Path]) -> list[Image.Image]:
    if len(reference_rgb) != len(reference_masks):
        raise ValueError(f"reference_rgb length {len(reference_rgb)} != reference_masks length {len(reference_masks)}")
    if len(reference_rgb) != 4:
        raise ValueError(f"Track2 expects exactly 4 views, got {len(reference_rgb)}")
    images: list[Image.Image] = []
    for rgb_path, mask_path in zip(reference_rgb, reference_masks):
        rgb = Path(rgb_path)
        if not rgb.is_file():
            raise FileNotFoundError(f"reference RGB missing: {rgb}")
        image = Image.open(rgb).convert("RGBA")
        image.putalpha(load_mask_alpha(mask_path, size=image.size))
        images.append(image)
    return images


def _alpha_bbox(image: Image.Image) -> dict[str, Any]:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.shape[0] == 0:
        raise ValueError("RGBA view has empty alpha foreground")
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0)
    return {
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "foreground_px": int(foreground.shape[0]),
        "image_size": [int(image.width), int(image.height)],
    }


def dump_preprocessed_dino_inputs(
    reference_rgb: Sequence[str | Path],
    reference_masks: Sequence[str | Path],
    *,
    out_dir: str | Path,
    prefix: str,
    view_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    images = load_rgba_views(reference_rgb, reference_masks)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, image in enumerate(images):
        view = int(view_indices[i]) if view_indices is not None else i
        processed = inference._preprocess_trellis_image(image)
        path = out / f"{prefix}_view{view}_dino_input.png"
        processed.save(path)
        saved.append(
            {
                "view_index": view,
                "rgb": str(Path(reference_rgb[i]).resolve()),
                "mask": str(Path(reference_masks[i]).resolve()),
                "alpha": _alpha_bbox(image),
                "dino_input": str(path.resolve()),
                "dino_input_size": [int(processed.width), int(processed.height)],
                "preprocess": "alpha bbox >0.8*255, 1.2 square crop, 518 LANCZOS, RGB premultiplied by alpha onto black",
            }
        )
    meta = {"views": saved}
    (out / f"{prefix}_dino_input_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def encode_official_rgba_tokens(
    reference_rgb: Sequence[str | Path],
    reference_masks: Sequence[str | Path],
    *,
    dump_dir: str | Path | None = None,
    prefix: str = "sample",
    view_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if dump_dir is not None:
        dump_preprocessed_dino_inputs(reference_rgb, reference_masks, out_dir=dump_dir, prefix=prefix, view_indices=view_indices)
    images = load_rgba_views(reference_rgb, reference_masks)
    tokens = inference._images_to_tokens(images).detach().float().cpu()
    if tuple(tokens.shape) != (4, 1374, 1024):
        raise ValueError(f"official TRELLIS RGBA tokens expected [4,1374,1024], got {tuple(tokens.shape)}")
    meta = {
        "token_source": "live_official_trellis_rgba",
        "preprocess": "TRELLIS RGBA alpha crop + black premultiply + 518 resize + ImageNet normalize + DINO x_prenorm layer_norm",
        "token_shape": list(tokens.shape),
        "view_indices": [int(v) for v in view_indices] if view_indices is not None else list(range(4)),
        "reference_rgb": [str(Path(path).resolve()) for path in reference_rgb],
        "reference_masks": [str(Path(path).resolve()) for path in reference_masks],
    }
    return tokens, meta
