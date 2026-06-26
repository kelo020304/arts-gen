#!/usr/bin/env python3
"""Check SAM3D SLat condition-token helper equivalence.

This loads the SAM3D pipeline and verifies that the new token helper keeps the
single-image path numerically identical to the original get_condition_input()
path. It also verifies that a one-item multi-view list is just the same tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


REPO = Path(__file__).resolve().parents[2]
GLUE = REPO / "submodules" / "sam3d-stage" / "infer_glue"
sys.path.insert(0, str(GLUE))

import slat_stage  # noqa: E402


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    return arr > 127


def _merge_rgba(image: Path, mask: Path) -> np.ndarray:
    rgb = _load_rgb(image)
    mask_arr = _load_mask(mask)
    if mask_arr.shape != rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask_arr.shape} does not match image shape {rgb.shape[:2]}"
        )
    return slat_stage._merge_mask_to_rgba(rgb, mask_arr)


def _first_manifest_row(path: Path) -> dict:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key in ("images", "masks"):
            if key not in row or not row[key]:
                raise ValueError(f"{path}:{line_no}: missing non-empty {key}")
        return row
    raise ValueError(f"manifest is empty: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = _first_manifest_row(args.manifest)
    pipe = slat_stage._build_pipeline(args.config, args.device, load_mesh_decoder=False)
    pipe.models.eval()
    for embedder in pipe.condition_embedders.values():
        if embedder is not None:
            embedder.eval()

    slat_input = pipe.preprocess_image(
        _merge_rgba(Path(row["images"][0]), Path(row["masks"][0])),
        pipe.slat_preprocessor,
    )

    with torch.no_grad():
        original_args, original_kwargs = pipe.get_condition_input(
            pipe.condition_embedders["slat_condition_embedder"],
            slat_input,
            pipe.slat_condition_input_mapping,
        )
        if original_kwargs or len(original_args) != 1:
            raise RuntimeError("original single-image condition path did not return one token tensor")
        original = original_args[0]
        single = pipe.get_slat_condition_tokens(slat_input)
        one_item_list = pipe.get_slat_condition_tokens([slat_input])

    if not torch.equal(original, single):
        raise AssertionError(
            f"single helper changed original output: max_abs_diff="
            f"{float((original - single).abs().max())}"
        )
    if not torch.equal(original, one_item_list):
        raise AssertionError(
            f"one-item list changed original output: max_abs_diff="
            f"{float((original - one_item_list).abs().max())}"
        )

    print(
        f"[ok] single-image condition tokens equivalent; shape={tuple(single.shape)} "
        f"dtype={single.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()
