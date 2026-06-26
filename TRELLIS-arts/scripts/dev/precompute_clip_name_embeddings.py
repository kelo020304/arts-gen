#!/usr/bin/env python3
"""Precompute frozen CLIP text token sequences for PartMMDiT part type names.

The script scans a part-completion manifest, reads each object's
``part_info.json``, collects unique ``parts[*].type`` strings, encodes them
with a local CLIP text encoder, and writes a CPU cache:

    {
        "dim": 768,
        "clip": "openai/clip-vit-large-patch14",
        "seq": {name: {"tokens": float32[L,768], "mask": bool[L]}},
    }

No network fallback is attempted. Missing data files are surfaced as normal
exceptions so dataset/manifest contract bugs fail loudly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer


DEFAULT_CLIP_DIR = "/robot/data-lab/jzh/art-gen/weights/clip-vit-large-patch14"
DEFAULT_RECON_SUBDIR = "reconstruction"
CLIP_NAME_DIM = 768
DEFAULT_OUT_NAME = "clip_vitl14_seq.pt"


def collect_types(manifest_path: Path, data_root: Path, recon_subdir: str) -> set[str]:
    """Collect unique part type strings from manifest-referenced part_info files."""

    types: set[str] = set()
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line_no, line in enumerate(manifest_file, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            obj_id = str(record.get("object_id", record.get("obj_id", "")))
            if not obj_id:
                raise KeyError(
                    f"{manifest_path}:{line_no} missing field 'object_id' or 'obj_id'"
                )
            part_info_rel = dict(record.get("paths", {})).get(
                "part_info",
                f"{recon_subdir}/part_info/{obj_id}/part_info.json",
            )
            part_info_path = Path(part_info_rel)
            if not part_info_path.is_absolute():
                part_info_path = data_root / part_info_path
            if not part_info_path.is_file():
                raise FileNotFoundError(f"part_info.json missing: {part_info_path}")

            with open(part_info_path, "r", encoding="utf-8") as part_info_file:
                part_info = json.load(part_info_file)
            for part in part_info["parts"].values():
                types.add(str(part["type"]))
    return types


@torch.no_grad()
def encode(
    types: list[str],
    clip_dir: str,
    device: str,
    batch_size: int = 64,
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode sorted part type strings as CLIP last_hidden_state token sequences."""

    tokenizer = CLIPTokenizer.from_pretrained(clip_dir, local_files_only=True)
    text_encoder = (
        CLIPTextModel.from_pretrained(clip_dir, local_files_only=True)
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    out: dict[str, dict[str, torch.Tensor]] = {}
    for start in range(0, len(types), batch_size):
        chunk = types[start : start + batch_size]
        inputs = tokenizer(chunk, padding=True, return_tensors="pt").to(device)
        encoded = text_encoder(**inputs)
        if not hasattr(encoded, "last_hidden_state"):
            raise AttributeError("CLIPTextModel output missing last_hidden_state")
        hidden = encoded.last_hidden_state
        mask = inputs["attention_mask"].bool()
        for row, name in enumerate(chunk):
            valid_len = int(mask[row].sum().item())
            if valid_len <= 0:
                raise ValueError(f"empty CLIP token sequence for part type {name!r}")
            out[name] = {
                "tokens": hidden[row, :valid_len].float().cpu(),
                "mask": mask[row, :valid_len].cpu(),
            }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen CLIP-L/14 pooled embeddings for part names."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest path, relative to data_root unless absolute.",
    )
    parser.add_argument("--recon_subdir", default=DEFAULT_RECON_SUBDIR)
    parser.add_argument("--clip_dir", default=DEFAULT_CLIP_DIR)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output .pt path. Defaults to "
            f"<data_root>/<recon_subdir>/name_emb_cache/{DEFAULT_OUT_NAME}"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path
    out_path = (
        Path(args.out)
        if args.out
        else data_root / args.recon_subdir / "name_emb_cache" / DEFAULT_OUT_NAME
    )

    types = sorted(collect_types(manifest_path, data_root, args.recon_subdir))
    print(f"[info] {len(types)} unique part types")
    embeddings = encode(types, args.clip_dir, args.device, args.batch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dim": CLIP_NAME_DIM,
            "clip": "openai/clip-vit-large-patch14",
            "seq": embeddings,
        },
        out_path,
    )
    print(f"[done] wrote {len(embeddings)} token sequences -> {out_path}")


if __name__ == "__main__":
    main()
