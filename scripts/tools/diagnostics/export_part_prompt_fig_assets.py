#!/usr/bin/env python3
"""Export real v5 object assets for the promptable part-seg architecture figure."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.diagnostics.render_joint_voxel_labels import (  # noqa: E402
    ENC_X,
    ENC_Y,
    adjust,
    bbox_project_extent,
    decode_key,
    face_corners,
    load_font,
    project,
    visible_faces,
)


PACKED_DIR = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5")
OUT_ROOT = Path("/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/fig_assets")
DOC_IMAGE = PROJECT_ROOT / "docs/images/part-promptable-seg-full-architecture-real-29921.png"
DATA_ROOT = Path(
    "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511"
)

BODY = (142, 142, 142)
BLUE = (31, 119, 180)
ORANGE = (255, 127, 14)
GREEN = (44, 160, 44)
PURPLE = (148, 103, 189)
PALETTE = [BODY, BLUE, ORANGE, GREEN, PURPLE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, default=PACKED_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--dataset-id", default="physx-0511-drawer-door")
    parser.add_argument("--obj-id", default="29921")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--part-name", default="")
    parser.add_argument("--doc-image", type=Path, default=DOC_IMAGE)
    return parser.parse_args()


def coord_keys(coords: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(coords.detach().cpu() if torch.is_tensor(coords) else coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    return arr[:, 0] * ENC_X + arr[:, 1] * ENC_Y + arr[:, 2]


def keys_to_coords(keys: set[int]) -> np.ndarray:
    out = []
    for key in sorted(keys):
        out.append(decode_key(int(key)))
    return np.asarray(out, dtype=np.int64)


def encode_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    return arr[:, 0] * ENC_X + arr[:, 1] * ENC_Y + arr[:, 2]


def load_index(packed_dir: Path) -> dict[str, Any]:
    return json.loads((packed_dir / "index.json").read_text(encoding="utf-8"))


def candidates(index: dict[str, Any], dataset_id: str) -> list[tuple[int, int, str, int, list[str], int]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in index["entries"]:
        if str(entry.get("dataset_id", "")) != dataset_id:
            continue
        grouped[(str(entry["obj_id"]), int(entry["angle_idx"]))].append(entry)
    rows = []
    for (obj_id, angle_idx), entries in grouped.items():
        parts = sorted({str(item["part_name"]) for item in entries})
        k = len(parts)
        text = " ".join(parts).lower()
        if not (1 <= k <= 2):
            continue
        if not any(word in text for word in ("drawer", "door", "cabinet", "lid")):
            continue
        raw_total = sum(int(item.get("raw_count", 0) or 0) for item in entries)
        rows.append((k, -raw_total, obj_id, angle_idx, parts, raw_total))
    return sorted(rows)


def load_record(packed_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    shard = torch.load(packed_dir / str(entry["shard"]), map_location="cpu")
    return shard[int(entry["index"])]


def load_object_records(
    packed_dir: Path,
    index: dict[str, Any],
    *,
    dataset_id: str,
    obj_id: str,
    angle_idx: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    entries = [
        entry
        for entry in index["entries"]
        if str(entry.get("dataset_id", "")) == dataset_id
        and str(entry["obj_id"]) == str(obj_id)
        and int(entry["angle_idx"]) == int(angle_idx)
    ]
    if not entries:
        raise ValueError(f"no entries for {dataset_id}|{obj_id}|angle={angle_idx}")
    entries = sorted(entries, key=lambda item: str(item["part_name"]))
    return [(entry, load_record(packed_dir, entry)) for entry in entries]


def color_for_label(label: int) -> tuple[int, int, int]:
    return PALETTE[int(label) % len(PALETTE)]


def render_voxel_labels(
    label_by_key: dict[int, int],
    out_path: Path,
    *,
    size: tuple[int, int] = (760, 620),
    margin: int = 34,
    background: tuple[int, int, int] = (255, 255, 255),
) -> None:
    if not label_by_key:
        raise ValueError("cannot render empty voxel label map")
    width, height = size
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    min_px, max_px, min_py, max_py = bbox_project_extent(set(label_by_key))
    proj_w = max_px - min_px
    proj_h = max_py - min_py
    scale = min((width - margin * 2) / max(proj_w, 1.0), (height - margin * 2) / max(proj_h, 1.0))
    scale = max(1.0, min(scale, 12.0))
    ox = margin - min_px * scale + (width - margin * 2 - proj_w * scale) * 0.5
    oy = margin - min_py * scale + (height - margin * 2 - proj_h * scale) * 0.5
    shade = {"x+": 0.90, "x-": 0.68, "y+": 0.82, "y-": 0.72, "z+": 1.12, "z-": 0.58}
    for _, face, label, (x, y, z) in visible_faces(label_by_key):
        pts = [project(p, scale, ox, oy) for p in face_corners(x, y, z, face)]
        fill = adjust(color_for_label(label), shade[face])
        draw.polygon(pts, fill=fill, outline=adjust(fill, 0.54))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def overlay_prompt_mask(
    rgb_path: Path,
    mask_path: Path,
    original_label: int,
    out_path: Path,
    *,
    color: tuple[int, int, int] = BLUE,
) -> int:
    rgb = Image.open(rgb_path).convert("RGB")
    label = np.asarray(np.load(mask_path))
    mask = label == int(original_label)
    if mask.shape[:2] != (rgb.height, rgb.width):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(rgb.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    base = np.asarray(rgb, dtype=np.float32)
    gray = np.asarray(rgb.convert("L").convert("RGB"), dtype=np.float32)
    canvas = gray * 0.72 + 255.0 * 0.28
    overlay = np.asarray(color, dtype=np.float32)
    canvas[mask] = canvas[mask] * 0.30 + overlay * 0.70
    # Draw a thin blue boundary by dilating in image space without scipy.
    m = mask
    pad = np.pad(m, 1, mode="constant")
    neigh = (
        pad[0:-2, 1:-1]
        | pad[2:, 1:-1]
        | pad[1:-1, 0:-2]
        | pad[1:-1, 2:]
        | pad[0:-2, 0:-2]
        | pad[0:-2, 2:]
        | pad[2:, 0:-2]
        | pad[2:, 2:]
    )
    edge = neigh & ~m
    canvas[edge] = np.asarray((15, 76, 129), dtype=np.float32)
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return int(mask.sum())


def source_render_dir(record: dict[str, Any]) -> Path:
    return DATA_ROOT / "renders" / str(record["obj_id"]) / f"angle_{int(record['angle_idx'])}"


def export_assets(
    packed_dir: Path,
    index: dict[str, Any],
    *,
    dataset_id: str,
    obj_id: str,
    angle_idx: int,
    part_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    records = load_object_records(packed_dir, index, dataset_id=dataset_id, obj_id=obj_id, angle_idx=angle_idx)
    selected_entry, selected = records[0]
    if part_name:
        matches = [(entry, rec) for entry, rec in records if str(entry["part_name"]) == part_name]
        if not matches:
            raise ValueError(f"part {part_name!r} not in {[entry['part_name'] for entry, _ in records]}")
        selected_entry, selected = matches[0]
    part_names = [str(entry["part_name"]) for entry, _ in records]
    whole_coords = np.asarray(selected["whole_coords"], dtype=np.int64)
    part_coord_items = [(str(entry["part_name"]), np.asarray(rec["raw_coords"], dtype=np.int64)) for entry, rec in records]
    whole_keys = set(int(x) for x in coord_keys(whole_coords))
    part_key_sets = [(name, set(int(x) for x in coord_keys(coords))) for name, coords in part_coord_items]
    movable_union = set().union(*(keys for _, keys in part_key_sets))
    body_keys = whole_keys - movable_union

    render_voxel_labels({key: 0 for key in whole_keys}, out_dir / "whole_voxel.png")

    exploded_labels: dict[int, int] = {int(key): 0 for key in body_keys}
    body_center = keys_to_coords(body_keys).mean(axis=0) if body_keys else whole_coords.mean(axis=0)
    for idx, (_, coords) in enumerate(part_coord_items, start=1):
        center = coords.mean(axis=0)
        direction = np.sign(center - body_center).astype(np.int64)
        direction[2] = 0
        if not direction[:2].any():
            direction = np.asarray([0, -1, 0], dtype=np.int64)
        shift = direction * 18
        shifted = coords + shift.reshape(1, 3)
        for key in encode_coords(shifted):
            exploded_labels[int(key)] = int(idx)
    render_voxel_labels(exploded_labels, out_dir / "parts_exploded.png")

    view_indices = [int(v) for v in np.asarray(selected["view_indices"]).reshape(-1).tolist()]
    render_dir = source_render_dir(selected)
    view_rows = []
    for view_idx in view_indices:
        mask_path = render_dir / "mask" / f"mask_{view_idx}.npy"
        rgb_path = render_dir / "rgb" / f"view_{view_idx}.png"
        if not mask_path.is_file() or not rgb_path.is_file():
            continue
        label_map = np.asarray(np.load(mask_path))
        area = int((label_map == int(selected["original_label"])).sum())
        view_rows.append((area, view_idx, rgb_path, mask_path))
    view_rows = sorted(view_rows, reverse=True)[:3]
    mask_paths = []
    for out_idx, (_, view_idx, rgb_path, mask_path) in enumerate(view_rows, start=1):
        out_path = out_dir / f"mask_v{out_idx}.png"
        area = overlay_prompt_mask(rgb_path, mask_path, int(selected["original_label"]), out_path)
        mask_paths.append({"view_idx": int(view_idx), "mask_pixels": int(area), "path": str(out_path)})

    manifest = {
        "dataset_id": dataset_id,
        "obj_id": str(obj_id),
        "angle_idx": int(angle_idx),
        "part_count_without_body": len(part_names),
        "part_names": part_names,
        "selected_prompt_part": str(selected_entry["part_name"]),
        "whole_voxels": int(len(whole_keys)),
        "body_voxels": int(len(body_keys)),
        "paths": {
            "whole_voxel": str((out_dir / "whole_voxel.png").resolve()),
            "parts_exploded": str((out_dir / "parts_exploded.png").resolve()),
            "prompt_masks": mask_paths,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def paste_fit(dst: Image.Image, src: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    src = src.convert("RGB")
    src.thumbnail((w, h), Image.Resampling.LANCZOS)
    px = x0 + (w - src.width) // 2
    py = y0 + (h - src.height) // 2
    dst.paste(src, (px, py))


def draw_arrow(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], fill=(90, 90, 90), width=4) -> None:
    draw.line((a[0], a[1], b[0], b[1]), fill=fill, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 12
    pts = [
        b,
        (int(b[0] - size * math.cos(ang - 0.45)), int(b[1] - size * math.sin(ang - 0.45))),
        (int(b[0] - size * math.cos(ang + 0.45)), int(b[1] - size * math.sin(ang + 0.45))),
    ]
    draw.polygon(pts, fill=fill)


def rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, outline) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=(255, 255, 255), outline=outline, width=3)
    lines = text.split("\n")
    y = xy[1] + (xy[3] - xy[1] - 26 * len(lines)) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((xy[0] + (xy[2] - xy[0] - (bbox[2] - bbox[0])) // 2, y), line, fill=(25, 25, 25), font=font)
        y += 27


def make_architecture(manifest: dict[str, Any], out_path: Path, doc_image: Path) -> None:
    out_dir = Path(manifest["paths"]["whole_voxel"]).parent
    whole = Image.open(out_dir / "whole_voxel.png")
    exploded = Image.open(out_dir / "parts_exploded.png")
    masks = [Image.open(item["path"]) for item in manifest["paths"]["prompt_masks"][:3]]

    width, height = 2500, 1180
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    font = load_font(24)
    small = load_font(19)
    tiny = load_font(16)

    stage_x = [50, 440, 790, 1180, 1560, 1960, 2360]
    for i, label in enumerate(["1 Inputs", "2 Encoders", "3 Queries", "4 Interaction", "5 Seg Head", "6 Output"]):
        x = stage_x[i]
        draw.text((x, 28), label, fill=(10, 10, 10), font=title_font)
        if i > 0:
            draw.line((x - 34, 20, x - 34, height - 120), fill=(218, 218, 218), width=2)

    draw.text((78, 96), "Multi-view part masks (prompt)", fill=(25, 25, 25), font=small)
    mask_boxes = [(55, 130, 175, 250), (188, 130, 308, 250), (321, 130, 441, 250)]
    for src, box in zip(masks, mask_boxes, strict=False):
        draw.rounded_rectangle(box, radius=10, fill=(250, 250, 250), outline=(210, 210, 210), width=2)
        paste_fit(image, src, (box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4))
    draw.text((78, 345), "Whole-object voxels (occupancy)", fill=(25, 25, 25), font=small)
    paste_fit(image, whole, (35, 380, 390, 780))

    rounded_box(draw, (475, 145, 690, 300), "2D mask\nencoder", font, (222, 173, 65))
    rounded_box(draw, (465, 530, 705, 720), "Sparse 3D\nencoder\n(sparse-conv +\ntransformer)", tiny, BLUE)
    draw_arrow(draw, (442, 190), (475, 222))
    draw_arrow(draw, (390, 570), (465, 625))

    grid_box = (830, 430, 1120, 735)
    for z in range(7):
        x0 = grid_box[0] + z * 26
        y0 = grid_box[1] + z * 16
        draw.rectangle((x0, y0, x0 + 165, y0 + 165), outline=(91, 154, 210), width=2)
    for i in range(6):
        draw.line((grid_box[0] + i * 33, grid_box[1], grid_box[0] + i * 33 + 156, grid_box[1] + 96), fill=(130, 184, 226), width=1)
        draw.line((grid_box[0], grid_box[1] + i * 33, grid_box[0] + 156, grid_box[1] + i * 33 + 96), fill=(130, 184, 226), width=1)
    draw.text((900, 760), "Trunk cells", fill=BLUE, font=font)
    draw_arrow(draw, (705, 625), (830, 590), fill=BLUE)

    q_y = 165
    query_pos = [
        (835, q_y, BODY, "body"),
        (945, q_y, BLUE, "drawer"),
        (1065, q_y, PURPLE, "global"),
    ]
    for x, y, color, label in query_pos:
        draw.rectangle((x, y, x + 42, y + 42), fill=adjust(color, 1.05), outline=(40, 40, 40), width=2)
        if label:
            bbox = draw.textbbox((0, 0), label, font=small)
            draw.text((x + 21 - (bbox[2] - bbox[0]) // 2, y + 52), label, fill=(20, 20, 20), font=small)
    draw.arc((790, 94, 1242, 232), start=190, end=350, fill=(20, 20, 20), width=3)
    draw.text((965, 92), "self-attend", fill=(20, 20, 20), font=small)
    for x, _, color, _ in query_pos:
        draw.line((x + 21, q_y + 42, x + 70, 430), fill=color, width=3)
    draw.text((940, 370), "cross-attend", fill=(20, 20, 20), font=small)
    for dot_x, target_x in [(115, 966), (248, 966), (381, 966)]:
        draw.line((dot_x, 262, dot_x, 310), fill=BLUE, width=2)
        draw.line((dot_x, 310, target_x, q_y), fill=BLUE, width=2)

    draw_arrow(draw, (1140, 590), (1245, 590))
    node_center = (1330, 560)
    nodes = []
    for i in range(8):
        a = i * math.tau / 8.0
        nodes.append((int(node_center[0] + 95 * math.cos(a)), int(node_center[1] + 82 * math.sin(a))))
    for a in nodes:
        for b in nodes:
            if a < b:
                draw.line((a[0], a[1], b[0], b[1]), fill=(205, 205, 205), width=1)
    for x, y in nodes:
        draw.rectangle((x - 13, y - 13, x + 13, y + 13), fill=(180, 180, 180), outline=(80, 80, 80))
    draw.line((1240, 455, 1430, 665), fill=(220, 30, 30), width=5)
    draw.line((1430, 455, 1240, 665), fill=(220, 30, 30), width=5)
    draw.text((1215, 705), "no voxel self-attention\n(O(S^2) removed)", fill=(20, 20, 20), font=small)

    draw_arrow(draw, (1460, 590), (1560, 590))
    draw.text((1615, 230), "dot product:\nvoxels x queries", fill=(20, 20, 20), font=small)
    mx, my, cell = 1595, 320, 34
    for r in range(8):
        for c in range(6):
            draw.rectangle((mx + c * cell, my + r * cell, mx + (c + 1) * cell, my + (r + 1) * cell), outline=(200, 200, 200))
    for i, color in enumerate([BODY, BLUE, PURPLE, BLUE, BODY, BLUE, PURPLE, BLUE]):
        draw.rectangle((mx + (i % 6) * cell, my + i * cell, mx + (i % 6 + 1) * cell, my + (i + 1) * cell), fill=color, outline=(200, 200, 200))
    bar_x = 1855
    for i, color in enumerate([BODY, BLUE, BLUE, BODY, BLUE, PURPLE, BLUE, BODY, BLUE, BLUE]):
        draw.rectangle((bar_x, 330 + i * 28, bar_x + 42, 358 + i * 28), fill=color, outline=(245, 245, 245))
    draw.text((1790, 250), "per-voxel softmax\n-> single owner", fill=(20, 20, 20), font=small)
    draw_arrow(draw, (1805, 455), (1855, 470))

    draw_arrow(draw, (1915, 470), (1975, 470))
    paste_fit(image, exploded, (1985, 150, 2460, 820))
    draw.text((2030, 90), "Labeled parts ->\nper-part meshes\n(sim-ready)", fill=(20, 20, 20), font=font)

    draw.rounded_rectangle((425, 945, 1000, 1035), radius=18, fill=(245, 250, 255), outline=BLUE, width=3)
    draw.text((515, 975), "One forward for all parts + body", fill=(20, 20, 20), font=font)
    draw.rounded_rectangle((1180, 945, 2010, 1035), radius=18, fill=(246, 254, 247), outline=GREEN, width=3)
    draw.text((1275, 975), "Single owner => no inter-part penetration", fill=(20, 90, 35), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    doc_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(doc_image)


def main() -> int:
    args = parse_args()
    index = load_index(args.packed_dir)
    cand = candidates(index, args.dataset_id)
    print("[CANDIDATES]")
    printed = 0
    seen: set[str] = set()
    for k, neg_total, obj_id, angle_idx, parts, raw_total in cand:
        kind = "drawer" if "drawer" in " ".join(parts).lower() else "door"
        key = f"{obj_id}:{kind}"
        if key in seen:
            continue
        seen.add(key)
        print(f"{args.dataset_id}|{obj_id}|angle={angle_idx}|K={k}|raw_total={raw_total}|parts={parts}")
        printed += 1
        if printed >= 5:
            break
    out_dir = args.out_root / str(args.obj_id)
    manifest = export_assets(
        args.packed_dir,
        index,
        dataset_id=args.dataset_id,
        obj_id=str(args.obj_id),
        angle_idx=int(args.angle_idx),
        part_name=str(args.part_name),
        out_dir=out_dir,
    )
    make_architecture(manifest, out_dir / "architecture_real_object.png", args.doc_image)
    print("[SELECTED]")
    print(
        f"{manifest['dataset_id']}|{manifest['obj_id']}|angle={manifest['angle_idx']} "
        f"K={manifest['part_count_without_body']} parts={manifest['part_names']} "
        f"prompt_part={manifest['selected_prompt_part']}"
    )
    print("[PATHS]")
    print(manifest["paths"]["whole_voxel"])
    for item in manifest["paths"]["prompt_masks"]:
        print(item["path"])
    print(manifest["paths"]["parts_exploded"])
    print(str((out_dir / "architecture_real_object.png").resolve()))
    print(str(args.doc_image.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
