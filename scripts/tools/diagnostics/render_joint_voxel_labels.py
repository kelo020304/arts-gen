#!/usr/bin/env python3
"""Render joint promptable segmentation owner labels for selected objects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    PackedPromptablePartDataset,
    PartRow,
    collate_promptable_parts,
    mask_morphology,
    part_row_key,
)
from scripts.train.part_promptable_seg.train_part_promptable_seg import (  # noqa: E402
    _candidate_cells_from_latent_mask,
    _coords_to_keys_tensor,
    _positions_for_keys,
)
from trellis.models.part_seg.promptable_latent_seg import (  # noqa: E402
    PromptablePartLatentSegNet,
    semantic_classes_from_ckpt,
    voxel_embedding_dim_from_ckpt,
)


ENC_Y = 64
ENC_X = 64 * 64
PALETTE = [
    (142, 142, 142),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (57, 59, 121),
    (82, 84, 163),
    (107, 110, 207),
    (156, 158, 222),
    (99, 121, 57),
]


@dataclass(frozen=True)
class RowSpec:
    dataset_id: str
    obj_id: str
    angle_idx: int
    part_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--packed-dir", type=Path, required=True)
    p.add_argument("--selection-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--objects", nargs="+", required=True, help="obj_id, dataset_id::obj_id, or obj_id:aN")
    p.add_argument("--angle-idx", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=1800)
    p.add_argument("--height", type=int, default=900)
    return p.parse_args()


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_model(ckpt_path: Path, device: torch.device) -> PromptablePartLatentSegNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(ckpt.get("args") or {})
    state = _clean_state_dict(ckpt["model"])
    stem = state.get("stem.weight")
    model = PromptablePartLatentSegNet(
        dim=int(args.get("dim", 256)),
        depth=int(args.get("depth", 6)),
        head_depth=int(args.get("head_depth", 2)),
        heads=int(args.get("heads", 8)),
        use_xyz=bool(torch.is_tensor(stem) and int(stem.shape[1]) == 11),
        use_voxel_head=True,
        voxel_depth=int(args.get("voxel_depth", 3)),
        refine_mode=str(args.get("refine_mode", "token")),
        spconv_depth=int(args.get("spconv_depth", 4)),
        mask_encoder=str(args.get("mask_encoder", "cnn_grid")),
        point_k_boundary=int(args.get("point_k_boundary", 32)),
        point_k_interior=int(args.get("point_k_interior", 32)),
        point_resample_points=bool(args.get("point_resample_points", False)),
        semantic_classes=semantic_classes_from_ckpt(ckpt),
        voxel_embedding_dim=voxel_embedding_dim_from_ckpt(ckpt),
        use_body_prompt=bool(args.get("joint_seg", False)) or "body_prompt" in state,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_selected_rows(selection_json: Path, packed_dir: Path) -> list[PartRow]:
    specs = json.loads(selection_json.read_text(encoding="utf-8"))
    index = json.loads((packed_dir / "index.json").read_text(encoding="utf-8"))
    by_key = {str(entry["key"]): entry for entry in index["entries"]}
    enriched: list[PartRow] = []
    for spec in specs:
        dataset_id = str(spec.get("dataset_id", ""))
        obj_id = str(spec["obj_id"])
        angle_idx = int(spec["angle_idx"])
        part_name = str(spec["part_name"])
        key = f"{dataset_id}::{obj_id}|{angle_idx}|{part_name}" if dataset_id else f"{obj_id}|{angle_idx}|{part_name}"
        entry = by_key[key]
        enriched.append(
            PartRow(
                sample_idx=int(entry.get("index", 0)),
                part_idx=0,
                obj_id=obj_id,
                angle_idx=angle_idx,
                sample_id=f"{obj_id}_angle_{angle_idx}",
                part_name=part_name,
                semantic_type="",
                original_label=0,
                raw_count=int(entry.get("raw_count", 0)),
                view_indices=tuple(),
                dataset_id=dataset_id,
                data_root="",
                manifest_path="",
                category="",
                object_name="",
                part_item_name="",
                part_joint="",
                sample_part_names="",
                visible_view_count=0,
            )
        )
    return enriched


def coord_keys(coords: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = np.asarray(coords.detach().cpu() if torch.is_tensor(coords) else coords, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    return arr[:, 0] * ENC_X + arr[:, 1] * ENC_Y + arr[:, 2]


def keys_to_coords(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    x = keys // ENC_X
    rem = keys - x * ENC_X
    y = rem // ENC_Y
    z = rem - y * ENC_Y
    return np.stack([x, y, z], axis=1).astype(np.int64, copy=False)


def decode_key(key: int) -> tuple[int, int, int]:
    x = key // ENC_X
    rem = key - x * ENC_X
    y = rem // ENC_Y
    z = rem - y * ENC_Y
    return int(x), int(y), int(z)


def face_corners(x: int, y: int, z: int, face: str) -> list[tuple[float, float, float]]:
    if face == "x+":
        return [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
    if face == "x-":
        return [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)]
    if face == "y+":
        return [(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)]
    if face == "y-":
        return [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]
    if face == "z+":
        return [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
    if face == "z-":
        return [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)]
    raise ValueError(face)


def project(point: tuple[float, float, float], scale: float, ox: float, oy: float) -> tuple[float, float]:
    x, y, z = point
    return (x - y) * scale + ox, (x + y) * 0.50 * scale - z * 0.92 * scale + oy


def adjust(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(c * factor)))) for c in rgb)


def label_color(label: int) -> tuple[int, int, int]:
    return PALETTE[int(label) % len(PALETTE)]


def visible_faces(label_by_key: dict[int, int]) -> list[tuple[float, str, int, tuple[int, int, int]]]:
    keys = set(label_by_key)
    faces: list[tuple[float, str, int, tuple[int, int, int]]] = []
    for key, label in label_by_key.items():
        x, y, z = decode_key(key)
        for face, missing in (
            ("x-", x == 0 or key - ENC_X not in keys),
            ("x+", x == 63 or key + ENC_X not in keys),
            ("y-", y == 0 or key - ENC_Y not in keys),
            ("y+", y == 63 or key + ENC_Y not in keys),
            ("z-", z == 0 or key - 1 not in keys),
            ("z+", z == 63 or key + 1 not in keys),
        ):
            if not missing:
                continue
            corners = face_corners(x, y, z, face)
            cx = sum(p[0] for p in corners) / 4.0
            cy = sum(p[1] for p in corners) / 4.0
            cz = sum(p[2] for p in corners) / 4.0
            faces.append((0.92 * cx + 0.92 * cy + cz, face, int(label), (x, y, z)))
    return sorted(faces, key=lambda item: item[0])


def bbox_project_extent(keys: set[int]) -> tuple[float, float, float, float]:
    xs, ys, zs = [], [], []
    for key in keys:
        x, y, z = decode_key(key)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    mnx, mxx = min(xs), max(xs) + 1
    mny, mxy = min(ys), max(ys) + 1
    mnz, mxz = min(zs), max(zs) + 1
    projected = [
        ((x - y), (x + y) * 0.50 - z * 0.92)
        for x in (mnx, mxx)
        for y in (mny, mxy)
        for z in (mnz, mxz)
    ]
    px = [p[0] for p in projected]
    py = [p[1] for p in projected]
    return min(px), max(px), min(py), max(py)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label_panel(
    draw: ImageDraw.ImageDraw,
    label_by_key: dict[int, int],
    *,
    bbox: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    x0, y0, w, h = panel
    draw.rectangle((x0, y0, x0 + w - 1, y0 + h - 1), fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((x0 + 12, y0 + 10), title, fill=(0, 0, 0), font=font)
    draw.text((x0 + 12, y0 + 34), subtitle, fill=(20, 20, 20), font=small)
    if not label_by_key:
        return
    min_px, max_px, min_py, max_py = bbox
    proj_w = max_px - min_px
    proj_h = max_py - min_py
    plot_y = y0 + 64
    plot_h = h - 78
    scale = min((w - 36) / max(proj_w, 1.0), (plot_h - 16) / max(proj_h, 1.0))
    scale = max(1.0, min(scale, 10.0))
    ox = x0 + 18 - min_px * scale + (w - 36 - proj_w * scale) * 0.5
    oy = plot_y + 8 - min_py * scale + (plot_h - 16 - proj_h * scale) * 0.5
    shade = {"x+": 0.90, "x-": 0.68, "y+": 0.82, "y-": 0.72, "z+": 1.12, "z-": 0.58}
    for _, face, label, (x, y, z) in visible_faces(label_by_key):
        pts = [project(p, scale, ox, oy) for p in face_corners(x, y, z, face)]
        fill = adjust(label_color(label), shade[face])
        draw.polygon(pts, fill=fill, outline=adjust(fill, 0.55))


def render_side_by_side(
    *,
    gt_labels: dict[int, int],
    pred_labels: dict[int, int],
    class_names: list[str],
    metrics: dict[str, Any],
    title: str,
    out_path: Path,
    width: int,
    height: int,
) -> None:
    all_keys = set(gt_labels) | set(pred_labels)
    if not all_keys:
        raise ValueError("no labels to render")
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = load_font(18)
    small = load_font(13)
    draw.text((16, 10), title, fill=(0, 0, 0), font=font)
    summary = (
        f"mean IoU={float(metrics['mean_iou']):.4f}  "
        f"body={float(metrics.get('body_iou', 0.0)):.4f}  "
        f"door={float(metrics.get('door_iou', 0.0)):.4f}  "
        f"drawer={float(metrics.get('drawer_iou', 0.0)):.4f}"
    )
    draw.text((16, 36), summary, fill=(0, 0, 0), font=small)
    legend_x, legend_y = 16, 60
    for idx, name in enumerate(class_names):
        lx = legend_x + (idx % 5) * 250
        ly = legend_y + (idx // 5) * 22
        draw.rectangle((lx, ly, lx + 16, ly + 16), fill=label_color(idx), outline=(40, 40, 40))
        draw.text((lx + 22, ly - 1), f"{idx}:{name[:28]}", fill=(0, 0, 0), font=small)
    panel_y = legend_y + (math.ceil(len(class_names) / 5) * 22) + 12
    panel_h = height - panel_y - 16
    gap = 16
    panel_w = (width - 32 - gap) // 2
    bbox = bbox_project_extent(all_keys)
    draw_label_panel(
        draw,
        gt_labels,
        bbox=bbox,
        panel=(16, panel_y, panel_w, panel_h),
        title="GT owner labels",
        subtitle=f"voxels={len(gt_labels)}",
        font=font,
        small=small,
    )
    draw_label_panel(
        draw,
        pred_labels,
        bbox=bbox,
        panel=(16 + panel_w + gap, panel_y, panel_w, panel_h),
        title="Pred owner labels",
        subtitle=f"voxels={len(pred_labels)}",
        font=font,
        small=small,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def class_kind(name: str) -> str:
    text = name.lower()
    if name == "body":
        return "body"
    if "door" in text:
        return "door"
    if "drawer" in text:
        return "drawer"
    if any(term in text for term in ("knob", "button", "handle", "switch")):
        return "small"
    return "part"


def object_matches(token: str, dataset_id: str, obj_id: str) -> bool:
    return token == obj_id or token == f"{dataset_id}::{obj_id}"


def parse_object_token(token: str, default_angle_idx: int) -> tuple[str, int]:
    text = str(token)
    if ":a" in text:
        obj, angle = text.rsplit(":a", 1)
        return obj, int(angle)
    return text, int(default_angle_idx)


@torch.no_grad()
def run_object(
    model: PromptablePartLatentSegNet,
    ds: PackedPromptablePartDataset,
    rows: list[PartRow],
    *,
    object_token: str,
    angle_idx: int,
    device: torch.device,
    out_dir: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    selected_indices = [
        idx
        for idx, row in enumerate(rows)
        if object_matches(object_token, row.dataset_id, row.obj_id) and int(row.angle_idx) == int(angle_idx)
    ]
    if not selected_indices:
        raise ValueError(f"no rows found for {object_token} angle {angle_idx}")
    batch = collate_promptable_parts([ds[idx] for idx in selected_indices])
    z_global = batch["z_global"][:1].to(device=device, dtype=torch.float32)
    masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
    whole_coords = batch["whole_coords"][0].to(device=device, dtype=torch.long)
    whole_keys = torch.unique(_coords_to_keys_tensor(whole_coords, device=device), sorted=True)
    group_m = batch["m_gt"].to(device=device, dtype=torch.float32)
    candidate = _candidate_cells_from_latent_mask(mask_morphology((group_m > 0.5).any(dim=0, keepdim=True).float(), "dilate"), device=device)
    full_occ = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    full_occ[0, 0, whole_coords[:, 0], whole_coords[:, 1], whole_coords[:, 2]] = 1.0
    out = model(
        z_global,
        masks2d,
        candidate_cells=candidate,
        full_occ=full_occ,
        max_voxels_per_sample=0,
        joint_voxels=True,
    )
    pred_coords = out["joint_coords"].to(device=device, dtype=torch.long)
    pred_keys = pred_coords[:, 0] * ENC_X + pred_coords[:, 1] * ENC_Y + pred_coords[:, 2]
    order = torch.argsort(pred_keys)
    pred_keys = pred_keys[order]
    pred_label = out["joint_logits"].float().argmax(dim=1)[order].to(device=device)
    target = torch.zeros((pred_keys.shape[0],), dtype=torch.long, device=device)
    for class_idx, raw in enumerate(batch["raw_coords"], start=1):
        raw_keys = torch.unique(_coords_to_keys_tensor(raw, device=device), sorted=True)
        pos = _positions_for_keys(pred_keys, raw_keys)
        if pos.numel() > 0:
            target[pos] = int(class_idx)
    class_names = ["body", *[str(name) for name in batch["part_name"]]]
    gt_labels = {int(k): int(v) for k, v in zip(pred_keys.detach().cpu().tolist(), target.detach().cpu().tolist())}
    pred_labels = {int(k): int(v) for k, v in zip(pred_keys.detach().cpu().tolist(), pred_label.detach().cpu().tolist())}
    per_class = []
    for class_idx, name in enumerate(class_names):
        p = pred_label == int(class_idx)
        g = target == int(class_idx)
        inter = int((p & g).sum().detach().item())
        union = int((p | g).sum().detach().item())
        gt_count = int(g.sum().detach().item())
        pred_count = int(p.sum().detach().item())
        per_class.append(
            {
                "class_idx": int(class_idx),
                "name": name,
                "kind": class_kind(name),
                "iou": float(inter / max(1, union)),
                "precision": float(inter / max(1, pred_count)),
                "recall": float(inter / max(1, gt_count)),
                "gt_count": gt_count,
                "pred_count": pred_count,
            }
        )
    metrics: dict[str, Any] = {"classes": per_class}
    metrics["mean_iou"] = float(np.mean([item["iou"] for item in per_class]))
    for kind in ("body", "door", "drawer", "small"):
        vals = [item["iou"] for item in per_class if item["kind"] == kind]
        if vals:
            metrics[f"{kind}_iou"] = float(np.mean(vals))
    dataset_id = str(batch["dataset_id"][0])
    obj_id = str(batch["obj_id"][0])
    stem = f"{dataset_id}__{obj_id}__a{int(angle_idx)}".replace("/", "_")
    render_side_by_side(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        class_names=class_names,
        metrics=metrics,
        title=f"{dataset_id}|{obj_id}|angle_{int(angle_idx)} joint owner labels",
        out_path=out_dir / f"{stem}_labels_gt_vs_pred.png",
        width=int(width),
        height=int(height),
    )
    payload = {
        "dataset_id": dataset_id,
        "obj_id": obj_id,
        "angle_idx": int(angle_idx),
        "class_names": class_names,
        "metrics": metrics,
        "png": str((out_dir / f"{stem}_labels_gt_vs_pred.png").resolve()),
    }
    (out_dir / f"{stem}_labels_gt_vs_pred.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_selected_rows(args.selection_json, args.packed_dir)
    ds = PackedPromptablePartDataset(args.packed_dir, rows)
    model = load_model(args.ckpt, device)
    reports = [
        run_object(
            model,
            ds,
            rows,
            object_token=parse_object_token(str(obj), int(args.angle_idx))[0],
            angle_idx=parse_object_token(str(obj), int(args.angle_idx))[1],
            device=device,
            out_dir=args.out_dir,
            width=int(args.width),
            height=int(args.height),
        )
        for obj in args.objects
    ]
    (args.out_dir / "joint_label_render_summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in reports:
        metrics = item["metrics"]
        print(
            f"{item['dataset_id']}|{item['obj_id']}|a{item['angle_idx']} "
            f"mean={metrics['mean_iou']:.4f} body={metrics.get('body_iou', float('nan')):.4f} "
            f"door={metrics.get('door_iou', float('nan')):.4f} drawer={metrics.get('drawer_iou', float('nan')):.4f} "
            f"png={item['png']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
