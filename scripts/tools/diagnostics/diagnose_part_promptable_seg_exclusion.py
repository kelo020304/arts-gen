#!/usr/bin/env python3
"""Read-only inference-time mutual-exclusion diagnostics for Route-V part seg."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
for path in (TRELLIS_PATH, DEV_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_part_promptable_seg_p0 import _dict_to_obj, dataset_source, load_model, raw_size_bucket  # noqa: E402
from scripts.train.part_promptable_seg.part_promptable_seg_utils import PackedPromptablePartDataset, collate_promptable_parts, format_table, latent_support_mask, mask_morphology  # noqa: E402
from scripts.train.part_promptable_seg.train_part_promptable_seg import full_occ_for_batch, partition_coords_by_embedding  # noqa: E402


DEFAULT_RUN_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0618-2")
DEFAULT_CKPT = DEFAULT_RUN_DIR / "ckpts" / "step_100000.pt"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "eval" / "step_100000" / "diagnostics_exclusion"
SIZE_ORDER = ("all", "tiny", "small", "medium", "large", "button")
RULE_ORDER = ("R0", "R1", "R2a", "R2b", "R2c")
NEI6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
PER_PART_BASELINE = 0.7923
PARTITION_BASELINE = 0.7727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-parts", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--visuals", type=int, default=6)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    print(f"\n[{title}]", flush=True)
    display_rows = [{key: fmt(row.get(key, "")) for key in headers} for row in rows]
    print(format_table(display_rows, headers) if display_rows else "(empty)", flush=True)


def key_array(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    coords64 = coords.astype(np.int64, copy=False)
    return coords64[:, 0] * 4096 + coords64[:, 1] * 64 + coords64[:, 2]


def keys_to_coords(keys: Iterable[int]) -> np.ndarray:
    keys_arr = np.fromiter((int(key) for key in keys), dtype=np.int64)
    if keys_arr.size == 0:
        return np.empty((0, 3), dtype=np.int16)
    x = keys_arr // 4096
    y = (keys_arr % 4096) // 64
    z = keys_arr % 64
    return np.stack([x, y, z], axis=1).astype(np.int16)


def coords_to_tuples(coords: np.ndarray) -> set[tuple[int, int, int]]:
    if coords.size == 0:
        return set()
    return {tuple(map(int, item)) for item in coords.tolist()}


def tuple_to_key(item: tuple[int, int, int]) -> int:
    x, y, z = item
    return int(x) * 4096 + int(y) * 64 + int(z)


def key_set_from_tuples(items: set[tuple[int, int, int]]) -> set[int]:
    return {tuple_to_key(item) for item in items}


def interface_keys_for_part(gt: set[tuple[int, int, int]], sibling_gt: set[tuple[int, int, int]]) -> set[int]:
    if not gt or not sibling_gt:
        return set()
    out: set[tuple[int, int, int]] = set()
    for x, y, z in gt:
        for dx, dy, dz in NEI6:
            if (x + dx, y + dy, z + dz) in sibling_gt:
                out.add((x, y, z))
                break
    return key_set_from_tuples(out)


def row_group_key(row: Any) -> str:
    dataset_id = getattr(row, "dataset_id", "")
    prefix = f"{dataset_id}::" if dataset_id else ""
    return f"{prefix}{getattr(row, 'obj_id')}|angle_{int(getattr(row, 'angle_idx'))}"


class PackedGroupBatchSampler(Sampler[list[int]]):
    """Pack complete object/angle groups into larger eval batches."""

    def __init__(self, rows: list[Any], *, batch_parts: int) -> None:
        groups: dict[str, list[int]] = {}
        order: list[str] = []
        for idx, row in enumerate(rows):
            key = row_group_key(row)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(idx)
        batches: list[list[int]] = []
        current: list[int] = []
        for key in order:
            group = groups[key]
            if current and len(current) + len(group) > int(batch_parts):
                batches.append(current)
                current = []
            if len(group) > int(batch_parts):
                if current:
                    batches.append(current)
                    current = []
                batches.append(group)
            else:
                current.extend(group)
        if current:
            batches.append(current)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def object_angle_key_from_batch(batch: dict[str, Any], idx: int) -> str:
    dataset_ids = batch.get("dataset_id", [""] * len(batch["obj_id"]))
    return f"{dataset_ids[idx]}::{batch['obj_id'][idx]}|angle_{int(batch['angle_idx'][idx])}"


def size_labels(row: dict[str, Any]) -> list[str]:
    labels = ["all", raw_size_bucket(row)]
    text = f"{row.get('part_name', '')} {row.get('semantic_type', '')}".lower()
    if "button" in text:
        labels.append("button")
    return labels


def pred_sets_from_partition(coords_list: list[torch.Tensor], partition_coords: list[torch.Tensor]) -> list[set[int]]:
    out: list[set[int]] = []
    for coords in partition_coords:
        arr = coords.detach().cpu().numpy().astype(np.int16, copy=False)
        out.append(set(key_array(arr).tolist()))
    return out


def argmax_assign(candidate_keys: list[np.ndarray], candidate_probs: list[np.ndarray], indices: list[int]) -> dict[int, set[int]]:
    keys_items = []
    prob_items = []
    part_items = []
    for idx in indices:
        keys = candidate_keys[idx]
        if keys.size == 0:
            continue
        keys_items.append(keys)
        prob_items.append(candidate_probs[idx])
        part_items.append(np.full((keys.shape[0],), idx, dtype=np.int32))
    out = {idx: set() for idx in indices}
    if not keys_items:
        return out
    keys_all = np.concatenate(keys_items)
    probs_all = np.concatenate(prob_items)
    parts_all = np.concatenate(part_items)
    order = np.lexsort((-probs_all, keys_all))
    sorted_keys = keys_all[order]
    first = np.empty((sorted_keys.shape[0],), dtype=bool)
    first[0] = True
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    for key, part in zip(sorted_keys[first], parts_all[order][first]):
        out[int(part)].add(int(key))
    return out


def threshold_argmax_assign(
    candidate_keys: list[np.ndarray],
    candidate_probs: list[np.ndarray],
    r0_sets: list[set[int]],
    indices: list[int],
    *,
    margin: float | None = None,
) -> dict[int, set[int]]:
    keys_items = []
    prob_items = []
    part_items = []
    for idx in indices:
        claimed = r0_sets[idx]
        if not claimed:
            continue
        keys = candidate_keys[idx]
        probs = candidate_probs[idx]
        keep = np.isin(keys, np.fromiter(claimed, dtype=np.int64), assume_unique=False)
        if not bool(keep.any()):
            continue
        keys_items.append(keys[keep])
        prob_items.append(probs[keep])
        part_items.append(np.full((int(keep.sum()),), idx, dtype=np.int32))
    out = {idx: set() for idx in indices}
    if not keys_items:
        return out
    keys_all = np.concatenate(keys_items)
    probs_all = np.concatenate(prob_items)
    parts_all = np.concatenate(part_items)
    order = np.lexsort((-probs_all, keys_all))
    sorted_keys = keys_all[order]
    sorted_probs = probs_all[order]
    sorted_parts = parts_all[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], sorted_keys.shape[0]]
    for start, end in zip(starts, ends):
        if end - start == 1:
            out[int(sorted_parts[start])].add(int(sorted_keys[start]))
            continue
        if margin is None:
            out[int(sorted_parts[start])].add(int(sorted_keys[start]))
            continue
        if float(sorted_probs[start] - sorted_probs[start + 1]) >= float(margin):
            out[int(sorted_parts[start])].add(int(sorted_keys[start]))
        else:
            key = int(sorted_keys[start])
            for pos in range(start, end):
                out[int(sorted_parts[pos])].add(key)
    return out


def overlap_counts(pred_sets: list[set[int]], indices: list[int]) -> dict[int, tuple[int, int]]:
    all_keys: dict[int, int] = defaultdict(int)
    for idx in indices:
        for key in pred_sets[idx]:
            all_keys[key] += 1
    overlap_keys = {key for key, count in all_keys.items() if count > 1}
    return {idx: (len(pred_sets[idx] & overlap_keys), len(pred_sets[idx])) for idx in indices}


def iou_from_sets(pred: set[int], gt: set[int]) -> float:
    union = len(pred | gt)
    if union == 0:
        return 1.0
    return float(len(pred & gt) / union)


def update_stats(
    stats: dict[tuple[str, str], dict[str, float]],
    *,
    rule: str,
    labels: list[str],
    iou: float,
    pred_count: int,
    overlap_count: int,
    interface_total: int,
    interface_error: int,
) -> None:
    for size in labels:
        item = stats[(rule, size)]
        item["n"] += 1
        item["iou_sum"] += float(iou)
        item["success"] += 1 if float(iou) >= 0.5 else 0
        item["pred_voxels"] += int(pred_count)
        item["overlap_voxels"] += int(overlap_count)
        item["interface_total"] += int(interface_total)
        item["interface_error"] += int(interface_error)


def summarize_stats(stats: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in RULE_ORDER:
        for size in SIZE_ORDER:
            item = stats.get((rule, size))
            if not item or int(item["n"]) == 0:
                continue
            predcand = float(item["iou_sum"] / max(1, item["n"]))
            rows.append(
                {
                    "rule": rule,
                    "size": size,
                    "n": int(item["n"]),
                    "Predcand_IoU": predcand,
                    "delta_vs_0p7923": float(predcand - PER_PART_BASELINE),
                    "delta_vs_0p7727": float(predcand - PARTITION_BASELINE),
                    "overlap_rate": float(item["overlap_voxels"] / item["pred_voxels"]) if item["pred_voxels"] else 0.0,
                    "interface_err": float(item["interface_error"] / item["interface_total"]) if item["interface_total"] else float("nan"),
                    "success_at_iou0p5": float(item["success"] / max(1, item["n"])),
                    "pred_voxels": int(item["pred_voxels"]),
                    "overlap_voxels": int(item["overlap_voxels"]),
                    "interface_voxels": int(item["interface_total"]),
                }
            )
    return rows


def sample_array(coords: np.ndarray, *, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if coords.shape[0] > max_points:
        coords = coords[rng.choice(coords.shape[0], size=max_points, replace=False)]
    return coords


def color_for_part(index: int) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return palette[index % len(palette)]


def write_group_visual(
    path: Path,
    *,
    group: dict[str, Any],
    max_points_per_part: int = 2500,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rng = np.random.default_rng(20260622)
    fig = plt.figure(figsize=(13, 4), dpi=150)
    panels = [
        ("R0 independent", group["R0"]),
        ("R2b threshold argmax", group["R2b"]),
    ]
    for panel_idx, (title, pred_sets_by_part) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
        for local_idx, part in enumerate(group["parts"]):
            coords = keys_to_coords(pred_sets_by_part[part["idx"]])
            coords = sample_array(coords, max_points=max_points_per_part, rng=rng)
            if coords.size:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, alpha=0.55, c=color_for_part(local_idx))
        if title.startswith("R0"):
            overlap = keys_to_coords(group["R0_overlap_keys"])
            overlap = sample_array(overlap, max_points=5000, rng=rng)
            if overlap.size:
                ax.scatter(overlap[:, 0], overlap[:, 1], overlap[:, 2], s=4, alpha=0.8, c="#000000")
        ax.set_title(title)
        ax.set_xlim(0, 63)
        ax.set_ylim(0, 63)
        ax.set_zlim(0, 63)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=42)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    removed_coords = keys_to_coords(group["removed_claim_keys"])
    removed_coords = sample_array(removed_coords, max_points=8000, rng=rng)
    if removed_coords.size:
        ax.scatter(removed_coords[:, 0], removed_coords[:, 1], removed_coords[:, 2], s=3, alpha=0.8, c="#d62728", label="removed duplicate claims")
    ax.set_title("R0 - R2b removed")
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=42)
    if removed_coords.size:
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(group["title"], fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")

    print(f"[exclusion] loading {ckpt_path} on {device}", flush=True)
    model, empty_code, ckpt_args, semantic_vocab = load_model(ckpt_path, device)
    packed_dir = Path(str(ckpt_args.get("packed_dir") or "/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v4"))
    rows_json = json.loads((run_dir / "full_eval_rows.json").read_text(encoding="utf-8"))
    row_objs = [_dict_to_obj(row) for row in rows_json]
    dataset = PackedPromptablePartDataset(packed_dir, row_objs, semantic_vocab=semantic_vocab)
    sampler = PackedGroupBatchSampler(row_objs, batch_parts=int(args.batch_parts))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_promptable_parts,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[exclusion] groups packed into {len(sampler)} batches for {len(row_objs)} rows", flush=True)

    support_multiplier = float(ckpt_args.get("support_multiplier", 4.0))
    use_packed_whole_occ = bool(ckpt_args.get("use_packed_whole_occ", True))
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    part_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    visual_candidates: list[dict[str, Any]] = []
    processed = 0
    processed_groups = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            z_global = batch["z_global"].to(device=device, dtype=torch.float32)
            masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
            latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
            m_raw = batch["m_gt"].to(device=device, dtype=torch.float32)
            if str(ckpt_args.get("mask_target", "support")) == "support":
                m_gt, _, _ = latent_support_mask(latent_gt, empty_code, m_raw, multiplier=support_multiplier)
            else:
                m_gt = m_raw
            full_occ = full_occ_for_batch(
                None,
                z_global,
                batch,
                device=device,
                use_packed_whole_occ=use_packed_whole_occ,
            )
            out_gt = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(m_gt, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=0,
            )
            pred_m = (out_gt["m_logit"].sigmoid() > float(args.threshold)).float().view(m_gt.shape)
            out_pred = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(pred_m, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=0,
            )

            coords_list = [coords.detach().long().cpu() for coords in out_pred["voxel_coords"]]
            logits = out_pred["voxel_logits"]
            embeddings = out_pred.get("voxel_embeddings")
            partition_coords = partition_coords_by_embedding(logits, out_pred["voxel_coords"], embeddings, batch, threshold=float(args.threshold))
            r1_sets = pred_sets_from_partition(coords_list, partition_coords)

            group_to_indices: dict[str, list[int]] = defaultdict(list)
            for idx in range(z_global.shape[0]):
                group_to_indices[object_angle_key_from_batch(batch, idx)].append(idx)

            rows: list[dict[str, Any]] = []
            candidate_keys: list[np.ndarray] = []
            candidate_probs: list[np.ndarray] = []
            r0_sets: list[set[int]] = []
            gt_sets: list[set[int]] = []
            gt_tuple_sets: list[set[tuple[int, int, int]]] = []
            for idx in range(z_global.shape[0]):
                row = {
                    "dataset": str(batch["dataset_id"][idx]),
                    "source": dataset_source(str(batch["dataset_id"][idx])),
                    "object_id": str(batch["obj_id"][idx]),
                    "angle": int(batch["angle_idx"][idx]),
                    "part_name": str(batch["part_name"][idx]),
                    "semantic_type": str(batch["semantic_type"][idx]),
                    "raw_count": int(batch["raw_count"][idx].item()),
                }
                rows.append(row)
                coords_np = coords_list[idx].numpy().astype(np.int16, copy=False)
                keys = key_array(coords_np)
                probs = logits[idx, : keys.shape[0]].float().sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)
                candidate_keys.append(keys)
                candidate_probs.append(probs)
                r0_sets.append(set(keys[probs > float(args.threshold)].tolist()))
                raw_np = torch.as_tensor(batch["raw_coords"][idx], dtype=torch.long).cpu().numpy().astype(np.int16, copy=False)
                raw_keys = set(key_array(raw_np).tolist())
                gt_sets.append(raw_keys)
                gt_tuple_sets.append(coords_to_tuples(raw_np))

            for group_key, indices in group_to_indices.items():
                processed_groups += 1
                sibling_gt_tuple: dict[int, set[tuple[int, int, int]]] = {}
                for idx in indices:
                    siblings: set[tuple[int, int, int]] = set()
                    for other in indices:
                        if other != idx:
                            siblings |= gt_tuple_sets[other]
                    sibling_gt_tuple[idx] = siblings
                interface_sets = {
                    idx: interface_keys_for_part(gt_tuple_sets[idx], sibling_gt_tuple[idx])
                    for idx in indices
                }
                r2a_map = argmax_assign(candidate_keys, candidate_probs, indices)
                r2b_map = threshold_argmax_assign(candidate_keys, candidate_probs, r0_sets, indices, margin=None)
                r2c_map = threshold_argmax_assign(candidate_keys, candidate_probs, r0_sets, indices, margin=float(args.margin))
                rule_sets: dict[str, list[set[int]]] = {
                    "R0": r0_sets,
                    "R1": r1_sets,
                    "R2a": [r2a_map.get(idx, set()) if idx in indices else set() for idx in range(z_global.shape[0])],
                    "R2b": [r2b_map.get(idx, set()) if idx in indices else set() for idx in range(z_global.shape[0])],
                    "R2c": [r2c_map.get(idx, set()) if idx in indices else set() for idx in range(z_global.shape[0])],
                }
                overlap_by_rule = {
                    rule: overlap_counts(pred_sets, indices)
                    for rule, pred_sets in rule_sets.items()
                }
                for idx in indices:
                    labels = size_labels({**rows[idx], "raw_count": rows[idx]["raw_count"]})
                    iface = interface_sets[idx]
                    for rule in RULE_ORDER:
                        pred_set = rule_sets[rule][idx]
                        iou = iou_from_sets(pred_set, gt_sets[idx])
                        overlap_count, pred_count = overlap_by_rule[rule][idx]
                        iface_error = len(iface - pred_set)
                        update_stats(
                            stats,
                            rule=rule,
                            labels=labels,
                            iou=iou,
                            pred_count=pred_count,
                            overlap_count=overlap_count,
                            interface_total=len(iface),
                            interface_error=iface_error,
                        )
                        part_rows.append(
                            {
                                **rows[idx],
                                "rule": rule,
                                "size": raw_size_bucket({**rows[idx], "raw_count": rows[idx]["raw_count"]}),
                                "is_button": int("button" in labels),
                                "iou": iou,
                                "pred_count": pred_count,
                                "gt_count": len(gt_sets[idx]),
                                "overlap_count": overlap_count,
                                "overlap_rate": float(overlap_count / pred_count) if pred_count else 0.0,
                                "interface_voxels": len(iface),
                                "interface_error": iface_error,
                                "interface_err": float(iface_error / len(iface)) if iface else float("nan"),
                            }
                        )

                group_overlap: dict[str, float] = {}
                for rule in RULE_ORDER:
                    unique_claimed: set[int] = set()
                    duplicate_claimed: set[int] = set()
                    seen: set[int] = set()
                    for idx in indices:
                        for key in rule_sets[rule][idx]:
                            if key in seen:
                                duplicate_claimed.add(key)
                            seen.add(key)
                            unique_claimed.add(key)
                    group_overlap[rule] = float(len(duplicate_claimed) / len(unique_claimed)) if unique_claimed else 0.0
                interface_total = sum(len(interface_sets[idx]) for idx in indices)
                medium_large = any(raw_size_bucket({**rows[idx], "raw_count": rows[idx]["raw_count"]}) in ("medium", "large") for idx in indices)
                r0_overlap_keys: set[int] = set()
                seen_once: set[int] = set()
                for idx in indices:
                    for key in r0_sets[idx]:
                        if key in seen_once:
                            r0_overlap_keys.add(key)
                        seen_once.add(key)
                removed_claim_keys = set()
                for idx in indices:
                    removed_claim_keys |= r0_sets[idx] - r2b_map.get(idx, set())
                group_rows.append(
                    {
                        "group_key": group_key,
                        "dataset": rows[indices[0]]["dataset"],
                        "source": rows[indices[0]]["source"],
                        "object_id": rows[indices[0]]["object_id"],
                        "angle": rows[indices[0]]["angle"],
                        "parts": len(indices),
                        "interface_voxels": interface_total,
                        "R0_object_overlap_rate": group_overlap["R0"],
                        "R1_object_overlap_rate": group_overlap["R1"],
                        "R2a_object_overlap_rate": group_overlap["R2a"],
                        "R2b_object_overlap_rate": group_overlap["R2b"],
                        "R2c_object_overlap_rate": group_overlap["R2c"],
                        "removed_claim_voxels_R2b": len(removed_claim_keys),
                    }
                )
                if medium_large and interface_total > 0 and len(indices) >= 2:
                    score = group_overlap["R0"] + min(1.0, interface_total / 500.0) + min(1.0, len(removed_claim_keys) / 500.0)
                    visual_candidates.append(
                        {
                            "score": score,
                            "group_key": group_key,
                            "title": f"{rows[indices[0]]['source']} {rows[indices[0]]['object_id']} angle={rows[indices[0]]['angle']} parts={len(indices)} overlap={group_overlap['R0']:.3f}",
                            "parts": [{"idx": idx, **rows[idx]} for idx in indices],
                            "R0": {idx: set(r0_sets[idx]) for idx in indices},
                            "R2b": {idx: set(r2b_map.get(idx, set())) for idx in indices},
                            "R0_overlap_keys": r0_overlap_keys,
                            "removed_claim_keys": removed_claim_keys,
                        }
                    )

            processed += int(z_global.shape[0])
            if batch_idx % 20 == 0:
                print(
                    f"[exclusion] batch={batch_idx + 1}/{len(sampler)} rows={processed}/{len(row_objs)} "
                    f"groups={processed_groups} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
            if int(args.max_groups) > 0 and processed_groups >= int(args.max_groups):
                break

    summary_rows = summarize_stats(stats)
    write_csv(out_dir / "rule_size_summary.csv", summary_rows)
    write_csv(out_dir / "part_level_rows.csv", part_rows)
    write_csv(out_dir / "group_overlap_summary.csv", group_rows)

    visual_candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    visual_rows: list[dict[str, Any]] = []
    for vis_idx, group in enumerate(visual_candidates[: max(0, int(args.visuals))]):
        stem = f"{vis_idx:02d}_{group['group_key']}"
        stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)[:180]
        png = out_dir / "visuals" / f"{stem}.png"
        write_group_visual(png, group=group)
        visual_rows.append(
            {
                "rank": vis_idx,
                "group_key": group["group_key"],
                "score": group["score"],
                "parts": len(group["parts"]),
                "removed_claim_voxels": len(group["removed_claim_keys"]),
                "r0_overlap_voxels": len(group["R0_overlap_keys"]),
                "png": str(png),
            }
        )
    write_csv(out_dir / "visuals_manifest.csv", visual_rows)

    payload = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "out_dir": str(out_dir),
        "rows_processed": processed,
        "groups_processed": processed_groups,
        "threshold": float(args.threshold),
        "margin": float(args.margin),
        "batch_parts": int(args.batch_parts),
        "summary": summary_rows,
        "visuals": visual_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")

    print_table(
        "Exclusion rule x size",
        summary_rows,
        [
            "rule",
            "size",
            "n",
            "Predcand_IoU",
            "delta_vs_0p7923",
            "delta_vs_0p7727",
            "overlap_rate",
            "interface_err",
            "success_at_iou0p5",
        ],
    )
    print(f"\n[exclusion] wrote {out_dir}", flush=True)
    for row in visual_rows:
        print(row["png"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
