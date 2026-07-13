#!/usr/bin/env python3
"""Smoke-test joint single-owner part segmentation with body prompted when available."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

from inference_pipeline.voxel_io import save_voxel  # noqa: E402
from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    PackedPromptablePartDataset,
    collate_promptable_parts,
    dense_occ_from_coords,
)
from scripts.train.part_promptable_seg.train_part_promptable_seg import (  # noqa: E402
    PartRow,
    _build_joint_group_prediction,
    joint_eval_table,
    joint_seg_eval_rows,
    joint_seg_loss,
)
from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet  # noqa: E402


def _load_rows(selection_json: Path) -> list[dict[str, Any]]:
    rows = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"selection_json must contain a non-empty list: {selection_json}")
    return rows


def _rows_from_packed_index(packed_dir: Path, selection_rows: list[dict[str, Any]]) -> list[PartRow]:
    index = json.loads((Path(packed_dir) / "index.json").read_text(encoding="utf-8"))
    entries = index.get("entries", [])
    out: list[PartRow] = []
    for spec in selection_rows:
        matches = [
            entry
            for entry in entries
            if str(entry.get("obj_id")) == str(spec["obj_id"])
            and str(entry.get("dataset_id", "")) == str(spec.get("dataset_id", entry.get("dataset_id", "")))
            and int(entry.get("angle_idx")) == int(spec["angle_idx"])
            and str(entry.get("part_name")) == str(spec["part_name"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"selection spec matched {len(matches)} packed rows: {spec}")
        entry = matches[0]
        out.append(
            PartRow(
                sample_idx=int(entry.get("sample_idx", 0)),
                part_idx=int(entry.get("part_idx", len(out))),
                obj_id=str(entry["obj_id"]),
                angle_idx=int(entry["angle_idx"]),
                sample_id=str(entry.get("sample_id", f"{entry['obj_id']}_angle_{entry['angle_idx']}")),
                part_name=str(entry["part_name"]),
                semantic_type=str(entry.get("semantic_type", str(entry["part_name"]).split("_")[0])),
                original_label=int(entry.get("original_label", entry.get("part_label", 0))),
                raw_count=int(entry.get("raw_count", 0)),
                view_indices=tuple(int(v) for v in entry.get("view_indices", (0, 1, 2, 3))),
                dataset_id=str(entry.get("dataset_id", "")),
                data_root=str(entry.get("data_root", "")),
                manifest_path=str(entry.get("manifest_path", "")),
                category=str(entry.get("category", "")),
                object_name=str(entry.get("object_name", "")),
                part_item_name=str(entry.get("part_item_name", "")),
                part_joint=str(entry.get("part_joint", "")),
                sample_part_names=str(entry.get("sample_part_names", "")),
                visible_view_count=int(entry.get("visible_view_count", 0)),
            )
        )
    return out


def _class_npz_export(
    out_dir: Path,
    pred: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = pred["coords"].detach().cpu().numpy().astype(np.int32)
    labels = pred["class_logits"].detach().float().argmax(dim=1).cpu().numpy().astype(np.int64)
    meta = pred["class_meta"]
    paths: dict[str, str] = {}
    sets: list[set[int]] = []
    label_map: dict[str, Any] = {"body": 1, "parts": {}}
    for class_idx, item in enumerate(meta):
        selected = coords[labels == int(class_idx)]
        if class_idx == 0:
            path = out_dir / "part_body_voxel.npz"
            save_voxel(out_dir, selected, resolution=64, source="joint_body_smoke", basename="part_body_voxel")
            label_map["body_file"] = str(path)
        else:
            part_name = str(item["name"])
            path = out_dir / f"part_{class_idx - 1:02d}_voxel.npz"
            np.savez_compressed(
                path,
                coords=selected.astype(np.int32),
                resolution=np.int32(64),
                coord_frame="canonical_grid",
                source="joint_body_smoke",
                part_index=np.int32(class_idx - 1),
                target_part_name=part_name,
            )
            label_map["parts"][str(class_idx)] = {
                "part_name": part_name,
                "render_label": int(item.get("render_label", class_idx + 1)),
            }
        paths[str(item["name"])] = str(path)
        keys = selected[:, 0].astype(np.int64) * 4096 + selected[:, 1].astype(np.int64) * 64 + selected[:, 2].astype(np.int64)
        sets.append(set(int(v) for v in keys.tolist()))
    overlap = 0
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            overlap += len(a & b)
    return {"paths": paths, "pairwise_overlap": int(overlap), "label_map": label_map}


def _joint_groups_from_batch(batch: dict[str, Any]) -> list[list[int]]:
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for idx, (dataset_id, obj_id, angle_idx) in enumerate(
        zip(batch["dataset_id"], batch["obj_id"], batch["angle_idx"])
    ):
        key = f"{dataset_id}::{obj_id}|angle_{int(angle_idx)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(idx))
    return [groups[key] for key in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-max-tokens", "--voxel-cap", dest="voxel_max_tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--body-class-weight", type=float, default=0.25)
    parser.add_argument("--joint-kmax", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.voxel_max_tokens) != 0:
        raise ValueError("joint-body smoke/probe uses full shared candidate S; pass --voxel-max-tokens 0")
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    rows = _rows_from_packed_index(args.packed_dir, _load_rows(args.selection_json))
    dataset = PackedPromptablePartDataset(args.packed_dir, rows)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_promptable_parts)
    batch = next(iter(loader))
    model = PromptablePartLatentSegNet(
        dim=256,
        depth=6,
        head_depth=2,
        heads=8,
        use_voxel_head=True,
        voxel_depth=3,
        mask_encoder="cnn_grid",
        voxel_embedding_dim=16,
        use_body_prompt=True,
        use_checkpoint=bool(args.use_checkpoint),
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    z_global = batch["z_global"].to(device=device, dtype=torch.float32)
    masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
    full_occ = dense_occ_from_coords(batch["whole_coords"], device=device)
    train_started = time.perf_counter()
    optim.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available() and str(device).startswith("cuda")):
        loss, items = joint_seg_loss(
            model,
            z_global=z_global,
            masks2d=masks2d,
            full_occ=full_occ,
            batch=batch,
            device=device,
            voxel_max_tokens=int(args.voxel_max_tokens),
            body_class_weight=float(args.body_class_weight),
            joint_kmax=int(args.joint_kmax),
            small_part_threshold=32,
            small_part_weight=0.25,
        )
    loss.backward()
    if float(args.grad_clip) > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
    else:
        grads = [p.grad.detach().float().norm(2) for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.vector_norm(torch.stack(grads), ord=2) if grads else loss.detach().new_tensor(0.0)
    optim.step()
    train_step_seconds = time.perf_counter() - train_started
    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"joint smoke loss is not finite: {float(loss.detach().item())}")

    model.eval()
    eval_rows, eval_meta = joint_seg_eval_rows(
        model,
        z_global=z_global,
        masks2d=masks2d,
        full_occ=full_occ,
        batch=batch,
        device=device,
        voxel_max_tokens=int(args.voxel_max_tokens),
        body_class_weight=float(args.body_class_weight),
        joint_kmax=int(args.joint_kmax),
        small_part_threshold=32,
        small_part_weight=0.25,
    )
    table = joint_eval_table(
        step=1,
        lr=1.0e-4,
        loss_total=float(loss.detach().item()),
        loss_joint_ce=float(items["joint_ce"]),
        grad_norm=float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
        train_rows=eval_rows,
        heldout_rows=eval_rows,
    )
    (args.out_dir / "joint_eval_table.txt").write_text(table + "\n", encoding="utf-8")

    groups = _joint_groups_from_batch(batch)
    export_indices = groups[0]
    pred = _build_joint_group_prediction(
        model,
        z_global=z_global,
        masks2d=masks2d,
        full_occ=full_occ,
        batch=batch,
        indices=export_indices,
        device=device,
        voxel_max_tokens=int(args.voxel_max_tokens),
        joint_kmax=int(args.joint_kmax),
        body_class_weight=float(args.body_class_weight),
        small_part_threshold=32,
        small_part_weight=0.25,
        subsample_parts=False,
    )
    npz = _class_npz_export(args.out_dir / "parts", pred)
    body_modes = [str(mode) for mode in eval_meta.get("body_modes", [])]
    if not body_modes:
        body_mode = "unknown"
    elif len(body_modes) == 1:
        body_mode = body_modes[0]
    else:
        body_mode = "mixed:" + ",".join(body_modes)

    ckpt_path = args.out_dir / "ckpts" / "latest.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_args = {
        "route": "voxel",
        "joint_seg": True,
        "dim": 256,
        "depth": 6,
        "head_depth": 2,
        "heads": 8,
        "voxel_depth": 3,
        "mask_encoder": "cnn_grid",
        "voxel_embedding_dim": 16,
        "body_class_weight": float(args.body_class_weight),
        "voxel_max_tokens": int(args.voxel_max_tokens),
        "infer_voxel_max_tokens": 0,
    }
    torch.save(
        {
            "step": 1,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": None,
            "args": ckpt_args,
            "empty_code": torch.zeros((8, 16, 16, 16), dtype=torch.float32),
            "metadata": {"body_mode": body_mode, "smoke": True},
        },
        ckpt_path,
    )
    reloaded = PromptablePartLatentSegNet(
        dim=256,
        depth=6,
        head_depth=2,
        heads=8,
        use_voxel_head=True,
        voxel_depth=3,
        mask_encoder="cnn_grid",
        voxel_embedding_dim=16,
        use_body_prompt=True,
    )
    reloaded.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"], strict=True)

    peak_gb = (
        float(torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024 ** 3))
        if torch.cuda.is_available() and str(device).startswith("cuda")
        else 0.0
    )
    report = {
        "body_mode": body_mode,
        "base_link_has_mask": "prompted-part" in set(body_modes),
        "rows": [row.__dict__ for row in rows],
        "part_names": [row.part_name for row in rows],
        "joint_items": items,
        "loss": float(loss.detach().item()),
        "no_nan": bool(torch.isfinite(loss.detach()).item()),
        "grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
        "grad_clip": float(args.grad_clip),
        "clip_enabled": bool(float(args.grad_clip) > 0),
        "peak_gb": peak_gb,
        "train_step_seconds": float(train_step_seconds),
        "step_per_s": float(1.0 / max(train_step_seconds, 1.0e-12)),
        "voxel_max_tokens": int(args.voxel_max_tokens),
        "voxel_cap": int(args.voxel_max_tokens),
        "batch_size": int(args.batch_size),
        "checkpoint_enabled": bool(args.use_checkpoint),
        "eval_meta": eval_meta,
        "eval_rows": eval_rows,
        "export_group_indices": export_indices,
        "export_group_obj": {
            "dataset_id": str(batch["dataset_id"][export_indices[0]]),
            "obj_id": str(batch["obj_id"][export_indices[0]]),
            "angle_idx": int(batch["angle_idx"][export_indices[0]]),
        },
        "argmax_has_body": bool(eval_meta["argmax_has_body"]),
        "npz_pairwise_disjoint": int(npz["pairwise_overlap"]) == 0,
        "npz_pairwise_overlap": int(npz["pairwise_overlap"]),
        "npz_paths": npz["paths"],
        "label_map": npz["label_map"],
        "ckpt": str(ckpt_path),
        "ckpt_strict_load": True,
        "seconds": round(time.time() - started, 3),
    }
    (args.out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
