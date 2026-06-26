from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")


def _setup_trellis_imports() -> None:
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines", "renderers"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules.setdefault(f"trellis.{sp}", mod)


_setup_trellis_imports()

import inference as trellis_inference  # noqa: E402
from inference_pipeline import inputs_materialize  # noqa: E402
from inference_pipeline.data_config_io import load_data_config  # noqa: E402
from inference_pipeline.object_inputs import _dataset_for  # noqa: E402
from inference_pipeline.part_flow_stage import save_part_voxels  # noqa: E402
from inference_pipeline.part_prompt_seg_stage import (  # noqa: E402
    _dense_occ_from_voxel_npz,
    _load_part_masks2d,
    _load_prompt_seg_model,
    _mask_morphology,
)
from inference_pipeline.voxel_io import save_voxel  # noqa: E402
from part_ss_eval_platform.eval_0615 import (  # noqa: E402
    BUCKETS,
    DEFAULT_DATA_CONFIG,
    DEFAULT_PART_SEG_CKPT,
    DEFAULT_SPLIT_JSON,
    DEFAULT_SS_DECODER_CKPT,
    DEFAULT_SS_FLOW_CKPT,
    part_bucket,
    render_table_png,
    size_bucket,
)
from part_ss_eval_platform.render_real_block_voxels import _draw_block_projection  # noqa: E402


DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0615-cfg")
DEFAULT_SS_FLOW_EMA2_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.9999_step0020000.pt"
)

LARGE_KEYWORDS = (
    "box",
    "carton",
    "cabinet",
    "drawer",
    "door",
    "lid",
    "flap",
    "panel",
    "shelf",
)
SMALL_KEYWORDS = ("button", "knob", "handle", "switch", "key")
SELECTION_BUCKETS = ("large", "medium", "small", "tiny", "button")
PART_COLORS = np.asarray(
    [
        (0.86, 0.16, 0.16, 0.96),
        (0.12, 0.38, 0.72, 0.96),
        (0.18, 0.62, 0.25, 0.96),
        (0.95, 0.49, 0.12, 0.96),
        (0.50, 0.28, 0.72, 0.96),
        (0.10, 0.66, 0.72, 0.96),
        (0.86, 0.36, 0.66, 0.96),
        (0.55, 0.32, 0.24, 0.96),
        (0.72, 0.67, 0.18, 0.96),
        (0.25, 0.50, 0.52, 0.96),
    ],
    dtype=np.float32,
)
BODY_RGBA = (150, 150, 150, 54)
GT_RGBA = (44, 44, 44, 210)


@dataclass(frozen=True)
class CrfSample:
    split: str
    obj_id: str
    angle_idx: int
    sample_bucket: str
    priority_bucket: str
    part_count: int
    min_raw_voxels: int
    max_raw_voxels: int
    has_button: bool
    has_large_keyword: bool
    selected_reason: str = ""


class VramSampler:
    def __init__(self, gpu: str, interval: float = 0.5) -> None:
        self.gpu = str(gpu).split(",")[0]
        self.interval = float(interval)
        self.max_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self.max_mib = max(self.max_mib, value)
            self._stop.wait(self.interval)

    def _query(self) -> int | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", self.gpu],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        values: list[int] = []
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip()))
            except ValueError:
                pass
        return max(values) if values else None


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _valid_coords(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    valid = np.all((coords >= 0) & (coords < resolution), axis=1)
    coords = coords[valid]
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(coords, axis=0)


def _coords_to_occ(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    occ = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = _valid_coords(coords, resolution)
    if coords.size:
        occ[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occ


def _occ_to_coords(occ: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(occ, dtype=bool)).astype(np.int32)


def _load_npz_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["coords"], dtype=np.int64).reshape(-1, 3)


def _sample_keyword_score(parts: list[dict[str, Any]]) -> tuple[bool, bool, int]:
    names = " ".join(str(part.get("part_name", "")).lower() for part in parts)
    has_large = any(key in names for key in LARGE_KEYWORDS)
    has_small = any(key in names for key in SMALL_KEYWORDS)
    score = sum(3 for key in LARGE_KEYWORDS if key in names) + sum(2 for key in SMALL_KEYWORDS if key in names)
    return has_large, has_small, score


def _sample_part_stats(ds: Any, sample: dict[str, Any]) -> dict[str, Any]:
    buckets: Counter[str] = Counter()
    raw_counts: list[int] = []
    has_button = False
    for part in sample["parts"]:
        raw = ds._load_raw_ind_coords(sample, part)
        raw_count = int(raw.shape[0])
        raw_counts.append(raw_count)
        bucket = part_bucket(str(part["part_name"]), part, raw_count)
        buckets[bucket] += 1
        has_button = has_button or bucket == "button"
    has_large_kw, has_small_kw, kw_score = _sample_keyword_score(sample["parts"])
    if has_button:
        sample_bucket = "button"
    elif raw_counts:
        sample_bucket = size_bucket(min(raw_counts))
    else:
        sample_bucket = "tiny"
    if buckets["large"] > 0 or has_large_kw:
        priority_bucket = "large"
    elif has_button:
        priority_bucket = "button"
    else:
        priority_bucket = sample_bucket
    return {
        "sample_bucket": sample_bucket,
        "priority_bucket": priority_bucket,
        "part_buckets": dict(buckets),
        "min_raw_voxels": int(min(raw_counts) if raw_counts else 0),
        "max_raw_voxels": int(max(raw_counts) if raw_counts else 0),
        "total_raw_voxels": int(sum(raw_counts)),
        "has_button": bool(has_button),
        "has_large_keyword": bool(has_large_kw),
        "has_small_keyword": bool(has_small_kw),
        "keyword_score": int(kw_score),
        "part_count": int(len(sample["parts"])),
    }


def _best_angle_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    def score(rec: dict[str, Any]) -> tuple[int, int, int, int, int]:
        st = rec["stats"]
        return (
            int(st["has_large_keyword"]),
            int(st["part_count"]),
            int(st["keyword_score"]),
            int(st["max_raw_voxels"]),
            -int(rec["angle_idx"]),
        )

    return max(records, key=score)


def build_unique_selection(
    ds: Any,
    split_json: Path,
    *,
    train_count: int = 85,
    val_count: int = 43,
) -> tuple[list[CrfSample], dict[str, Any]]:
    split_data = _read_json(split_json)
    split_ids = {
        "train": {str(x) for x in split_data["train_ids"]},
        "val": {str(x) for x in split_data["heldout_ids"]},
    }
    by_obj: dict[str, dict[str, list[dict[str, Any]]]] = {"train": defaultdict(list), "val": defaultdict(list)}
    for sample in ds.samples:
        obj_id = str(sample["obj_id"])
        split = "train" if obj_id in split_ids["train"] else "val" if obj_id in split_ids["val"] else ""
        if not split:
            continue
        stats = _sample_part_stats(ds, sample)
        by_obj[split][obj_id].append(
            {
                "split": split,
                "obj_id": obj_id,
                "angle_idx": int(sample["angle_idx"]),
                "stats": stats,
            }
        )

    obj_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for obj_id, records in by_obj[split].items():
            best = _best_angle_record(records)
            obj_records[split].append(best)

    quotas = {
        "train": {"large": 36, "medium": 16, "small": 12, "tiny": 8, "button": 13},
        "val": {"large": 18, "medium": 8, "small": 6, "tiny": 5, "button": 6},
    }
    targets = {"train": int(train_count), "val": int(val_count)}
    notes: list[str] = []
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    selected_keys: dict[str, set[str]] = {"train": set(), "val": set()}

    def priority_bucket(record: dict[str, Any]) -> str:
        return str(record["stats"]["priority_bucket"])

    def record_sort(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
        st = record["stats"]
        return (
            -int(st["has_large_keyword"]),
            -int(st["part_count"]),
            -int(st["keyword_score"]),
            -int(st["max_raw_voxels"]),
            str(record["obj_id"]),
        )

    for split in ("train", "val"):
        records = sorted(obj_records[split], key=record_sort)
        for bucket in SELECTION_BUCKETS:
            need = int(quotas[split].get(bucket, 0))
            bucket_records = [rec for rec in records if priority_bucket(rec) == bucket]
            for rec in bucket_records:
                if len(selected[split]) >= targets[split] or need <= 0:
                    break
                if rec["obj_id"] in selected_keys[split]:
                    continue
                selected[split].append({**rec, "selected_reason": f"quota:{bucket}"})
                selected_keys[split].add(str(rec["obj_id"]))
                need -= 1
            if need > 0:
                notes.append(f"{split}/{bucket} quota short by {need}")
        remaining = [rec for rec in records if str(rec["obj_id"]) not in selected_keys[split]]
        for rec in remaining:
            if len(selected[split]) >= targets[split]:
                break
            selected[split].append({**rec, "selected_reason": "fill:large_multipart_priority"})
            selected_keys[split].add(str(rec["obj_id"]))

    rows: list[CrfSample] = []
    for split in ("train", "val"):
        if len(selected[split]) < targets[split]:
            notes.append(f"{split} selected {len(selected[split])}/{targets[split]}")
        selected[split].sort(key=lambda rec: (priority_bucket(rec), str(rec["obj_id"])))
        for rec in selected[split]:
            st = rec["stats"]
            rows.append(
                CrfSample(
                    split=split,
                    obj_id=str(rec["obj_id"]),
                    angle_idx=int(rec["angle_idx"]),
                    sample_bucket=str(st["sample_bucket"]),
                    priority_bucket=str(st["priority_bucket"]),
                    part_count=int(st["part_count"]),
                    min_raw_voxels=int(st["min_raw_voxels"]),
                    max_raw_voxels=int(st["max_raw_voxels"]),
                    has_button=bool(st["has_button"]),
                    has_large_keyword=bool(st["has_large_keyword"]),
                    selected_reason=str(rec.get("selected_reason", "")),
                )
            )
    manifest = {
        "split_json": str(split_json),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "unique_obj": True,
        "one_angle_per_obj": True,
        "selection_policy": (
            "object-level dedup; choose one angle per object by large-part keyword, part_count, "
            "keyword score, max raw voxels; quotas prioritize large/multipart objects while retaining tiny/button cases"
        ),
        "quotas": quotas,
        "notes": notes,
        "counts": {
            split: {
                "objects": len([s for s in rows if s.split == split]),
                "sample_bucket": Counter(s.sample_bucket for s in rows if s.split == split),
                "priority_bucket": Counter(s.priority_bucket for s in rows if s.split == split),
                "has_button": sum(int(s.has_button) for s in rows if s.split == split),
                "has_large_keyword": sum(int(s.has_large_keyword) for s in rows if s.split == split),
            }
            for split in ("train", "val")
        },
        "samples": {split: [asdict(s) for s in rows if s.split == split] for split in ("train", "val")},
    }
    return rows, manifest


def load_selection_manifest(path: Path) -> tuple[list[CrfSample], dict[str, Any]]:
    manifest = _read_json(path)
    rows: list[CrfSample] = []
    for split in ("train", "val"):
        for item in manifest.get("samples", {}).get(split, []):
            payload = dict(item)
            payload.setdefault("priority_bucket", payload.get("sample_bucket", ""))
            rows.append(CrfSample(**payload))
    if not rows:
        raise ValueError(f"selection manifest has no samples: {path}")
    return rows, manifest


def _find_sample(ds: Any, obj_id: str, angle_idx: int) -> dict[str, Any]:
    for sample in ds.samples:
        if str(sample["obj_id"]) == str(obj_id) and int(sample["angle_idx"]) == int(angle_idx):
            return sample
    raise KeyError(f"sample not found: {obj_id} angle={angle_idx}")


def _run_dir(out_dir: Path, sample: CrfSample) -> Path:
    return out_dir / "platform_runs" / sample.split / sample.obj_id


def _load_official_tokens(ds: Any, sample: dict[str, Any]) -> np.ndarray:
    token_path = (
        Path(ds.data_root)
        / ds.recon_subdir
        / "dinov2_tokens_official_prenorm1374"
        / str(sample["obj_id"])
        / f"angle_{int(sample['angle_idx'])}"
        / "tokens.npz"
    )
    if not token_path.is_file():
        raise FileNotFoundError(f"official DINO tokens not found: {token_path}")
    with np.load(token_path, allow_pickle=False) as data:
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    view_indices = [int(v) for v in sample["view_indices"]]
    if len(view_indices) != 4:
        raise ValueError(f"{sample['obj_id']} angle={sample['angle_idx']} expected 4 views, got {view_indices}")
    if tokens.ndim != 3 or tokens.shape[1:] != (1374, 1024):
        raise ValueError(f"{token_path} expected [V,1374,1024], got {tokens.shape}")
    if min(view_indices) < 0 or max(view_indices) >= tokens.shape[0]:
        raise ValueError(f"{token_path} cannot select {view_indices} from {tokens.shape}")
    return np.ascontiguousarray(tokens[view_indices])


def run_ss_stage(
    ds: Any,
    data_config: dict[str, Any],
    ds_sample: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file() and not args.overwrite:
        return np.load(run_dir / "ss_latent.npy").astype(np.float32), _load_npz_coords(run_dir / "voxel.npz")

    print(f"[eval_crf_0615] ss materialize {ds_sample['obj_id']} angle={ds_sample['angle_idx']}", flush=True)
    inputs_materialize.materialize(
        run_dir,
        data_config,
        object_id=str(ds_sample["obj_id"]),
        angle_idx=int(ds_sample["angle_idx"]),
        view_indices=[int(v) for v in ds_sample["view_indices"]],
    )
    tokens = _load_official_tokens(ds, ds_sample)
    print(f"[eval_crf_0615] ss flow start {ds_sample['obj_id']} tokens={tokens.shape}", flush=True)
    z_global = trellis_inference.run_ss_flow_from_tokens(
        tokens,
        str(args.ss_flow_ckpt),
        num_steps=int(args.ss_steps),
        cfg_strength=float(args.ss_cfg),
        seed=int(args.ss_seed),
    )
    z_np = z_global.detach().float().cpu().numpy().astype(np.float32)
    if z_np.shape != (8, 16, 16, 16):
        raise ValueError(f"SS latent shape invalid: {z_np.shape}")
    np.save(run_dir / "ss_latent.npy", z_np)
    print(f"[eval_crf_0615] ss decode start {ds_sample['obj_id']}", flush=True)
    coords = trellis_inference.decode_ss(z_global, str(args.ss_decoder_ckpt), threshold=float(args.decode_threshold))
    whole_coords = coords.numpy().astype(np.int32)
    save_voxel(run_dir, whole_coords, resolution=64, source="trellis_ss_flow_ema2_cfg")
    if bool(getattr(args, "clear_ss_cache", False)):
        try:
            trellis_inference._load_ss_flow.cache_clear()
            trellis_inference._load_ss_decoder.cache_clear()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    meta = {
        "object_id": str(ds_sample["obj_id"]),
        "angle_idx": int(ds_sample["angle_idx"]),
        "view_indices": [int(v) for v in ds_sample["view_indices"]],
        "ss_flow_ckpt": str(args.ss_flow_ckpt),
        "ss_steps": int(args.ss_steps),
        "ss_cfg": float(args.ss_cfg),
        "ss_seed": int(args.ss_seed),
        "token_source": "dinov2_tokens_official_prenorm1374",
        "decode_threshold": float(args.decode_threshold),
        "whole_voxels": int(whole_coords.shape[0]),
    }
    _write_json(run_dir / "ss_meta.json", meta)
    print(f"[eval_crf_0615] ss done {ds_sample['obj_id']} whole_voxels={whole_coords.shape[0]}", flush=True)
    return z_np, whole_coords


def _dense_part_prob_from_output(out: dict[str, Any]) -> np.ndarray:
    dense = np.zeros((64, 64, 64), dtype=np.float32)
    logits = out["voxel_logits"][0].float().sigmoid()
    pad_mask = out["voxel_pad_mask"][0].bool()
    coords = out["voxel_coords"][0].long()
    valid_len = min(coords.shape[0], logits.shape[0], pad_mask.shape[0])
    keep = ~pad_mask[:valid_len]
    if bool(keep.any()):
        picked_coords = coords[:valid_len][keep].detach().cpu().numpy().astype(np.int64)
        picked_prob = logits[:valid_len][keep].detach().cpu().numpy().astype(np.float32)
        valid = np.all((picked_coords >= 0) & (picked_coords < 64), axis=1)
        picked_coords = picked_coords[valid]
        picked_prob = picked_prob[valid]
        dense[picked_coords[:, 0], picked_coords[:, 1], picked_coords[:, 2]] = picked_prob
    return dense


def run_part_probabilities(
    ds: Any,
    data_config: dict[str, Any],
    ds_sample: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    baseline_npz_dir = run_dir / "baseline_parts"
    prob_path = run_dir / "part_probs.npz"
    if prob_path.is_file() and baseline_npz_dir.is_dir() and not args.overwrite:
        with np.load(prob_path, allow_pickle=True) as data:
            names = [str(x) for x in data["part_names"].tolist()]
            probs = np.asarray(data["probs"], dtype=np.float32)
            full_occ = np.asarray(data["full_occ"], dtype=bool)
        return names, probs, full_occ

    print(f"[eval_crf_0615] part prob start {ds_sample['obj_id']} parts={len(ds_sample['parts'])}", flush=True)
    z_global = torch.from_numpy(np.load(run_dir / "ss_latent.npy")).float().unsqueeze(0).cuda()
    model, _empty_code, ckpt_args = _load_prompt_seg_model(str(args.part_seg_ckpt))
    route = str(ckpt_args.get("route", "latent"))
    if route != "voxel":
        raise ValueError(f"CRF eval requires route=voxel part-seg ckpt, got {route}")
    full_occ_t = _dense_occ_from_voxel_npz(run_dir / "voxel.npz", device=z_global.device)
    full_occ = full_occ_t[0, 0].detach().cpu().numpy().astype(bool)
    target_names = [str(part["part_name"]) for part in ds_sample["parts"]]
    probs: list[np.ndarray] = []
    baseline_coords: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for part in ds_sample["parts"]:
            name = str(part["part_name"])
            print(f"[eval_crf_0615] part prob {ds_sample['obj_id']} {name}", flush=True)
            masks2d = _load_part_masks2d(ds, ds_sample, part).unsqueeze(0).cuda()
            out_cell = model(
                z_global,
                masks2d,
                candidate_cells=torch.ones((1, 16, 16, 16), dtype=torch.float32, device=z_global.device),
                full_occ=full_occ_t,
            )
            pred_m = (out_cell["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
            out_voxel = model(
                z_global,
                masks2d,
                candidate_cells=_mask_morphology(pred_m, "dilate"),
                full_occ=full_occ_t,
            )
            prob = _dense_part_prob_from_output(out_voxel)
            prob[~full_occ] = 0.0
            probs.append(prob.astype(np.float32, copy=False))
            baseline_coords[name] = _occ_to_coords(prob > float(args.part_threshold))
    probs_arr = np.stack(probs, axis=0).astype(np.float32, copy=False) if probs else np.zeros((0, 64, 64, 64), dtype=np.float32)
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prob_path,
        part_names=np.asarray(target_names, dtype=str),
        probs=probs_arr.astype(np.float16, copy=False),
        full_occ=full_occ.astype(np.uint8),
        part_seg_ckpt=str(args.part_seg_ckpt),
        part_threshold=np.float32(args.part_threshold),
    )
    if baseline_npz_dir.exists():
        shutil.rmtree(baseline_npz_dir)
    save_part_voxels(baseline_npz_dir, baseline_coords, target_part_names=target_names, resolution=64)
    print(f"[eval_crf_0615] part prob done {ds_sample['obj_id']}", flush=True)
    return target_names, probs_arr, full_occ


def _shift_bool(arr: np.ndarray, axis: int, direction: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=bool)
    if direction > 0:
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(0, -1)
        dst[axis] = slice(1, None)
    else:
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(1, None)
        dst[axis] = slice(0, -1)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _binary_erosion6(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    eroded = mask.copy()
    for axis in range(3):
        eroded &= _shift_bool(mask, axis, 1)
        eroded &= _shift_bool(mask, axis, -1)
    return eroded


def _binary_dilation6(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    base = out
    for _ in range(int(radius)):
        grown = out.copy()
        for axis in range(3):
            grown |= _shift_bool(out, axis, 1)
            grown |= _shift_bool(out, axis, -1)
        out = grown
    return out if radius > 0 else base


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~_binary_erosion6(mask)


def _signed_distance(occ: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    occ = np.asarray(occ, dtype=bool)
    if not bool(occ.any()):
        return np.zeros_like(occ, dtype=np.float32)
    inside = ndimage.distance_transform_edt(occ).astype(np.float32)
    outside = ndimage.distance_transform_edt(~occ).astype(np.float32)
    return inside - outside


def _edge_weights_from_occ(whole_occ: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    sdf = _signed_distance(whole_occ)
    grads = np.gradient(sdf)
    grad_mag = np.sqrt(sum(g.astype(np.float32) ** 2 for g in grads)) + 1.0e-6
    nx, ny, nz = [g.astype(np.float32) / grad_mag for g in grads]
    weights: dict[str, np.ndarray] = {}
    for axis, key in enumerate(("x", "y", "z")):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(0, -1)
        slicer_b[axis] = slice(1, None)
        a = tuple(slicer_a)
        b = tuple(slicer_b)
        dist_jump = np.abs(sdf[a] - sdf[b])
        normal_dot = nx[a] * nx[b] + ny[a] * ny[b] + nz[a] * nz[b]
        normal_sim = np.clip((normal_dot + 1.0) * 0.5, 0.0, 1.0)
        both_occ = whole_occ[a] & whole_occ[b]
        weights[key] = (np.exp(-float(alpha) * dist_jump) * (0.25 + 0.75 * normal_sim) * both_occ).astype(np.float32)
    return weights


def apply_crf(
    probs: np.ndarray,
    whole_occ: np.ndarray,
    *,
    threshold: float,
    pairwise_weight: float,
    edge_alpha: float,
    iterations: int,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    whole_occ = np.asarray(whole_occ, dtype=bool)
    k = int(probs.shape[0])
    labels = np.zeros((64, 64, 64), dtype=np.int16)
    if k == 0 or not bool(whole_occ.any()):
        return labels
    probs = np.clip(probs, 1.0e-5, 1.0 - 1.0e-5)
    max_part = probs.max(axis=0)
    bg_prob = np.clip(1.0 - max_part, 1.0e-5, 1.0 - 1.0e-5)
    unary = np.concatenate([(-np.log(bg_prob))[None], -np.log(probs)], axis=0).astype(np.float32)
    unary[0, whole_occ] += 0.18
    unary[1:, ~whole_occ] += 12.0
    unary[0, ~whole_occ] = 0.0

    baseline_any = (max_part > float(threshold)) & whole_occ
    winner = np.argmax(probs, axis=0).astype(np.int16) + 1
    labels[baseline_any] = winner[baseline_any]
    weights = _edge_weights_from_occ(whole_occ, float(edge_alpha))

    num_labels = k + 1
    current = labels
    for _ in range(int(iterations)):
        costs = unary.copy()
        for label in range(num_labels):
            same = current == label
            vote = np.zeros((64, 64, 64), dtype=np.float32)
            sx = same[:-1, :, :]
            wx = weights["x"]
            vote[1:, :, :] += sx * wx
            vote[:-1, :, :] += same[1:, :, :] * wx
            sy = same[:, :-1, :]
            wy = weights["y"]
            vote[:, 1:, :] += sy * wy
            vote[:, :-1, :] += same[:, 1:, :] * wy
            sz = same[:, :, :-1]
            wz = weights["z"]
            vote[:, :, 1:] += sz * wz
            vote[:, :, :-1] += same[:, :, 1:] * wz
            costs[label] -= float(pairwise_weight) * vote
        updated = np.argmin(costs, axis=0).astype(np.int16)
        updated[~whole_occ] = 0
        updated[(updated > 0) & (max_part <= float(threshold) * 0.45)] = 0
        if np.array_equal(updated, current):
            current = updated
            break
        current = updated
    return current


def _coords_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int, int]:
    pred = _valid_coords(pred)
    gt = _valid_coords(gt)
    pred_set = set(map(tuple, pred.tolist()))
    gt_set = set(map(tuple, gt.tolist()))
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return (float(inter / union) if union else 1.0, int(inter), int(union), int(len(pred_set)))


def _boundary_iou(pred_occ: np.ndarray, gt_occ: np.ndarray, radius: int) -> float:
    pred_occ = np.asarray(pred_occ, dtype=bool)
    gt_occ = np.asarray(gt_occ, dtype=bool)
    pred_band = _binary_dilation6(_boundary(pred_occ), int(radius))
    gt_band = _binary_dilation6(_boundary(gt_occ), int(radius))
    pred_b = pred_occ & (pred_band | gt_band)
    gt_b = gt_occ & (pred_band | gt_band)
    inter = int((pred_b & gt_b).sum())
    union = int((pred_b | gt_b).sum())
    return float(inter / union) if union else 1.0


def _overlap_count(part_occs: list[np.ndarray]) -> int:
    if not part_occs:
        return 0
    counts = np.zeros((64, 64, 64), dtype=np.uint16)
    for occ in part_occs:
        counts += np.asarray(occ, dtype=bool)
    return int((counts > 1).sum())


def _label_part_occs(labels: np.ndarray, k: int) -> list[np.ndarray]:
    return [np.asarray(labels == (idx + 1), dtype=bool) for idx in range(k)]


def _save_part_npzs(root: Path, part_names: list[str], part_occs: list[np.ndarray]) -> None:
    if root.exists():
        shutil.rmtree(root)
    coords = {name: _occ_to_coords(occ) for name, occ in zip(part_names, part_occs)}
    save_part_voxels(root, coords, target_part_names=part_names, resolution=64)


def evaluate_sample(
    ds: Any,
    data_config: dict[str, Any],
    sample: CrfSample,
    args: argparse.Namespace,
    *,
    progress_path: Path,
) -> dict[str, Any]:
    run_dir = _run_dir(Path(args.out_dir), sample)
    result_path = run_dir / "result_metrics.json"
    if result_path.is_file() and not args.overwrite:
        return _read_json(result_path)

    ds_sample = _find_sample(ds, sample.obj_id, sample.angle_idx)
    started = time.time()
    with VramSampler(str(args.gpu)) as sampler:
        z_global, whole_coords = run_ss_stage(ds, data_config, ds_sample, run_dir, args)
        part_names, probs, full_occ = run_part_probabilities(ds, data_config, ds_sample, run_dir, args)
    print(f"[eval_crf_0615] crf start {sample.obj_id}", flush=True)
    baseline_occs = [(probs[idx] > float(args.part_threshold)) & full_occ for idx in range(len(part_names))]
    crf_labels = apply_crf(
        probs,
        full_occ,
        threshold=float(args.part_threshold),
        pairwise_weight=float(args.crf_pairwise),
        edge_alpha=float(args.crf_edge_alpha),
        iterations=int(args.crf_iters),
    )
    crf_occs = _label_part_occs(crf_labels, len(part_names))
    _save_part_npzs(run_dir / "crf_parts", part_names, crf_occs)

    baseline_overlap = _overlap_count(baseline_occs)
    crf_overlap = _overlap_count(crf_occs)
    rows: list[dict[str, Any]] = []
    gt_occs: list[np.ndarray] = []
    raw_counts_by_part: list[int] = []
    part_buckets: list[str] = []
    for part in ds_sample["parts"]:
        part_name = str(part["part_name"])
        raw_coords = ds._load_raw_ind_coords(ds_sample, part).numpy().astype(np.int64)
        raw_count = int(raw_coords.shape[0])
        gt_occs.append(_coords_to_occ(raw_coords))
        raw_counts_by_part.append(raw_count)
        part_buckets.append(part_bucket(part_name, part, raw_count))

    gt_union = np.logical_or.reduce(gt_occs) if gt_occs else np.zeros((64, 64, 64), dtype=bool)
    object_rows: list[dict[str, Any]] = []
    for method, occs, overlap in (
        ("baseline", baseline_occs, baseline_overlap),
        ("crf", crf_occs, crf_overlap),
    ):
        pred_union = np.logical_or.reduce(occs) if occs else np.zeros((64, 64, 64), dtype=bool)
        obj_iou, obj_inter, obj_union, obj_pred_count = _coords_iou(_occ_to_coords(pred_union), _occ_to_coords(gt_union))
        object_rows.append(
            {
                "split": sample.split,
                "obj_id": sample.obj_id,
                "angle": int(sample.angle_idx),
                "sample_bucket": sample.sample_bucket,
                "priority_bucket": sample.priority_bucket,
                "method": method,
                "num_parts": int(len(part_names)),
                "raw_union_voxels": int(gt_union.sum()),
                "pred_union_voxels": int(obj_pred_count),
                "object_GTcand_IoU": float(obj_iou),
                "object_intersection": int(obj_inter),
                "object_union": int(obj_union),
                "part_overlap_voxels": int(overlap),
                "run_dir": str(run_dir),
            }
        )

    for part_idx, part in enumerate(ds_sample["parts"]):
        part_name = str(part["part_name"])
        raw_occ = gt_occs[part_idx]
        raw_count = raw_counts_by_part[part_idx]
        bucket = part_buckets[part_idx]
        for method, occs, overlap in (
            ("baseline", baseline_occs, baseline_overlap),
            ("crf", crf_occs, crf_overlap),
        ):
            pred_occ = occs[part_idx]
            pred_coords = _occ_to_coords(pred_occ)
            iou, inter, union, pred_count = _coords_iou(pred_coords, raw_coords)
            rows.append(
                {
                    "split": sample.split,
                    "obj_id": sample.obj_id,
                    "angle": int(sample.angle_idx),
                    "part_index": int(part_idx),
                    "part_name": part_name,
                    "bucket": bucket,
                    "sample_bucket": sample.sample_bucket,
                    "priority_bucket": sample.priority_bucket,
                    "method": method,
                    "raw_voxels": raw_count,
                    "pred_voxels": int(pred_count),
                    "GTcand_IoU": float(iou),
                    "IoU": float(iou),
                    "hit@0.5": int(iou >= 0.5),
                    "Boundary_IoU": float(_boundary_iou(pred_occ, raw_occ, int(args.boundary_radius))),
                    "part_overlap_voxels": int(overlap),
                    "intersection": int(inter),
                    "union": int(union),
                    "run_dir": str(run_dir),
                }
            )

    result = {
        "sample": asdict(sample),
        "rows": rows,
        "object_rows": object_rows,
        "seconds": round(time.time() - started, 3),
        "peak_vram_mib": int(sampler.max_mib),
        "whole_voxels": int(len(whole_coords)),
        "num_parts": int(len(part_names)),
        "run_dir": str(run_dir),
    }
    _write_json(result_path, result)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "idx": None,
            "split": sample.split,
            "obj_id": sample.obj_id,
            "angle": int(sample.angle_idx),
            "seconds": result["seconds"],
            "peak_vram_mib": result["peak_vram_mib"],
            "run_dir": str(run_dir),
        }, ensure_ascii=False) + "\n")
    print(f"[eval_crf_0615] sample done {sample.obj_id} seconds={result['seconds']}", flush=True)
    return result


def _surface_only(occ: np.ndarray) -> np.ndarray:
    occ = np.asarray(occ, dtype=bool)
    if not bool(occ.any()):
        return occ
    return occ & ~_binary_erosion6(occ)


def _colored_layers_from_parts(
    whole_occ: np.ndarray,
    part_occs: list[np.ndarray],
    *,
    include_body: bool = True,
    body_alpha: int = 54,
) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    layers: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    part_union = np.zeros((64, 64, 64), dtype=bool)
    for idx, occ in enumerate(part_occs):
        surf = _surface_only(occ)
        if not bool(surf.any()):
            continue
        part_union |= np.asarray(occ, dtype=bool)
        rgb = tuple(int(round(float(c) * 255)) for c in PART_COLORS[idx % len(PART_COLORS)][:3])
        layers.append((_occ_to_coords(surf), (*rgb, 246), 7))
    if include_body:
        body = _surface_only(whole_occ & ~part_union)
        if bool(body.any()):
            grid = np.indices(body.shape)
            body &= (grid[0] + grid[1] + grid[2]) % 3 == 0
            layers.insert(0, (_occ_to_coords(body), (BODY_RGBA[0], BODY_RGBA[1], BODY_RGBA[2], body_alpha), 4))
    return layers


def render_crf_comparison(
    out_path: Path,
    *,
    whole_occ: np.ndarray,
    gt_occs: list[np.ndarray],
    baseline_occs: list[np.ndarray],
    crf_occs: list[np.ndarray],
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tmp_dir = out_path.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    panel_paths = []
    panels = [
        ("GT", _colored_layers_from_parts(np.logical_or.reduce(gt_occs), gt_occs, include_body=False)),
        ("baseline", _colored_layers_from_parts(whole_occ, baseline_occs)),
        ("+CRF", _colored_layers_from_parts(whole_occ, crf_occs)),
    ]
    for label, layers in panels:
        panel_path = tmp_dir / f"{out_path.stem}_{label.replace('+', 'plus')}.png"
        _draw_block_projection(panel_path, layers, title=label, image_size=820)
        panel_paths.append((label, panel_path))

    images = [Image.open(path).convert("RGB") for _label, path in panel_paths]
    header = 56
    gap = 12
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images) + header
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        small_font = font
    draw.text((16, 10), title, fill=(20, 20, 20), font=font)
    x = 0
    for label, img in zip([label for label, _path in panel_paths], images):
        draw.text((x + 16, 34), label, fill=(20, 20, 20), font=small_font)
        canvas.paste(img, (x, header))
        x += img.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    for _label, path in panel_paths:
        path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def render_visualizations(ds: Any, selected: list[CrfSample], out_dir: Path, *, max_count: int = 8) -> list[str]:
    candidates = sorted(
        selected,
        key=lambda s: (
            0 if s.has_large_keyword else 1,
            -int(s.part_count),
            -int(s.max_raw_voxels),
            s.split,
            s.obj_id,
        ),
    )[: int(max_count)]
    written: list[str] = []
    for sample in candidates:
        run_dir = _run_dir(out_dir, sample)
        prob_path = run_dir / "part_probs.npz"
        result_path = run_dir / "result_metrics.json"
        if not prob_path.is_file() or not result_path.is_file():
            continue
        ds_sample = _find_sample(ds, sample.obj_id, sample.angle_idx)
        with np.load(prob_path, allow_pickle=True) as data:
            probs = np.asarray(data["probs"], dtype=np.float32)
            full_occ = np.asarray(data["full_occ"], dtype=bool)
        baseline_occs = [(probs[idx] > 0.5) & full_occ for idx in range(probs.shape[0])]
        crf_parts_dir = run_dir / "crf_parts" / "parts"
        crf_occs: list[np.ndarray] = []
        gt_occs: list[np.ndarray] = []
        for part_idx, part in enumerate(ds_sample["parts"]):
            path = crf_parts_dir / f"part_{part_idx:02d}_voxel.npz"
            crf_occs.append(_coords_to_occ(_load_npz_coords(path)) if path.is_file() else np.zeros((64, 64, 64), dtype=bool))
            gt_occs.append(_coords_to_occ(ds._load_raw_ind_coords(ds_sample, part).numpy()))
        out_path = out_dir / "visualizations" / f"{sample.split}_{sample.obj_id}_angle_{sample.angle_idx}_gt_baseline_crf.png"
        render_crf_comparison(
            out_path,
            whole_occ=full_occ,
            gt_occs=gt_occs,
            baseline_occs=baseline_occs,
            crf_occs=crf_occs,
            title=f"{sample.split} {sample.obj_id} angle {sample.angle_idx}",
        )
        written.append(str(out_path))
    return written


def summarize(rows: list[dict[str, Any]], peak_vram_mib: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        bucket = str(row["bucket"])
        method = str(row["method"])
        groups[(split, bucket, method)].append(row)
        groups[(split, "all", method)].append(row)
    out: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for bucket in (*BUCKETS, "all"):
            for method in ("baseline", "crf"):
                group = groups.get((split, bucket, method), [])
                if not group:
                    continue
                out.append(
                    {
                        "split": split,
                        "bucket": bucket,
                        "method": method,
                        "n": len(group),
                        "mean_GTcand_IoU": float(np.mean([float(r["GTcand_IoU"]) for r in group])),
                        "mean_IoU": float(np.mean([float(r["IoU"]) for r in group])),
                        "success@IoU0.5": float(np.mean([int(r["hit@0.5"]) for r in group])),
                        "mean_Boundary_IoU": float(np.mean([float(r["Boundary_IoU"]) for r in group])),
                        "mean_part_overlap_voxels": float(np.mean([float(r["part_overlap_voxels"]) for r in group])),
                        "peak_vram_mib": int(peak_vram_mib),
                    }
                )
    return out


def summarize_objects(rows: list[dict[str, Any]], peak_vram_mib: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        bucket = str(row["priority_bucket"])
        method = str(row["method"])
        groups[(split, bucket, method)].append(row)
        groups[(split, "all", method)].append(row)
    out: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for bucket in (*SELECTION_BUCKETS, "all"):
            for method in ("baseline", "crf"):
                group = groups.get((split, bucket, method), [])
                if not group:
                    continue
                out.append(
                    {
                        "split": split,
                        "priority_bucket": bucket,
                        "method": method,
                        "n_objects": len(group),
                        "mean_object_GTcand_IoU": float(np.mean([float(r["object_GTcand_IoU"]) for r in group])),
                        "mean_part_overlap_voxels": float(np.mean([float(r["part_overlap_voxels"]) for r in group])),
                        "peak_vram_mib": int(peak_vram_mib),
                    }
                )
    return out


def write_report(
    out_dir: Path,
    *,
    selected: list[CrfSample],
    selection_manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    object_summary: list[dict[str, Any]],
    visual_paths: list[str],
    args: argparse.Namespace,
    peak_vram_mib: int,
) -> None:
    ckpt = torch.load(str(args.part_seg_ckpt), map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args") or {})
    rows = [
        "# 0615-cfg Part-Seg 3D CRF Eval",
        "",
        "## Platform Entry",
        "",
        f"- Batch driver: `{Path(__file__).resolve()}`",
        f"- Existing platform/API used for stage1 SS-flow: `{REPO_ROOT / 'TRELLIS-arts/inference.py'}::run_ss_flow_from_tokens(tokens, ckpt_path, num_steps, cfg_strength, seed)`",
        f"- Existing platform/API used for stage2 part-seg: `{REPO_ROOT / 'TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py'}` route=voxel forward; CRF consumes `voxel_logits` before threshold.",
        f"- CLI smoke/full command: `python {Path(__file__).resolve()} --out-dir {out_dir} --gpu GPU`",
        "",
        "## Weights",
        "",
        f"- SS-flow EMA2 ckpt: `{args.ss_flow_ckpt}`",
        f"- SS-flow rule: official DINO `[4,1374,1024]`, per-step velocity mean, zero-token CFG, steps={args.ss_steps}, cfg={args.ss_cfg}, seed={args.ss_seed}.",
        f"- Part-seg ckpt: `{args.part_seg_ckpt}`",
        f"- Part-seg checkpoint step: `{ckpt.get('step')}`",
        f"- Part-seg EMA: `not present in checkpoint keys`",
        f"- Part-seg args: route={ckpt_args.get('route')} dim={ckpt_args.get('dim')} depth={ckpt_args.get('depth')} mask_encoder={ckpt_args.get('mask_encoder')}",
        "",
        "## Selection",
        "",
        f"- Output: `{out_dir}`",
        f"- Unique objects: `{len({s.obj_id for s in selected})}`; one angle per object: `true`.",
        f"- Train/val counts: `{selection_manifest['counts']}`",
        "",
        "### train",
        "",
        "| obj_id | angle | sample_bucket | priority_bucket | part_count | min_raw | max_raw | reason |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for sample in [s for s in selected if s.split == "train"]:
        rows.append(
            f"| {sample.obj_id} | {sample.angle_idx} | {sample.sample_bucket} | {sample.priority_bucket} | {sample.part_count} | "
            f"{sample.min_raw_voxels} | {sample.max_raw_voxels} | {sample.selected_reason} |"
        )
    rows.extend([
        "",
        "### val",
        "",
        "| obj_id | angle | sample_bucket | priority_bucket | part_count | min_raw | max_raw | reason |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for sample in [s for s in selected if s.split == "val"]:
        rows.append(
            f"| {sample.obj_id} | {sample.angle_idx} | {sample.sample_bucket} | {sample.priority_bucket} | {sample.part_count} | "
            f"{sample.min_raw_voxels} | {sample.max_raw_voxels} | {sample.selected_reason} |"
        )
    rows.extend(
        [
            "",
            "## Summary",
            "",
            "| split | bucket | method | n | mean_GTcand_IoU | mean_IoU | success@IoU0.5 | mean_Boundary_IoU | mean_part_overlap_voxels | peak_vram_mib |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        rows.append(
            f"| {row['split']} | {row['bucket']} | {row['method']} | {row['n']} | "
            f"{row['mean_GTcand_IoU']:.4f} | {row['mean_IoU']:.4f} | {row['success@IoU0.5']:.4f} | "
            f"{row['mean_Boundary_IoU']:.4f} | {row['mean_part_overlap_voxels']:.2f} | {row['peak_vram_mib']} |"
        )
    rows.extend(
        [
            "",
            "## Object-Level GTcand",
            "",
            "| split | priority_bucket | method | n_objects | mean_object_GTcand_IoU | mean_part_overlap_voxels | peak_vram_mib |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in object_summary:
        rows.append(
            f"| {row['split']} | {row['priority_bucket']} | {row['method']} | {row['n_objects']} | "
            f"{row['mean_object_GTcand_IoU']:.4f} | {row['mean_part_overlap_voxels']:.2f} | {row['peak_vram_mib']} |"
        )
    lookup = {(r["split"], r["bucket"], r["method"]): r for r in summary}
    rows.extend(["", "## Deltas", ""])
    for split in ("train", "val"):
        for bucket in (*BUCKETS, "all"):
            base = lookup.get((split, bucket, "baseline"))
            crf = lookup.get((split, bucket, "crf"))
            if not base or not crf:
                continue
            rows.append(
                f"- {split}/{bucket}: CRF-baseline Boundary_IoU {crf['mean_Boundary_IoU'] - base['mean_Boundary_IoU']:+.4f}, "
                f"IoU {crf['mean_IoU'] - base['mean_IoU']:+.4f}, "
                f"overlap {crf['mean_part_overlap_voxels'] - base['mean_part_overlap_voxels']:+.2f}"
            )
    rows.extend(
        [
            "",
            "## VRAM",
            "",
            f"- Peak sampled VRAM: `{peak_vram_mib} MiB`.",
            "",
            "## Visualizations",
            "",
        ]
    )
    rows.extend([f"- `{p}`" for p in visual_paths] or ["- none"])
    rows.extend([
        "",
        "## Artifacts",
        "",
        f"- `metrics.csv`: `{out_dir / 'metrics.csv'}`",
        f"- `metrics_summary.csv`: `{out_dir / 'metrics_summary.csv'}`",
        f"- `object_metrics.csv`: `{out_dir / 'object_metrics.csv'}`",
        f"- `object_metrics_summary.csv`: `{out_dir / 'object_metrics_summary.csv'}`",
        f"- `metrics.png`: `{out_dir / 'metrics.png'}`",
        f"- `metrics_summary.png`: `{out_dir / 'metrics_summary.png'}`",
    ])
    (out_dir / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate part-seg baseline vs 3D CRF post-processing on 128 unique objects.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--gpu", default="0")
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    p.add_argument("--part-seg-ckpt", default=str(DEFAULT_PART_SEG_CKPT))
    p.add_argument("--ss-flow-ckpt", default=str(DEFAULT_SS_FLOW_EMA2_CKPT))
    p.add_argument("--ss-decoder-ckpt", default=str(DEFAULT_SS_DECODER_CKPT))
    p.add_argument("--train-count", type=int, default=85)
    p.add_argument("--val-count", type=int, default=43)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--ss-steps", type=int, default=20)
    p.add_argument("--ss-cfg", type=float, default=7.5)
    p.add_argument("--ss-seed", type=int, default=20260610)
    p.add_argument("--decode-threshold", type=float, default=0.0)
    p.add_argument("--part-threshold", type=float, default=0.5)
    p.add_argument("--boundary-radius", type=int, default=2)
    p.add_argument("--crf-iters", type=int, default=7)
    p.add_argument("--crf-pairwise", type=float, default=0.45)
    p.add_argument("--crf-edge-alpha", type=float, default=1.2)
    p.add_argument("--visual-count", type=int, default=8)
    p.add_argument("--clear-ss-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--selection-only", action="store_true")
    return p.parse_args()


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out_dir = str(Path(args.out_dir).expanduser())
    out_dir = Path(args.out_dir)
    _require_file(Path(args.data_config), "data config")
    _require_file(Path(args.split_json), "split json")
    _require_file(Path(args.part_seg_ckpt), "part-seg ckpt")
    _require_file(Path(args.ss_flow_ckpt), "SS-flow ckpt")
    _require_file(Path(args.ss_decoder_ckpt), "SS decoder ckpt")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_config = load_data_config(Path(args.data_config))
    ds = _dataset_for("four", data_config)
    selection_path = out_dir / "selection.json"
    if selection_path.is_file() and not args.overwrite:
        selected, selection_manifest = load_selection_manifest(selection_path)
        print(f"[eval_crf_0615] loaded existing selection: {selection_path}", flush=True)
    else:
        selected, selection_manifest = build_unique_selection(
            ds,
            Path(args.split_json),
            train_count=int(args.train_count),
            val_count=int(args.val_count),
        )
    if int(args.limit_samples) > 0:
        selected = selected[: int(args.limit_samples)]
        selection_manifest["limit_samples"] = int(args.limit_samples)
        selection_manifest["samples"] = {
            split: [asdict(s) for s in selected if s.split == split]
            for split in ("train", "val")
        }
    _write_json(out_dir / "selection.json", selection_manifest)
    _write_csv(
        out_dir / "selection.csv",
        [asdict(sample) for sample in selected],
        [
            "split",
            "obj_id",
            "angle_idx",
            "sample_bucket",
            "priority_bucket",
            "part_count",
            "min_raw_voxels",
            "max_raw_voxels",
            "has_button",
            "has_large_keyword",
            "selected_reason",
        ],
    )
    if args.selection_only:
        print(f"[eval_crf_0615] selection written -> {out_dir}", flush=True)
        return 0

    progress_path = out_dir / "progress.jsonl"
    rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    peak_vram = 0
    for idx, sample in enumerate(selected, 1):
        print(
            f"[eval_crf_0615] {idx}/{len(selected)} {sample.split} {sample.obj_id} "
            f"angle={sample.angle_idx} bucket={sample.sample_bucket}/{sample.priority_bucket} parts={sample.part_count}",
            flush=True,
        )
        result = evaluate_sample(ds, data_config, sample, args, progress_path=progress_path)
        for row in result["rows"]:
            row["sample_idx"] = int(idx)
            row["peak_vram_mib"] = int(result["peak_vram_mib"])
            rows.append(row)
        for row in result.get("object_rows", []):
            row["sample_idx"] = int(idx)
            row["peak_vram_mib"] = int(result["peak_vram_mib"])
            object_rows.append(row)
        peak_vram = max(peak_vram, int(result["peak_vram_mib"]))
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": idx, "done": True, "peak_vram_mib": peak_vram}, ensure_ascii=False) + "\n")

    summary = summarize(rows, peak_vram)
    object_summary = summarize_objects(object_rows, peak_vram)
    metric_fields = [
        "sample_idx",
        "split",
        "obj_id",
        "angle",
        "part_index",
        "part_name",
        "bucket",
        "sample_bucket",
        "priority_bucket",
        "method",
        "raw_voxels",
        "pred_voxels",
        "GTcand_IoU",
        "IoU",
        "hit@0.5",
        "Boundary_IoU",
        "part_overlap_voxels",
        "intersection",
        "union",
        "peak_vram_mib",
        "run_dir",
    ]
    summary_fields = [
        "split",
        "bucket",
        "method",
        "n",
        "mean_GTcand_IoU",
        "mean_IoU",
        "success@IoU0.5",
        "mean_Boundary_IoU",
        "mean_part_overlap_voxels",
        "peak_vram_mib",
    ]
    object_fields = [
        "sample_idx",
        "split",
        "obj_id",
        "angle",
        "sample_bucket",
        "priority_bucket",
        "method",
        "num_parts",
        "raw_union_voxels",
        "pred_union_voxels",
        "object_GTcand_IoU",
        "object_intersection",
        "object_union",
        "part_overlap_voxels",
        "peak_vram_mib",
        "run_dir",
    ]
    object_summary_fields = [
        "split",
        "priority_bucket",
        "method",
        "n_objects",
        "mean_object_GTcand_IoU",
        "mean_part_overlap_voxels",
        "peak_vram_mib",
    ]
    _write_csv(out_dir / "metrics.csv", rows, metric_fields)
    _write_json(out_dir / "metrics.json", rows)
    _write_csv(out_dir / "metrics_summary.csv", summary, summary_fields)
    _write_json(out_dir / "metrics_summary.json", summary)
    _write_csv(out_dir / "object_metrics.csv", object_rows, object_fields)
    _write_json(out_dir / "object_metrics.json", object_rows)
    _write_csv(out_dir / "object_metrics_summary.csv", object_summary, object_summary_fields)
    _write_json(out_dir / "object_metrics_summary.json", object_summary)
    render_table_png(
        out_dir / "metrics.png",
        rows,
        columns=["split", "obj_id", "angle", "part_name", "bucket", "method", "IoU", "Boundary_IoU", "part_overlap_voxels"],
        title="0615-cfg baseline vs CRF detail metrics",
        max_rows=220,
    )
    render_table_png(
        out_dir / "metrics_summary.png",
        summary,
        columns=summary_fields,
        title="0615-cfg baseline vs CRF summary",
        max_rows=None,
    )
    visual_paths = render_visualizations(ds, selected, out_dir, max_count=int(args.visual_count))
    (out_dir / "peak_vram.txt").write_text(f"{peak_vram}\n", encoding="utf-8")
    write_report(
        out_dir,
        selected=selected,
        selection_manifest=selection_manifest,
        summary=summary,
        object_summary=object_summary,
        visual_paths=visual_paths,
        args=args,
        peak_vram_mib=peak_vram,
    )
    print(f"[eval_crf_0615] done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
