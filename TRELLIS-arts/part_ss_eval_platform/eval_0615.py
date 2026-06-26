from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

from inference_pipeline.data_config_io import load_data_config  # noqa: E402
from inference_pipeline.object_inputs import _dataset_for  # noqa: E402
from part_ss_eval_platform.infer_jobs import InferJobRequest, build_infer_command  # noqa: E402
from trellis.trainers.arts.part_ss_latent_flow_eval import coords_iou  # noqa: E402


DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0615-1")
DEFAULT_SPLIT_JSON = Path("/robot/data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_v1.json")
DEFAULT_DATA_CONFIG = TRELLIS_PATH / "configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml"
DEFAULT_PART_SEG_CKPT = Path("/robot/data-lab/jzh/art-gen-output/part_promptable_seg_full_M_0612-3/ckpts/latest.pt")
DEFAULT_SS_FLOW_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.9999_step0020000.pt"
)
DEFAULT_SS_DECODER_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
BUCKETS = ("tiny", "small", "medium", "large", "button")
SIZE_BUCKETS = ("tiny", "small", "medium", "large")
BUCKET_ORDER = {name: idx for idx, name in enumerate(BUCKETS)}
DEFAULT_QUOTA = {"tiny": 12, "small": 12, "medium": 16, "large": 12, "button": 12}
HARD_OBJECT_IDS = ("101564", "103996")


@dataclass(frozen=True)
class SelectedSample:
    split: str
    obj_id: str
    angle_idx: int
    bucket: str
    part_count: int
    min_raw_voxels: int
    has_button: bool
    forced_reason: str = ""


class VramSampler:
    def __init__(self, gpu: str, interval: float = 0.5) -> None:
        self.gpu = str(gpu).split(",")[0]
        self.interval = float(interval)
        self.max_mib = 0
        self.samples: list[int] = []
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
                self.samples.append(value)
                self.max_mib = max(self.max_mib, value)
            self._stop.wait(self.interval)

    def _query(self) -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    self.gpu,
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    values.append(int(line))
                except ValueError:
                    pass
        return max(values) if values else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _part_is_button(part: dict[str, Any]) -> bool:
    values = [str(part.get("part_name", ""))]
    target = part.get("target_part")
    if isinstance(target, dict):
        values.extend(str(target.get(key, "")) for key in ("type", "name", "semantic_type"))
    values.append(str(part.get("type", "")))
    return any("button" in value.lower() for value in values)


def size_bucket(raw_count: int) -> str:
    raw_count = int(raw_count)
    if raw_count < 50:
        return "tiny"
    if raw_count < 500:
        return "small"
    if raw_count < 3000:
        return "medium"
    return "large"


def part_bucket(part_name: str, part: dict[str, Any], raw_count: int) -> str:
    if _part_is_button({"part_name": part_name, **part}):
        return "button"
    return size_bucket(raw_count)


def _sample_info(ds, sample: dict[str, Any]) -> tuple[str, int, bool]:
    raw_counts: list[int] = []
    has_button = False
    for part in sample["parts"]:
        raw = ds._load_raw_ind_coords(sample, part)
        raw_counts.append(int(raw.shape[0]))
        has_button = has_button or _part_is_button(part)
    min_raw = min(raw_counts) if raw_counts else 0
    if has_button:
        return "button", min_raw, True
    return size_bucket(min_raw), min_raw, False


def _sample_key(sample: SelectedSample) -> tuple[int, int, str]:
    return (BUCKET_ORDER[sample.bucket], int(sample.angle_idx), str(sample.obj_id))


def build_selection(
    ds,
    split_json: Path,
    *,
    per_split: int = 64,
    quota: dict[str, int] | None = None,
) -> tuple[list[SelectedSample], dict[str, Any]]:
    quota = dict(quota or DEFAULT_QUOTA)
    split_data = _read_json(split_json)
    split_ids = {
        "train": {str(x) for x in split_data["train_ids"]},
        "held": {str(x) for x in split_data["heldout_ids"]},
    }
    candidates: dict[str, dict[str, list[SelectedSample]]] = {
        split: {bucket: [] for bucket in BUCKETS}
        for split in ("train", "held")
    }
    hard_availability = {
        obj_id: {split: [] for split in ("train", "held")}
        for obj_id in HARD_OBJECT_IDS
    }
    for sample in ds.samples:
        obj_id = str(sample["obj_id"])
        split = "train" if obj_id in split_ids["train"] else "held" if obj_id in split_ids["held"] else ""
        if not split:
            continue
        bucket, min_raw, has_button = _sample_info(ds, sample)
        selected = SelectedSample(
            split=split,
            obj_id=obj_id,
            angle_idx=int(sample["angle_idx"]),
            bucket=bucket,
            part_count=len(sample["parts"]),
            min_raw_voxels=int(min_raw),
            has_button=bool(has_button),
        )
        candidates[split][bucket].append(selected)
        if obj_id in hard_availability:
            hard_availability[obj_id][split].append(selected)
    for split in candidates:
        for bucket in candidates[split]:
            candidates[split][bucket].sort(key=lambda s: (s.obj_id, s.angle_idx, s.part_count))

    selected_by_split: dict[str, list[SelectedSample]] = {"train": [], "held": []}
    notes: list[str] = []
    for split in ("train", "held"):
        chosen: dict[tuple[str, int], SelectedSample] = {}
        if split == "held":
            for obj_id in HARD_OBJECT_IDS:
                available = sorted(hard_availability[obj_id][split], key=lambda s: (s.bucket != "button", s.angle_idx))
                if available:
                    base = available[0]
                    chosen[(base.obj_id, base.angle_idx)] = SelectedSample(
                        **{**asdict(base), "forced_reason": "hard_button_panel"}
                    )
                else:
                    notes.append(f"{obj_id} has no held sample in split_official_v1")
        else:
            for obj_id in HARD_OBJECT_IDS:
                if not hard_availability[obj_id][split]:
                    notes.append(f"{obj_id} is not in train_ids of split_official_v1; train side cannot include it without split leakage")

        for bucket in BUCKETS:
            target = int(quota.get(bucket, 0))
            have = sum(1 for sample in chosen.values() if sample.bucket == bucket)
            need = max(0, target - have)
            for candidate in candidates[split][bucket]:
                key = (candidate.obj_id, candidate.angle_idx)
                if key in chosen:
                    continue
                chosen[key] = candidate
                need -= 1
                if need <= 0:
                    break
            actual = sum(1 for sample in chosen.values() if sample.bucket == bucket)
            if actual < target:
                notes.append(f"{split}/{bucket} quota short: requested {target}, got {actual}")

        if len(chosen) < int(per_split):
            remaining = [
                candidate
                for bucket in BUCKETS
                for candidate in candidates[split][bucket]
                if (candidate.obj_id, candidate.angle_idx) not in chosen
            ]
            remaining.sort(key=lambda s: (s.obj_id, s.angle_idx, BUCKET_ORDER[s.bucket]))
            for candidate in remaining:
                chosen[(candidate.obj_id, candidate.angle_idx)] = candidate
                if len(chosen) >= int(per_split):
                    break
        elif len(chosen) > int(per_split):
            forced_keys = {
                (sample.obj_id, sample.angle_idx)
                for sample in chosen.values()
                if sample.forced_reason
            }
            ordered = sorted(chosen.values(), key=lambda s: (0 if (s.obj_id, s.angle_idx) in forced_keys else 1, _sample_key(s)))
            chosen = {(sample.obj_id, sample.angle_idx): sample for sample in ordered[: int(per_split)]}
        selected_by_split[split] = sorted(chosen.values(), key=lambda s: (BUCKET_ORDER[s.bucket], s.obj_id, s.angle_idx))

    manifest = {
        "split_json": str(split_json),
        "per_split": int(per_split),
        "quota": quota,
        "bucket_definition": (
            "obj-angle bucket uses button priority when any target part is a button; otherwise the smallest "
            "target part raw voxel count decides tiny/small/medium/large with thresholds <50/<500/<3000."
        ),
        "notes": notes,
        "counts": {
            split: {bucket: sum(1 for sample in rows if sample.bucket == bucket) for bucket in BUCKETS}
            for split, rows in selected_by_split.items()
        },
        "samples": {
            split: [asdict(sample) for sample in rows]
            for split, rows in selected_by_split.items()
        },
    }
    return selected_by_split["train"] + selected_by_split["held"], manifest


def choose_b_subset(selected: list[SelectedSample], limit: int = 32) -> list[SelectedSample]:
    by_split_bucket: dict[tuple[str, str], list[SelectedSample]] = defaultdict(list)
    for sample in selected:
        by_split_bucket[(sample.split, sample.bucket)].append(sample)
    for key in by_split_bucket:
        by_split_bucket[key].sort(key=lambda s: (0 if s.forced_reason else 1, s.obj_id, s.angle_idx))

    chosen: dict[tuple[str, int], SelectedSample] = {}
    per_split_target = max(1, int(limit) // 2)
    per_split_quota = {"tiny": 4, "small": 3, "medium": 3, "large": 3, "button": 3}
    for sample in selected:
        if sample.obj_id in HARD_OBJECT_IDS:
            chosen[(sample.obj_id, sample.angle_idx)] = sample
    for split in ("train", "held"):
        for bucket in BUCKETS:
            have = sum(1 for sample in chosen.values() if sample.split == split and sample.bucket == bucket)
            need = max(0, int(per_split_quota.get(bucket, 0)) - have)
            for sample in by_split_bucket.get((split, bucket), []):
                if need <= 0:
                    break
                if (sample.obj_id, sample.angle_idx) in chosen:
                    continue
                chosen[(sample.obj_id, sample.angle_idx)] = sample
                need -= 1

    def _count_split(split: str) -> int:
        return sum(1 for sample in chosen.values() if sample.split == split)

    for split in ("train", "held"):
        remaining = [
            sample
            for sample in selected
            if sample.split == split and (sample.obj_id, sample.angle_idx) not in chosen
        ]
        remaining.sort(key=lambda s: (BUCKET_ORDER[s.bucket], s.obj_id, s.angle_idx))
        cursor = 0
        while _count_split(split) < per_split_target and cursor < len(remaining):
            sample = remaining[cursor]
            chosen[(sample.obj_id, sample.angle_idx)] = sample
            cursor += 1
            if len(chosen) >= int(limit):
                break

    remaining_all = sorted(selected, key=lambda s: (_count_split(s.split), s.split, BUCKET_ORDER[s.bucket], s.obj_id, s.angle_idx))
    for sample in remaining_all:
        chosen.setdefault((sample.obj_id, sample.angle_idx), sample)
        if len(chosen) >= int(limit):
            break
    return sorted(chosen.values(), key=lambda s: (s.split, BUCKET_ORDER[s.bucket], s.obj_id, s.angle_idx))[: int(limit)]


def run_dir_for(out_dir: Path, link: str, sample: SelectedSample) -> Path:
    root = out_dir / "platform_runs" / link / sample.split
    return root / f"{sample.obj_id}-{sample.angle_idx}" / f"eval-{link}"


def command_for(
    *,
    out_dir: Path,
    link: str,
    stage: str,
    sample: SelectedSample,
    data_config: Path,
    part_seg_ckpt: Path,
    ss_flow_ckpt: Path,
    ss_decoder_ckpt: Path,
    gpu: str,
) -> Any:
    req = InferJobRequest(
        stage=stage,
        object_id=sample.obj_id,
        root=str(out_dir / "platform_runs" / link / sample.split),
        run_id=f"eval-{link}",
        mode=link,
        view="four",
        data_config=str(data_config),
        angle_idx=int(sample.angle_idx),
        part_seg_ckpt=str(part_seg_ckpt),
        ss_flow_ckpt=str(ss_flow_ckpt) if link == "B" else "",
        ss_decoder_ckpt=str(ss_decoder_ckpt),
        part_backend="promptable_seg",
        decode_backend="trellis",
        gpu_ids=str(gpu),
        overwrite=True,
    )
    return build_infer_command(req, repo_root=REPO_ROOT)


def execute_command(spec, *, gpu: str, stage_name: str, progress_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(spec.env or {})
    Path(spec.run_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log, VramSampler(gpu) as sampler:
        log.write(f"\n[eval_0615] stage={stage_name} cmd={' '.join(spec.args)}\n")
        log.flush()
        proc = subprocess.Popen(
            spec.args,
            cwd=spec.cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    record = {
        "stage": stage_name,
        "returncode": int(return_code),
        "seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
        "run_dir": str(spec.run_dir),
        "peak_vram_mib": int(sampler.max_mib),
        "cmd": spec.args,
    }
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if return_code != 0:
        raise RuntimeError(f"{stage_name} failed with code {return_code}; log={log_path}")
    return record


def load_npz_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["coords"], dtype=np.int32).reshape(-1, 3)


def _find_sample(ds, obj_id: str, angle_idx: int) -> tuple[int, dict[str, Any]]:
    for idx, sample in enumerate(ds.samples):
        if str(sample["obj_id"]) == str(obj_id) and int(sample["angle_idx"]) == int(angle_idx):
            return idx, sample
    raise KeyError(f"sample not found: {obj_id} angle={angle_idx}")


def collect_metrics(ds, selected: list[SelectedSample], *, out_dir: Path, links: list[str], peak_vram: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_by_key = {(sample.obj_id, sample.angle_idx): sample for sample in selected}
    for link in links:
        for sample in selected:
            rd = run_dir_for(out_dir, link, sample)
            if not rd.is_dir():
                continue
            if not list((rd / "parts").glob("part_*_voxel.npz")):
                continue
            _, ds_sample = _find_sample(ds, sample.obj_id, sample.angle_idx)
            for part_idx, part in enumerate(ds_sample["parts"]):
                part_name = str(part["part_name"])
                raw_coords = ds._load_raw_ind_coords(ds_sample, part).numpy().astype(np.int32)
                pred_path = rd / "parts" / f"part_{part_idx:02d}_voxel.npz"
                if pred_path.is_file():
                    pred_coords = load_npz_coords(pred_path)
                    metric = coords_iou(pred_coords, raw_coords)
                    iou = float(metric["iou"])
                    pred_count = int(metric["pred_count"])
                else:
                    iou = 0.0
                    pred_count = 0
                raw_count = int(raw_coords.shape[0])
                bucket = part_bucket(part_name, part, raw_count)
                selected_sample = selected_by_key[(sample.obj_id, sample.angle_idx)]
                rows.append(
                    {
                        "split": sample.split,
                        "obj_id": sample.obj_id,
                        "angle": int(sample.angle_idx),
                        "part_name": part_name,
                        "bucket": bucket,
                        "sample_bucket": selected_sample.bucket,
                        "raw_voxels": raw_count,
                        "pred_voxels": pred_count,
                        "link": link,
                        "IoU": iou,
                        "hit@0.5": int(iou >= 0.5),
                        "run_dir": str(rd),
                        "peak_vram_mib": int(peak_vram.get(link, 0)),
                    }
                )
    return rows


def summarize_metrics(rows: list[dict[str, Any]], peak_vram: dict[str, int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["bucket"], row["link"])].append(row)
        groups[(row["split"], "all", row["link"])].append(row)
    summary: list[dict[str, Any]] = []
    for split in ("train", "held"):
        for bucket in (*BUCKETS, "all"):
            for link in ("A", "B"):
                group = groups.get((split, bucket, link), [])
                if not group:
                    continue
                ious = [float(row["IoU"]) for row in group]
                hits = [int(row["hit@0.5"]) for row in group]
                summary.append(
                    {
                        "split": split,
                        "bucket": bucket,
                        "link": link,
                        "n": len(group),
                        "mean_IoU": float(np.mean(ious)),
                        "success@IoU0.5": float(np.mean(hits)),
                        "peak_vram_mib": int(peak_vram.get(link, 0)),
                    }
                )
    return summary


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "obj_id",
        "angle",
        "part_name",
        "bucket",
        "raw_voxels",
        "link",
        "IoU",
        "hit@0.5",
        "pred_voxels",
        "sample_bucket",
        "peak_vram_mib",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "bucket", "link", "n", "mean_IoU", "success@IoU0.5", "peak_vram_mib"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_table_png(path: Path, rows: list[dict[str, Any]], *, columns: list[str], title: str, max_rows: int | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_rows = rows if max_rows is None else rows[:max_rows]
    data = []
    for row in display_rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        data.append(values)
    footer = ""
    if max_rows is not None and len(rows) > len(display_rows):
        footer = f"showing {len(display_rows)} of {len(rows)} rows; full table in JSON/CSV"
    fig_h = max(2.5, 0.26 * (len(display_rows) + 2))
    fig_w = max(10.0, 1.15 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title if not footer else f"{title}\n{footer}", fontsize=10, pad=8)
    if data:
        table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.15)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _project_coords(coords: np.ndarray, max_points: int = 18000) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    if coords.shape[0] > max_points:
        step = int(np.ceil(coords.shape[0] / max_points))
        coords = coords[::step]
    return coords


def plot_whole(path: Path, coords: np.ndarray, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    coords = _project_coords(coords)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if coords.size:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1.0, c="#2f6f95", alpha=0.75)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parts(path: Path, whole_coords: np.ndarray, part_coords: list[tuple[str, np.ndarray]], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#e377c2", "#8c564b"]
    path.parent.mkdir(parents=True, exist_ok=True)
    whole = _project_coords(whole_coords, max_points=12000)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if whole.size:
        ax.scatter(whole[:, 0], whole[:, 1], whole[:, 2], s=0.5, c="#bdbdbd", alpha=0.16)
    for idx, (name, coords) in enumerate(part_coords):
        pts = _project_coords(coords, max_points=5000)
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4.0, c=colors[idx % len(colors)], alpha=0.9, label=name[:20])
    if part_coords:
        ax.legend(loc="upper left", fontsize=5, markerscale=2.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_artifacts(ds, selected: list[SelectedSample], *, out_dir: Path, links: list[str]) -> None:
    for sample in selected:
        _, ds_sample = _find_sample(ds, sample.obj_id, sample.angle_idx)
        for link in links:
            rd = run_dir_for(out_dir, link, sample)
            if not rd.is_dir():
                continue
            artifact_dir = out_dir / sample.split / sample.obj_id / str(sample.angle_idx)
            suffix = "" if link == "A" else f"_{link}"
            whole_path = rd / "voxel.npz"
            if whole_path.is_file():
                whole = load_npz_coords(whole_path)
                plot_whole(
                    artifact_dir / f"stage1_whole{suffix}.png",
                    whole,
                    title=f"{sample.split} {sample.obj_id} angle {sample.angle_idx} link {link}",
                )
                part_items: list[tuple[str, np.ndarray]] = []
                for part_idx, part in enumerate(ds_sample["parts"]):
                    part_path = rd / "parts" / f"part_{part_idx:02d}_voxel.npz"
                    if part_path.is_file():
                        part_items.append((str(part["part_name"]), load_npz_coords(part_path)))
                if part_items:
                    plot_parts(
                        artifact_dir / f"stage2_parts{suffix}.png",
                        whole,
                        part_items,
                        title=f"{sample.split} {sample.obj_id} angle {sample.angle_idx} link {link}",
                    )


def _summary_lookup(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["split"], row["bucket"], row["link"]): row for row in summary}


def matrix_markdown(summary: list[dict[str, Any]]) -> str:
    rows = ["| split | bucket | link | n | mean_IoU | success@IoU0.5 | peak_vram_mib |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        rows.append(
            f"| {row['split']} | {row['bucket']} | {row['link']} | {row['n']} | "
            f"{row['mean_IoU']:.4f} | {row['success@IoU0.5']:.4f} | {row['peak_vram_mib']} |"
        )
    return "\n".join(rows)


def selection_markdown(samples: list[SelectedSample], split: str) -> str:
    rows = [f"### {split} selection", "| obj_id | angle | bucket | part_count | min_raw_voxels | note |", "| --- | ---: | --- | ---: | ---: | --- |"]
    for sample in [s for s in samples if s.split == split]:
        rows.append(
            f"| {sample.obj_id} | {sample.angle_idx} | {sample.bucket} | {sample.part_count} | "
            f"{sample.min_raw_voxels} | {sample.forced_reason} |"
        )
    return "\n".join(rows)


def gap_lines(summary: list[dict[str, Any]]) -> list[str]:
    lookup = _summary_lookup(summary)
    lines: list[str] = []
    for bucket in (*BUCKETS, "all"):
        train = lookup.get(("train", bucket, "A"))
        held = lookup.get(("held", bucket, "A"))
        if train and held:
            lines.append(
                f"- A train-held gap {bucket}: mean_IoU {train['mean_IoU'] - held['mean_IoU']:+.4f}, "
                f"success {train['success@IoU0.5'] - held['success@IoU0.5']:+.4f}"
            )
    for split in ("train", "held"):
        for bucket in (*BUCKETS, "all"):
            a = lookup.get((split, bucket, "A"))
            b = lookup.get((split, bucket, "B"))
            if a and b:
                lines.append(
                    f"- {split} A-B gap {bucket}: mean_IoU {a['mean_IoU'] - b['mean_IoU']:+.4f}, "
                    f"success {a['success@IoU0.5'] - b['success@IoU0.5']:+.4f}"
                )
    return lines


def append_report(
    path: Path,
    *,
    out_dir: Path,
    selected: list[SelectedSample],
    selection_manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    peak_vram: dict[str, int],
    b_status: str,
    b_error: str,
    args: argparse.Namespace,
) -> None:
    lines = [
        "",
        "# 0615-1 Part Promptable Seg Eval Platform Run",
        "",
        "## Platform Entry",
        "",
        f"- CLI entry: `{REPO_ROOT / 'scripts/inference/infer_stage.py'}`",
        "- Programmatic entry: `part_ss_eval_platform.infer_jobs.build_infer_command(req, repo_root=Path(...))` with `InferJobRequest`.",
        "- Command shape: `python scripts/inference/infer_stage.py --stage {ss,part,slat,assemble} --object-id OBJ --root ROOT --run-id RUN --mode {A,B} --view {single,four} --angle-idx N --data-config CONFIG --gpu GPU --part-backend {part_flow,promptable_seg} --part-seg-ckpt CKPT --ss-flow-ckpt CKPT --overwrite`.",
        f"- This batch driver: `{Path(__file__).resolve()}` calls that existing platform command sequentially; it does not implement a separate inference path.",
        "",
        "## Weights",
        "",
        f"- Stage1 SS-flow B ckpt: `{args.ss_flow_ckpt}`",
        f"- Stage2 promptable part seg ckpt: `{args.part_seg_ckpt}`",
        f"- SS decoder ckpt: `{args.ss_decoder_ckpt}`",
        "",
        "## Selection",
        "",
        f"- Split: `{args.split_json}`",
        f"- Bucket rule: {selection_manifest['bucket_definition']}",
        f"- Counts: `{json.dumps(selection_manifest['counts'], ensure_ascii=False)}`",
    ]
    if selection_manifest.get("notes"):
        lines.extend(["- Notes:"] + [f"  - {note}" for note in selection_manifest["notes"]])
    lines.extend(["", selection_markdown(selected, "train"), "", selection_markdown(selected, "held"), ""])
    lines.extend(
        [
            "## Summary Matrix",
            "",
            matrix_markdown(summary),
            "",
            "## Gaps",
            "",
            *gap_lines(summary),
            "",
            "## Link B Status",
            "",
            f"- Status: {b_status}",
        ]
    )
    if b_error:
        lines.append(f"- Error/blocker: `{b_error}`")
    lines.extend(
        [
            "",
            "## VRAM",
            "",
            f"- Peak VRAM A: {peak_vram.get('A', 0)} MiB.",
            f"- Peak VRAM B: {peak_vram.get('B', 0)} MiB.",
            "- 4090 deployment target: <= 8192 MiB. Values here are sampled from `nvidia-smi` process-window GPU memory and include baseline GPU occupancy.",
            "",
            "## Artifacts",
            "",
            f"- Output directory: `{out_dir}`",
            f"- Detail metrics: `{out_dir / 'metrics.png'}`, `{out_dir / 'metrics.csv'}`, `{out_dir / 'metrics.json'}`",
            f"- Summary metrics: `{out_dir / 'metrics_summary.png'}`, `{out_dir / 'metrics_summary.csv'}`, `{out_dir / 'metrics_summary.json'}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0615 part SS eval platform batch driver")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--part-seg-ckpt", default=str(DEFAULT_PART_SEG_CKPT))
    p.add_argument("--ss-flow-ckpt", default=str(DEFAULT_SS_FLOW_CKPT))
    p.add_argument("--ss-decoder-ckpt", default=str(DEFAULT_SS_DECODER_CKPT))
    p.add_argument("--gpu", default="0")
    p.add_argument("--per-split", type=int, default=64)
    p.add_argument("--b-limit", type=int, default=32)
    p.add_argument("--max-a", type=int, default=-1, help="debug cap for link A selected objects")
    p.add_argument("--max-b", type=int, default=-1, help="debug cap for link B selected objects")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    p.add_argument("--render-limit", type=int, default=-1, help="debug cap for artifact rendering")
    p.add_argument("--report-path", default=str(TRELLIS_PATH / "code_update/part_promptable_seg.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "stage_progress.jsonl"
    data_config = Path(args.data_config)
    split_json = Path(args.split_json)
    part_seg_ckpt = Path(args.part_seg_ckpt)
    ss_flow_ckpt = Path(args.ss_flow_ckpt)
    ss_decoder_ckpt = Path(args.ss_decoder_ckpt)

    dc = load_data_config(data_config)
    ds = _dataset_for("four", dc)
    selected, selection_manifest = build_selection(ds, split_json, per_split=int(args.per_split))
    b_subset = choose_b_subset(selected, limit=int(args.b_limit))
    if args.max_a >= 0:
        selected_a = selected[: int(args.max_a)]
    else:
        selected_a = selected
    if args.max_b >= 0:
        selected_b = b_subset[: int(args.max_b)]
    else:
        selected_b = b_subset

    selection_manifest["b_subset"] = [asdict(sample) for sample in b_subset]
    _write_json(out_dir / "selection.json", selection_manifest)
    print(f"[eval_0615] selected A={len(selected_a)}/{len(selected)} B={len(selected_b)}/{len(b_subset)} out={out_dir}", flush=True)
    print(f"[eval_0615] counts={selection_manifest['counts']}", flush=True)
    if args.plan_only:
        return 0

    peak_vram: dict[str, int] = {"A": 0, "B": 0}
    stage_records: list[dict[str, Any]] = []
    b_status = "skipped" if args.skip_b else "not_started"
    b_error = ""

    if not args.skip_a:
        for idx, sample in enumerate(selected_a, 1):
            print(f"[eval_0615] A {idx}/{len(selected_a)} {sample.split} {sample.obj_id} angle={sample.angle_idx} bucket={sample.bucket}", flush=True)
            for stage in ("ss", "part"):
                spec = command_for(
                    out_dir=out_dir,
                    link="A",
                    stage=stage,
                    sample=sample,
                    data_config=data_config,
                    part_seg_ckpt=part_seg_ckpt,
                    ss_flow_ckpt=ss_flow_ckpt,
                    ss_decoder_ckpt=ss_decoder_ckpt,
                    gpu=args.gpu,
                )
                rec = execute_command(spec, gpu=args.gpu, stage_name=f"A/{stage}/{sample.obj_id}/{sample.angle_idx}", progress_path=progress_path)
                peak_vram["A"] = max(peak_vram["A"], int(rec["peak_vram_mib"]))
                stage_records.append(rec)
                if stage == "part":
                    render_artifacts(ds, [sample], out_dir=out_dir, links=["A"])

    if not args.skip_b:
        b_status = "running"
        try:
            for idx, sample in enumerate(selected_b, 1):
                print(f"[eval_0615] B {idx}/{len(selected_b)} {sample.split} {sample.obj_id} angle={sample.angle_idx} bucket={sample.bucket}", flush=True)
                for stage in ("ss", "part"):
                    spec = command_for(
                        out_dir=out_dir,
                        link="B",
                        stage=stage,
                        sample=sample,
                        data_config=data_config,
                        part_seg_ckpt=part_seg_ckpt,
                        ss_flow_ckpt=ss_flow_ckpt,
                        ss_decoder_ckpt=ss_decoder_ckpt,
                        gpu=args.gpu,
                    )
                    rec = execute_command(spec, gpu=args.gpu, stage_name=f"B/{stage}/{sample.obj_id}/{sample.angle_idx}", progress_path=progress_path)
                    peak_vram["B"] = max(peak_vram["B"], int(rec["peak_vram_mib"]))
                    stage_records.append(rec)
                    if stage == "part":
                        render_artifacts(ds, [sample], out_dir=out_dir, links=["B"])
            b_status = "completed"
        except Exception as exc:
            b_status = "failed"
            b_error = str(exc)
            print(f"[eval_0615][WARN] B failed: {b_error}", flush=True)

    _write_json(out_dir / "stage_records.json", stage_records)
    _write_json(out_dir / "peak_vram.json", peak_vram)

    completed_links = []
    if not args.skip_a:
        completed_links.append("A")
    if not args.skip_b:
        completed_links.append("B")
    metrics_selection = list({(s.obj_id, s.angle_idx, s.split): s for s in [*selected_a, *selected_b]}.values())
    metric_rows = collect_metrics(ds, metrics_selection, out_dir=out_dir, links=completed_links, peak_vram=peak_vram)
    summary_rows = summarize_metrics(metric_rows, peak_vram)
    _write_json(out_dir / "metrics.json", metric_rows)
    _write_json(out_dir / "metrics_summary.json", summary_rows)
    write_metrics_csv(out_dir / "metrics.csv", metric_rows)
    write_summary_csv(out_dir / "metrics_summary.csv", summary_rows)
    render_table_png(
        out_dir / "metrics.png",
        metric_rows,
        columns=["split", "obj_id", "angle", "part_name", "bucket", "raw_voxels", "link", "IoU", "hit@0.5"],
        title="0615-1 detail metrics: one row per part",
        max_rows=220,
    )
    render_table_png(
        out_dir / "metrics_summary.png",
        summary_rows,
        columns=["split", "bucket", "link", "n", "mean_IoU", "success@IoU0.5", "peak_vram_mib"],
        title="0615-1 summary metrics",
        max_rows=None,
    )
    render_samples = metrics_selection if args.render_limit < 0 else metrics_selection[: int(args.render_limit)]
    render_artifacts(ds, render_samples, out_dir=out_dir, links=completed_links)
    append_report(
        Path(args.report_path),
        out_dir=out_dir,
        selected=selected,
        selection_manifest=selection_manifest,
        summary=summary_rows,
        peak_vram=peak_vram,
        b_status=b_status,
        b_error=b_error,
        args=args,
    )
    print(f"[eval_0615] done metrics={out_dir / 'metrics_summary.json'} report={args.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
