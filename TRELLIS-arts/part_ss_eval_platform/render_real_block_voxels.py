from __future__ import annotations

import argparse
import csv
import json
import sys
import types
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

from part_ss_eval_platform.eval_0615 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    load_data_config,
    _dataset_for,
    _find_sample,
    load_npz_coords,
    render_table_png,
    summarize_metrics,
    write_metrics_csv,
    write_summary_csv,
)


DEFAULT_ROOT = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0615-1")
COLORS = np.asarray(
    [
        (0.86, 0.16, 0.16, 0.95),
        (0.12, 0.38, 0.72, 0.95),
        (0.18, 0.62, 0.25, 0.95),
        (0.95, 0.49, 0.12, 0.95),
        (0.50, 0.28, 0.72, 0.95),
        (0.10, 0.66, 0.72, 0.95),
        (0.86, 0.36, 0.66, 0.95),
        (0.55, 0.32, 0.24, 0.95),
    ],
    dtype=np.float32,
)
BODY_COLOR = np.asarray((0.60, 0.60, 0.60, 0.16), dtype=np.float32)
WHOLE_COLOR = np.asarray((0.20, 0.47, 0.62, 0.88), dtype=np.float32)


def _coords_to_occ(coords: np.ndarray) -> np.ndarray:
    occ = np.zeros((64, 64, 64), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size:
        valid = np.all((coords >= 0) & (coords < 64), axis=1)
        coords = coords[valid]
        occ[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return occ


def _surface_only(occ: np.ndarray) -> np.ndarray:
    if not bool(occ.any()):
        return occ
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return occ & ~interior


def _project_iso(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    u = (x - y) * 0.86
    v = (x + y) * 0.43 - z * 0.92
    depth = x + y + z
    return u, v, depth


def _draw_block_projection(
    path: Path,
    layers: list[tuple[np.ndarray, tuple[int, int, int, int], int]],
    *,
    title: str,
    image_size: int = 1100,
) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    coords_all = [np.asarray(coords, dtype=np.float32).reshape(-1, 3) for coords, _color, _size in layers if np.asarray(coords).size]
    image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    if not coords_all:
        image.save(path)
        return
    all_coords = np.concatenate(coords_all, axis=0)
    all_u, all_v, _ = _project_iso(all_coords)
    margin = 70
    span = max(float(all_u.max() - all_u.min()), float(all_v.max() - all_v.min()), 1.0)
    scale = (image_size - 2 * margin) / span
    u_mid = float((all_u.max() + all_u.min()) * 0.5)
    v_mid = float((all_v.max() + all_v.min()) * 0.5)
    cx = cy = image_size * 0.5

    tiles: list[tuple[float, float, float, tuple[int, int, int, int], int]] = []
    for coords, color, size in layers:
        coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
        if coords.size == 0:
            continue
        u, v, depth = _project_iso(coords)
        px = (u - u_mid) * scale + cx
        py = (v - v_mid) * scale + cy
        for x, y, d in zip(px.tolist(), py.tolist(), depth.tolist()):
            tiles.append((float(d), float(x), float(y), color, int(size)))
    tiles.sort(key=lambda item: item[0])
    for _d, x, y, color, size in tiles:
        half = max(1, int(size) // 2)
        draw.rectangle((x - half, y - half, x + half, y + half), fill=color)
    draw.text((18, 18), title, fill=(20, 20, 20, 255))
    image.save(path)


def render_whole_block(path: Path, coords: np.ndarray, *, title: str) -> None:
    occ = _surface_only(_coords_to_occ(coords))
    coords = np.argwhere(occ).astype(np.int32)
    _draw_block_projection(path, [(coords, (44, 111, 145, 230), 6)], title=title)


def render_parts_block(path: Path, whole_coords: np.ndarray, part_items: list[tuple[str, np.ndarray]], *, title: str) -> None:
    whole_occ = _surface_only(_coords_to_occ(whole_coords))
    part_union = np.zeros((64, 64, 64), dtype=bool)
    layers: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for idx, (_name, coords) in enumerate(part_items):
        part_occ = _surface_only(_coords_to_occ(coords))
        if not bool(part_occ.any()):
            continue
        part_union |= part_occ
        rgb = tuple(int(round(float(c) * 255)) for c in COLORS[idx % len(COLORS)][:3])
        layers.append((np.argwhere(part_occ).astype(np.int32), (*rgb, 245), 7))
    body = whole_occ & ~part_union
    if bool(body.any()):
        grid = np.indices(body.shape)
        body &= (grid[0] + grid[1] + grid[2]) % 3 == 0
        layers.insert(0, (np.argwhere(body).astype(np.int32), (150, 150, 150, 48), 4))
    _draw_block_projection(path, layers, title=title)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_dir(root: Path, split: str, obj_id: str, angle_idx: int) -> Path:
    return root / "platform_runs" / "B" / split / f"{obj_id}-{int(angle_idx)}" / "eval-B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render real-inference B results as block voxel PNGs.")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--out-subdir", default="real_only_block")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_root = root / args.out_subdir
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    b_metrics = [row for row in metrics if row.get("link") == "B"]
    peak_vram = json.loads((root / "peak_vram.json").read_text(encoding="utf-8"))
    b_summary = summarize_metrics(b_metrics, {"B": int(peak_vram.get("B", 0))})
    b_summary = [row for row in b_summary if row.get("link") == "B"]

    _write_json(out_root / "metrics_real.json", b_metrics)
    _write_json(out_root / "metrics_summary_real.json", b_summary)
    write_metrics_csv(out_root / "metrics_real.csv", b_metrics)
    write_summary_csv(out_root / "metrics_summary_real.csv", b_summary)
    render_table_png(
        out_root / "metrics_real.png",
        b_metrics,
        columns=["split", "obj_id", "angle", "part_name", "bucket", "raw_voxels", "link", "IoU", "hit@0.5"],
        title="0615-1 real-inference B detail metrics",
        max_rows=None,
    )
    render_table_png(
        out_root / "metrics_summary_real.png",
        b_summary,
        columns=["split", "bucket", "link", "n", "mean_IoU", "success@IoU0.5", "peak_vram_mib"],
        title="0615-1 real-inference B summary metrics",
        max_rows=None,
    )

    dc = load_data_config(Path(args.data_config))
    ds = _dataset_for("four", dc)
    rendered = 0
    for row in selection["b_subset"]:
        split = str(row["split"])
        obj_id = str(row["obj_id"])
        angle_idx = int(row["angle_idx"])
        run_dir = _run_dir(root, split, obj_id, angle_idx)
        whole_path = run_dir / "voxel.npz"
        parts_dir = run_dir / "parts"
        if not whole_path.is_file() or not parts_dir.is_dir():
            print(f"[render_real_block_voxels][WARN] missing B run: {run_dir}", flush=True)
            continue
        _sample_idx, sample = _find_sample(ds, obj_id, angle_idx)
        whole_coords = load_npz_coords(whole_path)
        part_items: list[tuple[str, np.ndarray]] = []
        for part_idx, part in enumerate(sample["parts"]):
            part_path = parts_dir / f"part_{part_idx:02d}_voxel.npz"
            if part_path.is_file():
                part_items.append((str(part["part_name"]), load_npz_coords(part_path)))
        out_dir = out_root / split / obj_id / str(angle_idx)
        render_whole_block(
            out_dir / "stage1_whole.png",
            whole_coords,
            title=f"real B whole {split} {obj_id} angle {angle_idx}",
        )
        render_parts_block(
            out_dir / "stage2_parts.png",
            whole_coords,
            part_items,
            title=f"real B parts {split} {obj_id} angle {angle_idx}",
        )
        rendered += 1
        print(f"[render_real_block_voxels] {rendered}/{len(selection['b_subset'])} {split} {obj_id} angle={angle_idx}", flush=True)
    print(f"[render_real_block_voxels] done -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
