from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ROOT = Path("/robot/data-lab/jzh/art-gen-output/EE-eval/0615-1")
COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#bcbd22",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#c5b0d5",
    "#f7b6d2",
    "#9edae5",
]


def load_coords(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["coords"], dtype=np.int16).reshape(-1, 3)


def _hex_to_rgba(value: str, alpha: float = 0.96) -> tuple[float, float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        float(alpha),
    )


def label_grid(part_paths: list[Path]) -> np.ndarray:
    grid = np.zeros((64, 64, 64), dtype=np.int16)
    for label, path in enumerate(part_paths, start=1):
        coords = load_coords(path)
        if coords.size == 0:
            continue
        valid = np.all((coords >= 0) & (coords < 64), axis=1)
        coords = coords[valid]
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = int(label)
    return grid


def cube_faces_for_labels(grid: np.ndarray) -> tuple[list[list[tuple[int, int, int]]], list[tuple[float, float, float, float]]]:
    faces: list[list[tuple[int, int, int]]] = []
    colors: list[tuple[float, float, float, float]] = []
    dirs = [
        ((1, 0, 0), lambda x, y, z: [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]),
        ((-1, 0, 0), lambda x, y, z: [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)]),
        ((0, 1, 0), lambda x, y, z: [(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)]),
        ((0, -1, 0), lambda x, y, z: [(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)]),
        ((0, 0, 1), lambda x, y, z: [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]),
        ((0, 0, -1), lambda x, y, z: [(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)]),
    ]
    occupied = np.argwhere(grid > 0)
    for x, y, z in occupied:
        label = int(grid[x, y, z])
        color = _hex_to_rgba(COLORS[(label - 1) % len(COLORS)])
        for (dx, dy, dz), make_face in dirs:
            nx, ny, nz = int(x + dx), int(y + dy), int(z + dz)
            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64 or nz < 0 or nz >= 64 or grid[nx, ny, nz] == 0:
                faces.append(make_face(int(x), int(y), int(z)))
                colors.append(color)
    return faces, colors


def render_cube_result(path: Path, part_paths: list[Path], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path.parent.mkdir(parents=True, exist_ok=True)
    grid = label_grid(part_paths)
    faces, facecolors = cube_faces_for_labels(grid)
    fig = plt.figure(figsize=(7.4, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    if faces:
        poly = Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=(0.08, 0.08, 0.08, 0.20),
            linewidths=0.08,
        )
        ax.add_collection3d(poly)
    coords = np.argwhere(grid > 0)
    if coords.size:
        lo = coords.min(axis=0).astype(float)
        hi = coords.max(axis=0).astype(float) + 1.0
        center = (lo + hi) * 0.5
        radius = max(float((hi - lo).max()) * 0.58, 4.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    else:
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 64)
        ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-45)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render real B outputs as true cube voxel result images.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-subdir", default="real_infer_cubes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_root = root / args.out_subdir
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    metrics = [row for row in json.loads((root / "metrics.json").read_text(encoding="utf-8")) if row.get("link") == "B"]
    summary = [row for row in json.loads((root / "metrics_summary.json").read_text(encoding="utf-8")) if row.get("link") == "B"]
    write_csv(
        out_root / "metrics.csv",
        metrics,
        ["split", "obj_id", "angle", "part_name", "bucket", "raw_voxels", "IoU", "hit@0.5", "pred_voxels"],
    )
    write_csv(
        out_root / "metrics_summary.csv",
        summary,
        ["split", "bucket", "n", "mean_IoU", "success@IoU0.5", "peak_vram_mib"],
    )

    rendered = 0
    for row in selection["b_subset"]:
        split = str(row["split"])
        obj_id = str(row["obj_id"])
        angle = int(row["angle_idx"])
        run_dir = root / "platform_runs" / "B" / split / f"{obj_id}-{angle}" / "eval-B"
        part_paths = sorted((run_dir / "parts").glob("part_*_voxel.npz"))
        out_dir = out_root / split / obj_id / str(angle)
        render_cube_result(
            out_dir / "result.png",
            part_paths,
            title=f"real inference result: {split} {obj_id} angle {angle}",
        )
        rendered += 1
        print(f"[render_real_cube_results] {rendered}/32 {split} {obj_id} angle={angle}", flush=True)
    print(f"[render_real_cube_results] done -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
