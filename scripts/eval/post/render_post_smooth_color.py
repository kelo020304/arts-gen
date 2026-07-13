#!/usr/bin/env python3
"""Render colored before/after panels from existing post-smooth reports."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any

import trimesh

from scripts.eval.post.holopart_smooth import (
    _component_list,
    _load_json,
    _load_mesh,
    _load_render_camera,
    _safe_name,
    _write_component_panels,
    _write_exploded_panel,
    _write_overview_panel,
)


def _after_meshes_from_report(report: dict[str, Any]) -> dict[str, trimesh.Trimesh]:
    out: dict[str, trimesh.Trimesh] = {}
    for row in report.get("metrics") or []:
        label = str(row.get("component"))
        path = Path(str(row.get("after_mesh")))
        if label and path.is_file():
            out[label] = _load_mesh(path)
    return out


def render_report(report_path: Path, *, out_dir: Path | None, resolution: int, render_view: int) -> Path:
    report_path = Path(report_path).resolve()
    report = _load_json(report_path)
    target = Path(out_dir).resolve() if out_dir else report_path.parent / "color_renders"
    target.mkdir(parents=True, exist_ok=True)
    _run_dir, _whole_voxel, components, meta = _component_list(Path(report["summary_path"]))
    summary = meta["summary"]
    after_meshes = _after_meshes_from_report(report)
    if not after_meshes:
        raise ValueError(f"{report_path}: report has no matched after meshes")
    extrinsic, intrinsic = _load_render_camera(summary, render_view=int(render_view))
    _write_component_panels(
        components,
        after_meshes,
        target,
        method=str(report.get("method", "post")),
        max_faces=0,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
    )
    _write_overview_panel(
        components,
        after_meshes,
        target / "before_after_overview_color.png",
        method=str(report.get("method", "post")),
        max_faces=0,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
    )
    _write_exploded_panel(
        components,
        after_meshes,
        target / "after_exploded_overview_color.png",
        max_faces=0,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=int(resolution),
    )
    # Keep stable names near the old panels for quick browsing.
    if out_dir is None:
        for src_name, dst_name in (
            ("before_after_overview_color.png", "before_after_overview_color.png"),
            ("after_exploded_overview_color.png", "after_exploded_overview_color.png"),
        ):
            src = target / src_name
            if src.is_file():
                shutil.copy2(src, report_path.parent / dst_name)
        panel_src = target / "panels"
        panel_dst = report_path.parent / "panels_color"
        if panel_dst.exists():
            shutil.rmtree(panel_dst)
        if panel_src.is_dir():
            shutil.copytree(panel_src, panel_dst)
    manifest = {
        "report": str(report_path),
        "out_dir": str(target),
        "overview": str(target / "before_after_overview_color.png"),
        "exploded": str(target / "after_exploded_overview_color.png"),
        "panel_count": len(list((target / "panels").glob("*.png"))) if (target / "panels").is_dir() else 0,
    }
    (target / "color_render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--report-glob", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--render-view", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [Path(path) for path in args.report]
    if args.report_glob:
        reports.extend(Path(path) for path in sorted(glob.glob(args.report_glob)))
    if not reports:
        raise SystemExit("provide --report or --report-glob")
    for report in reports:
        out = render_report(report, out_dir=args.out_dir, resolution=int(args.resolution), render_view=int(args.render_view))
        print(f"[color-render] {report} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
