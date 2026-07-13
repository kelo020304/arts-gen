#!/usr/bin/env python3
"""Write final verdict tables for post-smoothing experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _collect_reports(root: Path, label: str, globpat: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for report_path in sorted(root.glob(globpat)):
        report = _load_json(report_path)
        rows = report.get("metrics") or []
        for row in rows:
            metrics.append({**row, "tool": label, "object_key": report_path.parent.name})
        summaries.append(
            {
                "tool": label,
                "object_key": report_path.parent.name,
                "report": str(report_path),
                "overview": str(report_path.parent / "before_after_overview_color.png"),
                "exploded": str(report_path.parent / "after_exploded_overview_color.png"),
                "components": len(rows),
            }
        )
    return metrics, summaries


def _rollup(label: str, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"tool": label}
    return {
        "tool": label,
        "objects": len(summaries),
        "components": len(rows),
        "watertight_after": sum(1 for row in rows if bool(row.get("after_is_watertight"))),
        "watertight_total": len(rows),
        "min_coverage_0p01": min(float(row.get("before_to_after_coverage_0p01")) for row in rows),
        "min_coverage_0p02": min(float(row.get("before_to_after_coverage_0p02")) for row in rows),
        "max_before_to_after_p95": max(float(row.get("before_to_after_p95")) for row in rows),
        "max_after_to_overall_p95": max(float(row.get("after_to_overall_p95")) for row in rows),
        "max_bidirectional_p95": max(float(row.get("bidirectional_chamfer_p95_max")) for row in rows),
        "mean_seconds": statistics.fmean(float(row.get("seconds")) for row in rows),
        "mean_before_dihedral": statistics.fmean(float(row.get("before_mean_dihedral_rad")) for row in rows),
        "mean_after_dihedral": statistics.fmean(float(row.get("after_mean_dihedral_rad")) for row in rows),
        "worst_component": min(rows, key=lambda row: float(row.get("before_to_after_coverage_0p01"))).get("component"),
        "worst_object": min(rows, key=lambda row: float(row.get("before_to_after_coverage_0p01"))).get("object_key"),
    }


def _xpart_curve(root: Path) -> list[dict[str, Any]]:
    objects = [
        "phyx-verse__0786542d0f7549208f889113fc384a7f__angle_00",
        "phyx-verse__0a46621504c24197b5653608f474f73b__angle_00",
    ]
    configs = [
        ("steps8_oct256", "xpart/{object}/report.json"),
        ("steps25_oct256", "xpart_sweep/steps25_oct256/{object}/report.json"),
        ("steps50_oct256", "xpart_sweep/steps50_oct256/{object}/report.json"),
        ("steps8_oct384", "xpart_sweep/steps8_oct384/{object}/report.json"),
    ]
    out = []
    for object_key in objects:
        for config, pattern in configs:
            path = root / pattern.format(object=object_key)
            if not path.is_file():
                continue
            report = _load_json(path)
            rows = report.get("metrics") or []
            worst = min(rows, key=lambda row: float(row.get("before_to_after_coverage_0p01")))
            out.append(
                {
                    "object_key": object_key,
                    "config": config,
                    "components": len(rows),
                    "watertight": sum(1 for row in rows if bool(row.get("after_is_watertight"))),
                    "min_coverage_0p01": min(float(row.get("before_to_after_coverage_0p01")) for row in rows),
                    "min_coverage_0p02": min(float(row.get("before_to_after_coverage_0p02")) for row in rows),
                    "max_before_to_after_p95": max(float(row.get("before_to_after_p95")) for row in rows),
                    "max_after_to_overall_p95": max(float(row.get("after_to_overall_p95")) for row in rows),
                    "mean_seconds": statistics.fmean(float(row.get("seconds")) for row in rows),
                    "worst_component": worst.get("component"),
                    "overview": str(path.parent / "before_after_overview_color.png"),
                }
            )
    return out


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    groups = [
        ("HoloPart component_scene", "holopart/*/report.json"),
        ("X-Part bbox steps8 oct256", "xpart/*/report.json"),
        ("X-Part bbox steps8 oct384 sweep2", "xpart_sweep/steps8_oct384/*/report.json"),
        ("classic trimesh_fill", "classic/trimesh_fill/*/report.json"),
        ("classic pymeshlab_close", "classic/pymeshlab_close/*/report.json"),
    ]
    all_metrics: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    rollups: list[dict[str, Any]] = []
    for label, pattern in groups:
        metrics, summaries = _collect_reports(root, label, pattern)
        all_metrics.extend(metrics)
        all_summaries.extend(summaries)
        rollups.append(_rollup(label, metrics, summaries))
    curve = _xpart_curve(root)
    _write_csv(root / "post_smooth_verdict_metrics.csv", all_metrics)
    _write_csv(root / "post_smooth_verdict_rollup.csv", rollups)
    _write_csv(root / "xpart_quality_time_curve.csv", curve)

    lines = [
        "# Post-Smoothing Fidelity Verdict",
        "",
        f"root: `{root}`",
        "",
        "## Metric Fix",
        "",
        "Watertight and one-way after->overall chamfer were insufficient: an empty frame can be watertight and every remaining after point can still sit near the original surface. The added completeness metric samples the before component surface and measures before->after distance with coverage@0.01/0.02. The left-door regression case is now caught.",
        "",
        "| case | old after->overall p95 | new before->after p95 | coverage@0.01 | coverage@0.02 | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    left_report = root / "xpart" / "phyx-verse__0786542d0f7549208f889113fc384a7f__angle_00" / "report.json"
    if left_report.is_file():
        rows = _load_json(left_report).get("metrics") or []
        left = next((row for row in rows if row.get("component") == "part_00_left_door_0"), None)
        if left:
            lines.append(
                "| X-Part steps8/oct256 left door | {a2o} | {b2a} | {cov01} | {cov02} | FAIL: deleted panel/empty frame |".format(
                    a2o=_fmt(left.get("after_to_overall_p95")),
                    b2a=_fmt(left.get("before_to_after_p95")),
                    cov01=_fmt(left.get("before_to_after_coverage_0p01"), 3),
                    cov02=_fmt(left.get("before_to_after_coverage_0p02"), 3),
                )
            )
    lines.extend(
        [
            "",
            "## Three-Way Rollup",
            "",
            "Fidelity priority is coverage first, then watertight, then smoothness. Bidir p95 is `max(after->overall p95, before->after p95)`.",
            "",
            "| tool | objects | components | watertight | min cov@0.01 | min cov@0.02 | max before->after p95 | max after->overall p95 | max bidir p95 | mean sec/component | dihedral before->after | worst component |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rollups:
        if "components" not in row:
            continue
        lines.append(
            "| {tool} | {objects} | {components} | {water}/{total} | {cov01} | {cov02} | {b2a} | {a2o} | {bidir} | {sec} | {bd}->{ad} | `{worst}` on `{obj}` |".format(
                tool=row["tool"],
                objects=row["objects"],
                components=row["components"],
                water=row["watertight_after"],
                total=row["watertight_total"],
                cov01=_fmt(row["min_coverage_0p01"], 3),
                cov02=_fmt(row["min_coverage_0p02"], 3),
                b2a=_fmt(row["max_before_to_after_p95"]),
                a2o=_fmt(row["max_after_to_overall_p95"]),
                bidir=_fmt(row["max_bidirectional_p95"]),
                sec=_fmt(row["mean_seconds"], 2),
                bd=_fmt(row["mean_before_dihedral"]),
                ad=_fmt(row["mean_after_dihedral"]),
                worst=row["worst_component"],
                obj=row["worst_object"],
            )
        )
    lines.extend(
        [
            "",
            "## X-Part Quality/Time Sweep",
            "",
            "| object | config | watertight | min cov@0.01 | min cov@0.02 | max before->after p95 | max after->overall p95 | sec/component | worst component | overview |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in curve:
        lines.append(
            "| `{object}` | `{config}` | {water}/{total} | {cov01} | {cov02} | {b2a} | {a2o} | {sec} | `{worst}` | `{overview}` |".format(
                object=row["object_key"],
                config=row["config"],
                water=row["watertight"],
                total=row["components"],
                cov01=_fmt(row["min_coverage_0p01"], 3),
                cov02=_fmt(row["min_coverage_0p02"], 3),
                b2a=_fmt(row["max_before_to_after_p95"]),
                a2o=_fmt(row["max_after_to_overall_p95"]),
                sec=_fmt(row["mean_seconds"], 2),
                worst=row["worst_component"],
                overview=row["overview"],
            )
        )
    lines.extend(
        [
            "",
            "## Classic Baseline Notes",
            "",
            "- `classic/trimesh_fill`: fidelity-first default candidate. It preserves all part surfaces in this set, catches no hallucination, is fastest, and keeps coverage high; it does not force every component watertight (12/14).",
            "- `classic/pymeshlab_close`: similar coverage and one extra watertight component (13/14), but much slower due to pymeshlab processing and subprocess overhead.",
            "- `classic/pymeshlab_close_remesh` was run on the double-door object only; it stayed faithful and watertight but body took about 70s.",
            "- `classic/o3d_alpha` and `classic/o3d_poisson` were smoke-tested on the double-door object. They preserved coverage but were not watertight; Poisson also worsened body after->overall distance, so they were not expanded to all 14 components.",
            "",
            "## Verdict",
            "",
            "- The fixed completeness metric invalidates the previous all-green X-Part conclusion. The original X-Part bbox steps8/oct256 deletes the left-door center panel: coverage@0.01 is about 0.598.",
            "- Raising X-Part steps alone does not fix the left door. Steps50/oct256 improves coverage but still fails. Octree384 fixes the double-door left panel at steps8, but hurts the multi-drawer body water/completeness. Use X-Part only behind per-component fidelity gates; no single tested global X-Part config dominates.",
            "- For pipeline default under fidelity-first criteria, use classic geometry repair first (`trimesh_fill` as fastest conservative default; `pymeshlab_close` if the extra watertight component is worth the cost). Keep X-Part as optional enhancement only when before->after coverage and bidirectional chamfer pass.",
            "- HoloPart remains out for this data: low coverage and 0/14 watertight on the appliance set despite official example passing.",
            "",
            "## Outputs",
            "",
            f"- `{root / 'post_smooth_verdict.md'}`",
            f"- `{root / 'post_smooth_verdict_rollup.csv'}`",
            f"- `{root / 'post_smooth_verdict_metrics.csv'}`",
            f"- `{root / 'xpart_quality_time_curve.csv'}`",
            "",
        ]
    )
    (root / "post_smooth_verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[verdict] wrote {root / 'post_smooth_verdict.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
