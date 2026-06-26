"""Aggregate KinematicSolver comparison rows into a compact report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize_rows(rows: list[dict]) -> dict:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    partial_rows = [row for row in rows if row.get("status") != "ok"]

    def success_fraction(side: str) -> str:
        eligible = [row for row in rows if row.get(f"status_{side}") == "ok"]
        successes = [row for row in eligible if row.get(f"success_{side}") is True]
        return f"{len(successes)}/{len(eligible)}" if eligible else "0/0"

    ious = [float(row["iou_range"]) for row in ok_rows if row.get("iou_range") is not None]
    return {
        "n_total": len(rows),
        "n_ok": len(ok_rows),
        "n_partial": len(partial_rows),
        "succ_upper_all": success_fraction("upper"),
        "succ_lower_all": success_fraction("lower"),
        "iou_mean_ok": statistics.mean(ious) if ious else None,
    }


def write_report_summary(run_output_dir: Path, rows: list[dict]) -> Path:
    summary = summarize_rows(rows)
    json_path = run_output_dir / "report_summary.json"
    md_path = run_output_dir / "report_summary.md"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(
        "# V1 KinematicSolver internal report\n\n"
        f"- total joints: **{summary['n_total']}**\n"
        f"- status ok / non-ok: {summary['n_ok']} / {summary['n_partial']}\n"
        f"- success rate upper / lower: "
        f"{summary['succ_upper_all']} / {summary['succ_lower_all']}\n"
        f"- mean range IoU on ok joints: {summary['iou_mean_ok']}\n"
    )
    return md_path


def _load_rows(run_output_dir: Path, object_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for object_id in object_ids:
        path = run_output_dir / object_id / "comparison.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V1 KinematicSolver comparisons")
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--object-ids", required=True)
    args = parser.parse_args()

    object_ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    report = write_report_summary(args.run_output_dir, _load_rows(args.run_output_dir, object_ids))
    print(report.read_text())


if __name__ == "__main__":
    main()
