"""End-to-end KinematicSolver V1 orchestration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import V1_COACD_RUN_PARAMS, V1_TEN_IDS, V1_VHACD_CACHE_METADATA, V1DatasetRoots
from .phase0_verify import preflight_inputs, verify_cache, write_dataset_fingerprint


def _shell(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_pipeline(
    *,
    converter_output_root: Path,
    source_root: Path,
    run_output_dir: Path,
    spike_result: Path | None,
    object_ids: list[str],
    skip_validate: bool = False,
    skip_visualization: bool = False,
    viz_stride: int = 5,
) -> None:
    roots = V1DatasetRoots(
        converter_output_root=converter_output_root,
        source_root=source_root,
    )
    preflight_inputs(roots, object_ids)
    common = [
        sys.executable,
        "-m",
        "post_process.kinematic_solver.utils.data_prep",
        "--converter-output-root",
        str(roots.converter_output_root),
        "--source-root",
        str(roots.source_root),
        "--object-ids",
        ",".join(object_ids),
    ]
    _shell(common + ["--stage", "default"])
    _shell(common + ["--stage", "vhacd"])

    if not (run_output_dir / "dataset_fingerprint.json").is_file():
        write_dataset_fingerprint(roots, run_output_dir, object_ids)
    verify_cache(
        roots,
        run_output_dir,
        object_ids,
        expected_coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        expected_vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
    )

    solver_cmd = [
        sys.executable,
        "-m",
        "post_process.kinematic_solver.utils.run_solver",
        "--converter-output-root",
        str(roots.converter_output_root),
        "--source-root",
        str(roots.source_root),
        "--run-output-dir",
        str(run_output_dir),
        "--object-ids",
        ",".join(object_ids),
    ]
    if spike_result is not None:
        solver_cmd.extend(["--spike-result", str(spike_result)])
    if not skip_visualization:
        solver_cmd.append("--write-visualization")
        solver_cmd.extend(["--viz-stride", str(viz_stride)])
    _shell(solver_cmd)

    if not skip_validate:
        _shell([
            sys.executable,
            "-m",
            "post_process.kinematic_solver.utils.run_validate",
            "--converter-output-root",
            str(roots.converter_output_root),
            "--source-root",
            str(roots.source_root),
            "--run-output-dir",
            str(run_output_dir),
            "--object-ids",
            ",".join(object_ids),
        ])

    compare_cmd = [
        sys.executable,
        "-m",
        "post_process.kinematic_solver.utils.run_compare",
        "--converter-output-root",
        str(roots.converter_output_root),
        "--source-root",
        str(roots.source_root),
        "--run-output-dir",
        str(run_output_dir),
        "--object-ids",
        ",".join(object_ids),
    ]
    if not skip_visualization:
        compare_cmd.append("--write-overlays")
    _shell(compare_cmd)
    _shell([
        sys.executable,
        "-m",
        "post_process.kinematic_solver.utils.report_summary",
        "--run-output-dir",
        str(run_output_dir),
        "--object-ids",
        ",".join(object_ids),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V1 KinematicSolver end to end")
    parser.add_argument("--converter-output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--spike-result", type=Path)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument(
        "--viz-stride",
        type=int,
        default=5,
        help="Render every Nth solver trace sample; use 1 for every step.",
    )
    args = parser.parse_args()

    run_pipeline(
        converter_output_root=args.converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
        run_output_dir=args.run_output_dir,
        spike_result=args.spike_result,
        object_ids=[s.strip() for s in args.object_ids.split(",") if s.strip()],
        skip_validate=args.skip_validate,
        skip_visualization=args.skip_visualization,
        viz_stride=args.viz_stride,
    )


if __name__ == "__main__":
    main()
