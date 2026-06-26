from pathlib import Path
from unittest.mock import patch

from post_process.kinematic_solver.utils.run_all import run_pipeline


def test_run_all_orders_subprocess_stages_and_can_skip_validate(tmp_path):
    calls = []

    with patch("post_process.kinematic_solver.utils.run_all.preflight_inputs") as preflight, \
         patch("post_process.kinematic_solver.utils.run_all.write_dataset_fingerprint") as fp, \
         patch("post_process.kinematic_solver.utils.run_all.verify_cache") as verify, \
         patch("post_process.kinematic_solver.utils.run_all._shell", side_effect=lambda cmd: calls.append(cmd)):
        run_pipeline(
            converter_output_root=tmp_path / "converter",
            source_root=tmp_path / "source",
            run_output_dir=tmp_path / "run",
            spike_result=Path("spike.md"),
            object_ids=["ra_007"],
            skip_validate=True,
        )

    preflight.assert_called_once()
    fp.assert_called_once()
    verify.assert_called_once()
    modules = [cmd[2] for cmd in calls]
    assert modules == [
        "post_process.kinematic_solver.utils.data_prep",
        "post_process.kinematic_solver.utils.data_prep",
        "post_process.kinematic_solver.utils.run_solver",
        "post_process.kinematic_solver.utils.run_compare",
        "post_process.kinematic_solver.utils.report_summary",
    ]


def test_run_all_skips_fingerprint_write_when_fingerprint_exists(tmp_path):
    calls = []
    run = tmp_path / "run"
    run.mkdir()
    (run / "dataset_fingerprint.json").write_text("{}\n")

    with patch("post_process.kinematic_solver.utils.run_all.preflight_inputs"), \
         patch("post_process.kinematic_solver.utils.run_all.write_dataset_fingerprint") as fp, \
         patch("post_process.kinematic_solver.utils.run_all.verify_cache"), \
         patch("post_process.kinematic_solver.utils.run_all._shell", side_effect=lambda cmd: calls.append(cmd)):
        run_pipeline(
            converter_output_root=tmp_path / "converter",
            source_root=tmp_path / "source",
            run_output_dir=run,
            spike_result=None,
            object_ids=["ra_007"],
            skip_validate=True,
        )

    fp.assert_not_called()
    assert calls


def test_run_all_skip_visualization_omits_solver_and_compare_visual_flags(tmp_path):
    calls = []

    with patch("post_process.kinematic_solver.utils.run_all.preflight_inputs"), \
         patch("post_process.kinematic_solver.utils.run_all.write_dataset_fingerprint"), \
         patch("post_process.kinematic_solver.utils.run_all.verify_cache"), \
         patch("post_process.kinematic_solver.utils.run_all._shell", side_effect=lambda cmd: calls.append(cmd)):
        run_pipeline(
            converter_output_root=tmp_path / "converter",
            source_root=tmp_path / "source",
            run_output_dir=tmp_path / "run",
            spike_result=None,
            object_ids=["ra_007"],
            skip_validate=True,
            skip_visualization=True,
        )

    solver_cmd = next(cmd for cmd in calls if cmd[2] == "post_process.kinematic_solver.utils.run_solver")
    compare_cmd = next(cmd for cmd in calls if cmd[2] == "post_process.kinematic_solver.utils.run_compare")
    assert "--write-visualization" not in solver_cmd
    assert "--write-overlays" not in compare_cmd


def test_run_all_passes_viz_stride_to_solver(tmp_path):
    calls = []

    with patch("post_process.kinematic_solver.utils.run_all.preflight_inputs"), \
         patch("post_process.kinematic_solver.utils.run_all.write_dataset_fingerprint"), \
         patch("post_process.kinematic_solver.utils.run_all.verify_cache"), \
         patch("post_process.kinematic_solver.utils.run_all._shell", side_effect=lambda cmd: calls.append(cmd)):
        run_pipeline(
            converter_output_root=tmp_path / "converter",
            source_root=tmp_path / "source",
            run_output_dir=tmp_path / "run",
            spike_result=None,
            object_ids=["ra_007"],
            skip_validate=True,
            viz_stride=1,
        )

    solver_cmd = next(cmd for cmd in calls if cmd[2] == "post_process.kinematic_solver.utils.run_solver")
    assert "--viz-stride" in solver_cmd
    assert solver_cmd[solver_cmd.index("--viz-stride") + 1] == "1"
