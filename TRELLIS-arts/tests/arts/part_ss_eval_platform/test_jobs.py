from pathlib import Path

import pytest

from part_ss_eval_platform.jobs import JobRequest, build_command


def test_legacy_eval_job_launcher_is_removed_from_active_tree():
    req = JobRequest(
        task_type="eval",
        view_mode="four",
        experiment_name="exp_eval4",
        output_root="/tmp/out",
        checkpoint="/tmp/ckpt.pt",
        gpu_ids="0,1",
        max_samples=100,
        sample_mode="first",
        object_ids="1,2",
        overrides=["loss.velocity_contrastive_weight=0.02"],
    )

    with pytest.raises(RuntimeError, match="Legacy part_ss_latent_flow"):
        build_command(req, repo_root=Path("/repo"))


def test_legacy_test_export_launcher_is_removed_from_active_tree():
    req = JobRequest(
        task_type="test",
        view_mode="single",
        experiment_name="exp_test1",
        output_root="/tmp/out",
        checkpoint="/tmp/ckpt.pt",
        gpu_ids="2",
    )

    with pytest.raises(RuntimeError, match="scripts/eval/run_ee_eval.bash"):
        build_command(req, repo_root=Path("/repo"))


def test_job_request_still_requires_checkpoint_or_load_dir_step():
    req = JobRequest(
        task_type="eval",
        view_mode="four",
        experiment_name="bad",
        output_root="/tmp/out",
    )

    with pytest.raises(ValueError, match="checkpoint or load_dir\\+step"):
        build_command(req, repo_root=Path("/repo"))
