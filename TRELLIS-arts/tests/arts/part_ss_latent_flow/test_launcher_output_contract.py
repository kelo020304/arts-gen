from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_legacy_part_ss_latent_flow_launchers_are_archived():
    archive = ROOT / "scripts/_archive/2026-06-train-launchers"
    assert archive.is_dir()
    assert (ROOT / "scripts/eval/run_ee_eval.bash").is_file()
    assert (ROOT / "scripts/train/part_promptable_seg/run_train.bash").is_file()
