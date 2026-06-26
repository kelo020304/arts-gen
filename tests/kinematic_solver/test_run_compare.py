import json
import subprocess
import sys

from post_process.kinematic_solver.utils.config import V1DatasetRoots
from post_process.kinematic_solver.utils.phase0_verify import write_dataset_fingerprint


def test_run_compare_writes_comparison_jsonl(tmp_path):
    converter = tmp_path / "converter"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    (obj_dir / "body.obj").write_text("v 0 0 0\n")
    (converter / "raw/gt_limits").mkdir(parents=True)
    (converter / "raw/gt_limits/ra_007.json").write_text(json.dumps({
        "object_id": "ra_007",
        "limits": {"joint0": {"lower": 0.0, "upper": 0.10}},
    }))
    (run / "ra_007").mkdir(parents=True)
    (run / "ra_007/predictions.jsonl").write_text(json.dumps({
        "object_id": "ra_007",
        "joint_name": "joint0",
        "type": "prismatic",
        "canonical_unit": "meters",
        "predicted_lower": 0.0,
        "predicted_upper": 0.10,
        "status": "ok",
        "status_lower": "ok",
        "status_upper": "ok",
    }) + "\n")
    write_dataset_fingerprint(
        V1DatasetRoots(converter_output_root=converter, source_root=tmp_path / "source"),
        run,
        ids=["ra_007"],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.utils.run_compare",
            "--converter-output-root", str(converter),
            "--run-output-dir", str(run),
            "--object-ids", "ra_007",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    rows = [
        json.loads(line)
        for line in (run / "ra_007/comparison.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["success"] is True
    assert rows[0]["iou_range"] == 1.0
