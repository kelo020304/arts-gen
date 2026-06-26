import json
import subprocess
import sys

from post_process.kinematic_solver.utils.config import V1_COACD_RUN_PARAMS, V1_VHACD_CACHE_METADATA, V1DatasetRoots
from post_process.kinematic_solver.utils.phase0_verify import write_dataset_fingerprint


def _cube(center):
    cx, cy, cz = center
    vertices = [
        [cx - 0.5, cy - 0.5, cz - 0.5], [cx + 0.5, cy - 0.5, cz - 0.5],
        [cx + 0.5, cy + 0.5, cz - 0.5], [cx - 0.5, cy + 0.5, cz - 0.5],
        [cx - 0.5, cy - 0.5, cz + 0.5], [cx + 0.5, cy - 0.5, cz + 0.5],
        [cx + 0.5, cy + 0.5, cz + 0.5], [cx - 0.5, cy + 0.5, cz + 0.5],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ]
    return vertices, faces


def _write_obj(path, center):
    vertices, faces = _cube(center)
    lines = []
    for x, y, z in vertices:
        lines.append(f"v {x} {y} {z}\n")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}\n")
    path.write_text("".join(lines))


def _write_cache(cache_root, part_name, center):
    vertices, faces = _cube(center)
    p = cache_root / f"{part_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "object_id": "ra_007",
        "part_name": part_name,
        "source_obj": f"{part_name}.obj",
        "source_sha256": "x",
        "vhacd_cache_metadata": dict(V1_VHACD_CACHE_METADATA),
        "coacd_run_params": dict(V1_COACD_RUN_PARAMS),
        "frame": "world_baked",
        "hulls": [{"hull_index": 0, "vertices": vertices, "faces": faces}],
        "n_hulls": 1,
    }))


def test_run_solver_writes_predictions_and_trace_for_synthetic_cache(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    _write_obj(obj_dir / "body.obj", [0.0, 0.0, 0.0])
    _write_obj(obj_dir / "part_00.obj", [2.0, 0.0, 0.0])
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps({
        "object_id": "ra_007",
        "joints": {
            "joint0": {
                "object_id": "ra_007",
                "joint_name": "joint0",
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [-1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            }
        },
    }))
    _write_cache(converter / "raw/vhacd/ra_007", "body", [0.0, 0.0, 0.0])
    _write_cache(converter / "raw/vhacd/ra_007", "part_00", [2.0, 0.0, 0.0])

    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.utils.run_solver",
            "--converter-output-root", str(converter),
            "--source-root", str(source),
            "--run-output-dir", str(run),
            "--object-ids", "ra_007",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    pred_path = run / "ra_007/predictions.jsonl"
    trace_path = run / "ra_007/trace.jsonl"
    assert pred_path.is_file()
    assert trace_path.is_file()
    pred = json.loads(pred_path.read_text().strip())
    assert pred["object_id"] == "ra_007"
    assert pred["joint_name"] == "joint0"
    assert pred["status"] == "ok"


def test_run_solver_writes_live_step_viewer_with_local_three_vendor(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    _write_obj(obj_dir / "body.obj", [0.0, 0.0, 0.0])
    _write_obj(obj_dir / "part_00.obj", [2.0, 0.0, 0.0])
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps({
        "object_id": "ra_007",
        "joints": {
            "joint0": {
                "object_id": "ra_007",
                "joint_name": "joint0",
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [-1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            }
        },
    }))
    _write_cache(converter / "raw/vhacd/ra_007", "body", [0.0, 0.0, 0.0])
    _write_cache(converter / "raw/vhacd/ra_007", "part_00", [2.0, 0.0, 0.0])

    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.utils.run_solver",
            "--converter-output-root", str(converter),
            "--source-root", str(source),
            "--run-output-dir", str(run),
            "--object-ids", "ra_007",
            "--write-visualization",
            "--viz-stride", "1",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert (run / "vendor/three.min.js").is_file()
    viewer = run / "ra_007/viz/joint0/step_viewer.html"
    assert viewer.is_file()
    manifest = json.loads((run / "ra_007/viz/joint0/step_manifest.json").read_text())
    assert manifest["render_hulls"]
    assert len(manifest["steps"]) > 2
    assert manifest["steps"][1]["frame"] is None
    assert "../../../vendor/three.min.js" in viewer.read_text()


def test_run_solver_applies_body0_gap_range_for_nested_prismatic_joint(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    _write_obj(obj_dir / "body.obj", [10.0, 0.0, 0.0])
    _write_obj(obj_dir / "part_00.obj", [0.0, 0.0, 0.0])
    _write_obj(obj_dir / "part_01.obj", [0.0, 0.0, 1.107])
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps({
        "object_id": "ra_007",
        "joints": {
            "joint_child": {
                "object_id": "ra_007",
                "joint_name": "joint_child",
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [0.0, 0.0, 1.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_01"],
                "static_parts": ["body", "part_00"],
                "body0_link_name": "part_00",
            }
        },
    }))
    _write_cache(converter / "raw/vhacd/ra_007", "body", [10.0, 0.0, 0.0])
    _write_cache(converter / "raw/vhacd/ra_007", "part_00", [0.0, 0.0, 0.0])
    _write_cache(converter / "raw/vhacd/ra_007", "part_01", [0.0, 0.0, 1.107])

    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.utils.run_solver",
            "--converter-output-root", str(converter),
            "--source-root", str(source),
            "--run-output-dir", str(run),
            "--object-ids", "ra_007",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    pred = json.loads((run / "ra_007/predictions.jsonl").read_text().strip())
    assert pred["predicted_lower"] == 0.0
    assert pred["predicted_upper"] == 0.1
    assert pred["status"] == "ok"


def test_run_solver_applies_pull_out_part_semantics_to_final_prismatic_limits(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    run = tmp_path / "run"
    obj_dir = converter / "raw/partseg/ra_007/objs"
    obj_dir.mkdir(parents=True)
    _write_obj(obj_dir / "body.obj", [0.0, 0.0, 0.0])
    _write_obj(obj_dir / "part_00.obj", [2.0, 0.0, 0.0])
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_007.json").write_text(json.dumps({
        "object_id": "ra_007",
        "joints": {
            "drawer_joint": {
                "object_id": "ra_007",
                "joint_name": "drawer_joint",
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [1.0, 0.0, 0.0],
                "origin_world": [0.0, 0.0, 0.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            }
        },
    }))
    (source / "source/model/007").mkdir(parents=True)
    (source / "source/model/007/gt_part.json").write_text(json.dumps({
        "pull-out drawer": "part_00",
    }))
    _write_cache(converter / "raw/vhacd/ra_007", "body", [0.0, 0.0, 0.0])
    _write_cache(converter / "raw/vhacd/ra_007", "part_00", [2.0, 0.0, 0.0])

    roots = V1DatasetRoots(converter_output_root=converter, source_root=source)
    write_dataset_fingerprint(roots, run, ids=["ra_007"])

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.utils.run_solver",
            "--converter-output-root", str(converter),
            "--source-root", str(source),
            "--run-output-dir", str(run),
            "--object-ids", "ra_007",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    pred = json.loads((run / "ra_007/predictions.jsonl").read_text().strip())
    trace = json.loads((run / "ra_007/trace.jsonl").read_text().strip())
    assert pred["predicted_lower"] == 0.0
    assert pred["predicted_upper"] > 0.0
    assert pred["motion_direction_prior"]["policy"] == "positive_only"
    assert pred["motion_direction_prior"]["raw_predicted_lower"] < 0.0
    assert trace["trace_lower"][1]["q"] < 0.0
