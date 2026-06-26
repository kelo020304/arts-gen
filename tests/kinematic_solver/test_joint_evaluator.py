import json

import numpy as np

from post_process.kinematic_solver.utils._fcl_backend import FclBackend
from post_process.kinematic_solver.utils.config import V1_COACD_RUN_PARAMS, V1_VHACD_CACHE_METADATA
from post_process.kinematic_solver.utils.constraints import CollisionConstraint
from post_process.kinematic_solver.utils.joint_evaluator import JointEvaluator


def _write_box_cache(root, object_id, part_name, center):
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
    p = root / object_id / f"{part_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "object_id": object_id,
        "part_name": part_name,
        "source_obj": f"{part_name}.obj",
        "source_sha256": "x",
        "vhacd_cache_metadata": dict(V1_VHACD_CACHE_METADATA),
        "coacd_run_params": dict(V1_COACD_RUN_PARAMS),
        "frame": "world_baked",
        "hulls": [{"hull_index": 0, "vertices": vertices, "faces": faces}],
        "n_hulls": 1,
    }))


def test_joint_evaluator_applies_prismatic_transform_before_collision_check(tmp_path):
    cache_root = tmp_path / "vhacd"
    _write_box_cache(cache_root, "ra_test", "body", [0.0, 0.0, 0.0])
    _write_box_cache(cache_root, "ra_test", "part_00", [2.0, 0.0, 0.0])

    backend = FclBackend()
    backend.load_model(
        object_id="ra_test",
        part_to_obj_path={"body": tmp_path / "body.obj", "part_00": tmp_path / "part_00.obj"},
        vhacd_cache_root=cache_root / "ra_test",
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
    )
    joint = {
        "type": "prismatic",
        "axis_world": [-1.0, 0.0, 0.0],
        "origin_world": [0.0, 0.0, 0.0],
        "moving_parts": ["part_00"],
        "static_parts": ["body"],
    }
    evaluator = JointEvaluator(
        joint=joint,
        constraints=[CollisionConstraint(["part_00"], ["body"], backend=backend)],
        backend=backend,
    )

    assert evaluator(0.0) is True
    assert evaluator(2.0) is False
