"""Geometric solver orchestration: vlm_oracle + VHACD cache -> predictions/trace."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .backend import make_backend
from .config import (
    CollisionConstraintConfig,
    SearchConfig,
    V1_COACD_RUN_PARAMS,
    V1_TEN_IDS,
    V1_VHACD_CACHE_METADATA,
    V1DatasetRoots,
)
from .constraints import (
    CollisionConstraint,
    RetainedOverlapConstraint,
    make_body0_gap_range_constraint,
)
from .joint_evaluator import JointEvaluator
from .motion_direction_prior import apply_motion_direction_prior, load_part_semantics
from .phase0_verify import assert_dataset_fingerprint_matches
from .solver import estimate_range
from .visualize import visualize_one_joint


def _part_to_obj_paths(roots: V1DatasetRoots, object_id: str) -> dict[str, Path]:
    obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        p.stem: p
        for p in sorted(obj_dir.glob("*.obj"))
        if p.stem == "body" or p.stem.startswith("part_")
    }


def _raw_vertices_by_part(part_to_obj: dict[str, Path]) -> dict[str, object]:
    import trimesh

    raw = {}
    for part, obj_path in part_to_obj.items():
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        raw[part] = mesh.vertices.copy()
    return raw


def _copy_viewer_vendor(run_output_dir: Path) -> None:
    vendor_src = (
        Path(__file__).resolve().parents[3]
        / "submodules/dataset_toolkits/vendor/three/0.160.0/classic/three.min.js"
    )
    if not vendor_src.is_file():
        return
    vendor_dst = run_output_dir / "vendor/three.min.js"
    vendor_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor_src, vendor_dst)


def _run_one_model(
    *,
    object_id: str,
    roots: V1DatasetRoots,
    run_output_dir: Path,
    spike_result: Path | None,
    enable_retained_overlap: bool,
    write_visualization: bool,
    viz_stride: int,
) -> None:
    out_dir = run_output_dir / object_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_visualization:
        _copy_viewer_vendor(run_output_dir)
    oracle = json.loads(
        (roots.converter_output_root / f"raw/vlm_oracle/{object_id}.json").read_text()
    )
    part_semantics = load_part_semantics(roots.source_root, object_id)
    part_to_obj = _part_to_obj_paths(roots, object_id)
    raw_vertices = _raw_vertices_by_part(part_to_obj)
    backend = make_backend(spike_result)
    backend.load_model(
        object_id=object_id,
        part_to_obj_path=part_to_obj,
        vhacd_cache_root=roots.converter_output_root / f"raw/vhacd/{object_id}",
        coacd_run_params=dict(V1_COACD_RUN_PARAMS),
        vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
    )

    pred_path = out_dir / "predictions.jsonl"
    trace_path = out_dir / "trace.jsonl"
    try:
        with pred_path.open("w") as pred_out, trace_path.open("w") as trace_out:
            for joint_name, joint in sorted(oracle["joints"].items()):
                joint = dict(joint)
                joint.setdefault("object_id", object_id)
                joint.setdefault("joint_name", joint_name)
                constraints = [CollisionConstraint(
                    list(joint["moving_parts"]),
                    list(joint["static_parts"]),
                    backend=backend,
                    config=CollisionConstraintConfig(allow_initial_penetration=True),
                )]
                body0_gap_constraint = make_body0_gap_range_constraint(
                    joint=joint,
                    raw_vertices_by_part=raw_vertices,
                )
                if body0_gap_constraint is not None:
                    constraints.append(body0_gap_constraint)
                if joint["type"] == "prismatic":
                    constraints.append(RetainedOverlapConstraint(
                        joint=joint,
                        raw_vertices_by_part=raw_vertices,
                        min_retained_ratio=0.2,
                    ))
                evaluator = JointEvaluator(
                    joint=joint,
                    constraints=constraints,
                    backend=backend,
                )
                prediction = estimate_range(
                    joint,
                    evaluator,
                    SearchConfig(
                        allow_initial_penetration=True,
                        viz_stride=viz_stride,
                    ),
                )
                trace = {
                    "object_id": object_id,
                    "joint_name": joint_name,
                    "trace_lower": prediction.pop("trace_lower"),
                    "trace_upper": prediction.pop("trace_upper"),
                }
                prediction = apply_motion_direction_prior(
                    prediction,
                    joint=joint,
                    part_semantics=part_semantics,
                )
                pred_out.write(json.dumps(prediction) + "\n")
                trace_out.write(json.dumps(trace) + "\n")
                if write_visualization:
                    backend.reset_to_identity()
                    visualize_one_joint(
                        backend=backend,
                        joint=joint,
                        prediction=prediction,
                        trace=trace,
                        out_dir=out_dir / "viz" / joint_name,
                        viz_stride=viz_stride,
                    )
    finally:
        backend.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 KinematicSolver geometric solver")
    parser.add_argument("--converter-output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--spike-result", type=Path)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    parser.add_argument(
        "--enable-retained-overlap",
        action="store_true",
        help="Deprecated: prismatic retained-overlap is enabled by default.",
    )
    parser.add_argument(
        "--write-visualization",
        action="store_true",
        help="Write solver-side PNG/GIF visualization artifacts.",
    )
    parser.add_argument(
        "--viz-stride",
        type=int,
        default=5,
        help="Render every Nth solver trace sample; use 1 for every step.",
    )
    args = parser.parse_args()

    roots = V1DatasetRoots(
        converter_output_root=args.converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
    )
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    assert_dataset_fingerprint_matches(roots, args.run_output_dir, ids)
    for object_id in ids:
        _run_one_model(
            object_id=object_id,
            roots=roots,
            run_output_dir=args.run_output_dir,
            spike_result=args.spike_result,
            enable_retained_overlap=args.enable_retained_overlap,
            write_visualization=args.write_visualization,
            viz_stride=args.viz_stride,
        )
        print(f"[OK] solver {object_id}")


if __name__ == "__main__":
    main()
