"""Comparison orchestration: predictions.jsonl x gt_limits -> comparison.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import make_backend
from .compare import compare
from .comparison_visualize import write_per_direction_overlays
from .config import V1_COACD_RUN_PARAMS, V1_TEN_IDS, V1_VHACD_CACHE_METADATA, V1DatasetRoots
from .phase0_verify import assert_dataset_fingerprint_matches


def _part_to_obj_paths(roots: V1DatasetRoots, object_id: str) -> dict[str, Path]:
    obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        p.stem: p
        for p in sorted(obj_dir.glob("*.obj"))
        if p.stem == "body" or p.stem.startswith("part_")
    }


def _run_one_model(
    object_id: str,
    roots: V1DatasetRoots,
    run_output_dir: Path,
    *,
    write_overlays: bool = False,
) -> None:
    gt = json.loads(
        (roots.converter_output_root / f"raw/gt_limits/{object_id}.json").read_text()
    )
    oracle = None
    backend = None
    if write_overlays:
        oracle = json.loads(
            (roots.converter_output_root / f"raw/vlm_oracle/{object_id}.json").read_text()
        )
        backend = make_backend()
        backend.load_model(
            object_id=object_id,
            part_to_obj_path=_part_to_obj_paths(roots, object_id),
            vhacd_cache_root=roots.converter_output_root / f"raw/vhacd/{object_id}",
            coacd_run_params=dict(V1_COACD_RUN_PARAMS),
            vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
        )
    pred_path = run_output_dir / object_id / "predictions.jsonl"
    cmp_path = run_output_dir / object_id / "comparison.jsonl"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pred_path.open() as pin, cmp_path.open("w") as out:
            for line in pin:
                line = line.strip()
                if not line:
                    continue
                prediction = json.loads(line)
                joint_name = prediction["joint_name"]
                gt_joint = gt["limits"][joint_name]
                out.write(json.dumps(compare(prediction, gt_joint)) + "\n")
                if write_overlays and backend is not None and oracle is not None:
                    backend.reset_to_identity()
                    write_per_direction_overlays(
                        backend=backend,
                        joint=oracle["joints"][joint_name],
                        prediction=prediction,
                        gt=gt_joint,
                        out_dir=run_output_dir / object_id / "comparison_overlay" / joint_name,
                    )
    finally:
        if backend is not None:
            backend.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 KinematicSolver GT comparison")
    parser.add_argument("--converter-output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    parser.add_argument("--write-overlays", action="store_true")
    args = parser.parse_args()

    roots = V1DatasetRoots(
        converter_output_root=args.converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
    )
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    assert_dataset_fingerprint_matches(roots, args.run_output_dir, ids)
    for object_id in ids:
        _run_one_model(
            object_id,
            roots,
            args.run_output_dir,
            write_overlays=args.write_overlays,
        )
        print(f"[OK] compare {object_id}")


if __name__ == "__main__":
    main()
