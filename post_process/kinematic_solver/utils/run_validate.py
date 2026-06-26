"""Validation orchestration for KinematicSolver predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import V1_COACD_RUN_PARAMS, V1_TEN_IDS, V1_VHACD_CACHE_METADATA, V1DatasetRoots
from .phase0_verify import assert_dataset_fingerprint_matches
from .validate import ValidationContext, validate_joint
from .write_predicted_usd import write_predicted_usd_for


def _part_to_obj_paths(roots: V1DatasetRoots, object_id: str) -> dict[str, Path]:
    obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
    return {
        p.stem: p
        for p in sorted(obj_dir.glob("*.obj"))
        if p.stem == "body" or p.stem.startswith("part_")
    }


def _run_one_model(object_id: str, roots: V1DatasetRoots, run_output_dir: Path) -> None:
    out_dir = run_output_dir / object_id
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = json.loads(
        (roots.converter_output_root / f"raw/vlm_oracle/{object_id}.json").read_text()
    )
    stage_metadata = json.loads(
        (roots.converter_output_root / f"raw/stage_metadata/{object_id}.json").read_text()
    )
    source_usd = roots.aligned_usd_for(object_id)
    pred_path = out_dir / "predictions.jsonl"
    val_path = out_dir / "validation.jsonl"

    with pred_path.open() as pred_in, val_path.open("w") as val_out:
        for line in pred_in:
            if not line.strip():
                continue
            prediction = json.loads(line)
            predicted_usd_path = None
            if prediction["status"] == "ok":
                predicted_usd_path = out_dir / "predicted_usd" / f"{prediction['joint_name']}.usd"
                write_predicted_usd_for(
                    prediction=prediction,
                    source_usd_path=source_usd,
                    stage_metadata=stage_metadata,
                    out_path=predicted_usd_path,
                )
            ctx = ValidationContext(
                prediction=prediction,
                vlm_oracle_model=oracle,
                joint_name=prediction["joint_name"],
                object_id=object_id,
                usd_path=source_usd,
                predicted_usd_path=predicted_usd_path,
                part_to_obj_path=_part_to_obj_paths(roots, object_id),
                vhacd_cache_root=roots.converter_output_root / f"raw/vhacd/{object_id}",
                coacd_run_params=dict(V1_COACD_RUN_PARAMS),
                vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
                stage_metadata=stage_metadata,
            )
            val_out.write(json.dumps(validate_joint(ctx)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 KinematicSolver validation")
    parser.add_argument("--converter-output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    args = parser.parse_args()

    roots = V1DatasetRoots(
        converter_output_root=args.converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
    )
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    assert_dataset_fingerprint_matches(roots, args.run_output_dir, ids)
    for object_id in ids:
        _run_one_model(object_id, roots, args.run_output_dir)
        print(f"[OK] validate {object_id}")


if __name__ == "__main__":
    main()
