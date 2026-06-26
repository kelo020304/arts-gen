"""Write predicted limits into a copy of a source USD file."""

from __future__ import annotations

import shutil
from pathlib import Path

from pxr import Usd

from .usd_limit_writer import wrap_joint_for_limit_write, write_predicted_limits


def write_predicted_usd_for(
    *,
    prediction: dict,
    source_usd_path: Path | None,
    stage_metadata: dict,
    out_path: Path | None,
) -> Path | None:
    """Copy a source stage and write limits only for successful predictions."""
    if prediction["status"] != "ok":
        return None
    assert source_usd_path is not None and out_path is not None, (
        "ok prediction requires source_usd_path and out_path"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_usd_path, out_path)

    stage = Usd.Stage.Open(str(out_path))
    if stage is None:
        raise FileNotFoundError(f"failed to open copied USD stage: {out_path}")
    joint_path = stage_metadata["joint_prim_paths"][prediction["joint_name"]]
    joint_prim = stage.GetPrimAtPath(joint_path)
    joint_api = wrap_joint_for_limit_write(joint_prim, prediction["type"])
    write_predicted_limits(
        joint_api,
        prediction["type"],
        pred_lower=prediction["predicted_lower"],
        pred_upper=prediction["predicted_upper"],
        meters_per_unit=stage_metadata.get("meters_per_unit", 1.0),
    )
    stage.GetRootLayer().Save()
    return out_path
