# TRELLIS-arts/inference_pipeline/transform_io.py
# NOTE on normalization: voxelize (submodules/dataset_toolkits/pipeline/04_voxelize.py movtran)
# maps world -> grid as grid = world*scale + offset; offset is the additive translation,
# NOT a geometric centroid. We store it under key "offset" (NOT "center") to avoid kin_test
# misreading it. grid_world_transform.build_grid_to_world inverts (n-offset)/scale.
#
# Step 5 finding (grep submodules/dataset_toolkits/pipeline/04_voxelize.py): per-object
# scale/offset ARE persisted on disk, but voxelize does NOT author them. It reads them from
# the renderer's output file:
#     renders/{object_id}/angle_{angle_idx}/camera_transforms.json   ->  keys "scale","offset","aabb"
# (load_canonical_normalization @ 04_voxelize.py:416; path built @ :910-911). voxelize itself
# only consumes scale/offset (movtran @ :471 applies scale then translation) and writes voxel
# .npy + a QC report — it never re-emits a scale/offset file. So the authoritative per-object
# normalization source for inference is camera_transforms.json (loadable via
# grid_world_transform.load_scale_offset). When that file is absent (e.g. mode-A pure-data /
# external inputs without a render pass), callers MUST pass scale=None so transform_source
# becomes "missing" rather than fabricating a normalization.
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .grid_world_transform import build_grid_to_world

_STAGES = ("ss", "part", "slat", "assemble")
def _meta_path(run_dir): return Path(run_dir) / "meta.json"

def write_meta(run_dir, *, mode, view, object_id, run_id, ckpts, angle_idx=0, part_backend="part_flow"):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    prev = read_meta(run_dir) if _meta_path(run_dir).is_file() else {}
    prev_same_contract = (
        prev.get("mode") == mode and
        prev.get("view") == view and
        str(prev.get("object_id")) == str(object_id) and
        prev.get("run_id") == run_id and
        int(prev.get("angle_idx", angle_idx)) == int(angle_idx)
    )
    meta = {"mode": mode, "view": view, "object_id": str(object_id), "run_id": run_id,
            "angle_idx": int(angle_idx), "ckpts": dict(ckpts),
            "part_backend": str(part_backend or "part_flow"),
            "stage_status": prev.get("stage_status", {s: "pending" for s in _STAGES}) if prev_same_contract else {s: "pending" for s in _STAGES},
            "transform": prev.get("transform") if prev_same_contract else None}
    _meta_path(run_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta

def read_meta(run_dir):
    p = _meta_path(run_dir)
    if not p.is_file(): raise KeyError(f"meta.json 不存在：{p}")
    return json.loads(p.read_text())

def set_stage_status(run_dir, stage, status):
    if stage not in _STAGES: raise ValueError(f"未知 stage：{stage}")
    if status not in ("pending","running","done","failed"): raise ValueError(f"未知 status：{status}")
    meta = read_meta(run_dir); meta["stage_status"][stage] = status
    _meta_path(run_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def write_transform(run_dir, *, resolution, scale, offset, axis_up="Z"):
    run_dir = Path(run_dir)
    if scale is None or offset is None:
        transform = {"voxel_resolution": int(resolution), "axis_up": axis_up,
                     "grid_to_world": None, "normalization": None,
                     "applied_to_assets": False, "transform_source": "missing"}
    else:
        g2w = build_grid_to_world(resolution=int(resolution), scale=float(scale),
                                  offset=list(offset), obj_up_axis=axis_up)
        transform = {"voxel_resolution": int(resolution), "axis_up": axis_up,
                     "grid_to_world": np.asarray(g2w, float).tolist(),
                     "normalization": {"offset": list(offset), "scale": float(scale)},
                     "applied_to_assets": False, "transform_source": "voxelize_scale_offset"}
    (run_dir/"transform.json").write_text(json.dumps(transform, ensure_ascii=False, indent=2))
    if _meta_path(run_dir).is_file():
        meta = read_meta(run_dir); meta["transform"] = transform
        _meta_path(run_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return transform
