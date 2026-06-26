# TRELLIS-arts/inference_pipeline/inputs_materialize.py
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def resolve_rgb_path(data_config: dict, *, object_id, angle_idx, view_index) -> Path:
    """重建 rgb 路径：<data_root>/<mask_subdir>/<id>/angle_<idx>/rgb/view_<v>.png。
    （manifest 的 image_paths 是别机器 stale 绝对路径，不用。）"""
    root = Path(data_config["data_root"]) / data_config.get("mask_subdir", "renders")
    p = root / str(object_id) / f"angle_{int(angle_idx)}" / "rgb" / f"view_{int(view_index)}.png"
    if not p.is_file():
        raise FileNotFoundError(f"rgb 不存在：{p}")
    return p

def materialize_from_paths(run_dir, *, rgb_src: Path, view_index: int) -> dict:
    run_dir = Path(run_dir); rgb_dir = run_dir/"input_rgb"; rgb_dir.mkdir(parents=True, exist_ok=True)
    dst = rgb_dir / f"view_{int(view_index)}.png"
    shutil.copyfile(rgb_src, dst)
    w, h = Image.open(dst).size                         # PIL size = (W,H)
    # 全幅 mask（用户确认：整图前景）。
    Image.fromarray(np.full((h, w), 255, np.uint8), mode="L").save(run_dir/"input_mask.png")
    return {"view_index": int(view_index), "rgb": str(dst), "mask": str(run_dir/"input_mask.png")}

def materialize(run_dir, data_config, *, object_id, angle_idx, view_indices) -> dict:
    if not view_indices: raise ValueError("view_indices 为空")
    v = int(view_indices[0])
    rgb = resolve_rgb_path(data_config, object_id=object_id, angle_idx=angle_idx, view_index=v)
    return materialize_from_paths(run_dir, rgb_src=rgb, view_index=v)
