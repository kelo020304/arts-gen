from __future__ import annotations
from typing import Any
from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset
from trellis.datasets.arts.part_ss_latent_flow_single_view import PartSSLatentFlowSingleViewDataset


# 模块级 dataset 缓存：以 (view_mode, data_root, manifest_path) 为 key 复用同一实例。
# 这避免了每次 /api/infer/inputs 预览和每次 /api/infer/jobs POST 校验都重新实例化整个
# eval dataset（读全量 manifest，~13k samples，耗时数十秒）。首次加载慢，之后命中即返回。
_DATASET_CACHE: dict[tuple[str, str, str], Any] = {}


def clear_dataset_cache() -> None:
    """清空模块级 dataset 缓存（测试隔离用）。"""
    _DATASET_CACHE.clear()


def _dataset_for(view_mode: str, data_config: dict):
    # 用 .get() 容错：单测可能传最小 config（无 data_root/manifest_path）。
    key = (view_mode, str(data_config.get("data_root", "")),
           str(data_config.get("manifest_path", "")))
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    cfg = dict(data_config)
    if view_mode == "single":
        cfg["num_views"] = 1
        ds = PartSSLatentFlowSingleViewDataset(cfg)
    else:
        cfg["num_views"] = int(cfg.get("num_views", 4))
        ds = PartSSLatentFlowDataset(cfg)
    _DATASET_CACHE[key] = ds
    return ds


def load_object_inputs(data_config: dict, *, object_id: str, angle_idx: int, view_mode: str) -> dict[str, Any]:
    """按 object_id+angle_idx 复用 eval dataset 取一物输入。返回 dataset[idx] 的 dict + dataset_index。

    找不到则 KeyError（失败暴露）。dataset.samples 中物体 id 字段名为 ``obj_id``。
    """
    ds = _dataset_for(view_mode, data_config)
    idx = next(
        (i for i, s in enumerate(ds.samples)
         if str(s.get("obj_id", s.get("object_id"))) == str(object_id) and int(s["angle_idx"]) == int(angle_idx)),
        None,
    )
    if idx is None:
        raise KeyError(f"manifest 中无 object_id={object_id} angle_idx={angle_idx}")
    item = dict(ds[idx])
    item["dataset_index"] = idx
    return item
