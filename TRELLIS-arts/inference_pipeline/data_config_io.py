# TRELLIS-arts/inference_pipeline/data_config_io.py
from __future__ import annotations
from pathlib import Path
import yaml


def load_data_config(yaml_path, *, data_root_override: str | None = None) -> dict:
    """读 eval yaml 的 data: 段；可仅覆盖 data_root（本地 smoke_test）。"""
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(cfg, dict) or "data" not in cfg:
        raise KeyError(f"{yaml_path} 缺 'data:' 段")
    dc = dict(cfg["data"])
    if data_root_override:
        dc["data_root"] = str(data_root_override)
    if "data_root" not in dc:
        raise KeyError(f"{yaml_path} data 段缺 data_root")
    return dc
