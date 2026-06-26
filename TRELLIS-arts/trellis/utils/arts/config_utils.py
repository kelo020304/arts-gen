"""
配置工具模块：加载 YAML 配置，支持 _base_ 递归继承和 CLI 覆盖。

用法:
    from trellis.utils.arts.config_utils import load_config, config_to_dict
    cfg = load_config('TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml', overrides=['training.lr=1e-4'])
    d = config_to_dict(cfg)
"""

import os
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig


def _load_with_base(path: str) -> DictConfig:
    """递归加载 YAML 配置，支持 _base_ 字段继承。

    如果 YAML 文件中包含 _base_ 字段（字符串或列表），会先递归加载基础配置，
    然后用当前配置覆盖。_base_ 路径相对于当前 YAML 文件所在目录。

    Args:
        path: YAML 配置文件的绝对或相对路径。

    Returns:
        合并后的 OmegaConf DictConfig。
    """
    cfg = OmegaConf.load(path)
    if '_base_' not in cfg:
        return cfg

    base_paths = cfg.pop('_base_')
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    # 基础配置路径相对于当前文件所在目录
    base_dir = os.path.dirname(os.path.abspath(path))
    merged = OmegaConf.create()
    for bp in base_paths:
        abs_bp = os.path.join(base_dir, bp)
        base_cfg = _load_with_base(abs_bp)
        merged = OmegaConf.merge(merged, base_cfg)

    # 当前配置覆盖基础配置
    merged = OmegaConf.merge(merged, cfg)
    return merged


def load_config(path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """加载 YAML 配置，支持 _base_ 递归继承和 CLI 覆盖。

    Args:
        path: YAML 配置文件路径。
        overrides: CLI 覆盖列表，格式为 'key=value'，例如 ['training.lr=1e-4']。

    Returns:
        最终合并后的 OmegaConf DictConfig。
    """
    cfg = _load_with_base(path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def config_to_dict(cfg) -> dict:
    """将 OmegaConf DictConfig 转为纯 Python dict。

    Args:
        cfg: OmegaConf DictConfig 或其子节点。

    Returns:
        纯 Python dict，所有 OmegaConf 特殊类型均已解析。
    """
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
