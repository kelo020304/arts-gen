#!/usr/bin/env python3
"""
训练前数据完整性和一致性验证脚本

独立运行（不嵌入训练流程），在训练前自动检查数据完整性、latent 有效性、
part label 一致性、坐标系对齐和维度一致性，防止错误数据进入训练流程。

5 项检查:
  1. completeness  — 5 种产物文件是否存在
  2. latent        — SS latent 和 DINOv2 tokens 中是否有 NaN/Inf
  3. partlabel     — part label 编号与 manifest k8_labels 是否一致
  4. coordinate    — 体素坐标 Z-up 约束和 label 数量匹配
  5. dimension     — 各产物的 shape 是否符合预期

数据路径模式 (每个 (obj_id, angle_idx) 对):
  - SS Latent:    arts/reconstruction/ss_latents_expanded/{obj_id}/angle_{angle_idx}/latent.npz
  - DINOv2:       arts/reconstruction/dinov2_tokens/{obj_id}/angle_{angle_idx}/tokens.npz
  - Part Labels:  arts/reconstruction/part_labels/{obj_id}/angle_{angle_idx}/part_labels_64.npy
  - Renders:      arts/reconstruction/renders/{obj_id}/angle_{angle_idx}/rgb/view_0.png
  - Voxel:        arts/reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/allind.npy

使用:
    # 全量验证
    python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility

    # 抽样 50 个样本
    python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --num_samples 50

    # 只运行 completeness 和 latent 检查
    python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --checks completeness,latent

    # 详细输出
    python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --verbose
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# ============================================================
# 常量定义
# ============================================================

# 重建数据相对路径前缀
RECON_PREFIX = "arts/reconstruction"

# 5 种产物的路径模板（相对于 data_root）
ARTIFACT_PATHS = {
    "ss_latent": os.path.join(
        RECON_PREFIX, "ss_latents_expanded", "{obj_id}", "angle_{angle_idx}", "latent.npz"
    ),
    "dinov2_tokens": os.path.join(
        RECON_PREFIX, "dinov2_tokens", "{obj_id}", "angle_{angle_idx}", "tokens.npz"
    ),
    "part_labels": os.path.join(
        RECON_PREFIX, "part_labels", "{obj_id}", "angle_{angle_idx}", "part_labels_64.npy"
    ),
    "renders": os.path.join(
        RECON_PREFIX, "renders", "{obj_id}", "angle_{angle_idx}", "rgb", "view_0.png"
    ),
    "voxel": os.path.join(
        RECON_PREFIX, "voxel_expanded", "{obj_id}", "angle_{angle_idx}", "64", "allind.npy"
    ),
}

# SS latent feats 的维度
SS_LATENT_FEAT_DIM = 8
# SS latent coords 的维度
SS_LATENT_COORD_DIM = 3
# 体素分辨率
VOXEL_RESOLUTION = 64
# SS latent feats 最大合理绝对值阈值
LATENT_ABS_MAX_THRESHOLD = 100.0

# 可接受的 DINOv2 token shape（第二维 = 特征维度）
DINOV2_FEAT_DIM = 1024


# ============================================================
# 工具函数
# ============================================================

def _artifact_path(data_root: str, artifact_key: str, obj_id: str, angle_idx: int) -> str:
    """根据产物类型、物体 ID 和角度索引，构造完整的文件路径。"""
    template = ARTIFACT_PATHS[artifact_key]
    rel = template.format(obj_id=obj_id, angle_idx=angle_idx)
    return os.path.join(data_root, rel)


def _safe_load_npz(path: str) -> Optional[dict]:
    """安全加载 npz 文件，失败返回 None。"""
    try:
        data = np.load(path, allow_pickle=False)
        return dict(data)
    except Exception as e:
        return None


def _safe_load_npy(path: str) -> Optional[np.ndarray]:
    """安全加载 npy 文件，失败返回 None。"""
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


# ============================================================
# 检查 1: 完整性检查
# ============================================================

def check_completeness(data_root: str, obj_id: str, angle_idx: int) -> List[str]:
    """
    检查 5 种产物文件是否存在。

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引

    Returns:
        缺失的产物名称列表（空列表表示全部存在）
    """
    missing = []
    for artifact_key in ARTIFACT_PATHS:
        path = _artifact_path(data_root, artifact_key, obj_id, angle_idx)
        if not os.path.exists(path):
            missing.append(artifact_key)
    return missing


# ============================================================
# 检查 2: Latent 有效性检查
# ============================================================

def check_latent_validity(data_root: str, obj_id: str, angle_idx: int) -> List[str]:
    """
    检查 SS latent 和 DINOv2 tokens 中是否有 NaN/Inf/异常值。

    检查内容:
      - SS latent feats: 无 NaN、无 Inf、绝对值 < 100
      - DINOv2 tokens: 各 view 无 NaN、无 Inf

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引

    Returns:
        问题描述列表（空列表表示无问题）
    """
    issues = []

    # --- SS Latent ---
    ss_path = _artifact_path(data_root, "ss_latent", obj_id, angle_idx)
    if os.path.exists(ss_path):
        ss_data = _safe_load_npz(ss_path)
        if ss_data is None:
            issues.append("ss_latent: 文件加载失败")
        else:
            # 检查 feats 字段
            if "feats" in ss_data:
                feats = ss_data["feats"]
                if np.any(np.isnan(feats)):
                    issues.append("ss_latent: NaN detected in feats")
                if np.any(np.isinf(feats)):
                    issues.append("ss_latent: Inf detected in feats")
                abs_max = np.abs(feats).max() if feats.size > 0 else 0
                if abs_max >= LATENT_ABS_MAX_THRESHOLD:
                    issues.append(
                        f"ss_latent: feats abs max = {abs_max:.2f} >= {LATENT_ABS_MAX_THRESHOLD} (warning)"
                    )
            else:
                issues.append("ss_latent: 缺少 feats 字段")

            # 检查 coords 字段
            if "coords" in ss_data:
                coords = ss_data["coords"]
                if np.any(np.isnan(coords)):
                    issues.append("ss_latent: NaN detected in coords")
                if np.any(np.isinf(coords)):
                    issues.append("ss_latent: Inf detected in coords")
            else:
                issues.append("ss_latent: 缺少 coords 字段")

    # --- DINOv2 Tokens ---
    dino_path = _artifact_path(data_root, "dinov2_tokens", obj_id, angle_idx)
    if os.path.exists(dino_path):
        dino_data = _safe_load_npz(dino_path)
        if dino_data is None:
            issues.append("dinov2_tokens: 文件加载失败")
        else:
            # DINOv2 tokens 存储格式: 单 key 'tokens', shape [V, T, D]
            # (也兼容旧版 view_* 多 key 格式)
            if 'tokens' in dino_data:
                tokens = dino_data['tokens']
                if np.any(np.isnan(tokens)):
                    issues.append("dinov2_tokens: NaN detected in tokens")
                if np.any(np.isinf(tokens)):
                    issues.append("dinov2_tokens: Inf detected in tokens")
                if tokens.ndim != 3:
                    issues.append(f"dinov2_tokens: tokens ndim={tokens.ndim}，期望 3 (V, T, D)")
                elif tokens.shape[-1] != DINOV2_FEAT_DIM:
                    issues.append(f"dinov2_tokens: 特征维度 {tokens.shape[-1]} != {DINOV2_FEAT_DIM}")
            else:
                # 兼容旧版 view_* 多 key 格式
                view_keys = [k for k in dino_data.keys() if k.startswith("view_")]
                if len(view_keys) == 0:
                    issues.append("dinov2_tokens: 未找到 'tokens' 或 'view_*' 字段")
                for vk in sorted(view_keys):
                    tokens = dino_data[vk]
                    if np.any(np.isnan(tokens)):
                        issues.append(f"dinov2_tokens: NaN detected in {vk}")
                    if np.any(np.isinf(tokens)):
                        issues.append(f"dinov2_tokens: Inf detected in {vk}")

    return issues


# ============================================================
# 检查 3: Part Label 一致性检查
# ============================================================

def check_part_label_consistency(
    data_root: str,
    obj_id: str,
    angle_idx: int,
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    检查 part label 基本合理性。

    检查内容:
      - 加载 part_labels_64.npy，获取实际 label 值集合
      - 如果 manifest_entry 含 k8_labels（旧版 manifest），校验 label 范围
      - 否则做 fallback 检查: 有效 label 是否存在、最大值是否异常、overlap 比例

    注意: 当前标准 assembler manifest 不提供 k8_labels 字段，
    因此大部分情况走 fallback 分支。这不是"part 编号与 2D mask 的一致性校验"，
    而是 part label 自身的基本合理性检查。

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引
        manifest_entry: manifest.json 中该物体的条目（可选）

    Returns:
        不一致描述列表
    """
    issues = []

    label_path = _artifact_path(data_root, "part_labels", obj_id, angle_idx)
    if not os.path.exists(label_path):
        # 完整性检查会捕获此问题，此处不重复报告
        return issues

    labels = _safe_load_npy(label_path)
    if labels is None:
        issues.append("part_labels: 文件加载失败")
        return issues

    # 获取实际 label 值集合（排除 0=空 和 -1=overlap）
    unique_labels = set(np.unique(labels).tolist())
    actual_part_labels = {v for v in unique_labels if v > 0}

    if manifest_entry is not None and "k8_labels" in manifest_entry:
        k8 = manifest_entry["k8_labels"]
        # k8_labels 的 key 是字符串形式的编号 "0", "1", ...
        # part_labels_64.npy 使用 part_id + 1 作为 label（1-indexed）
        expected_count = len(k8)
        expected_range = set(range(1, expected_count + 1))

        # 检查实际 label 是否超出期望范围
        out_of_range = actual_part_labels - expected_range
        if out_of_range:
            issues.append(
                f"part_labels: label 值 {sorted(out_of_range)} 超出期望范围 [1, {expected_count}]"
            )

        # 如果实际 label 为空但有体素数据
        if len(actual_part_labels) == 0 and labels.size > 0 and np.any(labels != 0):
            issues.append("part_labels: 未发现有效的 part label（所有非零值为 -1/overlap）")
    else:
        # 没有 manifest entry，只做基本范围检查
        if len(actual_part_labels) == 0 and labels.size > 0 and np.any(labels != 0):
            issues.append("part_labels: 未发现有效的 part label")

        # 检查是否有异常大的 label 值
        max_label = max(actual_part_labels) if actual_part_labels else 0
        if max_label > 20:
            issues.append(
                f"part_labels: 最大 label 值 {max_label} 异常偏大（期望 < 20）"
            )

    # 检查 overlap (-1) 占比
    if -1 in unique_labels:
        overlap_count = int(np.sum(labels == -1))
        total_occupied = int(np.sum(labels != 0))
        if total_occupied > 0:
            overlap_ratio = overlap_count / total_occupied
            if overlap_ratio > 0.5:
                issues.append(
                    f"part_labels: overlap 比例 {overlap_ratio:.1%} 过高（{overlap_count}/{total_occupied}）"
                )

    return issues


# ============================================================
# 检查 4: 坐标系对齐检查
# ============================================================

def check_coordinate_alignment(data_root: str, obj_id: str, angle_idx: int) -> List[str]:
    """
    简化版坐标系对齐检查（Z-up 约束 + label 数量匹配）。

    检查内容:
      - 体素坐标 Z 范围在 [0, 63]
      - 物体主体在 Z 方向有合理分布
      - part_labels 非零体素数量与 allind 行数一致

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引

    Returns:
        问题描述列表
    """
    issues = []

    # 加载体素坐标
    voxel_path = _artifact_path(data_root, "voxel", obj_id, angle_idx)
    if not os.path.exists(voxel_path):
        return issues

    allind = _safe_load_npy(voxel_path)
    if allind is None:
        issues.append("voxel: allind.npy 加载失败")
        return issues

    if allind.ndim != 2 or allind.shape[1] != 3:
        issues.append(f"voxel: allind shape {allind.shape} 不符合 (N, 3)")
        return issues

    n_voxels = allind.shape[0]

    # Z-up 约束检查: 体素坐标的 Z 轴范围应在 [0, 63]
    z_vals = allind[:, 2]  # 第三列为 Z
    z_min, z_max = int(z_vals.min()), int(z_vals.max())

    if z_min < 0 or z_max > (VOXEL_RESOLUTION - 1):
        issues.append(
            f"coordinate: Z 范围 [{z_min}, {z_max}] 超出 [0, {VOXEL_RESOLUTION - 1}]"
        )

    # Z 方向分布检查：物体至少占 Z 方向的一定范围
    z_span = z_max - z_min + 1
    if z_span < 2:
        issues.append(f"coordinate: Z 方向跨度仅 {z_span}，物体可能退化为平面")

    # X、Y 范围同样检查
    for axis_idx, axis_name in [(0, "X"), (1, "Y")]:
        vals = allind[:, axis_idx]
        a_min, a_max = int(vals.min()), int(vals.max())
        if a_min < 0 or a_max > (VOXEL_RESOLUTION - 1):
            issues.append(
                f"coordinate: {axis_name} 范围 [{a_min}, {a_max}] 超出 [0, {VOXEL_RESOLUTION - 1}]"
            )

    # 与 part_labels 数量一致性检查
    label_path = _artifact_path(data_root, "part_labels", obj_id, angle_idx)
    if os.path.exists(label_path):
        labels = _safe_load_npy(label_path)
        if labels is not None:
            if labels.ndim == 1:
                # 稀疏格式: shape (N,)
                n_labels = labels.shape[0]
                if n_labels != n_voxels:
                    issues.append(
                        f"coordinate: part_labels 数量 ({n_labels}) != allind 行数 ({n_voxels})"
                    )
            elif labels.ndim == 3:
                # 密集格式: shape (64, 64, 64)
                # 非零体素数量应与 allind 行数一致
                n_occupied = int(np.sum(labels != 0))
                # 允许一定误差（因为 overlap=-1 也算非零）
                if abs(n_occupied - n_voxels) > max(5, n_voxels * 0.1):
                    issues.append(
                        f"coordinate: part_labels 非零数 ({n_occupied}) 与 allind 行数 ({n_voxels}) 差异过大"
                    )

    return issues


# ============================================================
# 检查 5: 维度一致性检查
# ============================================================

def check_dimension_consistency(data_root: str, obj_id: str, angle_idx: int) -> List[str]:
    """
    检查各产物的 shape 和维度是否符合预期。

    检查内容:
      - SS latent: feats shape (N, 8), coords shape (N, 3)
      - DINOv2 tokens: 各 view shape (*, 1024)
      - Part labels: shape (N,) 且 N == allind 行数（稀疏格式）或 shape (64, 64, 64)（密集格式）
      - Voxel allind: shape (N, 3) 且值在 [0, 63]

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引

    Returns:
        不一致描述列表
    """
    issues = []

    # 获取 allind 行数作为基准
    voxel_path = _artifact_path(data_root, "voxel", obj_id, angle_idx)
    n_voxels = None
    if os.path.exists(voxel_path):
        allind = _safe_load_npy(voxel_path)
        if allind is not None:
            if allind.ndim != 2 or allind.shape[1] != 3:
                issues.append(f"dimension: allind shape {allind.shape} 不是 (N, 3)")
            else:
                n_voxels = allind.shape[0]
                # 值域检查
                if allind.min() < 0 or allind.max() > (VOXEL_RESOLUTION - 1):
                    issues.append(
                        f"dimension: allind 值域 [{allind.min()}, {allind.max()}] "
                        f"超出 [0, {VOXEL_RESOLUTION - 1}]"
                    )
        else:
            issues.append("dimension: allind.npy 加载失败")

    # --- SS Latent ---
    ss_path = _artifact_path(data_root, "ss_latent", obj_id, angle_idx)
    if os.path.exists(ss_path):
        ss_data = _safe_load_npz(ss_path)
        if ss_data is not None:
            if "feats" in ss_data:
                feats = ss_data["feats"]
                if feats.ndim != 2 or feats.shape[1] != SS_LATENT_FEAT_DIM:
                    issues.append(
                        f"dimension: ss_latent feats shape {feats.shape} 不是 (N, {SS_LATENT_FEAT_DIM})"
                    )
            if "coords" in ss_data:
                coords = ss_data["coords"]
                if coords.ndim != 2 or coords.shape[1] != SS_LATENT_COORD_DIM:
                    issues.append(
                        f"dimension: ss_latent coords shape {coords.shape} 不是 (N, {SS_LATENT_COORD_DIM})"
                    )
                # feats 和 coords 的 N 应一致
                if "feats" in ss_data:
                    if ss_data["feats"].shape[0] != coords.shape[0]:
                        issues.append(
                            f"dimension: ss_latent feats N={ss_data['feats'].shape[0]} "
                            f"!= coords N={coords.shape[0]}"
                        )

    # --- DINOv2 Tokens ---
    dino_path = _artifact_path(data_root, "dinov2_tokens", obj_id, angle_idx)
    if os.path.exists(dino_path):
        dino_data = _safe_load_npz(dino_path)
        if dino_data is not None:
            if 'tokens' in dino_data:
                tokens = dino_data['tokens']
                if tokens.ndim != 3:
                    issues.append(f"dimension: dinov2 tokens ndim={tokens.ndim}，期望 3 (V, T, D)")
                elif tokens.shape[-1] != DINOV2_FEAT_DIM:
                    issues.append(f"dimension: dinov2 特征维度 {tokens.shape[-1]} != {DINOV2_FEAT_DIM}")
            else:
                # 兼容旧版 view_* 格式
                view_keys = [k for k in dino_data.keys() if k.startswith("view_")]
                for vk in sorted(view_keys):
                    tokens = dino_data[vk]
                    if tokens.ndim != 2:
                        issues.append(f"dimension: dinov2 {vk} ndim={tokens.ndim} 不是 2")
                    elif tokens.shape[1] != DINOV2_FEAT_DIM:
                        issues.append(f"dimension: dinov2 {vk} 特征维度 != {DINOV2_FEAT_DIM}")

    # --- Part Labels ---
    label_path = _artifact_path(data_root, "part_labels", obj_id, angle_idx)
    if os.path.exists(label_path):
        labels = _safe_load_npy(label_path)
        if labels is not None:
            if labels.ndim == 1:
                # 稀疏格式
                if n_voxels is not None and labels.shape[0] != n_voxels:
                    issues.append(
                        f"dimension: part_labels shape ({labels.shape[0]},) "
                        f"!= allind rows ({n_voxels})"
                    )
            elif labels.ndim == 3:
                # 密集格式
                expected_shape = (VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
                if labels.shape != expected_shape:
                    issues.append(
                        f"dimension: part_labels shape {labels.shape} != {expected_shape}"
                    )
            else:
                issues.append(
                    f"dimension: part_labels ndim={labels.ndim}，期望 1（稀疏）或 3（密集）"
                )

    return issues


# ============================================================
# 检查调度器
# ============================================================

# 检查函数名 -> 实际函数的映射
CHECK_REGISTRY = {
    "completeness": check_completeness,
    "latent": check_latent_validity,
    "partlabel": check_part_label_consistency,
    "coordinate": check_coordinate_alignment,
    "dimension": check_dimension_consistency,
}


def run_checks(
    data_root: str,
    obj_id: str,
    angle_idx: int,
    checks: List[str],
    manifest_entry: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    对单个样本运行指定的检查。

    Args:
        data_root: 数据根目录
        obj_id: 物体 ID
        angle_idx: 角度索引
        checks: 要运行的检查名称列表
        manifest_entry: manifest.json 中该物体的条目
        verbose: 是否输出详细信息

    Returns:
        问题列表，每项包含 obj_id, angle_idx, check, issues
    """
    all_issues = []

    for check_name in checks:
        fn = CHECK_REGISTRY.get(check_name)
        if fn is None:
            continue

        # part label 一致性检查需要额外的 manifest_entry 参数
        if check_name == "partlabel":
            issues = fn(data_root, obj_id, angle_idx, manifest_entry)
        else:
            issues = fn(data_root, obj_id, angle_idx)

        if issues:
            all_issues.append({
                "obj_id": obj_id,
                "angle_idx": angle_idx,
                "check": check_name,
                "issues": issues,
            })

        if verbose and issues:
            for issue_desc in issues:
                print(f"  [{check_name}] {obj_id}/angle_{angle_idx}: {issue_desc}")

    return all_issues


# ============================================================
# Manifest 加载
# ============================================================

def load_manifest(data_root: str, manifest_rel: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    加载 manifest.json 并构建 obj_id -> entry 的查找表。

    Args:
        data_root: 数据根目录
        manifest_rel: manifest.json 相对于 data_root 的路径

    Returns:
        (manifest_entries, obj_id_lookup)
        - manifest_entries: manifest 中的所有条目
        - obj_id_lookup: {obj_id: manifest_entry} 查找表
    """
    manifest_path = os.path.join(data_root, manifest_rel)
    if not os.path.exists(manifest_path):
        print(f"[WARN] manifest 不存在: {manifest_path}")
        return [], {}

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    # 支持两种 manifest 格式:
    #   1. assembler 格式 (dict): {"samples": [{"object_id": ..., "angle_idx": ..., "complete": ...}]}
    #   2. 旧格式 (list): [{"id": ..., "angles": [...]}]
    entries = []
    lookup = {}
    if isinstance(manifest_data, dict) and 'samples' in manifest_data:
        # assembler 格式：只取 complete=True 的样本（与训练 dataset 对齐）
        for s in manifest_data['samples']:
            if not s.get('complete', False):
                continue  # 跳过未完成样本，和 MvImageConditionedSLatDataset 一致
            obj_id = str(s.get('object_id', ''))
            if obj_id:
                entry = {'id': obj_id, 'angle_idx': s.get('angle_idx', 0),
                         'complete': True}
                entries.append(entry)
                if obj_id not in lookup:
                    lookup[obj_id] = {'id': obj_id, 'angles': []}
                lookup[obj_id]['angles'].append(entry['angle_idx'])
    elif isinstance(manifest_data, list):
        # 旧格式
        entries = manifest_data
        for entry in entries:
            obj_id = str(entry.get("id", ""))
            if obj_id:
                lookup[obj_id] = entry
    else:
        print("[WARN] 无法识别的 manifest 格式")

    return entries, lookup


def enumerate_samples(
    data_root: str,
    manifest_entries: List[Dict],
    num_samples: int = 0,
) -> List[Tuple[str, int]]:
    """
    枚举所有需要验证的 (obj_id, angle_idx) 样本对。

    如果 manifest 有数据，使用 manifest 中的条目；
    否则从 voxel_expanded 目录枚举。

    Args:
        data_root: 数据根目录
        manifest_entries: manifest 条目
        num_samples: 采样数量（0 = 全量）

    Returns:
        [(obj_id, angle_idx), ...] 列表
    """
    samples = []

    if manifest_entries:
        # 从 manifest 枚举
        # 支持两种 entry 格式:
        #   assembler 扁平格式: {"id": "100015", "angle_idx": 0, "complete": true}
        #   旧格式: {"id": "100015", "angles": [0, 1, 2, ...]}
        for entry in manifest_entries:
            obj_id = str(entry.get("id", ""))
            if not obj_id:
                continue
            if "angle_idx" in entry:
                # assembler 扁平格式：每条 entry 是一个 (obj_id, angle_idx)
                samples.append((obj_id, int(entry["angle_idx"])))
            elif "angles" in entry:
                # 旧格式：每条 entry 有 angles 列表
                for angle_idx in entry["angles"]:
                    samples.append((obj_id, int(angle_idx)))
    else:
        # 从 voxel_expanded 目录枚举
        voxel_dir = os.path.join(data_root, RECON_PREFIX, "voxel_expanded")
        if os.path.isdir(voxel_dir):
            for obj_id in sorted(os.listdir(voxel_dir)):
                obj_dir = os.path.join(voxel_dir, obj_id)
                if not os.path.isdir(obj_dir):
                    continue
                for angle_name in sorted(os.listdir(obj_dir)):
                    if angle_name.startswith("angle_"):
                        try:
                            angle_idx = int(angle_name.split("_")[1])
                            samples.append((obj_id, angle_idx))
                        except (ValueError, IndexError):
                            pass
        else:
            print(f"[WARN] voxel_expanded 目录不存在: {voxel_dir}")

    # 抽样
    if num_samples > 0 and num_samples < len(samples):
        random.seed(42)  # 可复现的随机采样
        samples = random.sample(samples, num_samples)

    return samples


# ============================================================
# 报告生成
# ============================================================

def print_report(
    all_issues: List[Dict[str, Any]],
    total_samples: int,
    checks_run: List[str],
) -> int:
    """
    打印验证报告。

    Args:
        all_issues: 所有检查问题
        total_samples: 总检查样本数
        checks_run: 执行的检查名称列表

    Returns:
        失败样本数
    """
    # 按 (obj_id, angle_idx) 分组
    failed_samples = set()
    for issue_entry in all_issues:
        key = (issue_entry["obj_id"], issue_entry["angle_idx"])
        failed_samples.add(key)

    n_passed = total_samples - len(failed_samples)
    n_failed = len(failed_samples)

    print()
    print("=" * 60)
    print("=== 数据验证报告 ===")
    print("=" * 60)
    print(f"检查项目: {', '.join(checks_run)}")
    print(f"检查样本数: {total_samples}")
    print(f"通过: {n_passed}")
    print(f"失败: {n_failed}")

    if n_failed > 0:
        print()
        print("失败详情:")

        # 按检查类型统计
        check_counts: Dict[str, int] = {}
        for issue_entry in all_issues:
            check_name = issue_entry["check"]
            check_counts[check_name] = check_counts.get(check_name, 0) + 1

        print()
        print("  按检查类型统计:")
        for check_name in checks_run:
            count = check_counts.get(check_name, 0)
            if count > 0:
                print(f"    {check_name}: {count} 个样本失败")

        print()
        print("  具体失败样本:")
        for issue_entry in all_issues:
            obj_id = issue_entry["obj_id"]
            angle_idx = issue_entry["angle_idx"]
            check_name = issue_entry["check"]
            for desc in issue_entry["issues"]:
                print(f"  [FAIL] {obj_id}/angle_{angle_idx} {check_name}: {desc}")

    print()
    print("=" * 60)

    return n_failed


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="训练前数据完整性和一致性验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全量验证
  python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility

  # 抽样 50 个样本
  python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --num_samples 50

  # 只运行 completeness 和 latent 检查
  python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --checks completeness,latent

  # 详细输出
  python TRELLIS-arts/tests/arts/validate_data.py --data_root data/PhysX-Mobility --verbose
        """,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/PhysX-Mobility",
        help="数据根目录 (默认: data/PhysX-Mobility)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="arts/manifest.json",
        help="manifest.json 相对于 data_root 的路径 (默认: arts/manifest.json)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="验证样本数量，0 表示全量验证 (默认: 0)",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default="all",
        help="要运行的检查项，逗号分隔 (默认: all)。"
        "可选: completeness,latent,partlabel,coordinate,dimension",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息",
    )
    args = parser.parse_args()

    # 确定要运行的检查
    all_check_names = list(CHECK_REGISTRY.keys())
    if args.checks == "all":
        checks_to_run = all_check_names
    else:
        checks_to_run = [c.strip() for c in args.checks.split(",")]
        invalid = [c for c in checks_to_run if c not in CHECK_REGISTRY]
        if invalid:
            print(f"[ERROR] 未知检查项: {invalid}")
            print(f"[INFO] 可选: {all_check_names}")
            sys.exit(1)

    # 加载 manifest
    manifest_entries, manifest_lookup = load_manifest(args.data_root, args.manifest)
    if manifest_entries:
        print(f"[INFO] 从 manifest 加载了 {len(manifest_entries)} 个物体")
    else:
        print("[INFO] manifest 不可用，将从 voxel_expanded 目录枚举样本")

    # 枚举样本
    samples = enumerate_samples(args.data_root, manifest_entries, args.num_samples)
    if not samples:
        print("[ERROR] 未找到任何样本")
        sys.exit(1)

    mode_desc = f"抽样 {args.num_samples}" if args.num_samples > 0 else "全量"
    print(f"[INFO] {mode_desc}验证: {len(samples)} 个样本")
    print(f"[INFO] 检查项: {', '.join(checks_to_run)}")

    # 运行检查
    all_issues = []
    for i, (obj_id, angle_idx) in enumerate(samples):
        if args.verbose and (i + 1) % 100 == 0:
            print(f"[INFO] 进度: {i + 1}/{len(samples)}")

        # 获取对应的 manifest entry
        entry = manifest_lookup.get(obj_id, None)

        sample_issues = run_checks(
            data_root=args.data_root,
            obj_id=obj_id,
            angle_idx=angle_idx,
            checks=checks_to_run,
            manifest_entry=entry,
            verbose=args.verbose,
        )
        all_issues.extend(sample_issues)

    # 打印报告
    n_failed = print_report(all_issues, len(samples), checks_to_run)

    # 退出码: 有失败则 exit(1)
    if n_failed > 0:
        sys.exit(1)
    else:
        print("[DONE] 所有样本验证通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
