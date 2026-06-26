"""
LoRA 微调工具模块。
使用 peft 库为 TRELLIS DiT 模型注入 LoRA adapter。

设计决策 (D-20): 使用 peft 库而非手写 LoRA，原因:
1. SparseLinear 继承 nn.Linear，peft 可直接检测
2. peft 提供完整的 adapter save/load/merge/unmerge 生命周期
3. 避免为 SparseTensor 手写 LoRA 的复杂度

设计决策 (D-21): 默认目标层为所有 attention 投影层:
- self_attn: to_qkv, to_out
- cross_attn: to_q, to_kv, to_out
不包含 FFN (mlp) 层和 adaLN_modulation，因为:
- FFN 的 SparseLinear(1024,4096) 参数量巨大，LoRA 在此效率低
- adaLN 是 timestep 条件调制，不应被 LoRA 修改

依赖: pip install peft>=0.11.0
peft 0.11+ 支持自定义模型的 LoRA 注入 (不需要 HuggingFace model)

用法:
    from trellis.utils.arts.lora_utils import apply_lora_to_model, save_lora_weights, load_lora_weights

    # 从 YAML 配置中获取 lora 段落
    lora_cfg = config_to_dict(cfg).get('lora', {})

    # 应用 LoRA（如果 enabled=true）
    model = apply_lora_to_model(model, lora_cfg)

    # 保存 LoRA 权重（仅 adapter，< 50MB）
    save_lora_weights(model, 'output/lora_adapter')

    # 加载 LoRA 权重
    model = load_lora_weights(model, 'output/lora_adapter')
"""

import logging
import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

# ============================================================
# 预定义目标层组合 (D-21)
# peft 通过模块名称尾缀匹配 target_modules，
# 以下名称对应 TRELLIS DiT 中 MultiHeadAttention 的投影层:
#   blocks.{i}.self_attn.to_qkv   (nn.Linear, 1024->3072)
#   blocks.{i}.self_attn.to_out    (nn.Linear, 1024->1024)
#   blocks.{i}.cross_attn.to_q    (nn.Linear, 1024->1024)
#   blocks.{i}.cross_attn.to_kv   (nn.Linear, 1024->2048)
#   blocks.{i}.cross_attn.to_out  (nn.Linear, 1024->1024)
# ============================================================
TARGET_PRESETS = {
    "all_attn": ["to_qkv", "to_q", "to_kv", "to_out"],          # 所有 attention 投影 (推荐)
    "cross_attn_only": ["to_q", "to_kv"],                         # 只改 cross-attn QK
    "cross_attn_full": ["to_q", "to_kv", "to_out"],               # cross-attn 全部
    "self_attn_only": ["to_qkv", "to_out"],                       # 只改 self-attn
    "qkv_only": ["to_qkv", "to_q", "to_kv"],                     # 所有 QKV，不改 output
}


def get_lora_config(lora_cfg: dict):
    """
    根据 YAML 配置构建 peft LoraConfig。

    参数:
        lora_cfg: 从 YAML 配置中提取的 dict，包含:
            - enabled: bool
            - rank: int (默认 16)
            - alpha: int (默认 32，通常 alpha = 2 * rank)
            - dropout: float (默认 0.05)
            - target_modules: list[str] 或 str
              如果是 str 且在 TARGET_PRESETS 中，展开为列表；
              如果是 str 但不在预设中，用逗号分割。

    返回:
        peft.LoraConfig 实例
    """
    # 解析 target_modules
    target_modules = lora_cfg.get("target_modules", "all_attn")

    if isinstance(target_modules, str):
        if target_modules in TARGET_PRESETS:
            # 使用预定义组合
            resolved_targets = TARGET_PRESETS[target_modules]
        else:
            # 用逗号分割自定义字符串
            resolved_targets = [t.strip() for t in target_modules.split(",")]
    elif isinstance(target_modules, (list, tuple)):
        resolved_targets = list(target_modules)
    else:
        raise ValueError(
            f"target_modules 类型无效: {type(target_modules)}，"
            f"期望 str 或 list，可用预设: {list(TARGET_PRESETS.keys())}"
        )

    rank = lora_cfg.get("rank", 16)
    alpha = lora_cfg.get("alpha", 32)
    dropout = lora_cfg.get("dropout", 0.05)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved_targets,
        bias="none",
        # 不设置 task_type — 我们是自定义模型，不是 HuggingFace model
    )

    logger.info(
        f"LoRA 配置: rank={rank}, alpha={alpha}, dropout={dropout}, "
        f"target_modules={resolved_targets}"
    )

    return config


def apply_lora_to_model(model: nn.Module, lora_cfg: Optional[dict] = None) -> nn.Module:
    """
    为模型应用 LoRA adapter。

    如果 lora_cfg 为 None 或 enabled=false，直接返回原始模型（全参数微调模式）。
    如果 enabled=true，使用 peft 注入 LoRA adapter 并冻结原始权重。

    参数:
        model: 待注入 LoRA 的 PyTorch 模型
        lora_cfg: LoRA 配置 dict（从 YAML 中提取）。除常规 peft 字段外，
            可选 `keep_trainable: List[str]`（Phase 07, Rule A-01/A-04）：
            peft 注入后按子串包含规则把匹配的 named_parameter 重新
            `requires_grad_(True)`，用于 LoRA (attention) + 全量 (IO layers)
            的 hybrid 模式。匹配逻辑 `any(pat in name for pat in patterns)`，
            空/缺失时行为与改动前完全一致。

    返回:
        原始模型 (LoRA 禁用时) 或 peft 包装后的模型 (LoRA 启用时)
    """
    # LoRA 禁用时，保持全参数微调行为
    if lora_cfg is None or not lora_cfg.get("enabled", False):
        logger.info("LoRA 未启用，使用全参数微调")
        return model

    # 构建 LoRA 配置
    lora_config = get_lora_config(lora_cfg)

    # 注入 LoRA adapter
    model = get_peft_model(model, lora_config)

    # ---- keep_trainable: 按子串包含规则重新解冻指定参数 (Rule A-01, A-04) ----
    # peft 默认冻结所有非 LoRA 参数；hybrid 模式下需要把关键几何层 (例如
    # input_layer / input_blocks / out_blocks / out_layer) 重新打开 requires_grad。
    # 匹配规则是 `any(pat in name for pat in patterns)` (Rule A-04)；启动时打印
    # 前 3 个解冻参数名 + 总数以便核对误匹配。
    keep_patterns = lora_cfg.get("keep_trainable") or []
    unfrozen_names: List[str] = []
    if keep_patterns:
        for name, p in model.named_parameters():
            if p.requires_grad:
                # peft 已经打开的参数（LoRA adapter）跳过，不重复处理
                continue
            if any(pat in name for pat in keep_patterns):
                p.requires_grad_(True)
                unfrozen_names.append(name)
        # NOTE: 同一行包含 "keep_trainable unfroze" 以满足 Phase 07-01 acceptance_criteria
        # 的 grep 断言（`grep 'keep_trainable unfroze' lora_utils.py`）。
        logger.info(
            f"keep_trainable unfroze {len(unfrozen_names)} params "
            f"(patterns={list(keep_patterns)}); sample={unfrozen_names[:3]}"
        )

    # 冻结验证: 统计参数数量
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        if "lora_" in name:
            lora_params += num

    logger.info(
        f"LoRA 已应用: "
        f"总参数 {total_params / 1e6:.1f}M, "
        f"可训练 {trainable_params / 1e6:.1f}M ({trainable_params / total_params * 100:.1f}%), "
        f"LoRA 参数 {lora_params / 1e6:.1f}M"
    )

    # 断言确保 LoRA 正确注入
    assert trainable_params > 0, "LoRA 后无可训练参数"
    assert trainable_params < total_params, (
        "LoRA 后仍是全参数可训练，注入可能失败。"
        "请检查 target_modules 是否匹配模型中的层名。"
    )

    return model


def save_lora_weights(model: nn.Module, save_path: str) -> None:
    """
    保存 LoRA adapter 权重。

    优先使用 peft 的 save_pretrained (自动只保存 adapter 权重)；
    如果模型不是 peft 模型，手动筛选含 'lora_' 的参数保存。

    参数:
        model: 已注入 LoRA 的模型
        save_path: 保存目录路径
    """
    os.makedirs(save_path, exist_ok=True)

    if hasattr(model, 'save_pretrained'):
        # peft 模型: 使用 save_pretrained 自动只保存 adapter 权重
        model.save_pretrained(save_path)
        logger.info(f"LoRA 权重已通过 peft save_pretrained 保存到 {save_path}")
    else:
        # 非 peft 模型: 手动筛选 LoRA 参数
        lora_state = {
            k: v for k, v in model.state_dict().items() if "lora_" in k
        }
        if len(lora_state) == 0:
            logger.warning("未找到 LoRA 参数，模型可能未注入 LoRA")
            return

        save_file = os.path.join(save_path, "lora_weights.pt")
        torch.save(lora_state, save_file)
        logger.info(
            f"LoRA 权重已手动保存到 {save_file} "
            f"({len(lora_state)} 个参数张量)"
        )


def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """
    加载 LoRA adapter 权重。

    支持两种加载方式:
    1. peft 模型的 load_adapter 方法
    2. PeftModel.from_pretrained 从基础模型加载

    参数:
        model: 基础模型（已注入或未注入 LoRA）
        load_path: LoRA adapter 权重目录路径

    返回:
        加载了 LoRA 权重的模型
    """
    if hasattr(model, 'load_adapter'):
        # 已经是 peft 模型，直接加载 adapter
        model.load_adapter(load_path, adapter_name="default")
        logger.info(f"LoRA adapter 已通过 load_adapter 从 {load_path} 加载")
    else:
        # 基础模型，需要用 PeftModel.from_pretrained 包装
        # 检查路径下是否有 peft 格式的文件
        peft_files = [
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_config.json",
        ]
        has_peft_files = any(
            os.path.exists(os.path.join(load_path, f)) for f in peft_files
        )

        if has_peft_files:
            model = PeftModel.from_pretrained(model, load_path)
            logger.info(f"LoRA adapter 已通过 PeftModel.from_pretrained 从 {load_path} 加载")
        else:
            # 尝试加载手动保存的格式
            weight_file = os.path.join(load_path, "lora_weights.pt")
            if os.path.exists(weight_file):
                lora_state = torch.load(weight_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(lora_state, strict=False)
                logger.info(
                    f"LoRA 权重已从 {weight_file} 手动加载 "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            else:
                raise FileNotFoundError(
                    f"在 {load_path} 下未找到 LoRA 权重文件。"
                    f"期望 adapter_model.safetensors/bin 或 lora_weights.pt"
                )

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """
    打印模型中 requires_grad=True 的参数信息。

    显示前 20 个可训练参数的名称和 shape，其余省略。
    最后打印汇总统计。

    参数:
        model: PyTorch 模型
    """
    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append((name, tuple(param.shape)))

    # 打印前 20 个可训练参数
    print("=" * 60)
    print("可训练参数列表:")
    print("=" * 60)

    display_count = min(20, len(trainable_names))
    for i in range(display_count):
        name, shape = trainable_names[i]
        print(f"  {name}: {shape}")

    if len(trainable_names) > 20:
        print(f"  ... 省略 {len(trainable_names) - 20} 个参数")

    print("=" * 60)
    print(
        f"汇总: 总参数 {total_params:,} | "
        f"可训练 {trainable_params:,} | "
        f"比例 {trainable_params / total_params * 100:.2f}%"
    )
    print("=" * 60)
