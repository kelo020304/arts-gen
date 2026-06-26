#!/usr/bin/env python
"""
LoRA 集成测试脚本。
验证 peft LoRA 在 TRELLIS SparseStructureFlowModel 上的注入效果。

运行方式:
    pytest TRELLIS-arts/tests/arts/test_lora.py -v

依赖: peft>=0.11.0, torch, omegaconf
注意: 需要设置 ATTN_BACKEND=sdpa 避免 flash_attn 依赖（顶级 conftest 已处理）
"""
import os
import sys
import logging
import torch

# 顶级 conftest.py 已经注入 sys.path（TRELLIS-arts/）+ ATTN_BACKEND/TORCH_HOME 默认值。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置日志以便看到 lora_utils 的输出
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

# ---- 顶层导入（trainer/utils 已迁入 trellis 包内，无需 stub）----
import tempfile
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.utils.arts.lora_utils import (
    apply_lora_to_model, save_lora_weights, load_lora_weights,
    get_lora_config, TARGET_PRESETS,
)
from trellis.utils.arts.config_utils import load_config, config_to_dict


def _create_small_model():
    """创建小型 SparseStructureFlowModel 用于测试（避免 OOM）。"""
    return SparseStructureFlowModel(
        resolution=8,
        in_channels=1,
        model_channels=64,
        cond_channels=64,
        out_channels=1,
        num_blocks=2,
        num_heads=4,
        patch_size=2,
        pe_mode="ape",
        use_fp16=False,
    )


def test_lora_disabled():
    """Test 1: LoRA 禁用时模型不变。"""
    model = _create_small_model()
    original_params = sum(p.numel() for p in model.parameters())

    # 测试 enabled=False
    lora_cfg = {"enabled": False}
    model_out = apply_lora_to_model(model, lora_cfg)

    after_params = sum(p.numel() for p in model_out.parameters())
    assert original_params == after_params, \
        f"LoRA 禁用但参数数量变化: {original_params} -> {after_params}"

    trainable = sum(p.numel() for p in model_out.parameters() if p.requires_grad)
    assert trainable == original_params, "LoRA 禁用但部分参数被冻结"

    # 测试 lora_cfg=None
    model2 = _create_small_model()
    model2_out = apply_lora_to_model(model2, None)
    assert sum(p.numel() for p in model2_out.parameters()) == original_params, \
        "lora_cfg=None 时参数数量不应变化"

    print("[PASS] Test 1: LoRA 禁用时模型不变")


def test_lora_enabled():
    """Test 2: LoRA 启用时参数正确冻结。"""
    model = _create_small_model()
    original_params = sum(p.numel() for p in model.parameters())

    lora_cfg = {
        "enabled": True,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": "all_attn",
    }
    model = apply_lora_to_model(model, lora_cfg)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    assert total > original_params, \
        f"LoRA 后总参数应增加（含 adapter）: {original_params} -> {total}"
    assert trainable < total, \
        f"LoRA 后应有冻结参数: trainable={trainable}, total={total}"
    assert frozen >= original_params * 0.9, \
        f"大部分原始参数应被冻结: frozen={frozen}, original={original_params}"

    # 检查 LoRA 参数存在
    lora_param_names = [
        n for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    assert len(lora_param_names) > 0, "未找到 LoRA 参数"

    print(
        f"[PASS] Test 2: LoRA 启用 — "
        f"总参数 {total}, 可训练 {trainable} ({trainable / total * 100:.1f}%), "
        f"LoRA 层数 {len(lora_param_names)}"
    )


def test_target_presets():
    """Test 3: 不同 target_modules 预设。"""
    for preset_name, expected_targets in TARGET_PRESETS.items():
        cfg = {"enabled": True, "rank": 4, "alpha": 8, "target_modules": preset_name}
        lora_config = get_lora_config(cfg)
        actual_targets = list(lora_config.target_modules)
        assert set(actual_targets) == set(expected_targets), \
            f"预设 {preset_name}: 期望 {expected_targets}, 实际 {actual_targets}"

    # 测试自定义列表
    cfg = {"enabled": True, "rank": 4, "target_modules": ["to_q", "to_kv"]}
    lora_config = get_lora_config(cfg)
    assert set(lora_config.target_modules) == {"to_q", "to_kv"}

    # 测试逗号分割字符串
    cfg = {"enabled": True, "rank": 4, "target_modules": "to_q,to_kv"}
    lora_config = get_lora_config(cfg)
    assert set(lora_config.target_modules) == {"to_q", "to_kv"}

    print("[PASS] Test 3: 所有 target_modules 预设正确")


def test_save_load():
    """Test 4: LoRA 权重保存/加载。"""
    model = _create_small_model()
    lora_cfg = {
        "enabled": True,
        "rank": 4,
        "alpha": 8,
        "target_modules": "all_attn",
    }
    model = apply_lora_to_model(model, lora_cfg)

    # 修改 LoRA 权重（模拟训练后）
    for n, p in model.named_parameters():
        if "lora_" in n and p.requires_grad:
            p.data.fill_(0.42)
            break

    # 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "lora_adapter")
        save_lora_weights(model, save_path)

        # 验证保存目录存在且文件不为空
        assert os.path.exists(save_path), f"保存路径不存在: {save_path}"
        saved_files = os.listdir(save_path)
        assert len(saved_files) > 0, "保存目录为空"

        # 检查文件大小 (adapter 权重应该很小)
        total_size = sum(
            os.path.getsize(os.path.join(save_path, f))
            for f in saved_files
            if os.path.isfile(os.path.join(save_path, f))
        )
        assert total_size < 50 * 1024 * 1024, \
            f"保存的 LoRA 权重过大: {total_size / 1024 / 1024:.1f}MB"

        # 加载到新模型并验证权重恢复
        model2 = _create_small_model()
        model2 = load_lora_weights(model2, save_path)

        # 验证加载后的模型有 LoRA 参数
        lora_params2 = {n: p for n, p in model2.named_parameters() if 'lora_' in n}
        assert len(lora_params2) > 0, "加载后模型缺少 LoRA 参数"

        # 验证加载的权重值和保存的一致（之前 fill_(0.42)）
        found_042 = False
        for n, p in model2.named_parameters():
            if 'lora_' in n and p.requires_grad:
                if torch.allclose(p.data, torch.full_like(p.data, 0.42), atol=1e-5):
                    found_042 = True
                    break
        assert found_042, "加载后 LoRA 权重值未恢复（期望包含 0.42）"

        print(
            f"[PASS] Test 4: LoRA 保存/加载 — "
            f"保存文件: {saved_files}, 大小: {total_size / 1024:.1f}KB, "
            f"加载后 LoRA 参数: {len(lora_params2)}, 权重值恢复: OK"
        )


def test_config_integration():
    """Test 5: 配置集成测试（与 config_utils 联动）。"""
    # 加载 ss_flow_art 配置并验证 lora 段落从 base.yaml 继承
    cfg = load_config(
        os.path.join(PROJECT_ROOT, 'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml')
    )
    d = config_to_dict(cfg)

    assert 'lora' in d, "ss_flow_art 配置缺少 lora 段落（应从 base.yaml 继承）"
    assert d['lora']['enabled'] == False, \
        f"默认 lora.enabled 应为 False, 实际: {d['lora']['enabled']}"
    assert d['lora']['rank'] == 16, \
        f"默认 lora.rank 应为 16，实际: {d['lora']['rank']}"

    # 测试 CLI 覆盖 LoRA 配置
    cfg2 = load_config(
        os.path.join(PROJECT_ROOT, 'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml'),
        overrides=['lora.enabled=true', 'lora.rank=8'],
    )
    d2 = config_to_dict(cfg2)
    assert d2['lora']['enabled'] == True, "CLI 覆盖 lora.enabled 失败"
    assert d2['lora']['rank'] == 8, \
        f"CLI 覆盖 lora.rank 失败: {d2['lora']['rank']}"

    print("[PASS] Test 5: 配置集成 — lora 段落正确继承且可 CLI 覆盖")


def main():
    print("=" * 60)
    print("LoRA 集成测试")
    print("=" * 60)

    tests = [
        test_lora_disabled,
        test_lora_enabled,
        test_target_presets,
        test_save_load,
        test_config_integration,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
