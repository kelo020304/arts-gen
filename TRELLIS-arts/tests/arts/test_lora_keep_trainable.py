"""Unit tests for apply_lora_to_model + keep_trainable (Phase 7, Rule E-03)."""

import torch.nn as nn

# 顶级 conftest.py 已注入 TRELLIS-arts/ 到 sys.path。
from trellis.utils.arts.lora_utils import apply_lora_to_model


class MockTrellisDiT(nn.Module):
    """Mock model covering Stage 4 naming patterns."""

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(8, 16)
        self.input_blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.blocks = nn.ModuleList()
        for _ in range(3):
            block = nn.Module()
            block.to_qkv = nn.Linear(16, 48)
            block.to_out = nn.Linear(16, 16)
            self.blocks.append(block)
        self.out_blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.out_layer = nn.Linear(16, 8)


def _names_of_trainable(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]


def test_T1_no_keep_trainable_lora_only():
    model = MockTrellisDiT()
    cfg = {
        'enabled': True,
        'rank': 4,
        'alpha': 8,
        'dropout': 0.0,
        'target_modules': ['to_qkv', 'to_out'],
    }
    model = apply_lora_to_model(model, cfg)
    trainable = _names_of_trainable(model)
    assert trainable
    assert all('lora_' in name for name in trainable)


def test_T2_substring_match_precise():
    model = MockTrellisDiT()
    cfg = {
        'enabled': True,
        'rank': 4,
        'alpha': 8,
        'dropout': 0.0,
        'target_modules': ['to_qkv'],
        'keep_trainable': ['input_layer'],
    }
    model = apply_lora_to_model(model, cfg)
    trainable = _names_of_trainable(model)
    input_layer_names = [n for n in trainable if 'input_layer' in n and 'lora_' not in n]
    input_blocks_names = [n for n in trainable if 'input_blocks' in n and 'lora_' not in n]
    assert len(input_layer_names) == 2
    assert len(input_blocks_names) == 0


def test_T3_no_misfire_blocks_vs_input_blocks():
    model = MockTrellisDiT()
    cfg = {
        'enabled': True,
        'rank': 4,
        'alpha': 8,
        'dropout': 0.0,
        'target_modules': ['to_qkv'],
        'keep_trainable': ['input_blocks', 'out_blocks'],
    }
    model = apply_lora_to_model(model, cfg)
    trainable = _names_of_trainable(model)
    trunk_non_lora = [
        name for name in trainable
        if name.startswith('base_model.model.blocks.')
        and 'lora_' not in name
        and 'input_blocks' not in name
        and 'out_blocks' not in name
    ]
    assert len(trunk_non_lora) == 0
    assert any('input_blocks' in name and 'lora_' not in name for name in trainable)
    assert any('out_blocks' in name and 'lora_' not in name for name in trainable)


def test_T4_multi_pattern():
    model = MockTrellisDiT()
    cfg = {
        'enabled': True,
        'rank': 4,
        'alpha': 8,
        'dropout': 0.0,
        'target_modules': ['to_qkv'],
        'keep_trainable': ['input_layer', 'out_layer'],
    }
    model = apply_lora_to_model(model, cfg)
    trainable = _names_of_trainable(model)
    assert any('input_layer' in name and 'lora_' not in name for name in trainable)
    assert any('out_layer' in name and 'lora_' not in name for name in trainable)
    assert any('lora_' in name for name in trainable)


def test_T5_disabled_lora_no_keep_trainable_effect():
    model = MockTrellisDiT()
    before_ids = {name: id(param) for name, param in model.named_parameters()}
    cfg = {'enabled': False, 'keep_trainable': ['input_layer', 'out_layer']}
    model_after = apply_lora_to_model(model, cfg)
    after_ids = {name: id(param) for name, param in model_after.named_parameters()}
    assert model_after is model
    assert before_ids == after_ids


def main():
    tests = [
        test_T1_no_keep_trainable_lora_only,
        test_T2_substring_match_precise,
        test_T3_no_misfire_blocks_vs_input_blocks,
        test_T4_multi_pattern,
        test_T5_disabled_lora_no_keep_trainable_effect,
    ]
    for test in tests:
        test()
        print(f'[PASS] {test.__name__}')


if __name__ == '__main__':
    main()
