import pytest
import torch
import torch.nn as nn

from trellis.models.part_flow.part_ss_latent_flow import PartSSLatentFlowModel


class CaptureBackbone(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.latent_channels = latent_channels
        self.last_cond_shape = None
        self.last_cond_tokens = None
        self.last_attn_bias = None
        self.last_x_shape = None
        self.call_x_shapes = []

    def forward(self, x, t, cond_tokens, attn_bias=None):
        self.last_x_shape = tuple(x.shape)
        self.last_cond_shape = tuple(cond_tokens.shape)
        self.last_cond_tokens = cond_tokens.detach().clone()
        self.last_attn_bias = None if attn_bias is None else attn_bias.detach().clone()
        self.call_x_shapes.append(tuple(x.shape))
        return x[:, :self.latent_channels]


class HalfBackbone(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.latent_channels = latent_channels

    def forward(self, x, t, cond_tokens):
        return x[:, :self.latent_channels].half()


def _model():
    model = PartSSLatentFlowModel(
        resolution=16,
        latent_channels=8,
        model_channels=128,
        cond_dim=1024,
        num_blocks=2,
        num_heads=4,
        patch_size=1,
        num_views=4,
        max_parts=4,
        num_part_query_layers=1,
        part_label_vocab_size=64,
        require_part_token=True,
        use_fp16=False,
        use_checkpoint=False,
    )
    model.backbone = CaptureBackbone(latent_channels=8)
    return model


def test_joint_part_ss_latent_flow_forward_shape():
    model = _model()
    x_t = torch.randn(2, 3, 8, 16, 16, 16)
    z_global = torch.randn(2, 8, 16, 16, 16)
    cond = torch.randn(2, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(2, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    mask_token_labels[:, 40:60] = 2
    mask_token_labels[:, 70:90] = 3
    part_valid = torch.tensor([[True, True, False], [True, True, True]])
    target_slots = torch.tensor([[1, 2, 0], [1, 2, 3]])
    t = torch.tensor([0.25, 0.75])
    out = model(x_t, t, z_global, cond, mask_token_labels, part_valid, target_slots)
    assert out.shape == (2, 3, 8, 16, 16, 16)
    assert torch.all(out[0, 2] == 0)


def test_joint_model_uses_full_cond_tokens_and_all_part_queries():
    model = _model()
    x_t = torch.randn(2, 3, 8, 16, 16, 16)
    z_global = torch.randn(2, 8, 16, 16, 16)
    cond = torch.randn(2, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(2, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    mask_token_labels[:, 40:60] = 2
    mask_token_labels[:, 70:90] = 3
    part_valid = torch.tensor([[True, True, False], [True, True, True]])
    target_slots = torch.tensor([[1, 2, 0], [1, 2, 3]])
    _ = model(x_t, torch.tensor([0.1, 0.2]), z_global, cond, mask_token_labels, part_valid, target_slots)
    valid_parts = int(part_valid.sum().item())
    assert model.backbone.last_x_shape == (valid_parts, 8, 16, 16, 16)
    assert model.backbone.last_cond_shape == (valid_parts, 3 + cond.shape[1] + 512, 128)


def test_model_chunks_valid_parts_through_backbone():
    model = _model()
    model.max_part_forward_batch = 2
    x_t = torch.randn(2, 3, 8, 16, 16, 16)
    z_global = torch.randn(2, 8, 16, 16, 16)
    cond = torch.randn(2, 8, 1024)
    mask_token_labels = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 1, 2],
            [0, 1, 2, 3, 3, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    part_valid = torch.tensor([[True, True, False], [True, True, True]])
    target_slots = torch.tensor([[1, 2, 0], [1, 2, 3]])

    out = model(x_t, torch.tensor([0.1, 0.2]), z_global, cond, mask_token_labels, part_valid, target_slots)

    assert out.shape == x_t.shape
    assert torch.all(out[0, 2] == 0)
    assert [shape[0] for shape in model.backbone.call_x_shapes] == [2, 2, 1]


def test_chunk_zero_equals_chunked_when_backbone_deterministic():
    model = _model()
    x_t = torch.randn(2, 3, 8, 16, 16, 16)
    z_global = torch.randn(2, 8, 16, 16, 16)
    cond = torch.randn(2, 8, 1024)
    mask_token_labels = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 1, 2],
            [0, 1, 2, 3, 3, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    part_valid = torch.tensor([[True, True, False], [True, True, True]])
    target_slots = torch.tensor([[1, 2, 0], [1, 2, 3]])
    t = torch.tensor([0.1, 0.2])

    model.max_part_forward_batch = 0
    out_no_chunk = model(x_t, t, z_global, cond, mask_token_labels, part_valid, target_slots)
    model.backbone.call_x_shapes.clear()
    model.max_part_forward_batch = 2
    out_chunked = model(x_t, t, z_global, cond, mask_token_labels, part_valid, target_slots)

    assert torch.equal(out_chunked, out_no_chunk)
    assert [shape[0] for shape in model.backbone.call_x_shapes] == [2, 2, 1]


def test_model_can_optionally_concat_global_latent_to_backbone_input():
    model = PartSSLatentFlowModel(
        resolution=16,
        latent_channels=8,
        model_channels=128,
        cond_dim=1024,
        num_blocks=2,
        num_heads=4,
        patch_size=1,
        num_views=4,
        max_parts=2,
        num_part_query_layers=1,
        part_label_vocab_size=64,
        require_part_token=True,
        use_fp16=False,
        use_checkpoint=False,
        concat_global=True,
    )
    model.backbone = CaptureBackbone(latent_channels=8)
    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.tensor([[0, 1, 1, 2, 2, 0, 1, 2]], dtype=torch.long)
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])

    _ = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)

    assert model.backbone.last_x_shape == (2, 16, 16, 16, 16)


def test_model_builds_target_specific_cond_memory():
    model = _model()
    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(1, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    mask_token_labels[:, 40:60] = 2
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])

    _ = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)

    cond_tokens = model.backbone.last_cond_tokens
    assert model.backbone.last_cond_shape == (2, 2 + cond.shape[1] + 512, 128)
    assert not torch.allclose(cond_tokens[0, 2:2 + cond.shape[1]], cond_tokens[1, 2:2 + cond.shape[1]])


def test_target_specific_cond_memory_marks_only_current_slot_tokens():
    model = _model()
    with torch.no_grad():
        model.target_token_emb.fill_(2.0)
        model.context_token_emb.fill_(-3.0)

    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.tensor([[0, 1, 1, 2, 2, 0, 1, 2]], dtype=torch.long)
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])

    _ = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)

    prefix_len = int(part_valid.shape[1])
    slot_1_suffix = model.backbone.last_cond_tokens[0, prefix_len:]
    slot_2_suffix = model.backbone.last_cond_tokens[1, prefix_len:]
    diff = slot_1_suffix[:cond.shape[1]] - slot_2_suffix[:cond.shape[1]]
    labels = mask_token_labels[0]
    expected_delta = float(model.target_token_emb[0, 0] - model.context_token_emb[0, 0])

    assert torch.allclose(diff[labels == 0], torch.zeros_like(diff[labels == 0]))
    assert torch.allclose(diff[labels == 1], torch.full_like(diff[labels == 1], expected_delta))
    assert torch.allclose(diff[labels == 2], torch.full_like(diff[labels == 2], -expected_delta))


def test_part_queries_are_target_marked_without_duplicate_q_prefix():
    model = _model()
    with torch.no_grad():
        model.target_query_emb.fill_(4.0)
        model.context_query_emb.fill_(-1.0)

    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.tensor([[0, 1, 1, 2, 2, 0, 1, 2]], dtype=torch.long)
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])

    _ = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)

    query_tokens_for_slot_1 = model.backbone.last_cond_tokens[0, :2]
    query_tokens_for_slot_2 = model.backbone.last_cond_tokens[1, :2]
    diff = query_tokens_for_slot_1 - query_tokens_for_slot_2
    expected_delta = float(model.target_query_emb[0, 0] - model.context_query_emb[0, 0])

    assert model.backbone.last_cond_shape == (2, 2 + cond.shape[1] + 512, 128)
    assert torch.allclose(diff[0], torch.full_like(diff[0], expected_delta))
    assert torch.allclose(diff[1], torch.full_like(diff[1], -expected_delta))


def test_global_ss_latent_is_encoded_as_512_condition_tokens():
    model = _model()
    x_t = torch.randn(1, 1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    part_valid = torch.tensor([[True]])
    target_slots = torch.tensor([[1]])
    z_global_a = torch.zeros(1, 8, 16, 16, 16)
    z_global_b = z_global_a.clone()
    z_global_b[:, :, :2, :2, :2] = 1.0

    _ = model(x_t, torch.tensor([0.1]), z_global_a, cond, mask_token_labels, part_valid, target_slots)
    memory_a = model.backbone.last_cond_tokens.clone()
    _ = model(x_t, torch.tensor([0.1]), z_global_b, cond, mask_token_labels, part_valid, target_slots)
    memory_b = model.backbone.last_cond_tokens.clone()

    global_start = int(part_valid.shape[1]) + cond.shape[1]
    assert memory_a.shape == (1, global_start + 512, 128)
    assert not torch.allclose(memory_a[:, global_start:], memory_b[:, global_start:])
    assert torch.allclose(memory_a[:, :global_start], memory_b[:, :global_start])


def test_model_raises_when_any_valid_target_slot_has_no_mask_coverage():
    model = _model()
    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(1, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="zero 2D mask token coverage"):
        _ = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)


def test_model_uses_part_token_weights_when_hard_mask_has_zero_coverage():
    model = _model()
    x_t = torch.randn(1, 1, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.zeros(1, 8, dtype=torch.long)
    part_valid = torch.tensor([[True]])
    target_slots = torch.tensor([[1]])
    part_token_weights = torch.zeros(1, 1, 8)
    part_token_weights[0, 0, 3] = 1.0

    out = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )

    assert out.shape == x_t.shape


def test_mask_attention_bias_disabled_forwards_no_attn_bias():
    model = _model()
    x_t = torch.randn(1, 1, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.zeros(1, 8, dtype=torch.long)
    part_valid = torch.tensor([[True]])
    target_slots = torch.tensor([[1]])
    part_token_weights = torch.full((1, 1, 8), 0.125)

    _ = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )

    assert model.backbone.last_attn_bias is None


def test_mask_attention_bias_all_one_weights_are_equivalent():
    model = _model()
    model.mask_attention_bias_enabled = True
    x_t = torch.randn(1, 1, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 8, 1024)
    mask_token_labels = torch.zeros(1, 8, dtype=torch.long)
    part_valid = torch.tensor([[True]])
    target_slots = torch.tensor([[1]])
    part_token_weights = torch.ones(1, 1, 8)

    out = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )

    assert out.shape == x_t.shape
    assert model.backbone.last_attn_bias is None


def test_mask_attention_bias_targets_only_image_condition_segment():
    model = _model()
    model.mask_attention_bias_enabled = True
    model.mask_attention_bias_lambda = 2.0
    model.mask_attention_bias_eps = 1.0e-3
    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 4, 1024)
    mask_token_labels = torch.zeros(1, 4, dtype=torch.long)
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])
    part_token_weights = torch.tensor(
        [[[1.0, 0.25, 0.0, 0.5], [0.5, 0.0, 0.25, 1.0]]],
        dtype=torch.float32,
    )

    _ = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )

    bias = model.backbone.last_attn_bias
    assert bias is not None
    assert bias.shape == (2, 1, 1, 2 + 4 + 512)
    assert torch.all(bias[:, :, :, :2] == 0)
    assert torch.all(bias[:, :, :, 6:] == 0)
    expected = 2.0 * torch.log(part_token_weights[0].clamp_min(1.0e-3))
    assert torch.allclose(bias[:, 0, 0, 2:6], expected)


def test_mask_attention_bias_skips_all_zero_rows_and_warns_once(capsys):
    model = _model()
    model.mask_attention_bias_enabled = True
    x_t = torch.randn(1, 1, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 4, 1024)
    mask_token_labels = torch.zeros(1, 4, dtype=torch.long)
    part_valid = torch.tensor([[True]])
    target_slots = torch.tensor([[1]])
    part_token_weights = torch.zeros(1, 1, 4)

    _ = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )
    _ = model(
        x_t,
        torch.tensor([0.1]),
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        part_token_weights=part_token_weights,
    )

    captured = capsys.readouterr()
    assert captured.out.count("[WARN] mask_attention_bias skipped") == 1
    assert model.backbone.last_attn_bias is None


def test_model_raises_when_any_object_has_no_valid_parts():
    model = _model()
    x_t = torch.randn(2, 2, 8, 16, 16, 16)
    z_global = torch.randn(2, 8, 16, 16, 16)
    cond = torch.randn(2, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(2, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    part_valid = torch.tensor([[True, False], [False, False]])
    target_slots = torch.tensor([[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="at least one valid target part"):
        _ = model(x_t, torch.tensor([0.1, 0.2]), z_global, cond, mask_token_labels, part_valid, target_slots)


def test_model_scatter_accepts_amp_half_backbone_output():
    model = _model()
    model.backbone = HalfBackbone(latent_channels=8)
    x_t = torch.randn(1, 2, 8, 16, 16, 16)
    z_global = torch.randn(1, 8, 16, 16, 16)
    cond = torch.randn(1, 4 * 1370, 1024)
    mask_token_labels = torch.zeros(1, 4 * 1370, dtype=torch.long)
    mask_token_labels[:, 10:30] = 1
    mask_token_labels[:, 40:60] = 2
    part_valid = torch.tensor([[True, True]])
    target_slots = torch.tensor([[1, 2]])
    out = model(x_t, torch.tensor([0.1]), z_global, cond, mask_token_labels, part_valid, target_slots)
    assert out.dtype == x_t.dtype
    assert out.shape == x_t.shape
