"""Integration tests for the part-flow set-binding fixes.

These exercise the REAL ``PartSSLatentFlowModel`` backbone (not the capture
mocks used by ``test_model_forward.py``) so they verify the actual attention
wiring of the binding fixes:

  * fix 1 — joint cross-part attention couples parts of one object
            (perturbing part i changes part j) while ``cross_part_attention=False``
            keeps parts independent.
  * fix 2 — per-part identity embedding changes the output when the target slot
            id changes.
  * fix 7 — self-conditioning accepts ``x_self_cond`` with shape preserved and
            doubled backbone input channels handled.
  * fix 8 — classifier-free-guidance ``drop_part_cond`` produces a different
            output than the conditioned forward.
  * a no-OOM CPU sanity for K=8 parts at patch_size=2 (joint attention).

Loss-side fixes (part_shuffle, DeltaFM velocity-contrastive, logit-normal t,
per-channel latent norm) are exercised against the loss module here too, to
keep the binding-fix coverage in one place.

The real ``SparseStructureFlowModel`` backbone zero-inits ``out_layer`` so an
untrained model would emit all-zero velocity and make perturbation tests
vacuous; ``_break_zero_init`` randomizes ``out_layer`` so the network produces
observable signal. All tests run on CPU with ``ATTN_BACKEND=sdpa``.
"""

import pytest
import torch

from trellis.models.part_flow.part_ss_latent_flow import PartSSLatentFlowModel
from trellis.trainers.arts.part_ss_latent_flow_losses import (
    PartSSLatentRFLoss,
    sample_part_ss_latent,
)


# ----------------------------------------------------------------------
# Real-model helpers
# ----------------------------------------------------------------------
def _tiny_model(**overrides) -> PartSSLatentFlowModel:
    """Small CPU-runnable model with the REAL SparseStructureFlowModel backbone."""
    cfg = dict(
        resolution=16,
        latent_channels=4,
        model_channels=32,
        cond_dim=64,
        num_blocks=4,  # even+odd blocks so the joint parity alternation actually fires
        num_heads=4,
        patch_size=1,
        num_views=2,
        max_parts=8,
        num_part_query_layers=1,
        part_label_vocab_size=16,
        require_part_token=True,
        use_fp16=False,
        use_checkpoint=False,
    )
    cfg.update(overrides)
    model = PartSSLatentFlowModel(**cfg)
    _break_zero_init(model)
    model.eval()
    return model


def _break_zero_init(model: PartSSLatentFlowModel) -> None:
    """The backbone zero-inits out_layer/adaLN so it emits zeros until trained.
    Randomize them so forward produces observable, non-trivial signal."""
    bb = model.backbone
    with torch.no_grad():
        torch.nn.init.normal_(bb.out_layer.weight, std=0.1)
        torch.nn.init.normal_(bb.out_layer.bias, std=0.1)
        for block in bb.blocks:
            if not block.share_mod:
                torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
                torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        if bb.share_mod:
            torch.nn.init.normal_(bb.adaLN_modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(bb.adaLN_modulation[-1].bias, std=0.05)


def _inputs(model: PartSSLatentFlowModel, K: int = 3, B: int = 1, seed: int = 0):
    """Build a consistent (x, t, z, cond, mask, valid, slots) tuple for K parts."""
    torch.manual_seed(seed)
    C = model.latent_channels
    R = model.resolution
    V = model.num_views
    T = 8  # cond tokens per view
    x = torch.randn(B, K, C, R, R, R)
    z = torch.randn(B, C, R, R, R)
    cond = torch.randn(B, V * T, 64)
    mask = torch.zeros(B, V * T, dtype=torch.long)
    slots = torch.arange(1, K + 1).view(1, K).expand(B, K).contiguous()
    # give every target slot some 2D mask coverage so require_part_token is happy
    for k in range(K):
        mask[:, 2 * k : 2 * k + 2] = k + 1
    valid = torch.ones(B, K, dtype=torch.bool)
    t = torch.full((B,), 0.5)
    return x, t, z, cond, mask, valid, slots


# ----------------------------------------------------------------------
# Fix 1: cross-part attention couples parts (and OFF preserves independence)
# ----------------------------------------------------------------------
def test_joint_cross_part_attention_returns_correct_shape():
    model = _tiny_model(cross_part_attention=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    out = model(x, t, z, cond, mask, valid, slots)
    assert out.shape == (1, 3, model.latent_channels, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_cross_part_attention_couples_parts_when_on():
    """Perturbing part 0's latent input must change part 1's output when joint
    cross-part attention is enabled."""
    model = _tiny_model(cross_part_attention=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0, 0] += 5.0  # perturb only part 0's input
        pert = model(x2, t, z, cond, mask, valid, slots)
    delta_part1 = (pert[0, 1] - base[0, 1]).abs().max().item()
    delta_part2 = (pert[0, 2] - base[0, 2]).abs().max().item()
    assert delta_part1 > 1e-4, f"part1 should react to part0 perturbation, got {delta_part1}"
    assert delta_part2 > 1e-4, f"part2 should react to part0 perturbation, got {delta_part2}"


def test_independent_forward_keeps_parts_independent_when_off():
    """With cross_part_attention=False the legacy per-part path must NOT let
    part 0's perturbation leak into part 1 / part 2."""
    model = _tiny_model(cross_part_attention=False)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0, 0] += 5.0
        pert = model(x2, t, z, cond, mask, valid, slots)
    assert (pert[0, 0] - base[0, 0]).abs().max().item() > 1e-4  # part0 itself changes
    assert torch.equal(pert[0, 1], base[0, 1])  # part1 untouched
    assert torch.equal(pert[0, 2], base[0, 2])  # part2 untouched


def test_cross_part_attention_couples_via_condition_perturbation():
    """Joint attention must also propagate a per-part CONDITION change across
    parts (not only the latent input)."""
    model = _tiny_model(cross_part_attention=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        cond2 = cond.clone()
        cond2[0, 0:2] += 5.0  # perturb part 0's mask-covered condition tokens
        pert = model(x, t, z, cond2, mask, valid, slots)
    assert (pert[0, 1] - base[0, 1]).abs().max().item() > 1e-5
    assert (pert[0, 2] - base[0, 2]).abs().max().item() > 1e-5


def test_independent_forward_isolates_per_object_under_cross_part():
    """Joint forward must isolate DIFFERENT objects: perturbing object 0 must
    not change object 1's output (batch isolation)."""
    model = _tiny_model(cross_part_attention=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2, B=2)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0] += 5.0  # perturb whole object 0
        pert = model(x2, t, z, cond, mask, valid, slots)
    assert (pert[0] - base[0]).abs().max().item() > 1e-4  # object 0 changes
    assert torch.equal(pert[1], base[1])  # object 1 isolated


# ----------------------------------------------------------------------
# Fix 2: per-part identity embedding
# ----------------------------------------------------------------------
def test_identity_embedding_changes_output_when_slot_changes():
    """Changing target_slots (and thus the per-part identity index) must change
    the output when token_identity_embedding=True."""
    model = _tiny_model(cross_part_attention=True, token_identity_embedding=True)
    # Make the identity embedding non-degenerate so distinct slots map distinctly.
    with torch.no_grad():
        torch.nn.init.normal_(model.part_slot_emb.weight, std=1.0)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2)
    mask = torch.zeros_like(mask)
    mask[:, 0:2] = 5
    mask[:, 2:4] = 6
    with torch.no_grad():
        out_a = model(x, t, z, cond, mask, valid, torch.tensor([[5, 6]]))
        out_b = model(x, t, z, cond, mask, valid, torch.tensor([[6, 5]]))
    assert not torch.allclose(out_a, out_b), "identity embedding had no effect on output"


def test_identity_embedding_off_path_runs_and_changes_with_slot():
    """Identity embedding on the independent (legacy) path too."""
    model = _tiny_model(cross_part_attention=False, token_identity_embedding=True)
    with torch.no_grad():
        torch.nn.init.normal_(model.part_slot_emb.weight, std=1.0)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2)
    mask = torch.zeros_like(mask)
    mask[:, 0:2] = 5
    mask[:, 2:4] = 6
    with torch.no_grad():
        out_a = model(x, t, z, cond, mask, valid, torch.tensor([[5, 6]]))
        out_b = model(x, t, z, cond, mask, valid, torch.tensor([[6, 5]]))
    assert not torch.allclose(out_a, out_b)


# ----------------------------------------------------------------------
# Fix 7: self-conditioning
# ----------------------------------------------------------------------
def test_self_conditioning_doubles_backbone_input_channels():
    model = _tiny_model(self_conditioning=True)
    # x_t (4) + x_self_cond (4) = 8 in_channels.
    assert model.backbone.in_channels == model.latent_channels * 2
    assert model.in_channel_mult == 2


def test_self_conditioning_accepts_x_self_cond_and_preserves_shape():
    model = _tiny_model(cross_part_attention=True, self_conditioning=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    x_self = torch.randn_like(x)
    with torch.no_grad():
        out = model(x, t, z, cond, mask, valid, slots, x_self_cond=x_self)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_self_conditioning_estimate_changes_output():
    """A non-zero x_self_cond must change the prediction vs. the zeros default."""
    model = _tiny_model(cross_part_attention=True, self_conditioning=True)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2)
    with torch.no_grad():
        out_zero = model(x, t, z, cond, mask, valid, slots, x_self_cond=torch.zeros_like(x))
        out_nonzero = model(x, t, z, cond, mask, valid, slots, x_self_cond=torch.randn_like(x))
    assert not torch.allclose(out_zero, out_nonzero)


def test_self_conditioning_raises_when_flag_off():
    model = _tiny_model(self_conditioning=False)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2)
    with pytest.raises(ValueError, match="self_conditioning"):
        model(x, t, z, cond, mask, valid, slots, x_self_cond=torch.zeros_like(x))


# ----------------------------------------------------------------------
# Fix 8: classifier-free guidance drop_part_cond
# ----------------------------------------------------------------------
def test_cfg_drop_part_cond_changes_output_joint():
    model = _tiny_model(cross_part_attention=True, classifier_free_guidance=True)
    # Make the null embeddings distinct from the real condition.
    with torch.no_grad():
        model.null_part_query.normal_(0, 1.0)
        model.null_cond_token.normal_(0, 1.0)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        cond_out = model(x, t, z, cond, mask, valid, slots, drop_part_cond=False)
        drop_out = model(x, t, z, cond, mask, valid, slots, drop_part_cond=True)
    assert not torch.allclose(cond_out, drop_out), "CFG drop did not change output"


def test_cfg_drop_part_cond_changes_output_independent():
    model = _tiny_model(cross_part_attention=False, classifier_free_guidance=True)
    with torch.no_grad():
        model.null_part_query.normal_(0, 1.0)
        model.null_cond_token.normal_(0, 1.0)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        cond_out = model(x, t, z, cond, mask, valid, slots, drop_part_cond=False)
        drop_out = model(x, t, z, cond, mask, valid, slots, drop_part_cond=True)
    assert not torch.allclose(cond_out, drop_out)


def test_cfg_per_part_drop_only_changes_dropped_part_under_independent():
    """Under the INDEPENDENT path, dropping only part 0's condition must not
    change part 1's output (per-part isolation), proving the mask is per-part."""
    model = _tiny_model(cross_part_attention=False, classifier_free_guidance=True)
    with torch.no_grad():
        model.null_part_query.normal_(0, 1.0)
        model.null_cond_token.normal_(0, 1.0)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=2)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots, drop_part_cond=False)
        drop0 = model(
            x, t, z, cond, mask, valid, slots,
            drop_part_cond=torch.tensor([[True, False]]),
        )
    assert not torch.allclose(drop0[0, 0], base[0, 0])  # dropped part changes
    assert torch.allclose(drop0[0, 1], base[0, 1])  # kept part unchanged


# ----------------------------------------------------------------------
# No-OOM sanity: K=8, patch_size=2, joint attention runs on CPU
# ----------------------------------------------------------------------
def test_joint_attention_k8_patch2_runs_on_cpu():
    model = _tiny_model(cross_part_attention=True, patch_size=2, max_parts=8)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=8)
    with torch.no_grad():
        out = model(x, t, z, cond, mask, valid, slots)
    assert out.shape == (1, 8, model.latent_channels, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_joint_attention_backward_finite_grads():
    """Gradient checkpointing + joint forward must give finite grads end-to-end."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        use_checkpoint=True,
    )
    model.train()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    x.requires_grad_(True)
    out = model(x, t, z, cond, mask, valid, slots)
    out.sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert torch.isfinite(x.grad).all()


# ----------------------------------------------------------------------
# Loss-side binding fixes against the real loss module
# ----------------------------------------------------------------------
def _loss_batch(x_1: torch.Tensor, valid: torch.Tensor):
    B = x_1.shape[0]
    return {
        "x_1_parts": x_1,
        "part_valid": valid,
        "part_raw_voxel_counts": torch.full(valid.shape, 10.0),
        "part_fg_mask": torch.ones(
            x_1.shape[0],
            x_1.shape[1],
            *x_1.shape[3:],
            dtype=torch.bool,
        ),
        "z_global": torch.zeros(B, x_1.shape[2], 16, 16, 16),
        "cond": torch.zeros(B, 12, 1024),
        "mask_token_labels": torch.ones(B, 12, dtype=torch.long),
        "target_slots": torch.ones(x_1.shape[:2], dtype=torch.long),
        "debug_t": torch.full((B,), 0.5),
        "debug_noise": torch.zeros_like(x_1),
    }


class _TrainableConstVelocity(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = torch.nn.Parameter(output.clone())

    def forward(self, x_t_parts, t, z_global, cond, mask_token_labels, part_valid, target_slots, **kwargs):
        return self.output.to(device=x_t_parts.device, dtype=x_t_parts.dtype).expand_as(x_t_parts)


def test_loss_part_shuffle_keeps_loss_finite_and_grads_flow():
    x_1 = torch.zeros(1, 3, 2, 1, 1, 1)
    x_1[0, 0] = 1.0
    x_1[0, 1] = 2.0
    x_1[0, 2] = 3.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    batch = _loss_batch(x_1, valid)
    model = _TrainableConstVelocity(torch.zeros(1, 3, 2, 1, 1, 1))
    criterion = PartSSLatentRFLoss(
        t_min=0.0, t_max=1.0, part_shuffle=True, velocity_contrastive_weight=0.0
    )
    loss, metrics = criterion(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.output.grad).all()
    assert metrics["loss_total"] == pytest.approx(loss.detach().item())


def test_velocity_contrastive_finite_and_lower_when_aligned():
    """DeltaFM velocity-contrastive: finite, and aligned v_pred gives a lower
    contrastive loss than swapped (mis-bound) v_pred."""
    x_1 = torch.zeros(1, 2, 4, 1, 1, 1)
    x_1[0, 0, :, 0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x_1[0, 1, :, 0, 0, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    valid = torch.ones(1, 2, dtype=torch.bool)
    batch = _loss_batch(x_1, valid)
    batch["debug_t"] = torch.ones(1)  # t=1, noise=0 -> v_target == x_1
    criterion = PartSSLatentRFLoss(
        t_min=0.0, t_max=1.0, velocity_contrastive_weight=1.0, velocity_contrastive_lambda=0.05
    )

    class _Fixed(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, x_t_parts, *a, **k):
            return self.out.to(x_t_parts)

    aligned_loss, aligned_m = criterion(_Fixed(x_1), batch)
    swapped_loss, swapped_m = criterion(_Fixed(x_1.flip(dims=[1])), batch)
    assert torch.isfinite(aligned_loss) and torch.isfinite(swapped_loss)
    assert aligned_m["velocity_contrastive_loss"] < swapped_m["velocity_contrastive_loss"]
    assert aligned_m["velocity_contrastive_acc"] == 1.0
    assert swapped_m["velocity_contrastive_acc"] == 0.0


def test_logit_normal_t_schedule_in_open_unit_interval():
    torch.manual_seed(0)
    criterion = PartSSLatentRFLoss(t_min=0.0, t_max=1.0, t_schedule="logit_normal")
    t = criterion._sample_t(4096, device=torch.device("cpu"), dtype=torch.float32)
    assert bool((t > 0.0).all()) and bool((t < 1.0).all())
    assert abs(float(t.mean()) - 0.5) < 0.03


def test_per_channel_latent_norm_round_trips_identity():
    """normalize then denormalize must recover the raw latent per channel."""
    channels = 4
    mean = torch.tensor([1.0, -2.0, 3.0, 0.5])
    std = torch.tensor([2.0, 0.5, 1.0, 3.0])
    criterion = PartSSLatentRFLoss(
        t_min=0.0, t_max=1.0, latent_norm_mode="per_channel",
        latent_channels=channels, latent_mean=mean, latent_std=std,
        velocity_contrastive_weight=0.0,
    )
    raw = torch.randn(2, 3, channels, 4, 4, 4)
    norm = criterion._normalize_latent(raw)
    back = criterion._denormalize_latent(norm)
    assert torch.allclose(back, raw, atol=1e-5)
    # normalization actually moves the data (not a no-op) for non-trivial stats.
    assert not torch.allclose(norm, raw)


def test_sampler_runs_with_self_conditioning_and_cfg_on_real_model():
    """End-to-end eval path: sampler with self-conditioning + CFG on the real
    model returns finite latents of the right shape (no OOM / no crash)."""
    model = _tiny_model(
        cross_part_attention=True, self_conditioning=True, classifier_free_guidance=True
    )
    z = sample_part_ss_latent(
        model,
        z_global=torch.randn(1, model.latent_channels, 16, 16, 16),
        cond=torch.randn(1, 2 * 8, 64),
        mask_token_labels=_inputs(model, K=2)[4],
        part_valid=torch.ones(1, 2, dtype=torch.bool),
        target_slots=torch.tensor([[1, 2]]),
        num_steps=2,
        noise_scale=1.0,
        self_conditioning=True,
        cfg_scale=2.0,
    )
    assert z.shape == (1, 2, model.latent_channels, 16, 16, 16)
    assert torch.isfinite(z).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA fp16 autocast")
def test_grad_checkpoint_under_fp16_autocast_self_cond_double_pass():
    """Regression for the CheckpointError that crashed real training: gradient
    checkpointing + cuda fp16 autocast + the loss's no_grad self-conditioning
    pre-pass. With autocast weight-cache ON the pre-pass poisons the cache so
    checkpoint recompute mismatches; the trainer fixes this with
    cache_enabled=False. This test reproduces the exact path (use_checkpoint=True,
    self_conditioning, cross_part_attention) and asserts a finite backward. The
    prior CPU tests missed it because they never ran cuda fp16 autocast."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        self_conditioning=True,
        classifier_free_guidance=True,
        use_checkpoint=True,
        num_blocks=4,
    ).cuda().train()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    x = x.cuda().requires_grad_(True)
    t, z, cond, mask, valid, slots = (v.cuda() for v in (t, z, cond, mask, valid, slots))
    x_self = torch.randn_like(x)
    with torch.cuda.amp.autocast(enabled=True, cache_enabled=False):
        with torch.no_grad():  # self-cond first pass poisons the autocast cache
            model(x, t, z, cond, mask, valid, slots,
                  x_self_cond=torch.zeros_like(x), drop_part_cond=False)
        out = model(x, t, z, cond, mask, valid, slots, x_self_cond=x_self, drop_part_cond=False)
        loss = out.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).item()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ----------------------------------------------------------------------
# Summary-token (ISAB / Perceiver) cross-part interaction for EVEN blocks
# ----------------------------------------------------------------------
def _summary_model(**overrides) -> PartSSLatentFlowModel:
    """Tiny model with the summary-token path on. patch_size=4 keeps T=4^3=64
    small on CPU; n_summary_tokens=8 is a coarse 2x2x2 grid (a perfect cube so
    the spatial pos emb stays well-defined)."""
    cfg = dict(
        cross_part_attention=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        patch_size=4,
        max_parts=8,
    )
    cfg.update(overrides)
    return _tiny_model(**cfg)


def test_summary_path_returns_correct_shape_and_finite():
    model = _summary_model()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        out = model(x, t, z, cond, mask, valid, slots)
    assert out.shape == (1, 3, model.latent_channels, 16, 16, 16)
    assert torch.isfinite(out).all()


def test_summary_path_requires_cross_part_attention():
    """summary_cross_part_attention without cross_part_attention must fail loud."""
    with pytest.raises(ValueError, match="requires cross_part_attention"):
        _tiny_model(
            cross_part_attention=False,
            summary_cross_part_attention=True,
            n_summary_tokens=8,
            patch_size=4,
        )


def test_summary_path_preserves_cross_part_coupling():
    """THE POINT: with the summary path on, perturbing part 0's input must still
    change parts 1 AND 2 (broadcast carries cross-part info through the summaries)."""
    model = _summary_model()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0, 0] += 5.0  # perturb only part 0
        pert = model(x2, t, z, cond, mask, valid, slots)
    delta1 = (pert[0, 1] - base[0, 1]).abs().max().item()
    delta2 = (pert[0, 2] - base[0, 2]).abs().max().item()
    assert delta1 > 1e-4, f"part1 must react to part0 perturbation via summaries, got {delta1}"
    assert delta2 > 1e-4, f"part2 must react to part0 perturbation via summaries, got {delta2}"


def test_summary_path_preserves_per_part_identity():
    """Perturbing ONLY part 0 changes part 0's output more than a no-op, and parts
    with DIFFERENT footprints get DIFFERENT outputs (summaries not collapsed)."""
    model = _summary_model()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0, 0] += 5.0
        pert = model(x2, t, z, cond, mask, valid, slots)
    self_delta = (pert[0, 0] - base[0, 0]).abs().max().item()
    sibling_delta = (pert[0, 1] - base[0, 1]).abs().max().item()
    # The perturbed part must move more than the unperturbed sibling (the summary
    # pool/broadcast preserve where the change came from rather than washing it
    # uniformly across all parts).
    assert self_delta > sibling_delta, (
        f"perturbed part0 should move most, got self={self_delta} sibling={sibling_delta}"
    )
    # Distinct per-part inputs/footprints produce distinct per-part outputs
    # (not a single collapsed summary broadcast to identical parts).
    assert not torch.allclose(base[0, 0], base[0, 1])
    assert not torch.allclose(base[0, 1], base[0, 2])


def test_summary_path_off_matches_full_joint_behavior():
    """summary_cross_part_attention=False must reproduce the full [1,K*T,C] joint
    path EXACTLY (same weights, same inputs, identical output)."""
    base = _tiny_model(cross_part_attention=True, patch_size=4, max_parts=8)
    summ = _tiny_model(
        cross_part_attention=True,
        summary_cross_part_attention=False,
        n_summary_tokens=8,
        patch_size=4,
        max_parts=8,
    )
    # Copy base weights into summ so only the flag differs. The OFF path must not
    # have created any summary params (so state_dicts match) and must run the
    # legacy global self-attn.
    assert not hasattr(summ, "summary_queries")
    summ.load_state_dict(base.state_dict())
    x, t, z, cond, mask, valid, slots = _inputs(base, K=3)
    with torch.no_grad():
        out_base = base(x, t, z, cond, mask, valid, slots)
        out_summ = summ(x, t, z, cond, mask, valid, slots)
    assert torch.equal(out_base, out_summ)


def test_summary_path_grad_finite_through_summary_params():
    """Backward must give finite grads through summary_queries / summary_pool_attn
    / summary_broadcast_attn."""
    model = _summary_model(token_identity_embedding=True, use_checkpoint=True)
    model.train()
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    x.requires_grad_(True)
    out = model(x, t, z, cond, mask, valid, slots)
    out.sum().backward()
    for pname in ("summary_queries",):
        g = getattr(model, pname).grad
        assert g is not None and torch.isfinite(g).all(), f"bad grad in {pname}"
    for mod_name in ("summary_pool_attn", "summary_broadcast_attn"):
        mod = getattr(model, mod_name)
        saw_grad = False
        for name, p in mod.named_parameters():
            if p.grad is not None:
                saw_grad = True
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {mod_name}.{name}"
        assert saw_grad, f"no grad reached {mod_name}"
    assert torch.isfinite(x.grad).all()


def test_soft_role_marking_destarves_subpatch_parts():
    """Soft role marking: the cross-attn target/context role tokens use the soft
    overlap weights instead of the hard min_fg vote, so a part whose slot is ABSENT
    from mask_token_labels (sub-patch / starved under the hard vote) still gets a
    distinct, weight-driven role -> its output differs from the hard-vote path."""
    model = _tiny_model(cross_part_attention=True, token_identity_embedding=True,
                        soft_role_marking=True, require_part_token=False)
    x, t, z, cond, mask, valid, slots = _inputs(model, K=3)
    T = cond.shape[1]
    # hard vote contains ONLY slot 1 -> slots 2,3 are starved (all-context under hard)
    mask = torch.zeros(1, T, dtype=torch.long)
    mask[0, 1] = 1
    # each part covers a DIFFERENT single token (distinct soft footprint)
    w = torch.zeros(1, 3, T)
    w[0, 0, 2] = 1.0
    w[0, 1, 3] = 1.0
    w[0, 2, T // 2 + 2] = 1.0
    with torch.no_grad():
        model.soft_role_marking = False
        out_hard = model(x, t, z, cond, mask, valid, slots, part_token_weights=w)
        model.soft_role_marking = True
        out_soft = model(x, t, z, cond, mask, valid, slots, part_token_weights=w)
    # role path is active overall
    assert (out_soft - out_hard).abs().max().item() > 1e-4
    # starved parts (slots 2,3, absent from the hard vote) are de-starved: their
    # output responds to the soft role, unlike the all-context hard path
    assert (out_soft[0, 1] - out_hard[0, 1]).abs().max().item() > 1e-4
    assert (out_soft[0, 2] - out_hard[0, 2]).abs().max().item() > 1e-4
