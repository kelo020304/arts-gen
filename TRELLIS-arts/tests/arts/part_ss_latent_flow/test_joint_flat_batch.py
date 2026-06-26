"""Gate tests for the flat (cross-object batched) joint forward path.

The serial joint forward (``_forward_joint``, ``for b in range(B)``) processes one
object's K parts at a time. ``joint_flat_batch=True`` instead packs ALL objects'
parts into one flat ``[N_total, T, C]`` batch so the expensive per-part ops
(within-part self-attn, cross-attn, MLP) run as a single batched call, while only
the cheap summary-token cross-part mix/broadcast stays grouped per object.

The contract is **numerical equivalence to the serial path**: the flat path does
the exact same math, just batched, so its output must equal the serial path's
output to floating-point tolerance. That single equality is the strongest
possible gate — if flat == serial then flat has

  * no cross-object attention leakage (serial has none), and
  * no within-object binding regression (serial couples parts), and
  * no implementation bug (it reproduces the trusted reference).

Two behavioural gates (object isolation, within-object coupling) are asserted on
the flat path directly as defense in depth. All tests run on CPU.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_binding_fix_integration import _tiny_model


def _multi_inputs(model, valid_counts, seed=0):
    """Multi-object batch with VARYING parts-per-object K_b (the case the flat
    path's object grouping must handle). ``valid_counts[b]`` = #valid parts of
    object b; trailing parts are masked invalid."""
    torch.manual_seed(seed)
    B = len(valid_counts)
    Kmax = max(valid_counts)
    C = model.latent_channels
    R = model.resolution
    V = model.num_views
    T = 8  # cond tokens per view
    x = torch.randn(B, Kmax, C, R, R, R)
    z = torch.randn(B, C, R, R, R)
    cond = torch.randn(B, V * T, 64)
    mask = torch.zeros(B, V * T, dtype=torch.long)
    slots = torch.arange(1, Kmax + 1).view(1, Kmax).expand(B, Kmax).contiguous()
    valid = torch.zeros(B, Kmax, dtype=torch.bool)
    for b, kb in enumerate(valid_counts):
        valid[b, :kb] = True
        for k in range(kb):  # give each valid slot 2D mask coverage
            mask[b, 2 * k : 2 * k + 2] = k + 1
    t = torch.rand(B)
    return x, t, z, cond, mask, valid, slots


def _serial_vs_flat(model, inputs, **fwd_kw):
    x, t, z, cond, mask, valid, slots = inputs
    with torch.no_grad():
        model.joint_flat_batch = False
        out_serial = model(x, t, z, cond, mask, valid, slots, **fwd_kw)
        model.joint_flat_batch = True
        out_flat = model(x, t, z, cond, mask, valid, slots, **fwd_kw)
    return out_serial, out_flat


# ----------------------------------------------------------------------
# Equivalence gate (the contract)
# ----------------------------------------------------------------------
def test_flat_matches_serial_summary_on():
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        joint_flat_batch=True,
    )
    inputs = _multi_inputs(model, valid_counts=[2, 4, 3])
    out_serial, out_flat = _serial_vs_flat(model, inputs)
    diff = (out_serial - out_flat).abs().max().item()
    assert diff < 1e-4, f"flat path must match serial; max abs diff={diff}"


def test_flat_matches_serial_summary_off():
    """summary_cross_part_attention=False uses the per-object [1,K*T,C] reshape
    global attention; flat must group that per object too."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=False,
        joint_flat_batch=True,
    )
    inputs = _multi_inputs(model, valid_counts=[2, 4, 3])
    out_serial, out_flat = _serial_vs_flat(model, inputs)
    diff = (out_serial - out_flat).abs().max().item()
    assert diff < 1e-4, f"flat path must match serial (summary off); max abs diff={diff}"


def test_flat_matches_serial_with_self_cond_and_cfg():
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        self_conditioning=True,
        classifier_free_guidance=True,
        joint_flat_batch=True,
    )
    inputs = _multi_inputs(model, valid_counts=[3, 1, 4])
    x = inputs[0]
    sc = torch.randn_like(x)
    out_serial, out_flat = _serial_vs_flat(
        model, inputs, x_self_cond=sc, drop_part_cond=True
    )
    diff = (out_serial - out_flat).abs().max().item()
    assert diff < 1e-4, f"flat must match serial (self-cond+cfg); max abs diff={diff}"


def test_flat_matches_serial_under_checkpoint_grad():
    """With use_checkpoint and grad enabled the flat path checkpoints per block
    over the whole flat batch; it must still match the serial reference."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        use_checkpoint=True,
        joint_flat_batch=True,
    )
    model.train()
    x, t, z, cond, mask, valid, slots = _multi_inputs(model, valid_counts=[2, 3, 4])
    model.joint_flat_batch = False
    out_serial = model(x, t, z, cond, mask, valid, slots)
    model.joint_flat_batch = True
    out_flat = model(x, t, z, cond, mask, valid, slots)
    diff = (out_serial - out_flat).abs().max().item()
    assert diff < 1e-4, f"flat must match serial under checkpoint+grad; max abs diff={diff}"


# ----------------------------------------------------------------------
# Behavioural gates (defense in depth)
# ----------------------------------------------------------------------
def test_flat_isolates_objects():
    """Flat path: perturbing object 0's parts must NOT change object 1/2 output
    (no cross-object attention leakage)."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        joint_flat_batch=True,
    )
    x, t, z, cond, mask, valid, slots = _multi_inputs(model, valid_counts=[2, 3, 2])
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[0] += 5.0  # perturb ALL of object 0
        pert = model(x2, t, z, cond, mask, valid, slots)
    assert (pert[0] - base[0]).abs().max().item() > 1e-4  # object 0 changes
    assert torch.equal(pert[1], base[1]), "object 1 must be isolated from object 0"
    assert torch.equal(pert[2], base[2]), "object 2 must be isolated from object 0"


def test_flat_couples_within_object():
    """Flat path: perturbing part 0 of object 1 must change part 1 of object 1
    (within-object binding preserved)."""
    model = _tiny_model(
        cross_part_attention=True,
        token_identity_embedding=True,
        summary_cross_part_attention=True,
        n_summary_tokens=8,
        joint_flat_batch=True,
    )
    x, t, z, cond, mask, valid, slots = _multi_inputs(model, valid_counts=[2, 3, 2])
    with torch.no_grad():
        base = model(x, t, z, cond, mask, valid, slots)
        x2 = x.clone()
        x2[1, 0] += 5.0  # perturb only part 0 of object 1
        pert = model(x2, t, z, cond, mask, valid, slots)
    assert (pert[1, 1] - base[1, 1]).abs().max().item() > 1e-4, "within-object coupling lost"
    assert torch.equal(pert[0], base[0]), "object 0 must stay isolated"


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_flat_batch_requires_cross_part_attention():
    with pytest.raises(ValueError, match="joint_flat_batch"):
        _tiny_model(cross_part_attention=False, joint_flat_batch=True)
