"""End-to-end variable-K integration test.

Builds a synthetic batch with mixed K_b per sample, runs:
  - Loss forward+backward for each active flow family (gumbel, fisher)
  - flow_sample (ODE integration) for each
  - Verifies padding invariants hold throughout
  - Verifies predicted labels never exceed K_b (no padding leak)

No real data / GPU required; runs entirely on CPU.

Legacy dirichlet/sfm are not covered here — they're no longer active.
"""

from __future__ import annotations

import torch

def _setup():
    """Trainer code now lives inside the trellis package — direct imports work
    via the parent conftest's sys.path injection. Force the sdpa_loop backend
    so model tests stay CPU-safe."""
    from trellis.models.part_flow import attention_utils as au
    au._get_attn_backend = lambda: 'sdpa_loop'
    from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor
    from trellis.models.part_flow.bridges import build_bridge
    from trellis.trainers.arts.part_flow_losses import FlowMatchingLoss, flow_sample
    return PartFlowPredictor, FlowMatchingLoss, flow_sample, build_bridge


def _build_synth_batch(k_max, num_parts, n_per, VT=200, cond_dim=64):
    """Build a synthetic batch with variable K per sample."""
    B = len(num_parts)
    N_total = sum(n_per)

    # Per-voxel labels in {0..K_b-1}
    per_voxel_labels = torch.zeros(N_total, dtype=torch.long)
    voxel_layout = []
    offset = 0
    for b, (K_b, n_b) in enumerate(zip(num_parts, n_per)):
        voxel_layout.append(slice(offset, offset + n_b))
        per_voxel_labels[offset:offset + n_b] = torch.randint(0, K_b, (n_b,))
        offset += n_b

    batch_idx = torch.cat([torch.full((n_per[b],), b, dtype=torch.long) for b in range(B)])
    coords = torch.cat([batch_idx.unsqueeze(-1), torch.randint(0, 64, (N_total, 3))], dim=-1).int()
    cond = torch.randn(B, VT, cond_dim)

    # mask_token_labels: bg by default; distribute real parts deterministically.
    # Phase 8 num_parts includes empty slot 0, so real mask labels are 1..K_b-1.
    mask_labels = torch.zeros(B, VT, dtype=torch.long)
    for b, K_b in enumerate(num_parts):
        # 10% of tokens are part-labeled; guarantee each real part has >=1 token
        for j in range(1, K_b):
            positions = torch.arange(j * 3, VT, K_b * 3 + 1)
            mask_labels[b, positions] = j
        for j in range(1, K_b):
            if (mask_labels[b] == j).sum() == 0:
                mask_labels[b, j - 1] = j

    return {
        'coords': coords,
        'cond': cond,
        'mask_token_labels': mask_labels,
        'per_voxel_labels': per_voxel_labels,
        'is_on_surface': (torch.arange(N_total) % 3 == 0).long(),
        'voxel_layout': voxel_layout,
        'num_parts': num_parts,
    }


def _run_one_bridge(bridge_name, build_bridge, PartFlowPredictor, FlowMatchingLoss, flow_sample):
    torch.manual_seed(0)
    k_max = 8
    num_parts = [2, 4, 7]
    n_per = [10, 15, 20]

    bridge_kwargs = {'k_max': k_max, 't_max': 1.0}
    if bridge_name == 'gumbel':
        bridge_kwargs.update(tau_max=10.0, decay_rate=3.0, noise_scale=2.0, tau_min=0.01)
    bridge = build_bridge(bridge_name, **bridge_kwargs)

    batch = _build_synth_batch(k_max, num_parts, n_per, VT=200, cond_dim=64)
    cond_dim = batch['cond'].shape[-1]

    model = PartFlowPredictor(
        k_max=k_max, hidden_dim=32, num_layers=2, num_heads=4,
        cond_dim=cond_dim, dropout=0.0,
    )
    # Perturb voxel_score_proj and part_score_proj so logits are non-trivial
    with torch.no_grad():
        model.voxel_score_proj.weight.add_(torch.randn_like(model.voxel_score_proj.weight) * 0.05)
        model.part_score_proj.weight.add_(torch.randn_like(model.part_score_proj.weight) * 0.05)

    criterion = FlowMatchingLoss(bridge)

    # Forward + backward
    loss, metrics = criterion(model, batch)
    assert torch.isfinite(loss), f'{bridge_name}: loss not finite'
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and p.grad.abs().max() > 0)
    total = sum(1 for _ in model.parameters())
    assert n_with_grad > 0.5 * total, \
        f'{bridge_name}: only {n_with_grad}/{total} params got grad'

    # Sampling (short ODE)
    model.eval()
    with torch.no_grad():
        labels, soft = flow_sample(
            model, bridge,
            coords=batch['coords'],
            cond=batch['cond'],
            mask_token_labels=batch['mask_token_labels'],
            voxel_layout=batch['voxel_layout'],
            num_parts=batch['num_parts'],
            is_on_surface=batch['is_on_surface'],
            num_steps=5,
            solver='euler',
        )
    # Each predicted label must be < K_b for its sample
    for sl, K_b in zip(batch['voxel_layout'], batch['num_parts']):
        max_label = labels[sl].max().item()
        assert max_label < K_b, \
            f'{bridge_name}: predicted label {max_label} >= K_b={K_b} (padding leak)'
        # Soft probs should have 0 on padding
        pad_mass = soft[sl, K_b:].abs().max().item()
        assert pad_mass < 1e-5, f'{bridge_name}: padding soft prob leak {pad_mass}'
    print(f'[PASS] {bridge_name}: loss={metrics["loss"]:.4f} '
          f'ep_acc={metrics["endpoint_acc"]:.3f} grad_frac={n_with_grad}/{total}')


def test_variable_k_e2e_all_bridges():
    """Active flow families only: gumbel (default) + fisher."""
    PartFlowPredictor, FlowMatchingLoss, flow_sample, build_bridge = _setup()
    for br_name in ('gumbel', 'fisher'):
        _run_one_bridge(br_name, build_bridge, PartFlowPredictor, FlowMatchingLoss, flow_sample)


if __name__ == '__main__':
    test_variable_k_e2e_all_bridges()
    print('Variable-K E2E test passed for gumbel + fisher.')
