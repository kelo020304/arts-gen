import torch

from trellis.models.part_flow.bridges import build_bridge
from trellis.trainers.arts.part_flow_losses import flow_sample


class CountingModel(torch.nn.Module):
    def __init__(self, k_max):
        super().__init__()
        self.k_max = k_max
        self.voxel_chunk_size = 4
        self.calls = []

    def forward(self, x_t, t, coords, cond, mask_token_labels, num_parts, is_on_surface):
        self.calls.append(coords.shape[0])
        logits = torch.zeros(coords.shape[0], self.k_max, device=coords.device)
        logits[:, 0] = 1.0
        return {'endpoint_logits': logits}


def test_flow_sample_chunks_eval_model_forward():
    bridge = build_bridge('fisher', k_max=3)
    model = CountingModel(k_max=3)
    coords = torch.cat([
        torch.zeros(10, 1, dtype=torch.long),
        torch.randint(0, 64, (10, 3), dtype=torch.long),
    ], dim=1)
    labels, soft = flow_sample(
        model,
        bridge,
        coords=coords,
        cond=torch.randn(1, 8, 16),
        mask_token_labels=torch.zeros(1, 8, dtype=torch.long),
        voxel_layout=[slice(0, 10)],
        num_parts=[3],
        is_on_surface=torch.zeros(10, dtype=torch.long),
        num_steps=1,
        solver='euler',
    )
    assert model.calls == [4, 4, 2]
    assert labels.shape == (10,)
    assert soft.shape == (10, 3)
