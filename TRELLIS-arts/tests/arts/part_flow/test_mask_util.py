"""D-22: patch_aggregate_foreground_wins spec tests."""

import torch

from trellis.utils.arts.mask_utils import patch_aggregate_foreground_wins


def test_output_shape():
    out = patch_aggregate_foreground_wins(torch.zeros(512, 512, dtype=torch.long))
    assert out.shape == (37, 37)


def test_all_background_stays_zero():
    out = patch_aggregate_foreground_wins(torch.zeros(512, 512, dtype=torch.long))
    assert out.sum().item() == 0


def test_min_fg_threshold_at_2_drops_label():
    mask = torch.zeros(512, 512, dtype=torch.long)
    mask[0, 0:2] = 5
    out = patch_aggregate_foreground_wins(mask, min_fg=3)
    assert out[0, 0].item() == 0


def test_min_fg_threshold_at_3_passes_label():
    mask = torch.zeros(512, 512, dtype=torch.long)
    mask[0, 0:3] = 5
    out = patch_aggregate_foreground_wins(mask, min_fg=3)
    assert out[0, 0].item() == 5


def test_majority_label_wins():
    mask = torch.zeros(512, 512, dtype=torch.long)
    mask[0, 0:4] = 2
    mask[1, 0:5] = 3
    out = patch_aggregate_foreground_wins(mask, min_fg=3)
    assert out[0, 0].item() == 3


def test_boundary_patch_padded_to_518():
    mask = torch.zeros(512, 512, dtype=torch.long)
    mask[510:512, 510:512] = 7
    out = patch_aggregate_foreground_wins(mask, min_fg=3)
    assert out[36, 36].item() == 7
