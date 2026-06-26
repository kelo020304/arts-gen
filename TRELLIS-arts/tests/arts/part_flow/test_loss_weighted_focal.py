"""D-19/D-20: weighted focal endpoint CE."""

import torch
import torch.nn.functional as F

from trellis.trainers.arts.part_flow_losses import _class_weights, weighted_focal_endpoint_ce


def test_gamma0_uniform_matches_cross_entropy():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 2])
    valid = torch.ones(2, 3, dtype=torch.bool)
    weights = torch.ones(3)
    got = weighted_focal_endpoint_ce(logits, labels, valid, weights, gamma=0.0)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(got, expected)


def test_empty_weight_reduces_loss_on_empty_dominated_batch():
    logits = torch.zeros(4, 3)
    labels = torch.zeros(4, dtype=torch.long)
    valid = torch.ones(4, 3, dtype=torch.bool)
    w_full = _class_weights(3, empty_weight=1.0, part_weight=1.0, device=torch.device('cpu'))
    w_low = _class_weights(3, empty_weight=0.05, part_weight=1.0, device=torch.device('cpu'))
    full = weighted_focal_endpoint_ce(logits, labels, valid, w_full, gamma=0.0)
    low = weighted_focal_endpoint_ce(logits, labels, valid, w_low, gamma=0.0)
    assert low < full


def test_focal_downweights_easy_examples():
    valid = torch.ones(1, 3, dtype=torch.bool)
    weights = torch.ones(3)
    labels = torch.tensor([1])
    easy = torch.tensor([[-5.0, 5.0, -5.0]])
    uniform = torch.zeros(1, 3)
    easy_loss = weighted_focal_endpoint_ce(easy, labels, valid, weights, gamma=2.0)
    uniform_loss = weighted_focal_endpoint_ce(uniform, labels, valid, weights, gamma=2.0)
    assert easy_loss < uniform_loss


def test_ignore_index_excludes_rows():
    logits = torch.tensor([[1.0, 0.0], [-100.0, 100.0]])
    labels = torch.tensor([0, -1])
    valid = torch.ones(2, 2, dtype=torch.bool)
    weights = torch.ones(2)
    got = weighted_focal_endpoint_ce(logits, labels, valid, weights, gamma=0.0, ignore_index=-1)
    expected = F.cross_entropy(logits[:1], labels[:1])
    assert torch.allclose(got, expected)


def test_class_balanced_reduction_averages_present_classes_not_voxels():
    logits = torch.tensor([
        [4.0, 0.0],
        [4.0, 0.0],
        [4.0, 0.0],
        [4.0, 0.0],
        [4.0, 0.0],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1])
    valid = torch.ones(5, 2, dtype=torch.bool)
    weights = torch.ones(2)

    got = weighted_focal_endpoint_ce(
        logits, labels, valid, weights, gamma=0.0,
        reduction='class_balanced',
    )
    per_row = F.cross_entropy(logits, labels, reduction='none')
    expected = 0.5 * (per_row[labels == 0].mean() + per_row[labels == 1].mean())

    assert torch.allclose(got, expected)
