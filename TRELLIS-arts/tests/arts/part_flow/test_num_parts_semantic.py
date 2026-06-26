"""D-11: num_parts[b] means K_b+1 including empty slot."""

from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor


def test_build_part_valid_mask_slot_0_always_true():
    model = PartFlowPredictor(k_max=8)
    valid = model.build_part_valid_mask([1, 3, 5], device='cpu')
    assert valid[:, 0].all().item()
    assert valid[0, 1:].sum().item() == 0
    assert valid[1, :3].all().item() and not valid[1, 3:].any().item()
    assert valid[2, :5].all().item() and not valid[2, 5:].any().item()


def test_num_parts_minimum_is_1():
    model = PartFlowPredictor(k_max=4)
    valid = model.build_part_valid_mask([1], device='cpu')
    assert valid.tolist() == [[True, False, False, False]]


def test_kmax_unchanged_at_128():
    model = PartFlowPredictor()
    assert model.k_max == 128
