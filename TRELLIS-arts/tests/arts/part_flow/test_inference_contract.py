"""Inference API must preserve the Part Flow conditioning contract."""

import importlib

import pytest
import torch


def _patch_part_flow_loader(monkeypatch, *, k_max=8):
    inference = importlib.import_module("inference")

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    class FakeBridge:
        pass

    FakeBridge.k_max = k_max

    monkeypatch.setattr(
        inference,
        "_load_part_flow",
        lambda _ckpt: (object(), FakeBridge(), 1),
    )

    return inference


def test_run_part_flow_enumerates_dense_grid_and_scatters_surface(monkeypatch):
    inference = _patch_part_flow_loader(monkeypatch)
    losses = importlib.import_module("trellis.trainers.arts.part_flow_losses")

    captured = {}

    def fake_flow_sample(
        model,
        bridge,
        coords,
        cond,
        mask_token_labels,
        voxel_layout,
        num_parts,
        is_on_surface,
        num_steps,
        solver,
    ):
        captured["coords"] = coords.detach().clone()
        captured["mask_token_labels"] = mask_token_labels.detach().clone()
        captured["num_parts"] = list(num_parts)
        captured["is_on_surface"] = is_on_surface.detach().clone()
        n = coords.shape[0]
        return torch.zeros(n, dtype=torch.long), torch.zeros(n, bridge.k_max)

    monkeypatch.setattr(losses, "flow_sample", fake_flow_sample)

    surface_coords = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)
    mask_token_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    inference.run_part_flow(
        surface_coords,
        cond_tokens,
        "dummy.pt",
        mask_token_labels=mask_token_labels,
        num_parts=3,
        num_steps=1,
    )

    assert captured["coords"].shape == (64 ** 3, 4)
    assert captured["coords"][0].tolist() == [0, 0, 0, 0]
    assert captured["coords"][-1].tolist() == [0, 63, 63, 63]
    assert captured["mask_token_labels"].shape == (1, 4)
    assert torch.equal(captured["mask_token_labels"][0], mask_token_labels)
    assert captured["num_parts"] == [3]
    surface = captured["is_on_surface"].reshape(64, 64, 64)
    assert int(surface.sum().item()) == 2
    assert surface[1, 2, 3].item() == 1
    assert surface[4, 5, 6].item() == 1


def test_run_part_flow_requires_mask_token_labels():
    inference = importlib.import_module("inference")

    coords = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)

    with pytest.raises(TypeError, match="mask_token_labels"):
        inference.run_part_flow(
            coords,
            cond_tokens,
            "dummy.pt",
            num_parts=3,
            num_steps=1,
        )


@pytest.mark.parametrize("num_parts", [1, 9])
def test_run_part_flow_rejects_num_parts_outside_valid_range(monkeypatch, num_parts):
    inference = _patch_part_flow_loader(monkeypatch, k_max=8)

    coords = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)
    mask_token_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="outside valid range"):
        inference.run_part_flow(
            coords,
            cond_tokens,
            "dummy.pt",
            mask_token_labels=mask_token_labels,
            num_parts=num_parts,
            num_steps=1,
        )


def test_run_part_flow_rejects_mask_token_labels_outside_num_parts(monkeypatch):
    inference = _patch_part_flow_loader(monkeypatch)

    coords = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)
    mask_token_labels = torch.tensor([0, 1, 3, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="values must be in"):
        inference.run_part_flow(
            coords,
            cond_tokens,
            "dummy.pt",
            mask_token_labels=mask_token_labels,
            num_parts=3,
            num_steps=1,
        )


def test_run_part_flow_rejects_mask_token_labels_length_mismatch(monkeypatch):
    inference = _patch_part_flow_loader(monkeypatch)

    coords = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)
    mask_token_labels = torch.tensor([0, 1, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="does not match cond token count"):
        inference.run_part_flow(
            coords,
            cond_tokens,
            "dummy.pt",
            mask_token_labels=mask_token_labels,
            num_parts=3,
            num_steps=1,
        )
