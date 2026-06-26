import inspect

import pytest
import torch


def test_run_part_ss_latent_flow_requires_real_mask_labels():
    import inference

    with pytest.raises(ValueError, match="mask_token_labels"):
        inference.run_part_ss_latent_flow(
            torch.randn(8, 16, 16, 16),
            torch.randn(4 * 1370, 1024),
            ckpt_path="missing.pt",
            target_slots=[1],
            mask_token_labels=None,
            target_part_names=["wheel_0"],
            ss_decoder_ckpt="missing.safetensors",
        )


def test_run_slat_flow_from_tokens_rejects_empty_coords():
    import inference

    with pytest.raises(ValueError, match="coords must contain at least one voxel"):
        inference.run_slat_flow_from_tokens(
            torch.randn(1370, 1024),
            torch.zeros(0, 3, dtype=torch.long),
            ckpt_path="missing.safetensors",
        )


def test_slat_initial_feats_seed_is_local_and_repeatable():
    import inference

    torch.manual_seed(123)
    before = torch.randn(1)
    seeded_a = inference._make_slat_initial_feats(
        2,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=99,
    )
    after = torch.randn(1)

    torch.manual_seed(123)
    before_again = torch.randn(1)
    seeded_b = inference._make_slat_initial_feats(
        2,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=99,
    )
    after_again = torch.randn(1)
    seeded_c = inference._make_slat_initial_feats(
        2,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=100,
    )

    assert torch.equal(before, before_again)
    assert torch.equal(after, after_again)
    assert torch.equal(seeded_a, seeded_b)
    assert not torch.equal(seeded_a, seeded_c)


def test_decode_slat_paths_share_sparse_tensor_warmup_helper():
    import inference

    assert hasattr(inference, "_ensure_sparse_tensor_init")
    decode_slat_source = inspect.getsource(inference.decode_slat)
    decode_assets_source = inspect.getsource(inference.decode_slat_assets)

    assert "_ensure_sparse_tensor_init()" in decode_slat_source
    assert "_ensure_sparse_tensor_init()" in decode_assets_source
    assert "SparseTensor(coords=_dummy_coords" not in decode_slat_source
    assert "SparseTensor(coords=_dummy_coords" not in decode_assets_source


def test_run_part_ss_latent_flow_samples_all_parts_jointly_once(monkeypatch):
    import inference
    import trellis.trainers.arts.part_ss_latent_flow_losses as losses

    monkeypatch.setattr(
        inference,
        "_load_part_ss_latent_flow",
        lambda ckpt_path: (object(), {"flow": {"num_steps": 3, "noise_scale": 0.0, "latent_scale": 8.0}}),
    )
    monkeypatch.setattr(inference, "_load_ss_decoder", lambda ckpt_path: object())
    monkeypatch.setattr(
        inference,
        "decode_ss",
        lambda latent, ckpt_path, threshold=0.0: torch.zeros(1, 3, dtype=torch.long),
    )
    calls = []

    def fake_sample(
        model,
        *,
        z_global,
        cond,
        mask_token_labels,
        part_valid,
        target_slots,
        num_steps,
        noise_scale,
        latent_scale,
        part_token_weights=None,
        **_sampler_kwargs,  # latent_norm_mode / latent_mean / ... from build_part_ss_sampler_kwargs
    ):
        calls.append({
            "part_valid": part_valid.detach().cpu().clone(),
            "target_slots": target_slots.detach().cpu().clone(),
            "num_steps": num_steps,
            "latent_scale": latent_scale,
            "part_token_weights": None if part_token_weights is None else part_token_weights.detach().cpu().clone(),
        })
        return torch.zeros(1, 2, 8, 16, 16, 16, device=z_global.device)

    monkeypatch.setattr(losses, "sample_part_ss_latent", fake_sample)
    labels = torch.zeros(4 * 1370, dtype=torch.long)
    labels[10:20] = 1
    labels[30:40] = 2
    part_token_weights = torch.zeros(2, 4 * 1370, dtype=torch.float32)
    part_token_weights[0, 10:20] = 0.1
    part_token_weights[1, 30:40] = 0.1
    result = inference.run_part_ss_latent_flow(
        torch.randn(8, 16, 16, 16),
        torch.randn(4 * 1370, 1024),
        ckpt_path="fake.pt",
        target_slots=[1, 2],
        mask_token_labels=labels,
        target_part_names=["wheel_0", "wheel_1"],
        part_token_weights=part_token_weights,
        ss_decoder_ckpt="fake.safetensors",
    )
    assert len(calls) == 1
    assert calls[0]["part_valid"].tolist() == [[True, True]]
    assert calls[0]["target_slots"].tolist() == [[1, 2]]
    assert calls[0]["num_steps"] == 3
    assert calls[0]["latent_scale"] == 8.0
    assert calls[0]["part_token_weights"].shape == (1, 2, 4 * 1370)
    assert torch.equal(calls[0]["part_token_weights"][0], part_token_weights)
    assert set(result["part_latents"]) == {"wheel_0", "wheel_1"}
