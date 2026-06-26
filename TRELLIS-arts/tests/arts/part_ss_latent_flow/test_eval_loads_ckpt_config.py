"""Eval must load a checkpoint using the config EMBEDDED in that checkpoint.

The eval platform lets the user pick ANY checkpoint from a dropdown, so the
checkpoint's architecture (e.g. trained WITHOUT summary-token) routinely differs
from the current default YAML (summary-token ON). The model must be built from
``ckpt['config']['model']`` (the trainer persists it), not the eval YAML, or
``load_state_dict(strict=True)`` fails on the missing summary_* keys. The latent
(de)normalization stats must likewise come from the ckpt so sampling matches
training (per_channel needs the persisted mean/std).
"""

import torch
from omegaconf import OmegaConf

from eval_part_ss_latent_flow import _apply_ckpt_latent_norm, _load_model
from trellis.models.part_flow.part_ss_latent_flow import PartSSLatentFlowModel


_TRAIN_MODEL_CFG = dict(
    resolution=16,
    latent_channels=4,
    model_channels=32,
    cond_dim=64,
    num_blocks=4,
    num_heads=4,
    patch_size=1,
    num_views=2,
    max_parts=8,
    num_part_query_layers=1,
    part_label_vocab_size=16,
    require_part_token=True,
    use_fp16=False,
    use_checkpoint=False,
    cross_part_attention=True,
    token_identity_embedding=True,
    summary_cross_part_attention=False,  # <- trained WITHOUT summary-token
)


def _save_ckpt(path, *, with_config: bool):
    trained = PartSSLatentFlowModel(**_TRAIN_MODEL_CFG)
    payload = {"step": 5, "model": trained.state_dict()}
    if with_config:
        payload["config"] = {
            "model": dict(_TRAIN_MODEL_CFG),
            "flow": {
                "latent_norm_mode": "per_channel",
                "latent_mean": [0.0, 0.1, 0.2, 0.3],
                "latent_std": [1.0, 1.1, 1.2, 1.3],
                "latent_scale": 8.0,
            },
        }
    torch.save(payload, path)


def test_load_model_builds_from_embedded_ckpt_config(tmp_path):
    ckpt = tmp_path / "step_5.pt"
    _save_ckpt(ckpt, with_config=True)
    # Eval YAML asks for summary-token ON — a mismatch the loader must ignore in
    # favour of the ckpt's own architecture.
    eval_cfg = OmegaConf.create(
        {"model": {**_TRAIN_MODEL_CFG, "summary_cross_part_attention": True, "n_summary_tokens": 8}}
    )
    model, step, ckpt_cfg = _load_model(eval_cfg, ckpt, torch.device("cpu"))
    assert step == 5
    assert model.summary_cross_part_attention is False  # built from ckpt cfg, not eval cfg
    assert ckpt_cfg["flow"]["latent_mean"] == [0.0, 0.1, 0.2, 0.3]


def test_load_model_falls_back_to_passed_config_for_legacy_ckpt(tmp_path):
    """A checkpoint without an embedded config (legacy) still loads from the YAML."""
    ckpt = tmp_path / "legacy.pt"
    _save_ckpt(ckpt, with_config=False)
    eval_cfg = OmegaConf.create({"model": dict(_TRAIN_MODEL_CFG)})
    model, step, ckpt_cfg = _load_model(eval_cfg, ckpt, torch.device("cpu"))
    assert step == 5
    assert ckpt_cfg is None
    assert model.summary_cross_part_attention is False


def test_apply_ckpt_latent_norm_fills_stats(tmp_path):
    flow_cfg = {"latent_norm_mode": "per_channel", "latent_mean": None, "latent_std": None, "cfg_scale": 2.0}
    ckpt_cfg = {
        "flow": {
            "latent_norm_mode": "per_channel",
            "latent_mean": [0.0, 0.1, 0.2, 0.3],
            "latent_std": [1.0, 1.1, 1.2, 1.3],
            "latent_scale": 8.0,
        }
    }
    _apply_ckpt_latent_norm(flow_cfg, ckpt_cfg)
    assert flow_cfg["latent_mean"] == [0.0, 0.1, 0.2, 0.3]
    assert flow_cfg["latent_std"] == [1.0, 1.1, 1.2, 1.3]
    assert flow_cfg["latent_scale"] == 8.0
    assert flow_cfg["cfg_scale"] == 2.0  # eval-time guidance NOT overwritten by ckpt


def test_apply_ckpt_latent_norm_noop_without_config():
    flow_cfg = {"latent_norm_mode": "scalar"}
    _apply_ckpt_latent_norm(flow_cfg, None)
    assert flow_cfg == {"latent_norm_mode": "scalar"}
