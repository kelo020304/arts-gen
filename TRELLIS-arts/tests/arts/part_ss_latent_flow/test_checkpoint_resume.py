from pathlib import Path

import pytest
import torch


def _make_stepped_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_resolve_resume_checkpoint_prefers_ckpts_subdir(tmp_path):
    from trellis.trainers.arts import part_ss_latent_flow as trainer

    root = tmp_path / "run"
    (root / "ckpts").mkdir(parents=True)
    preferred = root / "ckpts" / "step_50000.pt"
    legacy = root / "step_50000.pt"
    preferred.write_bytes(b"preferred")
    legacy.write_bytes(b"legacy")

    assert trainer._resolve_resume_checkpoint(root, 50000) == preferred


def test_load_resume_checkpoint_restores_training_state(tmp_path):
    from trellis.trainers.arts import part_ss_latent_flow as trainer

    source_model, source_optimizer, source_scheduler = _make_stepped_state()
    ckpt_dir = tmp_path / "run" / "ckpts"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {
            "step": 7,
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "config": {},
        },
        ckpt_dir / "step_7.pt",
    )

    target_model = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9.0e-4)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lr_lambda=lambda step: 1.0)

    start_step = trainer._load_resume_checkpoint(
        target_model,
        target_optimizer,
        target_scheduler,
        {"load_dir": str(ckpt_dir.parent), "resume_step": 7},
        torch.device("cpu"),
        rank=1,
    )

    assert start_step == 7
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        torch.testing.assert_close(source_param, target_param)
    assert target_optimizer.state_dict()["state"]
    assert target_scheduler.state_dict()["last_epoch"] == source_scheduler.state_dict()["last_epoch"]


def test_load_resume_checkpoint_weights_only_resets_optimizer_and_scheduler(tmp_path):
    from trellis.trainers.arts import part_ss_latent_flow as trainer

    source_model, source_optimizer, source_scheduler = _make_stepped_state()
    ckpt_dir = tmp_path / "run" / "ckpts"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {
            "step": 7,
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "scheduler": source_scheduler.state_dict(),
            "config": {},
        },
        ckpt_dir / "step_7.pt",
    )

    target_model = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9.0e-4)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(target_optimizer, lr_lambda=lambda step: 1.0)

    start_step = trainer._load_resume_checkpoint(
        target_model,
        target_optimizer,
        target_scheduler,
        {"load_dir": str(ckpt_dir.parent), "resume_step": 7, "resume_weights_only": True},
        torch.device("cpu"),
        rank=0,
    )

    assert start_step == 0
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        torch.testing.assert_close(source_param, target_param)
    assert target_optimizer.state_dict()["state"] == {}
    assert target_scheduler.state_dict()["last_epoch"] == 0


def test_load_resume_checkpoint_missing_file_fails_loudly(tmp_path):
    from trellis.trainers.arts import part_ss_latent_flow as trainer

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        trainer._load_resume_checkpoint(
            model,
            optimizer,
            scheduler,
            {"load_dir": str(tmp_path / "run"), "resume_step": 9},
            torch.device("cpu"),
            rank=0,
        )
