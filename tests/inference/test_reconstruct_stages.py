from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.inference.reconstruct import CkptConfig
from scripts.inference.reconstruct_stages import (
    STAGES,
    _signature,
    _save_token_visualizations,
    _stage_config,
    invalidate_downstream,
    pipeline_status,
    run_stage,
    stage_status,
)


def test_pipeline_status_defaults_and_reads_progress(tmp_path: Path) -> None:
    assert list(pipeline_status(tmp_path)) == list(STAGES)
    assert stage_status(tmp_path, "ss_decode")["state"] == "not_started"

    root = tmp_path / "ss_decode"
    root.mkdir()
    (root / "status.json").write_text(
        json.dumps({"state": "running", "progress": 42, "message": "decoding"}),
        encoding="utf-8",
    )
    status = stage_status(tmp_path, "ss_decode")
    assert status["progress"] == 42
    assert status["message"] == "decoding"


def test_invalidate_downstream_preserves_stage_and_upstream(tmp_path: Path) -> None:
    for stage in STAGES:
        root = tmp_path / stage
        root.mkdir()
        (root / "artifact.bin").write_bytes(b"x")

    removed = invalidate_downstream(tmp_path, "ss_decode")

    assert removed == ["part_prompt_seg", "slat_decode"]
    assert (tmp_path / "dino_ss_flow" / "artifact.bin").is_file()
    assert (tmp_path / "ss_decode" / "artifact.bin").is_file()
    assert not (tmp_path / "part_prompt_seg").exists()
    assert not (tmp_path / "slat_decode").exists()


def test_stage_signature_is_stable_and_sensitive_to_upstream() -> None:
    payload = {"stage": "ss_decode", "dependency_signature": "first", "config": {"threshold": 0}}
    assert _signature(payload) == _signature(dict(reversed(list(payload.items()))))
    assert _signature(payload) != _signature({**payload, "dependency_signature": "second"})


def test_downstream_rejects_inputs_that_do_not_match_dependency(tmp_path: Path) -> None:
    images = []
    masks = []
    for index in range(4):
        image = tmp_path / f"image_{index}.png"
        mask = tmp_path / f"mask_{index}.npy"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(image)
        np.save(mask, np.ones((4, 4), dtype=np.int32))
        images.append(image)
        masks.append(mask)
    dependency = tmp_path / "pipeline" / "dino_ss_flow"
    dependency.mkdir(parents=True)
    (dependency / "status.json").write_text(
        json.dumps({"state": "complete", "signature": "old", "input_signature": "old-input"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="inputs changed"):
        run_stage(
            "ss_decode",
            pipeline_root=tmp_path / "pipeline",
            images=images,
            masks=masks,
            part_info=None,
            ckpt_config=CkptConfig(),
        )

    failed = stage_status(tmp_path / "pipeline", "ss_decode")
    assert failed["state"] == "failed"
    assert failed["progress"] == 0


def test_token_visualization_uses_spatial_tokens_and_records_semantics(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    tokens = rng.normal(size=(2, 14, 8)).astype(np.float32)  # 5 special + 3x3 spatial

    metadata = _save_token_visualizations(tokens, tmp_path)["token_visualization"]

    assert metadata["special_tokens_excluded"] == 5
    assert metadata["spatial_grid"] == [3, 3]
    assert "not source RGB" in metadata["semantics"]
    assert (tmp_path / "token_pca.png").is_file()
    assert (tmp_path / "token_norm.png").is_file()


def test_part_stage_signature_fingerprints_checkpoint_content_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_100.pt"
    checkpoint.write_bytes(b"first")
    config = CkptConfig(part_seg_ckpt=checkpoint)

    first = _stage_config("part_prompt_seg", config)["part_seg_ckpt"]
    checkpoint.write_bytes(b"updated checkpoint")
    second = _stage_config("part_prompt_seg", config)["part_seg_ckpt"]

    assert first["path"] == str(checkpoint.resolve())
    assert first["size"] != second["size"]
    assert first != second
