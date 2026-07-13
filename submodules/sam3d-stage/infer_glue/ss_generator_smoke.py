#!/usr/bin/env python
"""Smoke-test SAM3D SS generator and optional SS decoder.

This deliberately does NOT instantiate ``pipeline.yaml``'s full
``InferencePipelinePointMap`` target, so it does not require pytorch3d, MoGe, or
Blender. It only proves that the SAM3D sparse-structure generator checkpoint can
be loaded and sampled in the current Python environment. With ``--decode`` it
also proves the standalone SS decoder can run on the sampled shape latent.

The condition tokens used here are synthetic, so this is an environment/weights
smoke test rather than useful image-conditioned generation.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path


DEFAULT_WEIGHTS_DIR = Path("/robot/data-lab/jzh/art-gen/weights")


def _configure_runtime(attention_backend: str) -> None:
    # Must be set before importing sam3d_objects.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ["ATTN_BACKEND"] = attention_backend
    os.environ["SPARSE_ATTN_BACKEND"] = attention_backend


def _load_prefixed_state_dict(ckpt_path: Path, prefix: str):
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    n = len(prefix)
    filtered = {
        key[n:]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not filtered:
        raise KeyError(f"checkpoint has no weights with prefix {prefix!r}: {ckpt_path}")
    return filtered


def _build_generator(config_path: Path, ckpt_path: Path, device: str):
    import hydra
    import torch
    from omegaconf import OmegaConf

    import sam3d_objects  # noqa: F401 -- package init side-effects, init skipped

    config = OmegaConf.load(str(config_path))["module"]["generator"]["backbone"]
    generator = hydra.utils.instantiate(config)
    state_dict = _load_prefixed_state_dict(ckpt_path, "_base_models.generator.")
    missing, unexpected = generator.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"strict generator load failed: missing={missing}, unexpected={unexpected}"
        )
    return generator.to(torch.device(device)).eval(), len(state_dict)


def _build_decoder(config_path: Path, ckpt_path: Path, device: str):
    import hydra
    import torch
    from omegaconf import OmegaConf

    import sam3d_objects  # noqa: F401 -- package init side-effects, init skipped

    decoder = hydra.utils.instantiate(OmegaConf.load(str(config_path)))
    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
    return decoder.to(torch.device(device)).eval(), len(missing), len(unexpected)


def _latent_shape_dict(generator, batch_size: int) -> dict[str, tuple[int, ...]]:
    latent_mapping = generator.reverse_fn.backbone.latent_mapping
    return {
        key: (batch_size,) + (latent.pos_emb.shape[0], latent.input_layer.in_features)
        for key, latent in latent_mapping.items()
    }


def run_smoke(
    *,
    weights_dir: Path,
    generator_config: Path,
    generator_ckpt: Path,
    decoder_config: Path,
    decoder_ckpt: Path,
    device: str,
    inference_steps: int,
    condition: str,
    condition_tokens: int,
    seed: int,
    decode: bool,
    require_nonempty_voxel: bool,
) -> dict:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator, generator_state_keys = _build_generator(
        generator_config, generator_ckpt, device
    )
    generator.no_shortcut = True
    generator.inference_steps = inference_steps
    generator.reverse_fn.strength = 0.0
    generator.reverse_fn.interval = [0, 500]
    generator.reverse_fn.unconditional_handling = "add_flag"

    device_t = torch.device(device)
    batch_size = 1
    cond_channels = int(getattr(generator.reverse_fn.backbone, "cond_channels", 1024))
    if condition == "zero":
        condition_tensor = torch.zeros(
            (batch_size, condition_tokens, cond_channels), device=device_t
        )
    elif condition == "random":
        condition_tensor = torch.randn(
            (batch_size, condition_tokens, cond_channels), device=device_t
        )
    else:
        raise ValueError(f"unknown condition mode: {condition}")

    latent_shapes = _latent_shape_dict(generator, batch_size)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device_t.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        generated = generator(latent_shapes, device_t, condition_tensor)

    result = {
        "weights_dir": str(weights_dir),
        "device": str(device_t),
        "generator_state_keys": generator_state_keys,
        "generator_outputs": {
            key: tuple(value.shape) for key, value in generated.items()
        },
    }

    if decode:
        decoder, missing, unexpected = _build_decoder(decoder_config, decoder_ckpt, device)
        shape_latent = generated["shape"].float()
        zt = (
            shape_latent.permute(0, 2, 1)
            .contiguous()
            .view(shape_latent.shape[0], 8, 16, 16, 16)
        )
        with torch.no_grad():
            voxel = decoder(zt)
        coords = torch.argwhere(voxel > 0)[:, [0, 2, 3, 4]].int()
        if require_nonempty_voxel and int(coords.shape[0]) == 0:
            raise RuntimeError(
                "decoder ran but produced 0 voxels from the synthetic condition"
            )
        result.update(
            {
                "decoder_missing": missing,
                "decoder_unexpected": unexpected,
                "decoder_output": tuple(voxel.shape),
                "decoder_num_voxels": int(coords.shape[0]),
            }
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ss-generator-smoke",
        description=(
            "Load and run SAM3D ss_generator, optionally followed by ss_decoder, "
            "without instantiating the full pointmap pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--ss-generator-config", type=Path, default=None)
    parser.add_argument("--ss-generator-ckpt", type=Path, default=None)
    parser.add_argument("--ss-decoder-config", type=Path, default=None)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=1)
    parser.add_argument("--condition", choices=["zero", "random"], default="zero")
    parser.add_argument("--condition-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument(
        "--require-nonempty-voxel",
        action="store_true",
        help="Fail if --decode produces 0 voxels. Usually leave this off for synthetic conditions.",
    )
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        help="Pinned for minimal compiled dependencies; applied to ATTN_BACKEND and SPARSE_ATTN_BACKEND.",
    )
    args = parser.parse_args()

    _configure_runtime(args.attention_backend)

    weights_dir = args.weights_dir
    generator_config = args.ss_generator_config or weights_dir / "ss_generator.yaml"
    generator_ckpt = args.ss_generator_ckpt or weights_dir / "ss_generator.ckpt"
    decoder_config = args.ss_decoder_config or weights_dir / "ss_decoder.yaml"
    decoder_ckpt = args.ss_decoder_ckpt or weights_dir / "ss_decoder.ckpt"

    for path in (generator_config, generator_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.decode:
        for path in (decoder_config, decoder_ckpt):
            if not path.is_file():
                raise FileNotFoundError(path)

    result = run_smoke(
        weights_dir=weights_dir,
        generator_config=generator_config,
        generator_ckpt=generator_ckpt,
        decoder_config=decoder_config,
        decoder_ckpt=decoder_ckpt,
        device=args.device,
        inference_steps=args.inference_steps,
        condition=args.condition,
        condition_tokens=args.condition_tokens,
        seed=args.seed,
        decode=args.decode,
        require_nonempty_voxel=args.require_nonempty_voxel,
    )
    print("[INFO] ss_generator loaded strict=True")
    print(f"[INFO] generator_state_keys={result['generator_state_keys']}")
    for key, shape in result["generator_outputs"].items():
        print(f"[INFO] output {key}: {shape}")
    if args.decode:
        print(
            "[INFO] ss_decoder loaded strict=False "
            f"missing={result['decoder_missing']} unexpected={result['decoder_unexpected']}"
        )
        print(f"[INFO] decoder_output={result['decoder_output']}")
        print(f"[INFO] decoder_num_voxels={result['decoder_num_voxels']}")
    print("[INFO] DONE")


if __name__ == "__main__":
    main()
