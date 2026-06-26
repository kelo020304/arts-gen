#!/usr/bin/env python3
"""Verify the decode-aware OOM fix on a CUDA box before building the training image.

The decode-aware loss decodes selected parts through the (frozen) SS conv3d
decoder *with the autograd graph live* so the occupancy loss can backprop into the
flow model. Before the fix, every selected part's 64^3 decoder activation graph
stayed resident until a single combined backward, so high part counts OOM'd once
the loss turned on at decode_aware_start_step. The fix gradient-checkpoints each
per-chunk decoder call (trellis/trainers/arts/part_ss_latent_flow_losses.py:
_decode_logits), so peak decoder activation memory is bounded to ~one chunk
regardless of part count, with identical gradients.

This script exercises the REAL _decode_logits path and reports peak GPU memory.

Examples
--------
# Real SS decoder + production settings (20 parts, chunk_size=1):
python scripts/tools/verify_decode_aware_oom_fix.py \
    --ss-decoder-ckpt TRELLIS-arts/pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
    --parts 20 --chunk-size 1

# Quick sanity without the checkpoint (synthetic decoder, no ckpt needed):
python scripts/tools/verify_decode_aware_oom_fix.py --synthetic --parts 20 --chunk-size 1

# Show the pre-fix behavior for contrast (checkpoint disabled — expect a much
# higher peak, and likely OOM at high --parts on a small GPU):
python scripts/tools/verify_decode_aware_oom_fix.py --synthetic --parts 20 --baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import types

# Allocator config must be set before the first CUDA allocation (mirror train_arts.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRELLIS_PATH = os.path.join(REPO_ROOT, "TRELLIS-arts")
if TRELLIS_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_PATH)

# Minimal-deps trellis registration (same trick as train_arts.py) so importing the
# loss / decoder loader does not trigger trellis/__init__.py's heavy eager imports.
_pkg = types.ModuleType("trellis")
_pkg.__path__ = [os.path.join(TRELLIS_PATH, "trellis")]
_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _pkg)
for _sp in ("models", "modules", "trainers", "utils", "datasets"):
    _m = types.ModuleType(f"trellis.{_sp}")
    _m.__path__ = [os.path.join(TRELLIS_PATH, "trellis", _sp)]
    _m.__package__ = f"trellis.{_sp}"
    sys.modules.setdefault(f"trellis.{_sp}", _m)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import trellis.trainers.arts.part_ss_latent_flow_losses as losses_mod  # noqa: E402
from trellis.trainers.arts.part_ss_latent_flow_losses import PartSSLatentRFDecodeAwareLoss  # noqa: E402


class _GN32(nn.GroupNorm):
    """GroupNorm that upcasts to fp32, matching trellis GroupNorm32."""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1, self.c1 = _GN32(32, ch), nn.Conv3d(ch, ch, 3, padding=1)
        self.n2, self.c2 = _GN32(32, ch), nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class _Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c = nn.Conv3d(ci, co, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode="nearest"))


class SyntheticSSDecoder(nn.Module):
    """Structurally mirrors SparseStructureDecoder channels [512,128,32], 16->32->64."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Conv3d(8, 512, 3, padding=1)
        self.mid = nn.Sequential(_ResBlock(512), _ResBlock(512))
        self.b16 = nn.Sequential(_ResBlock(512), _ResBlock(512))
        self.up1 = _Up(512, 128)
        self.b32 = nn.Sequential(_ResBlock(128), _ResBlock(128))
        self.up2 = _Up(128, 32)
        self.b64 = nn.Sequential(_ResBlock(32), _ResBlock(32))
        self.out = nn.Conv3d(32, 1, 3, padding=1)

    def forward(self, x):
        h = self.mid(self.inp(x))
        h = self.up1(self.b16(h))
        h = self.up2(self.b32(h))
        h = self.b64(h)
        return self.out(h)  # [N,1,64,64,64]


def build_decoder(args, device):
    if args.synthetic or not args.ss_decoder_ckpt:
        dec = SyntheticSSDecoder().to(device).half().eval()
        for mod in dec.modules():
            if isinstance(mod, _GN32):
                mod.float()  # norms stay fp32, like the real decoder
        for p in dec.parameters():
            p.requires_grad_(False)
        print("[decoder] synthetic SparseStructureDecoder-shaped stub (channels [512,128,32])")
        return dec
    from trellis.trainers.arts.part_ss_latent_flow_eval import load_ss_decoder
    dec = load_ss_decoder(args.ss_decoder_ckpt)  # .cuda().eval(), params frozen
    print(f"[decoder] real SparseStructureDecoder from {args.ss_decoder_ckpt}")
    return dec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ss-decoder-ckpt", default="", help="path to ss_dec_conv3d_16l8_fp16.safetensors")
    ap.add_argument("--synthetic", action="store_true", help="use a structural stub decoder (no ckpt needed)")
    ap.add_argument("--parts", type=int, default=20, help="parts to decode (decode_aware_max_parts_per_object)")
    ap.add_argument("--chunk-size", type=int, default=1, help="decode_aware_decode_chunk_size")
    ap.add_argument("--baseline", action="store_true", help="disable gradient checkpointing (pre-fix behavior)")
    ap.add_argument("--max-peak-gb", type=float, default=8.0, help="PASS threshold for peak alloc")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda")
    print(f"[env] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"[gpu] {torch.cuda.get_device_name(0)}")

    if args.baseline:
        # Restore the pre-fix path: call the decoder directly (no checkpoint).
        losses_mod.checkpoint = lambda fn, *a, **kw: fn(*a)
        print("[mode] BASELINE (checkpoint disabled) — expect a high peak / possible OOM")
    else:
        print("[mode] FIXED (per-chunk gradient checkpoint)")

    decoder = build_decoder(args, device)
    K = int(args.parts)
    loss_obj = PartSSLatentRFDecodeAwareLoss(
        ss_decoder=decoder,
        decode_aware_weight=0.1,
        decode_aware_voxel_resolution=64,
        decode_aware_max_parts_per_object=K,
        decode_aware_decode_chunk_size=int(args.chunk_size),
    )
    selections = [(0, k) for k in range(K)]
    # endpoint_raw shape in the real model: [B, K, latent_channels=8, 16, 16, 16].
    base = torch.randn(1, K, 8, 16, 16, 16, device=device, dtype=torch.float32, requires_grad=True)

    torch.cuda.reset_peak_memory_stats()
    chunks = [selections[i:i + int(args.chunk_size)] for i in range(0, K, int(args.chunk_size))]
    # Mirror the real loss path: decode each chunk, cat all logits, single backward,
    # all under fp16 autocast like the trainer.
    with torch.autocast("cuda", dtype=torch.float16):
        logits = torch.cat([loss_obj._decode_logits(base, c) for c in chunks], dim=0)  # noqa: SLF001
    logits.float().sum().backward()

    peak = torch.cuda.max_memory_allocated() / 1e9
    grad_ok = base.grad is not None and bool(torch.isfinite(base.grad).all().item())
    shape_ok = tuple(logits.shape) == (K, 1, 64, 64, 64)
    print(f"[decode] parts={K} chunk_size={args.chunk_size} res=64^3  logits={tuple(logits.shape)}")
    print(f"[result] peak_alloc={peak:.2f} GB  grad_finite={grad_ok}  shape_ok={shape_ok}")
    ok = grad_ok and shape_ok and (args.baseline or peak < args.max_peak_gb)
    verdict = "PASS" if ok else "FAIL"
    if not args.baseline:
        print(f"RESULT: {verdict} (peak {peak:.2f} GB vs {args.max_peak_gb:.1f} GB threshold)")
    else:
        print(f"RESULT: {verdict} (baseline run — peak {peak:.2f} GB shown for contrast)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
