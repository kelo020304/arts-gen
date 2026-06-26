#!/usr/bin/env python3
"""Bucket a PyTorch profiler key-averages text file for TC optimization triage.

This is a scratchpad diagnostic helper. It does not import or modify the
training path. Feed it ``profiler/key_averages_self_cuda.txt`` from
``scripts.tools.train_part_kin --torch-profiler``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


MMA_PATTERNS = (
    "hmma",
    "mma",
    "tensorop",
    "cutlass",
    "flash",
    "gemm",
    "matmul",
    "bmm",
    "mm",
    "addmm",
)
MEM_PATTERNS = (
    "layer_norm",
    "native_layer_norm",
    "softmax",
    "gelu",
    "silu",
    "sigmoid",
    "where",
    "masked_fill",
    "index",
    "gather",
    "scatter",
    "nonzero",
    "copy",
    "cast",
    "to",
    "cat",
    "slice",
    "select",
    "repeat",
    "fill",
    "zero",
    "sum",
    "binary_cross_entropy",
    "clamp",
    "div",
    "mul",
    "add",
    "sub",
)
CPU_SYNC_PATTERNS = (
    "item",
    "tolist",
    "empty_cache",
    "linear_sum_assignment",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
)


def parse_time_us(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s)\s*$", text.strip())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    return value * 1_000_000.0


def bucket_name(op: str) -> str:
    low = op.lower()
    if any(pattern in low for pattern in CPU_SYNC_PATTERNS):
        return "cpu_sync_or_idle"
    if any(pattern in low for pattern in MMA_PATTERNS):
        return "mma"
    if any(pattern in low for pattern in MEM_PATTERNS):
        return "memory_bound"
    return "other"


def parse_key_averages(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("aten::") and not line.startswith("cuda") and "::" not in line:
            continue
        parts = [piece.strip() for piece in line.split()]
        if not parts:
            continue
        op = parts[0]
        # PyTorch table columns vary. The self CUDA time is the first duration
        # after the self CUDA percentage column in the common text table; if the
        # format changes, fall back to the last duration on the row.
        times = [parse_time_us(piece) for piece in parts]
        times = [value for value in times if value is not None]
        if not times:
            continue
        rows.append((op, float(times[0])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key_averages", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = parse_key_averages(args.key_averages)
    totals: dict[str, float] = defaultdict(float)
    by_bucket: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for op, us in rows:
        bucket = bucket_name(op)
        totals[bucket] += us
        by_bucket[bucket].append((op, us))
    total = sum(totals.values())
    print("bucket,self_cuda_ms,pct")
    for bucket, us in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * us / total if total > 0 else 0.0
        print(f"{bucket},{us / 1000.0:.3f},{pct:.2f}")
    print()
    print("bucket,op,self_cuda_ms,pct")
    for bucket in ("mma", "memory_bound", "cpu_sync_or_idle", "other"):
        for op, us in sorted(by_bucket.get(bucket, []), key=lambda item: item[1], reverse=True)[: int(args.top)]:
            pct = 100.0 * us / total if total > 0 else 0.0
            print(f"{bucket},{op},{us / 1000.0:.3f},{pct:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
