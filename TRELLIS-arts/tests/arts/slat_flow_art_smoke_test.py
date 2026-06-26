#!/usr/bin/env python3
"""Phase 3 Stage 4 dual-mode Smoke Test Harness.

5 subtests:
  1. dataset+collate sanity (unit, ~10s)
  2. pretrained load (subprocess, max_steps=0, ~2min)
  3. memory+elastic+grad (subprocess, max_steps=20, ~3min)
  4. full mode 100 steps (subprocess, ~8min)
  5. lora mode 100 steps + elastic wired (subprocess, ~8min)

Usage:
  python TRELLIS-arts/tests/arts/slat_flow_art_smoke_test.py               # run all
  python TRELLIS-arts/tests/arts/slat_flow_art_smoke_test.py --test 1       # run single test
  python TRELLIS-arts/tests/arts/slat_flow_art_smoke_test.py --test full    # shortcut for Test 4
  python TRELLIS-arts/tests/arts/slat_flow_art_smoke_test.py --test lora    # shortcut for Test 5
  python TRELLIS-arts/tests/arts/slat_flow_art_smoke_test.py --test all-fast  # Tests 1-3 only

Outputs:
  output/smoke_test_phase3_full.json  - Test 4 result
  output/smoke_test_phase3_lora.json  - Test 5 result

Sources:
  - 03-RESEARCH.md section 8 Validation Architecture
  - 03-VALIDATION.md task map
  - TRELLIS-arts/tests/arts/smoke_test.py (parse_log_losses/parse_param_stats REUSE)
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

# Reuse existing parsers (MUST NOT re-implement)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from smoke_test import parse_log_losses, parse_param_stats  # noqa: E402

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
PYTHON = sys.executable  # use the same python that runs this script
TRAIN_PY = os.path.join(PROJECT_ROOT, 'TRELLIS-arts', 'train_arts.py')
CONFIG = os.path.join(
    PROJECT_ROOT, 'TRELLIS-arts', 'configs', 'arts', 'slat_flow_art', 'smoke_test.yaml'
)
PRETRAINED = 'pretrained/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors'


def _run_train(
    out_dir: str,
    max_steps: int,
    lora: bool,
    dump_param_stats: bool = False,
    log_file: str = None,
    timeout: int = 1200,
) -> Tuple[int, str]:
    """Run train.py as subprocess, return (returncode, combined stdout+stderr)."""
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env['TORCH_HOME'] = os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1')
    # Disable wandb for smoke tests
    env['WANDB_IGNORE_GLOBS'] = '*.pt,*.safetensors,*.ckpt'

    cmd = [PYTHON, TRAIN_PY, '--config', CONFIG]
    if dump_param_stats:
        cmd.append('--dump-param-stats')

    # All positional overrides go at the end (argparse nargs='*' requirement)
    overrides = [
        f'training.max_steps={max_steps}',
        'training.i_log=1',
        'training.i_save=50',
        f'training.pretrained_ckpt={PRETRAINED}',
        f'training.output_dir={out_dir}',
        'wandb.mode=disabled',
        f'lora.enabled={"true" if lora else "false"}',
    ]
    if lora:
        overrides += ['lora.rank=16', 'lora.target_modules=all_attn']
    cmd.extend(overrides)

    print(f'[slat_flow_art_smoke] CMD: {" ".join(cmd)}')
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        stdout_text = proc.stdout + '\n----STDERR----\n' + proc.stderr
    except subprocess.TimeoutExpired:
        stdout_text = f'TIMEOUT after {timeout}s'
        return -1, stdout_text

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(stdout_text)

    return proc.returncode, stdout_text


def _parse_peak_mem(stdout: str) -> float:
    """Parse `[Stage4 Train] peak_mem_mb=<val>` from stdout.

    M-5 fix: train.py unconditionally prints this line.
    Returns MB if found, -1.0 if not (indicates abnormal exit before printing).
    """
    m = re.search(r'peak_mem_mb=(\d+\.?\d*)', stdout)
    if m:
        return float(m.group(1))
    return -1.0


def _parse_elastic_events(stdout: str) -> int:
    """Count elastic-related log entries suggesting mem_ratio < 1.0 or checkpointing."""
    count = 0
    for line in stdout.splitlines():
        if 'mem_ratio' in line:
            m = re.search(r'mem_ratio[\'":\s]+([\d.]+)', line)
            if m and float(m.group(1)) < 1.0:
                count += 1
        if 'elastic' in line.lower() and ('checkpoint' in line.lower() or 'mem_ratio' in line.lower()):
            count += 1
    return count


def _check_pretrained_used(stdout: str) -> bool:
    """Check if pretrained weights were loaded based on stdout markers."""
    return 'loaded pretrained' in stdout or 'missing keys' in stdout


def _check_elastic_wired_lora(stdout: str) -> bool:
    """Parse the unconditional `elastic_wired_check: wired=<bool>` line.

    M-2 fix: _ensure_elastic_wired in train.py prints exactly one line containing
    `elastic_wired_check: wired=True|False reason=<reason>` in ALL code paths.
    """
    m = re.search(r'elastic_wired_check: wired=(\w+)', stdout)
    if m is None:
        return False
    return m.group(1).lower() == 'true'


# ============================================================
# TEST 1 -- Dataset + collate unit sanity
# ============================================================
def test_1_dataset_collate() -> Dict[str, Any]:
    """Construct dataset and verify sample format + collate produces SparseTensor."""
    print('\n[Test 1] dataset + collate_fn unit sanity')
    # Use inline subprocess to avoid polluting parent env with trellis stub
    env = os.environ.copy()
    env['TORCH_HOME'] = os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1')
    probe = f'''
import os, sys
PROJECT_ROOT = "{PROJECT_ROOT}"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "TRELLIS-arts"))
# Manual trellis stub (same as train_arts.py — avoids eager pipelines/rembg pull-in)
import types
for name in ["trellis"]:
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(PROJECT_ROOT, "TRELLIS-arts", name)]
    mod.__package__ = name
    sys.modules[name] = mod
for sub in ["models", "modules", "trainers", "utils", "datasets"]:
    m = types.ModuleType(f"trellis.{{sub}}")
    m.__path__ = [os.path.join(PROJECT_ROOT, "TRELLIS-arts", "trellis", sub)]
    m.__package__ = f"trellis.{{sub}}"
    sys.modules[f"trellis.{{sub}}"] = m
pip_pkg = types.ModuleType("trellis.pipelines")
pip_pkg.__path__ = [os.path.join(PROJECT_ROOT, "TRELLIS-arts", "trellis", "pipelines")]
pip_pkg.__package__ = "trellis.pipelines"
sys.modules["trellis.pipelines"] = pip_pkg

from trellis.datasets.arts.slat_flow_art import MvImageConditionedSparseLatentDataset
cfg = {{
    "data_root": os.path.join(PROJECT_ROOT, "data/smoke_test"),
    "recon_subdir": "reconstruction",
    "manifest_path": None,
    "test_obj_ids": ["100013","100015","100017","100021","100023"],
    "num_views": 4,
    "min_views": 1,
    "view_dropout": False,
    "max_num_voxels": 32768,
}}
ds = MvImageConditionedSparseLatentDataset(cfg)
assert len(ds) >= 5, f"dataset too small: {{len(ds)}}"
s = ds[0]
assert set(s.keys()) >= {{"coords","feats","cond"}}, f"keys: {{s.keys()}}"
assert s["coords"].shape[1] == 3, f"coords shape {{s['coords'].shape}}"
assert s["feats"].shape[1] == 8, f"feats shape {{s['feats'].shape}}"
assert s["cond"].ndim == 2, f"cond ndim {{s['cond'].ndim}}"
# Collate
from trellis.datasets.structured_latent import SLat
pack = SLat.collate_fn([ds[0], ds[1]])
assert pack["cond"].shape[0] == 2, f"cond batch {{pack['cond'].shape}}"
assert hasattr(pack["x_0"], "coords"), "x_0 not SparseTensor"
print(f"[Test 1] PASS len(ds)={{len(ds)}} N_sample0={{s['coords'].shape[0]}}")
'''
    proc = subprocess.run(
        [PYTHON, '-c', probe], cwd=PROJECT_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    stdout = proc.stdout + proc.stderr
    print(stdout[-1000:])
    return {
        'test': 1,
        'name': 'dataset_collate',
        'exit_code': proc.returncode,
        'stdout_tail': stdout[-500:],
        'passed': proc.returncode == 0 and '[Test 1] PASS' in stdout,
    }


# ============================================================
# TEST 2 -- pretrained load (max_steps=0)
# ============================================================
def test_2_pretrained_load() -> Dict[str, Any]:
    """Verify pretrained safetensors load with max_steps=0."""
    print('\n[Test 2] pretrained load (max_steps=0)')
    out_dir = os.path.join(PROJECT_ROOT, 'output', 'slat_flow_art_smoke_probe_load')
    log_file = os.path.join(out_dir, 'stdout.log')
    rc, stdout = _run_train(out_dir, max_steps=0, lora=False, log_file=log_file)

    missing_m = re.search(r'missing keys:\s*(\d+)', stdout)
    missing_n = int(missing_m.group(1)) if missing_m else 9999
    unexpected_m = re.search(r'unexpected keys:\s*(\d+)', stdout)
    unexpected_n = int(unexpected_m.group(1)) if unexpected_m else 0
    pretrained_used = _check_pretrained_used(stdout)

    # ElasticSLatFlowModel has a few extra buffers vs the pretrained SLatFlowModel,
    # so missing < 50 is acceptable. Unexpected keys MUST be 0 — they indicate
    # a model/ckpt architecture mismatch (Review Round 1 Finding 2 fix).
    passed = rc == 0 and pretrained_used and missing_n < 50 and unexpected_n == 0
    print(f'[Test 2] rc={rc} pretrained_used={pretrained_used} '
          f'missing_keys={missing_n} unexpected_keys={unexpected_n}')
    return {
        'test': 2,
        'name': 'pretrained_load',
        'exit_code': rc,
        'pretrained_used': pretrained_used,
        'missing_keys': missing_n,
        'unexpected_keys': unexpected_n,
        'passed': passed,
    }


# ============================================================
# TEST 3 -- memory + elastic + grad (max_steps=20)
# ============================================================
def test_3_mem_elastic_grad() -> Dict[str, Any]:
    """Run 20 steps to verify memory, elastic controller, and gradient clipping."""
    print('\n[Test 3] memory + elastic + grad (max_steps=20)')
    out_dir = os.path.join(PROJECT_ROOT, 'output', 'slat_flow_art_smoke_probe_mem')
    log_file = os.path.join(out_dir, 'stdout.log')
    rc, stdout = _run_train(out_dir, max_steps=20, lora=False, log_file=log_file)

    losses = [l for _, l in parse_log_losses(out_dir)]
    nan_count = sum(1 for l in losses if not math.isfinite(l))
    elastic_events = _parse_elastic_events(stdout)
    oom = 'out of memory' in stdout.lower() or 'OOM' in stdout

    peak_mem_mb = _parse_peak_mem(stdout)

    # M-5 fix: peak_mem_mb must be printed (not -1) AND < 22000 MB
    peak_mem_enforced = peak_mem_mb >= 0 and peak_mem_mb < 22000
    # Review Round 1 Finding 2 fix: elastic_events must be > 0 to prove
    # the LinearMemoryController is actually firing. If elastic never logs,
    # it means the config wiring is broken (D-18 violation).
    elastic_active = elastic_events > 0
    passed = rc == 0 and nan_count == 0 and not oom and peak_mem_enforced and elastic_active
    print(f'[Test 3] rc={rc} nan={nan_count} elastic_events={elastic_events} '
          f'elastic_active={elastic_active} oom={oom} peak_mem={peak_mem_mb:.1f} '
          f'peak_mem_enforced={peak_mem_enforced}')
    return {
        'test': 3,
        'name': 'mem_elastic_grad',
        'exit_code': rc,
        'num_loss_records': len(losses),
        'nan_count': nan_count,
        'elastic_events': elastic_events,
        'elastic_active': elastic_active,
        'oom_detected': oom,
        'peak_mem_mb': round(peak_mem_mb, 1),
        'peak_mem_mb_enforced': peak_mem_enforced,
        'passed': passed,
    }


# ============================================================
# TEST 4 -- full mode 100 steps
# ============================================================
def test_4_full_100() -> Dict[str, Any]:
    """Full mode 100-step training: loss trend + pretrained load verification."""
    print('\n[Test 4] full mode 100 steps')
    out_dir = os.path.join(PROJECT_ROOT, 'output', 'slat_flow_art_smoke_full')
    log_file = os.path.join(out_dir, 'stdout.log')
    rc, stdout = _run_train(out_dir, max_steps=100, lora=False, log_file=log_file)

    losses = [l for _, l in parse_log_losses(out_dir)]
    head = sum(losses[:10]) / max(len(losses[:10]), 1) if losses else float('inf')
    tail = sum(losses[-10:]) / max(len(losses[-10:]), 1) if losses else float('inf')
    trend_ok = (tail < head * 1.05) if len(losses) >= 20 else False

    ckpt_dir = os.path.join(out_dir, 'ckpts')
    ckpts = sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []

    pretrained_used = _check_pretrained_used(stdout)
    nan_count = sum(1 for l in losses if not math.isfinite(l))
    elastic_events = _parse_elastic_events(stdout)
    peak_mem_mb = _parse_peak_mem(stdout)

    passed = (rc == 0 and len(losses) >= 90 and nan_count == 0
              and pretrained_used and trend_ok)

    result = {
        'mode': 'full',
        'test': 4,
        'name': 'full_100',
        'exit_code': rc,
        'pretrained_used': pretrained_used,
        'pretrained_ckpt_path': PRETRAINED,
        'max_steps': 100,
        'num_loss_records': len(losses),
        'loss_head_mean': round(head, 6),
        'loss_tail_mean': round(tail, 6),
        'loss_trend_ok': trend_ok,
        'nan_count': nan_count,
        'elastic_events': elastic_events,
        'peak_mem_mb': round(peak_mem_mb, 1),
        'ckpt_count': len(ckpts),
        'passed': passed,
    }
    out_json = os.path.join(PROJECT_ROOT, 'output', 'smoke_test_phase3_full.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[Test 4] rc={rc} passed={passed} head={head:.4f} tail={tail:.4f} -> {out_json}')
    return result


# ============================================================
# TEST 5 -- lora mode 100 steps + elastic wired + freeze check
# ============================================================
def test_5_lora_100() -> Dict[str, Any]:
    """LoRA mode 100 steps: freeze verification + elastic wiring + loss trend."""
    print('\n[Test 5] lora mode 100 steps + elastic wired')
    out_dir = os.path.join(PROJECT_ROOT, 'output', 'slat_flow_art_smoke_lora')
    log_file = os.path.join(out_dir, 'stdout.log')
    rc, stdout = _run_train(
        out_dir, max_steps=100, lora=True,
        dump_param_stats=True, log_file=log_file,
    )

    losses = [l for _, l in parse_log_losses(out_dir)]
    head = sum(losses[:10]) / max(len(losses[:10]), 1) if losses else float('inf')
    tail = sum(losses[-10:]) / max(len(losses[-10:]), 1) if losses else float('inf')
    trend_ok = (tail < head * 1.05) if len(losses) >= 20 else False

    # Reuse smoke_test.parse_param_stats
    stats = parse_param_stats(stdout)
    trainable_ratio_pct = stats.get('trainable_ratio', -1.0)  # already in percentage
    trainable_ratio = trainable_ratio_pct / 100.0 if trainable_ratio_pct > 0 else -1.0
    non_lora_changed = stats.get('non_lora_changed', -1)
    lora_changed = stats.get('lora_changed', -1)

    ckpt_dir = os.path.join(out_dir, 'ckpts')
    ckpts = sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []

    pretrained_used = _check_pretrained_used(stdout)
    nan_count = sum(1 for l in losses if not math.isfinite(l))
    elastic_events = _parse_elastic_events(stdout)
    elastic_wired = _check_elastic_wired_lora(stdout)
    peak_mem_mb = _parse_peak_mem(stdout)

    freeze_ok = non_lora_changed == 0 and lora_changed > 0
    trainable_ok = 0.001 < trainable_ratio < 0.05

    passed = (rc == 0 and len(losses) >= 90 and nan_count == 0
              and pretrained_used and trend_ok
              and freeze_ok and trainable_ok and elastic_wired)

    result = {
        'mode': 'lora',
        'test': 5,
        'name': 'lora_100',
        'exit_code': rc,
        'pretrained_used': pretrained_used,
        'pretrained_ckpt_path': PRETRAINED,
        'max_steps': 100,
        'num_loss_records': len(losses),
        'loss_head_mean': round(head, 6),
        'loss_tail_mean': round(tail, 6),
        'loss_trend_ok': trend_ok,
        'nan_count': nan_count,
        'elastic_events': elastic_events,
        'elastic_wired_lora': elastic_wired,
        'peak_mem_mb': round(peak_mem_mb, 1),
        'ckpt_count': len(ckpts),
        'trainable_ratio': trainable_ratio,
        'non_lora_changed': non_lora_changed,
        'lora_changed': lora_changed,
        'freeze_ok': freeze_ok,
        'passed': passed,
    }
    out_json = os.path.join(PROJECT_ROOT, 'output', 'smoke_test_phase3_lora.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[Test 5] rc={rc} passed={passed} trainable_ratio={trainable_ratio:.4f} '
          f'non_lora_changed={non_lora_changed} elastic_wired={elastic_wired} -> {out_json}')
    return result


def main():
    ap = argparse.ArgumentParser(
        description='Phase 3 Stage 4 dual-mode smoke test harness'
    )
    ap.add_argument(
        '--test', default='all',
        help='Test selector: 1/2/3/4/5 or full/lora or all or all-fast'
    )
    args = ap.parse_args()

    results = []
    if args.test in ['1', 'all', 'all-fast']:
        results.append(test_1_dataset_collate())
    if args.test in ['2', 'all', 'all-fast']:
        results.append(test_2_pretrained_load())
    if args.test in ['3', 'all', 'all-fast']:
        results.append(test_3_mem_elastic_grad())
    if args.test in ['4', 'full', 'all']:
        results.append(test_4_full_100())
    if args.test in ['5', 'lora', 'all']:
        results.append(test_5_lora_100())

    all_passed = all(r.get('passed', False) for r in results)
    n_passed = sum(1 for r in results if r.get('passed', False))
    n_total = len(results)
    print(f'\n[slat_flow_art_smoke] {"ALL PASS" if all_passed else "FAIL"} -- '
          f'{n_passed}/{n_total} tests passed')
    for r in results:
        status = 'PASS' if r.get('passed') else 'FAIL'
        print(f'  Test {r["test"]} ({r.get("name", "?")}): {status}')

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
