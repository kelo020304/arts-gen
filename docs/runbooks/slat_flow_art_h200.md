# SLat Flow Art -- H200 Production Training Runbook

> 历史背景：本阶段曾命名为 stage4（SLat Flow），Phase 9 hard cut 后统一更名为 SLat Flow Art；
> 所有命令引用已切换到新路径（launcher: `scripts/train/slat_flow_art_train.bash`），仅个别历史背景说明保留旧名以便回溯。

**Phase 3 deliverable** -- independent from SS Flow Art runbook (`docs/runbooks/ss_flow_art_h200_production.md`) per CONTEXT D-10.

---

## 0. Prerequisites (one-time per machine)

### 0.1 Conda environment

```bash
# Use the trellis conda env (shared with Stage 1/2)
conda activate trellis
python --version  # 3.10
```

### 0.2 Verify CUDA extensions

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import diff_gaussian_rasterization; print('dgr OK')"
python -c "import spconv.pytorch; print('spconv OK')"
python -c "import torchmetrics; print(torchmetrics.__version__)"
```

If `torchmetrics` missing:
```bash
pip install torchmetrics
```

### 0.3 Path layout

- `TORCH_HOME` -> `{repo_root}/TRELLIS-arts/submodules/TRELLIS.1` (DINOv2 hub cache; must be
  importable even though Stage 4 does not encode images -- trainer MRO still touches the module)
- `pretrained/ckpts/` -> symlink to shared ckpt storage (see section 0.4)
- `data/PhysX-Mobility/` -> production data root (full ~2000 obj x 10 angles, data team owned)

### 0.4 Download pretrained ckpts (3 files, ~1.54 GB total)

```bash
REPO=JeffreyXiang/TRELLIS-image-large
CKPT_DIR=/path/to/shared/pretrained/ckpts
mkdir -p "$CKPT_DIR"

# --- Option A: huggingface-cli (recommended) ---
for base in slat_flow_img_dit_L_64l8p2_fp16 slat_enc_swin8_B_64l8_fp16 slat_dec_gs_swin8_B_64l8gs32_fp16; do
  for ext in safetensors json; do
    huggingface-cli download "$REPO" "ckpts/${base}.${ext}" \
      --local-dir "$(dirname $CKPT_DIR)" --local-dir-use-symlinks False
  done
done

# --- Option B: wget fallback ---
BASE_URL="https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/ckpts"
for base in slat_flow_img_dit_L_64l8p2_fp16 slat_enc_swin8_B_64l8_fp16 slat_dec_gs_swin8_B_64l8gs32_fp16; do
  wget -c "$BASE_URL/$base.safetensors" -O "$CKPT_DIR/$base.safetensors"
  wget -c "$BASE_URL/$base.json" -O "$CKPT_DIR/$base.json"
done

# --- Verify ---
ls -lh "$CKPT_DIR"/slat_*
# Expected: 6 files total, main DiT ~1.2 GB, encoder ~173 MB, decoder ~171 MB

# --- Symlink into repo ---
cd /path/to/arts-reconstruction
[ -L pretrained/ckpts ] || ln -s "$CKPT_DIR" pretrained/ckpts
```

### 0.5 Data readiness check

```bash
test -d data/PhysX-Mobility/arts/reconstruction/slat_latents_expanded/ || \
  { echo "[BLOCKER] slat_latents_expanded not delivered by data team"; exit 1; }

test -d data/PhysX-Mobility/arts/reconstruction/dinov2_tokens/ || \
  { echo "[BLOCKER] dinov2_tokens not delivered"; exit 1; }

test -d data/PhysX-Mobility/arts/reconstruction/renders/ || \
  echo "[WARN] renders not available -- eval phase needs this"
```

---

## 1. Pre-Training Split (MANDATORY first step)

**Reuse Phase 2's `scripts/train/split_manifest.py` directly -- SLat Flow Art does NOT ship its own copy (CONTEXT D-09).**

```bash
python scripts/train/split_manifest.py \
  --input data/PhysX-Mobility/arts/manifest.json \
  --output_dir data/PhysX-Mobility/arts \
  --val_ratio 0.1 \
  --seed 42
```

Output: `data/PhysX-Mobility/arts/manifest_train.json` + `manifest_val.json`

Same splits are used by Stage 2 and Stage 4 (obj_id-level md5 hash is deterministic and stage-agnostic).

---

## 2. Launch Full Fine-tune

```bash
cd /path/to/arts-reconstruction
TORCH_HOME=TRELLIS-arts/submodules/TRELLIS.1 \
bash scripts/train/launchers/launch_ddp.sh 8 \
  scripts/train/slat_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/slat_flow_art/mv_4view.yaml
```

Settings come from `mv_4view.yaml`:
- model: ElasticSLatFlowModel (24 blocks, 1024 channels, fp16)
- batch_size_per_gpu: 8, batch_split: 4
- fp16_mode: inflat_all, fp16_scale_growth: 1e-3
- grad_clip: AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
- elastic: LinearMemoryController(target_ratio=0.75, max_mem_ratio_start=0.5)
- lr: 1e-4 AdamW, max_steps: 50000
- i_sample: 10000 (decoder-based snapshot, renders 4 views x 512^2 to wandb)
- i_save: 10000, pretrained_ckpt: slat_flow_img_dit_L_64l8p2_fp16.safetensors

---

## 3. Launch LoRA Fine-tune

```bash
TORCH_HOME=TRELLIS-arts/submodules/TRELLIS.1 \
bash scripts/train/launchers/launch_ddp.sh 8 \
  scripts/train/slat_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/slat_flow_art/mv_4view_lora.yaml
```

Overrides from `mv_4view_lora.yaml`:
- lora: enabled=true, rank=16, alpha=32, target_modules=all_attn
- output_dir: output/slat_flow_art_mv_4view_lora
- wandb.name: slat-flow-art-mv4view-lora, tags: [slat_flow_art, mv4view, mode=lora]

Same data, same optimizer, same elastic config -- **only trainable params differ (~0.9%).**

---

## 4. Expected Walltime

- 8x H200, batch_size=8 x split=4 = effective 32 per step
- elastic target 0.75 -> mostly full throughput, some blocks checkpointed under pressure
- 50k steps ~ ~2.5 days (estimate -- refine after 1k-step warmup)
- Speed printout every 100 steps (`i_print: 100`)

---

## 5. Checkpoint and Resume

Auto-save at `i_save: 10000` -> `output_dir/ckpts/denoiser_step{step:07d}.pt`

**Resume:**
```bash
TORCH_HOME=TRELLIS-arts/submodules/TRELLIS.1 \
bash scripts/train/launchers/launch_ddp.sh 8 \
  scripts/train/slat_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/slat_flow_art/mv_4view.yaml \
  --load-dir output/slat_flow_art_mv_4view \
  --resume-step 20000
```

**CRITICAL**: use `--load-dir` / `--resume-step` as top-level argparse flags, NOT OmegaConf overrides (`training.load_dir=...`). The latter is silently swallowed by args.overrides and never injected into the trainer -- this is the same gotcha as SS Flow Art (docs/runbooks/ss_flow_art_h200_production.md section 5).

When `is_resuming=True`, train.py automatically skips `pretrained_ckpt` to avoid overwriting checkpoint weights (train_arts.py main() around line 250 — historically stage4/train.py).

**LoRA resume note**: peft-wrapped checkpoint keys look like `base_model.model.blocks.{i}.self_attn.to_qkv.lora_A.default.weight`. These reload cleanly ONLY if the same peft wrapper is re-applied in the same order on resume. The train.py flow (build -> load pretrained -> apply_lora -> build Trainer with load_dir) guarantees this.

---

## 6. Elastic / Memory Troubleshooting

Watch `step_log['elastic']` in stdout or wandb:
- `mem_ratio` -- predicted memory usage ratio (target 0.75)
- Lower values -> more blocks checkpointed -> slower but safer
- `params/k`, `params/b` -- LinearMemoryController regression coefficients (converge after ~500 steps)

**Symptoms -> Actions:**

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Persistent OOM | Effective batch too high | Lower `batch_size_per_gpu`, raise `batch_split` (keep product constant) |
| `mem_ratio` stays at 1.0, still OOM | Elastic not engaged | Check `trainer.elastic_controller is not None` after init; if LoRA mode, verify `_ensure_elastic_wired` fired (stdout `elastic_wired_check: wired=True`) |
| Loss NaN first step | fp16 underflow or bad pretrained match | Check `fp16_scale_growth` in step log; reduce lr to 5e-5; verify `missing keys < 50` on ckpt load |
| `adaptive grad clip` always returns max_norm=1.0 | Buffer not filled (first 1000 steps) | Expected -- adapts after step 1000 |

---

## 7. Evaluation (post-training)

```bash
TORCH_HOME=TRELLIS-arts/submodules/TRELLIS.1 \
python scripts/train/eval_slat_render.py \
  --config TRELLIS-arts/configs/arts/slat_flow_art/mv_4view.yaml \
  --ckpt output/slat_flow_art_mv_4view/ckpts/denoiser_step050000.pt \
  --decoder pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 \
  --output output/slat_flow_art_eval_final_full.json
```

For LoRA (using `--lora_ckpt` for TRELLIS BasicTrainer .pt format):
```bash
python scripts/train/eval_slat_render.py \
  --config TRELLIS-arts/configs/arts/slat_flow_art/mv_4view_lora.yaml \
  --ckpt pretrained/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \
  --lora_ckpt output/slat_flow_art_mv_4view_lora/ckpts/denoiser_step050000.pt \
  --decoder pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 \
  --output output/slat_flow_art_eval_final_lora.json
```

**Output JSON schema** (aligned with Phase 5 consumers):
```json
{
  "ckpt": "...",
  "lora_ckpt": "...",
  "lora_adapter": null,
  "decoder": "...",
  "n_samples": 200,
  "n_valid": 198,
  "psnr_mean": 24.5,
  "ssim_mean": 0.812,
  "per_sample": [{"obj_id": "...", "angle_idx": 0, "psnr": 25.1, "ssim": 0.83, "status": "ok"}],
  "camera_alignment_warning": "..."
}
```

**IMPORTANT**: at production scale, camera alignment between generated renders and GT Blender renders must be verified by Phase 5 infrastructure. The `camera_alignment_warning` field in the JSON output is a reminder -- do not quote PSNR/SSIM numbers in the paper until Phase 5 sign-off.

---

## 8. Monitoring (wandb)

- Project: `arts-reconstruction`
- Tags: `slat_flow_art`, `mv4view`, `mode=full` or `mode=lora`
- Watched metrics: `loss/mse`, `elastic/mem_ratio`, `grad_clip/max_norm`
- Alert: loss NaN for 3 consecutive steps (manual -- wandb dashboard alerts)

---

## 9. Known Issues / FAQ

**Q: Why does train.py not register `trellis.representations` stub like SS Flow Art train.py?**
A: SLat Flow Art snapshot/eval path needs `from trellis.representations import Gaussian` to resolve (decoder_gs.py:8). The stub blocks this. Real `representations/__init__.py` only imports plyfile/utils3d/diffoctreerast -- all safe in `trellis` conda env. See 03-RESEARCH.md section 6.3.

**Q: Can I use plain `SLatFlowModel` instead of `ElasticSLatFlowModel`?**
A: No. SLat Flow Art smoke was tuned for elastic memory controller; removing it risks OOM even on H200 at batch=8x4. Also `with_mem_ratio` context is a critical part of the forward path.

**Q: Do I need to override `get_cond` AND `get_inference_cond` in Stage4Trainer?**
A: Yes, both. `get_cond` is called from `training_losses` (every step); `get_inference_cond` is called from `run_snapshot` (every `i_sample` steps). Missing either one causes DINOv2 download on first occurrence. See 03-RESEARCH.md section 1.2.

**Q: LoRA x Elastic -- how is the isinstance mismatch handled?**
A: `train.py::_ensure_elastic_wired()` walks into `PeftModel.base_model.model` to find the real `ElasticSLatFlowModel`, then manually calls `register_memory_controller` if needed. Smoke test (`slat_flow_art_smoke_test.py` (历史名 stage4_smoke_test.py) Test 5) asserts this via the `elastic_wired_lora: true` field. See 03-RESEARCH.md section 2.3.

**Q: How do I verify normalization mean/std are correct for PhysX-Mobility?**
A: The YAML defaults are TRELLIS Objaverse statistics (TODO marker in mv_4view.yaml). After H200 first run, inspect initial loss -- if > 2.0 (canonical is ~1.2), stats are likely off. Run a one-shot statistics computation pass over `slat_latents_expanded/` feats and override in a new YAML variant.

**Q: How do I load LoRA checkpoints for eval?**
A: Two paths supported (Phase 2 Review Round 1 #1 lesson):
1. `--lora_ckpt path/to/denoiser_stepN.pt` (TRELLIS BasicTrainer output, peft-wrapped state_dict)
2. `--lora_adapter path/to/adapter_dir/` (PEFT standard format with adapter_model.safetensors)
Path 1 is the default for checkpoints produced by `train.py`. Path 2 is for separately saved adapters via `peft.save_pretrained()`.

---

## 10. Contact

- Data team: slat_latents_expanded / dinov2_tokens / renders delivery
- H200 ops: walltime / resume / node allocation
- SLat Flow Art ownership: see 03-CONTEXT.md + 03-01-SUMMARY.md + 03-02-SUMMARY.md

---
*Phase: 03-stage-4-slat-flow, Plan 02*
*Runbook version: v0.1*
*Date: 2026-04-12*
