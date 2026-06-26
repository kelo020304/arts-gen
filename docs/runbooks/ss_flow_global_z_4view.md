# SS Flow Global-Z 4View Runbook

## Goal

Fine-tune TRELLIS `SparseStructureFlowModel` from 4 pre-encoded DINOv2 views to
whole-object `z_global` SS latent. Inputs are
`reconstruction/dinov2_tokens/<id>/angle_<i>/tokens.npz["tokens"]`; targets are
`reconstruction/ss_latents_expanded/<id>/angle_<i>/latent.npz["mean"]`.

DINOv2 and SS VAE are not trained. Snapshot decode must use the TRELLIS
decoder paired with the encoder that produced `ss_latents_expanded`:
`pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors`.

## Smoke

```bash
python TRELLIS-arts/train_arts.py \
  --config TRELLIS-arts/configs/arts/ss_flow_global_z/smoke_test.yaml \
  training.max_steps=30 \
  training.output_dir=output/ss_flow_global_z_smoke_30step
```

Expected outputs:

- `output/ss_flow_global_z_smoke_30step/log.txt`
- `output/ss_flow_global_z_smoke_30step/samples/init/global_z_decode_init.png`
- `output/ss_flow_global_z_smoke_30step/samples/final/global_z_decode_final.png`
- matching `.json` files with decoded voxel counts and logit stats

Previous local smoke evidence with the invalid snapshot decoder:

- 30 losses logged.
- first 5 mean loss: `0.1776868388`
- last 5 mean loss: `0.0277049880`
- final decoded sample voxel count: `1870`
- GT decoded voxel count: `3511`

The loss trend is still useful, but the decoded GT count above is not valid:
that run used `/robot/data-lab/jzh/art-gen/weights/ss_decoder.ckpt`, a SAM3D/TDFY
decoder, to decode TRELLIS SS latents. The data pipeline's TRELLIS decoder
output for `100015/angle_0` matches `surface.npy` exactly: `11798` voxels,
IoU `1.0`.

Corrected single-object evidence:

- `output/ss_flow_global_z_smoke_30step_trellisdec`
  - GT voxels: `11798`
  - final sample voxels: `49417`
  - loss last: `0.05335`
- `output/ss_flow_global_z_overfit_1000_trellisdec`
  - step 250 sample voxels: `12072`
  - step 500 sample voxels: `11798`
  - step 750 sample voxels: `11801`
  - step 1000/final sample voxels: `11798`
  - loss step 1: `0.28537`
  - loss step 1000: `0.00092976`

## Multi-Object Overfit Gate

```bash
ATTN_BACKEND=sdpa PYTHONPATH=TRELLIS-arts \
python TRELLIS-arts/train_arts.py \
  --config TRELLIS-arts/configs/arts/ss_flow_global_z/multi_overfit_8obj.yaml
```

Output:

- `output/ss_flow_global_z_multi_overfit_8obj_800/loss_curve.png`
- `output/ss_flow_global_z_multi_overfit_8obj_800/samples/step0000600/global_z_decode_step0000600.png`
- `output/ss_flow_global_z_multi_overfit_8obj_800/samples/final/global_z_decode_final.png`

Result:

- loss step 1: `0.2478547990`
- loss step 800: `0.0030767135`
- first 20 mean: `0.1129645270`
- last 20 mean: `0.0050521164`
- step 600 mean own IoU: `0.8529`
- step 600 max cross IoU: `0.0670`
- final mean own IoU: `0.7524`
- final max cross IoU: `0.0740`

Judgement: passed. The eight samples remain object-specific and do not collapse
to one shared shape. Pot/chair are the hardest under stochastic sampling, so the
full run uses fixed snapshot seed `1234` for stable val trend tracking.

The full run no longer uses this 8-object set as validation. It uses a
deterministic 200-object object-level holdout:

```bash
TRELLIS-arts/configs/arts/ss_flow_global_z/splits/val_objects_seed20260605_n200.txt
```

## Train

```bash
python TRELLIS-arts/train_arts.py \
  --config TRELLIS-arts/configs/arts/ss_flow_global_z/mv_4view.yaml \
  training.output_dir=output/ss_flow_global_z_mv_4view
```

Prepared full-data config:

```bash
TRELLIS-arts/configs/arts/ss_flow_global_z/full_train.yaml
```

Do not launch full training until the run owner confirms output/checkpoint
location. Current prepared output directory is:

```bash
/robot/data-lab/jzh/art-gen-output/ss_flow_global_z_full_4view
```

Single GPU launch:

```bash
ATTN_BACKEND=sdpa PYTHONPATH=TRELLIS-arts \
python TRELLIS-arts/train_arts.py \
  --config TRELLIS-arts/configs/arts/ss_flow_global_z/full_train.yaml
```

8 GPU launch:

```bash
ATTN_BACKEND=sdpa PYTHONPATH=TRELLIS-arts \
torchrun --nproc_per_node=8 TRELLIS-arts/train_arts.py \
  --config TRELLIS-arts/configs/arts/ss_flow_global_z/full_train.yaml
```

Important config paths:

- SS flow init: `/robot/data-lab/jzh/art-gen/weights/ss_flow_img_dit_L_16l8_fp16.safetensors`
- SS decoder snapshot: `pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors`
- data root: `/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511`

The trainer overrides `get_cond`, `get_inference_cond`, `snapshot_dataset`, and
`snapshot` so pre-encoded tokens are never routed through DINOv2/image encoding.
Full training excludes every angle of the 200 held-out object IDs via
`data.exclude_obj_ids_file` and loads the same objects through
`snapshot.data.test_obj_ids_file` with `one_sample_per_object: true`, so the
split is object-level and snapshots are not leaked from training samples.
Snapshot JSON computes IoU/voxel metrics for all 200 validation objects; PNGs
render the first 16 pairs only. Training keeps `ema_rate: 0.9999` enabled, and
validation snapshots use `snapshot.use_ema: true`, so offline eval should load
`denoiser_ema0.9999_stepXXXXXXX.pt`.
