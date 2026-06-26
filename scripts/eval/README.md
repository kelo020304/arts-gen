# Eval 推理脚本

三个独立的推理脚本：输出指标 JSON + 可视化 PNG，不依赖 YAML 配置。

## 环境要求

**所有脚本必须在 `trellis` conda 环境运行**（需要 flash_attn 或 xformers；sparse attention 不支持 sdpa）。

```bash
TRELLIS_PY=/home/jiziheng/anaconda3/envs/trellis/bin/python
```

脚本启动时会自动检测并选择 flash_attn / xformers backend；若两者都缺失会直接报错。

---

## 快速开始

### Stage 2 — SS Flow (Voxel IoU)

```bash
# 单样本
$TRELLIS_PY scripts/eval/stage2/infer.py \
  --ckpt pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \
  --data_root data/smoke_test \
  --obj_id 100015 --angle_idx 0 \
  --output output/eval_stage2/

# 批量
$TRELLIS_PY scripts/eval/stage2/infer.py \
  --ckpt pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \
  --data_root data/smoke_test \
  --manifest data/smoke_test/reconstruction/manifest.json \
  --output output/eval_stage2/
```

输出：`metrics.json` + 单样本模式下 `viz_{obj}_{angle}.png` + `pred_{obj}_{angle}.npz`。

### Stage 4 — SLat Flow (渲染 + PSNR/SSIM)

```bash
$TRELLIS_PY scripts/eval/stage4/infer.py \
  --ckpt pretrained/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \
  --decoder pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 \
  --data_root data/smoke_test \
  --obj_id 100015 --angle_idx 0 \
  --output output/eval_stage4/
```

输出：`metrics.json`（有 GT renders 时含 PSNR/SSIM）+ `viz_{obj}_{angle}.png`（4 view 渲染）。

### Part Predictor (Part mIoU)

```bash
$TRELLIS_PY scripts/eval/part_predictor/infer.py \
  --ckpt output/part_predictor_smoke/ckpts/step_5.pt \
  --data_root data/smoke_test \
  --obj_id 100015 --angle_idx 0 \
  --output output/eval_pp/
```

输出：`metrics.json`（per-part IoU + mean）+ `viz_{obj}_{angle}.png`（pred vs GT 着色体素）。

---

## 数据路径约定

```
{data_root}/reconstruction/
├── ss_latents_expanded/{obj}/angle_{N}/latent.npz       # key='mean' [8,16,16,16]
├── slat_latents_expanded/{obj}/angle_{N}/latent.npz     # coords [N,3] + feats [N,8]
├── dinov2_tokens/{obj}/angle_{N}/tokens.npz             # key='tokens' [V,T,D]
├── part_labels/{obj}/angle_{N}/part_labels_64.npy       # [64,64,64] int64
├── part_info/{obj}/part_info.json
├── renders/{obj}/angle_{N}/rgb/                         # 可选（stage4 GT 对比）
└── manifest.json                                        # 批量模式枚举样本
```

---

## 关键设计

- **无 YAML**：每个脚本顶部 `DEFAULT_MODEL_ARGS` dict，架构匹配标准预训练 L 模型
- **单/批量一键切换**：`--obj_id` + `--angle_idx` 走单样本，`--manifest` 走批量，互斥
- **可视化仅单样本模式**：批量模式只输出 `metrics.json`，避免生成大量 PNG
- **checkpoint 兼容**：自动识别 `.safetensors` 与 `.pt`（part_predictor 的 `.pt` 从 `raw['model']` 提取）
