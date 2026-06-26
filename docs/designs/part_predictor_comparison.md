# 性能对比 Checklist

## 一、Part Flow vs Part Predictor

| 指标 | Part Predictor | Part Flow |
|---|---|---|
| val mIoU |  |  |
| per-part IoU (mean) |  |  |
| class accuracy |  | N/A |
| 收敛 step |  |  |
| 推理时间 (ms, A100 fp16) |  |  |
| 显存峰值 (GB, train batch=4) |  |  |
| 参数量 (M) |  |  |

## 二、Part Predictor 内部三方案

- **A** = 当前串行多路 cross-attn (voxel → rgb → mask,独立残差)
- **B** = Concat KV (三模态 feats 拼起来,共享 Q/K/V,单 softmax)
- **C** = MMDiT (模态独立 K/V + QK-Norm,joint softmax)

| 指标 | A (当前) | B (Concat) | C (MMDiT) |
|---|---|---|---|
| val mIoU |  |  |  |
| per-part IoU (mean) |  |  |  |
| class accuracy |  |  |  |
| 训练 loss @ 50k step |  |  |  |
| 显存峰值 (GB, batch=4) |  |  |  |
| 单 step 时间 (ms, A100) |  |  |  |
| 参数量 (M) |  |  |  |
| 是否训练稳定 (Y/N) |  |  |  |

## 已落地:YAML 切换

三方案代码已实现,通过 `scripts/train/configs/part_predictor/base.yaml` 的 `model.fusion_mode` 字段切换:

| 值 | 方案 | Decoder layer class |
|---|---|---|
| `serial` (默认) | A (当前) | `PartDecoderLayer` |
| `concat_kv` | B | `PartDecoderLayerConcatKV` |
| `mmdit` | C | `PartDecoderLayerMMDiT` (含 QK-Norm + joint_o zero-init) |

代码位置: `TRELLIS-arts/trellis/models/part_predictor/part_predictor.py`。
实现细节见 `docs/archive/phase04/part_predictor_fusion_modes.md`。

## Serial 模式注入顺序消融 (fusion_mode="serial")

Serial 模式下三路 cross-attn 的顺序通过 `model.serial_order` 切换。6 种排列都可跑消融，默认 `["voxel", "rgb", "mask"]` 与升级前完全一致。

| 顺序 | `serial_order` 值 | val mIoU | 训练 loss @ 50k | 显存峰值 (GB) |
|---|---|---|---|---|
| voxel → rgb → mask (默认) | `["voxel", "rgb", "mask"]` |  |  |  |
| voxel → mask → rgb | `["voxel", "mask", "rgb"]` |  |  |  |
| rgb → voxel → mask | `["rgb", "voxel", "mask"]` |  |  |  |
| rgb → mask → voxel | `["rgb", "mask", "voxel"]` |  |  |  |
| mask → voxel → rgb | `["mask", "voxel", "rgb"]` |  |  |  |
| mask → rgb → voxel | `["mask", "rgb", "voxel"]` |  |  |  |

备注: 仅在 `fusion_mode="serial"` 时生效；对 `concat_kv` 和 `mmdit` 无意义（跨模态联合 softmax 对顺序天然不敏感）。

## 测试约束

- 同一 split、同一 seed (42)、同一 lr (1e-4)、同一 max_steps (50k)
- 只改 `PartDecoderLayer` 内部融合方式,其余全锁
