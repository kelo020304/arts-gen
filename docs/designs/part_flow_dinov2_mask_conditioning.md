# Part Flow — DINOv2 + Mask 条件机制

本文档说明 PartFlow 训练 / 推理时，**DINOv2 特征**和 **2D part mask** 是怎么组合成条件信号、再喂进 velocity predictor 的。面向首次看这个模块的人，不假设读过 Gumbel/Fisher categorical FM 的论文。

对比参考：[`part_predictor_dinov2_multiview_fusion.md`](part_predictor_dinov2_multiview_fusion.md)（Part Predictor 的 DINOv2 用法）。

---

## 0. 一张图看懂

```
DINOv2 tokens [B, V·T, 1024]              mask_token_labels [B, V·T] int64
        │                                         │
        │  rgb_proj (1024→256)                    │ （0 = bg/CLS, 1..K_b = part id）
        ▼                                         │
cond_proj [B, V·T, 256] ──────────┬───────────────┘
                                  │
                                  │   (按 part id 做 masked average pooling)
                                  ▼
                           build_part_tokens()
                                  │
                                  ▼
                 part_tokens [B, k_max=128, 256]  ← 每个 sample 只用前 K_b 个 slot
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  ① x_t 编码           ② 每层 cross_part KV        ③ 分类头 key
  x_t_emb = Σ_k x_t·pt  （解码时读 part 特征）    （voxel·part 点积出 logits）

cond_proj 本身 ───► 每层 cross_rgb KV（不过 mask，保留全局上下文）
```

两句话总结：
- **条件有两份并用**：一份是全局视觉（`cond_proj`，所有 DINOv2 token），一份是 per-part 摘要（`part_tokens`，mask 池化后的 K_b 个向量）。
- **mask 的唯一职责**：把 DINOv2 token 按 part id 重组成可按 index 查表的 per-part 字典。除此之外，cross_rgb 那条路 mask 一次也没用。

---

## 1. 输入数据怎么来

> 数据侧代码：[scripts/train/part_predictor/dataset.py:195-255](../scripts/train/part_predictor/dataset.py#L195-L255)
> （PartFlow 复用 `PartPredictorDataset` 作为父类）

### 1.1 `cond`：DINOv2 tokens

- Shape: `[B, V·T, 1024]`
  - `V = num_views = 4`（训练时按 view dropout 从 12 个象限视角里随机选 4 个；不足补 0）
  - `T = 1370`（= 37×37 patch tokens + 1 CLS token，DINOv2-L/14-reg @ 518 px）
  - `D = 1024`（DINOv2-L 的 token 维度）
- 离线预编码产出于 `data/.../dinov2_tokens/{obj}/angle_{a}/tokens.npz`。

### 1.2 `mask_token_labels`：2D mask 下采样到 patch 分辨率

- Shape: `[B, V·T]`, dtype int64
- 每个 DINOv2 patch token 带一个 part id：
  - `0` → 背景 / CLS token / 被 dropout 掉的视角
  - `1..K_b` → **0-indexed part 索引 +1**（所以 id=1 对应第 0 个 part）
- 生成方式：把原始 512×512 的 int32 mask（每个像素的 raw part label）用 **nearest interpolation** 下采样到 37×37 patch grid，再通过 `label_to_idx` 重映射为 0-indexed。CLS token 固定写 0。

### 1.3 `num_parts` 与 `x_t`

- `num_parts: List[int]`：每个 sample 的实际 part 数 K_b（来自 `part_info.json`，跨 sample 变化）。
- `x_t [N_total, k_max=128]`：当前 flow 时刻每个 voxel 在 K_b 个 part 上的概率（padding 维强制为 0）。`x_t` 由 bridge 在训练时按 `t ∈ [0,1]` 采样生成，本文不展开。

---

## 2. 两步处理：投影 + pooling

模型入口 [`PartFlowPredictor.forward`](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L326)：

### 2.1 降维

```python
cond_proj = self.rgb_proj(cond)          # [B, V·T, 256]  —— 线性 1024→256
```

这是唯一一次对 DINOv2 tokens 做变换。之后 `cond_proj` 被复用在两条路上。

### 2.2 mask-guided pooling

核心函数 [`build_part_tokens`](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L267)：

```python
# 伪代码（真实代码用 one-hot + einsum 向量化）
for b in range(B):
    for j in range(1, num_parts[b] + 1):
        # 在 sample b 里找所有 mask == j 的 patch token，取均值
        mask_ij = (mask_token_labels[b] == j)             # [V·T] bool
        if mask_ij.sum() > 0:
            part_tokens[b, j-1] = cond_proj[b, mask_ij].mean(0)
        else:
            # 该 part 在所有视角里都被遮挡
            part_tokens[b, j-1] = slot_emb[j-1]  if use_slot_embedding_fallback else 0
```

输出 `part_tokens [B, k_max=128, 256]`。细节：

| 情形 | 结果 |
|---|---|
| `mask == 0`（bg/CLS） | 被排除在任何 part 的 pooling 之外 |
| `mask == j`, 有覆盖（`count > 0`） | `part_tokens[b, j-1]` = 被 pool 到的 patch tokens 的平均 |
| `mask == j`, 无覆盖（该 part 在所有视角都被遮挡） | 如果 `use_slot_embedding_fallback=True`，用 learnable `slot_emb[j-1]` 兜底；否则置 0 |
| `j >= num_parts[b]`（padding slot） | 置 0，并在 `part_valid_mask` 里标 False |

> **为什么要 `slot_emb` 兜底？**
> 没有 fallback 时，被完全遮挡的 part 对应一个 0 向量：x_t 在它上面的概率权重会让 `x_t_emb` 吃到 0 贡献，cross_part attention 看到的 KV 也是 0 → 模型永远没法把 voxel 判给这个 part。slot_emb 给模型一个"虽然 2D 看不见但 3D 有体素"的先验。

---

## 3. 两条条件通路在 decoder 里怎么用

### 3.1 路径 A：`cross_rgb` — 全局上下文

每层 [`PartFlowDecoderLayer`](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L67) 的第 2 子层：

```python
# voxel_tokens 作为 Q，cond_proj 所有 token 作为 K/V
cross_rgb(voxel_tokens, cond_proj_packed)
```

特点：
- **K/V = 全部 DINOv2 tokens**，包括背景、CLS、被 dropout 的视角。
- **没做 mask-based gating**。这是故意的：即使 2D mask 偶尔漏分或错分，voxel 仍能从原始 DINOv2 里捞几何线索（物体外形、边界等）。
- 每个 voxel 都能看所有 V·T 个视觉 token（varlen attention，跨 batch 不串扰）。

### 3.2 路径 B：`cross_part` — per-part 字典检索

每层第 3 子层：

```python
# voxel_tokens 作为 Q，当前 sample 有效的 K_b 个 part_tokens 作为 K/V
cross_part(voxel_tokens, part_tokens[valid_slots])
```

特点：
- **K/V = sample-specific，长度 K_b 不定**（sample A 的 K_b=3，sample B 的 K_b=7，各自只用各自的 slot）。
- `cross_part_k/v` 的 `bias=False`（[line 112-113](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L112-L113)）——当某 sample K_b=0 时 KV 是空序列，保证输出严格为 0，不会因为 bias 产生非零漂移。
- 语义类比：`part_tokens[b, j]` 是"part j 的视觉签名"，voxel 通过 cross-attn 问"我更像哪个 part"。

### 3.3 额外用法：`part_tokens` 也进 x_t 编码和分类头

除了两条 cross-attn，`part_tokens` 还在另外两处被用：

1. **x_t 编码**（[line 380-384](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L380-L384)）——decoder 的输入就靠它：

   ```python
   x_t_emb = einsum('nk, bkh -> nh', x_t, part_tokens[batch_idx])
   voxel_tokens = x_t_emb + pos_embed(xyz)
   ```

   `x_t[n, :]` 是 voxel n 在 K_b 个 part 上的概率，`part_tokens[b, :]` 是这些 part 的特征字典。两者点乘 = "当前 x_t 概率下，这个 voxel 的期望特征向量"。

2. **分类头 key**（[line 412-417](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L412-L417)）——最终输出 logits 就是 voxel 向量和 part 向量的点积：

   ```python
   voxel_q  = voxel_score_proj(out_norm(voxel_tokens))   # [N, 256]
   part_k   = part_score_proj(part_tokens)               # [B, k_max, 256]
   logits   = einsum('nh, nkh -> nk', voxel_q, part_k[batch_idx]) / sqrt(256)
   logits   = logits.masked_fill(~valid_per_voxel, -1e4)  # padding slot = -inf
   ```

所以 `part_tokens` 总共被用了 **4 次**：encode x_t、cross_part KV（每层）、分类头 key、还有作为检测 `present` 的统计来源。

---

## 4. 和 Part Predictor 的对照

| 维度 | Part Predictor | PartFlow |
|---|---|---|
| 任务类型 | 判别式（Mask2Former-style） | 生成式（categorical flow matching） |
| Query 来源 | k_max 个 **learnable** 参数（样本无关） | 每个 **实际 voxel**（样本相关） |
| Query 数 | 固定 k_max | 变长 N_total，per-sample K_b 变化 |
| Mask 用途 | 独立第三路 cross-attn（gather mask>0 的 raw DINOv2 tokens 做 KV） | 不作为独立分支，而是**pooling 指令**生成 per-part 字典 |
| 跨 part 对应关系 | Hungarian matching（后验分配） | 直接靠 mask id（前验对齐） |
| Fusion mode | serial / concat_kv / mmdit 可选 | 只有 serial（rgb + part 两路） |
| 参数量 | ~253M（12 层 × 1024 dim） | ~7.5M（4 层 × 256 dim） |

**为什么 PartFlow 不照抄 Part Predictor 的三路融合**：

1. **K 维对齐是硬约束**：x_t 是 K_b 维概率，`x_t_emb = Σ_k x_t[k] · pt[k]` 必须拿到形状 `[K_b, H]` 的字典。Part Predictor 的 mask 分支是扁平 gather（`[Σ_coverage, H]`），没有 part 轴，**数学上没法替代 pooling**。
2. **没有 Hungarian 的自由**：Part Predictor 用 Hungarian 允许 query 和 part 对应关系后验确定；PartFlow 的 part 身份由 mask id **前验锁定**（id=j → 第 j-1 个 slot），模型必须照这个身份学。
3. **MMDiT 不合适**：MMDiT 假设两条 stream 共进化，PartFlow 的 part_tokens 是纯条件，没有独立监督信号，强行 MMDiT 等于让未监督的 part_tokens 每层漂移，训练不稳定。

详细讨论见对话记录（2026-04-23 会话）或 [part_predictor.py 的 fusion_mode 部分](../TRELLIS-arts/trellis/models/part_predictor/part_predictor.py#L355)。

---

## 5. 常见问题

### Q1: 那条 rgb 路既然没过 mask，是不是冗余？

不是。rgb 路看到的包含 **bg tokens + CLS + padded-view 零向量**，这些信息在 pool 出的 part_tokens 里都被 mask 过滤掉了。如果 2D mask 有错，rgb 路是唯一的 backup。

### Q2: 可以只用 cross_part，去掉 cross_rgb 吗？

不建议。这么做相当于让模型完全依赖 2D mask 的质量。如果 mask 某 part 没覆盖到（全遮挡）或 mask 边界不精，cross_rgb 的全局信息就补不回来了。消融实验要做这个对照时，保留开关但默认开启。

### Q3: `mask_token_labels` 的精度有多重要？

关键但不致命。pooling 对个别像素的 label 错误鲁棒（平均会抹平），但**系统性错误**（比如某 part 整体漏分）会让对应 `part_tokens[b, j]` 走 `slot_emb` 兜底或为 0 —— 模型没有视觉信号可依赖。如果数据上 mask 精度低，考虑提高 `use_slot_embedding_fallback` 的容量（扩 `slot_emb` 初始化范围）或在 train loss 里不强制该 part 的监督权重。

### Q4: 如何扩容量？

看 [`part_flow/base.yaml`](../scripts/train/configs/part_flow/base.yaml) 的 `model.hidden_dim` / `model.num_layers`。当前 256 + 4 层 = 7.5M，属于"轻量 velocity 预测器"档位。如果想向 Part Predictor（253M）靠拢：

| 配置 | 参数量 | 瓶颈 |
|---|---|---|
| 256 + 4（当前） | 7.5M | cond 1024→256 压缩比 4×，可能丢视觉细节 |
| 384 + 6 | ~25M | 中档 |
| 512 + 8 | ~70M | 适合刷点数 |
| 768 + 12 | ~200M | 接近 Part Predictor 规模 |

注意：推理 ODE 每步一次 forward，`eval_ode_steps × 模型前向`，扩容量的推理成本是线性涨的。

---

## 6. 代码定位

| 角色 | 文件 | 行号 |
|---|---|---|
| 模型总入口 | `TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py` | [L326+](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L326) |
| Decoder 单层 | 同上 | [L67-180](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L67-L180) |
| `build_part_tokens` | 同上 | [L267-314](../TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py#L267-L314) |
| DINOv2 预编码 | `scripts/data_process/encode_dinov2_mobility.py` | — |
| mask patch-grid 下采样 | `scripts/train/part_predictor/dataset.py` | [L195-255](../scripts/train/part_predictor/dataset.py#L195-L255) |
| Bridge (x_t 采样) | `TRELLIS-arts/trellis/models/part_flow/bridges.py` | — |
| 训练入口 | `scripts/train/part_flow/train_part_flow.py` | — |
| 配置 | `scripts/train/configs/part_flow/base.yaml` | — |
