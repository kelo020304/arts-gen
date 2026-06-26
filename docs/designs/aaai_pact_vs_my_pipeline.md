# E2E Articulated Asset Reconstruction：论文框架与实验设计

> 端到端系统：Fine-tuned VLM → 多视角重建（per-part decode）→ 仿真资产生成。本文档聚焦重建模块的技术方案及其在系统中的角色。

---

## 1. 系统全景

```
Wild Image(s)
    │
    ▼
┌───────────────────┐
│  Fine-tuned Qwen   │  fine-tune 后具备铰链物体的 part/joint 推理能力
│  (VLM 前端)        │  输出：part 数量、语义类别、joint 类型/轴向、part 拓扑关系
└────────┬──────────┘
         │  structured part/joint prior
         ▼
┌───────────────────────────────────────────────────────────────┐
│  TRELLIS-based Articulated Reconstruction                      │
│                                                                │
│  Stage 1-2 (frozen): SS-VAE encoder                            │
│       │  多视角图像 → structure latent → 64³ sparse active voxels│
│       ▼                                                        │
│  Stage 3 (fine-tuned for multi-view): SLat Flow Model          │
│       │  多视角 DINOv2 cross-attention conditioning              │
│       │  flow matching → detailed structured latent             │
│       ▼                                                        │
│  ★ Part Predictor (全新独立模型): Query-based Transformer       │
│       │  在 active voxels 上预测 per-voxel part label           │
│       │  可接收 VLM prior 作为 condition                        │
│       ▼                                                        │
│  按 part label 拆分 structured latent                           │
│       ▼                                                        │
│  Stage 4 (fine-tuned for multi-view): per-part decode          │
│       │  每个 part 的 latent 子集 → 独立 decode → per-part mesh  │
│       │  所有 part 在同一 voxel grid 坐标系 → 尺度天然一致       │
│       ▼                                                        │
└───────┬───────────────────────────────────────────────────────┘
        │  per-part mesh + part labels + joint info
        ▼
┌───────────────────┐
│  仿真后端          │  per-part mesh + joint info → URDF/MJCF → Isaac Sim / MuJoCo
└───────────────────┘
```

---

## 2. 各模块的角色与训练策略

### 2.1 四个模块，各司其职

| 模块 | 职责 | 训练策略 | 与 part 的关系 |
|---|---|---|---|
| **Stage 1-2 (SS-VAE)** | 多视角 → structure latent → 哪些 voxel 是 active | **Frozen** | 无关，通用 3D structure |
| **Stage 3 (SLat Flow)** | Flow matching 生成 detailed structured latent | **Fine-tune for multi-view** | 无关，只是从 single-view 改为 multi-view conditioning |
| **Part Predictor** | 在 active voxels 上预测 per-voxel part label | **全新训练** | **核心 part 模块** |
| **Stage 4 (Decoder)** | Structured latent → mesh geometry | **Fine-tune for multi-view** | 间接相关：接收 per-part 拆分后的 latent 子集 |

**关键澄清**：
- Stage 3 和 Stage 4 的 fine-tune 目的是 **适配多视角输入**，不是为了 part
- Part 分解由 **独立的 Part Predictor** 完成，它是 Stage 3 和 Stage 4 之间的全新模型
- Per-part decode 是在 Part Predictor 输出 label 之后，用 label 拆分 Stage 3 的 structured latent，再分别送 Stage 4

### 2.2 数据流详解

```
多视角图像 x_mv (4 views)
    │
    ├──→ DINOv2 encode → multi-view tokens [4, 1370, 1024]
    │                          │
    │                          │ (cross-attention condition)
    │                          ▼
    └──→ Stage 1-2 (frozen) → sparse active voxels (64³)
                │                      │
                │                      ▼
                │              Stage 3 (fine-tuned MV)
                │              ├── input: structure latent + noise
                │              ├── condition: multi-view DINOv2 tokens (cross-attn)
                │              └── output: detailed structured latent (per active voxel)
                │                      │
                │                      ▼
                │              Part Predictor (Query-based Transformer)
                │              ├── input: active voxel features (from Stage 3 output 或 Stage 2)
                │              ├── optional condition: VLM part/joint prior
                │              └── output: per-voxel part label
                │                      │
                │                      ▼
                │              按 part label 拆分 structured latent
                │              ├── part_0: voxel 子集 + latent 子集
                │              ├── part_1: voxel 子集 + latent 子集
                │              └── ...
                │                      │
                │                      ▼
                │              Stage 4 (fine-tuned MV): 每个 part 独立 decode
                │              ├── part_0 latent → mesh_0
                │              ├── part_1 latent → mesh_1
                │              └── ...（同一坐标系，尺度一致）
                │                      │
                │                      ▼
                └──────────→  per-part meshes → 组装 → URDF
```

---

## 3. 三个核心技术点

### 3.1 Stage 3 Multi-view Fine-tune

**目标**：让原本 single-view conditioned 的 SLat Flow Model 学会利用多视角约束。

**做法**：将 4 个视角的 DINOv2 tokens concat 作为 cross-attention 的 KV（参考 TRELLIS 2.0 的做法），fine-tune Stage 3 的 cross-attention layers。

**技术点**：
- 不是简单的多图 feature 拼接——flow matching 去噪过程需要学会利用多视角几何一致性
- View Dropout：训练时随机 drop 1-3 个视角，增强鲁棒性
- LoRA 微调：只改 cross-attention 的 Q/K/V projections，保留原有 3D prior

**为什么不改 Stage 1-2**：SS-VAE 已经能生成足够好的 sparse structure（"哪些 voxel active"），多视角的增量主要体现在 detailed latent 的质量上，这是 Stage 3 的职责。

### 3.2 Part Predictor（全新独立模型）

**架构**：Query-based Transformer Decoder + 可选 decode-aware loss。详细设计见 [phase2_part_predictor_plan.md](docs/archive/phase02/part_predictor_plan.md)。

**核心设计**：
- **Query-based**：K 个 part queries（K 由 VLM/GT 给出，可变），通过 cross-attention 从 voxel features 聚合信息，输出 per-query soft mask
- **Query 初始化**：用 VLM/GT 的 part class embedding 初始化 queries，语义对齐且天然支持可变 part 数量
- **Decode-aware loss（可选）**：Frozen Stage 4 作为 evaluator，decode quality 梯度回传到 Part Predictor。通过 YAML `decode_aware.enabled` 开关，两版代码对比实验
- **独立模块**：不干扰 Stage 3 的 Flow Model 质量；训练和消融完全解耦
- **只对 active voxel**：非 active voxel 无需 part label

**与标准 3D segmentation 的区别**：
- 不是 per-voxel classification（UNet 式），而是 instance-level query prediction
- Loss 不只有 mask CE，还有下游 decode quality 信号（decode-aware 版本）
- Query 数量由上游 VLM 动态决定，不是固定类别数

**输入选择**（通过实验确定）：
- 来自 Stage 3 输出的 structured latent features
- 或来自 Stage 2 的 structure features

### 3.3 Per-part Decode（Stage 4）

**做法**：Part Predictor 输出 label 后，按 label 将 Stage 3 的 structured latent 拆分为多个子集，每个子集独立送 fine-tuned Stage 4 decode。

**尺度一致性**：所有 part 的 voxel 来自同一个 64³ grid，坐标是全局的。即使独立 decode，空间位置和尺度天然一致，不需要 post-hoc alignment。

**Stage 4 fine-tune 的目标是多视角适配**，不是为了 per-part。但 fine-tune 后的 decoder 需要能处理 voxel 子集输入——这是一个需要验证的假设（P0）。

---

## 4. 与 PAct 的核心区别

### 4.1 改动位置对比

```
PAct:
  Stage 1 (改): part-aware latent dynamics → 生成过程本身变成 part-aware
  Stage 2-4:    基本不改
  结果:         backbone 被改写，part 信息隐式编码在 latent 中

Ours:
  Stage 1-2 (不改): frozen，保持通用 3D structure prior
  Stage 3 (fine-tune): 目的是多视角 conditioning，不是 part
  ★ Part Predictor (全新): 独立模型，在 3D voxel 空间显式预测 part label
  Stage 4 (fine-tune): 目的是多视角适配，per-part decode 是下游使用方式
  结果:         part 信息是显式的 3D label，由独立模型预测
```

### 4.2 对比表

| 维度 | PAct | Ours |
|---|---|---|
| Part-aware 改动位置 | Stage 1 生成动力学 | Stage 3→4 之间的独立模型 |
| 生成过程是否 part-aware | 是，backbone 被改写 | 否，Stage 1-3 都不 part-aware |
| Part 信息性质 | 隐式，编码在 latent dynamics 中 | **显式，3D voxel 空间上的 label** |
| Part 信息来源 | 生成过程内部涌现 | **独立模型显式预测** |
| 输入 | 单图 | 多视角（Stage 3/4 fine-tuned） |
| Decode 方式 | 整体 decode | **Per-part decode（按 label 拆分后独立 decode）** |
| 尺度一致性 | 需要后处理 | **天然保持（同一 voxel grid）** |
| Part Predictor 可替换/升级 | 不可（耦合在 backbone 中） | **可（独立模块）** |

### 4.3 设计哲学

**PAct**：让生成模型本身学会 "生成 part-decomposed 的铰链物体"。Part awareness 从生成过程中涌现。

**Ours**：生成模型只负责 "生成高质量 3D structure"（Stage 1-2 通用 prior + Stage 3 多视角 fine-tune）。Part 分解由一个**独立的 3D segmentation 模型**在生成结果上显式完成，然后按 part 拆分 decode。

这不是高低之分。区别在于：
- PAct 更适合 "从零生成铰链物体"（generation）
- Ours 更适合 "从多视角观测重建 + 接仿真"（reconstruction → simulation），因为：
  - 多视角提供更强观测约束
  - 显式 part label 可检查、可监督、可接仿真
  - 尺度一致性对 URDF 生成至关重要

---

## 5. Technical Contributions（重建模块部分）

### TC1: Multi-view Adapted SLat Flow

Fine-tune TRELLIS Stage 3 从 single-view 扩展为 multi-view conditioning（4-view DINOv2 tokens concat as cross-attention KV + view dropout）。让 flow matching 去噪过程学会利用多视角几何一致性生成更准确的 structured latent。

### TC2: Query-based Decode-aware Part Prediction

在 Stage 3 和 Stage 4 之间引入 query-based part predictor。K 个 part queries（K 由 VLM 动态给出）通过 Transformer Decoder 的 cross-attention 从 3D voxel features 聚合信息，输出 instance-level part masks。可选 decode-aware loss：frozen Stage 4 作为 decode quality evaluator，梯度回传使分割对下游 per-part decode 最优——Part Predictor 学会预测 "Stage 4 能 decode 好的分割"，而非只 "几何上正确的分割"。这是一个只在本 pipeline 中存在的问题和方案。

### TC3: Per-part Decode with Scale Consistency

利用 Part Predictor 的 label 拆分 structured latent，按 part 独立送 Stage 4 decode。所有 part 在同一 64³ voxel grid 的坐标系下 decode，尺度和位置天然一致，可直接组装为仿真资产。

### TC4: VLM-guided Structured Prior

Fine-tune Qwen 使其输出 structured part/joint prior（数量、类别、拓扑、轴向），作为 Part Predictor 的 condition，桥接 2D 语义理解和 3D 结构分解。

---

## 6. 实验设计

### 6.1 对比模型

| 模型 | 定义 | 验证什么 |
|---|---|---|
| **Base TRELLIS (SV)** | 原始 TRELLIS single-view，无 part | 原始基线 |
| **Base TRELLIS (MV)** | Stage 3/4 fine-tuned for multi-view，无 part | 隔离多视角 fine-tune 的增量 |
| **Entangled Baseline** | Part-aware bias 写进 Stage 1（模拟 PAct） | PAct 方法论对比 |
| **Ours w/o VLM** | Full pipeline，Part Predictor 无 VLM condition | 纯几何 part prediction |
| **Ours w/ VLM** | Full pipeline + VLM condition | 完整方案 |

### 6.2 Group A: 重建质量

**数据**：PartNet-Mobility 测试集 + 小米汽车零件

| 指标 | 衡量什么 |
|---|---|
| Voxel IoU | 整体结构准确性 |
| Part Label mIoU | Part 分解准确性 |
| Per-part Chamfer Distance | 单 part 几何精度 |
| Per-part F-Score | 单 part 表面质量 |
| Part Count Accuracy | 部件数量正确率 |

**关键对比**：
- Base TRELLIS (MV) vs (SV)：多视角 fine-tune 的增量
- Ours vs Base TRELLIS (MV)：Part Predictor + per-part decode 的增量
- Ours vs Entangled Baseline：独立 Part Predictor vs 写进 backbone 的 part awareness
- Ours w/ VLM vs w/o VLM：VLM prior 的增量

### 6.3 Group B: 尺度一致性与可组装性

| 指标 | 衡量什么 |
|---|---|
| Inter-part scale consistency | 各 part 尺度一致性 |
| Assembly gap/overlap | 组装后交界处间隙/重叠 |
| Joint alignment error | Joint 位置/轴向对齐误差 |

**期望**：Ours 天然尺度一致（同一 voxel grid），Entangled Baseline 需要后处理对齐。

### 6.4 Group C: 端到端仿真验证

| 指标 | 衡量什么 |
|---|---|
| URDF 生成成功率 | per-part mesh → 合法 URDF |
| 关节运动范围误差 | 与 GT 关节参数偏差 |
| 仿真稳定性 | Isaac Sim 中无穿模/爆炸 |
| 操作任务成功率 | 开关门/抽屉等任务 |

### 6.5 Ablation

| 实验 | 验证什么 |
|---|---|
| Per-part decode vs 整体 decode + 后处理拆分 | Per-part decode 的必要性 |
| Part Predictor: query-based vs UNet baseline | Query 架构的优势 |
| Part Predictor: +decode-aware loss vs 不加 | Decode quality 监督的增量 |
| Part Predictor: VLM query init vs learnable query init | VLM 语义对齐的增量 |
| Part Predictor 输入: Stage 3 output vs Stage 2 output | 哪个 feature 更适合 |
| Part Predictor: 独立模块 vs Stage 3 dual-head | 解耦设计的优势 |
| Boundary voxel assignment 可视化 | decode-aware 改变了哪些 boundary voxel 的 assignment |
| VLM condition: 有 vs 无 vs GT part info | VLM prior 的质量和上限 |
| 视角数量: 1 / 2 / 4 views | 多视角边际收益 |
| Stage 3: LoRA vs full fine-tune vs frozen | 多视角适配策略 |

### 6.6 小米数据专项

| 实验 | 目的 |
|---|---|
| PartNet-Mobility 训练 → 小米 zero-shot | 泛化能力 |
| 混合训练 | 工业数据兼容性 |
| 端到端 demo | 工业场景 showcase |

---

## 7. 核心假设（按验证优先级）

### P0: Per-part decode 可行性
> Fine-tuned Stage 4 decoder 对 structured latent 的 part 子集 decode 出合理几何，各 part 尺度一致。

方案地基。最早验证：用 GT part label 拆分 GT latent → Stage 4 decode → 检查质量。

### P1: Part Predictor 准确性
> Sparse 3D UNet 在 active voxels 上达到足够高的 part label mIoU。

### P2: Stage 3 多视角 fine-tune 有效性
> Multi-view conditioned Stage 3 生成的 structured latent 显著优于 single-view。

### P3: VLM prior 增量
> Fine-tuned Qwen 的 part/joint prior 显著提升 Part Predictor。

### P4: 端到端仿真可用性
> 从 wild image 到仿真中正常运作的铰链物体资产。

---

## 8. 训练方案

### 8.1 损失函数

| Loss | 作用域 | 说明 |
|---|---|---|
| $\mathcal{L}_{flow}$ | Stage 3 fine-tune | Multi-view conditioned flow matching loss |
| $\mathcal{L}_{part}$ | Part Predictor | Hungarian matching + mask CE + dice + class CE + 可选 decode-aware Chamfer |
| $\mathcal{L}_{decode}$ | Stage 4 fine-tune | Multi-view conditioned decode loss（与 part 无关） |

### 8.2 训练阶段（来自 ROADMAP.md）

| Phase | 做什么 | 硬件 |
|---|---|---|
| **Phase 1** | 训练基础设施：OmegaConf + Wandb + 数据验证 + LoRA | RTX 4090 |
| **Phase 2** | Stage 3 多视角 fine-tune（LoRA on cross-attention） | RTX 4090 smoke → H200 |
| **Phase 3** | Part Predictor 训练 + Stage 4 多视角 fine-tune + per-part decode | RTX 4090 smoke → H200 |
| **Phase 4** | 推理管线 + 评估系统 | RTX 4090 |
| **Phase 5** | 消融实验 | H200 |

---

## 9. 已识别风险（来自 PITFALLS.md）

### FATAL 级

| 风险 | 应对 |
|---|---|
| 小数据集过拟合（~2000 物体） | LoRA + early stopping + N-part = N 样本的数据增量 |
| 坐标系错位（Y-up vs Z-up） | 训练前自动验证脚本 |
| Flow matching timestep sampling 不当 | Curriculum Sampling |
| P0 不成立（per-part decode 失败） | 1) context voxel 填充；2) 退化为整体 decode + 后处理拆分 |

### SEVERE 级

| 风险 | 应对 |
|---|---|
| Part Predictor class imbalance（背景 voxel 多） | 只对 active voxel 预测 + Focal Loss |
| Part boundary 歧义 | Boundary-aware loss |
| GPU 显存不足（550M DiT + multi-view） | Gradient checkpointing |
| DDP 同步 bug | no_sync() + torchrun |

---

## 10. 论文材料清单（重建模块需提供）

| 材料 | 用途 |
|---|---|
| Pipeline 全景图（Stage 1-2 → Stage 3 MV → Part Predictor → Per-part Stage 4） | Method Figure |
| Per-part decode 可视化（各 part 独立 + 组装后） | 核心 Figure |
| Multi-view vs single-view 的重建质量对比 | Stage 3 fine-tune 效果 |
| Part label 在 3D voxel 上的可视化 | Part Predictor 效果 |
| 与 entangled baseline 的指标对比表 | 主 Table |
| 尺度一致性对比（Ours vs 后处理对齐） | 关键 Figure |
| VLM condition 有/无对比 | Ablation |
| 小米零件重建 demo | 工业场景 Figure |
| 失败案例分析 | Limitation |
| 端到端 demo（image → recon → sim） | 系统 Figure + 补充视频 |
