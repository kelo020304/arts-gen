# Part Predictor — DINOv2 Multi-View 融合与 Query 初始化

本文档详细描述 Part Predictor 中 DINOv2 特征的提取、多视角融合、以及 query 初始化的完整流程。

---

## 0. 概览

```
12 张多视角 RGB 图片
    ↓ [离线] DINOv2-L/14-reg 编码
tokens.npz: [12, 1370, 1024]
    ↓ [训练时] view dropout + 分两路
    │
    ├── 路线 A（全局条件）: flatten → cond_proj → cross-attention KV
    │
    └── 路线 B（query 初始化）: 2D mask × spatial tokens → masked pooling
            → query_init_proj → 初始 part queries
    │
    ↓ Transformer Decoder（12 层，两路信息汇合）
    ↓ mask head + class head
    输出: 每个 voxel 的 part 归属 + part 类别
```

---

## 1. DINOv2 特征提取（离线预编码）

> 代码: `scripts/data_process/encode_dinov2_mobility.py`

### 1.1 模型

使用 **DINOv2-L/14-reg**（Facebook Research），关键参数：

| 参数 | 值 |
|------|------|
| Backbone | ViT-Large |
| Patch size | 14×14 像素 |
| 输入分辨率 | 518×518（= 37 patches × 14 px） |
| 输出 spatial tokens | 37×37 = **1369** 个 |
| CLS token | **1** 个 |
| 总 token 数 | **1370** |
| 每个 token 维度 | **1024** |

### 1.2 图像预处理

对每张渲染图做如下变换：

```python
transforms.Compose([
    # 1. 缩放到 518×518（DINOv2 要求的输入分辨率）
    transforms.Resize((518, 518), interpolation=BICUBIC),
    # 2. 转 tensor: [H,W,3] uint8 → [3,H,W] float32, 值域 [0,1]
    transforms.ToTensor(),
    # 3. ImageNet 标准化
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

> **注意 RGBA 处理**: 渲染图可能有透明通道。加载时先合成到白色背景上再转 RGB：
>
> ```python
> bg = Image.new('RGB', img.size, (255, 255, 255))
> bg.paste(img, mask=img.split()[3])
> ```

### 1.3 前向推理

```python
with torch.no_grad():
    features = model.forward_features(images)  # images: [B, 3, 518, 518]
    patch_tokens = features['x_norm_patchtokens']  # [B, 1369, 1024]
    cls_token    = features['x_norm_clstoken']      # [B, 1024]
    # 拼接: CLS 放在 position 0
    all_tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
    # → [B, 1370, 1024]
```

`forward_features` 返回经过 LayerNorm 的特征（`x_norm_*`），而非原始特征，这样各 token 的 scale 一致。

#### CLS token 与 patch tokens 的区别

DINOv2 的输出由两类 token 组成：

**CLS token**（1 个，`[1024]`）：
- ViT 的特殊 token，不对应图像的任何具体区域
- 在每一层 Transformer 中与所有 patch token 做 self-attention，逐层聚合全图信息
- 最终编码整张图片的**全局语义摘要**——物体类别、整体形状、场景布局等
- 可以理解为"这张图整体是什么"的一个 1024 维向量
- 类比：一段文章的标题/摘要

**Patch tokens**（1369 个，`[1369, 1024]`）：
- 每个 token 对应图像中一个 14×14 像素的 patch
- 排列成 37×37 的网格，保持空间位置关系（第 0 个 = 左上角，第 1368 个 = 右下角）
- 每个 token 编码对应区域的**局部语义**——该 patch 位置的纹理、颜色、边缘、物体部件等
- 经过多层 self-attention 后，每个 patch token 也融合了周围上下文，但仍以局部信息为主
- 类比：文章中每个段落的内容

**在本项目中两者的不同用途**：

| | CLS token (position 0) | Patch tokens (position 1~1369) |
|---|---|---|
| 含义 | 全局语义：整张图"是什么" | 局部语义：每个 14×14 区域"有什么" |
| 空间信息 | 无，不对应任何位置 | 有，37×37 网格保持空间关系 |
| 路线 A（cond cross-attn） | **保留**，和 patch tokens 一起作为 KV。decoder 中 queries 可以 attend 到它获取全局上下文 | **保留**，提供位置相关的 2D 外观细节 |
| 路线 B（masked pooling） | **丢弃**（`tokens[:, 1:, :]`）。CLS 不对应空间位置，无法用 2D mask 选取 | **使用**。用 part mask 在 37×37 网格上选中属于该 part 的 patches，做 mean pooling |

> **为什么路线 B 要丢弃 CLS？**
> Masked pooling 的核心操作是"用 2D mask 选出属于某个 part 的 patch tokens"。CLS token 没有空间位置，无法被 mask 选中或排除——它代表的是整张图而非某个区域。如果把 CLS 也放进 spatial tokens 里做 pooling，会引入与该 part 无关的全局信息，污染 query 初始化。
>
> **为什么路线 A 保留 CLS？**
> 在 decoder cross-attention 中，attention 机制自动学习每个 query 该关注哪些 KV。CLS token 作为一个额外的 KV entry，让 query 在需要全局信息时可以 attend 到它（比如判断"这个 part 在整个物体中的相对位置"），但不会被强制使用。

### 1.4 12 个视角的编码

每个物体的每个关节角度有 12 张渲染图（4 象限 × 3 仰角）：

```
view_0 ~ view_2:   象限 0（方位角 0°~90°，3 个仰角）
view_3 ~ view_5:   象限 1（方位角 90°~180°）
view_6 ~ view_8:   象限 2（方位角 180°~270°）
view_9 ~ view_11:  象限 3（方位角 270°~360°）
```

分 batch 编码（每次 4 张，3 批）：

```python
for batch_start in range(0, 12, 4):
    batch = torch.stack(view_tensors[batch_start:batch_start+4])  # [4, 3, 518, 518]
    tokens = encode_batch(model, batch)  # [4, 1370, 1024]
    all_tokens.append(tokens)
tokens = np.concatenate(all_tokens, axis=0)  # [12, 1370, 1024]
```

### 1.5 存储

```python
np.savez(write_path, tokens=tokens)  # key='tokens', shape=[12, 1370, 1024], float32
```

每个样本约 **66 MB**（12 × 1370 × 1024 × 4 bytes）。不做压缩（NVMe 上 CPU 压缩成本不值得）。

---

## 2. 训练时加载与 View Dropout

> 代码: `scripts/train/part_predictor/dataset.py` — `__getitem__`

### 2.1 加载

```python
tokens_data = np.load(tokens_path)
tokens = torch.from_numpy(tokens_data['tokens']).float()  # [12, 1370, 1024]
```

### 2.2 View Dropout（视角采样）

不是用全部 12 个视角，而是 **每个象限随机选 1 个**，保证 360° 覆盖同时增加多样性：

```python
QUADRANTS = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
selected_views = [random.choice(q) for q in QUADRANTS]  # 4 个视角索引

# 未选中的视角 token 全部清零（保持 tensor shape 不变）
mask_views = torch.zeros(12, dtype=torch.bool)
mask_views[selected_views] = True
tokens[~mask_views] = 0.0
```

**为什么清零而不是去掉？** 保持 `[V*T, D]` shape 固定，方便 batch collate。清零的 token 在 attention 中贡献接近于零（softmax 后权重极小）。

### 2.3 Flatten 为条件向量

```python
cond = tokens.reshape(-1, D)  # [12*1370, 1024] = [16440, 1024]
```

这个 `cond` 会被送入模型，作为 **路线 A**（全局 2D 条件）。

---

## 3. 路线 A — 全局条件 cross-attention

> 代码: `TRELLIS-arts/trellis/models/part_predictor/part_predictor.py`

### 3.1 投影到 query 空间

```python
self.cond_proj = nn.Linear(1024, 256)     # cond_dim → query_dim
cond_proj = self.cond_proj(cond)           # [B, 16440, 256]
cond_feats_packed = cond_proj.reshape(B * VT, -1)  # packed for varlen attn
```

全部 12×1370 个 token（含清零的）投影到 256 维。

### 3.2 在 Decoder 中使用

每层 `PartDecoderLayer` 的第 3 步，part queries 通过 cross-attention attend to 这些 cond tokens：

```python
# queries [sum_K, D] × cond_feats [sum_cond, D] → refined queries
q = self.cond_q(queries)       # query: "我这个 part 需要什么信息？"
k = self.cond_k(cond_feats)    # key: "每个 patch 有什么信息？"
v = self.cond_v(cond_feats)    # value: "对应的特征内容"
out = varlen_attention(q, k, v)
queries = queries + dropout(out)
```

这让每个 part query 能看到全部视角的所有 patch，自动学习从哪些视角、哪些区域提取有用信息。

---

## 4. 路线 B — Masked DINOv2 Pooling → Query 初始化

> 代码: `dataset.py` — `_compute_query_init()`

这是核心的多视角融合初始化流程。目标：为每个 part 生成一个 **语义初始化向量**，而不是用通用的 learnable query 或 GT type embedding。

### 4.1 输入

| 输入 | Shape | 来源 |
|------|-------|------|
| `tokens` | [V=12, T=1370, D=1024] | Step 2 加载的 DINOv2 tokens（已 view dropout） |
| 2D part mask | [H=512, W=512] int32 | GT 渲染的 Object Index mask（训练）/ VLM 预测（推理） |
| `sorted_parts` | list of K parts | part_info.json 中按 label 排序的 part 列表 |
| `selected_views` | list of 4 ints | view dropout 选中的视角索引 |

### 4.2 Step-by-step

#### Step B1: 取 spatial tokens，去掉 CLS

```python
spatial_tokens = tokens[:, 1:, :]   # [12, 1369, 1024]
patch_grid = 37                      # sqrt(1369)
```

CLS token 是全图级别的语义总结，对 part-level 的 masked pooling 没有用，所以去掉。

#### Step B2: 对每个 part、每个选中的视角

```python
for k, part in enumerate(sorted_parts):    # 遍历 K 个 part
    label_val = part['label']               # 这个 part 在 mask 中的整数值
    pooled_list = []

    for v in selected_views:                # 遍历 4 个选中的视角
```

#### Step B3: 加载 2D mask 并提取该 part 的区域

```python
mask_2d = np.load(mask_path)                      # [512, 512] int32
binary = (mask_2d == label_val).astype(np.float32) # [512, 512] 0/1
```

mask 中，0 是背景，其他整数值标记不同 part。取出当前 part 对应的二值 mask。

#### Step B4: 下采样 mask 到 DINOv2 patch 网格

```python
patch_mask = F.interpolate(
    binary[None, None],             # [1, 1, 512, 512]
    size=(37, 37),                  # 下采样到 patch 网格大小
    mode='nearest',                 # 最近邻，保持二值性
).squeeze().flatten().bool()        # [1369] bool
```

**为什么是 37×37？** DINOv2-L/14 把 518×518 的图切成 37×37 = 1369 个 patch。每个 patch 覆盖原图 14×14 像素的区域。mask 需要下采样到同样的网格，才能知道哪些 patch 属于这个 part。

**下采样的具体行为（nearest 模式）：**

注意这里是对**单个 part 的二值 mask** 做下采样，不是对多 label 的原始 mask 做下采样。流程是：

1. 先提取：`binary = (mask_2d == label_val)` → 512×512 的 0/1 图（只有当前 part 是 1）
2. 再下采样：`F.interpolate(binary, size=(37, 37), mode='nearest')` → 37×37 的 0/1 图

nearest 下采样的含义：对于输出 37×37 中的每个位置 (i, j)，回到 512×512 上找最近邻的像素值。如果该像素恰好属于当前 part → 1，否则 → 0。

**一个 patch 内有多个 part 的情况：**

实际场景中，一个 14×14 像素的 patch 内很可能同时包含多个 part（比如 door 和 handle 的边界区域）：

```
原图 512×512 局部             对应的 1 个 DINOv2 patch (14×14 px)
┌────────────┐
│ 2 2 2 2 2 2│  2=door
│ 2 2 3 3 3 3│  3=handle        ← 这个 patch 同时覆盖了 door 和 handle
│ 2 2 3 3 3 3│
│ 2 2 2 2 2 2│
└────────────┘
```

因为我们是对每个 part **独立做二值 mask 再下采样**，所以不存在"多个 label 竞争"的问题：

- 处理 door (label=2) 时：binary 在这个 patch 区域内有 1 也有 0，nearest 采样到一个值：
  - 采到 1 → 这个 patch 被认为属于 door → door 的 pooling 会包含这个 patch 的 token
  - 采到 0 → 这个 patch 不算 door 的
- 处理 handle (label=3) 时：同理，独立判断

**结果：同一个 patch token 可以同时被 door 和 handle 的 pooling 选中。** 这在语义上是合理的——边界处的 patch token 本身就同时编码了两个 part 的视觉信息（DINOv2 的感受野远大于单个 patch），所以两个 part 的 query 初始化都包含它的贡献是正确的。

**nearest 采样的不精确性：**

nearest 模式下，37×37 中每个位置只看 512×512 上的一个采样点。即使某个 patch 覆盖的 14×14 区域内大部分是 door，但如果采样点恰好落在 handle 的像素上，door 的 mask 就会漏掉这个 patch。反过来也一样。

这种不精确在实践中问题不大，原因：
1. **边界 patch 数量少**——大多数 patch 完全在一个 part 内部，不受影响
2. **有多个视角**——一个视角漏掉的边界 patch，其他视角可能会覆盖到
3. **pooling 是 mean**——少一两个边界 patch 对平均值影响很小
4. **query 初始化只是起点**——后续 12 层 decoder 会通过 cross-attention 继续精炼

> 如果追求更精确，可以用 `mode='area'`（面积平均）代替 `nearest`，得到 0~1 的连续值，再用阈值（如 >0.3）判断归属。但当前实现用 nearest 已足够。

#### Step B5: Masked mean pooling

```python
if patch_mask.sum() > 0:
    pooled = spatial_tokens[v, patch_mask].mean(dim=0)  # [1024]
    pooled_list.append(pooled)
```

**`spatial_tokens[v]` 的结构：**

`spatial_tokens = tokens[:, 1:, :]` 去掉 CLS 后，每个视角剩下 1369 个 token，逻辑上排列成 37×37 的网格。每个 token 是一个 1024 维向量，编码了对应 patch 位置的视觉语义。

以视角 v 为例，假设图中有 door（label=2）和 handle（label=3）：

```
视角 v 的渲染图 (518×518)                spatial_tokens[v] (37×37 = 1369 个 token)
┌─────────────────────────┐              每个格子 = 1 个 1024 维向量
│                         │
│    ┌──────────┐         │              token 排列 (37×37 网格):
│    │          │         │              ┌──┬──┬──┬──┬──┬──┬──┐
│    │  door    │         │              │bg│bg│bg│bg│bg│bg│bg│  bg = 背景区域的 token
│    │ (label=2)│         │              ├──┼──┼──┼──┼──┼──┼──┤      编码: 白色/空背景
│    │   ┌───┐  │         │              │bg│d │d │d │d │bg│bg│
│    │   │hdl│  │         │              ├──┼──┼──┼──┼──┼──┼──┤  d  = door 区域的 token
│    │   └───┘  │         │              │bg│d │d │d │d │bg│bg│      编码: 门板的纹理/颜色/形状
│    │          │         │              ├──┼──┼──┼──┼──┼──┼──┤
│    └──────────┘         │              │bg│d │d │dh│dh│bg│bg│  h  = handle 区域的 token
│                         │              ├──┼──┼──┼──┼──┼──┼──┤      编码: 把手的金属质感/形状
└─────────────────────────┘              │bg│d │d │dh│h │bg│bg│
                                         ├──┼──┼──┼──┼──┼──┼──┤  dh = 边界 patch, 同时包含
 每个 patch 覆盖 14×14 像素               │bg│bg│bg│bg│bg│bg│bg│      door 和 handle 的像素
                                         └──┴──┴──┴──┴──┴──┴──┘      → token 编码两者混合语义
                                         (实际是 37×37，这里简化画 7×7)
```

**关键理解**: 每个 token 不是简单的像素颜色——经过 ViT 212 层 self-attention 后，每个 token 融合了大范围上下文。一个 door 区域的 token 不仅编码了自己 14×14 像素内的纹理，还"知道"周围有 handle、整体是个柜子等信息。但它的主要语义仍以自身 patch 位置的内容为主。

**`spatial_tokens[v, patch_mask]` 的选取过程：**

当处理 door (label=2) 时，`patch_mask` 标记了上图中所有 `d` 和 `dh` 位置为 True：

```
patch_mask (37×37 → flatten 为 [1369] bool):

F F F F F F F
F T T T T F F       T = True (属于 door)
F T T T T F F       F = False (不属于 door)
F T T T T F F       注意 dh 位置也可能被选中
F T T T T F F       (取决于 nearest 采样点)
F F F F F F F
```

`spatial_tokens[v, patch_mask]` 取出所有 True 位置的 token → `[P, 1024]`（P ≈ 上图中 T 的个数）：

```
spatial_tokens[v, patch_mask]:

┌──────────────────┐
│ token_d1  [1024] │  ← door 左上 patch 的特征
│ token_d2  [1024] │  ← door 右侧 patch 的特征
│ token_d3  [1024] │  ← ...
│ ...              │
│ token_dh1 [1024] │  ← door/handle 边界 patch（混合语义）
│ token_dP  [1024] │
└──────────────────┘
shape: [P, 1024]，P = door 命中的 patch 数
```

**`.mean(dim=0)` → `[1024]`：**

对 P 个 token 逐维平均，得到一个 1024 维向量——door 在这个视角下的"平均视觉语义"：

```
pooled = mean([token_d1, token_d2, ..., token_dP])  → [1024]

这个向量大致编码了:
- door 的主要颜色/纹理（木纹？白色？玻璃？）
- door 的大致形状（方形？圆弧？）
- door 在图中的相对位置和大小
- （弱）周围上下文（旁边有 handle、属于柜子的一部分）
```

**含义**: 这个 1024 维向量编码了"这个 part 在视角 v 下的视觉外观"——形状、纹理、颜色的混合语义。

> 如果该 part 在这个视角完全不可见（patch_mask 全 False），就跳过，不贡献。

#### Step B6: 跨视角平均

```python
if pooled_list:
    query_init[k] = torch.stack(pooled_list).mean(dim=0)  # [1024]
# else: 保持 zero vector（极少见——该 part 在所有视角都不可见）
```

把所有可见视角的 pooled 向量取平均。通常一个 part 在 2~4 个视角可见。

**最终输出**: `query_init` shape = `[K, 1024]`，每行是一个 part 的多视角融合 DINOv2 语义表示。

### 4.3 投影到 query 空间

在模型的 `forward()` 中：

```python
self.query_init_proj = nn.Linear(1024, 256)  # 可学习投影
q_b = self.query_init_proj(qi_b)              # [K, 1024] → [K, 256]
```

从 DINOv2 的 1024 维投影到 Transformer Decoder 的 query 维度 256。这个投影层是可学习的，会随训练调整。

---

## 5. Query 初始化的三种模式

模型支持三种 query 初始化方式，按优先级排列：

| 优先级 | 方式 | 输入 | 适用场景 |
|--------|------|------|----------|
| 1（最高） | **Masked DINOv2 pooling** | 2D part masks + DINOv2 tokens | 正常训练和推理 |
| 2 | **GT type embedding** | part_type_ids → nn.Embedding | 消融实验（泄漏 GT 信息） |
| 3（兜底） | **Learnable fallback** | nn.Parameter [max_K, 256] | 无 mask 时的备用 |

```python
if query_init is not None:          # 优先级 1: masked pooling
    q_b = self.query_init_proj(qi_b)
elif part_type_ids is not None:     # 优先级 2: GT type embedding
    q_b = self.part_type_embed(ids_b)
else:                               # 优先级 3: learnable fallback
    q_b = self.fallback_queries[:K_b]
```

---

## 6. 进入 Transformer Decoder

初始化完的 queries 经过 **12 层 PartDecoderLayer**，每层包含：

```
Input: queries [sum_K, 256]
  │
  ├─ (1) Self-Attention: K 个 queries 之间互相交互
  │      → 学习 part 间关系（"门把手在门旁边"）
  │
  ├─ (2) Cross-Attention → voxel features [sum_N, 256]
  │      → voxel features 来自坐标的位置编码 MLP(xyz / 64.0)
  │      → 学习"每个 query 应该关注哪些 3D 位置"
  │
  ├─ (3) Cross-Attention → DINOv2 cond tokens [sum_VT, 256]
  │      → 路线 A 的全量 tokens，补充 2D 视觉上下文
  │      → 学习"从哪些视角的哪些区域获取额外信息"
  │
  └─ (4) FFN: 非线性变换 (Linear → GELU → Linear)

Output: refined queries [sum_K, 256]
```

12 层之后，每个 query 已经充分融合了：
- 自身的 DINOv2 语义初始化（Step 4）
- 其他 part 的上下文（self-attention）
- 3D 空间位置信息（voxel cross-attention）
- 多视角 2D 外观细节（cond cross-attention）

---

## 7. 最终输出

```python
# Mask prediction: dot product between queries and voxel features
mask_logits = einsum('kd,nd->kn', queries, voxel_feats) / sqrt(256)  # [K, N]
soft_masks  = softmax(mask_logits, dim=0)  # 每个 voxel 上 K 个 part 的概率分布

# Class prediction: query → part type
class_logits = class_head(class_norm(queries))  # [K, 33]  (32 types + 1 "other")
```

---

## 8. 数据流维度速查表

| 阶段 | 数据 | Shape | 说明 |
|------|------|-------|------|
| 渲染图 | RGB | [512, 512, 3] | Blender 渲染 |
| 预处理 | 归一化图 | [3, 518, 518] | Resize + ImageNet norm |
| DINOv2 输出 | patch tokens | [1369, 1024] | 37×37 spatial |
| DINOv2 输出 | CLS token | [1024] | 全局语义 |
| 拼接 | all tokens | [1370, 1024] | CLS + spatial |
| 12 视角 | tokens.npz | **[12, 1370, 1024]** | 存储到磁盘 |
| View dropout | 活跃 tokens | [12, 1370, 1024] | 8 个视角清零 |
| Flatten | cond | [16440, 1024] | 路线 A 输入 |
| cond_proj | cond_feats | [16440, 256] | 模型内部 |
| 2D mask | binary | [512, 512] | 每 part 每视角 |
| 下采样 mask | patch_mask | [1369] bool | 37×37 flatten |
| Masked pool | per-view | [1024] | 1 个 part 1 个视角 |
| 跨视角平均 | per-part | [1024] | 1 个 part |
| 全部 parts | query_init | **[K, 1024]** | K 个 part |
| 投影 | queries | **[K, 256]** | 进入 decoder |
