# Data Spec: Part Flow surface part labeling 数据契约

> **For data team + 模型组.** Phase 8/9 Part Flow 数据契约（manifest-driven，2026-05-07 修订）。
> 联系人：模型组。
> 状态：v2，已与数据同学对齐。
>
> **重命名说明**（2026-05-07）：本文件由 v1 的 `part_labels_solid_64.md` 重命名而来。
> v1 草案设计目标是 dense solid（实心填充）GT，文件名沿用 "solid" 一词；**v2 修正
> 后实际契约是 surface 含内壁的 part labeling**，旧名字会持续误导后续维护者，
> 因此随 v2 一起改名 `part_labels_solid_64.md` → `part_labels_surface_64.md`。
> 历史归档（`.gsd/`、`code_update/`、`code_review/`）里的旧名字保持不动（frozen
> archive），代码 / 配置 / 测试 / 文档主路径全部已更新为新文件名。
> 文件 `part_labels_solid_64.npy` 不再由数据组离线生成，由 PartFlowDataset 在线
> 合成（见 §3）。

## TL;DR

每个 `(object_id, angle_idx)` 样本，`reconstruction/` 下数据组需要 ship 的
文件：

```
reconstruction/voxel_expanded/{oid}/angle_{N}/64/surface.npy        ← 整体外壳 + 内壁的并集
reconstruction/voxel_expanded/{oid}/angle_{N}/64/ind_<target>_<i>.npy  ← 每个 manifest target 部件的 surface (含内壁)
reconstruction/part_info/{oid}/part_info.json                       ← 部件元信息
reconstruction/dinov2_tokens/{oid}/angle_{N}/tokens.npz             ← DINOv2 12 视图 tokens
renders/{oid}/angle_{N}/mask/mask_{v}.npy                           ← manifest view_indices 对应的 2D part mask
renders/{oid}/angle_{N}/rgb/view_{v}.png                            ← 同上 4 视图 RGB（仅 inspect / inference 用）
manifests/part_completion/{name}.train.jsonl                        ← 每行一个 sample 的训练入口
```

非 manifest target 的部件（如 `base_body_0`、`handle_0`）**有意不输出独立
ind 文件**，由 dataset 在线合并到 body slot。

数据组**不再需要**离线生成 `part_labels_solid_64.npy` —— 该 supervision tensor
由 PartFlowDataset.__getitem__ 在线合成（见 §3）。

---

## 1. 文件契约

### 1.1 `surface.npy`

| 字段 | 值 |
|---|---|
| **Path** | `voxel_expanded/{obj_id}/angle_{N}/64/surface.npy` |
| **Shape** | `[N_surface, 3]` 稀疏坐标 |
| **Dtype** | `int32` 或 `int64` |
| **取值范围** | 每行 `[x, y, z]`，每个分量 `0 ≤ v < 64` |
| **语义** | 该 obj 在该 angle 下，**整体物体 mesh surface（含内壁）**占据的 voxel 坐标。所有 part（target + 非 target）surface 的并集。 |
| **典型大小** | 64³ 下 1k–15k voxels（100075=3656，100368=13541） |

注：v1 spec 误用"外表面（一层壳）"措辞，v2 修正为"整体 surface（含内壁）"。
对薄壳/空腔几何（如抽屉、盒子），TRELLIS 原始 watertight surface 体素化只
留外壁；当前数据组 04_voxelize 输出**含内壁**（drawer 内壁、盒子内表面），
这就是 surface dropout + 监督预测能成立的几何来源。

### 1.2 `ind_<part>_<i>.npy`（仅 manifest target 部件）

| 字段 | 值 |
|---|---|
| **Path** | `voxel_expanded/{obj_id}/angle_{N}/64/ind_<part_name>_<inst>.npy` |
| **Shape** | `[N_part, 3]` 稀疏坐标 |
| **Dtype** | `int32` 或 `int64` |
| **取值** | 该 part mesh 自己的 surface 体素化（含内壁），voxel 范围 `[0, 64)` |
| **语义** | 单个 part 的 surface voxel 集合。`union(全部 ind_*) ⊆ surface.npy`（理论上等于，但 04_voxelize 的 unique-vstack 让两者一致）。 |
| **覆盖范围** | **只对 manifest 的 `target_part_names` 输出**。非 target 部件（base_body / 某些 handle）有意省略，由 dataset 现场算 body slot 几何。 |

### 1.3 `part_info.json`

raw label 必须 contiguous `1..num_parts`（由 `04_voxelize.load_part_specs:
expected_label = part_index + 1` 强制）。PartFlowDataset 在线 assert 此
invariant，drift 时 fail loud。

### 1.4 `mask_{v}.npy`（per-view 2D 标注）

| 字段 | 值 |
|---|---|
| **Path** | `renders/{oid}/angle_{N}/mask/mask_{v}.npy` |
| **Shape** | `[H, W]`（典型 [512, 512]） |
| **Dtype** | `int32` |
| **取值** | `0 = bg / 非物体`，`1..num_parts = part_info raw label` |
| **覆盖范围** | manifest `view_indices` 对应的 4 个视图必须都在；其余视图**可省略**（数据组 curate 时只 ship 这 4 个，节省体积） |

注：manifest 行里的 `mask_rule: "remap target original labels to local 1..K;
all other labels become 0"` 是 OmniPart **任务运行时** 的 remap 指令，**不是**
mask 文件的 on-disk 语义。on-disk mask 始终是 part_info raw label。

### 1.5 `dinov2_tokens.npz`

| 字段 | 值 |
|---|---|
| **Path** | `reconstruction/dinov2_tokens/{oid}/angle_{N}/tokens.npz` |
| **Key** | `tokens` |
| **Shape** | `[12, T, D]`，T=1370 (1 CLS + 1369 patches，DINOv2-L/14 在 518² 输入下) |
| **Dtype** | `float32` |
| **覆盖范围** | **12 视图全在**（不是只 manifest 的 4 个），dataset 在 __getitem__ 里按 manifest view_indices 切片。 |

### 1.6 manifest jsonl（per-sample 训练入口）

每行一个 sample，必备字段：

```json
{
  "sample_id": "physx-mobility_100075_angle_0",
  "object_id": "100075",
  "angle_idx": 0,
  "target_part_names": ["wheel_0", "wheel_1"],
  "view_indices": [2, 3, 8, 10],
  ...
}
```

`view_indices` 必须长度 4、唯一、≥0；`target_part_names` 必须非空、唯一、所有
名字都在 `part_info["parts"]` 里；不满足任何一条 PartFlowDataset 直接 raise。

`view_indices` 数据组按"每象限 1 个 + 目标可见"挑选，PartFlowDataset 训练时
deterministic 用这 4 个（不再随机 quadrant pick），保证 mask + DINOv2 + RGB
完全对应。

---

## 2. 监督 supervision slot 设计（per-sample local，body 合并）

PartFlowDataset 现场把 raw label 重映射到 supervision slot：

| slot | 来源 | 语义 |
|---|---|---|
| `0` | surface 之外 + 物体外背景 | empty |
| `1..K_target` | manifest `target_part_names` 顺序 | 每个 target 自己的 slot |
| `K_target + 1` | `surface − union(target ind)`、所有非 target part 的 raw label | body slot（合并 base_body_0 / handle_0 等） |
| `-1` | 多个 target ind 重叠的 voxel | overlap ignore（loss 不计） |

`num_parts_phase8 = K_target + 2`（empty + targets + body）。slot 编号是
**sample-local** 的，跨样本不共享语义（slot 1 在 sample A 是 wheel，在 sample B
可能是 lid），符合 Part Flow 的 "object-local part token" 设计。

---

## 3. dataset 在线合成（`__getitem__` 30 行）

```python
# 1. 用 manifest 的 target_part_names → 1..K_target
target_to_slot = {name: i + 1 for i, name in enumerate(target_part_names)}
body_slot = K_target + 1

# 2. 每个 target 部件的 ind 文件 → 对应 slot
per_voxel_labels = zeros([64, 64, 64], int64)
overlap_mask = zeros_like(...)
for name in target_part_names:
    ind = np.load(f"ind_{name}_*.npy")
    existing = per_voxel_labels[ind] != 0
    overlap_mask[ind[existing]] = True
    per_voxel_labels[ind] = target_to_slot[name]

# 3. body slot = surface − union(target ind)
surface = np.load("surface.npy")
body_voxels = surface − union(target_ind_arrays)
per_voxel_labels[body_voxels] = body_slot

# 4. overlap → -1 ignore
per_voxel_labels[overlap_mask] = -1
```

每 sample CPU < 50ms，DataLoader workers 并行做不会成为瓶颈。

---

## 4. 验证

dataset 在线检查（PartFlowDataset.__getitem__ 已经做的）：

```python
# part_info raw labels 必须 contiguous 1..K_real
observed_labels = sorted(int(p["label"]) for p in parts.values())
assert observed_labels == list(range(1, K_real + 1))

# target_part_names ⊆ part_info parts
unknown = [n for n in target_part_names if n not in parts]
assert not unknown

# per_voxel_labels 取值 ⊆ {-1, 0, 1, ..., K_target+1}
assert per_voxel_labels.min() >= -1
assert per_voxel_labels.max() < K_target + 2

# mask 上的 raw label 必须都在 part_info 里
assert set(mask_uniques) ⊆ raw_to_slot.keys()

# manifest view_indices 不能超出 dinov2 V_total
assert max(view_indices) < V_total
```

任意一条不满足直接 raise，训练 fail loud。

---

## 5. 已废弃 / 未实现的部分

下列 v1 草案条目在 v2 已废弃：

- ❌ `part_labels_solid/{oid}/angle_{N}/part_labels_solid_64.npy`：v1 让数据
  组离线生成 dense [64,64,64] int64 GT，v2 改成 dataset 在线合成，**这个文件
  不需要数据组生成**。
- ❌ "实心 solid 填充" 语义：v1 设计 GT 是实心 part volume（含内部 voxel），
  v2 实际是 surface（含内壁）labeling，内部 voxel 标 0。
- ❌ §2.1 `build_part_labels_solid_64()` 的离线脚本：v2 不需要。

下列 v1 条目在 v2 仍然有效：

- ✓ `surface.npy` 文件契约（§1.1，仅措辞修正"外表面 1 层壳" → "整体含内壁
  surface"）。
- ✓ `ind_<part>_<i>.npy` 文件名规则与正则解析（`^ind_(.+)_(\d+)\.npy$`）。
- ✓ "raw label 必须 1..K_real contiguous" 由 04_voxelize 强制。
- ✓ 多 part overlap 时按 ignore（slot -1）处理。

---

*Spec v2 — 2026-05-07 — 基于 manifest-driven dataset + 数据同学反馈。
v1 (2026-04-26) 草案 archive 到 `docs/archive/phase08/solid_voxel_spec.md`。*
