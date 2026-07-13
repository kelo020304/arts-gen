# Dataset Toolkits Pipeline

这份文档描述 `dataset_toolkits` 当前实际可用的数据处理流程。目标读者是后续接手 PhysX-Mobility / HSSD 数据处理的人：读完以后应该能判断每一步该不该跑、会读写哪些产物、以及默认流程不会误触发哪些高成本分支。

所有命令默认在仓库根目录执行，并且必须使用官方 conda 环境：

```bash
conda activate dataset_toolkits
```

当前 PhysX-Mobility 配置入口是：

```bash
configs/PhysX-Mobility.yaml
```

配置文件里的 `data_root` 是数据根目录；脚本不要硬编码旧路径。原始数据在 `raw/` 下，视为只读；`joint_transforms/`、`part_info/`、`canonical_transforms/`、`renders/`、`reconstruction/`、`manifests/`、`vlm/`、`preview/` 是派生产物，可以按需要重建。

---

## 1. 当前主线结构

`pipeline/` 根目录只保留当前主线入口，按 01-11 编号组织；总入口只调度，
具体重活保留在独立子脚本里。旧 quadrant / 4-view / 老 VLM 脚本已移除；
当前 Step 09 需要复用的 manifest helper 统一放在 `utils/vlm_manifest_helpers.py`。

```text
pipeline/
├── 01_joint_transformation.py
├── 02_build_canonical_transforms.py
├── 03_voxelize.py
├── 04_build_valid_parts_manifest.py
├── 05_render.py
│   ├── 05_render_part_complete_rgb_mask.py      # 默认：16-view RGB + valid part masks + remaining mask
│   ├── 05_render_full_object_all_views.py       # 可选：整体 150-view RGB
│   └── 05_render_valid_parts_all_views.py       # 可选：有效可动 part 150-view RGB
├── 06_extract_feature.py                        # 默认：part_complete 16-view RGB DINOv2 feature
├── 07_encode_ss_latents_per_part.py
├── 08_decode_ss_latents.py
├── 09_build_vlm_dataset_manifest.py
├── 10_build_part_completion_manifest.py
├── 11_web_preview.py
│   ├── 11_web_preview_vlm_dataset.py
│   └── 11_web_preview_part_completion.py
├── 12_encode_part_synthesis_slat.py             # 可选后续分支，默认不跑
└── 13_build_part_synthesis_manifest.py          # 可选后续分支，默认不跑
```

核心约定：

- Step 04 的 valid-parts manifest 是后续筛选的 source of truth。
- “有效 target component” 的定义是：`可动部件 ∩ has_voxel_ind=true ∩ num_voxels > 5`。
- 每个物体的 angle 数量必须来自 `cfg.get_num_angles(object_id)`：可动物体通常是多 angle，静态物体只有 angle 0。
- Step 05 默认只跑 `part_complete`，不会默认跑整体 150-view，也不会默认跑有效 part 150-view。
- Step 12、13 是后续 part synthesis 分支，`run_pipeline.sh` 默认 profile 不执行。

---

## 2. 推荐入口

### 2.1 快速主线：构建 VLM + Part Completion 数据和预览

这是当前最常用、成本相对较低的训练数据主线。它会提取默认
`part_complete` 16-view DINOv2 feature，并继续生成 / 解码 SS latent，
用于 Part Completion 预览里的 GT vs SS decoder 对比；它只跳过可选
150-view 分支和 Step 12/13 part synthesis 分支。

```bash
bash run_pipeline.sh \
  --config configs/PhysX-Mobility.yaml \
  --profile base
```

单物体调试：

```bash
bash run_pipeline.sh \
  --config configs/PhysX-Mobility.yaml \
  --profile base \
  --object-ids 102377 \
  --workers 1
```

`base` 和 `preview-base` 当前等价，都会跑：

```text
1,2,3,4,5,6,7,8,9,10,11
```

### 2.2 默认 profile

不传 `--profile` 时，`run_pipeline.sh` 使用 default profile：

```text
1,2,3,4,5,6,7,8,9,10,11
```

注意：

- Step 05 在 default profile 中仍然只跑 `part_complete`。
- Step 07/08 不可跳过：Part Completion 预览需要 SS latent decode 结果做 GT vs decoder 对比。
- Step 12、13 不在 default/full/stable/base/preview-base 任何默认 profile 里。
- Step 06 默认提取 `part_complete` 16-view RGB 特征；150-view 特征需要通过 `--sets full_object_all_views,valid_parts_all_views` 显式请求。

### 2.3 显式选择步骤

```bash
bash run_pipeline.sh \
  --config configs/PhysX-Mobility.yaml \
  --steps 1,2,3,4,5,6,7,8,9,10,11 \
  --object-ids 102377
```

`--steps` 和 `--profile` 互斥。Step 04 是全局 manifest 构建，`run_pipeline.sh` 不会把 `--object-ids` 传给 Step 04。

---

## 3. Step-by-step 说明

### Step 01：joint transformation + part metadata

入口：

```bash
python pipeline/01_joint_transformation.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

输入：

- `raw/finaljson/<object_id>.json`
- `raw/partseg/<object_id>/...`

输出：

```text
joint_transforms/<object_id>.json
part_info/<object_id>/part_info.json
```

作用：

- `joint_transforms` 记录每个 object / angle / part 的关节变换矩阵，用于把部件放到对应 angle 的正确位姿。
- `part_info.json` 记录 canonical part 名、label、part index、类别、joint/motion 元信息等。后续 mask label、voxel 文件名、manifest target part 都依赖它。
- 这两个产物现在都在 `data_root` 一级派生目录下，不再放在 `reconstruction/` 下。

---

### Step 02：canonical transform

入口：

```bash
python pipeline/02_build_canonical_transforms.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

输出：

```text
canonical_transforms/<object_id>/angle_<i>/canonical_transform.json
```

作用：

- 为每个 object / angle 生成统一的 geometry normalization。
- 把 joint 变换后的 raw mesh 坐标映射到 pipeline canonical cube：`[-0.5, 0.5]^3`。
- 渲染和体素化都应使用这个 canonical 坐标约定，避免以前依赖 render 侧 `camera_transforms.json` 的隐式耦合。

常用调试参数：

```bash
# 只看计划，不写文件
python pipeline/02_build_canonical_transforms.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --dry-run
```

---

### Step 03：voxelize

入口：

```bash
python pipeline/03_voxelize.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --workers 1
```

主要输出：

```text
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/surface.npy
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/ind_<part_key>.npy
```

作用：

- 体素化整体 surface。
- 体素化每个 part。
- 小于阈值的 part 不会保留 `ind_<part>.npy`；当前阈值与 manifest 对齐为 `num_voxels > 5`。

下游影响：

- Step 04 根据这些 voxel 产物判断 `has_voxel_ind`。
- Step 07 根据这些 voxel 产物生成 SS latent。
- Step 10 的 Part Completion 样本只会绑定该视角可见且有有效 voxel 的 target part。

---

### Step 04：valid-parts manifest

入口：

```bash
python pipeline/04_build_valid_parts_manifest.py \
  --config configs/PhysX-Mobility.yaml
```

输出：

```text
manifests/<dataset_name>.json
```

对 PhysX-Mobility，默认是：

```text
manifests/PhysX-Mobility.json
```

作用：

- 汇总每个 object / angle / part 的有效性。
- 记录 `has_voxel_ind`、`num_voxels`、`voxel_ind_path`、过滤原因等。
- 作为 Step 05、09、10 的有效部件 source of truth。

有效 target component 判定：

```text
finaljson 中 motion type 属于 A/B/C 的可动 part
∩ Step 04 manifest 中 has_voxel_ind=true
∩ num_voxels > 5
```

---

### Step 05：render 总入口

总入口：

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --workers 1
```

默认行为：

```text
--sets part_complete
```

也就是说默认只调用：

```text
05_render_part_complete_rgb_mask.py
```

不会默认调用：

```text
05_render_full_object_all_views.py
05_render_valid_parts_all_views.py
```

#### 5.1 默认 render set：part_complete

输出：

```text
renders/<object_id>/angle_<i>/part_complete/
├── rgb/view_0.png ... rgb/view_15.png
├── mask/<valid_part_key>/mask_0.npy ... mask_15.npy
├── mask/<valid_part_key>/mask_0.png ... mask_15.png
├── mask/remaining/mask_0.npy ... mask_15.npy
├── mask/remaining/mask_0.png ... mask_15.png
├── camera_transforms.json
└── mask_labels.json
```

语义：

- RGB 是完整物体的 16 个视角。
- 对每个有效可动 target part 单独输出一个 binary mask。
- `remaining` mask 是所有非 target 可见物体像素的合并，包括固定部件和没有有效 voxel 的可动部件。
- 背景不属于任何 mask。
- 后续 Step 09、10 都基于这个 render set。

#### 5.2 可选 render set：full_object_all_views

入口：

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --sets full_object_all_views
```

输出：

```text
renders/<object_id>/angle_<i>/render_full_obj_all_view/
├── 000.png ... 149.png
└── transforms.json
```

语义：

- 整体物体 150-view RGB。
- 默认不跑，避免不必要渲染成本。
- 主要服务后续 150-view / SLAT / synthesis 分支。

#### 5.3 可选 render set：valid_parts_all_views

入口：

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --sets valid_parts_all_views
```

输出：

```text
renders/<object_id>/angle_<i>/render_part_all_view/<part_key>/
├── 000.png ... 149.png
└── transforms.json
```

语义：

- 只渲染有效可动 target part。
- part 会应用 joint transform，保留它在整体物体 canonical frame 中的正确位置。
- 采样 seed 与 full-object 150-view 对齐，保证同一个 view index 表示同一个 camera view。
- 默认不跑。

#### 5.4 Step 05 dry-run

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --angle-ids 0 \
  --dry-run \
  --workers 1
```

---

### Step 06：DINOv2 feature extraction

入口：

```bash
python pipeline/06_extract_feature.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

默认行为：

```text
--sets part_complete
```

默认输入：

```text
renders/<object_id>/angle_<i>/part_complete/rgb/view_0.png ... view_15.png
```

默认输出：

```text
reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz
reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens_npz_meta.json
```

默认 token shape：

```text
(16, 1370, 1024)
```

可选 render set：

```bash
# 整体 150-view RGB feature，要求 Step 05 已显式跑过 full_object_all_views
python pipeline/06_extract_feature.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --sets full_object_all_views

# 有效 part 150-view RGB feature，要求 Step 05 已显式跑过 valid_parts_all_views
python pipeline/06_extract_feature.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --sets valid_parts_all_views
```

可选 150-view 输出：

```text
reconstruction/dinov2_tokens/<object_id>/angle_<i>/full_object/tokens.npz
reconstruction/dinov2_tokens/<object_id>/angle_<i>/valid_parts/<part_key>/tokens.npz
```

历史 12-view quadrant 布局只通过 Step 06 的 `--sets legacy_quadrant` 显式读取旧数据产物，
不属于当前默认主线，也不依赖已移除的旧脚本目录。

---

### Step 07：encode SS latents

入口示例：

```bash
python pipeline/07_encode_ss_latents_per_part.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --latent-scope all
```

输入：

```text
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/surface.npy
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/ind_<part_key>.npy
```

输出：

```text
reconstruction/ss_latents_expanded/<object_id>/angle_<i>/latent.npz
reconstruction/ss_latents_per_part/<object_id>/angle_<i>/<part_key>.npy
```

作用：

- 对整体 surface 编码 SS latent。
- 对每个 part 编码 SS latent。

常用 scope：

- `--latent-scope all`：整体 + parts。
- `--latent-scope overall`：只重建整体 surface latent。
- `--latent-scope parts`：只处理 per-part latent。

注意：这个步骤需要 TRELLIS 依赖和 GPU 环境。可先用 `--dry-run` 检查枚举结果。

---

### Step 08：decode SS latents

入口示例：

```bash
python pipeline/08_decode_ss_latents.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --scope all-parts
```

输出：

```text
reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/overall.npy
reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/parts/<part_key>.npy
reconstruction/ss_latent_decoded/<object_id>/angle_<i>/64/metrics.json
```

作用：

- 把 SS latent decode 回 voxel coords。
- 为 Step 11 Part Completion preview 提供 GT vs SS decoder 的交互式对比。

推荐：

- 当前 Part Completion 预览若要完整显示 GT/SS 重合度，用 `--scope all-parts`。
- 只检查整体 surface 时，用 `--scope overall-only`。
- 默认 `vlm-targets` 是旧 VLM JSONL 相关路径，当前新 VLM 单图样本流程建议显式指定 scope。

---

### Step 09：build VLM dataset manifest

入口：

```bash
python pipeline/09_build_vlm_dataset_manifest.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

输入：

```text
manifests/<dataset_name>.json
renders/<object_id>/angle_<i>/part_complete/rgb/view_<j>.png
renders/<object_id>/angle_<i>/part_complete/mask/<valid_part_key>/mask_<j>.npy
```

输出：

```text
vlm/training_json/arts_mllm_<dataset_slug>_part_complete_8view_1img.jsonl
vlm/training_json/arts_mllm_<dataset_slug>_part_complete_8view_1img.jsonl.meta.json
```

当前规则：

- 使用 `part_complete` 的前 8 个固定视角。
- 1 张 RGB = 1 个 VLM 训练样本。
- 只有该视角可见的有效 target part 会进入样本。
- bbox 从对应 part binary mask 计算，不再使用旧 `bbox_gt`。
- 如果某个 view 没有任何可见有效 target part，则跳过该 view。

注意：Step 09 会写一个辅助 label mask 到 `part_complete/mask/label/`。Step 10 会生成 Part Completion 使用的最终 label mask，并包含 `remaining` label。

---

### Step 10：build Part Completion manifest

入口：

```bash
python pipeline/10_build_part_completion_manifest.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --overwrite
```

输入：

```text
manifests/<dataset_name>.json
renders/<object_id>/angle_<i>/part_complete/rgb/view_<j>.png
renders/<object_id>/angle_<i>/part_complete/mask/<valid_part_key>/mask_<j>.npy
renders/<object_id>/angle_<i>/part_complete/mask/remaining/mask_<j>.npy
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/ind_<part_key>.npy
reconstruction/voxel_expanded/<object_id>/angle_<i>/64/surface.npy
reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz
```

输出：

```text
manifests/part_completion/arts_pc_<dataset_slug>_train.jsonl
manifests/part_completion/manifest_meta.json
manifests/part_completion/skip_report.json
renders/<object_id>/angle_<i>/part_complete/mask/label/mask_<j>.npy
renders/<object_id>/angle_<i>/part_complete/mask/label/mask_<j>.png
```

样本规则：

- 1 张 RGB + 1 张 label mask = 1 个样本。
- 每个样本记录对应的 `part_complete/tokens.npz` 和 `view_idx`；训练时用 `tokens[view_idx]` 取得该 RGB 的 DINO 特征。
- 默认严格要求 `reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz` 已存在；仅调试/预览可传 `--allow-missing-dinov2` 放宽。
- 每个样本只绑定该 view 中可见的有效 target part voxel。
- 如果该 view 没有任何可见有效 target part，则跳过。
- `remaining` 出现在 label mask 中，但不会绑定 target voxel。

label mask 约定：

```text
0 = background
part_info.label = visible valid target part
remaining_label = merged visible non-target object pixels
```

这里的 non-target 包括固定部件、不可动部件、没有有效 voxel 的部件等。

---

### Step 11：web preview 总入口

入口：

```bash
python pipeline/11_web_preview.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

默认输出：

```text
preview/index.html
preview/preview_manifest.json
preview/vlm_training/index.html
preview/part_completion/index.html
```

模式：

```bash
# 只生成 VLM 预览
python pipeline/11_web_preview.py \
  --config configs/PhysX-Mobility.yaml \
  --mode vlm

# 只生成 Part Completion 预览
python pipeline/11_web_preview.py \
  --config configs/PhysX-Mobility.yaml \
  --mode pc

# 默认：两个都生成，并写一个 switch index
python pipeline/11_web_preview.py \
  --config configs/PhysX-Mobility.yaml \
  --mode both
```

VLM 预览：

- 入口：`preview/vlm_training/index.html`
- 展示 JSONL 中真实使用的单视角 RGB。
- bbox 会画在 RGB 上。
- 不加载 voxel，也不展示 voxel viewer。

Part Completion 预览：

- 入口：`preview/part_completion/index.html`
- 按样本展示 RGB、label mask、separated masks。
- 对每个有效 target part 展示一个 interactive voxel viewer：GT part voxel vs SS decoder voxel。
- 对 `remaining` 展示整体 surface：GT surface voxel vs SS decoder overall voxel。
- 保留 IoU / Precision / Recall / GT / SS / intersection / FP / FN 等指标。

如果 Step 07、08 没有跑，Part Completion 预览仍可用于检查 RGB/mask/manifest，但 SS decoder voxel 对比可能缺失或报 warning。

---

## 4. 可选后续分支：Step 12 / Step 13

这两个步骤保留在 pipeline 根目录，但不属于当前默认主线。

```text
12_encode_part_synthesis_slat.py
13_build_part_synthesis_manifest.py
```

它们面向后续 part synthesis / SLAT 分支。只有明确需要时才单独调用，`run_pipeline.sh` 默认 profile 不会执行。

---

## 5. 常用命令清单

### 查看 pipeline 帮助

```bash
./run_pipeline.sh --help
python pipeline/05_render.py --help
python pipeline/11_web_preview.py --help
```

### 单物体完整生成 VLM + Part Completion + preview

```bash
bash run_pipeline.sh \
  --config configs/PhysX-Mobility.yaml \
  --profile base \
  --object-ids 102377 \
  --workers 1
```

### 只重建 render part_complete

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --angle-ids 0 \
  --sets part_complete \
  --workers 1
```

### dry-run render，不写数据

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --angle-ids 0 \
  --dry-run \
  --workers 1
```

### 显式跑 150-view 可选 render

```bash
python pipeline/05_render.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --angle-ids 0 \
  --sets full_object_all_views,valid_parts_all_views \
  --workers 1
```

### 只重建 VLM JSONL

```bash
python pipeline/09_build_vlm_dataset_manifest.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377
```

### 只重建 Part Completion manifest

```bash
python pipeline/10_build_part_completion_manifest.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --overwrite
```

### 只重建网页预览

```bash
python pipeline/11_web_preview.py \
  --config configs/PhysX-Mobility.yaml \
  --object-ids 102377 \
  --mode both
```

---

## 6. 当前数据流图

```text
raw/finaljson + raw/partseg
        │
        ▼
01_joint_transformation
        ├── joint_transforms/<object>.json
        └── part_info/<object>/part_info.json
        │
        ▼
02_build_canonical_transforms
        └── canonical_transforms/<object>/angle_i/canonical_transform.json
        │
        ▼
03_voxelize
        └── reconstruction/voxel_expanded/<object>/angle_i/64/{surface.npy,ind_<part>.npy}
        │
        ▼
04_build_valid_parts_manifest
        └── manifests/<dataset>.json
        │
        ▼
05_render --sets part_complete
        └── renders/<object>/angle_i/part_complete/{rgb,mask}
        │
        ├──► 09_build_vlm_dataset_manifest
        │        └── vlm/training_json/arts_mllm_<dataset>_part_complete_8view_1img.jsonl
        │
        └──► 10_build_part_completion_manifest
                 └── manifests/part_completion/arts_pc_<dataset>_train.jsonl
                         │
                         ▼
                  11_web_preview
                         ├── preview/vlm_training/index.html
                         └── preview/part_completion/index.html
```

SS latent / decoder QC 分支：

```text
03_voxelize
   └── 07_encode_ss_latents_per_part
           ├── reconstruction/ss_latents_expanded/...
           └── reconstruction/ss_latents_per_part/...
                  │
                  ▼
           08_decode_ss_latents
                  └── reconstruction/ss_latent_decoded/...
                         │
                         ▼
           11_web_preview_part_completion 的 GT vs SS decoder voxel 对比
```

150-view 可选分支：

```text
05_render --sets full_object_all_views,valid_parts_all_views
   ├── renders/<object>/angle_i/render_full_obj_all_view/000..149.png
   └── renders/<object>/angle_i/render_part_all_view/<part>/000..149.png
```

---

## 7. 交接注意事项

1. 不要把历史 quadrant/4-view 数据产物和当前 Step 05 混为一谈。当前主线 VLM/PC 都使用 `part_complete`。
2. 不要假设全局 `num_angles`；每个 object 的 angle 数量都通过 config 的 object-aware 逻辑决定。
3. 不要把所有可动部件都当 target。target 必须同时满足可动和有效 voxel。
4. `remaining` 是训练 mask 的 label 区域，不是 target voxel。
5. Step 05 的 150-view render 是显式 opt-in。默认和 `run_pipeline.sh --profile base` 都不会跑 150-view；base 只提取 `part_complete` 16-view DINOv2 feature。
6. Step 06 默认提取 `part_complete` 16-view RGB features；如果需要 150-view feature，必须先显式跑对应 Step 05 render set，再在 Step 06 里显式传 `--sets full_object_all_views` 或 `--sets valid_parts_all_views`。
7. 如果要看 Part Completion 的 interactive voxel GT/SS 对比，需要先跑 Step 07 和 Step 08 生成 decoded voxel；否则网页只能检查 RGB/mask/manifest 部分。
