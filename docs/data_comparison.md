# 小米数据 vs PhysX 格式对比

> 范围: 把小米交付的铰链物体数据接入 `submodules/dataset_toolkits` 管线。本文档列出格式差距、转换规则、数据质量反馈点、跑通 pipeline 1→11 的验收步骤。
> 最后更新: 2026-05-06

## 概述

小米交付的两个示例位于 `/media/mi/E2AB72E695F22B61/data_sda/aticulated_data/Mi/{uuid}/`，
JSON 顶层字段几乎完全对齐 PhysX 格式（[`docs/DATA_FORMAT.md`](DATA_FORMAT.md)）。
真正需要处理的差距分四类：

1. **结构性**: 目录布局、`.convex.stl` 命名 — 转换脚本可机械修复
2. **语义性**: prismatic joint type 用 `P` 不是 `B`；revolute 角度用度数不是 pi 倍数 — 转换脚本可修复
3. **轴系**: 模型实际是 Z-up，但 `model_info.json` 错标 `y-up` — 通过 dataset_toolkits config `obj_up_axis: Z` 处理
4. **元数据/命名质量**: object_name / category / part name / description 多处错误或模板化 — **必须反馈给数据提供方人工修**，pipeline 帮不上。详见 [`docs/data_review_checklist.md`](data_review_checklist.md)。

## 关键决策

- **接入路径**（2026-05-06）: 转换脚本放 `submodules/dataset_toolkits/converters/convert_mi2physx/`（与上游 `convert_hssd.py` 同级），把小米数据扁平化重排到 `data/Mi-PhysX/raw/`，再用我们 wrapper 跑 pipeline。— 不改上游已有代码（仅在 `converters/` 增加），未来 upstream 升级也不冲突。
- **一键运行**: [`scripts/ops/data_pipeline/run_mi_pipeline.sh`](../scripts/ops/data_pipeline/run_mi_pipeline.sh) 把"转换 → validator → step 6 → step 7-8 → gap-fill → step 9-11 → Z-up viewer patch"封装为单条命令；要走完整 pipeline 1→11 用它。
- **conda 环境**（2026-05-06）: 本项目用 `arts-gen`，wrapper 通过 `CONDA_DEFAULT_ENV=dataset_toolkits` 伪装绕开 upstream 校验。— 维持单一统一环境。
- **up axis**（2026-05-06）: 同学澄清后确认小米数据**全部 Z-up**（`model_info.json` 的 `y-up` 字段不准）。在 `Mi.yaml` 设 `render.obj_up_axis: Z`，转换脚本不再做几何旋转。
- **upstream pipeline gap**（2026-05-06）: 默认 profile 1-11 缺一步——**没有任何脚本写 `reconstruction/ss_latents_expanded/{id}/angle_{k}/latent.npz`**，但 step 9/10 都依赖它。我们补了 `submodules/dataset_toolkits/utils/encode_ss_latents_expanded.py`，复用 step 8 的 TRELLIS encoder 跑整体表面体素，必须在 step 8 之后、step 9 之前手动调用。
- **Step 6 manifest 必须有 validator 报告**（2026-05-06）: step 7 拒绝 `validator_status=UNKNOWN` 的 manifest。upstream `run_pipeline.sh` 调 step 6 时不带 `--validator-report`，所以默认产出 UNKNOWN。**正确流程**：先 `python utils/validate_dataset.py --steps render,voxel`，再 `python pipeline/06_build_manifest.py --validator-report <path>`，然后才能跑 step 7。

## 当前设计

### 输入形态（小米交付）

```text
{Mi_root}/{uuid}/
├── {uuid}.json                  # 主元数据（顶层字段对齐 PhysX，但有元数据/命名问题，见下文）
├── {uuid}.urdf                  # URDF（pipeline 不消费，可选保留）
├── model_info.json              # 多余字段；其中 "up axis: y-up" 不可信
├── config.yaml                  # Isaac Sim 配置，丢弃
├── .asset_hash                  # 多余
├── objs/                        # 部件级 OBJ + 贴图（PhysX 想要的几何）
│   ├── {part_stem}.obj
│   ├── {part_stem}.mtl
│   ├── {part_stem}.convex.stl   # ★ 命名差异（少一个 .obj 后缀）
│   └── *.png                    # MTL 引用的贴图
├── obj/                         # 整体合并 OBJ + 贴图（不需要）
├── images/                      # camera renders（不是贴图）
└── usd/                         # USD 资产（不需要）
```

### 目标形态（PhysX raw/）

```text
data/Mi-PhysX/raw/
├── finaljson/{object_id}.json
├── partseg/{object_id}/
│   └── objs/
│       ├── {part_stem}.obj
│       ├── {part_stem}.mtl
│       ├── {part_stem}.obj.convex.stl
│       └── *.png
└── urdf/{object_id}.urdf                  # 可选
```

`{object_id}` 直接用小米 UUID。pipeline 把 ID 当字符串使用，不要求是数字。

### 差异点对比

| # | 差异维度 | 小米现状 | PhysX 期望 | 修法 | 来源证据 |
|---|---|---|---|---|---|
| 1 | 顶层目录布局 | `{id}/{id}.json` + `{id}/objs/` | `raw/finaljson/{id}.json` + `raw/partseg/{id}/objs/` | 转换脚本扁平化 | [DATA_FORMAT.md](DATA_FORMAT.md) §目录结构 |
| 2 | 凸包文件名 | `{stem}.convex.stl` | `{stem}.obj.convex.stl` | 重命名时拼上 `.obj` | [DATA_FORMAT.md](DATA_FORMAT.md) §partseg |
| 3 | Prismatic joint type | `"P"` | `"B"` | group_info 里 `P → B` | [`config_loader.py:26`](../submodules/dataset_toolkits/utils/config_loader.py) `VALID_FINALJSON_JOINT_TYPES = {"A","B","C","CB","D","E"}` |
| 4 | Revolute (`C`) 角度单位 | 度数（如 `-121.058`、`45.0`、`-30.0`、`90.0`） | pi 倍数 | params[6:8] 除以 180 | [`joint_utils.py:440-441`](../submodules/dataset_toolkits/utils/joint_utils.py) `angle_lo = angle_range[0] * math.pi` |
| 5 | up 轴约定 | `model_info.json` 标 `y-up` 但**实际是 Z-up** | dataset_toolkits 接受 `Y` 或 `Z`（全局，每 yaml 一个值）| `Mi.yaml` 设 `render.obj_up_axis: Z` | 同学口头确认 + body 几何不对称（`Z` 轴 0 起的"地面" pattern）+ 渲染验证 |
| 6 | URDF / USD / config.yaml / model_info.json | 都有 | 不消费 | 转换时丢弃，URDF 可保留进 `raw/urdf/` | pipeline 不读这些文件 |
| 7 | `images/` | camera renders | MTL 贴图（可选） | 丢弃 | 实际目录 ls 验证 |

### 转换映射规则（per object）

[`submodules/dataset_toolkits/converters/convert_mi2physx/convert.py`](../submodules/dataset_toolkits/converters/convert_mi2physx/convert.py) 做的：

```text
{Mi_root}/{uuid}/{uuid}.json          → raw/finaljson/{uuid}.json   （含字段 fix）
{Mi_root}/{uuid}/objs/{stem}.obj      → raw/partseg/{uuid}/objs/{stem}.obj
{Mi_root}/{uuid}/objs/{stem}.mtl      → raw/partseg/{uuid}/objs/{stem}.mtl
{Mi_root}/{uuid}/objs/{stem}.convex.stl → raw/partseg/{uuid}/objs/{stem}.obj.convex.stl  ★ 重命名
{Mi_root}/{uuid}/objs/*.png           → raw/partseg/{uuid}/objs/*.png
{Mi_root}/{uuid}/{uuid}.urdf          → raw/urdf/{uuid}.urdf              （可选）
```

JSON 字段 fix:
1. `group_info` 中 `joint_type == "P"` 改成 `"B"`。
2. `group_info` 中 `joint_type == "C"` 的 `params[6]` 和 `params[7]` 各除 `180.0`（带幂等守卫：值 ≤ 1.0 不再除）。

**几何不旋转**——up axis 通过 yaml 配置告诉 pipeline，源 OBJ 字节级保留。

## 约定

- 转换脚本必须**显式失败**而不是兜底：未知 joint type、缺 OBJ、缺 mesh stem 都直接 `raise`。
- 转换脚本必须**幂等**：重跑覆盖目标目录里的同名文件；`P→B` 自然幂等；`C` 角度除 180 用值守卫，不会双除。
- 不修改 submodule 里的**已有代码**或上游 `PhysX-Mobility.yaml`；只在 `submodules/dataset_toolkits/utils/` 和 `submodules/dataset_toolkits/configs/` 增加新文件。
- 同一物体 ID 必须在 `finaljson/`、`partseg/`、`urdf/` 三处保持一致。

## 数据质量反馈 checklist

> 这部分**已搬到独立文档** [`docs/data_review_checklist.md`](data_review_checklist.md)，按"元数据 / part 命名 / description / 关节几何 / 单位"分组，并给出通用 review 流程和反馈话术模板。

## 接入验收 checklist（pipeline 跑通）

### A. 转换前

- [ ] `ls /media/mi/E2AB72E695F22B61/data_sda/aticulated_data/Mi/` 至少能看到 2 个 UUID 子目录
- [ ] 每个 UUID 子目录里都有 `{uuid}.json` 和非空的 `objs/`
- [ ] `{uuid}.json` 里 `parts[].obj` 引用的每个 stem 都能在 `objs/{stem}.obj` 找到
- [ ] `group_info["0"]` 存在，根组里至少有 1 个 part label
- [ ] 所有 part 的 `label` 唯一，且 `group_info` 引用的 part label 都在 `parts[].label` 里

### B. 转换后（静态检查）

- [ ] `data/Mi-PhysX/raw/finaljson/{uuid}.json` 存在
- [ ] **没有任何 `"P"` joint type**：`grep -n '"P"' data/Mi-PhysX/raw/finaljson/*.json` 应空
- [ ] 所有 `C` 类型 group 的 `params[6:7]` 绝对值都 `≤ 1.0`
- [ ] `data/Mi-PhysX/raw/partseg/{uuid}/objs/` 里每个 `{stem}.obj` 都有对应的 `{stem}.obj.convex.stl`
- [ ] `Mi.yaml` 里 `render.obj_up_axis: Z`、`data_root` 是绝对路径、`articulated_objects: all`、`static_objects: []`

### C. Pipeline 跑通（按顺序）

- [ ] Blender 4.4.0 在 `software/blender-4.4.0-linux-x64/blender`
- [ ] DINOv2 + TRELLIS 权重在 `pretrained/`（DINOv2 1.2 GB + ss_enc 119 MB + ss_dec 147 MB）
- [ ] `conda activate arts-gen`
- [ ] **执行顺序**（不能直接 `--steps 1-11` 一把跑，因为 step 6/7 中间需要 validator dance + step 8/9 中间需要 gap-fill）。**用 [`scripts/ops/data_pipeline/run_mi_pipeline.sh`](../scripts/ops/data_pipeline/run_mi_pipeline.sh) 一条命令搞定所有顺序**；它内部会按下面顺序调：
  1. `converters/convert_mi2physx/convert.py` (Xiaomi → PhysX)
  2. `run_pipeline.sh --steps 1,2,3,4,5`
  3. `utils/validate_dataset.py --steps render,voxel`
  4. `pipeline/06_build_manifest.py --validator-report <path>` ← 必须传，否则 status=UNKNOWN
  5. `run_pipeline.sh --steps 7,8`
  6. `utils/encode_ss_latents_expanded.py` ← 我们补的 gap-fill
  7. `run_pipeline.sh --steps 9,10,11`
  8. sed 注入 `camera.up.set(0,0,1)` 到生成的 HTML
- [ ] `data/Mi-PhysX/joint_transforms/{uuid}.json` 存在
- [ ] `data/Mi-PhysX/renders/{uuid}/angle_*/rgb/view_*.png` 共 N×12 张（N=num_angles）
- [ ] `data/Mi-PhysX/manifests/Mi.json` 的 `summary.validator_status == "PASS"`
- [ ] `data/Mi-PhysX/vlm/training_json/arts_mllm_mi.jsonl` 行数 > 0
- [ ] `data/Mi-PhysX/preview/vlm_training/index.html` 存在
- [ ] HTTP server 起来后浏览器打开能渲染

### D. 数据语义肉眼验收（前端里）

- [ ] **物体朝向正确**：用 Z-up viewer patch 后，body 的"地面"在 Z=低的方向，"顶部"在 Z=高的方向
- [ ] Jewelry Box 的 lid 在 `angle_*` 间确实绕 hinge 轴小幅转动
- [ ] Coffee Machine 的 3 个 button 在 `angle_*` 间小距离平移（如果 visibility filter 没把它们 ban 掉）
- [ ] 每个 part 的颜色编号和 `parts[].label` 一致
- [ ] bbox 没有明显错位、没有空 bbox

## 流程陷阱（operator 必须知道）

1. **不能 `--steps 1,2,3,4,5,6,7,8,9,10,11` 一把跑** —— step 6 需要 validator 报告先就位（否则 step 7 拒绝），step 8 后需要手动调 `encode_ss_latents_expanded.py`（否则 step 9 全部 skip）。详见上面 C 段的执行顺序。
2. **wrapper `scripts/ops/data_pipeline/launch_dataset_preview.sh` 默认只跑 step 11** —— 第一次跑数据要显式传 `--steps 1,2,3,4,5,6,7,8,9,10,11`（但仍然要走上面的拆分调用）。
3. **`render.obj_up_axis` 是全局配置**，dataset_toolkits 不支持 per-object override。如果一个 dataset 里同时混 Y-up 和 Z-up 的物体，必须在转换时旋转到统一 up axis。当前小米数据全 Z-up，所以单 yaml 就够。
4. **Three.js viewer 默认相机 Y-up**，会让 Z-up 数据看起来"歪倒"。修法是 sed 注入 `camera.up.set(0,0,1)` 到 `preview/vlm_training/index.html`：
   ```bash
   sed -i 's#camera.position.set(128,96,128);#camera.up.set(0,0,1);camera.position.set(96,-128,96);#' \
     data/Mi-PhysX/preview/vlm_training/index.html
   ```

## 待解问题

- 是否要保留小米的 `model_info.json`（含 `mass(kg)`、`hwl`、`friction_coefficient` 等物理量）？当前 pipeline 不消费，但下游仿真可能用得上。先保留在源目录，转换脚本不复制。
- Step 7 visibility filter 把"任意 movable target part 在所有 12 视图都 invisible"的物体整体过滤。Coffee Machine 的 3 个 button 命中此条件——下次数据需要把 button 几何放大或者换个角度采样让 button 至少在某些视图可见。
- `encode_ss_latents_expanded.py` 是我们的本地 gap-fill。**长期方案**应该是 PR 给 upstream，把它正式做成 step `8b`（或者改 step 8 让它同时产 per-part 和 expanded 两份 latent）。
- Blender 当前 `software/` 目录下手动安装；应整合到 setup 脚本里。
