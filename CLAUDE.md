# Arts-Reconstruction Project Instructions

## Overview

PhysX-Anything 铰链物体重建项目：从多视角图像 + 2D 子部件分割 mask，重建尺度一致的、各部件完整的铰链物体。基于 TRELLIS 架构，在 64^3 体素空间下保持正确的相对尺寸。

## Project Structure

```
arts-reconstruction/
├── TRELLIS-arts/          # Training code (TRELLIS-based backbone)
├── scripts/
│   ├── launch_dataset_preview.sh   # ★ One-shot launcher: dataset_toolkits step 11 + http server
│   ├── setup_arts_gen.sh           # ★ Install conda env arts-gen (run --check first)
│   ├── train/             # scripts/train/ — Training scripts
│   │   ├── configs/
│   │   │   └── stage3_mv_partlabel.yaml
│   │   ├── train_stage3_mv_dataset.py
│   │   └── train_stage3_mv.py
│   ├── inference/         # Inference scripts (future)
│   ├── eval/  ops/  dev/  # Evaluation / ops / dev tools
│   └── sync_to_moganshan.sh
├── submodules/
│   ├── dataset_toolkits/  # ★ Canonical data pipeline (mlpchenxl/dataset_toolkits, pinned)
│   ├── PhysX-Anything/    # PhysX-Anything reference
│   ├── TRELLIS.1/         # Original TRELLIS clone
│   └── TRELLIS.2/         # TRELLIS variant clone
├── pretrained/            # Pretrained weights (SS-VAE encoder)
├── data/                  # Generated datasets (not tracked)
├── .gsd/                  # Frozen GSD-2 planning (read-only reference, M001/S02 done — see Workflow section)
└── docs/archive/v0.1.0-planning/  # Legacy GSD-1 planning (v0.1.0 milestone, archived)
```

## Data Pipeline

Canonical implementation lives in `submodules/dataset_toolkits/pipeline/` (mlpchenxl/dataset_toolkits). Run via `scripts/ops/data_pipeline/launch_dataset_preview.sh` or directly with `submodules/dataset_toolkits/run_pipeline.sh`.

| Step | Script (under `submodules/dataset_toolkits/pipeline/`) | Purpose |
| --- | --- | --- |
| 1 | `01_joint_transformation.py` | Joint-angle expansion, generates `joint_transforms/{id}.json` + `part_info.json` |
| 2 | `02_render_quadrant_views.py` | 4 quadrants × 3 views = 12 views RGB + int32 Object Index mask |
| 3 | `03_extract_bbox_gt.py` | Extract bbox GT from masks |
| 4 | `04_voxelize.py` | Per-part 64³ voxel + label volume |
| 5 | `05_extract_feature.py` | DINOv2 feature encoding |
| 6 | `06_build_manifest.py` | Build PhysX delivery manifest (mask/bbox/voxel availability) |
| 7 | `07_build_vlm_dataset.py` | Build VLM JSONL dataset (4-view groups) |
| 8 | `08_encode_ss_latents_per_part.py` | Per-part SS latent (GPU/TRELLIS encoder) |
| 9 | `09_build_part_completion_manifest.py` | Part Completion training manifest from confirmed VLM JSONL |
| 10 | `10_decode_ss_latents.py` | Decode SS latents back to 64³ voxels for QC |
| 11 | `11_web_preview_dataset.py` | **HTML preview** (consumed by `launch_dataset_preview.sh`) |
| 12 (dev) | `12_encode_part_synthesis_slat.py` | OmniPart/TRELLIS part-synthesis SLat caches |
| 13 (dev) | `13_build_part_synthesis_manifest.py` | Part Synthesis manifest + OmniPart mesh lists |

Default profile runs steps 1–11; steps 12/13 are development-only.
Profiles available via `run_pipeline.sh --profile`:
- `default`/`full`/`stable`: 1,2,3,4,5,6,7,8,9,10,11
- `base`/`preview-base`: 1,2,3,4,5,6,7,11 (skip GPU encoding/decoding)

Steps 5/8/10/12 need GPU + the `arts-gen` env (we ignore upstream's
`EXPECTED_CONDA_ENV=dataset_toolkits` and reuse our unified env).

## Key Conventions

- **Coordinate system**: PartNet-Mobility = Y-up, TRELLIS/voxel = Z-up (convert via +90 deg X rotation)
- **Material preservation**: Blender loads original OBJ with .mtl + textures from raw/partseg/{id}/objs/
- **Data path**: All generated data under `data/PhysX-Mobility/arts/reconstruction/`
- **Part mapping**: part_info.json is single source of truth for part labels
- **Mask format**: [512,512] int32, 0=background, label values from part_info.json
- **Code update logs**: On the dev machine, update `.txt` maintenance logs by default. For PartMMDiT, use `TRELLIS-arts/code_update/code_update_part_mmdit.txt` as the canonical log; do not rely on the `.md` copy unless the user explicitly asks to sync it.

## Environment

- **Conda env**: `arts-gen` — **single unified env** for data processing + TRELLIS training/inference
  - Install: `bash scripts/ops/setup/setup_arts_gen.sh` (use `--check` first to dry-run; ~30-60 min)
  - Activate: `conda activate arts-gen`
  - Verify: `python -c "import torch, trimesh, yaml, transformers; print(torch.__version__)"`
  - Replaces legacy envs `artsvox` (data) and `trellis` (training); both deprecated.
- **Blender**: `software/blender-4.4.0-linux-x64/blender` (project-relative symlink)
- **MuJoCo** (optional, for URDF inspection): `~/mujoco-3.2.7/bin/simulate`
- **GPU**: Required for DINOv2 / SS-VAE encoding (steps 5/8) + training
- **Headless GPU rendering (VolcEngine ML container only)**: Blender EEVEE_NEXT
  needs a GPU graphics context, but VolcEngine's ML containers ship CUDA compute
  only (no `/dev/dri`, `nvidia-drm.modeset=N`, cgroup blocks DRM). Workaround:
  Xvfb virtual display + NVIDIA Vulkan driver. First time in a container:
  ```bash
  sudo bash scripts/ops/setup/setup_blender_headless_gpu.sh
  ```
  Installs `xvfb`/`vulkan-tools`, downloads `libnvidia-gpucomp.so.550.144.03`
  (Container Toolkit's CSV doesn't list this lib so it's not auto-mounted),
  writes Vulkan ICD config, starts Xvfb on `:99`. Idempotent — re-run after
  container restart (Xvfb dies on restart; lib + ICD persist in `/usr/lib`,
  `/etc`). `run_physx_mobility_cloud.sh` exports the required env vars itself
  (`DISPLAY=:99`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, etc.) and passes
  `--gpu-backend vulkan` to Blender. For ad-hoc Blender calls outside the
  pipeline export them manually (see setup script output).

## Quick Start

```bash
# 1. Install env (first time only, ~30-60 min)
bash scripts/ops/setup/setup_arts_gen.sh --check     # see commands first
bash scripts/ops/setup/setup_arts_gen.sh             # actually install

# 2. View any dataset's HTML preview
bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root /absolute/path/to/dataset_root
# → opens http://localhost:8000/index.html

# 3. Run a pipeline step (e.g. voxelize + preview)
bash scripts/ops/data_pipeline/launch_dataset_preview.sh --data-root /path --steps 4,11
```

## Workflow (Superpowers)

This project uses [obra/superpowers](https://github.com/obra/superpowers) v5+ (installed via `obra/superpowers-marketplace`, user scope) as its development methodology. Each capability is exposed two ways:
- **User types** `/superpowers:<skill-name>` (e.g. `/superpowers:brainstorming`) — slash command form, visible under `/help` → `custom-commands`
- **Model invokes** the `Skill` tool with `<skill-name>` — programmatic form, auto-triggered by the `using-superpowers` bootstrap at session start

Core skills you should reach for (full list in `~/.claude/plugins/cache/superpowers-marketplace/superpowers/<version>/skills/`):
- `brainstorming` — scope a problem before writing code
- `writing-plans` — design before implementing
- `executing-plans` — drive a written plan to completion
- `systematic-debugging` — methodical bug investigation
- `test-driven-development` — TDD loop
- `verification-before-completion` — confirm work meets the goal before declaring done
- `subagent-driven-development` / `dispatching-parallel-agents` — parallelize via Task agents
- `using-git-worktrees` — isolate risky work
- `requesting-code-review` / `receiving-code-review` — review loop
- `finishing-a-development-branch` — wrap up before PR

If a skill applies even at 1% probability, invoke it (per `using-superpowers` policy). User instructions in this CLAUDE.md still override skill defaults.

### Frozen planning history

- `.gsd/` — GSD-2 milestones/slices/tasks, last active up to **M001/S02 done, S03 not started**. Treat as **read-only**: don't write new `S##-PLAN.md` / `T##-PLAN.md` / SUMMARY / UAT files there. When you need context on what's been decided or implemented, read `.gsd/STATE.md` and `.gsd/DECISIONS.md`.
- `docs/archive/v0.1.0-planning/` — older GSD-1 phase artifacts (v0.1.0 milestone, 10 phases all complete). Same rule: read-only.
- The `M001 S03` work (External Data Conversion) is still real outstanding work — when you pick it up, plan it via superpowers (`brainstorming` skill → `writing-plans` skill), not by extending `.gsd/milestones/M001/`.

### Codex review (still recommended)

Independent of methodology: when you make non-trivial code changes, asking a Codex/secondary review pass is still useful. The prior GSD-2 rule mandated a `codex_review.post_execution` block in plan frontmatter — under superpowers there's no equivalent frontmatter slot, so just include 3-4 specific review prompts (in Chinese) inside whatever plan/PR description you write. Focus them on:
- 核心逻辑改动是否正确
- 跨脚本的数据路径一致性
- 是否有非预期的副作用

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Arts-Reconstruction: 铰链物体多视角重建与部件分割**

基于 TRELLIS 1.0 框架，从多视角图片重建铰链物体的带部件标签的 3D 结构（labeled voxel），按 part 拆分后独立 decode 为 mesh，用于下游物理仿真（Isaac Sim / MuJoCo）。面向 AAAI 2027 投稿，同时服务小米工厂场景（车辆零件等工业铰链物体）。

**Core Value:** 端到端的铰链物体重建系统：多视角图片 → 带部件标签的 3D 结构 → 按 part 独立 decode → 可用于仿真的铰链物体资产。

### Constraints

- **GPU**: 本地 RTX 4090 做开发和 smoke test，后续用小米 H200 集群全量训练
- **时间线**: AAAI 2027 投稿（预计 2026 年 8 月前，~4 个月）
- **框架**: 基于 TRELLIS 1.0，Stage 1/2 不动
- **代码组织**: 训练代码在 scripts/，模型定义在 TRELLIS-arts/trellis/models/
- **训练设施**: 参考 TRELLIS 原版 trainer，但各阶段必须独立，YAML 配置驱动
- **坐标系**: PartNet-Mobility Y-up → TRELLIS Z-up，+90° X 旋转
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

> 注：本节原本由 GSD-1 的 codebase mapper 自动生成，源文件 `.planning/codebase/STACK.md`
> 已归档到 `docs/archive/v0.1.0-planning/codebase/STACK.md`，本节改为手维护快照。

### 语言 / 运行时
- Python 3.10 - 数据处理脚本、训练代码、模型定义
- Bash - 管线编排（`run_pipeline.sh`、`scripts/ops/data_pipeline/launch_dataset_preview.sh`、`scripts/inference/*.bash`）
- YAML/JSON - 训练 / 推理配置（`scripts/train/configs/`、`TRELLIS-arts/configs/arts/`）
- Conda 环境 `arts-gen` —— **统一环境**，覆盖数据处理 + TRELLIS 训练 + 推理
  - 老的 `artsvox` / `trellis` 双环境已 deprecated（见 Environment 节）
  - 上游 `submodules/dataset_toolkits/run_pipeline.sh` 写死 `EXPECTED_CONDA_ENV="dataset_toolkits"`，
    本项目忽略此校验，继续用 `arts-gen`

### 核心框架
- PyTorch 2.4.0 + CUDA 11.8 - 模型训练 / 推理
- TRELLIS 1.0 (改造版，位于 `TRELLIS-arts/`) - Flow Matching + DiT 架构
- Open3D / trimesh - 点云、网格、体素操作
- Blender 4.4.0 (项目相对 symlink `software/blender-4.4.0-linux-x64/blender`) - 多视角渲染
- torch.distributed - 分布式训练
- DINOv2-L/14-reg - 图像特征提取
- 冻结 CLIP text encoder (ViT-L/14) - PartMMDiT 的 part name 语义条件
- spconv (cu118) / kaolin / nvdiffrast / diffoctreerast / vox2seq - 稀疏 3D 与可微光栅
- safetensors - 模型权重序列化
- 训练日志: tensorboard + wandb
- 测试框架: pytest（散落使用，无统一配置）；本项目 `arts-gen` 默认未装 pytest

### 关键依赖（按需）
torch / torchvision / xformers / flash-attn / open3d / trimesh / spconv / kaolin /
nvdiffrast / diffoctreerast / vox2seq / safetensors / Pillow / pillow-simd /
opencv-python-headless / imageio[+ffmpeg] / numpy / scipy / transformers / tqdm /
easydict / pandas / lpips / rembg + onnxruntime / ninja

### 数据 / 权重路径
- 原始数据: `data/PhysX-Mobility/raw/partseg/{object_id}/`
- 生成数据: `data/PhysX-Mobility/arts/reconstruction/`
- 预训练权重: `pretrained/ckpts/`（需单独下载）
- CLIP ViT-L/14 本地权重: `/robot/data-lab/jzh/art-gen/weights/clip-vit-large-patch14`
- `TORCH_HOME` 指向 `submodules/TRELLIS.1`（DINOv2 hub cache）

### 平台
- Linux (Ubuntu) + NVIDIA GPU + CUDA 11.8
- 本地开发: RTX 4090（24 GB，batch_size=1）
- 全量训练: 小米 H200 集群（参考；尚未上线）
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

> 注：原 GSD-1 自动生成块已归档至 `docs/archive/v0.1.0-planning/codebase/CONVENTIONS.md`；
> 本节改为手维护精简版。详情仍以归档文档为准。

### 命名 / 风格
- 全部 Python 用 `snake_case`（脚本名、函数、变量）；TRELLIS 框架类用 `PascalCase`，Mixin 类后缀 `Mixin`
- 私有 helper 加下划线前缀（`_empty_voxel_result()`）
- 常量 `UPPER_SNAKE_CASE`（`RESOLUTION = 64`, `NUM_VIEWS = 12`, `DINOV2_DIM = 1024`）
- 路径变量后缀统一：`*_dir`（目录）/ `*_path`（文件）
- 缩进 4 空格，行宽 ~100-120，无强制 linter / formatter / pre-commit

### 错误处理 / 日志
- 数据处理脚本实现「断点续传」：检查输出文件存在则跳过
- 失败暴露：禁止用大块 try/except 或静默 fallback 隐藏 bug（见 `/home/mi/AGENTS.md` 与 v0.1.0 lessons）
- 日志：`print()` + `[INFO]/[WARN]/[ERROR]` 前缀 是默认；训练用 wandb + tensorboard

### 入口脚本
- 数据处理脚本均为独立可执行文件（`if __name__ == '__main__': main()`），统一 `argparse`
- 常用参数: `--data_root`, `--num_angles`, `--workers`, `--object_ids`, `--start_idx`, `--end_idx`
- 配置：训练用 YAML（`scripts/train/configs/`、`TRELLIS-arts/configs/arts/`），CLI flag 覆盖
- 模块注册: `TRELLIS-arts/trellis/models/__init__.py` 是工厂中心，`from_pretrained()` 加载 `{path}.json` + `{path}.safetensors`

### 并行
- CPU 任务: `multiprocessing.Pool` + worker 接收单 tuple 参数 + 返回 `{'id', 'status', 'errors'}` 字典
- GPU 任务: 单进程 + `tqdm` + batch 切分
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

> 注：原 GSD-1 自动生成块已归档至 `docs/archive/v0.1.0-planning/codebase/ARCHITECTURE.md`；
> 本节改为手维护精简版，描述与代码现状一致的高层结构。

### 模式概览
- **数据管线**：11 步 default profile（+ 2 步 dev-only，共 13 步）串行管线，
  权威实现在 `submodules/dataset_toolkits/`；详细 step 列表见上方 Data Pipeline 节
- **训练框架**：基于 TRELLIS 1.0 改造的 Sparse Latent Flow Matching 系统
- **部署**：本地 RTX 4090 做开发 / smoke test；全量训练规划用小米 H200 集群
- **坐标系**：PartNet-Mobility Y-up → TRELLIS Z-up（绕 X 轴 +90°）

### 层次
- 用途: 数据处理（PartNet-Mobility → TRELLIS 训练数据）
- 位置: `submodules/dataset_toolkits/pipeline/` (canonical)
- 入口: `submodules/dataset_toolkits/run_pipeline.sh` 或本仓库 wrapper `scripts/ops/data_pipeline/launch_dataset_preview.sh`
- 依赖: Open3D, Trimesh, Blender, DINOv2, TRELLIS SS-VAE encoder
- 消费: 训练层（通过 part_completion / part_synthesis manifests）

- 用途: 模型定义（Sparse Structure VAE、Structured Latent Flow、Part SS Latent Flow 等）
- 位置: `TRELLIS-arts/trellis/models/`
- 包含: SLatFlowModel (DiT)、SparseStructureFlowModel、SS-VAE encoder/decoder、ElasticSLatFlowModel、PartSSLatentFlowModel、PartMMDiTModel
- 依赖: `TRELLIS-arts/trellis/modules/`（sparse conv、transformer、attention）

- 用途: PyTorch Dataset 实现
- 位置: `TRELLIS-arts/trellis/datasets/`
- 包含: StandardDatasetBase 基类, SLat / ImageConditionedSLat / PartSSLatentFlowDataset
- 依赖: manifest jsonl + render / token / latent 产物

- 用途: 训练循环 / 损失 / Flow Matching 扩散
- 位置: `TRELLIS-arts/trellis/trainers/`
- 继承链: Trainer → BasicTrainer → FlowMatchingTrainer + (CFG / Text / Image) Mixin

- 用途: 底层神经网络组件
- 位置: `TRELLIS-arts/trellis/modules/`
- 包含: SparseTensor, SparseConv3d, ModulatedSparseTransformerCrossBlock, attention kernels

- 用途: 3D 表示格式
- 位置: `TRELLIS-arts/trellis/representations/`
- 包含: Gaussian, Mesh/FlexiCubes, Radiance Field, Octree

- 用途: 端到端推理 API
- 位置: `TRELLIS-arts/inference.py`（公共函数：`run_ss_flow` / `decode_ss` / `run_part_ss_latent_flow` / `run_slat_flow` / `decode_slat`）
- 编排: `pipeline/0[1-3]_*.py` 是 thin orchestrator，`run_pipeline.sh`（项目根，**不**是 dataset_toolkits 那个）链 5 步推理

### 关键抽象
- **SparseTensor**：`coords (N×4, batch+xyz) + feats (N×C)`，替代 dense 张量节省显存（`TRELLIS-arts/trellis/modules/sparse/basic.py`）
- **Flow Matching DiT**：SparseResBlock3d 输入 / 输出 + ModulatedSparseTransformerCrossBlock 中间 + skip connection
- **Model loader**：`{path}.json` (config) + `{path}.safetensors` (weights)，`from_pretrained()` 工厂

### 错误处理
- 数据脚本: 单样本失败 → log + `continue`（断点续传由"输出已存在则跳过"实现）
- 渲染调度: `subprocess.run()` 检查 returncode，失败 sample 打 `[ERROR]` 跳过
- Dataset 加载: 单样本异常时随机另选样本，避免阻塞训练
- Trainer: 内置 gradient clip、NaN 检测、checkpoint 自动保存
- **推理 API**：禁止静默兜底（见 v0.1.0 lessons "数据契约必须端到端对齐" 与 `run_part_ss_latent_flow` mask/token 契约）
<!-- GSD:architecture-end -->

## Lessons Learned (v0.1.0 Review, 5 Rounds)

### TRELLIS Trainer MRO 必须一次性审完

继承 TRELLIS Trainer 时，**必须追踪所有调用路径**，不能只看 `training_losses()`：

- `get_cond()` — 训练时条件编码（ImageConditionedMixin 会调 encode_image）
- `get_inference_cond()` — 推理/snapshot 时条件编码（同上）
- `vis_cond()` — 可视化时条件处理
- `snapshot_dataset()` — 训练启动前的数据集可视化
- `snapshot()` — init/final/resume 时的模型采样可视化
- `run_snapshot()` — 实际采样逻辑

**规则**: 继承 ImageConditioned*Trainer 但使用预编码 tokens 时，必须 override: `get_cond`, `snapshot_dataset`, `snapshot`。否则预编码 tokens 会被当作原始图像送进 DINOv2。

### 训练入口必须最小依赖

`import trellis` 会通过 `__init__.py` 拉入 pipelines/renderers/rembg 等推理依赖。训练入口应使用**手动包注册**（`types.ModuleType`），只注册 models/modules/trainers/utils/datasets。参考 `scripts/train/tests/test_lora.py` 的 `_setup_trellis_imports()`。

### 数据契约必须端到端对齐

写新代码时必须确认**上游产出格式**和**下游消费格式**一致：
- assembler manifest 格式 (dict + samples) vs dataset/validate 期望的格式
- DINOv2 tokens 存储格式 (单 key 'tokens' [V,T,D]) vs 检查脚本假设的格式 (多 key 'view_*')
- checkpoint 目录名 (TRELLIS 用 `ckpts/` 不是 `checkpoints/`)
- resume 时 `load_dir` + `step` 必须都传，且不能再被 `pretrained_ckpt` 覆盖

### 文档不能说高

文档描述的能力必须和代码实际能力对齐。如果某项检查在当前数据格式下只走 fallback 分支，就写清楚是 fallback，不要写成"完整校验"。

## Workflow Enforcement (Superpowers)

For non-trivial work, default to the superpowers flow:

1. **Skim `.gsd/STATE.md` once for context** (frozen, read-only) — it summarizes what was already decided/built up to M001/S02.
2. **For new work**, invoke the relevant superpowers skill via the `Skill` tool — `brainstorming` to scope, `writing-plans` to design, `executing-plans` to drive it. Do **not** create new files under `.gsd/milestones/...`.
3. **Tiny ad-hoc edits** (1 file, no design needed) → just do it, no plan needed.
4. **Multi-step or risky changes** → produce a plan via `writing-plans`, then execute via `executing-plans`. Use `verification-before-completion` before declaring done.

Legacy GSD plugin commands — `/gsd:quick`, `/gsd:debug`, `/gsd:plan-phase`, `/gsd:execute-phase`, `/gsd:profile-user`, etc. — are **deprecated for this project**. Don't invoke them; their `.gsd/`-writing side effects would re-activate the abandoned hierarchy.

<!-- Removed: GSD:profile block — superseded by superpowers workflow (see Workflow section above). -->
