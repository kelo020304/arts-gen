# arts-gen

`arts-gen` 是一个面向可动 3D 物体的多视角重建仓库。当前开源主线是：

```text
多视角 RGB + 2D part label mask
  -> 带 part 标签的 64^3 voxel
  -> 按 part 解码 mesh / Gaussian
```

关节估计不走 part-seg 主线，保留在 `scripts/kinematic_solver/` 与 `kin_test/` 作为单独入口。本文档记录的是 2026-06-26 在 dev 机 `/root/code/arts-gen` 上实跑过的命令、路径和模型契约。

## 模型结构

核心模型类名和 import path 是权重加载契约的一部分，不改名：

```python
trellis.models.part_seg.promptable_latent_seg.PromptablePartLatentSegNet
```

文字框图：

```text
多视角 RGB images
  -> run_ss_flow / run_ss_flow_from_tokens
  -> z_global: SS latent [B, 8, 16, 16, 16]

2D part masks [B, V=4, 512, 512]
  -> PointMaskEncoder(fg_points) 或 MaskEncoder2D(cnn_grid)
  -> mask prompt tokens

PromptablePartLatentSegNet
  z_global
    -> encode_cells: 1x1x1 3D conv + 3D position
    -> 16^3 trunk tokens
    -> Transformer blocks:
         LocalConv + self-attn + cross-attn(mask prompt tokens) + MLP
    -> head1: 16^3 cell mask
    -> voxel head: 64^3 occupied voxel candidates 上逐体素 part / background 分类
    -> per-part labeled voxel

per-part voxel + whole SLat token
  -> run_slat_flow_from_tokens / run_slat_flow_per_part
  -> decode_slat_assets
  -> mesh.glb / Gaussian .ply 或 ee-eval preview PNG
```

推理管线：

```text
images
  -> run_ss_flow
  -> decode_ss(whole occupancy coords, 64^3)
  -> part seg(mask prompt)
  -> run_slat_flow_from_tokens once for whole object
  -> subset SLat by each part voxel coords
  -> decode_slat_assets(mesh + gaussian per part)
```

`PromptablePartLatentSegNet` 的 `z_global` 输入固定为 `[B,8,16,16,16]`。`encode_cells` 把 SS latent 和可选 xyz 坐标编码成 `16^3` 个 trunk tokens。每个 block 对 trunk tokens 做局部 3D conv、自注意力，再对 2D mask prompt tokens 做 cross-attn。当前 ee-eval 使用 voxel route：先预测 `16^3` cell，再在 whole occupancy 的 `64^3` 候选点上做逐体素 part 分类，输出 part-labeled voxel。

从 ckpt strict-load 得到的关键超参：

| ckpt | step | dim | depth | heads | mask_encoder | semantic_classes | voxel_embedding_dim | 说明 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `part_promptable_seg_full_S_0616-1` | 50000 | 256 | 6 | 8 | `fg_points` | 4245 | 0 | ee-eval 默认 ckpt |
| `part_promptable_seg_full_S_0618-1` | 100000 | 256 | 6 | 8 | `fg_points` | 4365 | 16 | S + voxel embedding |
| `part_promptable_seg_full_S_0618-2` | 100000 | 256 | 6 | 8 | `fg_points` | 4365 | 16 | S + voxel embedding |
| `part_promptable_seg_full_M_0612-2` | 6000 | 384 | 8 | 8 | `fg_points` | 320 | 0 | M size，不同 dim/depth |

ckpt 加载契约：

- 模型类名和 import path 不变：`trellis.models.part_seg.promptable_latent_seg.PromptablePartLatentSegNet`。
- 推理 parser 在 `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py`，通过 `_model_args_from_ckpt(ckpt)` 从 `ckpt["args"]` 和 state dict 解析 `dim/depth/head_depth/heads/mask_encoder/use_xyz/use_voxel_head/semantic_classes/voxel_embedding_dim`。
- strict-load 使用 `_clean_state_dict(ckpt["model"])` 去掉 `module.` 前缀，然后 `model.load_state_dict(..., strict=True)`。S 和 M 都不能写死模型尺寸。

## 仓库结构

```text
TRELLIS-arts/                         TRELLIS fork；模型、inference.py、0617 eval platform
scripts/inference/                    VLM reconstruct API 与推理入口
scripts/eval/                         统一 eval CLI、regression smoke、ee-eval launcher
scripts/eval/tasks/                   0617 ee-eval single/batch runner
scripts/train/part_promptable_seg/    part-prompt-seg 训练、打包、split 工具和 launcher
scripts/train/                        SS/SLat 核心训练 launcher 与 DDP/Slurm helper
scripts/tools/                        diagnostics/qc/render/export/roundtrip/report 等工具
scripts/ops/                          setup、TOS 同步、数据流水线脚本
scripts/kinematic_solver/             关节/运动学后处理入口
docs/runbooks/                        当前 runbook
docs/refactor/                        清理报告和重构记录
submodules/                           上游依赖源码快照
```

本地数据和权重路径：

```text
/mnt/robot-data-lab/jzh/art-gen/data/phyx-verse
/mnt/robot-data-lab/jzh/art-gen/data/realappliance
/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5
/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg
/mnt/robot-data-lab/jzh/art-gen/ckpt/tre-ss-flow/tre-ss-concat-0616-1
/root/code/arts-gen/pretrained/TRELLIS-image-large
```

## 环境

主环境是 CUDA 11.8 / Python 3.10：

```bash
/opt/venvs/arts-gen/bin/python --version
```

常用环境变量：

```bash
export PYTHONPATH=/root/code/arts-gen:/root/code/arts-gen/TRELLIS-arts
export SPCONV_ALGO=native
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SS_FLOW_FUSION_MODE=concat
```

SAM3D 使用独立环境。本仓 VLM reconstruct API 接收调用方传入的 part masks，不在 API 内部 import 或运行 SAM3D。

## 大依赖安装

大依赖和本地资产不进入开源仓库，已由 `.gitignore` 与 `.dockerignore` 排除：

```text
software/
sam3d_cu118_deps/
sam3d_cu118_src_deps/
libnvidia-gpucomp.so.*
nvdiffrast-*.whl
pretrained/
data/
checkpoints/
output/
outputs/
runs/
```

dev 机当前把这些大件移到了：

```text
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/
```

并在仓库根目录保留本地软链接，方便现有脚本继续跑。新机器恢复方式：

```bash
bash scripts/ops/tos/tos_pull_software.sh
bash scripts/ops/tos/tos_pull_sam3d_cu118_deps.sh
bash scripts/ops/tos/tos_pull_weights.sh
bash scripts/ops/setup/setup_arts_gen.sh
```

Dockerfile 不再 `COPY sam3d_cu118_src_deps/`，因为该 bundle 是本地大依赖。需要在镜像内安装 SAM3D 环境时，先把 bundle 挂载或复制到 `/workspace/arts-gen/sam3d_cu118_deps`，再用 `--build-arg INSTALL_SAM3D_ENV=1` 构建。

## 加载权重

下面命令会逐个目录找最新 `step_*.pt`，按推理 parser 解析模型参数，并 strict-load：

```bash
PYTHONPATH=/root/code/arts-gen:/root/code/arts-gen/TRELLIS-arts \
SPCONV_ALGO=native ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/opt/venvs/arts-gen/bin/python - <<'PY'
from pathlib import Path
import gc, json, re, torch
from inference_pipeline.part_prompt_seg_stage import _clean_state_dict, _model_args_from_ckpt
from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet

roots = [
    Path("/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1"),
    Path("/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0618-1"),
    Path("/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0618-2"),
    Path("/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_M_0612-2"),
]

def step_of(path: Path) -> int:
    m = re.search(r"step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1

for root in roots:
    ckpt_path = sorted((root / "ckpts").glob("step_*.pt"), key=step_of)[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = _model_args_from_ckpt(ckpt)
    model = PromptablePartLatentSegNet(**model_args)
    incompatible = model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=True)
    print(json.dumps({
        "name": root.name,
        "path": str(ckpt_path),
        "class": f"{PromptablePartLatentSegNet.__module__}.{PromptablePartLatentSegNet.__name__}",
        "step": int(ckpt.get("step") or step_of(ckpt_path)),
        "model_args": model_args,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }, ensure_ascii=False))
    del model, ckpt
    gc.collect()
PY
```

2026-06-26 实跑日志：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/partseg_ckpt_strict_load_all_0626.log
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_strict_load_after.log
```

四个 ckpt 的 `missing=[]`、`unexpected=[]`，其中 M ckpt 解析为 `dim=384 depth=8 semantic_classes=320`。

## 训练 part-prompt-seg

标准 launcher：

```text
scripts/train/part_promptable_seg/run_train.bash
```

单卡 5 step smoke：

```bash
SMOKE=1 GPU_IDS=0 bash scripts/train/part_promptable_seg/run_train.bash
```

8 卡训练：

```bash
MODEL_SIZE=S NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 \
OUT_DIR=/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/my_run \
STEPS=100000 BATCH=16 \
bash scripts/train/part_promptable_seg/run_train.bash
```

2026-06-26 实跑输出：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/run_train_bash_smoke.log
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T091158Z
```

5 步 loss 均为有限值，`ckpts/latest.pt` 和 `ckpts/step_5.pt` 都包含 `model`、`optimizer`、`step=5`。

## ee-eval

标准 launcher：

```text
scripts/eval/run_ee_eval.bash
```

默认 part-seg ckpt：

```text
/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt
```

单 object smoke：

```bash
SMOKE=1 GPUS=0 bash scripts/eval/run_ee_eval.bash
```

1024 object run：

```bash
OUT_DIR=/mnt/robot-data-lab/jzh/art-gen/ee-eval/0626-1024-1 \
LIMIT=1024 TRAIN_COUNT=1024 HELD_COUNT=0 GPUS=0,1,2,3 \
bash scripts/eval/run_ee_eval.bash
```

2026-06-26 实跑输出：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/run_ee_eval_bash_smoke.log
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T091241Z/metrics.json
```

状态为 `passed`，`done=1 failed=0`。产物包括：

```text
metrics.json
*__summary.json
*__mesh.png
*__gaussian.png
*__diagnostic.png
_platform_runs/.../voxel.npz
_platform_runs/.../parts/part_*_voxel.npz
```

ee-eval 默认输出 mesh/Gaussian preview PNG；如需 MuJoCo mesh asset，可加 `--export-mujoco`。

## VLM reconstruct API

接口文档：

```text
INTERFACE.md
```

Python API：

```python
from scripts.inference.reconstruct import CkptConfig, ReconstructInput, reconstruct

result = reconstruct(
    ReconstructInput(
        images=["view_0.png", "view_1.png", "view_2.png", "view_3.png"],
        masks=["mask_0.npy", "mask_1.npy", "mask_2.npy", "mask_3.npy"],
        part_info="part_info.json",
        ckpt_config=CkptConfig(output_dir="/mnt/robot-data-lab/jzh/art-gen/vlm-smoke/my_object"),
    )
)

print(result.labeled_voxel.shape)  # (64, 64, 64)
```

mask 契约：

- `masks` 和 `images` 一一对应。
- 每个 mask 是 `[H,W] int32` label map。
- `0` 是背景，正整数是 part id。
- 同一个 part 在所有视角必须使用同一个 label id，这个跨视角一致性由 VLM 侧保证。

接口 smoke：

```bash
OUT=/mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/readme-verify-$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/code/arts-gen:/root/code/arts-gen/TRELLIS-arts \
SPCONV_ALGO=native ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa SS_FLOW_FUSION_MODE=concat \
/opt/venvs/arts-gen/bin/python scripts/inference/reconstruct.py \
  --smoke-from-dataset \
  --out-dir "$OUT" \
  --quick-steps
```

2026-06-26 实跑输出：

```text
/mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/readme-default-verify-20260626T080052Z/labeled_voxel.npy
/mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/readme-default-verify-20260626T080052Z/part_01_flower_petals/mesh.glb
/mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/readme-default-verify-20260626T080052Z/part_01_flower_petals/gaussian.ply
```

`--quick-steps` 只用于接口 smoke。正式重建可以去掉该参数，使用默认 `ss_steps=20` 和 `slat_steps=25`。

## License / Provenance

根目录 `LICENSE` 是本仓代码许可。`NOTICE` 记录第三方来源和本地权重/数据再分发注意事项。TRELLIS 原始许可保留在：

```text
TRELLIS-arts/LICENSE
submodules/TRELLIS.1/LICENSE
submodules/TRELLIS.2/LICENSE
```

其他上游来源包括：

```text
submodules/Hunyuan3D-2/LICENSE
submodules/Hunyuan3D-2/NOTICE
submodules/PhysX-Anything/LICENSE
```

`/mnt/robot-data-lab` 下的数据、模型权重和生成产物不因本仓 README 自动获得重新分发授权，发布前需要按各自来源检查许可。
