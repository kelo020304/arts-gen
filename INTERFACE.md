# FROZEN v1: VLM -> arts-gen reconstruct 接口

本文档冻结 VLM 对接的第一版重建接口。调用方只负责提供多视角 RGB 图像和同顺序的 part label mask；`arts-gen` 不在这个接口内运行 SAM3D，也不直接 `import sam3d`。

## 输入约定

0617 accepted e2e 管线使用 4 个输入视角。外部调用时必须按同一物体、同一姿态的 4 视角顺序传入 `images` 和 `masks`，两者一一对应。盘上训练/eval 样本的渲染分辨率是 512x512；接口允许其他分辨率，但 part prompt 会按现有 promptable segmentation 训练契约下采样/池化到 512x512。

跨视角铁律：同一个 part 在所有视角必须使用同一个正整数 label id，0 固定表示背景。VLM 侧负责保证该一致性，`reconstruct()` 会校验 mask 形状、dtype、label 集合和空 part。

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CkptConfig:
    # 默认值来自 docs/runbooks/0617_128ee_correct_slat_flow.md。
    ss_flow_ckpt: str | Path = "/robot/data-lab/jzh/art-gen-output/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
    part_seg_ckpt: str | Path = "/robot/data-lab/jzh/art-gen-output/part_promptable_seg_full_S_0615-5/ckpts/step_50000.pt"
    ss_decoder_ckpt: str | Path = "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    slat_flow_ckpt: str | Path = "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
    slat_mesh_decoder_ckpt: str | Path = "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
    slat_gaussian_decoder_ckpt: str | Path = "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
    ss_steps: int = 20
    ss_cfg_strength: float = 7.5
    ss_fusion_mode: str = "concat"
    slat_steps: int = 25
    slat_seed: int = 42
    part_voxel_threshold: float = 0.5
    part_joint_candidate_mode: str = "proposal"  # proposal | full_occ ablation
    part_joint_refine: bool = False
    part_joint_refine_iters: int = 1
    part_joint_refine_pairwise: float = 3.0
    part_joint_refine_margin: float = 0.0
    part_joint_refine_margin_quantile: float = 0.01
    part_joint_refine_neighborhood: int = 6
    part_joint_refine_min_vote_gain: float = 0.0
    part_joint_refine_preserve_small_classes: int = 32
    part_joint_save_logits: bool = False
    output_dir: str | Path | None = None


@dataclass(frozen=True)
class ReconstructInput:
    images: Sequence[str | Path | Image.Image]
    masks: Sequence[str | Path | np.ndarray]  # 每张 [H,W] int32: 0=bg, 正整数=part id
    part_info: Mapping[str, Any] | str | Path | None = None
    ckpt_config: CkptConfig | Mapping[str, Any] | None = None
```

The `part_joint_*` controls require a checkpoint trained with
`args.joint_seg=true`. Enabling refinement/logit export, or selecting
`full_occ`, with a legacy independent or latent checkpoint is an error rather
than a silent no-op. The guarded refiner remains disabled by default.

`part_info` 可选；如果提供，推荐沿用数据盘 `reconstruction/part_info/{object_id}/part_info.json` 的结构，其中 `parts[*].label` 是 mask 中的正整数 label，`parts[*].type`/key 作为 part 名称来源。

## 输出约定

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Part:
    part_id: int
    label: str
    voxel_coords: np.ndarray      # [N,3], int32, canonical 64^3 grid
    mesh: Any | None              # decoder 返回的 mesh 对象；未请求/失败则抛错，不静默 None
    gaussian: Any | None          # decoder 返回的 Gaussian 对象
    mesh_path: Path | None
    gaussian_path: Path | None
    joint: dict | None            # 当前无 cotrain joint ckpt，返回 None
    metadata: dict


@dataclass(frozen=True)
class ArtObject:
    labeled_voxel: np.ndarray     # [64,64,64] int32: 0=bg, 正整数=part id
    whole_voxel_coords: np.ndarray
    parts: list[Part]
    scale: dict
    metadata: dict
```

坐标系固定为 `canonical_grid`，分辨率固定为 64。`metadata["joint_status"] == "TODO_no_cotrain_ckpt"` 表示当前接口尚未接 cotrain joint ckpt，所有 `Part.joint` 为 `None`。

## 调用示例

```python
from scripts.inference.reconstruct import CkptConfig, ReconstructInput, reconstruct

result = reconstruct(
    ReconstructInput(
        images=[
            "view_0.png",
            "view_1.png",
            "view_2.png",
            "view_3.png",
        ],
        masks=[
            "mask_0.npy",
            "mask_1.npy",
            "mask_2.npy",
            "mask_3.npy",
        ],
        part_info="part_info.json",
        ckpt_config=CkptConfig(output_dir="/mnt/robot-data-lab/jzh/art-gen/vlm-smoke/my_object"),
    )
)

print(result.labeled_voxel.shape)
for part in result.parts:
    print(part.part_id, part.label, part.mesh_path, part.gaussian_path)
```

CLI smoke 示例：

```bash
/opt/venvs/arts-gen/bin/python scripts/inference/reconstruct.py \
  --smoke-from-dataset \
  --out-dir /mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/part0
```

## 管线边界

- SS stage：复用 `TRELLIS-arts/inference.py::run_ss_flow_from_tokens`，0617 concat checkpoint，4 视角 official DINOv2 tokens。
- Part stage：复用 `PromptablePartLatentSegNet`，外部 `masks` 作为 2D prompt；不跑内部 SAM3D。
- SLat stage：复用 `run_slat_flow_from_tokens`，对 whole-object voxel 坐标只跑一次 SLat flow，然后按 exact sparse coords 切出每个 part 的 SLat。
- Decode：复用 `decode_slat_assets`，写出 mesh/gaussian 资产时使用现有 asset writer。
- SAM3D：本接口不导入、不启动 SAM3D。若以后需要 SAM3D 跨进程服务，沿用 `scripts/inference/infer_stage.py` 的 `SAM3D_VENV_PYTHON`/glue 契约另开 v2。
