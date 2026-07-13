# dataset_toolkits

`dataset_toolkits` 是一套用于统一处理多关节 3D 物体数据集的工具链，当前面向
`PhysX-Mobility`、`HSSD` 等数据源。它将不同数据集的原始结构转换为统一格式，
并串联当前主线处理阶段：关节变换、canonical transform、体素化、valid-parts manifest、
`part_complete` 16-view 渲染、DINOv2 特征抽取、SS latent 编解码、VLM 单图 JSONL、
Part Completion 单图 manifest 和 Web 预览。

Step 12/13 的 part-synthesis SLat cache / manifest 仍在开发中，暂不列入稳定 pipeline。

## Stable pipeline stages

当前主线结构以 `docs/PIPELINE.md` 为准。默认 profile 跑 Step 01-11；Step 12/13 是可选后续分支，不在默认流程中。

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `pipeline/01_joint_transformation.py` | 生成 `joint_transforms` 和 `part_info.json` |
| 2 | `pipeline/02_build_canonical_transforms.py` | 生成 canonical transform，统一渲染/体素坐标 |
| 3 | `pipeline/03_voxelize.py` | 生成整体 surface 和 per-part voxel |
| 4 | `pipeline/04_build_valid_parts_manifest.py` | 生成 valid-parts manifest，作为有效 target source of truth |
| 5 | `pipeline/05_render.py` | 默认只跑 `part_complete` 16-view RGB + mask；150-view 需显式 `--sets` |
| 6 | `pipeline/06_extract_feature.py` | 默认提取 `part_complete` 16-view DINOv2 features |
| 7 | `pipeline/07_encode_ss_latents_per_part.py` | 生成 overall/per-part SS latent |
| 8 | `pipeline/08_decode_ss_latents.py` | 将 SS latent 解码回 voxel 并计算 QC 指标 |
| 9 | `pipeline/09_build_vlm_dataset_manifest.py` | 基于 `part_complete` 前 8 固定视角构建单图 VLM JSONL |
| 10 | `pipeline/10_build_part_completion_manifest.py` | 构建单图 Part Completion manifest 和 label masks |
| 11 | `pipeline/11_web_preview.py` | 生成 VLM / Part Completion Web 预览入口 |

快速主线建议使用 `--profile base`，跑 `1,2,3,4,5,6,7,8,9,10,11`，适合构建当前 VLM + Part Completion 训练数据和预览；Step 07/08 不跳过，用于生成 SS decoder 对比结果。

## 第一次使用：配置官方环境

```bash
cd dataset_toolkits
bash scripts/bootstrap_dataset_toolkits_env.sh
conda activate dataset_toolkits
python scripts/check_environment.py --config configs/PhysX-Mobility.yaml
```

`dataset_toolkits` 是唯一支持的 conda 环境。不要把稳定 pipeline 拆到多个 conda 环境里运行。
`run_pipeline.sh` 只接受当前激活环境里的 `python3`，不支持 `PYTHON=/other/python` 覆盖。
`scripts/bootstrap_dataset_toolkits_env.sh` 会先创建同名 conda env，再在 torch 已安装后单独安装
`flash-attn`；不要把 `flash-attn` 合并回 `envs/dataset_toolkits.yaml` 的同一轮 pip 安装。
Blender 是外部可执行文件，不属于 conda 环境；路径在对应数据集配置文件的 `render.blender`
字段里设置，例如 `configs/PhysX-Mobility.yaml`、`configs/HSSD.yaml`。
preflight 不会下载模型、不会猜路径、不会自动兜底；缺少环境、CUDA、Blender、数据目录或权重时直接失败。

preflight 通过后运行默认完整 pipeline：

```bash
bash run_pipeline.sh --config configs/PhysX-Mobility.yaml
```

只运行指定步骤：

```bash
bash run_pipeline.sh --config configs/PhysX-Mobility.yaml --steps 4
```

运行单个步骤仍然使用同一个激活后的 `dataset_toolkits` 环境：

```bash
python pipeline/07_encode_ss_latents_per_part.py \
  --config configs/PhysX-Mobility.yaml
```

如只需要重写整体表面 latent，而不触碰 per-part latent：

```bash
python pipeline/07_encode_ss_latents_per_part.py \
  --config configs/PhysX-Mobility.yaml \
  --latent-scope overall \
  --overwrite
```

本地模型代码和权重不提交到 git。DINOv2 / TRELLIS 路径必须显式写在对应 dataset config 里；
推荐使用项目相对路径指向 `pretrained/` 下的本地 checkpoint 或 symlink（该目录已由 `.gitignore` 排除）。
Step 11 所需 Three.js vendor 文件已放在 `vendor/three/0.160.0/`，运行时不会联网下载。

## 数据集位置在哪里改

数据集根目录只在对应的 YAML 配置里改，不要在 `run_pipeline.sh` 或 pipeline 脚本里硬改路径。

配置文件在 `configs/`：

- PhysX-Mobility：`configs/PhysX-Mobility.yaml`
- HSSD：`configs/HSSD.yaml`
- 3D-FUTURE：`configs/3D-FUTURE.yaml`

每个配置文件顶部都有：

```yaml
dataset_name: PhysX-Mobility
data_root: /home/cfy/cfy/ccc/nip/base_line/arts-reconstruction/data/PhysX-Mobility
```

把 `data_root` 改成该数据集实际所在的**绝对路径**即可，例如：

```yaml
dataset_name: PhysX-Mobility
data_root: /your/absolute/path/to/PhysX-Mobility
```

pipeline 的所有输入和派生产物都会从这个 `data_root` 派生出来，包括：

```text
<data_root>/raw/finaljson/
<data_root>/raw/partseg/
<data_root>/joint_transforms/
<data_root>/part_info/
<data_root>/canonical_transforms/
<data_root>/renders/
<data_root>/reconstruction/
<data_root>/vlm/
<data_root>/manifests/
<data_root>/preview/
```

所以迁移数据集位置时，只需要：

1. 把数据目录移动到新位置；
2. 修改对应 `configs/<Dataset>.yaml` 的 `data_root`；
3. 运行时继续传同一个配置文件：

```bash
bash run_pipeline.sh --config configs/PhysX-Mobility.yaml
```

新增数据集时，新建一个 `configs/<Dataset>.yaml`，至少要设置正确的 `dataset_name`、绝对路径
`data_root`、`joint_transform`、`render`、`voxel`、`feature`、`trellis` 和 `vlm` 配置。
数据目录需要满足统一 raw 输入约定：`<data_root>/raw/finaljson/*.json` 和 `<data_root>/raw/partseg/<object_id>/objs/`。

同一个 config 还必须显式配置本地可执行文件和权重路径：

```yaml
render:
  resolution: 512
  blender: /absolute/path/to/blender

feature:
  model: dinov2_vitl14_reg
  dinov2_repo: pretrained/dinov2
  torch_hub_dir: pretrained/torch_hub

trellis:
  root: pretrained/TRELLIS
  ss_encoder: pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16
  ss_decoder: pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16
  slat_encoder: pretrained/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16

vlm:
  image_prefix: /path/prefix/seen/by/training
```

`data_root` 和 `render.blender` 必须是绝对路径；`feature.*` 与 `trellis.*` 可以是绝对路径或项目相对路径。
这些字段都是强约束：路径缺失就失败，不做隐式下载、不切换备用路径。
`vlm.image_prefix` 只影响 Step 09 写入 VLM JSONL 的图片路径：脚本会把本地 `data_root` 中从
`data/<Dataset>` 开始的相对路径接到这个 prefix 后面。训练机看到的数据根前缀不同，就改这里；
如果训练就在当前机器当前目录跑，应让它和 `data_root` 的上级路径保持一致。

## More docs

- [docs/PIPELINE.md](docs/PIPELINE.md)：11 个稳定 pipeline 步骤的详细说明。
- [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)：输入、中间产物和最终输出的数据格式规范。
