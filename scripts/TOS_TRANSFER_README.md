# TOS 传输脚本

把本地 arts-reconstruction 项目（含 conda env、pretrained、Blender、代码）打包上传到 TOS，供云开发机拉下来跑批量 pipeline。

## 文件清单

| 脚本 | 干啥 | 大小估计（压缩后）|
|---|---|---|
| `tos_push_env.sh` | conda-pack `arts-gen` env，上传到 `tos://.../env/arts_gen_env.tar.gz` | ~7 GB |
| `tos_push_weights.sh` | tar `pretrained/`（DINOv2 + TRELLIS encoder/decoder + dinov2 repo） | ~1.4 GB |
| `tos_push_software.sh` | tar `software/`（Blender 4.4.0 linux-x64） | ~300 MB |
| `tos_push_code.sh` | tar 项目代码（含 submodule 工作树，跳过 data/pretrained/software/.git） | ~50 MB |
| `tos_pull_env.sh` | 在 dev 机上下载并解压到 `/opt/venvs/arts-gen/`，自动 `conda-unpack` | — |
| `tos_pull_weights.sh` | 下载并解压到 `pretrained/` | — |
| `tos_pull_software.sh` | 下载并解压到 `software/`，验证 `blender --version` | — |
| `tos_pull_code.sh` | 下载并解压到 `/root/code/arts-gen/`，调 setup_cloud_storage 链接 vePFS | — |
| `tos_push_data.sh` | tar `data/smoke_test/`（PartNet-Mobility smoke 集，含 renders + voxels + DINOv2 tokens + SS latents 全部预处理产物）| ~1 GB |
| `tos_pull_data.sh` | 下载并解压回 `<repo>/data/smoke_test/` | — |

默认 TOS 根：`tos://robot-data-lab/arts-reconstruction`。所有脚本都用环境变量 `TOS_ROOT` / `TOS_URI` 可覆盖。

## 本地（push 端）一次性准备

```bash
# 1. tosutil 已装在 /usr/local/bin/tosutil ✓
# 2. conda-pack 已装在 /home/mi/anaconda3/bin/conda-pack ✓
# 没装就：/home/mi/anaconda3/bin/python -m pip install conda-pack
```

## 本地：上传所有 4 个 archive

```bash
cd /home/mi/jzh/AAAI2027/arts-reconstruction

bash scripts/ops/tos/tos_push_env.sh        # arts-gen → TOS
bash scripts/ops/tos/tos_push_weights.sh    # pretrained/ → TOS
bash scripts/ops/tos/tos_push_software.sh   # software/ (Blender) → TOS
bash scripts/ops/tos/tos_push_code.sh       # repo 代码 → TOS
```

每个脚本独立可重跑。env 是最慢的（conda-pack ~3-5 min + 上传 ~5-10 min 视带宽），其他都很快。

## Dev 机（pull 端）从零起步

```bash
# 0. tosutil 凭证已就位（dev 机一般已配好）
which tosutil

# 1. bootstrap：先抓一份 tos_pull_code.sh 拉代码到 /root/code/arts-gen/
mkdir -p /tmp/bootstrap && cd /tmp/bootstrap
tosutil cp tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz arts_code.tar.gz
tar -xzf arts_code.tar.gz ./scripts/ops/tos/tos_pull_code.sh
bash scripts/ops/tos/tos_pull_code.sh        # 默认 REMOTE_DIR=/root/code/arts-gen
cd /root/code/arts-gen

# 2. 拉 env、weights、software（解压到 /root/code/arts-gen/{.venv,pretrained,software}）
bash scripts/ops/tos/tos_pull_env.sh         # → /opt/venvs/arts-gen/   (5.6 GB)
bash scripts/ops/tos/tos_pull_weights.sh     # → pretrained/        (1.3 GB)
bash scripts/ops/tos/tos_pull_software.sh    # → software/          (68 MB)
bash scripts/ops/tos/tos_pull_data.sh        # → data/smoke_test/    (smoke-test 数据)

# 3. 激活环境 + smoke check
source /opt/venvs/arts-gen/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
./software/blender-4.4.0-linux-x64/blender --version

# 4. data/smoke_test/ 已含 PartNet-Mobility 的 renders/voxels/tokens/latents
#    可以直接用于 inference / 评估。如果要再跑 Mi pipeline 1→11，需要单独
#    传 Xiaomi raw 数据（DATA_DIR=/path/to/Mi 走一次 tos_push_data.sh）
ls data/smoke_test/
```

## TOS 路径布局（预期）

```
tos://robot-data-lab/arts-reconstruction/
├── env/arts_gen_env.tar.gz         # conda env（含 Python 3.10.20 + torch+cu118 + ...）
├── weights/arts_pretrained.tar.gz  # pretrained/
├── software/arts_software.tar.gz   # software/blender-4.4.0-linux-x64/
└── code/latest.tar.gz              # 代码 + submodule 工作树
```

## 注意

- **Python 版本**：env 自带 Python 3.10.20（不动 dev 机的 system 3.10.12）。`source /opt/venvs/arts-gen/bin/activate` 后 `python` 解析到 env 的 3.10.20。
- **CUDA 11.8**：env 里 torch 是 `2.4.0+cu118`，dev 机驱动需 ≥ 520。
- **跨 distro**：conda-pack 输出包含 libstdc++ / libc++ 等运行时。Ubuntu / Debian / CentOS 7+ 都 OK；非常老的发行版（CentOS 6 这种 glibc < 2.17）可能链接失败。
- **headless OK**：Blender 用 `--background` 渲染，无桌面环境也能跑。
- **重传 = 覆盖**：`tosutil cp` 默认覆盖；想保留旧版本就改 `TOS_URI` 加版本后缀。
