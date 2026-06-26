# TOS 代码同步记录

> 最后更新: 2026-05-06
> 范围: `arts-reconstruction` 代码同步到 TOS / 开发机的脚本、约定和执行记录。

## 当前约定

- 本 repo 使用本地脚本同步代码：
  - `scripts/ops/tos/tos_push_code.sh`
  - `scripts/ops/tos/tos_pull_code.sh`
  - `scripts/ops/setup/setup_cloud_storage.sh`
- 默认 TOS 路径：
  - `tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz`
- 默认开发机代码目录：
  - `$HOME/code/arts-reconstruction`
- 默认 vePFS 大文件目录：
  - `/robot/data-lab/arts-reconstruction`
- 当前开发机无 vePFS 写权限时，使用本地盘模式：
  - `USE_LOCAL_STORAGE=1`
  - `data/`、`runs/`、`outputs/`、`checkpoints/` 直接创建在 `$HOME/code/arts-reconstruction` 下。
- 代码包不包含文档：
  - `docs/` 被排除。
  - `code_update/` 被排除。
- 代码包包含本地 submodule working tree：
  - `submodules/` 下已 checkout 的代码会一起打包。
  - submodule 的 `.git`、`.cache`、`ckpts`、`checkpoints` 会被排除。
  - 目标是让无公网开发机不需要 `git submodule update --init`。
- 代码包不包含大文件和运行产物：
  - `data/`
  - `runs/`
  - `outputs/`
  - `checkpoints/`
  - `software/`
  - `.git/`
  - cache directories

## 本地上传命令

```bash
bash scripts/ops/tos/tos_push_code.sh
```

自定义快照路径：

```bash
TOS_URI=tos://robot-data-lab/arts-reconstruction/code/flow_fix_plan0.tar.gz \
  bash scripts/ops/tos/tos_push_code.sh
```

## 开发机首次拉取命令

```bash
CODE_DIR="${CODE_DIR:-$HOME/code/arts-reconstruction}"
mkdir -p "$CODE_DIR"
tosutil cp tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz /tmp/arts_reconstruction_code.tar.gz
tar -xzf /tmp/arts_reconstruction_code.tar.gz -C "$CODE_DIR"
cd "$CODE_DIR"
USE_LOCAL_STORAGE=1 bash scripts/ops/setup/setup_cloud_storage.sh
```

## 开发机后续更新命令

```bash
cd "$HOME/code/arts-reconstruction"
USE_LOCAL_STORAGE=1 bash scripts/ops/tos/tos_pull_code.sh
```

## Round 5 - 排除本地 software 工具目录

日期: 2026-05-06

### 内容
- `scripts/ops/tos/tos_push_code.sh` 新增排除 `software/`。
- 原因是本地 `software/blender-4.4.0-linux-x64*` 会把代码包从约 `123MB` 放大到约 `933MB`。
- 重新上传干净的 latest 代码包。

### 上传结果
- 上传目标:
  - `tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz`
- 上传大小:
  - `122.71MB`
- 上传结果:
  - TOS 返回 `Upload successfully`，status `200`。

### 验证
- `tar -tzf /tmp/arts_reconstruction_code.tar.gz | rg '^\\./software(/|$)'`: 无输出，确认未包含 `software/`。
- `tar -xOzf /tmp/arts_reconstruction_code.tar.gz ./TRELLIS-arts/inference.py | rg 'strict=True|num_steps: int \\| None'`: 确认 latest 包含 review 修复。

## Round 4 - 支持开发机本地盘存储模式

日期: 2026-05-06

### 内容
- `scripts/ops/setup/setup_cloud_storage.sh` 新增 `USE_LOCAL_STORAGE=1` 模式。
- 该模式不访问 `/robot/data-lab/arts-reconstruction`，而是在开发机代码目录下直接创建：
  - `data/`
  - `runs/`
  - `outputs/`
  - `checkpoints/`
- `scripts/ops/tos/tos_pull_code.sh` 会把 `USE_LOCAL_STORAGE` 传给 `setup_cloud_storage.sh`，并输出本地盘存储提示。

### 当前推荐命令
- 开发机首次拉取后执行：

```bash
USE_LOCAL_STORAGE=1 bash scripts/ops/setup/setup_cloud_storage.sh
```

- 开发机后续更新执行：

```bash
USE_LOCAL_STORAGE=1 bash scripts/ops/tos/tos_pull_code.sh
```

### 注意
- 本地盘模式适合当前没有 vePFS 权限的开发机。
- 代码上传包仍然排除 `data/`、`runs/`、`outputs/`、`checkpoints/`，所以开发机本地生成的数据和 checkpoint 不会被下一次代码同步覆盖上传。

### 上传结果
- 已重新执行 `bash scripts/ops/tos/tos_push_code.sh`。
- 上传目标:
  - `tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz`
- 上传大小:
  - `122.71MB`
- 上传结果:
  - TOS 返回 `Upload successfully`，status `200`。

## Round 1 - 新增同步脚本

日期: 2026-05-06

### 内容
- 参考 `/home/mi/jzh/AAAI2027/scene_gen/scripts` 新增 repo-local TOS 同步脚本。
- `tos_push_code.sh` 打包当前 repo 并上传到 TOS。
- `tos_pull_code.sh` 在开发机拉取代码包并解压。
- `setup_cloud_storage.sh` 将 `data/runs/outputs/checkpoints` 链接到 vePFS。

### 注意
- 本轮未执行真实 TOS 上传。
- 后续完成代码验证后再上传。

## Round 2 - 上传 flow fix 代码包

日期: 2026-05-06

### 内容
- 执行 `bash scripts/ops/tos/tos_push_code.sh`。
- 本地 archive:
  - `/tmp/arts_reconstruction_code.tar.gz`
- 上传目标:
  - `tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz`
- 上传大小:
  - `122.71MB`
- 上传结果:
  - TOS 返回 `Upload successfully`，status `200`。

### 注意
- 代码包由脚本排除 `docs/` 和 `code_update/`。
- 本地环境缺少 `torch` 和 `pytest`，因此上传的是已通过语法检查的 WIP 代码包，不是完整 pytest 验证后的最终包。

## Round 3 - 明确 submodule working tree 随代码包上传

日期: 2026-05-06

### 内容
- 更新 `scripts/ops/tos/tos_push_code.sh` 注释和 exclude 规则。
- 明确 `submodules/` 下本地已 checkout 的工作树会随代码包上传。
- 继续排除 submodule git metadata、cache 和 checkpoint 目录。
- 重新执行 `bash scripts/ops/tos/tos_push_code.sh`。

### 上传结果
- 上传目标:
  - `tos://robot-data-lab/arts-reconstruction/code/latest.tar.gz`
- 上传大小:
  - `122.71MB`
- 上传结果:
  - TOS 返回 `Upload successfully`，status `200`。

### 验证
- `bash -n scripts/ops/tos/tos_push_code.sh`: 通过。
- `tar -tzf /tmp/arts_reconstruction_code.tar.gz | rg '^\\./submodules/'`: 确认 archive 包含 submodule 内容。
- `tar -tzf /tmp/arts_reconstruction_code.tar.gz | rg '^\\./submodules/.*/\\.git(/|$)'`: 无输出，确认未包含 submodule `.git` 目录。
