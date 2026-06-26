# Object Post Process — MJCF 关节编辑器

针对重建后的 3D 物体资产，提供基于 Web 的 MJCF 关节编辑工具。加载物体的 MJCF XML 和 OBJ 网格，在浏览器中可视化编辑关节锚点、轴向、范围和 body 层级结构，保存修改后的 XML 或导出为 USD 供 Isaac Sim 使用。

## 功能

```
输入:  assets/object_assets/<name>/mjcf/<name>.xml + OBJ 网格 + 贴图
编辑:  Web 3D 关节编辑器 (Three.js)
输出:  修改后的 MJCF XML  或  USD (通过 Isaac Sim 导出)
```

- 铰链关节 (hinge) 和滑动关节 (slide) 的实时滑块预览
- 关节锚点编辑：拖拽手柄或直接输入 x/y/z 数值
- 关节轴向编辑：拖拽旋转环或直接输入 x/y/z 数值
- Body 合并：将多个 body 合并到同一关节下
- 每次保存自动通过 MjSpec.compile() 校验 XML 合法性
- 通过单进程 Isaac Sim GUI 服务导出 USD 并自动可视化

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10 - 3.12 | 运行时 |
| NVIDIA GPU | CUDA 12.4+ | MuJoCo GPU 渲染 / Isaac Sim |
| NVIDIA 驱动 | >= 535 | CUDA 12.4 支持 |
| [uv](https://docs.astral.sh/uv/) | >= 0.4 | Python 包管理 |
| Conda | Anaconda / Miniconda | Isaac Sim 环境管理（仅 USD 导出需要） |

### Python 依赖（自动安装）

| 包 | 版本约束 | 用途 |
|----|---------|------|
| flask | >= 3 | Web 服务 |
| mujoco | >= 3 | MJCF 解析与编译校验 |
| numpy | >= 1.26, < 2 | 数值计算 |
| torch | == 2.4.0 (cu124) | Isaac Sim 依赖的 PyTorch |
| plyfile | >= 1.1 | PLY 点云读写 |
| scipy | >= 1.12 | 空间计算 |
| Pillow | >= 10 | 图像处理 |
| packaging | >= 24 | 版本解析 |
| setuptools | >= 70 | 构建工具 |

### Conda 环境（仅 USD 导出需要）

USD 导出功能需要 Isaac Sim / IsaacLab 环境：

| 环境名 | 用途 | 关键依赖 |
|--------|------|---------|
| `env_isaaclab` | Isaac Sim 导出服务 | IsaacLab, isaacsim.asset.importer.mjcf, omni.usd |

如果不需要 USD 导出，可以跳过 Conda 环境，只使用 Web 编辑器。

## 新机器从零配置（推荐流程）

先判断你要跑哪一档：

| 目标 | 必须配置 | 不需要配置 |
|------|----------|------------|
| 只用 MJCF Web 编辑器 / 3DGS Viewer | `uv` + `.post_process` | Conda、Isaac Sim |
| 还要点击 **Export USD** 导出到 Isaac Sim | `uv` + `.post_process` + Conda `env_isaaclab` | 无 |

### 0. 准备代码和资产

项目代码和 `assets/` 目录需要放在同一个 `post_process` 根目录下：

```bash
cd /path/to/post_process

# 应该能看到这些路径
ls README.md pyproject.toml
ls assets/object_assets
ls assets/scene_assets
```

如果 `assets/` 是单独同步/拷贝的，先把它放回项目根目录；否则编辑器没有可加载的物体或场景。

### 1. 安装 uv（主项目包管理器）

如果机器上还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

如果已经有 `uv`，直接确认版本即可：

```bash
uv --version
```

### 2. 创建项目 Python 环境

```bash
cd /path/to/post_process

# Python 运行环境固定放在 .post_process
uv venv --python 3.10 .post_process

# 安装本项目及 pyproject.toml 中声明的依赖
uv pip install --python .post_process/bin/python -e .
```

安装完成后做一次最小验证：

```bash
.post_process/bin/python -c "import flask, mujoco, numpy; import object_post_process.server, scene_reconstruction.server; print('OK')"
```

看到 `OK` 就表示 Web 编辑器和 3DGS Viewer 的 Python 依赖已经装好。

### 3. 启动 MJCF Web 编辑器

本机使用：

```bash
.post_process/bin/python object_post_process/web_editor.py --port 8080
```

远程机器使用，例如通过 Tailscale IP 访问：

```bash
.post_process/bin/python object_post_process/web_editor.py --host 0.0.0.0 --port 8080 --no-browser
```

然后在浏览器打开：

```text
http://<机器IP>:8080/object-post-process/
```

例如 Tailscale IP 是 `100.x.x.x` 时：

```text
http://100.x.x.x:8080/object-post-process/
```

只在可信网络或 Tailscale 内使用 `--host 0.0.0.0`。

### 4. 启动 3DGS Scene Viewer

```bash
.post_process/bin/python -m scene_reconstruction.web_viewer --host 0.0.0.0 --port 8083 --no-browser
```

浏览器打开：

```text
http://<机器IP>:8083/scenes/
```

### 5. 可选：配置 Conda / Isaac Sim USD 导出

只有需要 **Export USD** 时才做这一步。主项目依赖仍然由 `uv` 管理；Conda 只负责 Isaac Sim / IsaacLab。

本项目**不负责安装 Isaac Sim / IsaacLab / `env_isaaclab`**。请用户按自己机器的 CUDA、驱动、Isaac Sim、IsaacLab 版本要求自行安装，并保证最终存在一个可用的 Conda 环境：

```text
env_isaaclab
```

本项目只做两件事：

1. 启动时执行 `conda activate env_isaaclab`
2. 在该环境里运行 `utils/isaac_export_service.py`

安装好后，用下面命令确认环境存在：

```bash
conda --version
conda env list | grep env_isaaclab
```

确认已有 `env_isaaclab` 后，用完整启动脚本：

```bash
scripts/start_object_post_process.sh
```

脚本会自动寻找常见位置的 `conda.sh`，包括：

- `$HOME/miniconda3/etc/profile.d/conda.sh`
- `$HOME/anaconda3/etc/profile.d/conda.sh`
- `/opt/conda/etc/profile.d/conda.sh`

如果 Conda 装在其他位置，手动指定：

```bash
CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh scripts/start_object_post_process.sh
```

完整启动后：

- Web 编辑器：`http://127.0.0.1:8080/object-post-process/`
- Isaac 导出服务：`http://127.0.0.1:8081/health`

### 6. 常见问题快速判断

| 现象 | 处理 |
|------|------|
| `uv: command not found` | 重新执行 uv 安装命令，并 `export PATH="$HOME/.local/bin:$PATH"` |
| `ModuleNotFoundError` | 确认使用 `.post_process/bin/python`，并重新执行 `uv pip install --python .post_process/bin/python -e .` |
| 页面资产列表为空 | 检查 `assets/object_assets/<name>/mjcf/` 是否存在 |
| 3DGS 场景为空 | 检查 `assets/scene_assets/<scene>/3dgs_compressed.ply` 或 `3dgs_standard.ply` 是否存在 |
| `Export USD` 不可用 | 说明 Isaac 导出服务没启动；不影响普通编辑和保存 XML |
| `conda.sh not found` | 设置 `CONDA_SH=/path/to/conda.sh` 后重新运行完整启动脚本 |

## 日常重建 Python 环境

```bash
cd /path/to/post_process

# 1. 创建虚拟环境（Python 3.10）
uv venv --python 3.10 .post_process

# 2. 激活并安装依赖
source .post_process/bin/activate
uv pip install -e .
```

安装完成后验证：

```bash
python -c "import flask, mujoco, numpy; import object_post_process.server, scene_reconstruction.server; print('OK')"
```

## 使用

### 仅 Web 编辑器（不需要 Isaac Sim）

```bash
source .post_process/bin/activate
python object_post_process/web_editor.py --port 8080
```

浏览器打开 `http://127.0.0.1:8080/object-post-process/`，从下拉菜单选择资产，编辑关节，点击 **Save XML** 保存。

### 完整启动（编辑器 + Isaac Sim USD 导出）

```bash
scripts/start_object_post_process.sh
```

该脚本启动两个进程：

1. **Web 编辑器**（端口 8080）— 使用 `.post_process` 虚拟环境
2. **Isaac Sim 导出服务**（端口 8081）— 使用 Conda `env_isaaclab` 环境

Isaac Sim 首次启动需要 60-90 秒加载。就绪后页面上的 **Export USD** 按钮变为可用，点击后导出到 `assets/object_assets/<name>/usd/<name>.usd`，并在 Isaac Sim GUI 窗口中自动打开可视化。

按 `Ctrl+C` 同时停止两个进程。

### 自定义端口

```bash
scripts/start_object_post_process.sh --port 9000 --isaac-port 9001
```

## Scene Reconstruction (3DGS Viewer)

这是独立于 `object_post_process` 的 3DGS PLY 查看器，只负责浏览 `assets/scene_assets/<name>/` 下的场景，不包含选点、SAM3、提取或其他编辑功能。

启动方式：

```bash
python -m scene_reconstruction.web_viewer --port 8083
```

浏览器打开：

```text
http://127.0.0.1:8083
```

## 资产目录结构

```
assets/object_assets/<name>/
    mjcf/
        <name>.xml              MJCF 源文件（编辑器读写的对象）
        assets/                 OBJ 网格和贴图文件
    usd/                        导出的 USD 文件（自动生成）
        <name>.usd
```

## 项目结构

```
object_post_process/
    server.py               Flask 服务（API 路由 + Isaac 代理）
    mjcf_parser.py          MJCF XML 解析 -> 预览 manifest
    xml_saver.py            编辑器状态 -> 校验后的 MJCF XML
    web_editor.py           CLI 启动入口

utils/
    frontend/
        mjcf_joint_editor.html   Three.js 编辑器前端
    shared_libs/                 Three.js / OBJLoader / TransformControls 等
    isaac_export_service.py      单进程 Isaac Sim GUI + Flask 导出服务
    usd_exporter.py              MjcfConverter 封装
    isaaclab_launcher.py         Isaac 环境自动探测

scripts/
    start_object_post_process.sh 双进程启动脚本（含信号清理）
```

## 编辑器操作

| 操作 | 控制方式 |
|------|---------|
| 旋转视角 | 鼠标左键拖拽 |
| 平移视角 | 鼠标右键拖拽 |
| 缩放 | 滚轮 |
| 选中 body | 点击右侧面板的 body 列表 |
| 编辑锚点 | 点击 **Edit Anchor**，拖拽手柄或输入 x/y/z 数值 |
| 编辑轴向 | 点击 **Edit Axis**，拖拽旋转环或输入 x/y/z 数值 |
| 预览关节运动 | 拖动关节滑块 |
| 合并 body | 选中多个 body 后点击 **Merge** |
| 保存 | 点击 **Save XML**（自动 MjSpec 编译校验） |
| 导出 USD | 点击 **Export USD**（需要 Isaac Sim 就绪） |
