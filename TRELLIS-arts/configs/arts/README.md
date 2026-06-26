# arts/ 训练配置

本目录承载 arts-reconstruction 项目所有训练 stage 的 YAML 配置：

- `base/base.yaml` — 全局默认值（optimizer / fp16_mode / wandb / lora 默认开关 / `stage:` 默认）
- `ss_flow_art/` — 多视角 SS Flow（原 stage2，paper Stage 2 SS-Flow）
- `slat_flow_art/` — 多视角 SLat Flow（原 stage4，paper Stage 4 SLat-Flow）
- `part_ss_latent_flow/` — 每个 target part 的 SS latent Rectified Flow（替代旧 dense categorical Part Flow）
- `part_predictor/` — Hungarian 部件预测（独立 stage）

## 为什么用 YAML + `_base_` 而非 JSON

TRELLIS 原版 `configs/{vae,generation}/*.json` 是**模型架构描述符**（DiT 层数、通道数等不变量）。
arts/ 是**训练 recipe**（lr / batch / lora rank / max_steps / output_dir / wandb name），需要 thin override 复用基线（`mv_4view_lora.yaml` 仅 4 行覆盖 `mv_4view.yaml`）。
YAML `_base_` 继承提供这个语义；JSON 没有原生继承。两者关注点不同，并存兄弟目录而非合并。

## `stage:` 字段（D-11）

`train_arts.py` dispatch 依据 YAML 顶层 `stage:` 字段（不接 `--stage` CLI flag）。
每个 stage 子目录的 mid-level yaml（`mv_4view.yaml` / `base.yaml`）显式声明 `stage:`；
thin override yaml（`mv_4view_lora.yaml`）通过 `_base_` 继承自动获得正确 stage 名。

合法值：`ss_flow_art` / `slat_flow_art` / `part_ss_latent_flow` / `part_predictor`。

## 对应表（旧路径 → 新路径）

| 旧 | 新 | 备注 |
|---|---|---|
| `scripts/train/configs/base/base.yaml` | `TRELLIS-arts/configs/arts/base/base.yaml` | + 新增 `stage:` 字段 |
| `scripts/train/configs/stage2/*.yaml` | `TRELLIS-arts/configs/arts/ss_flow_art/*.yaml` | output_dir / wandb 重命名；修 `stage3` 拼写 bug |
| `scripts/train/configs/stage4/*.yaml` | `TRELLIS-arts/configs/arts/slat_flow_art/*.yaml` | output_dir / wandb 重命名 |
| `scripts/train/configs/part_flow/*.yaml` | `TRELLIS-arts/configs/arts/part_ss_latent_flow/*.yaml` | 旧 dense categorical 路线已废弃 |
| `scripts/train/configs/part_predictor/*.yaml` | `TRELLIS-arts/configs/arts/part_predictor/*.yaml` | （命名不变） |

## `_base_` 继承规则

`_base_:` 路径 **相对当前 YAML 文件所在目录**（参考 `trellis.utils.arts.config_utils.load_config`）。
本目录结构与原 `scripts/train/configs/` 同构，所有 `_base_:` 引用在迁移后保持 0 修改：

- `ss_flow_art/mv_4view.yaml` 引用 `../base/base.yaml` → 解析到 `arts/base/base.yaml` ✓
- `ss_flow_art/mv_4view_lora.yaml` 引用 `mv_4view.yaml` → 同目录解析 ✓
- `part_predictor/decode_aware.yaml` 引用 `base.yaml` → 同目录解析 ✓
