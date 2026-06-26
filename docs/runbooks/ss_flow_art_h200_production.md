# SS Flow Art H200 生产训练 Runbook

> 历史背景：本阶段曾命名为 stage2（SS Flow），Phase 9 hard cut 后统一更名为 SS Flow Art；
> 本 runbook 中所有命令引用已切换到新路径（launcher: `scripts/train/ss_flow_art_train.bash`），
> 仅个别历史背景说明保留旧名以便回溯。

> 本 runbook 不在本地 4090 执行，只在 H200 集群按 slot 调度时使用。
> 本地 4090 的 smoke test 验证（100 步 × 2 模式）记录在
> `docs/archive/phase02/ss_flow_test_smoke_checklist.md` 和 `output/smoke_test_phase2_{full,lora}.json`。

**覆盖 v1 需求:** S3-01, S3-02, S3-03, S3-04, S3-05, S3-06

---

## Prerequisites

- [ ] 集群节点已挂载 `data/PhysX-Mobility/arts/reconstruction/`（全量 ~2000 obj × 10 angles；assembler 产出）
- [ ] 集群节点已挂载 `data/PhysX-Mobility/arts/manifest.json`（assembler 新格式
      `{"samples": [{"object_id", "angle_idx", "complete"}]}`）
- [ ] 已下载 `pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors`（1.08 GB，
      TRELLIS 1.0 官方 HuggingFace: JeffreyXiang/TRELLIS-image-large）
- [ ] conda env `trellis` 已就绪（参考 `TRELLIS-arts/setup.sh`）
- [ ] 集群已配置 `TORCH_HOME=submodules/TRELLIS.1`（TRELLIS DINOv2 hub 缓存路径）
- [ ] 至少 8 × H200 80GB 同节点或 DDP 同步可达
- [ ] wandb 账号已 login（或设置 `WANDB_API_KEY`；离线模式见 Troubleshooting）

---

## Step 0: Train/Val Split (MUST run BEFORE first training)

**这是 H200 启动训练的前置步骤**。生产 manifest 在 assembler 产出后必须先走
obj_id 级确定性 md5 hash 切分（Phase 2 CONTEXT.md D-07/08/09）：

```bash
cd /path/to/arts-reconstruction && \
python scripts/train/split_manifest.py \
  --input data/PhysX-Mobility/arts/manifest.json \
  --output_dir data/PhysX-Mobility/arts/ \
  --val_ratio 0.1 \
  --seed 42
```

产出两个文件：

- `data/PhysX-Mobility/arts/manifest_train.json`
- `data/PhysX-Mobility/arts/manifest_val.json`

此后两种模式训练都在 CLI 里显式设置 `data.manifest_path=arts/manifest_train.json`
覆盖 `mv_4view.yaml` 的默认 `arts/manifest.json`。

**确定性保证**：同一个 obj_id 跨机器跨运行始终落同一 split（md5 hash 直接决定 bucket，
不依赖随机数；见 `split_manifest.py::_hash_bucket`）。

---

## Step 1: Full Fine-tune (50k steps)

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  scripts/train/ss_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml \
  data.manifest_path=arts/manifest_train.json
```

- **Checkpoint 路径约定**：`output/ss_flow_art_mv_4view/ckpts/{denoiser,denoiser_ema0.9999,misc}_step{N:07d}.pt`
  （TRELLIS Trainer 的 `ckpts/` 子目录约定，CLAUDE.md Lessons Learned #3；注意是 `ckpts/` 不是 `checkpoints/`）
- **Wandb**: `project=arts-reconstruction`, `name=ss-flow-art-mv4view`, `tags=[ss_flow_art, mv4view, mode=full]`
- **预估 walltime**: ~48-72 h on 8×H200（50k steps × batch_size_per_gpu=1 ×
  8 gpu = 400k effective samples；实测 4090 单卡 100 步 ~100 秒，H200 8 卡保守外推）
- **峰值 VRAM**: < 40 GB per H200（fp16 AMP + batch=1）
- **磁盘占用**: 每 `i_save=5000` 快照产出 ~8.5 GB × 10 snaps = **85 GB** 量级
  （`fp16_mode=inflat_all` 下 denoiser + denoiser_ema + misc 三份，参考 Phase 1 checklist
  §Checkpoint 大小分析）。注意启动前检查 H200 节点磁盘剩余。

---

## Step 2: LoRA Fine-tune (并行，独立 run)

```bash
torchrun --nproc_per_node=8 --master_port=29501 \
  scripts/train/ss_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view_lora.yaml \
  data.manifest_path=arts/manifest_train.json
```

- **Checkpoint 路径**: `output/ss_flow_art_mv_4view_lora/ckpts/` （同目录结构）
- **Wandb**: `name=ss-flow-art-mv4view-lora`, `tags=[ss_flow_art, mv4view, mode=lora]`
- **预估 walltime**: ~36-60 h（trainable 参数 < 2%，forward 时间主导；反向和 optimizer 显著更快）
- **VRAM**: < 30 GB per H200（LoRA 下 optimizer 状态只覆盖 ~5 M 参数）
- **磁盘占用**: ~4.4 GB × 10 snaps = **44 GB** 量级（SS Flow Art 4090 smoke 实测）
- **冻结守恒证据**: SS Flow Art smoke test 已证明 100 步内
  `non_lora_changed=0 / lora_changed=240 / freeze_ok=true / trainable_params_ratio=0.9118%`
  （见 `output/smoke_test_phase2_lora.json`）

**LoRA 和 full 可以并行跑**（不同 `master_port`，不同 `output_dir`，不同 wandb run）。

---

## Step 3: Evaluate (任一 ckpt → Voxel IoU on val set)

```bash
# Full ckpt, 4 视角（SS Flow Art 主工况）
python scripts/train/eval_voxel_iou.py \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml \
  --ckpt output/ss_flow_art_mv_4view/ckpts/denoiser_step0050000.pt \
  --manifest data/PhysX-Mobility/arts/manifest_val.json \
  --views 4 \
  --num_steps 25 \
  --output output/eval/ss_flow_art_full_v4.json

# 同一 ckpt，1 视角（验证 view dropout 鲁棒性，Phase 2 CONTEXT D-13 (历史阶段编号)）
python scripts/train/eval_voxel_iou.py \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml \
  --ckpt output/ss_flow_art_mv_4view/ckpts/denoiser_step0050000.pt \
  --manifest data/PhysX-Mobility/arts/manifest_val.json \
  --views 1 \
  --num_steps 25 \
  --output output/eval/ss_flow_art_full_v1.json

# LoRA ckpt（需要 base pretrained + LoRA 训练产物 两个文件）
python scripts/train/eval_voxel_iou.py \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view_lora.yaml \
  --ckpt pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \
  --lora_ckpt output/ss_flow_art_mv_4view_lora/ckpts/denoiser_step0050000.pt \
  --manifest data/PhysX-Mobility/arts/manifest_val.json \
  --views 4 \
  --num_steps 25 \
  --output output/eval/ss_flow_art_lora_v4.json
```

`eval_voxel_iou.py` 特点：

- 按 `x_0` / `cond` 读 dataset（Phase 2 D-42 (历史阶段编号)，绝不引用 legacy `part_labels` 键）
- 加载顺序和 `train.py:229-298` 对称：build → load base safetensors (strict=False) →
  apply_lora (if `lora.enabled=true`) → `model.load_state_dict(torch.load(lora_ckpt), strict=False)` → eval()
- `--lora_ckpt` 指向的是 **TRELLIS `BasicTrainer.save()` 产出的单文件 `.pt`**
  （peft-wrapped full state_dict，键名前缀 `base_model.model.*`），**不是 PEFT
  规范的 adapter 目录**。Review Round 1 #1 发现旧版本 `--lora_adapter` 和
  `lora_utils.load_lora_weights` 的目录语义不兼容，已统一到 `.pt` 单文件契约。
- 用 `trellis.pipelines.samplers.FlowEulerSampler`（手动 `types.ModuleType` 注册，避免
  `import trellis` 拉入 pipelines 重依赖）
- 输出 JSON: `{config, ckpt, lora_ckpt, manifest, views, num_steps, n_samples, mean_iou, per_sample}`

**IoU 数字量级参考**：latent 空间 IoU（per-voxel `abs().sum(0) > 1e-3` 的
occupancy 重叠），不是 64³ 原始体素空间 IoU —— decode 回 64³ 需要 Stage 1 VAE
decoder，SS Flow Art scope 不做（D-31）。真正的 64³ IoU 留给 Phase 5 评估系统。

---

## Step 4: Resume (集群 walltime 中断后)

**CRITICAL — resume 参数是 argparse 顶层 flag，不是 OmegaConf override。**
`train.py:173-178` 把 `--load-dir` 和 `--resume-step` 定义为
`parser.add_argument(...)`；`train.py:243-244` 直接读 `args.load_dir` / `args.resume_step`。

写成 `training.load_dir = ...`（带空格，OmegaConf 形式 —— 这是错误写法的示例） 的 OmegaConf 形式会被 `args.overrides` 吞掉并**静默忽略**，
导致"resume 命令"实际走 fresh run 并覆盖原 checkpoint。**这会丢数据，务必按下面写法。**

正确写法：

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  scripts/train/ss_flow_art_train.bash \
  --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml \
  --load-dir output/ss_flow_art_mv_4view \
  --resume-step 25000
```

说明：

- `--load-dir` 指向 output **根目录**（不是 `ckpts/` 子目录）。Trainer 会自己拼
  `{load_dir}/ckpts/denoiser_step{resume_step:07d}.pt`
- `--resume-step` 必须和 `--load-dir` **一起传**；单传任一个都会被视为 fresh run
- **不需要** `training.pretrained_ckpt=null` 的额外 override —— `train.py:243-250`
  的 Round 14 fix 已经在 `is_resuming=True` 时自动跳过 pretrained_ckpt 加载，
  避免覆盖已恢复的 checkpoint 权重
- 两条参数是顶层 argparse，位置可以放在 `--config` 之前或之后，但**必须**带
  `--` 前缀，**不能**写成 `training.load_dir = ...`（带空格，OmegaConf 形式 —— 这是错误写法的示例）
- LoRA 模式 resume 用同样的 flag，只是 `--config` 换成 `mv_4view_lora.yaml`

**已知非阻塞问题**（Phase 1 checklist §3 记录）: Python `random` module 的 RNG
状态未被 BasicTrainer 保存到 checkpoint，resume 后 `random.sample(range(12), 4)`
选出的视角组合与连续跑不同，loss 数值会和未中断的连续训练有轻微差异。生产训练
不影响收敛，但如果将来需要严格 bit-reproducible resume，需要 checkpoint 额外
存 `random.getstate() / torch.get_rng_state()`。

---

## Troubleshooting

| 症状 | 处理 |
|------|------|
| **CUDA OOM** | 已是最小 batch（`batch_size_per_gpu=1`）。尝试 `model.args.use_checkpoint=true` 开 gradient checkpointing。仍不行则减 `num_views=2` 或 LoRA-only。 |
| **Wandb 离线 / 连不上** | `wandb.mode=offline` 作为 CLI override；训练结束后 `wandb sync output/ss_flow_art_*/wandb/` 补传。 |
| **Flash-attn 报错** | 降级到 `ATTN_BACKEND=sdpa` 环境变量。SS Flow Art smoke test 已验证 `flash_attn` 正常工作（Round 21 移除硬编码默认）。 |
| **Loss NaN in step 0-5** | 检查 `training.grad_clip=1.0` 生效；必要时降 lr 到 5e-6；再不行切 `fp16_mode=amp`。 |
| **Resume 没生效 / 从 step 0 重新开始** | 十有八九是写成了 `training.load_dir = ...`（带空格，OmegaConf 形式 —— 这是错误写法的示例） 的 OmegaConf 形式。改成顶层 `--load-dir` / `--resume-step` 即可（见 Step 4）。 |
| **LoRA trainable_ratio > 5%** | `target_modules` 匹配到非 attention 层（例如 MLP）。切回 `lora.target_modules=all_attn` 预设。 |
| **LoRA non_lora_changed > 0** | peft 注入顺序错或 base 权重未正确冻结。查 `train.py:290-298` 确认 `apply_lora_to_model` 在 `load_state_dict` 之后（Phase 1 Round 8 fix）。 |
| **H200 节点磁盘满** | Full 模式 ~85 GB / 50k 步，LoRA ~44 GB。把 `training.i_save` 从 5000 改大（例如 10000）或把老 checkpoint 自动迁移到共享存储。 |
| **Snapshot 跳过 "预编码 tokens 无法可视化"** | 已知非阻塞行为（Phase 1 Round 21）。Stage 2 dataset 存的是 DINOv2 tokens 不是图像，trainer snapshot 无法生成 PIL 可视化。训练本身不受影响。 |

---

## References

- **Phase 2 smoke 验证记录**: `docs/archive/phase02/ss_flow_test_smoke_checklist.md`, `output/smoke_test_phase2_full.json`, `output/smoke_test_phase2_lora.json`
- **Phase 1 Trainer lessons**: `CLAUDE.md` §Lessons Learned, `docs/archive/v0.1.0_milestone.md`
- **配置继承链**: `scripts/train/configs/base/base.yaml` → `mv_4view.yaml` → `mv_4view_lora.yaml`
- **Resume argparse 定义**: `scripts/train/ss_flow_art_train.bash:173-178`
- **Resume guard (pretrained_ckpt 自动跳过)**: `scripts/train/ss_flow_art_train.bash:243-250` (Round 14 fix)
- **Trellis 手动 import shim**: `scripts/train/ss_flow_art_train.bash:43-73` / `scripts/train/eval_voxel_iou.py::_setup_trellis_imports`
- **Dataset 接口契约**: `scripts/train/dataset.py::MvImageConditionedSLatDataset.__getitem__` 返回 `{'x_0': [C,H,W,D], 'cond': [V*T, D]}`（**不**返回 legacy `part_labels` 键，Round 21 删除）
- **DDP launcher**: `scripts/train/launchers/launch_ddp.sh`

---

*Phase: 02-stage2-mv-finetune / Plan 02-01 / Task 7 (历史阶段编号；现 SS Flow Art)*
*Status: H200 生产启动手册，可直接复制粘贴执行*
*不包含: baseline_sv.py（deferred to Phase 5 评估系统，CONTEXT D-37）*
