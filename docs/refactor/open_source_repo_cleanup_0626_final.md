# arts-gen 开源收尾报告 2026-06-26

证据目录：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626
```

## 入口补齐

新增并实跑：

- `scripts/train/part_promptable_seg/run_train.bash`
- `scripts/eval/run_ee_eval.bash`

训练 launcher smoke：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/run_train_bash_smoke_after.log
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z
```

ee-eval launcher smoke：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/run_ee_eval_bash_smoke_after.log
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z
```

## scripts/train 清理

保留：

```text
scripts/train/_ddp_common.sh
scripts/train/_slurm_common.sh
scripts/train/part_promptable_seg/
scripts/train/part_promptable_seg/run_train.bash
scripts/train/ss_flow_art_train.bash
scripts/train/slat_flow_art_train.bash
scripts/train/part_ss_eval_platform.bash
```

归档到 `scripts/_archive/2026-06-train-launchers/`：

```text
part_flow_train.bash
part_mmdit_train.bash
part_predictor_train.bash
part_ss_latent_flow_eval_decode.bash
part_ss_latent_flow_full_eval.bash
part_ss_latent_flow_single_view_full_eval.bash
part_ss_latent_flow_single_view_test_export.bash
part_ss_latent_flow_single_view_train.bash
part_ss_latent_flow_test_export.bash
part_ss_latent_flow_train.bash
ss_flow_global_z_train.bash
```

最终计数：

```text
scripts/dev_files=0
scripts_train_files=12
archive_train_launchers=11
trash_files=0
```

## 根目录和大依赖

root scratch 先进入 `scripts/_trash/2026-06-open-source-root-cleanup/` 登记，再删除。清单在：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/move_delete_manifest.txt
```

大依赖移出仓库，根目录保留本地软链：

```text
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/software
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/sam3d_cu118_deps
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/sam3d_cu118_src_deps
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/libnvidia-gpucomp.so.550.144.03
/mnt/robot-data-lab/jzh/art-gen/local-deps/arts-gen-root/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl
```

`.gitignore` 和 `.dockerignore` 已覆盖数据、输出、权重、大依赖、个人工具和 scratch。diff：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/gitignore.diff
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/dockerignore.diff
```

Dockerfile 不再 `COPY sam3d_cu118_src_deps/`，需要 SAM3D 环境时挂载或复制 bundle 到 `/workspace/arts-gen/sam3d_cu118_deps` 后再 `INSTALL_SAM3D_ENV=1`。

## 残留引用检查

活跃树 grep 结果为 0：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/grep_scripts_dev_run0617_after.txt
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/grep_legacy_launchers_after.txt
```

## 回归验证

strict-load 四个最新 part-seg ckpt：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_strict_load_after.log
```

结果：

```text
part_promptable_seg_full_S_0616-1 step=50000 missing=[] unexpected=[]
part_promptable_seg_full_S_0618-1 step=100000 missing=[] unexpected=[]
part_promptable_seg_full_S_0618-2 step=100000 missing=[] unexpected=[]
part_promptable_seg_full_M_0612-2 step=6000 dim=384 depth=8 missing=[] unexpected=[]
```

训练 smoke：

```text
step 1/5 total 8.2201
step 2/5 total 7.1198
step 3/5 total 6.8534
step 4/5 total 5.7179
step 5/5 total 5.7820
```

ckpt：

```text
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z/ckpts/latest.pt
/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z/ckpts/step_5.pt
```

二者均包含 `model`、`optimizer`、`step=5`。

ee-eval smoke：

```text
status=passed
done=1
failed=0
part_seg_ckpt=/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt
```

产物：

```text
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/metrics.json
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__mesh.png
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__gaussian.png
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__diagnostic.png
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/_platform_runs/held/004d1e9e13934e319094151a4fad823f-0/real-B/voxel.npz
/mnt/robot-data-lab/jzh/art-gen/ee-eval/run-ee-eval-smoke-20260626T093309Z/_platform_runs/held/004d1e9e13934e319094151a4fad823f-0/real-B/parts/part_00_voxel.npz
```

静态检查：

```text
bash -n 新 launcher 和保留 launcher：通过
python -m py_compile 关键 py：通过
python -c import trellis + PromptablePartLatentSegNet：通过
pytest launcher/articulator 小测试：43 passed
pytest legacy launcher smoke：4 passed
```

## README

根 `README.md` 已重写为中文开源说明，包含：

- 项目输入输出和模型结构文字框图
- S/M ckpt 关键超参
- strict-load 契约
- 新训练 launcher 与 ee-eval launcher
- 大依赖安装/恢复说明
- 数据权重路径、环境、license/provenance
