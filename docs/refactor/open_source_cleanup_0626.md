# Open-source cleanup report 2026-06-26

## 1. Before / after

- Before snapshot: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/before_scripts_dev_files.txt`
- Before legacy dev-tree top-level Python count: 103.
- After legacy dev-tree top-level Python count: 0. The legacy dev tree is absent.
- Active `scripts/eval/` is self-contained for EE eval: `scripts/eval/run_eval.py` calls `scripts/eval/tasks/ee_0617_batch.py`, which calls `scripts/eval/tasks/ee_0617_single.py`.
- Residual old-path grep for the legacy dev path and moved-shim marker returned no active-tree matches, excluding bundled external dependency folders.

Current active anchors:

```text
scripts/eval/run_eval.py
scripts/eval/tasks/ee_0617.py
scripts/eval/tasks/ee_0617_batch.py
scripts/eval/tasks/ee_0617_single.py
scripts/train/part_promptable_seg/train_part_promptable_seg.py
scripts/train/part_promptable_seg/part_promptable_seg_utils.py
scripts/train/part_promptable_seg/pack_part_promptable_seg_dataset.py
scripts/train/part_promptable_seg/make_part_promptable_official_split.py
scripts/tools/lib/{joint_head.py,ss_vae_roundtrip.py,slat_vae_roundtrip.py}
scripts/tools/roundtrip/trellis_full_voxel_mesh_roundtrip.py
```

## 2. Delete / archive / move manifest

- Full recoverable archive: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/arts-gen-open-source-cleanup-trash-0626.tar.gz` (20 MB).
- Full archived/deleted manifest: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/deleted_archived_manifest_0626.txt` (740 entries).
- Archived legacy dev top-level Python list: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/deleted_dev_top_py_0626.txt` (88 entries after moving retained files out).
- Archived legacy dev all files list: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/deleted_dev_all_files_0626.txt` (214 entries).
- Deleted top-level moved shims: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/deleted_top_level_shims_0626.txt` (35 entries). Canonical implementations remain under `scripts/ops/`.
- Removed generated cache dirs: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/deleted_cache_dirs_0626.txt` (101 active-source cache dirs; bundled dependency caches excluded where appropriate).

Main moves:

```text
legacy trainer -> scripts/train/part_promptable_seg/train_part_promptable_seg.py
legacy trainer utils -> scripts/train/part_promptable_seg/part_promptable_seg_utils.py
legacy packer -> scripts/train/part_promptable_seg/pack_part_promptable_seg_dataset.py
legacy split maker -> scripts/train/part_promptable_seg/make_part_promptable_official_split.py
legacy EE single/batch runners -> scripts/eval/tasks/ee_0617_{single,batch}.py
scripts/eval/*diagnose*.py -> scripts/tools/diagnostics/
```

## 3. Part-seg original checkpoint load

Command used the existing inference ckpt parser path, then strict-loaded the model state:

```text
PART_SEG_CKPT_LOAD_OK
ckpt= /mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt
step= 50000
class= PromptablePartLatentSegNet
model_args= {'dim': 256, 'depth': 6, 'head_depth': 2, 'heads': 8, 'use_xyz': True, 'use_voxel_head': True, 'voxel_depth': 3, 'mask_encoder': 'fg_points', 'point_k_boundary': 32, 'point_k_interior': 32, 'point_resample_points': True, 'semantic_classes': 4245, 'voxel_embedding_dim': 0}
missing= []
unexpected= []
from_torch_load= True
```

Import smoke also passed:

```text
import ok PromptablePartLatentSegNet /mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-128ee True
```

## 4. EE eval 1-object proof

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venvs/arts-gen/bin/python scripts/eval/run_eval.py ee_0617 \
  --out-dir /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj \
  --limit 1 --train-count 1 --held-count 0 --gpus 0 \
  --selection-mode samples --sample-selection-unit objects \
  --slat-token-source live --overwrite-selection --force
```

Result:

```text
status=passed
object=phyx-verse 004d1e9e13934e319094151a4fad823f angle=0
backend=scripts/eval/tasks/ee_0617_batch.py
summary_count=1 done=1 failed=0
slat_flow_calls=1
component_count=5
```

Artifacts:

```text
metrics: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/metrics.json
summary: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__summary.json
voxel: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/_platform_runs/held/004d1e9e13934e319094151a4fad823f-0/real-B/voxel.npz
mesh: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__mesh.png
gaussian: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__gaussian.png
diagnostic: /mnt/robot-data-lab/jzh/art-gen/ee-eval/open-source-cleanup-1obj/phyx-verse__004d1e9e13934e319094151a4fad823f__angle_00__diagnostic.png
```

## 5. Verification

Passed:

```text
python -m py_compile scripts/eval/run_eval.py scripts/eval/tasks/ee_0617.py scripts/eval/tasks/ee_0617_single.py scripts/eval/tasks/ee_0617_batch.py scripts/train/part_promptable_seg/*.py scripts/tools/lib/*.py scripts/tools/roundtrip/trellis_full_voxel_mesh_roundtrip.py scripts/tools/render/render_voxel_eval_tripanel_flat.py scripts/tools/render/render_glb_open3d_preview.py TRELLIS-arts/part_ss_eval_platform/eval_ss_flow_ema_0615.py
python -m pytest tests/part_promptable_seg -q  # 9 passed
legacy dev-path and moved-shim grep  # no active-tree matches
legacy dev-tree top-level Python count  # 0
```

## 6. Open-source checklist

- [x] Legacy dev tree removed from active source tree.
- [x] cotrain/joint/stage experiment trainers removed from active tree and archived.
- [x] old compatibility shims removed; canonical ops paths are under `scripts/ops/`.
- [x] part-prompt-seg class path unchanged: `trellis.models.part_seg.promptable_latent_seg.PromptablePartLatentSegNet`.
- [x] original part-seg checkpoint strict-loaded through existing ckpt parser.
- [x] `scripts/eval/` contains real 0617 EE batch/single logic and no longer imports/calls old development runners.
- [x] 1-object EE eval produced voxel, mesh, Gaussian, diagnostic, summary, and metrics.
- [x] root `README.md`, `LICENSE`, and `NOTICE` are present and current.
- [x] old logs/docs/dev scratch content archived outside active repo in a tarball with manifests.
