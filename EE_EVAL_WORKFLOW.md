# EE-Eval / RealAppliance MuJoCo Delivery Workflow

This repo snapshot is intended to preserve the ee-eval path and the RealAppliance
R-D MuJoCo delivery constraints. The short rule is:

```text
decoded SLat whole/part meshes + decoded appearance -> ee-eval outputs -> optional R-D MuJoCo export -> validator
```

Do not deliver raw/GT meshes from `raw/partseg` or `reconstruction/part_info` as
final MuJoCo assets.

## Canonical Entry Points

Single smoke:

```bash
cd /root/code/arts-gen
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /opt/venvs/arts-gen

SMOKE=1 GPUS=0 bash scripts/eval/run_ee_eval.bash
```

RA-40 prismatic selection:

```bash
/opt/venvs/arts-gen/bin/python scripts/eval/selection/select_ra_prismatic_equal40.py \
  --out-dir /robot/data-lab/jzh/art-gen/ee-eval/0708-ra-40 \
  --limit 40 --train-count 40
```

RA-40 ee-eval with textured MuJoCo export:

```bash
PYTHONPATH=/root/code/arts-gen:/root/code/arts-gen/TRELLIS-arts:${PYTHONPATH:-} \
/opt/venvs/arts-gen/bin/python scripts/eval/run_eval.py ee_0617 \
  --out-dir /robot/data-lab/jzh/art-gen/ee-eval/0708-ra-40 \
  --limit 40 --train-count 40 --held-count 0 --gpus 0,1,2,3,4,5,6,7 \
  --allowed-datasets realappliance --selection-mode samples --sample-selection-unit objects \
  --split-json /robot/data-lab/jzh/art-gen/ee-eval/0708-ra-40/_data_configs/realappliance/ra_prismatic_one_part_40objects_split.json \
  --slat-token-source live \
  --ss-decoder-ckpt /robot/data-lab/jzh/art-gen/third-party-weights/trellis/pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --ss-flow-ckpt /robot/data-lab/jzh/art-gen/ckpts/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt \
  --export-mujoco --mujoco-textured-assets --mujoco-appearance-source gaussian-texture \
  --mujoco-texture-size 1024 --mujoco-texture-render-resolution 768 \
  --mujoco-texture-nviews 32 --mujoco-texture-mode fast \
  --force --force-export
```

## Required R-D MuJoCo Rules

- Meshes must come from decoded SLat geometry and decoded appearance/texture.
- Do not rotate or bake OBJ vertices to fix direction. Put the direction
  correction on the MJCF root object body:

```xml
<body name="object" pos="0 0 0" quat="0.707106781 0.707106781 0 0">
```

- Drawer/prismatic parts must have the slide joint inside the drawer body with
  local `axis="0 0 1"`. Do not ship final XML with drawer axis `0 1 0` or
  `0 -1 0`.
- Remove floor/blue checker assets/geoms.
- Remove all `inertia="shell"` for MuJoCo 3.2.7 compatibility.
- Use visual `group="0"` and:

```xml
<compiler angle="radian" meshdir="." texturedir="." balanceinertia="true" inertiagrouprange="3 5" />
```

- If a light-blue/glass component remains inside predicted body mesh, cut it
  from the predicted body textured OBJ and attach it as `drawer_glass_visual`
  under the drawer body so it moves with qpos.

## Validation

Always validate final XML:

```bash
/opt/venvs/arts-gen/bin/python scripts/eval/post/validate_mujoco_rd.py \
  --xml path/to/object.xml
```

On machines with `/home/mi/mujoco-3.2.7/bin/compile`, the validator also runs
the MuJoCo 3.2.7 compile binary.

Expected checks include:

```text
nq=1, nv=1, nu=1, njnt=1
jnt_axis=[0,0,1]
floor_count=0
shell_count=0
missing_assets=[]
drawer glass, if present, is attached under the drawer body
```

## RA-40 Definition

RA-40 means 40 unique RealAppliance objects with one angle per object, not 40
object-angle pairs. Use:

```text
scripts/eval/selection/select_ra_prismatic_equal40.py
```

The filtered split must merge fixed child labels whose `parent_group` points to
the moving group, or whose fixed `joint_group_id` equals the moving group, into
the moving target label.

## Promptable Part Seg Data Default

Promptable part segmentation training should default to v6:

```text
split:  /robot/data-lab/jzh/art-gen/data/part_promptable_seg_manifests/v6/split_official_verse_realappliance_0511dd_v6.json
packed: /robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6
```

v6 keeps high-part-count/button-heavy objects and includes the RealAppliance
fixed-child union repair.

## Joint Part Boundary Refinement

Joint checkpoints (`ckpt args.joint_seg=true`) must use the true shared
`body + K parts` head. Do not combine a joint checkpoint with
`--part-t0-filter`; T0 uses the legacy independent voxel head.

The joint stage supports two explicit candidate modes:

- `proposal`: current production-compatible cell proposal union.
- `full_occ`: whole occupied-grid ablation for measuring proposal truncation.

It can also save `parts/joint_partition.npz`, containing shared coords, fp16
joint logits, raw/refined labels, top-2 margin, and class names. This is the
required artifact for offline boundary sweeps; per-part hard voxel files alone
discard the confidence needed for a guarded refinement.

The current guarded refiner is opt-in. Its proxy-held locked parameters are one
6-neighbor iteration over the lowest raw-logit-margin 1% of voxels, pairwise
weight 3.0, and protection for predicted classes with at most 32 voxels. The
global default remains off until the same setting passes the fixed RA-40 and
general heldout gates.

Example one-object or batch launcher configuration:

```bash
PART_SEG_CKPT=/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint/ckpts/latest.pt \
PART_JOINT_REFINE=1 \
PART_JOINT_SAVE_LOGITS=1 \
PART_JOINT_CANDIDATE_MODE=proposal \
bash scripts/eval/run_ee_eval.bash
```

For the next joint training run, do not restore the old all-label Potts term.
Enable the complete boundary auxiliary explicitly; setting only a sub-weight
while leaving `JOINT_SMOOTH_WEIGHT=0` is a no-op:

```bash
JOINT_SMOOTH_WEIGHT=0.2 \
JOINT_SMOOTH_SAME_LABEL_WEIGHT=1.5 \
JOINT_SMOOTH_ALL_LABEL_WEIGHT=0 \
JOINT_SMOOTH_CROSS_LABEL_WEIGHT=1 \
bash scripts/train/part_promptable_seg/run_train.bash
```

Multi-claim/contact voxels are not assigned by part list order. Hard CE ignores
their arbitrary single owner, while a partial-label unary keeps probability
inside the set of claiming parts. Hard/overlap and overlap/overlap neighbor
terms propagate a locally coherent split across the contact band. Training logs
expose `joint_overlap_voxels`, `joint_overlap_ratio`,
`joint_overlap_supervised_voxels`, `joint_overlap_claim_mass`, and the overlap
spatial pair count `joint_smooth_overlap_pairs`.

EE stage reuse is guarded by `parts/part_stage_signature.json`. Changing the
joint checkpoint, candidate mode, refiner parameters, or logits artifact flag
invalidates cached part voxels and reruns the part stage.
