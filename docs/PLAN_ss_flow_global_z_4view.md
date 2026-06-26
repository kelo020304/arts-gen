# PLAN: 4view -> global z SS flow fine-tune

## Scope

Fine-tune TRELLIS `SparseStructureFlowModel` from `ss_flow_img_dit_L_16l8_fp16`
so pre-encoded 4-view DINOv2 tokens generate whole-object `z_global`
(`mean: [8,16,16,16]`). Freeze SS VAE and DINOv2; the dataset consumes only
precomputed tokens and SS latents. The `z_global` target is canonical normalized
structure; absolute scale remains out of scope.

## Files

1. Add `TRELLIS-arts/trellis/datasets/arts/ss_flow_global_z.py`.
   - Strict manifest/directory dataset for:
     - `reconstruction/dinov2_tokens/<id>/angle_<i>/tokens.npz["tokens"]`
     - `reconstruction/ss_latents_expanded/<id>/angle_<i>/latent.npz["mean"]`
   - Return `{"x_0": z_global, "cond": concat_4view_tokens, metadata...}`.
   - No recursive retry/fallback on sample load failure; raise readable
     `FileNotFoundError`/`KeyError`/`ValueError`.
   - Keep fixed 4-view concat for training, with optional deterministic
     `test_obj_ids`/`max_samples` for smoke.

2. Add tests under `TRELLIS-arts/tests/arts/ss_flow_global_z/`.
   - Minimal temp-data dataset contract test:
     - 1 object, 1 angle, `tokens: [4,T,D]`, `mean: [8,16,16,16]`.
     - Validate shapes, collate output, metadata, and failure on missing keys.
   - Trainer import/override smoke:
     - Verify trainer class routes pre-encoded `cond` directly into CFG and
       does not call DINOv2/image encoding.

3. Add `TRELLIS-arts/trellis/trainers/arts/ss_flow_global_z.py`.
   - Start from `trellis/trainers/arts/ss_flow_art.py`.
   - Build `SparseStructureFlowModel`, load `ss_flow_img_dit_L_16l8_fp16`.
   - Use dense `ImageConditionedFlowMatchingCFGTrainer` parent, but override:
     - `get_cond`: pass pre-encoded tokens to `ClassifierFreeGuidanceMixin`
       with zero negative tokens for cond-dropout/CFG training.
     - `get_inference_cond`: same for sampling, include `neg_cond`.
     - `snapshot_dataset`: skip image visualization.
     - `snapshot`: sample latents from tokens, decode via frozen `ss_dec`, and
       write voxel PNG plus a small JSON summary.
   - Keep DINOv2 absent from the training path.

4. Update dispatch and imports.
   - Add stage key `ss_flow_global_z` to `TRELLIS-arts/train_arts.py`.
   - Add dataset export in `trellis/datasets/arts/__init__.py` only if that
     package uses explicit exports.

5. Add configs under `TRELLIS-arts/configs/arts/ss_flow_global_z/`.
   - `mv_4view.yaml`: production full fine-tune config.
   - `smoke_test.yaml`: batch 1, fixed object subset/max samples, `i_print=1`,
     short run, snapshots enabled at init/final, wandb disabled.
   - Config includes:
     - pretrained SS flow ckpt path
     - frozen SS decoder ckpt path for snapshot decode
     - CFG dropout (`p_uncond`) and snapshot CFG strength/steps
     - output/inspection directories

6. Add smoke runner/docs.
   - Add a small script or pytest smoke that runs:
     - dataset import/shape check
     - 1 sample train for tens of steps
     - init/final snapshot decode through `ss_dec`
   - Update `docs/README.md` or add a short runbook section with:
     - training command
     - smoke command
     - where to inspect loss, ckpts, and voxel PNGs.

## Incremental Order And Validation

1. Dataset first.
   - Run:
     `PYTHONPATH=TRELLIS-arts pytest -q TRELLIS-arts/tests/arts/ss_flow_global_z/test_dataset.py`
   - Expected: shapes pass; bad files fail loudly.

2. Trainer skeleton and stage dispatch.
   - Run:
     `PYTHONPATH=TRELLIS-arts pytest -q TRELLIS-arts/tests/arts/ss_flow_global_z/test_train_import.py`
   - Expected: trainer imports, stage dispatch resolves, `get_cond` accepts
     token tensors without image/DINOv2 code.

3. Snapshot decode only.
   - Run a tiny local decode using one ground-truth latent before training.
   - Expected: `ss_dec` loads frozen, decoded voxel count is nonzero or the
     JSON records threshold counts explaining emptiness; PNG is written.

4. Short training smoke.
   - Run:
     `python TRELLIS-arts/train_arts.py --config TRELLIS-arts/configs/arts/ss_flow_global_z/smoke_test.yaml training.max_steps=30`
   - Expected:
     - pretrained SS flow loads, or fails before training with explicit path.
     - loss logs every step.
     - no DINOv2/image encoder call.
     - final snapshot decode PNG exists.

5. Evidence check before done.
   - Record first/last loss window from stdout/log.
   - Report snapshot PNG path and decode JSON path.
   - If loss does not decrease in the short run, report the actual log instead
     of claiming convergence.

## Known Local Constraint

`/root/code/arts-gen` and `TRELLIS-arts/` are not git repositories in this
workspace (`git status` fails with "not a git repository"). I can keep changes
small and inspect diffs manually, but atomic commits are blocked unless a git
worktree is provided or initialized by the user.
