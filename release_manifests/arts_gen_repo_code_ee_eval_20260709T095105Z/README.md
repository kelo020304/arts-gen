Arts-Gen Repo Code Snapshot
===========================

Release id: arts-gen-code-ee-eval-20260709T095105Z

Purpose
-------

This archive is a repo code snapshot with the ee-eval / RealAppliance R-D MuJoCo
workflow highlighted at the repository root in `EE_EVAL_WORKFLOW.md`.

Primary workflow docs included:

- `EE_EVAL_WORKFLOW.md`
- `README.md`
- `docs/runbooks/ee-eval-right.txt`
- `scripts/eval/run_ee_eval.bash`
- `scripts/eval/run_eval.py`
- `scripts/eval/tasks/ee_0617.py`
- `scripts/eval/tasks/ee_0617_batch.py`
- `scripts/eval/tasks/ee_0617_single.py`
- `scripts/eval/post/validate_mujoco_rd.py`
- `scripts/eval/selection/select_ra_prismatic_equal40.py`

Included Scope
--------------

- Repo source code under `scripts/`, `TRELLIS-arts/`, `pipeline/`,
  `post_process/`, `tests/`, `workbenches/`, and lightweight docs/runbooks.
- Lightweight submodule/source snapshots except large dataset/tool cache folders.
- Current part promptable segmentation training code and release manifests.

Excluded From The Code Archive
------------------------------

These are not repo code and are intentionally excluded:

- `.git/`, caches, `__pycache__/`, virtualenvs.
- Packed data, model checkpoints, run outputs, generated eval outputs.
- `core` crash dump.
- `software/`.
- `sam3d_cu118_deps/wheelhouse/`.
- `submodules/dataset_toolkits/`.
- `docs/inspections/` image-heavy inspection artifacts.
- Archive files and wheel files already present in the working tree.

Important EE-Eval Delivery Rule
-------------------------------

For RealAppliance ee-eval MuJoCo delivery, final assets must use decoded SLat
mesh and decoded appearance/texture. Do not deliver raw/GT mesh from
`raw/partseg` or `reconstruction/part_info`.

