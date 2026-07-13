Part Promptable Seg L Joint Release
===================================

Release id: part-prompt-seg-L-0709-1-joint-20260709T093348Z

Contents
--------

- ckpts/latest.pt
  - Source: /robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint/ckpts/latest.pt
  - Warm-start source: /robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/ckpts/latest.pt
  - Checkpoint step reported by training: 5000
- code/
  - scripts/train/part_promptable_seg
  - scripts/train/_ddp_common.sh
  - TRELLIS-arts/trellis/models/part_seg
- eval/step_005000
  - Small evaluation artifacts and tables only.
- config/
  - proxy_balanced_three_datasets_v6_eval_stratified.json
- run_h200_joint_L.sh
  - Repro launch command for the L joint-seg run.

Not Included
------------

- Packed dataset files.
- train_rows.json and full_eval_rows.json from the run directory.
- Large source data or generated meshes.

Key 5k Eval Snapshot
--------------------

- held_cell: 0.4007
- realappliance held_cell: 0.4900
- held_boundary_err: 0.4718
- held_cross_same: 0.8089
- tiny held_cell: 0.0977
- button held_cell: 0.1455
- peakGB: 52.4492

