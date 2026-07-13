

# Promptable Part Seg Gate2 Eval step 5

out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/partseg_train_smoke_5step_20260626T074757Z/eval/step_000005`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.777`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 1       | 0.0000     | 0.2872       | 0.0000         | 0.0000     | 1.0      | 1.0           | 1      | 0.0000    | 0.2872      | 0.0000        | 0.0000    | 1.0     | 1.0         
medium | 2       | 0.0000     | 0.2507       | 0.0000         | 0.0000     | 0.5      | 1.0           | 2      | 0.0000    | 0.2507      | 0.0000        | 0.0000    | 0.5     | 1.0         
large  | 1       | 0.0000     | 0.7282       | 0.0000         | 0.0000     | 0.0      | 1.0           | 1      | 0.0000    | 0.7282      | 0.0000        | 0.0000    | 0.0     | 1.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 4       | 0.0000     | 0.3792       | 0.0000         | 0.0000     | 0.5      | 1.0           | 4      | 0.0000    | 0.3792      | 0.0000        | 0.0000    | 0.5     | 1.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 762  | 0.0000 | 0.3163 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.0000 | 0.1851 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 450  | 0.0000 | 0.2872 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.0000 | 0.7282 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `gate1`
out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/partseg_train_smoke_5step_20260626T074757Z`
latest: `/mnt/robot-data-lab/jzh/art-gen/open-source-cleanup-0626/partseg_train_smoke_5step_20260626T074757Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 5

out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T091158Z/eval/step_000005`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.777`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 1       | 0.0000     | 0.2865       | 0.0000         | 0.0000     | 1.0      | 1.0           | 1      | 0.0000    | 0.2865      | 0.0000        | 0.0000    | 1.0     | 1.0         
medium | 2       | 0.0000     | 0.2364       | 0.0000         | 0.0000     | 0.5      | 1.0           | 2      | 0.0000    | 0.2364      | 0.0000        | 0.0000    | 0.5     | 1.0         
large  | 1       | 0.0000     | 0.7217       | 0.0000         | 0.0000     | 0.0      | 1.0           | 1      | 0.0000    | 0.7217      | 0.0000        | 0.0000    | 0.0     | 1.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 4       | 0.0000     | 0.3702       | 0.0000         | 0.0000     | 0.5      | 1.0           | 4      | 0.0000    | 0.3702      | 0.0000        | 0.0000    | 0.5     | 1.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 762  | 0.0000 | 0.3070 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.0000 | 0.1658 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 450  | 0.0000 | 0.2865 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.0000 | 0.7217 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `gate1`
out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T091158Z`
latest: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T091158Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 5

out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z/eval/step_000005`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.777`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 1       | 0.0000     | 0.2872       | 0.0000         | 0.0000     | 1.0      | 1.0           | 1      | 0.0000    | 0.2872      | 0.0000        | 0.0000    | 1.0     | 1.0         
medium | 2       | 0.0000     | 0.2413       | 0.0000         | 0.0000     | 0.5      | 1.0           | 2      | 0.0000    | 0.2413      | 0.0000        | 0.0000    | 0.5     | 1.0         
large  | 1       | 0.0000     | 0.7235       | 0.0000         | 0.0000     | 0.0      | 1.0           | 1      | 0.0000    | 0.7235      | 0.0000        | 0.0000    | 0.0     | 1.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 4       | 0.0000     | 0.3733       | 0.0000         | 0.0000     | 0.5      | 1.0           | 4      | 0.0000    | 0.3733      | 0.0000        | 0.0000    | 0.5     | 1.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 762  | 0.0000 | 0.3116 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.0000 | 0.1710 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 450  | 0.0000 | 0.2872 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.0000 | 0.7235 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `gate1`
out_dir: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z`
latest: `/mnt/robot-data-lab/jzh/art-gen/open-source-repo-cleanup-0626/partseg_run_train_smoke_20260626T093210Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 2

out_dir: `/tmp/part_prompt_seg_joint_linear_smoke/eval/step_000002`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.748`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 2       | 0.0197     | 0.0197       | 0.0197         | 0.0197     | 0.0      | 0.0           | 2      | 0.0197    | 0.0197      | 0.0197        | 0.0197    | 0.0     | 0.0         
medium | 2       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 2      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
large  | 1       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 1      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 5       | 0.0079     | 0.0079       | 0.0079         | 0.0079     | 0.0      | 0.0           | 5      | 0.0079    | 0.0079      | 0.0079        | 0.0079    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/body               | small  | 180  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 608  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 339  | 0.0394 | 0.0394 | 0.0394  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/tmp/part_prompt_seg_joint_linear_smoke`
latest: `/tmp/part_prompt_seg_joint_linear_smoke/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 250

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_000250`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 9      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
small  | 10      | 0.4262     | 0.4262       | 0.4262         | 0.4262     | 0.0      | 0.0           | 10     | 0.4262    | 0.4262      | 0.4262        | 0.4262    | 0.0     | 0.0         
medium | 18      | 0.8252     | 0.8252       | 0.8252         | 0.8252     | 0.0      | 0.0           | 18     | 0.8252    | 0.8252      | 0.8252        | 0.8252    | 0.0     | 0.0         
large  | 23      | 0.7721     | 0.7721       | 0.7721         | 0.7721     | 0.0      | 0.0           | 23     | 0.7721    | 0.7721      | 0.7721        | 0.7721    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.6146     | 0.6146       | 0.6146         | 0.6146     | 0.0      | 0.0           | 60     | 0.6146    | 0.6146      | 0.6146        | 0.6146    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.0000 | 0.0000 | 0.0000  
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_1_0          | tiny   | 25  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_2_0          | tiny   | 19  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_3_0          | tiny   | 31  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 500

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_000500`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 9      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
small  | 10      | 0.7815     | 0.7815       | 0.7815         | 0.7815     | 0.0      | 0.0           | 10     | 0.7815    | 0.7815      | 0.7815        | 0.7815    | 0.0     | 0.0         
medium | 18      | 0.9372     | 0.9372       | 0.9372         | 0.9372     | 0.0      | 0.0           | 18     | 0.9372    | 0.9372      | 0.9372        | 0.9372    | 0.0     | 0.0         
large  | 23      | 0.8833     | 0.8833       | 0.8833         | 0.8833     | 0.0      | 0.0           | 23     | 0.8833    | 0.8833      | 0.8833        | 0.8833    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.7500     | 0.7500       | 0.7500         | 0.7500     | 0.0      | 0.0           | 60     | 0.7500    | 0.7500      | 0.7500        | 0.7500    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.0000 | 0.0000 | 0.0000  
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_1_0          | tiny   | 25  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_2_0          | tiny   | 19  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_3_0          | tiny   | 31  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 750

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_000750`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.0080     | 0.0080       | 0.0080         | 0.0080     | 0.0      | 0.0           | 9      | 0.0080    | 0.0080      | 0.0080        | 0.0080    | 0.0     | 0.0         
small  | 10      | 0.8457     | 0.8457       | 0.8457         | 0.8457     | 0.0      | 0.0           | 10     | 0.8457    | 0.8457      | 0.8457        | 0.8457    | 0.0     | 0.0         
medium | 18      | 0.9561     | 0.9561       | 0.9561         | 0.9561     | 0.0      | 0.0           | 18     | 0.9561    | 0.9561      | 0.9561        | 0.9561    | 0.0     | 0.0         
large  | 23      | 0.9269     | 0.9269       | 0.9269         | 0.9269     | 0.0      | 0.0           | 23     | 0.9269    | 0.9269      | 0.9269        | 0.9269    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.7843     | 0.7843       | 0.7843         | 0.7843     | 0.0      | 0.0           | 60     | 0.7843    | 0.7843      | 0.7843        | 0.7843    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.0000 | 0.0000 | 0.0000  
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_2_0          | tiny   | 19  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_4_0          | tiny   | 29  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_5_0          | tiny   | 18  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 1000

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_001000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.3042     | 0.3042       | 0.3042         | 0.3042     | 0.0      | 0.0           | 9      | 0.3042    | 0.3042      | 0.3042        | 0.3042    | 0.0     | 0.0         
small  | 10      | 0.8128     | 0.8128       | 0.8128         | 0.8128     | 0.0      | 0.0           | 10     | 0.8128    | 0.8128      | 0.8128        | 0.8128    | 0.0     | 0.0         
medium | 18      | 0.9737     | 0.9737       | 0.9737         | 0.9737     | 0.0      | 0.0           | 18     | 0.9737    | 0.9737      | 0.9737        | 0.9737    | 0.0     | 0.0         
large  | 23      | 0.9779     | 0.9779       | 0.9779         | 0.9779     | 0.0      | 0.0           | 23     | 0.9779    | 0.9779      | 0.9779        | 0.9779    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.8481     | 0.8481       | 0.8481         | 0.8481     | 0.0      | 0.0           | 60     | 0.8481    | 0.8481      | 0.8481        | 0.8481    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_3_0          | tiny   | 31  | 0.1905 | 0.1905 | 0.1905  
101808/a0/knob_2_0          | tiny   | 19  | 0.1923 | 0.1923 | 0.1923  
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.3000 | 0.3000 | 0.3000  
101808/a0/knob_7_0          | tiny   | 30  | 0.3611 | 0.3611 | 0.3611  
```


# Promptable Part Seg Gate2 Eval step 1250

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_001250`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.3699     | 0.3699       | 0.3699         | 0.3699     | 0.0      | 0.0           | 9      | 0.3699    | 0.3699      | 0.3699        | 0.3699    | 0.0     | 0.0         
small  | 10      | 0.8864     | 0.8864       | 0.8864         | 0.8864     | 0.0      | 0.0           | 10     | 0.8864    | 0.8864      | 0.8864        | 0.8864    | 0.0     | 0.0         
medium | 18      | 0.9881     | 0.9881       | 0.9881         | 0.9881     | 0.0      | 0.0           | 18     | 0.9881    | 0.9881      | 0.9881        | 0.9881    | 0.0     | 0.0         
large  | 23      | 0.9886     | 0.9886       | 0.9886         | 0.9886     | 0.0      | 0.0           | 23     | 0.9886    | 0.9886      | 0.9886        | 0.9886    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.8787     | 0.8787       | 0.8787         | 0.8787     | 0.0      | 0.0           | 60     | 0.8787    | 0.8787      | 0.8787        | 0.8787    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101808/a0/knob_3_0          | tiny   | 31  | 0.1951 | 0.1951 | 0.1951  
101808/a0/knob_2_0          | tiny   | 19  | 0.2800 | 0.2800 | 0.2800  
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.2857 | 0.2857 | 0.2857  
101808/a0/knob_4_0          | tiny   | 29  | 0.4000 | 0.4000 | 0.4000  
```


# Promptable Part Seg Gate2 Eval step 1500

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/eval/step_001500`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.600`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 9       | 0.3948     | 0.3948       | 0.3948         | 0.3948     | 0.0      | 0.0           | 9      | 0.3948    | 0.3948      | 0.3948        | 0.3948    | 0.0     | 0.0         
small  | 10      | 0.8941     | 0.8941       | 0.8941         | 0.8941     | 0.0      | 0.0           | 10     | 0.8941    | 0.8941      | 0.8941        | 0.8941    | 0.0     | 0.0         
medium | 18      | 0.9897     | 0.9897       | 0.9897         | 0.9897     | 0.0      | 0.0           | 18     | 0.9897    | 0.9897      | 0.9897        | 0.9897    | 0.0     | 0.0         
large  | 23      | 0.9905     | 0.9905       | 0.9905         | 0.9905     | 0.0      | 0.0           | 23     | 0.9905    | 0.9905      | 0.9905        | 0.9905    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 60      | 0.8848     | 0.8848       | 0.8848         | 0.8848     | 0.0      | 0.0           | 60     | 0.8848    | 0.8848      | 0.8848        | 0.8848    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                          | bucket | raw | cell   | GTcand | Predcand
--------------------------- | ------ | --- | ------ | ------ | --------
10068/a0/side_controls_0    | tiny   | 10  | 0.0000 | 0.0000 | 0.0000  
101584/a0/switch_(handle)_0 | tiny   | 20  | 0.2381 | 0.2381 | 0.2381  
101808/a0/knob_3_0          | tiny   | 31  | 0.2683 | 0.2683 | 0.2683  
101808/a0/knob_2_0          | tiny   | 19  | 0.2800 | 0.2800 | 0.2800  
101808/a0/knob_4_0          | tiny   | 29  | 0.4412 | 0.4412 | 0.4412  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-overfit/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 2

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-ddp-debug2-20260701T025933Z/eval/step_000002`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `11.700`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 10      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 12     | 0.0136    | 0.0136      | 0.0136        | 0.0136    | 0.0     | 0.0         
small  | 76      | 0.0869     | 0.0869       | 0.0869         | 0.0869     | 0.0      | 0.0           | 45     | 0.0101    | 0.0101      | 0.0101        | 0.0101    | 0.0     | 0.0         
medium | 73      | 0.1509     | 0.1509       | 0.1509         | 0.1509     | 0.0      | 0.0           | 103    | 0.1029    | 0.1029      | 0.1029        | 0.1029    | 0.0     | 0.0         
large  | 21      | 0.3503     | 0.3503       | 0.3503         | 0.3503     | 0.0      | 0.0           | 18     | 0.0924    | 0.0924      | 0.0924        | 0.0924    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 180     | 0.1387     | 0.1387       | 0.1387         | 0.1387     | 0.0      | 0.0           | 178    | 0.0724    | 0.0724      | 0.0724        | 0.0724    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00619c9de6f14f03940b6cf72575d822/a0/coffin_0                      | medium | 994 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-ddp-debug2-20260701T025933Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-ddp-debug2-20260701T025933Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 2

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-bg-debug-20260701T030452Z/eval/step_000002`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `11.700`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 10      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 12     | 0.0136    | 0.0136      | 0.0136        | 0.0136    | 0.0     | 0.0         
small  | 76      | 0.0868     | 0.0868       | 0.0868         | 0.0868     | 0.0      | 0.0           | 45     | 0.0101    | 0.0101      | 0.0101        | 0.0101    | 0.0     | 0.0         
medium | 73      | 0.1509     | 0.1509       | 0.1509         | 0.1509     | 0.0      | 0.0           | 103    | 0.1029    | 0.1029      | 0.1029        | 0.1029    | 0.0     | 0.0         
large  | 21      | 0.3503     | 0.3503       | 0.3503         | 0.3503     | 0.0      | 0.0           | 18     | 0.0924    | 0.0924      | 0.0924        | 0.0924    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 180     | 0.1387     | 0.1387       | 0.1387         | 0.1387     | 0.0      | 0.0           | 178    | 0.0724    | 0.0724      | 0.0724        | 0.0724    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00619c9de6f14f03940b6cf72575d822/a0/coffin_0                      | medium | 994 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-bg-debug-20260701T030452Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part-propt-seg-S-0701-bg-debug-20260701T030452Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 120

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget240000-scale128-20260701T041510Z/eval/step_000120`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `17.835`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 62     | 0.0027    | 0.0027      | 0.0027        | 0.0027    | 0.0     | 0.0         
small  | 192     | 0.1072     | 0.1072       | 0.1072         | 0.1072     | 0.0      | 0.0           | 124    | 0.1431    | 0.1431      | 0.1431        | 0.1431    | 0.0     | 0.0         
medium | 112     | 0.2764     | 0.2764       | 0.2764         | 0.2764     | 0.0      | 0.0           | 124    | 0.2629    | 0.2629      | 0.2629        | 0.2629    | 0.0     | 0.0         
large  | 21      | 0.4396     | 0.4396       | 0.4396         | 0.4396     | 0.0      | 0.0           | 30     | 0.6603    | 0.6603      | 0.6603        | 0.6603    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 337     | 0.1804     | 0.1804       | 0.1804         | 0.1804     | 0.0      | 0.0           | 340    | 0.2068    | 0.2068      | 0.2068        | 0.2068    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0      | small  | 136 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget240000-scale128-20260701T041510Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget240000-scale128-20260701T041510Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 120

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget400000-scale128-20260701T042625Z/eval/step_000120`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `27.850`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 62     | 0.0029    | 0.0029      | 0.0029        | 0.0029    | 0.0     | 0.0         
small  | 192     | 0.1376     | 0.1376       | 0.1376         | 0.1376     | 0.0      | 0.0           | 124    | 0.1141    | 0.1141      | 0.1141        | 0.1141    | 0.0     | 0.0         
medium | 112     | 0.2528     | 0.2528       | 0.2528         | 0.2528     | 0.0      | 0.0           | 124    | 0.3108    | 0.3108      | 0.3108        | 0.3108    | 0.0     | 0.0         
large  | 21      | 0.6657     | 0.6657       | 0.6657         | 0.6657     | 0.0      | 0.0           | 30     | 0.6495    | 0.6495      | 0.6495        | 0.6495    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 337     | 0.2039     | 0.2039       | 0.2039         | 0.2039     | 0.0      | 0.0           | 340    | 0.2128    | 0.2128      | 0.2128        | 0.2128    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0      | small  | 136 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget400000-scale128-20260701T042625Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget400000-scale128-20260701T042625Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 120

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget600000-scale128-20260701T044501Z/eval/step_000120`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `43.699`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 62     | 0.0031    | 0.0031      | 0.0031        | 0.0031    | 0.0     | 0.0         
small  | 192     | 0.0717     | 0.0717       | 0.0717         | 0.0717     | 0.0      | 0.0           | 124    | 0.0666    | 0.0666      | 0.0666        | 0.0666    | 0.0     | 0.0         
medium | 112     | 0.2003     | 0.2003       | 0.2003         | 0.2003     | 0.0      | 0.0           | 124    | 0.2989    | 0.2989      | 0.2989        | 0.2989    | 0.0     | 0.0         
large  | 21      | 0.5856     | 0.5856       | 0.5856         | 0.5856     | 0.0      | 0.0           | 30     | 0.5937    | 0.5937      | 0.5937        | 0.5937    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 337     | 0.1439     | 0.1439       | 0.1439         | 0.1439     | 0.0      | 0.0           | 340    | 0.1862    | 0.1862      | 0.1862        | 0.1862    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0      | small  | 136 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget600000-scale128-20260701T044501Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/sweep-fp16-S-b256-budget600000-scale128-20260701T044501Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 5

out_dir: `/tmp/part_promptable_seg_smoke_S_0701/eval/step_000005`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `1.313`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 2       | 0.0231     | 0.0231       | 0.0231         | 0.0231     | 0.0      | 0.0           | 2      | 0.0231    | 0.0231      | 0.0231        | 0.0231    | 0.0     | 0.0         
medium | 2       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 2      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
large  | 1       | 0.2051     | 0.2051       | 0.2051         | 0.2051     | 0.0      | 0.0           | 1      | 0.2051    | 0.2051      | 0.2051        | 0.2051    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 5       | 0.0502     | 0.0502       | 0.0502         | 0.0502     | 0.0      | 0.0           | 5      | 0.0502    | 0.0502      | 0.0502        | 0.0502    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/body               | small  | 180  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 608  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 339  | 0.0461 | 0.0461 | 0.0461  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.2051 | 0.2051 | 0.2051  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/tmp/part_promptable_seg_smoke_S_0701`
latest: `/tmp/part_promptable_seg_smoke_S_0701/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 5

out_dir: `/tmp/part_promptable_seg_smoke_M_0701/eval/step_000005`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `2.500`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 2       | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 2      | 0.0000    | 0.0000      | 0.0000        | 0.0000    | 0.0     | 0.0         
medium | 2       | 0.0873     | 0.0873       | 0.0873         | 0.0873     | 0.0      | 0.0           | 2      | 0.0873    | 0.0873      | 0.0873        | 0.0873    | 0.0     | 0.0         
large  | 1       | 0.5727     | 0.5727       | 0.5727         | 0.5727     | 0.0      | 0.0           | 1      | 0.5727    | 0.5727      | 0.5727        | 0.5727    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 5       | 0.1495     | 0.1495       | 0.1495         | 0.1495     | 0.0      | 0.0           | 5      | 0.1495    | 0.1495      | 0.1495        | 0.1495    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                     | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
004d1e9e13934e319094151a4fad823f/a0/body               | small  | 180  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/rotational_shaft_0 | medium | 608  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/top_hatch_panel_0  | small  | 339  | 0.0000 | 0.0000 | 0.0000  
004d1e9e13934e319094151a4fad823f/a0/gun_mount_frame_0  | medium | 2288 | 0.1746 | 0.1746 | 0.1746  
004d1e9e13934e319094151a4fad823f/a0/main_gun_housing_0 | large  | 5194 | 0.5727 | 0.5727 | 0.5727  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/tmp/part_promptable_seg_smoke_M_0701`
latest: `/tmp/part_promptable_seg_smoke_M_0701/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 100

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-ctrl-0707/eval/step_000100`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.858`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 1       | 0.8284     | 0.6682       | 0.6682         | 0.6682     | 594.0    | 1603.0        | 0      | nan       | nan         | nan           | nan       | nan     | nan         
medium | 2       | 0.7781     | 0.7341       | 0.7341         | 0.7341     | 690.5    | 1603.0        | 1      | 0.8177    | 0.7948      | 0.7948        | 0.7948    | 0.0     | 0.0         
large  | 1       | 0.7892     | 0.8543       | 0.8543         | 0.8543     | 1231.0   | 1603.0        | 0      | nan       | nan         | nan           | nan       | nan     | nan         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 4       | 0.7934     | 0.7477       | 0.7477         | 0.7477     | 801.5    | 1603.0        | 1      | 0.8177    | 0.7948      | 0.7948        | 0.7948    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                           | bucket | raw | cell   | GTcand | Predcand
-------------------------------------------- | ------ | --- | ------ | ------ | --------
00619c9de6f14f03940b6cf72575d822/a0/coffin_0 | medium | 994 | 0.8177 | 0.7948 | 0.7948  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-ctrl-0707`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-ctrl-0707/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 100

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-t1-0707/eval/step_000100`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `0.895`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 1       | 0.7075     | 0.7396       | 0.7396         | 0.7396     | 316.0    | 1156.0        | 0      | nan       | nan         | nan           | nan       | nan     | nan         
medium | 2       | 0.6600     | 0.6997       | 0.6997         | 0.6997     | 595.5    | 1156.0        | 1      | 0.6929    | 0.7611      | 0.7611        | 0.7611    | 0.0     | 0.0         
large  | 1       | 0.6928     | 0.8924       | 0.8924         | 0.8924     | 805.0    | 1156.0        | 0      | nan       | nan         | nan           | nan       | nan     | nan         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 4       | 0.6801     | 0.7579       | 0.7579         | 0.7579     | 578.0    | 1156.0        | 1      | 0.6929    | 0.7611      | 0.7611        | 0.7611    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                           | bucket | raw | cell   | GTcand | Predcand
-------------------------------------------- | ------ | --- | ------ | ------ | --------
00619c9de6f14f03940b6cf72575d822/a0/coffin_0 | medium | 994 | 0.6929 | 0.7611 | 0.7611  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-t1-0707`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/smoke-S-t1-0707/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 300

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_000300`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 57     | 0.0778    | 0.0778      | 0.0778        | 0.0778    | 0.0     | 0.0         
small  | 149     | 0.4029     | 0.4029       | 0.4029         | 0.4029     | 0.0      | 0.0           | 54     | 0.1694    | 0.1694      | 0.1694        | 0.1694    | 0.0     | 0.0         
medium | 77      | 0.5650     | 0.5650       | 0.5650         | 0.5650     | 0.0      | 0.0           | 123    | 0.5191    | 0.5191      | 0.5191        | 0.5191    | 0.0     | 0.0         
large  | 21      | 0.7580     | 0.7580       | 0.7580         | 0.7580     | 0.0      | 0.0           | 25     | 0.7924    | 0.7924      | 0.7924        | 0.7924    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.4612     | 0.4612       | 0.4612         | 0.4612     | 0.0      | 0.0           | 259    | 0.3755    | 0.3755      | 0.3755        | 0.3755    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                  | bucket | raw | cell   | GTcand | Predcand
------------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0          | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0   | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0          | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0  | small  | 182 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red_with_bow)_0 | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 600

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_000600`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 57     | 0.0539    | 0.0539      | 0.0539        | 0.0539    | 0.0     | 0.0         
small  | 149     | 0.4606     | 0.4606       | 0.4606         | 0.4606     | 0.0      | 0.0           | 54     | 0.2847    | 0.2847      | 0.2847        | 0.2847    | 0.0     | 0.0         
medium | 77      | 0.6557     | 0.6557       | 0.6557         | 0.6557     | 0.0      | 0.0           | 123    | 0.6322    | 0.6322      | 0.6322        | 0.6322    | 0.0     | 0.0         
large  | 21      | 0.7908     | 0.7908       | 0.7908         | 0.7908     | 0.0      | 0.0           | 25     | 0.8233    | 0.8233      | 0.8233        | 0.8233    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.5241     | 0.5241       | 0.5241         | 0.5241     | 0.0      | 0.0           | 259    | 0.4510    | 0.4510      | 0.4510        | 0.4510    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                 | bucket | raw | cell   | GTcand | Predcand
------------------------------------------------------------------ | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0        | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0  | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0         | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0       | small  | 136 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0 | small  | 182 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 900

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_000900`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0000     | 0.0000       | 0.0000         | 0.0000     | 0.0      | 0.0           | 57     | 0.1089    | 0.1089      | 0.1089        | 0.1089    | 0.0     | 0.0         
small  | 149     | 0.4622     | 0.4622       | 0.4622         | 0.4622     | 0.0      | 0.0           | 54     | 0.2849    | 0.2849      | 0.2849        | 0.2849    | 0.0     | 0.0         
medium | 77      | 0.6102     | 0.6102       | 0.6102         | 0.6102     | 0.0      | 0.0           | 123    | 0.5912    | 0.5912      | 0.5912        | 0.5912    | 0.0     | 0.0         
large  | 21      | 0.7809     | 0.7809       | 0.7809         | 0.7809     | 0.0      | 0.0           | 25     | 0.7800    | 0.7800      | 0.7800        | 0.7800    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.5107     | 0.5107       | 0.5107         | 0.5107     | 0.0      | 0.0           | 259    | 0.4394    | 0.4394      | 0.4394        | 0.4394    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0       | small  | 262 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_red)_0        | small  | 299 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0 | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0        | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0      | small  | 136 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 1200

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_001200`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0886     | 0.0886       | 0.0886         | 0.0886     | 0.0      | 0.0           | 57     | 0.1442    | 0.1442      | 0.1442        | 0.1442    | 0.0     | 0.0         
small  | 149     | 0.4915     | 0.4915       | 0.4915         | 0.4915     | 0.0      | 0.0           | 54     | 0.3830    | 0.3830      | 0.3830        | 0.3830    | 0.0     | 0.0         
medium | 77      | 0.7167     | 0.7167       | 0.7167         | 0.7167     | 0.0      | 0.0           | 123    | 0.6608    | 0.6608      | 0.6608        | 0.6608    | 0.0     | 0.0         
large  | 21      | 0.8486     | 0.8486       | 0.8486         | 0.8486     | 0.0      | 0.0           | 25     | 0.8077    | 0.8077      | 0.8077        | 0.8077    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.5688     | 0.5688       | 0.5688         | 0.5688     | 0.0      | 0.0           | 259    | 0.5034    | 0.5034      | 0.5034        | 0.5034    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                  | bucket | raw | cell   | GTcand | Predcand
------------------------------------------------------------------- | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0   | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red)_0          | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0        | small  | 136 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0  | small  | 182 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red_with_bow)_0 | tiny   | 44  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 1400

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_001400`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0091     | 0.0091       | 0.0091         | 0.0091     | 0.0      | 0.0           | 57     | 0.1876    | 0.1876      | 0.1876        | 0.1876    | 0.0     | 0.0         
small  | 149     | 0.4869     | 0.4869       | 0.4869         | 0.4869     | 0.0      | 0.0           | 54     | 0.3692    | 0.3692      | 0.3692        | 0.3692    | 0.0     | 0.0         
medium | 77      | 0.7696     | 0.7696       | 0.7696         | 0.7696     | 0.0      | 0.0           | 123    | 0.6791    | 0.6791      | 0.6791        | 0.6791    | 0.0     | 0.0         
large  | 21      | 0.8677     | 0.8677       | 0.8677         | 0.8677     | 0.0      | 0.0           | 25     | 0.8141    | 0.8141      | 0.8141        | 0.8141    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.5797     | 0.5797       | 0.5797         | 0.5797     | 0.0      | 0.0           | 259    | 0.5194    | 0.5194      | 0.5194        | 0.5194    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                  | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------------------- | ------ | ---- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(small,_red)_0         | small  | 262  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0  | small  | 182  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red_with_bow)_0 | tiny   | 44   | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/napkin_0                        | medium | 1183 | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/spoon_1                         | small  | 75   | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 1600

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_001600`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0746     | 0.0746       | 0.0746         | 0.0746     | 0.0      | 0.0           | 57     | 0.1419    | 0.1419      | 0.1419        | 0.1419    | 0.0     | 0.0         
small  | 149     | 0.5196     | 0.5196       | 0.5196         | 0.5196     | 0.0      | 0.0           | 54     | 0.3525    | 0.3525      | 0.3525        | 0.3525    | 0.0     | 0.0         
medium | 77      | 0.7872     | 0.7872       | 0.7872         | 0.7872     | 0.0      | 0.0           | 123    | 0.7352    | 0.7352      | 0.7352        | 0.7352    | 0.0     | 0.0         
large  | 21      | 0.8669     | 0.8669       | 0.8669         | 0.8669     | 0.0      | 0.0           | 25     | 0.8080    | 0.8080      | 0.8080        | 0.8080    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.6067     | 0.6067       | 0.6067         | 0.6067     | 0.0      | 0.0           | 259    | 0.5319    | 0.5319      | 0.5319        | 0.5319    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                 | bucket | raw | cell   | GTcand | Predcand
------------------------------------------------------------------ | ------ | --- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(cylindrical,_red)_0  | small  | 233 | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tall,_narrow,_red)_0 | small  | 189 | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/spoon_0                        | small  | 76  | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/spoon_1                        | small  | 75  | 0.0000 | 0.0000 | 0.0000  
0276b7595fe44ae39e0f812b25727e54/a0/body                           | small  | 108 | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 1800

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_001800`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.0507     | 0.0507       | 0.0507         | 0.0507     | 0.0      | 0.0           | 57     | 0.2367    | 0.2367      | 0.2367        | 0.2367    | 0.0     | 0.0         
small  | 149     | 0.5341     | 0.5341       | 0.5341         | 0.5341     | 0.0      | 0.0           | 54     | 0.3693    | 0.3693      | 0.3693        | 0.3693    | 0.0     | 0.0         
medium | 77      | 0.7855     | 0.7855       | 0.7855         | 0.7855     | 0.0      | 0.0           | 123    | 0.7403    | 0.7403      | 0.7403        | 0.7403    | 0.0     | 0.0         
large  | 21      | 0.8612     | 0.8612       | 0.8612         | 0.8612     | 0.0      | 0.0           | 25     | 0.8141    | 0.8141      | 0.8141        | 0.8141    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.6130     | 0.6130       | 0.6130         | 0.6130     | 0.0      | 0.0           | 259    | 0.5592    | 0.5592      | 0.5592        | 0.5592    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                 | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------------------ | ------ | ---- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red)_0       | small  | 136  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0 | small  | 182  | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/spoon_1                        | small  | 75   | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/cup_0                          | medium | 2052 | 0.0000 | 0.0000 | 0.0000  
0276b7595fe44ae39e0f812b25727e54/a0/body                           | small  | 108  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 2000

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_002000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `7.371`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.1675     | 0.1675       | 0.1675         | 0.1675     | 0.0      | 0.0           | 57     | 0.2531    | 0.2531      | 0.2531        | 0.2531    | 0.0     | 0.0         
small  | 149     | 0.5409     | 0.5409       | 0.5409         | 0.5409     | 0.0      | 0.0           | 54     | 0.3875    | 0.3875      | 0.3875        | 0.3875    | 0.0     | 0.0         
medium | 77      | 0.7760     | 0.7760       | 0.7760         | 0.7760     | 0.0      | 0.0           | 123    | 0.7043    | 0.7043      | 0.7043        | 0.7043    | 0.0     | 0.0         
large  | 21      | 0.8604     | 0.8604       | 0.8604         | 0.8604     | 0.0      | 0.0           | 25     | 0.8225    | 0.8225      | 0.8225        | 0.8225    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.6194     | 0.6194       | 0.6194         | 0.6194     | 0.0      | 0.0           | 259    | 0.5504    | 0.5504      | 0.5504        | 0.5504    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                                  | bucket | raw  | cell   | GTcand | Predcand
------------------------------------------------------------------- | ------ | ---- | ------ | ------ | --------
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(medium,_red,_rear)_0  | small  | 182  | 0.0000 | 0.0000 | 0.0000  
00dfee50afad4153880d3a04d9a040aa/a0/gift_box_(tiny,_red_with_bow)_0 | tiny   | 44   | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/napkin_0                        | medium | 1183 | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/cup_0                           | medium | 2052 | 0.0000 | 0.0000 | 0.0000  
0276b7595fe44ae39e0f812b25727e54/a0/body                            | small  | 108  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 2200

out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/eval/step_002200`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `8.187`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 12      | 0.2197     | 0.2197       | 0.2197         | 0.2197     | 0.0      | 0.0           | 57     | 0.2453    | 0.2453      | 0.2453        | 0.2453    | 0.0     | 0.0         
small  | 149     | 0.5428     | 0.5428       | 0.5428         | 0.5428     | 0.0      | 0.0           | 54     | 0.3879    | 0.3879      | 0.3879        | 0.3879    | 0.0     | 0.0         
medium | 77      | 0.7906     | 0.7906       | 0.7906         | 0.7906     | 0.0      | 0.0           | 123    | 0.7058    | 0.7058      | 0.7058        | 0.7058    | 0.0     | 0.0         
large  | 21      | 0.8708     | 0.8708       | 0.8708         | 0.8708     | 0.0      | 0.0           | 25     | 0.8073    | 0.8073      | 0.8073        | 0.8073    | 0.0     | 0.0         
button | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
all    | 259     | 0.6281     | 0.6281       | 0.6281         | 0.6281     | 0.0      | 0.0           | 259    | 0.5480    | 0.5480      | 0.5480        | 0.5480    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                           | bucket | raw  | cell   | GTcand | Predcand
-------------------------------------------- | ------ | ---- | ------ | ------ | --------
018f683e86ea42eb8e98935e890eb9fd/a0/napkin_0 | medium | 1183 | 0.0000 | 0.0000 | 0.0000  
018f683e86ea42eb8e98935e890eb9fd/a0/spoon_1  | small  | 75   | 0.0000 | 0.0000 | 0.0000  
0276b7595fe44ae39e0f812b25727e54/a0/body     | small  | 108  | 0.0000 | 0.0000 | 0.0000  
0295f923f1384c1089dddd697ce71f11/a1/body     | tiny   | 0    | 0.0000 | 0.0000 | 0.0000  
0295f923f1384c1089dddd697ce71f11/a5/body     | tiny   | 0    | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Train Complete

mode: `train`
out_dir: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z`
latest: `/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg-S-t1-0708-joint-smoke-20260708T132409Z/ckpts/latest.pt`


# Promptable Part Seg Gate2 Eval step 5000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_005000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `52.449`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
small  | 5       | 0.3831     | 0.3831       | 0.3831         | 0.3831     | 0.0      | 0.0           | 6      | 0.3879    | 0.3879      | 0.3879        | 0.3879    | 0.0     | 0.0         
medium | 2       | 0.5145     | 0.5145       | 0.5145         | 0.5145     | 0.0      | 0.0           | 2      | 0.7260    | 0.7260      | 0.7260        | 0.7260    | 0.0     | 0.0         
large  | 0       | nan        | nan          | nan            | nan        | nan      | nan           | 0      | nan       | nan         | nan           | nan       | nan     | nan         
button | 1       | 0.4869     | 0.4869       | 0.4869         | 0.4869     | 0.0      | 0.0           | 4      | 0.3431    | 0.3431      | 0.3431        | 0.3431    | 0.0     | 0.0         
all    | 7       | 0.4207     | 0.4207       | 0.4207         | 0.4207     | 0.0      | 0.0           | 8      | 0.4724    | 0.4724      | 0.4724        | 0.4724    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | - | -------- | ---------- | ---------- | ----------
tiny   | 0 | nan      | nan        | nan        | nan       
small  | 0 | nan      | nan        | nan        | nan       
medium | 0 | nan      | nan        | nan        | nan       
large  | 0 | nan      | nan        | nan        | nan       
button | 0 | nan      | nan        | nan        | nan       
all    | 0 | nan      | nan        | nan        | nan       
```

worst heldout:
```
id                                                    | bucket | raw | cell   | GTcand | Predcand
----------------------------------------------------- | ------ | --- | ------ | ------ | --------
517d9d39ae3643559f10d7927a548bba/a0/preset_button_3_0 | small  | 90  | 0.2258 | 0.2258 | 0.2258  
517d9d39ae3643559f10d7927a548bba/a0/preset_button_1_0 | small  | 106 | 0.3580 | 0.3580 | 0.3580  
517d9d39ae3643559f10d7927a548bba/a0/preset_button_2_0 | small  | 106 | 0.3911 | 0.3911 | 0.3911  
517d9d39ae3643559f10d7927a548bba/a0/power_button_0    | small  | 90  | 0.3974 | 0.3974 | 0.3974  
517d9d39ae3643559f10d7927a548bba/a0/volume_knob_0     | small  | 102 | 0.4722 | 0.4722 | 0.4722  
```


# Promptable Part Seg Gate2 Eval step 10000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_010000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `51.146`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.3244     | 0.3244       | 0.3244         | 0.3244     | 0.0      | 0.0           | 27     | 0.1204    | 0.1204      | 0.1204        | 0.1204    | 0.0     | 0.0         
small  | 10      | 0.6308     | 0.6308       | 0.6308         | 0.6308     | 0.0      | 0.0           | 30     | 0.5047    | 0.5047      | 0.5047        | 0.5047    | 0.0     | 0.0         
medium | 6       | 0.8146     | 0.8146       | 0.8146         | 0.8146     | 0.0      | 0.0           | 8      | 0.7298    | 0.7298      | 0.7298        | 0.7298    | 0.0     | 0.0         
large  | 4       | 0.9139     | 0.9139       | 0.9139         | 0.9139     | 0.0      | 0.0           | 11     | 0.6472    | 0.6472      | 0.6472        | 0.6472    | 0.0     | 0.0         
button | 4       | 0.4277     | 0.4277       | 0.4277         | 0.4277     | 0.0      | 0.0           | 26     | 0.1782    | 0.1782      | 0.1782        | 0.1782    | 0.0     | 0.0         
all    | 28      | 0.6231     | 0.6231       | 0.6231         | 0.6231     | 0.0      | 0.0           | 76     | 0.4125    | 0.4125      | 0.4125        | 0.4125    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.2586   | nan        | 0.2586     | 0.2586    
small  | 5  | 0.6410   | nan        | 0.6410     | 0.6410    
medium | 2  | 0.8608   | nan        | 0.8608     | 0.8608    
large  | 2  | 0.7536   | nan        | 0.7536     | 0.7536    
button | 9  | 0.2586   | nan        | 0.2586     | 0.2586    
all    | 18 | 0.4867   | nan        | 0.4867     | 0.4867    
```

worst heldout:
```
id                                                         | bucket | raw | cell   | GTcand | Predcand
---------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0  | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/control_knob_(right)_0 | small  | 53  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0          | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0      | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                         | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 15000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_015000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `51.146`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.5306     | 0.5306       | 0.5306         | 0.5306     | 0.0      | 0.0           | 27     | 0.1531    | 0.1531      | 0.1531        | 0.1531    | 0.0     | 0.0         
small  | 10      | 0.7764     | 0.7764       | 0.7764         | 0.7764     | 0.0      | 0.0           | 30     | 0.5305    | 0.5305      | 0.5305        | 0.5305    | 0.0     | 0.0         
medium | 6       | 0.9204     | 0.9204       | 0.9204         | 0.9204     | 0.0      | 0.0           | 8      | 0.7511    | 0.7511      | 0.7511        | 0.7511    | 0.0     | 0.0         
large  | 4       | 0.9706     | 0.9706       | 0.9706         | 0.9706     | 0.0      | 0.0           | 11     | 0.6633    | 0.6633      | 0.6633        | 0.6633    | 0.0     | 0.0         
button | 4       | 0.7005     | 0.7005       | 0.7005         | 0.7005     | 0.0      | 0.0           | 26     | 0.1943    | 0.1943      | 0.1943        | 0.1943    | 0.0     | 0.0         
all    | 28      | 0.7647     | 0.7647       | 0.7647         | 0.7647     | 0.0      | 0.0           | 76     | 0.4389    | 0.4389      | 0.4389        | 0.4389    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.2981   | nan        | 0.2981     | 0.2981    
small  | 5  | 0.7839   | nan        | 0.7839     | 0.7839    
medium | 2  | 0.8996   | nan        | 0.8996     | 0.8996    
large  | 2  | 0.8176   | nan        | 0.8176     | 0.8176    
button | 9  | 0.2981   | nan        | 0.2981     | 0.2981    
all    | 18 | 0.5576   | nan        | 0.5576     | 0.5576    
```

worst heldout:
```
id                                                         | bucket | raw | cell   | GTcand | Predcand
---------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0  | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/control_knob_(right)_0 | small  | 53  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0          | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0      | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                         | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 20000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_020000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `52.436`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.5440     | 0.5440       | 0.5440         | 0.5440     | 0.0      | 0.0           | 27     | 0.1915    | 0.1915      | 0.1915        | 0.1915    | 0.0     | 0.0         
small  | 10      | 0.8014     | 0.8014       | 0.8014         | 0.8014     | 0.0      | 0.0           | 30     | 0.5642    | 0.5642      | 0.5642        | 0.5642    | 0.0     | 0.0         
medium | 6       | 0.9313     | 0.9313       | 0.9313         | 0.9313     | 0.0      | 0.0           | 8      | 0.7820    | 0.7820      | 0.7820        | 0.7820    | 0.0     | 0.0         
large  | 4       | 0.9597     | 0.9597       | 0.9597         | 0.9597     | 0.0      | 0.0           | 11     | 0.6633    | 0.6633      | 0.6633        | 0.6633    | 0.0     | 0.0         
button | 4       | 0.6315     | 0.6315       | 0.6315         | 0.6315     | 0.0      | 0.0           | 26     | 0.2217    | 0.2217      | 0.2217        | 0.2217    | 0.0     | 0.0         
all    | 28      | 0.7783     | 0.7783       | 0.7783         | 0.7783     | 0.0      | 0.0           | 76     | 0.4691    | 0.4691      | 0.4691        | 0.4691    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.4065   | nan        | 0.4065     | 0.4065    
small  | 5  | 0.8002   | nan        | 0.8002     | 0.8002    
medium | 2  | 0.9102   | nan        | 0.9102     | 0.9102    
large  | 2  | 0.8424   | nan        | 0.8424     | 0.8424    
button | 9  | 0.4065   | nan        | 0.4065     | 0.4065    
all    | 18 | 0.6203   | nan        | 0.6203     | 0.6203    
```

worst heldout:
```
id                                                        | bucket | raw | cell   | GTcand | Predcand
--------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0 | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0         | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0     | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                        | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_1                                        | tiny   | 12  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 25000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_025000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `54.112`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.7252     | 0.7252       | 0.7252         | 0.7252     | 0.0      | 0.0           | 27     | 0.2015    | 0.2015      | 0.2015        | 0.2015    | 0.0     | 0.0         
small  | 10      | 0.8931     | 0.8931       | 0.8931         | 0.8931     | 0.0      | 0.0           | 30     | 0.5077    | 0.5077      | 0.5077        | 0.5077    | 0.0     | 0.0         
medium | 6       | 0.9728     | 0.9728       | 0.9728         | 0.9728     | 0.0      | 0.0           | 8      | 0.7611    | 0.7611      | 0.7611        | 0.7611    | 0.0     | 0.0         
large  | 4       | 0.9968     | 0.9968       | 0.9968         | 0.9968     | 0.0      | 0.0           | 11     | 0.6911    | 0.6911      | 0.6911        | 0.6911    | 0.0     | 0.0         
button | 4       | 0.8323     | 0.8323       | 0.8323         | 0.8323     | 0.0      | 0.0           | 26     | 0.2117    | 0.2117      | 0.2117        | 0.2117    | 0.0     | 0.0         
all    | 28      | 0.8770     | 0.8770       | 0.8770         | 0.8770     | 0.0      | 0.0           | 76     | 0.4521    | 0.4521      | 0.4521        | 0.4521    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.4048   | nan        | 0.4048     | 0.4048    
small  | 5  | 0.8005   | nan        | 0.8005     | 0.8005    
medium | 2  | 0.9184   | nan        | 0.9184     | 0.9184    
large  | 2  | 0.7809   | nan        | 0.7809     | 0.7809    
button | 9  | 0.4048   | nan        | 0.4048     | 0.4048    
all    | 18 | 0.6136   | nan        | 0.6136     | 0.6136    
```

worst heldout:
```
id                                                        | bucket | raw | cell   | GTcand | Predcand
--------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0 | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0         | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0     | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                        | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_1                                        | tiny   | 12  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 30000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_030000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `54.112`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.7966     | 0.7966       | 0.7966         | 0.7966     | 0.0      | 0.0           | 27     | 0.2248    | 0.2248      | 0.2248        | 0.2248    | 0.0     | 0.0         
small  | 10      | 0.9370     | 0.9370       | 0.9370         | 0.9370     | 0.0      | 0.0           | 30     | 0.5331    | 0.5331      | 0.5331        | 0.5331    | 0.0     | 0.0         
medium | 6       | 0.9769     | 0.9769       | 0.9769         | 0.9769     | 0.0      | 0.0           | 8      | 0.7500    | 0.7500      | 0.7500        | 0.7500    | 0.0     | 0.0         
large  | 4       | 0.9975     | 0.9975       | 0.9975         | 0.9975     | 0.0      | 0.0           | 11     | 0.6623    | 0.6623      | 0.6623        | 0.6623    | 0.0     | 0.0         
button | 4       | 0.9094     | 0.9094       | 0.9094         | 0.9094     | 0.0      | 0.0           | 26     | 0.2351    | 0.2351      | 0.2351        | 0.2351    | 0.0     | 0.0         
all    | 28      | 0.9141     | 0.9141       | 0.9141         | 0.9141     | 0.0      | 0.0           | 76     | 0.4651    | 0.4651      | 0.4651        | 0.4651    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.3822   | nan        | 0.3822     | 0.3822    
small  | 5  | 0.8098   | nan        | 0.8098     | 0.8098    
medium | 2  | 0.9226   | nan        | 0.9226     | 0.9226    
large  | 2  | 0.7877   | nan        | 0.7877     | 0.7877    
button | 9  | 0.3822   | nan        | 0.3822     | 0.3822    
all    | 18 | 0.6061   | nan        | 0.6061     | 0.6061    
```

worst heldout:
```
id                                                        | bucket | raw | cell   | GTcand | Predcand
--------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0 | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0         | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0     | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                        | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_1                                        | tiny   | 12  | 0.0000 | 0.0000 | 0.0000  
```


# Promptable Part Seg Gate2 Eval step 35000

out_dir: `/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0708/eval/step_035000`
metric: `voxel_iou_proxy`
full_eval: `False`
peak_memory_gb_batch1: `54.112`

```
bucket | train_n | train_cell | train_GTcand | train_Predcand | train_part | train_ov | train_part_ov | held_n | held_cell | held_GTcand | held_Predcand | held_part | held_ov | held_part_ov
------ | ------- | ---------- | ------------ | -------------- | ---------- | -------- | ------------- | ------ | --------- | ----------- | ------------- | --------- | ------- | ------------
tiny   | 8       | 0.9066     | 0.9066       | 0.9066         | 0.9066     | 0.0      | 0.0           | 27     | 0.2399    | 0.2399      | 0.2399        | 0.2399    | 0.0     | 0.0         
small  | 10      | 0.9584     | 0.9584       | 0.9584         | 0.9584     | 0.0      | 0.0           | 30     | 0.5519    | 0.5519      | 0.5519        | 0.5519    | 0.0     | 0.0         
medium | 6       | 0.9906     | 0.9906       | 0.9906         | 0.9906     | 0.0      | 0.0           | 8      | 0.7584    | 0.7584      | 0.7584        | 0.7584    | 0.0     | 0.0         
large  | 4       | 0.9989     | 0.9989       | 0.9989         | 0.9989     | 0.0      | 0.0           | 11     | 0.6906    | 0.6906      | 0.6906        | 0.6906    | 0.0     | 0.0         
button | 4       | 0.9489     | 0.9489       | 0.9489         | 0.9489     | 0.0      | 0.0           | 26     | 0.2532    | 0.2532      | 0.2532        | 0.2532    | 0.0     | 0.0         
all    | 28      | 0.9563     | 0.9563       | 0.9563         | 0.9563     | 0.0      | 0.0           | 76     | 0.4829    | 0.4829      | 0.4829        | 0.4829    | 0.0     | 0.0         
```

RealAppliance heldout:
```
bucket | n  | cell_iou | support_l1 | gtm_decode | e2e_decode
------ | -- | -------- | ---------- | ---------- | ----------
tiny   | 9  | 0.4352   | nan        | 0.4352     | 0.4352    
small  | 5  | 0.8055   | nan        | 0.8055     | 0.8055    
medium | 2  | 0.9218   | nan        | 0.9218     | 0.9218    
large  | 2  | 0.7985   | nan        | 0.7985     | 0.7985    
button | 9  | 0.4352   | nan        | 0.4352     | 0.4352    
all    | 18 | 0.6325   | nan        | 0.6325     | 0.6325    
```

worst heldout:
```
id                                                        | bucket | raw | cell   | GTcand | Predcand
--------------------------------------------------------- | ------ | --- | ------ | ------ | --------
50be70931dcb4a838b906382233a28f1/a0/control_knob_(left)_0 | small  | 52  | 0.0000 | 0.0000 | 0.0000  
50be70931dcb4a838b906382233a28f1/a0/oven_handle_0         | small  | 432 | 0.0000 | 0.0000 | 0.0000  
5ee9aa73c0344183ac2b14355b32e5ed/a0/control_buttons_0     | tiny   | 24  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_0                                        | tiny   | 17  | 0.0000 | 0.0000 | 0.0000  
101971/a0/button_1                                        | tiny   | 12  | 0.0000 | 0.0000 | 0.0000  
```
