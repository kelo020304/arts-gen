

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
