# RealAppliance Door/Lid Panel Repair

Date: 2026-06-18

## Root Cause

Some RealAppliance logical door/lid/glass parts were represented in `part_info` as a frame or ring parent plus fixed child parts containing the middle panel/glass voxels or attached structure such as a handle. The VLM target points at the parent part, so the GT `ind_<parent>.npy` looked like an empty frame and missed visible geometry.

This is not a connectivity problem. The earlier forced-connectivity/bridge idea is not the right fix for these cases because it changes geometry without assigning the missing panel to the correct semantic part.

## Repair Rule

For reviewed panel candidates in:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_audit_20260618/likely_panel_child_candidates.json`

and anonymous fixed structural children from:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_audit_20260618/non_panel_group_children.json`

apply:

1. Identify the target parent door/lid/glass part and its selected fixed panel/structural child parts.
2. Load the true pre-repair parent and child voxels. Prefer `connectivity_repair_20260618/backup` to avoid carrying forward the earlier bridge repair; fall back to the panel-repair backup, then live data.
3. Set `ind_<parent>.npy = pre_parent union child_union`.
4. Delete the selected child `ind_<child>.npy` files so the same panel is not supervised twice.
5. Recompute `surface.npy` as the union of all remaining `ind_*.npy` files for that object angle.
6. Validate parent equality, removed child files, and `surface.npy == union(ind_*.npy)`.

The structural-child rule intentionally stays narrow: `joint_type == E`, anonymous `part_*` name, not a known control/button/knob, average voxels at least 80, and 2D area ratio at least 0.09. This covers door/lid handles and small fixed structural pieces without absorbing control buttons.

## Applied Result

Repair report:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_repair_20260618`

Final validation:

- 35 target parent parts after the structural-child extension
- 40 fixed child panel/structural parts
- 35 objects
- 341 checked object-angle records
- 0 final validation errors
- 298,959 voxels added to parent targets
- affected objects: `004,005,006,010,016,020,023,024,028,029,032,034,038,039,049,050,054,055,056,057,058,072,076,078,082,088,090,092,095,099`

The initial panel repair report remains at:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_repair_20260618`

The structural-child extension report is:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_structural_child_patch_20260618`

Packed data is stale after this repair. Rebuild or incrementally update:

`/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v4`

Cached SS latents/decoded preview targets are also stale after GT voxel edits.
For object `016`, the old decoded target for `玻璃盖_0` came from 2026-06-15 and
matched the old frame-only GT. On 2026-06-22, `016` was refreshed through Step 08
per-part latent encoding and Step 10 decoding, then the full preview was rebuilt.
After refresh, `016/玻璃盖_0` GT and decoded counts match again.

## Repro Script

The reproducible script is:

`scripts/data/repair_realappliance_door_lid_panel_children.py`

Dry-run:

```bash
python scripts/data/repair_realappliance_door_lid_panel_children.py
```

Apply:

```bash
python scripts/data/repair_realappliance_door_lid_panel_children.py --apply
```

The script writes `summary.json`, `repair_records.json`, and `validation.json` to a report directory. In apply mode it also backs up current live files before modification.

## Preview Update

The RealAppliance static preview reads GT target voxels from:

`preview/vlm_training/voxels/sample_*.js`

Refreshing only `decoded_voxels/` is not enough. After this repair, force a full static preview rebuild so `data/sample_*.js`, `voxels/sample_*.js`, overlay PNGs, and `index.html` all reflect the updated GT.

The 2D overlay masks still contain the original renderer labels. The preview uses:

`/robot/data-lab/jzh/art-gen/data/realappliance/manifests/door_lid_panel_repair_20260618/preview_label_merges.json`

to color selected child labels as their repaired parent target. Without this, repaired 3D voxels are correct but the 2D preview can still look like only the frame/ring is highlighted.
