# PartSeg Method Overview Fact Sheet

This file is the sole source for labels used in `partseg_arch.svg` / `partseg_arch.png`.
All module labels below come from code; the VLM card text is the user-specified fixed text.

## RUN-T1 Context

- Training command uses `MODEL_SIZE=S`, so PartSeg uses `dim=256`, `depth=6`, `heads=8`; `run_train.bash` maps model size S to these defaults at `scripts/train/part_promptable_seg/run_train.bash:67`.
- T1 default route is `ROUTE=voxel`, prompt encoder is `MASK_ENCODER=fg_points`, whole occupancy comes from packed whole coords, and route voxel is passed to the training script at `scripts/train/part_promptable_seg/run_train.bash:55`, `scripts/train/part_promptable_seg/run_train.bash:56`, `scripts/train/part_promptable_seg/run_train.bash:57`, `scripts/train/part_promptable_seg/run_train.bash:157`.
- T1 passes `--head-depth`, `--voxel-depth`, `--mask-encoder`, `--point-resample-points`, `--voxel-embedding-dim ${VOXEL_EMBEDDING_DIM:-0}`, and `--spconv-depth` at `scripts/train/part_promptable_seg/run_train.bash:154` through `scripts/train/part_promptable_seg/run_train.bash:165`.
- T1 NEW flags are mapped at `scripts/train/part_promptable_seg/run_train.bash:194` through `scripts/train/part_promptable_seg/run_train.bash:203`: `BOUNDARY_BAND_RADIUS`, `BOUNDARY_HARD_MINING`, `BOUNDARY_HARD_MINING_TOPK`, `BOUNDARY_HARD_MINING_WEIGHT`, `NEGATIVE_PROMPT_CHANNEL`, `NEGATIVE_PROMPT_EQUIV_CHECK`, `VOXEL_CORRUPT`, `VOXEL_CORRUPT_DROP_PROB`, `VOXEL_CORRUPT_SHELL_PROB`, `VOXEL_CORRUPT_SPECKLE_PROB`.
- User RUN-T1 values for the figure: `route=voxel`, `mask_encoder=fg_points`, `head_depth=2`, `voxel_depth=3`, `spconv_depth=4`, `voxel_embedding_dim=0`, `semantic_aux=1`, `mask_target=support`, `support_multiplier=4.0`, `boundary_weight=2.0`, `boundary_band_radius=2`, `boundary_hard_mining=1`, `boundary_hard_mining_topk=0.2`, `boundary_hard_mining_weight=2.0`, `negative_prompt_channel=1`, `negative_prompt_equivalence_check=1`, `voxel_corrupt=1`, `drop/shell/speckle=0.03/0.08/0.0003`, `SEG_DISCRIMINATIVE=0` so `joint_seg=False`.

## VLM Panel Fixed Text

- VLM input label: `multi-view renders`.
- VLM outputs: `selected 4-view group`, `per-part semantic names`, `per-view 2D part masks`.
- The `selected 4-view group` line goes to `DINOv2 frozen` and `SS Flow trainable multi-view concat`.
- The semantic names plus 2D masks are packed as `part prompts` and cross into Panel (b).
- Required note: `VLM output is the single semantic source for both SS conditioning and seg prompts`.

## Module Facts For Panel (a)

### DINOv2 / SS Flow Conditioning

- Input: selected render views. In ee-eval, `slat_view_indices` come from CLI or manifest `view_indices`, then tokens are loaded from live TRELLIS image preprocessing or cached DINO tokens at `scripts/eval/tasks/ee_0617_single.py:1187` through `scripts/eval/tasks/ee_0617_single.py:1199` and `scripts/eval/tasks/ee_0617_single.py:348` through `scripts/eval/tasks/ee_0617_single.py:402`.
- Token shape fact: cached DINO tokens must be `[V,T,1024]`; selected tokens are flattened for flow as `[V*T,1024]` at `scripts/eval/tasks/ee_0617_single.py:387` through `scripts/eval/tasks/ee_0617_single.py:401`.
- SS stage record says the input is `4-view DINO tokens`, fusion mode is `concat`, at `scripts/eval/tasks/ee_0617_single.py:1472` through `scripts/eval/tasks/ee_0617_single.py:1475`.
- Output for PartSeg is saved/loaded as `ss_latent.npy` with shape `[8,16,16,16]` and whole `voxel.npz` at `scripts/eval/tasks/ee_0617_single.py:293` and `scripts/eval/tasks/ee_0617_single.py:637` through `scripts/eval/tasks/ee_0617_single.py:639`.

### SS Encoder / Decoder

- SS encoder checkpoint defaults to `ss_enc_conv3d_16l8_fp16.safetensors`; decoder defaults to `ss_dec_conv3d_16l8_fp16.safetensors` at `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:53` through `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:69`.
- Encoder config file has `SparseStructureEncoder`, `in_channels=1`, `latent_channels=8`; decoder config has `SparseStructureDecoder`, `out_channels=1`, `latent_channels=8`.
- SS encoder forward consumes an occupancy grid and outputs posterior mean `z` when `sample_posterior=False` at `TRELLIS-arts/trellis/models/sparse_structure_vae.py:186` through `TRELLIS-arts/trellis/models/sparse_structure_vae.py:207`.
- Empty code is computed by encoding a zero grid `[1,1,64,64,64]`; returned latent is used as `[8,16,16,16]` at `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:1308` through `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:1311`.
- SS encoder and decoder are loaded in eval mode and all parameters are frozen at `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:1280` through `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:1305`.

## Module Facts For Panel (b)

### PartSeg Inputs

- `z_global`: input latent grid shape `[B,8,16,16,16]` in T1. The PartSeg model checks `[B, latent_channels, 16,16,16]`, with default `latent_channels=8`, at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:261` and `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:663` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:671`.
- `masks2d`: prompt mask input shape `[B,4,512,512]` for T1, because model defaults are `num_views=4`, `mask_size=512` at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:263` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:267`, and `PointMaskEncoder` checks `[B,V,H,W]` at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:350` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:358`.
- `full_occ`: voxel-route input shape `[B,1,64,64,64]`; `forward_voxels` checks it at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:780` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:797`.
- `candidate_cells`: voxel-route input shape `[B,16,16,16]`; `forward_voxels` checks it at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:780` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:795`.

### Prompt Encoder: `fg_points`

- T1 selects `PointMaskEncoder` when `mask_encoder="fg_points"` at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:300` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:313`.
- The encoder samples only foreground mask pixels and one centroid token per visible view; this is stated in the module header at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:1` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:6`.
- Point types are `interior`, `boundary`, `centroid` at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:26` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:29`.
- Sampling per 2D view: erode mask by 3x3, boundary is `mask XOR eroded`, split connected components, allocate up to `k_boundary=32` and `k_interior=32`, sample boundary along contour order, sample interior by FPS, append centroid, and store distance-to-boundary and log area at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:203` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:272`.
- Feature encoding: `[u,v]`, Fourier features with 10 bands, distance-to-boundary, normalized log area, plus view embedding and point-type embedding, then LayerNorm at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:319` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:420`.
- Output shape: variable tokens `[B,T,256]`, key padding mask `[B,T]`; maximum T1 tokens per sample is `4*(32+32+1)=260` at `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:281` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:317` and `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:360` through `TRELLIS-arts/trellis/models/part_seg/point_mask_encoder.py:420`.
- Trainability: prompt encoder is part of `PromptablePartLatentSegNet` and is trainable in route voxel T1 unless a checkpoint freeze applies externally; route-freeze only freezes latent-only modules, not prompt encoder, at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4605` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4666`.

### NEW: Negative Prompt Channel

- Training builds `negative_masks2d` as the union of other parts' 2D masks from the same `(dataset_id, obj_id, angle_idx)` group after view dropout; output shape matches `masks2d [B,V,H,W]` at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:1081` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:1119`.
- The model encodes negative masks with the same prompt encoder, pools them to one summary per sample, applies `negative_prompt_proj`, and adds the projected context to all positive prompt tokens at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:435` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:459`.
- Zero initialization: `negative_prompt_proj = nn.Linear(256,256,bias=False)` and its weight is zero-initialized at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:328` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:330`.
- Equivalence check: when enabled, training compares outputs with and without negative masks and requires max absolute difference `<= 1e-6` at init at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:3897` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:3978`.
- Constraint: negative prompt channel is rejected with joint segmentation at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4694` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4695`.

### Cell Trunk And Per-cell Logits

- Stem input is latent grid plus xyz coordinates when `use_xyz=True`: `[B,8+3,16,16,16] -> [B,4096,256]`, with learned 3D position embedding, at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:314` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:325` and `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:673` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:686`.
- Trunk has `depth=6` `TrunkBlock`s for RUN-T1. Each block applies local 3D depthwise+pointwise conv, self-attention, cross-attention to prompt tokens, and MLP at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:131` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:188` and `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:331`.
- Cell mask logits shape is `[B,4096]` from `head1(LN(feat))` at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:739` and `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:805` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:806`.
- Trainability: the cell trunk and `head1` are trainable for route voxel T1; route freeze only freezes latent-only modules at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4627` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4636`.

### Active T1 Voxel Route

- T1 active output path is route `voxel`, not route `latent`. The model is constructed with `use_voxel_head=True` when `args.route == "voxel"` at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4995` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:5013`.
- Training candidate cells are `dilate(m_gt)` and full occupancy is loaded from packed whole coords when `use_packed_whole_occ=True`; corruption is applied before model forward at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2629` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2650`.
- Candidate 16^3 cells are expanded to 64^3 by repeating each cell 4 times per axis, then intersected with `full_occ > 0.5`; valid 64^3 coords are packed per sample at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:811` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:852`.
- Voxel token feature is parent cell feature plus Fourier 3D coordinate projection plus 5x5x5 full-occupancy patch projection in token refine mode at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:854` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:881`.
- RUN-T1 voxel refiner is `voxel_depth=3` `SparseTokenBlock`s; each block cross-attends voxel tokens to prompt tokens and applies an MLP at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:191` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:225`, `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:356` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:360`, and `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:884` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:894`.
- Output is `voxel_logits [B,max_len]`, `voxel_coords` list of packed 64^3 coords, and `voxel_pad_mask [B,max_len]` at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:927` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:945`.
- Trainability: voxel head/refiner is trainable for route voxel T1. Voxel embedding head is off in RUN-T1 because `run_train.bash` passes default `--voxel-embedding-dim 0` at `scripts/train/part_promptable_seg/run_train.bash:160`.

### Inactive Latent Route / SS Decoder Path

- Route latent path computes `m_embedding`, `head2_in`, `head2_blocks`, `delta [B,8,16,16,16]`, and `part_latent = m*(z_global+delta)+(1-m)*empty` at `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:755` through `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py:778`.
- This is not the active RUN-T1 path. In route voxel training, latent-only modules `m_emb`, `head2_in`, `head2_blocks`, `head2_norm`, and `delta` are frozen at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4627` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4636`.
- SS decoder is frozen when loaded, and may be used for latent-route decode or fallback whole occupancy decode, but route voxel T1 main output is `voxel_logits`, not `part_latent -> SS decoder`.

### NEW: Boundary-band Hard Supervision

- Boundary band is computed on 16^3 masks. It detects 6-neighbor label changes, then dilates the boundary by max-pooling with kernel `2*radius+1` at `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:376` through `scripts/train/part_promptable_seg/part_promptable_seg_utils.py:409`.
- In RUN-T1, radius is `2`; because `mask_target=support`, the boundary band is computed from raw `m_raw` before support-target replacement, while the loss target may become support-expanded `m_gt` at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2524` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2544`.
- Loss implementation: BCE uses per-sample positive weight `(neg/pos).clamp(4,1000)`, optional focal term, boundary BCE multiplier `boundary_weight`, and top-k hard mining within the boundary band by detached BCE loss at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:921` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:993`.
- RUN-T1 label values: `boundary_weight=2.0`, `boundary_hard_mining=True`, `topk=0.2`, `hard_mining_weight=2.0`, `focal_gamma=1.5`.

### NEW: Structured Voxel Corruption

- Applies to `full_occ [B,1,64,64,64]` before route voxel forward, not to target part voxels, at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2629` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:2642`.
- Corruption steps: randomly drop occupied voxels with `drop_prob`; compute 3x3x3 dilation shell and randomly add shell voxels with `shell_prob`; randomly add exterior speckles outside the dilated occupancy with `speckle_prob`; restore original occupancy for any sample that becomes empty at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:1143` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:1202`.
- RUN-T1 label values: `drop=0.03`, `shell=0.08`, `speckle=0.0003`.
- Constraint: voxel corruption requires `route=voxel` and is rejected with `joint_seg=True` at `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4696` through `scripts/train/part_promptable_seg/train_part_promptable_seg.py:4699`.

## Inference / ee-eval Facts

- Part stage loads the PartSeg checkpoint, constructs `PromptablePartLatentSegNet` from checkpoint args, sets eval mode, and freezes all parameters at `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:57` through `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:84`.
- Part prompt masks are loaded for each part from dataset mask paths, downsampled to `512x512`, stacked as `[V,512,512]`, and must have at least one prompt view at `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:103` through `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:113`.
- Route voxel default inference is two-forward per part: first forward with all 16^3 candidate cells to get `m_logit`, threshold `sigmoid(m_logit)>0.5` to `pred_m`; second forward with `candidate_cells=dilate(pred_m)` and `full_occ`; threshold voxel logits at `voxel_threshold=0.5` to write part voxels at `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:296` through `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:307` and `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:373` through `TRELLIS-arts/inference_pipeline/part_prompt_seg_stage.py:389`.
- `ee_0617_single.py` ensures SS and part stages have produced `ss_latent.npy`, `voxel.npz`, and part voxel files at `scripts/eval/tasks/ee_0617_single.py:289` through `scripts/eval/tasks/ee_0617_single.py:316`.
- Optional T0-lite postprocess (`--part-t0-filter`) reruns per-part logits, stacks a default body logit with independent part logits, performs argmax over body+parts, sets outside whole occupancy to body, optionally smooths ambiguous top-2 score-margin voxels by 26-neighborhood majority, and optionally connected-component filters small remote islands to body at `scripts/eval/tasks/ee_0617_single.py:630` through `scripts/eval/tasks/ee_0617_single.py:762`.
- T0-lite defaults: threshold `0.5`, margin `0.35`, smooth iters `1`, CC min component voxels `32`, min fraction `0.05`, max component distance `2` at `scripts/eval/tasks/ee_0617_single.py:1096` through `scripts/eval/tasks/ee_0617_single.py:1117`.
- CC filter keeps the largest component plus components that are large enough or close to the largest; removed small remote islands become body residual at `scripts/eval/tasks/ee_0617_single.py:765` through `scripts/eval/tasks/ee_0617_single.py:852`.
- Final body is not a direct PartSeg output in the per-part route. It is `whole_coords minus union(part_coords)` at `scripts/eval/tasks/ee_0617_single.py:443` through `scripts/eval/tasks/ee_0617_single.py:459`, used at `scripts/eval/tasks/ee_0617_single.py:1238` through `scripts/eval/tasks/ee_0617_single.py:1274`, and recorded at `scripts/eval/tasks/ee_0617_single.py:1445` through `scripts/eval/tasks/ee_0617_single.py:1448`.

## Draft-vs-Code Differences To Reflect In The Figure

1. The draft main path says `64^3 Voxel -> SS Encoder -> Latent Grid` inside Panel (b). Code/eval usually passes precomputed `ss_latent.npy` / `z_global [8,16^3]` into PartSeg; SS encoder is frozen and can be shown as producing the latent upstream, not as an operation inside every PartSeg forward.
2. The draft says `per-cell logits -> Part Latent -> SS Decoder -> Part Voxels 64^3`. That is the latent route, but RUN-T1 active route is voxel: `per-cell logits` select/dilate candidate 16^3 cells, packed 64^3 voxel tokens predict `voxel_logits`, and threshold/argmax postprocess produces part voxels. The latent route modules are frozen in RUN-T1.
3. The draft names `Negative Prompt Channel` as "other part mask union, zero-init". Code confirms the union and zero-init, but injection is not a raw voxel/input channel; it is encoded with the same 2D prompt encoder, pooled, projected by zero-init `negative_prompt_proj`, and added to positive prompt tokens.
4. Boundary-band supervision is on 16^3 cell masks from 6-neighbor boundary dilation, not a 64^3 geometric surface loss. With RUN-T1 `mask_target=support`, the band is computed from raw cell mask before support-target replacement.
5. Structured voxel corruption modifies the full occupancy input before voxel-route forward; it does not corrupt target part voxels. Actual perturbations are occupied dropout, shell additions, and exterior speckles.
6. `Argmax + CC` is optional T0-lite ee-eval postprocess, not the default training graph. Default part stage thresholds each part independently; T0-lite stacks body+part logits for single-owner labels, then optional smoothing and CC filtering.
7. `Body = Residual` is computed in ee-eval as whole occupancy coords minus union of final part coords; it is not emitted by the standard per-part PartSeg forward.
8. The VLM selected 4-view group applies to SS/SLat conditioning by manifest or CLI view indices. Part prompt masks are loaded from dataset mask paths for each part; the PartSeg model itself does not choose the prompt views.

## Figure Checklist

- VLM has three outgoing lines: selected 4-view group; per-part semantic names; per-view 2D part masks.
- SS encoder/decoder are marked frozen.
- Panel (b) main path uses route voxel: `z_global [8 x 16^3]`, `fg point prompts [<=260 x 256]`, `6 TrunkBlocks`, `m_logit [4096]`, packed `64^3` voxel tokens, `3 SparseTokenBlocks`, `voxel_logits`, threshold/T0-lite postprocess, part voxels.
- NEW components match code: negative prompt prompt-token context; boundary-band hard supervision with r=2 / weight=2 / top20% x2; full occupancy corruption with drop/shell/speckle probabilities.
- Body residual and optional T0-lite argmax+CC are labeled as inference/postprocess, not training forward.
- All dimensions shown in the figure appear in this file.
