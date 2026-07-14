# Joint Prompt Seg Local Refinement Spec

## Objective

Reduce same-part A/B/A/B fragmentation without weakening real part boundaries.
The joint multiclass softmax, body query, part queries, cosine similarity scoring,
and overlap partial-label behavior remain unchanged.

## Backward Compatibility

- `joint_local_mode` defaults to `none`.
- A checkpoint with no local-mode field and no `joint_local_*` state keys must
  reconstruct the legacy model and load with `strict=True`.
- The 0709 joint checkpoint is a warm-start source for new models. It is not a
  strict resume source because the new models add parameters and optimizer state.
- New checkpoints save their local mode and depth. Inference reconstructs the
  matching topology before strict state loading.
- Training-only boundary and affinity losses do not alter legacy inference.

## Model Variants

Both variants are inserted after joint voxel/query cross-attention and its MLP,
and before normalized voxel/query similarity scoring.

### `post_spconv`

Two residual `3x3x3` `SubMConv3d` blocks over active candidate voxels:

```text
h <- h + alpha_i * GELU(LayerNorm(SubMConv3d(h)))
```

Each scalar `alpha_i` is initialized to zero. The warm-start function is therefore
an identity at initialization. The first backward pass trains the gates; convolution
weights begin receiving non-zero gradients after a gate moves away from zero.

### `edge_graph`

Two residual graph blocks over occupied 6-neighbor voxel edges. Each block computes
a feature-similarity edge gate, aggregates neighbor values, subtracts the center
value, and applies an identity-initialized residual:

```text
g_ij = sigmoid(scale * cosine(LN(h_i), LN(h_j)) + bias)
h_i <- h_i + alpha_i * W_out(weighted_neighbor_mean_i - W_value(LN(h_i)))
```

This variant tests whether boundary-aware axis-neighbor propagation is preferable
to the fixed 3D convolution neighborhood.

## Loss

New runs use:

```text
L = joint_CE
  + lambda_overlap * overlap_partial_unary
  + lambda_boundary * boundary_CE
  + lambda_affinity * supervised_neighbor_affinity
```

- Boundary voxels are endpoints of valid GT cross-label neighbor edges.
- Boundary CE is normalized by the boundary voxel count within each object, then
  joint groups are averaged. This intentionally keeps the loss object-balanced.
- `target=-100` overlap/ignored voxels do not participate in boundary CE or
  supervised affinity.
- Same-label affinity minimizes `1 - dot(p_i, p_j)`.
- Cross-label affinity minimizes `dot(p_i, p_j)`.
- Affinity uses softmax probabilities and only the two supervised edge types.
- The legacy smooth same/all/cross terms are zero in new launchers. Its weight is
  retained only for the existing overlap partial-label unary.
- CRF evaluation is disabled in the new launchers.

Default new-run weights are:

```text
lambda_overlap = 0.2
lambda_boundary = 0.5
lambda_affinity = 0.2
same affinity weight = 1.0
cross affinity weight = 1.0
neighborhood = 6
```

## Metrics

- `joint_same_label_diff_pred_rate`: GT same-label neighbor pairs predicted as
  different labels.
- `joint_cross_label_same_pred_rate`: GT cross-label neighbor pairs predicted as
  the same label.
- `joint_boundary_iou_at1`: IoU between one-neighbor-dilated predicted and GT
  boundary voxel sets.
- `part` recall aggregates every non-body class, regardless of semantic label.
- `small` recall is a size bucket over non-body parts with source `raw_count < 500`.
- `drawer` recall includes non-body parts identified by drawer naming or by
  `prismatic`/`slide` joint metadata. Buckets overlap: a small prismatic part is
  included in `part`, `small`, and `drawer`.

Online metrics operate on the joint candidate set. Candidate and boundary coverage
must be considered when interpreting them; full-grid diagnostics remain the final
boundary evaluation authority.

## Training Entrypoints

- `scripts/train/part_promptable_seg/run_joint_local_spconv_L.bash`
- `scripts/train/part_promptable_seg/run_joint_edge_graph_L.bash`

Both default to the canonical v6 packed dataset and split, model size L, eight GPUs,
bf16, and warm-start from the 0709 joint step-100000 checkpoint.

Queue-ready commands:

```bash
bash scripts/train/part_promptable_seg/run_joint_local_spconv_L.bash
bash scripts/train/part_promptable_seg/run_joint_edge_graph_L.bash
```

Use `WARM_START` for 0709-to-new-topology initialization. Use `RESUME` only for
continuing a checkpoint produced by the same local mode and depth.

## Verification

- Prompt segmentation tests: `41 passed`.
- The real 0709 step-100000 checkpoint reconstructs as `joint_local_mode=none`
  and strict-loads with 474 state entries.
- Final 8-GPU, bf16, four-step v6 smokes completed forward, backward, evaluation,
  and checkpoint save for both variants.
- The final `post_spconv` and `edge_graph` checkpoints reconstruct their topology
  from checkpoint metadata/state and strict-load with 482 and 488 state entries.
- Non-zero optimizer moments on internal local-block parameters after four steps
  confirm that training progressed beyond the initially zero gates.
