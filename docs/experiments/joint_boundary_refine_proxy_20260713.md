# Joint Boundary Refiner Proxy Check (2026-07-13)

Checkpoint:

```text
/robot/data-lab/jzh/art-gen/ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint/ckpts/latest.pt
```

Evaluation used 11 proxy-held objects, 76 classes, and 74,518 shared candidate
voxels. Parameters were selected on an interleaved 6-object development subset
and checked on the remaining 5 objects.

Locked guarded-refiner parameters:

```text
raw logit top-2 margin quantile: 0.01
iterations: 1
pairwise weight: 3.0
neighborhood: 6
minimum vote gain: 0.0
preserve predicted classes <= 32 voxels
probability margin threshold: 0.0
```

## Full Proxy-Held Result

| method | mean IoU | part mean IoU | boundary error | cross-label same-pred | changed voxels |
|---|---:|---:|---:|---:|---:|
| raw argmax | 0.473340 | 0.432517 | 0.363478 | 0.626857 | 0 |
| existing CRF (5 iter, 0.3) | 0.479166 | 0.439005 | 0.362430 | 0.638559 | 474 |
| guarded refiner | 0.477877 | 0.437743 | 0.362197 | 0.631872 | 281 |

Relative to raw argmax, the guarded refiner improved mean IoU by 0.004537 and
reduced boundary error by 0.001280. Relative to the existing CRF it changed
40.7% fewer voxels, had lower boundary error, and had lower cross-label
same-pred rate, while mean IoU was lower by 0.001289.

The 5-object untouched audit showed the same direction: guarded refinement
improved mean IoU by 0.008739 and reduced boundary error by 0.001061 relative
to raw argmax.

## Decision

The refiner remains opt-in. The proxy result supports using it for an RA-40 and
general heldout A/B, but it is not strong enough to change the delivery default
without that gate. The more important training fix is partial-label supervision
for multi-claim contact voxels, local propagation across the overlap band,
same-label attraction, and cross-label repulsion. The boundary auxiliary must
be enabled with `JOINT_SMOOTH_WEIGHT>0`; the next-run locked starting point is
`0.2` with same-label `1.5`, all-label `0`, and cross-label `1.0`.
