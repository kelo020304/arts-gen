# Kin Agent v2: decoded-geometry self refinement

This path addresses a structural limitation in the original harness: its
`LimitEstimate` action only changes axis and range, while joint type and origin
come from upstream context.  The v2 numerical core proposes all four fields:

- joint type: `prismatic` or `revolute`;
- signed world axis;
- world origin (the hinge line point for revolute joints);
- lower and upper motion limits.

It consumes decoded moving-part and decoded body geometry.  It does not read
dataset joint annotations, source USD joint properties, `part_info` axis/range,
or GT mesh.  The default budget is seven iterations and the API rejects a
budget of ten or more.

## Iteration design

Iteration 1 builds a compact proposal bank from world axes, moving-part PCA,
the moving/body center relation, and the nearest moving/static contact strip.
Subsequent rounds consume structured critic feedback and make bounded axis
rotations and hinge-origin offsets.  Prismatic joints now use a shrinking
`12/6/4...` degree trust region instead of stopping after the initial bank;
revolute joints retain the calibrated `8/iteration` schedule.  Every trace row
records the incumbent, issues/actions, generated proposals, validation score
gain, accept/keep decision, and stop reason.  Convergence stops early when no
legal revision exists, score gain is negligible, or the acceptance score is
reached.

Nearest-neighbor collision queries use a `cKDTree` when SciPy is available,
with the original blocked NumPy implementation retained as a fallback.  This
reduced the representative 15-part run from minutes to tens of seconds without
changing the collision metric.

The LLM-facing layer should only choose a proposal or request one of these
bounded refinements.  Geometry code owns transforms, collision checks and
scores.  This prevents free-form numerical guesses and makes an under-ten-turn
agent reproducible.

## Metric separation

The following signals are legal during inference because they need no GT:

- excess moving/static collision over the rest pose;
- endpoint displacement and non-degenerate motion;
- usable collision-free range;
- moving-shape/type consistency;
- hinge/contact-strip or slider/support-axis consistency;
- adaptive decoded-mesh collision sweep and exact Manifold overlap volume for
  watertight decoded body/part meshes;
- multi-view silhouette/reprojection consistency when calibrated observations
  are supplied by the ee-eval frontend.

The following metrics are benchmark-only and must never enter prompts, candidate
generation, stopping decisions, or reranking:

- joint type accuracy against dataset labels;
- signed/unsigned axis angular error against GT;
- hinge-line/origin distance against GT;
- lower/upper endpoint error or range IoU against GT;
- success measured by matching source USD authored joint properties.

The current eval-platform helper that creates `vlm_initial_from_gt.json` from
dataset `part_info` is therefore not a valid v2 input or accuracy experiment.

## Multi-state observation critic

Dataset-mode runs can use the legal observation bundle under
`renders/<object>/angle_*/`: calibrated `camera_transforms.json` plus per-part
2D boxes in `bbox_gt.json`.  These files contain imagespace observations, not
joint parameters.  The critic triangulates a coarse part center for every
state from multiple camera rays, then fits either a line trajectory or a 3D
circle.  It only overrides the decoded-mesh proposal when the trajectory is
observable: translations use the dominant displacement, while revolute
motions require a low plane residual.  Pure knob spin is explicitly treated
as centroid-unobservable.

This evidence is explicit in the workbench. Dataset runs default to
`dataset_motion_states` when those states exist, while the user can switch to
`static_decoded_geometry`; wild uploads only expose the static mode. The result
JSON records `evidence_mode` and `motion_observation_root`, so cached results
and reported metrics cannot silently mix the two conditions.

For semantically ambiguous lids, the critic now fits both hypotheses to the
same ordered trajectory.  A train-selected secondary/primary SVD ratio of
`0.10` separates line-like slides from curved hinges; values near the threshold
remain marked for review.  Doors use a bounded axis-family critic: when the
motion and geometry families disagree, moderate motion evidence may select an
already-generated cardinal proposal, but it cannot invent a free-form axis.

The observed interval is expressed relative to the decoded `angle_0` pose.  It
is an observed state span, not proof of a mechanical stop.  Range prior v3,
built only from canonical split `train_ids`, stores signed lower/upper
quantiles in a canonical decoded axis frame.  It uses signed intervals for
centroid-unobservable knobs/lids, a group-CV Ridge calibrator for 0511 drawers,
and a bounded span envelope for noisy door arcs.  Result JSON keeps the raw
observed interval, estimated usable interval, q90 prediction interval, export
interval, and `mechanical_stop_confirmed=false` separate.

PhyX rotary knobs use an additional decoded-only thin-axis critic.  It takes
the smallest-variance PCA family, then selects an existing cardinal proposal
only when family confidence is at least `0.80` and score drop is at most
`0.20`; it never invents an unconstrained axis.  RealAppliance keeps its
separate train-only axis-family model. If that model abstains, RealAppliance
uses the same decoded-only critic only under a stricter `0.95` confidence and
`0.15` score-drop gate. This fallback left the frozen expanded benchmark
metrics unchanged while covering a live decoded-mesh domain gap.

Static PhyX doors use a separate contact-axis critic only when no articulated
motion observation is available. It requires the moving-part dominant PCA and
the nearest 25% body-contact strip to agree on the same non-Y cardinal family,
with confidence at least `0.65`. It can only select an existing validated
proposal whose score is within `0.15` of the incumbent. Disagreement is kept as
a review signal instead of forcing a hinge axis.

Before XML/USD export, the bundle runs an adaptive Open3D sweep over decoded
meshes.  Watertight broad-phase hits are confirmed with Manifold intersection
volume; non-watertight or unavailable exact checks are never silently passed.
Moving-part pairs are also checked over bounded lower/mid/upper combinations.
Collision hits now enter one final bounded feedback round. It finds the
zero-connected clear interval with at most six Manifold bisections per side.
The interval is only written to XML/USD when it retains at least 35% of the
proposal, preserves the observed motion interval, and the axis has independent
support. Otherwise the incumbent is kept and the trace records a segmentation,
body-boundary, axis, or range conflict for review. Geometry, evidence fusion,
and collision feedback share one global 1--9 round budget.

Knob axis families use a second frozen train-only model.  Its input is the
visible label, calibrated part location, decoded body/part bounds, and the
existing proposal-bank scores.  It predicts only X/Y/Z family and then reranks
the numerical proposals; it never regresses a free-form axis.  The artifact
contains vectorizer/scaler/model coefficients and no per-object GT rows.  Its
official-train object-group 5-fold accuracy is 96.8%.

## Run

```bash
python -m post_process.kinematic_solver.run_kin_agent \
  --body-obj /path/to/decoded_body.obj \
  --moving-obj /path/to/decoded_part.obj \
  --out-dir /tmp/kin_agent_run \
  --object-name ra_001 \
  --joint-name drawer_0 \
  --max-iterations 7
```

The run writes `kinematic_result.json`, a combined `object.xml`, and a portable
`object.usda`.  USDA embeds decoded mesh geometry and decoded vertex color; it
does not merely store GLB/OBJ paths as custom metadata.  Its joints use standard
USD Physics axis/local-frame attributes.  For RealAppliance delivery only, both
exporters place the Z-forward correction on the root object and override the
final prismatic joint to local `axis="0 0 1"`.  That delivery override is kept
out of inference scoring. Legacy RealAppliance benchmark OBJs that had this
rotation baked into their vertices are explicitly mapped back to the current
unbaked decoded frame before inference, then mapped through the root rotation
into `part_info` coordinates for evaluation.

## Expanded decoded benchmark

`benchmark_kin_agent.py` enforces two processes: `infer` freezes predictions
from decoded SLat OBJ files, then `evaluate` is allowed to open annotations.
The primary benchmark is now the fixed 139-part / 82-object expanded cohort
(130 RealAppliance parts and 9 PhyX parts). The corrected delivery-frame
multi-state result is:

- type accuracy: 100%;
- axis error median / mean: 0 / 7.55 degrees;
- drawer axis mean: 0 degrees;
- door axis median / mean / p90: 0 / 9.61 / 28.25 degrees;
- knob axis median / mean / p90: 0 / 10.70 / 35.0 degrees;
- lid axis median / mean / p90: 0 / 0.16 / 0 degrees;
- prismatic range endpoint error mean: 0.037 normalized asset units;
- revolute range endpoint error mean: 0.207 radians;
- iterations mean/max: 5.12/7.

These numbers use nine to ten articulated render states per object. They are
not static 1--4 image results. The full zero-observation pipeline on this same
RA-heavy cohort is substantially weaker: type 96.40%, axis mean/median
62.63/90 degrees, with 101 errors above 30 degrees. It must not be presented as
equivalent to the multi-state benchmark or as evidence for wild-image range
recovery.

The canonical split diagnostic contains 117 train parts and 22 heldout parts.
For multi-state inference, heldout type accuracy is 100% and axis mean/median
is 10.68/0 degrees. For static inference, heldout type accuracy is 95.45% and
axis mean/median is 52.68/82.5 degrees. This is not an untouched final test:
earlier aggregate benchmark feedback exposed the cohort during solver
development. It remains useful as an object-disjoint diagnostic.

A second decoded-only manifest was frozen from `0707-64-cfy` before opening
annotations.  It contains 137 parts / 44 objects not present in expanded-139,
including 32 parts / 11 objects from the canonical official heldout split.
After applying the fixed PhysX-0511 frame convention, the multi-state result is
type 99.27% with axis mean/median 1.94/0 degrees. Official-heldout type is 100%,
axis mean/median is 0/0 degrees, and range endpoint mean is `0.096`. The full
static pipeline on this cohort is also strong: type 99.27%, axis mean/median
1.83/0 degrees, three errors above 30 degrees, and heldout axis mean/median 0/0
degrees. Its heldout range endpoint mean is `0.162`, reflecting the absence of
observed stops. This cohort is dominated by 80 convention-consistent 0511
drawers and contains only a small set of PhyX knobs, so it does not contradict
the weak RA-heavy static result and must not be generalized to arbitrary wild
objects.

The earlier 15-part manifest remains a fast smoke/diagnostic set only.  It is
not used to claim aggregate accuracy because it substantially under-sampled
hard lids and knob/door axis-family failures.

Static-only geometry still reaches the tested search boundary without proving
a mechanical stop.  Dataset multi-state observations substantially improve
axis and relative-pose range, but the result remains censored unless a stop is
directly observed. The corrected expanded multi-state long tail contains three
door-axis and eleven knob-axis errors above 30 degrees. Wild runs with only one
state continue to use geometry plus train-only priors and must not be described
as recovering exact mechanical limits.

Revolute-origin comparison against `part_info` remains diagnostic only. The
decoded delivery mesh and annotation frames do not yet have a fully audited
object-level similarity registration for every dataset, so normalized origin
offsets are not evidence that a delivered hinge line is accurate. Decoded-mesh
motion and collision validation is authoritative for the exported XML/USD
frame.

The workbench displays type/axis/range confidence, observed/estimated/q90
intervals, stop status, decoded collision results, review reasons, the delivered
axis/range (including RealAppliance local-Z canonicalization), a per-round
feedback timeline, an axis/origin guide, and a MuJoCo qpos validation image.
Matching inputs and solver format reuse the cached result; changing decoded
inputs, priors, solver format, or decoded-collision audit version invalidates it.
