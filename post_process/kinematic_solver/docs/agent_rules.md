# Agent Rules

- Edit only the editable region between `# >>> USER_CODE_START` and
  `# >>> USER_CODE_END` in `post_process/kinematic_solver/estimate_limit.py`.
- Define `estimate_limits(ctx)`.
- Return `list[LimitEstimate]`, one per joint.
- Use `ctx.joints` and `ctx.evidence`; do not open source USD files or GT limit
  JSON.
- Positive prismatic motion follows `axis_world * q`.
- Keep reasoning in `LimitEstimate.reason` so compile feedback is auditable.
- Treat this as an iterative action loop. For each joint, maintain an implicit
  state in the returned reasoning: `need_fix` or `correct`.
- In one iteration, each joint may either remain unchanged or take this bounded
  action budget:
  - `axis_world` starts from the SDK geometry baseline or the VLM/authored axis
    when no relation-based SDK baseline exists. The agent may micro-rotate
    the current unit vector by no more than 5 degrees, but must not switch to a
    different axis family. The axis can be any normalized 3D direction, not only
    `X/Y/Z`;
  - change `lower` at most once;
  - change `upper` at most once.
- Prismatic limit actions are exactly `+/-10mm`, `+/-5mm`, `+/-2.5mm`,
  `+/-1mm`, or `+/-0.5mm`, expressed in meters in `LimitEstimate`.
- Revolute limit actions use the same numeric ladder in degrees, expressed in
  radians in `LimitEstimate`.
- If `ctx.evidence[joint]["initial_estimate"]` exists, treat its limit as the
  rough VLM starting range. The SDK may replace the VLM axis with a
  geometry-derived baseline before the agent loop starts. This baseline uses
  Articraft-style rest-pose relations described in `docs/axis_validation.md`;
  PCA must not be treated as an axis source of truth.
- Joints that are already correct should stay unchanged in later iterations.
- Validation follows the Articraft compile/QC pattern, not a one-shot numeric
  guess. The harness compiles the current editable code, drives sampled
  articulation poses, checks actual overlap/contact outcomes, and returns a
  structured `<compile_signals>` block with `failure`, `warning`, and `note`
  records. Blocking failures are authoritative and must be fixed before a joint
  is marked `correct`.
- For pull-out prismatic joints, an axis action is not valid just because a
  short segment can move without creating a new pair. The SDK also probes the
  searched endpoint: if the moving part still overlaps the parent/static
  geometry, that axis trial is reported as `endpoint_overlap` and cannot be the
  selected action.
- For prismatic joints with a clear rest-face exit, validation also rejects
  candidate axes that deviate from the SDK exit axis by more than 5 degrees.
  This catches cases such as a drawer moving along `+Z` or a slanted axis
  through the body when the actual exit face is `+X`.
- This mirrors the useful part of Articraft's `TestContext` behavior:
  candidate joints are authored by the agent, then judged by driven poses and
  sampled geometry QC. The SDK may provide axis trials and selected actions as
  feedback, but the agent must still make bounded edits in `estimate_limit.py`.
- Validation writes per-joint `correct` / `need_fix` state. Use those states,
  the `<compile_signals>` block, and the visual result for the next iteration.
- During API runs, the harness may call the model for multiple iterations, but
  each iteration may only replace the editable region in `estimate_limit.py`.
  It must not edit `sdk/`, `docs/`, tests, or the harness wrapper.
