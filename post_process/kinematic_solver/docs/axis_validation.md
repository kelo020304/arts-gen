# Axis Validation

> Last updated: 2026-05-28

This document records the runtime contract for KinematicSolver joint-axis
validation. Tests may cover these rules, but the executable logic lives in
`post_process/kinematic_solver/sdk/`.

## Runtime Ownership

- `sdk/axis_candidates.py` owns SDK-generated axis baselines.
- `sdk/motion_validation.py` owns validation of driven poses.
- `sdk/agent_loop.py` owns the bounded agent action contract.
- `estimate_limit.py` is the only agent-maintained Python file. The agent may
  edit only its editable region.
- `tests/kinematic_solver/` contains regression tests only. Runtime code must
  not import from tests.

## Axis Rule

The SDK should not trust PCA as an axis source. PCA can pick a visually plausible
axis that reduces AABB overlap while still driving the part through the body, or
can make a knob look valid even when it rotates around the wrong physical mount
axis.

The current SDK baseline uses Articraft-style pose relations:

- For prismatic joints, find the body face where the moving part already
  protrudes at rest.
- Require projected overlap on the two non-motion axes, similar in spirit to
  Articraft `expect_overlap(...)` and `expect_within(...)`.
- Use that signed face normal as the recommended prismatic axis.
- For revolute controls, use the body-relative mount/outward relation as the
  candidate axis. If there is no body/mount relation and no authored/VLM axis,
  the SDK should not invent one from PCA.
- PCA may be used only as debug evidence during investigation, not as a
  recommended axis or pass criterion.

Example: for `ra_036 part_07`, the drawer/basket protrudes through the `+X`
face, so the SDK baseline is `axis_world=[1, 0, 0]`.

If the initial JSON marks such a part as `revolute`, the SDK does not silently
change the joint type. Instead it records `joint_type_warning` in evidence so
the run can tell the user that the VLM/type input is inconsistent with the
rest-pose slider geometry.

## Validation Rule

Continuous-axis validation must reject a prismatic candidate whose axis does not
match the rest-face exit axis within `5` degrees. The failure message must expose:

- `target_axis_world`
- `candidate_axis_world`
- `max_angle_degrees`

After axis validation, endpoint motion is still checked by sampled geometry:

- the moving part must not still intersect parent/static geometry at the driven
  endpoint;
- the endpoint overlap ratio must drop enough to prove visible clearance;
- otherwise the joint remains `need_fix`.

This mirrors the useful part of Articraft's validator: a candidate is judged by
driven poses and explicit geometry relations, not by a one-shot numeric guess.

## Agent Action Boundary

The agent receives the SDK baseline and may only:

- keep the SDK/VLM axis;
- micro-rotate the current axis by at most `5` degrees per iteration;
- change `lower` once;
- change `upper` once.

For prismatic limits the allowed deltas are `+/-10mm`, `+/-5mm`, `+/-2.5mm`,
`+/-1mm`, and `+/-0.5mm`. Values in `LimitEstimate` are meters.

The agent must not edit SDK files, docs, tests, or utility code during a run.
