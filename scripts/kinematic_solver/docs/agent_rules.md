# Agent Rules

- Edit only the editable region between `# >>> USER_CODE_START` and
  `# >>> USER_CODE_END` in `post_process/kinematic_solver/estimate_limit.py`.
- Define `estimate_limits(ctx)`.
- Return `list[LimitEstimate]`, one per joint.
- Use `ctx.joints` and `ctx.evidence`; do not open source USD files or GT limit
  JSON.
- Positive prismatic motion follows `axis_world * q`. For closed-rest drawers,
  baskets, pans, and trays, prefer `lower=0.0` and positive `upper` when the
  evidence says positive q opens outward.
- Keep reasoning in `LimitEstimate.reason` so compile feedback is auditable.
- When the visible motion direction is wrong, set `LimitEstimate.axis_world` to
  one of the signed action axes (`+/-X`, `+/-Y`, `+/-Z`) and set
  `axis_label`. The harness visualization uses this axis override.
- For knobs/dials, do not blindly trust the authored USD axis. Prefer the top
  signed action in `ctx.evidence[joint]["axis_candidates"]`; validation rejects
  rotary-control axes that do not match the recommended geometry action.
- Use coarse-to-fine action search when possible: try signed x/y/z translation
  or rotation, scan outward with a large step, and halve the interval after the
  first invalid pose until the requested resolution.
- During API runs, the harness may call the model for multiple iterations, but
  each iteration may only replace the editable region in `estimate_limit.py`.
  It must not edit `sdk/`, `docs/`, tests, or the harness wrapper.
