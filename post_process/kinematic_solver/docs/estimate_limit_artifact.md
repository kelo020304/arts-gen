# estimate_limit.py Artifact

`post_process/kinematic_solver/estimate_limit.py` is both:

- the agent-maintained estimator artifact, inside `USER_CODE_START/END`
- the local harness entrypoint, outside that editable region

The agent may edit only:

```python
# >>> USER_CODE_START
def estimate_limits(ctx):
    ...
# >>> USER_CODE_END
```

The stable `sdk/` code compiles this file, validates the returned
`LimitEstimate` objects, and writes `predictions.jsonl`.

`LimitEstimate` may include an explicit action axis:

```python
LimitEstimate(
    joint_name="part_02",
    lower=0.0,
    upper=0.16,
    axis_world=[0.0, -1.0, 0.0],
    axis_label="-Y",
)
```

Use this when the default oracle `axis_world` plays the right range in the
wrong visible direction. The axis is no longer limited to signed cardinal
`X/Y/Z` actions: it can be any normalized 3D vector. In API mode, one iteration
may rotate each joint's current axis by at most 5 degrees.

When `--initial-joints-json` is provided, the harness stores the rough VLM
starting point in `ctx.evidence[joint]["initial_estimate"]`. Prismatic JSON
limits are millimeters and revolute JSON limits are degrees; both are converted
to meters/radians before the agent sees them. Missing limits become a zero range
so the agent expands them through bounded actions.

When `--agent-loop` is enabled, the API loop still edits only this region. The
stable harness records each attempt in `agent_events.jsonl` and writes
`frontend_state.json` plus run-scoped MJCF assets. The existing `post_process`
MJCF/Three.js frontend at `/kinematic-agent/<run_id>` shows the live iteration,
API/compile/validation status, latest estimates, code diff, and automatically
plays the full lower-to-upper-to-lower range for each non-final iteration.
