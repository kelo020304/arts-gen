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

`LimitEstimate` may include an explicit signed action axis:

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
wrong visible direction. For semantic rotary controls such as knobs and dials,
the SDK ranks the geometry thin axis as the preferred signed action while still
keeping the authored USD axis as a candidate. A candidate that uses the wrong
knob/dial axis fails validation and the next agent turn must choose a different
axis from `ctx.evidence[joint]["axis_candidates"]`.

When `--agent-loop` is enabled, the API loop still edits only this region. The
stable harness records each attempt in `agent_events.jsonl` and writes
`frontend_state.json` plus run-scoped MJCF assets. The existing `post_process`
MJCF/Three.js frontend at `/kinematic-agent/<run_id>` shows the live iteration,
API/compile/validation status, latest estimates, code diff, and automatically
plays the full lower-to-upper-to-lower range for each iteration.
