# KinematicSolver Agent Harness

For the decoded-mesh, GT-free type/axis/origin/range optimizer and combined
MJCF/USD export, see `kin_agent_v2.md` and run
`python -m post_process.kinematic_solver.run_kin_agent`.  The legacy harness
below remains available for compatibility and limit-focused experiments.

This folder follows the Articraft-style split:

- `docs/` describes the task and rules.
- `sdk/` provides stable compile, context, and schema helpers.
- `utils/` contains solver, validation, comparison, USD, and data-prep internals
  that the agent must not edit.
- `estimate_limit.py` is the only file the agent may edit, and only inside its
  editable region.

Axis selection and validation are SDK responsibilities, not test-script logic.
See `docs/axis_validation.md` for the runtime contract. In short: axis
selection uses Articraft-style geometry relations and driven-pose validation,
not PCA as a source of truth. Wrong-face motion is rejected even when a short
segment looks collision-free.

The editable region in `estimate_limit.py` must define:

```python
def estimate_limits(ctx):
    return [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.15)]
```

If the candidate moves in the wrong visible direction, return an explicit
signed action axis:

```python
LimitEstimate(
    joint_name="part_02",
    lower=0.0,
    upper=0.16,
    axis_world=[0.0, -1.0, 0.0],
    axis_label="-Y",
)
```

Run local validation with:

```bash
/home/mi/anaconda3/envs/env-isaacsim/bin/python -m post_process.kinematic_solver.estimate_limit \
  --object-id ra_063 \
  --converter-output-root /tmp/ks063_work_fix.BffLua \
  --source-root data/RealAppliance \
  --out-dir /tmp/ks_estimate_limit
```

Run with the live dashboard:

```bash
/home/mi/anaconda3/envs/env-isaacsim/bin/python -m post_process.kinematic_solver.estimate_limit \
  --object-id ra_063 \
  --converter-output-root /tmp/ks063_work_fix.BffLua \
  --source-root data/RealAppliance \
  --out-dir /tmp/ks_estimate_limit_live \
  --live-viewer \
  --open-live-viewer \
  --live-hold-seconds 30
```

Run from a rough VLM initial JSON:

```json
{
  "object_id": "ra_063",
  "initial_joints": {
    "part_00": {"type": "revolute", "axis": [0, 0, 1], "limit": null, "parent": "body"},
    "part_01": {"type": "revolute", "axis": [1, 0, 0], "limit": [-360, 360], "parent": "body"},
    "part_02": {"type": "prismatic", "axis": [1, 0, 0], "limit": [0, 30], "parent": "body"}
  }
}
```

Prismatic limits are millimeters in the JSON and meters in `LimitEstimate`.
Revolute limits are degrees in the JSON and radians in `LimitEstimate`.
The live viewer writes iteration `0` as `vlm_initial`, so the first visible
state is the rough VLM axis/range before the agent edits `estimate_limit.py`.
For pull-out drawers, SDK validation now follows the Articraft-style sampled
pose check: after each signed-axis search it drives the endpoint and requires
the drawer to actually clear parent/static geometry. A side-axis trial that
still intersects the body is reported as `endpoint_overlap`, even if a short
partial motion looked collision-free.

```bash
/home/mi/anaconda3/envs/env-isaacsim/bin/python -m post_process.kinematic_solver.estimate_limit \
  --object-id ra_063 \
  --converter-output-root /tmp/ks063_work_fix.BffLua \
  --source-root data/RealAppliance \
  --initial-joints-json /tmp/ks063_vlm_initial.json \
  --out-dir /tmp/ks_estimate_limit_agent \
  --agent-loop \
  --max-agent-iterations 10 \
  --api-heartbeat-seconds 1 \
  --live-viewer \
  --open-live-viewer \
  --live-hold-seconds 300
```

The live viewer opens the existing `post_process` MJCF/Three.js frontend at
`/kinematic-agent/<run_id>`. It polls `agent_events.jsonl` plus
`frontend_state.json`, shows compact Articraft-style `failure` / `warning` /
`note` compile signals, and automatically plays each iteration's full estimated
joint range in the viewer. The progress bar tracks agent iteration progress, not
animation playback. No standalone `agent_live.html` is generated.

Run the controlled API loop with Articraft-compatible environment variables:

```bash
OPENROUTER_BASE_URL=https://api-router.evad.mioffice.cn/v1 \
ARTICRAFT_MODEL=gpt-5.5 \
ARTICRAFT_THINKING_LEVEL=high \
/home/mi/anaconda3/envs/env-isaacsim/bin/python -m post_process.kinematic_solver.estimate_limit \
  --object-id ra_063 \
  --converter-output-root /tmp/ks063_work_fix.BffLua \
  --source-root data/RealAppliance \
  --out-dir /tmp/ks_estimate_limit_agent \
  --agent-loop \
  --max-agent-iterations 3 \
  --api-heartbeat-seconds 2 \
  --live-viewer \
  --open-live-viewer \
  --live-hold-seconds 60
```

Set `OPENROUTER_API_KEY` in the shell before running the command. The harness
does not print API keys.
