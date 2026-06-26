# KinematicSolver Agent Harness

The implementation lives in `post_process/kinematic_solver`. This compatibility
folder follows the Articraft-style split:

- `docs/` describes the task and rules.
- `sdk/` provides stable compile, context, and schema helpers.
- Runtime internals live under `post_process/kinematic_solver/utils/`; they are
  not agent-editable.
- `estimate_limit.py` is the only file the agent may edit, and only inside its
  editable region.

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

The live viewer opens the existing `post_process` MJCF/Three.js frontend at
`/kinematic-agent/<run_id>`. It polls `agent_events.jsonl` plus
`frontend_state.json`, shows the current iteration and warning/note/error
signals, and automatically plays each iteration's full estimated joint range.

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
