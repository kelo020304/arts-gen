"""Solver-side visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from html import escape

import numpy as np

from .backend import CollisionBackend
from .manual_transform import apply_joint_transform_world_baked
from .render import render_backend_frame


def _apply_pose(*, backend: CollisionBackend, joint: dict, q_signed: float) -> None:
    direction = 1 if q_signed >= 0 else -1
    rotation, translation = apply_joint_transform_world_baked(
        joint_type=joint["type"],
        direction=direction,
        q_abs=abs(float(q_signed)),
        axis_world=np.asarray(joint["axis_world"], dtype=np.float64),
        origin_world=np.asarray(joint["origin_world"], dtype=np.float64),
    )
    for part in joint["moving_parts"]:
        backend.set_pose(part, rotation, translation)


def render_final_predicted(
    *,
    backend: CollisionBackend,
    joint: dict,
    q_signed: float | None,
    out_path: Path,
) -> None:
    """Render the final pose for one direction."""
    if q_signed is None:
        raise ValueError("q_signed is None; non-ok directions must not be rendered")
    _apply_pose(backend=backend, joint=joint, q_signed=float(q_signed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_backend_frame(backend, out_path)


def write_visualization_manifest(
    *,
    manifest_path: Path,
    object_id: str,
    joint_name: str,
    upper_status: str,
    upper_gif: Path | None,
    upper_final: Path | None,
    lower_status: str,
    lower_gif: Path | None,
    lower_final: Path | None,
) -> None:
    def side_payload(status: str, gif: Path | None, final: Path | None) -> dict:
        if status == "ok":
            assert gif is not None and final is not None
            return {"status": "ok", "gif": gif.name, "final_png": final.name}
        return {"status": status, "gif": None, "final_png": None}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "object_id": object_id,
        "joint_name": joint_name,
        "upper": side_payload(upper_status, upper_gif, upper_final),
        "lower": side_payload(lower_status, lower_gif, lower_final),
    }, indent=2))


def _write_gif(path: Path, frame_paths: list[Path]) -> None:
    if not frame_paths:
        raise RuntimeError(f"empty frames for ok-status GIF: {path}")
    from PIL import Image

    frames = [Image.open(frame_path).convert("P") for frame_path in frame_paths]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )


def _render_trace_frames(
    *,
    backend: CollisionBackend,
    joint: dict,
    samples: list[dict],
    out_dir: Path,
    side: str,
    viz_stride: int,
) -> list[dict]:
    frames: list[dict] = []
    for idx, sample in enumerate(samples):
        if idx % max(viz_stride, 1) != 0 and idx != len(samples) - 1:
            continue
        q = sample.get("q")
        if q is None:
            continue
        frame_path = out_dir / f"frame_{side}_{idx:03d}.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=float(q),
            out_path=frame_path,
        )
        frames.append({
            "side": side,
            "sample_index": idx,
            "q": float(q),
            "valid": bool(sample.get("valid")),
            "frame": frame_path.name,
        })
    return frames


def write_step_viewer(
    *,
    out_dir: Path,
    object_id: str,
    joint_name: str,
    prediction: dict,
    steps: list[dict],
) -> None:
    payload = {
        "object_id": object_id,
        "joint_name": joint_name,
        "prediction": prediction,
        "steps": [
            {**step, "turn": idx + 1, "total_turns": len(steps)}
            for idx, step in enumerate(steps)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_manifest.json").write_text(json.dumps(payload, indent=2))
    data_json = json.dumps(payload["steps"])
    title = f"{object_id} / {joint_name}"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KinematicSolver Step Viewer - {escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #62717f;
      --line: #d9e2dc;
      --ok: #16866f;
      --bad: #c84d42;
      --paper: #fbfcf8;
      --panel: rgba(255, 255, 255, 0.92);
      --accent: #13977f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(rgba(20, 30, 38, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20, 30, 38, 0.04) 1px, transparent 1px),
        var(--paper);
      background-size: 28px 28px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    main {{
      width: min(1180px, calc(100vw - 36px));
      margin: 38px auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .stage {{
      display: grid;
      grid-template-columns: minmax(300px, 0.75fr) minmax(420px, 1.25fr);
      gap: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.52);
      box-shadow: 0 20px 44px rgba(44, 58, 72, 0.12);
      padding: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 22px;
      min-height: 520px;
    }}
    .left {{
      border-left: 7px solid var(--accent);
      display: flex;
      flex-direction: column;
      gap: 22px;
    }}
    .turn {{ color: var(--muted); font-size: 18px; }}
    .action {{ font-size: 30px; line-height: 1.12; }}
    .pillrow {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      min-width: 92px;
      background: #f7faf7;
      color: var(--muted);
      font-size: 12px;
    }}
    .pill strong {{ display: block; color: var(--ink); font-size: 16px; margin-top: 4px; }}
    .status-ok {{ color: var(--ok); }}
    .status-bad {{ color: var(--bad); }}
    .log {{
      margin-top: auto;
      background: #f6faf6;
      border: 1px solid #e5eee7;
      min-height: 160px;
      padding: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.8;
    }}
    .viewer {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .viewerTop {{
      display: flex;
      justify-content: space-between;
      color: var(--accent);
      font-size: 14px;
    }}
    img {{
      width: 100%;
      height: 465px;
      object-fit: contain;
      background: #edf4f7;
      border: 1px solid var(--line);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 44px 1fr 44px;
      gap: 12px;
      align-items: center;
    }}
    button {{
      height: 40px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      font-size: 20px;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    @media (max-width: 900px) {{
      .stage {{ grid-template-columns: 1fr; }}
      .card {{ min-height: auto; }}
      img {{ height: 360px; }}
      header {{ align-items: start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>KinematicSolver Step Viewer</h1>
        <div class="sub">{escape(title)} &middot; trace samples rendered locally</div>
      </div>
      <div class="sub">Use left/right arrow keys or the slider</div>
    </header>
    <section class="stage">
      <aside class="card left">
        <div class="turn" id="turn">TURN 1 / {len(steps)}</div>
        <div class="action" id="action">Loading</div>
        <div class="pillrow">
          <div class="pill">SIDE<strong id="side">-</strong></div>
          <div class="pill">Q<strong id="q">-</strong></div>
          <div class="pill">VALID<strong id="valid">-</strong></div>
        </div>
        <div class="log" id="log"></div>
      </aside>
      <section class="card viewer">
        <div class="viewerTop">
          <span>VIEWER RENDER</span>
          <span>{escape(object_id)}</span>
        </div>
        <img id="frame" alt="Rendered solver trace frame">
        <div class="controls">
          <button id="prev" aria-label="Previous step">&lsaquo;</button>
          <input id="slider" type="range" min="0" max="{max(len(steps) - 1, 0)}" value="0">
          <button id="next" aria-label="Next step">&rsaquo;</button>
        </div>
      </section>
    </section>
  </main>
  <script>
    const steps = {data_json};
    let current = 0;
    const el = (id) => document.getElementById(id);
    function show(i) {{
      if (!steps.length) return;
      current = Math.max(0, Math.min(i, steps.length - 1));
      const s = steps[current];
      el('turn').textContent = `TURN ${{s.turn}} / ${{s.total_turns}}`;
      el('action').textContent = `${{s.side.toUpperCase()}} scan sample ${{s.sample_index}}`;
      el('side').textContent = s.side;
      el('q').textContent = Number(s.q).toFixed(4);
      el('valid').textContent = s.valid ? 'OK' : 'BLOCK';
      el('valid').className = s.valid ? 'status-ok' : 'status-bad';
      el('log').innerHTML = [
        `q = ${{Number(s.q).toFixed(6)}}`,
        `sample index = ${{s.sample_index}}`,
        `collision/sanity = ${{s.valid ? 'valid pose' : 'invalid pose'}}`,
        `frame = ${{s.frame}}`
      ].join('<br>');
      el('frame').src = s.frame;
      el('slider').value = current;
    }}
    el('prev').onclick = () => show(current - 1);
    el('next').onclick = () => show(current + 1);
    el('slider').oninput = (event) => show(Number(event.target.value));
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') show(current - 1);
      if (event.key === 'ArrowRight') show(current + 1);
    }});
    show(0);
  </script>
</body>
</html>
"""
    (out_dir / "step_viewer.html").write_text(html)


def visualize_one_joint(
    *,
    backend: CollisionBackend,
    joint: dict,
    prediction: dict,
    trace: dict,
    out_dir: Path,
    viz_stride: int = 5,
) -> None:
    """Write final PNGs, lightweight GIF placeholders, and a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    upper_gif = upper_final = lower_gif = lower_final = None
    steps: list[dict] = []

    upper_frames = _render_trace_frames(
        backend=backend,
        joint=joint,
        samples=trace.get("trace_upper", []),
        out_dir=out_dir,
        side="upper",
        viz_stride=viz_stride,
    )
    lower_frames = _render_trace_frames(
        backend=backend,
        joint=joint,
        samples=trace.get("trace_lower", []),
        out_dir=out_dir,
        side="lower",
        viz_stride=viz_stride,
    )
    steps.extend(upper_frames)
    steps.extend(lower_frames)

    if prediction.get("status_upper") == "ok":
        upper_final = out_dir / "final_predicted_upper.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=prediction.get("predicted_upper"),
            out_path=upper_final,
        )
        upper_gif = out_dir / "gif_upper.gif"
        _write_gif(upper_gif, [out_dir / frame["frame"] for frame in upper_frames])

    if prediction.get("status_lower") == "ok":
        lower_final = out_dir / "final_predicted_lower.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=prediction.get("predicted_lower"),
            out_path=lower_final,
        )
        lower_gif = out_dir / "gif_lower.gif"
        _write_gif(lower_gif, [out_dir / frame["frame"] for frame in lower_frames])

    write_visualization_manifest(
        manifest_path=out_dir / "visualization_manifest.json",
        object_id=prediction["object_id"],
        joint_name=prediction["joint_name"],
        upper_status="ok" if prediction.get("status_upper") == "ok" else "skipped_non_ok",
        upper_gif=upper_gif,
        upper_final=upper_final,
        lower_status="ok" if prediction.get("status_lower") == "ok" else "skipped_non_ok",
        lower_gif=lower_gif,
        lower_final=lower_final,
    )
    write_step_viewer(
        out_dir=out_dir,
        object_id=prediction["object_id"],
        joint_name=prediction["joint_name"],
        prediction=prediction,
        steps=steps,
    )
