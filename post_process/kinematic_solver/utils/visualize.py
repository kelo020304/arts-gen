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


def _render_hulls_payload(backend: CollisionBackend) -> list[dict]:
    iter_hulls = getattr(backend, "iter_render_hulls", None)
    if iter_hulls is None:
        return []
    hulls = []
    for hull in iter_hulls():
        hulls.append({
            "part_name": hull.part_name,
            "vertices": np.asarray(hull.vertices, dtype=float).tolist(),
            "faces": np.asarray(hull.faces, dtype=int).tolist(),
            "rotation": np.asarray(hull.rotation, dtype=float).tolist(),
            "translation": np.asarray(hull.translation, dtype=float).tolist(),
        })
    return hulls


def _trace_steps(
    *,
    joint: dict,
    samples: list[dict],
    side: str,
    viz_stride: int,
) -> list[dict]:
    steps = []
    for idx, sample in enumerate(samples):
        if idx % max(viz_stride, 1) != 0 and idx != len(samples) - 1:
            continue
        q = sample.get("q")
        if q is None:
            continue
        direction = 1 if float(q) >= 0 else -1
        rotation, translation = apply_joint_transform_world_baked(
            joint_type=joint["type"],
            direction=direction,
            q_abs=abs(float(q)),
            axis_world=np.asarray(joint["axis_world"], dtype=np.float64),
            origin_world=np.asarray(joint["origin_world"], dtype=np.float64),
        )
        steps.append({
            "side": side,
            "sample_index": idx,
            "q": float(q),
            "valid": bool(sample.get("valid")),
            "frame": None,
            "moving_parts": list(joint["moving_parts"]),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        })
    return steps


def _final_range_steps(
    *,
    joint: dict,
    prediction: dict,
    trace: dict,
    viz_stride: int,
) -> list[dict]:
    lower = prediction.get("predicted_lower")
    upper = prediction.get("predicted_upper")
    has_ok_final_range = (
        lower is not None
        and upper is not None
        and (
            prediction.get("status") == "ok"
            or (
                prediction.get("status_lower") == "ok"
                and prediction.get("status_upper") == "ok"
            )
        )
    )
    if not has_ok_final_range:
        return (
            _trace_steps(
                joint=joint,
                samples=trace.get("trace_upper", []),
                side="upper",
                viz_stride=viz_stride,
            )
            + _trace_steps(
                joint=joint,
                samples=trace.get("trace_lower", []),
                side="lower",
                viz_stride=viz_stride,
            )
        )

    lower_f = float(lower)
    upper_f = float(upper)
    samples_by_q: dict[float, dict] = {}
    for sample in trace.get("trace_upper", []) + trace.get("trace_lower", []):
        q = sample.get("q")
        if q is None or not sample.get("valid"):
            continue
        q_f = float(q)
        if lower_f - 1e-12 <= q_f <= upper_f + 1e-12:
            samples_by_q[q_f] = {"q": q_f, "valid": True}

    samples = [
        {"q": q, "valid": samples_by_q[q]["valid"]}
        for q in sorted(samples_by_q)
    ]
    return _trace_steps(
        joint=joint,
        samples=samples,
        side="final_range",
        viz_stride=viz_stride,
    )


def _render_step_frames(
    *,
    backend: CollisionBackend,
    joint: dict,
    steps: list[dict],
    out_dir: Path,
) -> list[dict]:
    rendered = []
    for step in steps:
        frame_path = out_dir / f"frame_{step['side']}_{step['sample_index']:03d}.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=float(step["q"]),
            out_path=frame_path,
        )
        rendered.append({**step, "frame": frame_path.name})
    return rendered


def _raw_scan_evidence(*, prediction: dict, trace: dict) -> dict:
    return {
        "final_range": {
            "lower": prediction.get("predicted_lower"),
            "upper": prediction.get("predicted_upper"),
        },
        "direction_prior": prediction.get("motion_direction_prior"),
        "directions": {
            "upper": _direction_evidence(trace.get("trace_upper", [])),
            "lower": _direction_evidence(trace.get("trace_lower", [])),
        },
    }


def _direction_evidence(samples: list[dict]) -> dict:
    valid_qs = [
        float(sample["q"])
        for sample in samples
        if sample.get("q") is not None and bool(sample.get("valid"))
    ]
    invalid_qs = [
        float(sample["q"])
        for sample in samples
        if sample.get("q") is not None and not bool(sample.get("valid"))
    ]
    return {
        "n_samples": len(samples),
        "last_valid_q": valid_qs[-1] if valid_qs else None,
        "first_invalid_q": invalid_qs[0] if invalid_qs else None,
    }


def write_step_viewer(
    *,
    out_dir: Path,
    object_id: str,
    joint_name: str,
    prediction: dict,
    steps: list[dict],
    render_hulls: list[dict] | None = None,
    raw_scan_evidence: dict | None = None,
) -> None:
    payload = {
        "object_id": object_id,
        "joint_name": joint_name,
        "prediction": prediction,
        "render_hulls": render_hulls or [],
        "raw_scan_evidence": raw_scan_evidence or {},
        "steps": [
            {**step, "turn": idx + 1, "total_turns": len(steps)}
            for idx, step in enumerate(steps)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_manifest.json").write_text(json.dumps(payload, indent=2))
    payload_json = json.dumps(payload)
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
    #frame, #canvas {{
      width: 100%;
      height: 465px;
      background: #edf4f7;
      border: 1px solid var(--line);
    }}
    #frame {{ object-fit: contain; display: none; }}
    #canvas {{ display: block; }}
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
        <div class="sub">{escape(title)} &middot; final range samples rendered locally</div>
      </div>
      <div class="sub">Use left/right arrow keys or the slider</div>
    </header>
    <section class="stage">
      <aside class="card left">
        <div class="turn" id="turn">TURN 1 / {len(steps)}</div>
        <div class="action" id="action">Loading</div>
        <div class="pillrow">
          <div class="pill">RANGE<strong id="side">-</strong></div>
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
        <canvas id="canvas"></canvas>
        <img id="frame" alt="Rendered solver trace frame">
        <div class="controls">
          <button id="prev" aria-label="Previous step">&lsaquo;</button>
          <input id="slider" type="range" min="0" max="{max(len(steps) - 1, 0)}" value="0">
          <button id="next" aria-label="Next step">&rsaquo;</button>
        </div>
      </section>
    </section>
  </main>
  <script src="../../../vendor/three.min.js"></script>
  <script>
    const payload = {payload_json};
    const steps = payload.steps || [];
    const render_hulls = payload.render_hulls || [];
    let current = 0;
    const el = (id) => document.getElementById(id);
    let renderer = null;
    let scene = null;
    let camera = null;
    let groups = {{}};
    let baseMatrices = {{}};

    function mat3ToMat4(rotation, translation) {{
      const m = new THREE.Matrix4();
      m.set(
        rotation[0][0], rotation[0][1], rotation[0][2], translation[0],
        rotation[1][0], rotation[1][1], rotation[1][2], translation[1],
        rotation[2][0], rotation[2][1], rotation[2][2], translation[2],
        0, 0, 0, 1
      );
      return m;
    }}

    function initThree() {{
      if (!render_hulls.length || typeof THREE === 'undefined') return false;
      el('frame').style.display = 'none';
      el('canvas').style.display = 'block';
      renderer = new THREE.WebGLRenderer({{ canvas: el('canvas'), antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0xedf4f7);
      camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
      camera.up.set(0, 0, 1);
      scene.add(new THREE.AmbientLight(0xffffff, 0.7));
      const light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(2, -3, 4);
      scene.add(light);

      const palette = [0x4C78A8, 0xF58518, 0x54A24B, 0xE45756, 0x72B7B2, 0xB279A2, 0xFF9DA6, 0x9D755D];
      const partNames = [];
      for (const hull of render_hulls) {{
        if (!groups[hull.part_name]) {{
          const group = new THREE.Group();
          group.matrixAutoUpdate = false;
          groups[hull.part_name] = group;
          partNames.push(hull.part_name);
          scene.add(group);
        }}
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(hull.vertices.flat(), 3));
        geometry.setIndex(hull.faces.flat());
        geometry.computeVertexNormals();
        const color = palette[partNames.indexOf(hull.part_name) % palette.length];
        const material = new THREE.MeshLambertMaterial({{
          color,
          transparent: true,
          opacity: 0.68,
          side: THREE.DoubleSide,
        }});
        const mesh = new THREE.Mesh(geometry, material);
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(mat3ToMat4(hull.rotation, hull.translation));
        groups[hull.part_name].add(mesh);
        baseMatrices[hull.part_name] = new THREE.Matrix4();
      }}

      const box = new THREE.Box3().setFromObject(scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.01);
      camera.position.set(center.x + radius * 1.4, center.y - radius * 2.4, center.z + radius * 1.2);
      camera.lookAt(center);
      resizeThree();
      window.addEventListener('resize', resizeThree);
      return true;
    }}

    function resizeThree() {{
      if (!renderer || !camera) return;
      const canvas = el('canvas');
      const box = canvas.getBoundingClientRect();
      renderer.setSize(box.width, box.height, false);
      camera.aspect = Math.max(box.width, 1) / Math.max(box.height, 1);
      camera.updateProjectionMatrix();
    }}

    function renderThreeStep(s) {{
      if (!renderer || !scene || !camera) return false;
      for (const [part, group] of Object.entries(groups)) {{
        group.matrix.identity();
        group.matrixAutoUpdate = false;
      }}
      const motion = mat3ToMat4(s.rotation, s.translation);
      for (const part of s.moving_parts || []) {{
        if (groups[part]) {{
          groups[part].matrix.copy(motion);
        }}
      }}
      renderer.render(scene, camera);
      return true;
    }}

    const hasThree = initThree();
    function show(i) {{
      if (!steps.length) return;
      current = Math.max(0, Math.min(i, steps.length - 1));
      const s = steps[current];
      el('turn').textContent = `TURN ${{s.turn}} / ${{s.total_turns}}`;
      el('action').textContent = `${{s.side.toUpperCase()}} sample ${{s.sample_index}}`;
      el('side').textContent = s.side;
      el('q').textContent = Number(s.q).toFixed(4);
      el('valid').textContent = s.valid ? 'OK' : 'BLOCK';
      el('valid').className = s.valid ? 'status-ok' : 'status-bad';
      el('log').innerHTML = [
        `q = ${{Number(s.q).toFixed(6)}}`,
        `sample index = ${{s.sample_index}}`,
        `final lower/upper = ${{payload.raw_scan_evidence?.final_range?.lower}} / ${{payload.raw_scan_evidence?.final_range?.upper}}`,
        `collision/sanity = ${{s.valid ? 'valid pose' : 'invalid pose'}}`,
        `frame = ${{s.frame || 'live geometry'}}`
      ].join('<br>');
      if (s.frame) {{
        el('canvas').style.display = 'none';
        el('frame').style.display = 'block';
        el('frame').src = s.frame;
      }} else if (hasThree) {{
        el('frame').style.display = 'none';
        el('canvas').style.display = 'block';
        renderThreeStep(s);
      }}
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
    render_frames: bool = False,
) -> None:
    """Write solver-side visualization artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    upper_gif = upper_final = lower_gif = lower_final = None
    steps: list[dict] = []
    render_hulls = _render_hulls_payload(backend)

    upper_frames = _final_range_steps(
        joint=joint,
        prediction=prediction,
        trace=trace,
        viz_stride=viz_stride,
    )
    lower_frames: list[dict] = []
    if render_frames:
        upper_frames = _render_step_frames(
            backend=backend,
            joint=joint,
            steps=upper_frames,
            out_dir=out_dir,
        )
    steps.extend(upper_frames)
    steps.extend(lower_frames)

    if render_frames and prediction.get("status_upper") == "ok":
        upper_final = out_dir / "final_predicted_upper.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=prediction.get("predicted_upper"),
            out_path=upper_final,
        )
        upper_gif = out_dir / "gif_upper.gif"
        _write_gif(upper_gif, [out_dir / frame["frame"] for frame in upper_frames])

    if render_frames and prediction.get("status_lower") == "ok":
        lower_final = out_dir / "final_predicted_lower.png"
        render_final_predicted(
            backend=backend,
            joint=joint,
            q_signed=prediction.get("predicted_lower"),
            out_path=lower_final,
        )
        lower_gif = out_dir / "gif_lower.gif"
        _write_gif(lower_gif, [lower_final])

    write_visualization_manifest(
        manifest_path=out_dir / "visualization_manifest.json",
        object_id=prediction["object_id"],
        joint_name=prediction["joint_name"],
        upper_status=(
            "ok" if render_frames and prediction.get("status_upper") == "ok"
            else "live_geometry"
        ),
        upper_gif=upper_gif,
        upper_final=upper_final,
        lower_status=(
            "ok" if render_frames and prediction.get("status_lower") == "ok"
            else "live_geometry"
        ),
        lower_gif=lower_gif,
        lower_final=lower_final,
    )
    write_step_viewer(
        out_dir=out_dir,
        object_id=prediction["object_id"],
        joint_name=prediction["joint_name"],
        prediction=prediction,
        steps=steps,
        render_hulls=render_hulls,
        raw_scan_evidence=_raw_scan_evidence(prediction=prediction, trace=trace),
    )
