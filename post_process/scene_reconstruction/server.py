"""Flask server for the standalone 3DGS scene viewer."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, send_file, send_from_directory

_SCENE_LIST_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scene List</title>
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #06111d;
      --bg-1: #0d1726;
      --panel: rgba(16, 24, 38, 0.9);
      --panel-border: rgba(102, 160, 222, 0.18);
      --text: #edf5ff;
      --muted: #8ea3ba;
      --accent: #6cd4ff;
      --accent-strong: #9be7ff;
      --danger: #ff8f8f;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(61, 133, 198, 0.16), transparent 32rem),
        linear-gradient(180deg, var(--bg-0), var(--bg-1));
    }

    main {
      width: min(960px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 48px 0 64px;
    }

    .hero {
      margin-bottom: 24px;
    }

    .eyebrow {
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }

    h1 {
      margin: 0;
      font-size: clamp(32px, 6vw, 48px);
      line-height: 1.05;
    }

    .subtitle {
      margin: 12px 0 0;
      max-width: 640px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }

    .scene-grid {
      display: grid;
      gap: 12px;
    }

    .scene-card {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px;
      border-radius: 18px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      backdrop-filter: blur(14px);
      text-decoration: none;
      color: inherit;
      transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
    }

    a.scene-card:hover {
      transform: translateY(-1px);
      border-color: rgba(108, 212, 255, 0.4);
      background: rgba(18, 28, 44, 0.96);
    }

    .scene-card.disabled {
      cursor: not-allowed;
      opacity: 0.7;
    }

    .scene-name {
      font-size: 18px;
      font-weight: 600;
      line-height: 1.3;
    }

    .scene-meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }

    .badge {
      flex-shrink: 0;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(108, 212, 255, 0.28);
      color: var(--accent-strong);
      background: rgba(11, 38, 58, 0.55);
    }

    .badge.missing {
      border-color: rgba(255, 143, 143, 0.22);
      color: var(--danger);
      background: rgba(58, 18, 18, 0.45);
    }

    .empty-state {
      padding: 28px;
      border-radius: 18px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      color: var(--muted);
      line-height: 1.7;
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">3DGS Viewer</p>
      <h1>Scene List</h1>
      <p class="subtitle">
        Browse standalone Gaussian splat scenes found under <code>assets/scene_assets</code>.
      </p>
    </section>

    {% if scenes %}
      <section class="scene-grid">
        {% for scene in scenes %}
          {% if scene.has_ply %}
            <a class="scene-card" href="{{ url_for('scene_viewer', scene_name=scene.name) }}">
              <div>
                <div class="scene-name">{{ scene.name }}</div>
                <div class="scene-meta">3DGS PLY available</div>
              </div>
              <span class="badge">Open</span>
            </a>
          {% else %}
            <div class="scene-card disabled">
              <div>
                <div class="scene-name">{{ scene.name }}</div>
                <div class="scene-meta">PLY file missing</div>
              </div>
              <span class="badge missing">Missing</span>
            </div>
          {% endif %}
        {% endfor %}
      </section>
    {% else %}
      <section class="empty-state">
        No scene directories were found under <code>assets/scene_assets</code>.
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


def create_app(scene_assets_root: Path) -> Flask:
    """Create and return the standalone scene viewer Flask application."""
    app = Flask(__name__)

    scene_assets_root = scene_assets_root.resolve()
    gen_obj_root = scene_assets_root.parent.parent
    frontend_dir = gen_obj_root / "utils" / "frontend"
    supersplat_dir = frontend_dir / "supersplat-viewer"

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "ready": True}), 200

    @app.get("/")
    def index():
        return redirect("/scenes/")

    @app.get("/scenes/")
    def scene_list():
        return render_template_string(_SCENE_LIST_TEMPLATE, scenes=_list_scenes(scene_assets_root))

    @app.get("/scenes/<scene_name>")
    def scene_viewer(scene_name: str):
        scene_dir = _resolve_scene_dir(scene_assets_root, scene_name)
        if scene_dir is None or _preferred_ply_name(scene_dir) is None:
            abort(404, description=f"Scene not found or missing PLY: {scene_name}")
        return send_file(frontend_dir / "scene_viewer.html")

    @app.get("/api/scenes")
    def api_scenes():
        return jsonify(_list_scenes(scene_assets_root))

    @app.get("/scenes/<scene_name>/ply")
    def scene_ply(scene_name: str):
        scene_dir = _resolve_scene_dir(scene_assets_root, scene_name)
        if scene_dir is None:
            abort(404, description=f"Scene not found: {scene_name}")

        ply_name = _preferred_ply_name(scene_dir)
        if ply_name is None:
            abort(404, description=f"No supported PLY file found for scene: {scene_name}")

        return send_from_directory(str(scene_dir), ply_name, conditional=True)

    @app.get("/static/supersplat/<path:filename>")
    def supersplat_static(filename: str):
        return send_from_directory(str(supersplat_dir), filename)

    return app


def _list_scenes(scene_assets_root: Path) -> list[dict[str, bool | str]]:
    scenes: list[dict[str, bool | str]] = []
    if not scene_assets_root.is_dir():
        return scenes

    for child in sorted(scene_assets_root.iterdir()):
        if not child.is_dir():
            continue
        scenes.append({"name": child.name, "has_ply": _preferred_ply_name(child) is not None})
    return scenes


def _preferred_ply_name(scene_dir: Path) -> str | None:
    for candidate in ("3dgs_standard.ply", "3dgs_compressed.ply"):
        if (scene_dir / candidate).is_file():
            return candidate
    return None


def _resolve_scene_dir(scene_assets_root: Path, scene_name: str) -> Path | None:
    scene_dir = (scene_assets_root / scene_name).resolve()
    try:
        scene_dir.relative_to(scene_assets_root)
    except ValueError:
        return None
    return scene_dir if scene_dir.is_dir() else None
