#!/usr/bin/env python3
"""Unified static web preview entrypoint with VLM/PC switch tabs."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402


PIPELINE_ROOT = Path(__file__).resolve().parent


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.lower().replace(" ", "_")


def default_vlm_jsonl(config) -> Path:
    return (
        Path(config.vlm_dir)
        / "training_json"
        / f"arts_mllm_{dataset_slug(config.dataset_name)}_part_complete_8view_1img.jsonl"
    )


def default_pc_jsonl(config) -> Path:
    return Path(config.data_root) / "manifests" / "part_completion" / f"arts_pc_{dataset_slug(config.dataset_name)}_train.jsonl"


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _run(cmd: list[str]) -> None:
    print("[preview] command: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_vlm_preview(args: argparse.Namespace, config, output_root: Path) -> dict[str, Any]:
    jsonl = Path(args.vlm_jsonl) if args.vlm_jsonl else default_vlm_jsonl(config)
    out_dir = output_root / "vlm_training"
    cmd = [
        sys.executable,
        str(PIPELINE_ROOT / "11_web_preview_vlm_dataset.py"),
        "--config",
        args.config,
        "--jsonl",
        str(jsonl),
        "--output-dir",
        str(out_dir),
    ]
    if args.object_ids:
        cmd.extend(["--object-ids", args.object_ids])
    _run(cmd)
    return {
        "mode": "vlm",
        "title": "VLM training preview",
        "jsonl": str(jsonl),
        "sample_count": _line_count(jsonl),
        "output_dir": str(out_dir),
        "index": str(out_dir / "index.html"),
        "relative_index": "vlm_training/index.html",
    }


def build_pc_preview(args: argparse.Namespace, config, output_root: Path) -> dict[str, Any]:
    jsonl = Path(args.pc_jsonl) if args.pc_jsonl else default_pc_jsonl(config)
    out_dir = output_root / "part_completion"
    cmd = [
        sys.executable,
        str(PIPELINE_ROOT / "11_web_preview_part_completion.py"),
        "--config",
        args.config,
        "--manifest",
        str(jsonl),
        "--output-dir",
        str(out_dir),
    ]
    if args.object_ids:
        cmd.extend(["--object-ids", args.object_ids])
    if args.angle_ids:
        cmd.extend(["--angle-ids", args.angle_ids])
    if args.view_ids:
        cmd.extend(["--view-ids", args.view_ids])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    _run(cmd)
    return {
        "mode": "pc",
        "title": "Part completion preview",
        "jsonl": str(jsonl),
        "sample_count": _line_count(jsonl),
        "output_dir": str(out_dir),
        "index": str(out_dir / "index.html"),
        "relative_index": "part_completion/index.html",
    }


def write_switch_index(output_root: Path, built: list[dict[str, Any]], config, args: argparse.Namespace) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.html"
    manifest_path = output_root / "preview_manifest.json"
    manifest = {
        "schema_version": "v1-preview-switch",
        "dataset": config.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "object_ids": args.object_ids,
        "previews": built,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    default_mode = built[0]["mode"] if built else ""
    buttons = []
    panels = []
    cards = []
    for item in built:
        mode = html.escape(str(item["mode"]))
        title = html.escape(str(item["title"]))
        rel = html.escape(str(item["relative_index"]))
        jsonl = html.escape(str(item["jsonl"]))
        active = " active" if item["mode"] == default_mode else ""
        hidden = "" if item["mode"] == default_mode else " hidden"
        buttons.append(f'<button class="tab{active}" data-target="{mode}">{title}</button>')
        cards.append(
            f'<a class="card" href="{rel}"><strong>{title}</strong><span>{rel}</span><small>{jsonl}</small></a>'
        )
        panels.append(
            f'<section class="panel{active}" id="panel-{mode}"{hidden}>'
            f'<div class="panel-head"><h2>{title}</h2><a href="{rel}" target="_blank">Open in new tab</a></div>'
            f'<iframe src="{rel}" title="{title}"></iframe>'
            f'</section>'
        )

    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config.dataset_name)} preview switch</title>
<style>
:root {{ color-scheme: light; --bg:#f6f7fb; --panel:#fff; --ink:#18202f; --muted:#687386; --accent:#315bff; --border:#dce1ea; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:18px 22px 12px; border-bottom:1px solid var(--border); background:linear-gradient(180deg,#fff,#f9fbff); position:sticky; top:0; z-index:5; }}
h1 {{ margin:0 0 6px; font-size:22px; }}
.meta {{ color:var(--muted); font-size:13px; }}
.tabs {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
.tab {{ border:1px solid var(--border); background:#fff; color:var(--ink); border-radius:999px; padding:9px 14px; cursor:pointer; font-weight:700; }}
.tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); box-shadow:0 6px 16px rgba(49,91,255,.22); }}
main {{ padding:16px 18px 22px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; margin-bottom:14px; }}
.card {{ display:flex; flex-direction:column; gap:4px; text-decoration:none; color:inherit; background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }}
.card span {{ color:var(--accent); font-size:13px; }}
.card small {{ color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:16px; overflow:hidden; box-shadow:0 10px 24px rgba(23,31,48,.08); }}
.panel[hidden] {{ display:none; }}
.panel-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:10px 14px; border-bottom:1px solid var(--border); background:#fff; }}
.panel-head h2 {{ margin:0; font-size:16px; }}
.panel-head a {{ color:var(--accent); font-weight:700; text-decoration:none; }}
iframe {{ display:block; width:100%; height:calc(100vh - 220px); min-height:620px; border:0; background:#fff; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(config.dataset_name)} unified preview</h1>
  <div class="meta">Generated {html.escape(manifest['created_at'])}; data root: {html.escape(config.data_root)}</div>
  <nav class="tabs">{''.join(buttons)}</nav>
</header>
<main>
  <div class="cards">{''.join(cards)}</div>
  {''.join(panels)}
</main>
<script>
const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.panel')];
function activate(mode) {{
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.target === mode));
  panels.forEach(panel => {{
    const on = panel.id === `panel-${{mode}}`;
    panel.classList.toggle('active', on);
    panel.hidden = !on;
  }});
}}
tabs.forEach(tab => tab.addEventListener('click', () => activate(tab.dataset.target)));
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"[preview] switch index: {index_path}")
    print(f"[preview] manifest: {manifest_path}")
    return index_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VLM/PC static previews plus a switch page.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--mode", choices=("vlm", "pc", "both"), default="both")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset for both previews.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset for PC preview.")
    parser.add_argument("--view-ids", help="Optional comma-separated view subset for PC preview.")
    parser.add_argument("--max-samples", type=int, help="Optional sample cap for PC preview.")
    parser.add_argument("--vlm-jsonl", help="Override VLM JSONL path.")
    parser.add_argument("--pc-jsonl", help="Override part-completion JSONL path.")
    parser.add_argument("--output-dir", help="Output root. Default: <data_root>/preview")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_root = Path(args.output_dir) if args.output_dir else Path(config.preview_dir)

    built: list[dict[str, Any]] = []
    if args.mode in {"vlm", "both"}:
        built.append(build_vlm_preview(args, config, output_root))
    if args.mode in {"pc", "both"}:
        built.append(build_pc_preview(args, config, output_root))
    write_switch_index(output_root, built, config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
