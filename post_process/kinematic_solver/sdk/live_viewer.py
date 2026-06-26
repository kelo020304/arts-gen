"""Live event stream for post_process kinematic agent runs."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .candidate_file import extract_editable_code
from .schemas import CandidateReport, EstimateContext


EVENTS_FILENAME = "agent_events.jsonl"


@dataclass(frozen=True)
class LiveViewerServer:
    """Small local post_process Flask server for live browser polling."""

    url: str
    _httpd: Any
    _thread: threading.Thread

    def stop(self) -> None:
        shutdown = getattr(self._httpd, "shutdown", None)
        if callable(shutdown):
            shutdown()
        server_close = getattr(self._httpd, "server_close", None)
        if callable(server_close):
            server_close()
        self._thread.join(timeout=2.0)


class LiveViewer:
    """Append-only JSONL event stream consumed by the post_process viewer."""

    def __init__(self, out_dir: Path, *, run_id: str | None = None):
        self.out_dir = Path(out_dir)
        self.events_path = self.out_dir / EVENTS_FILENAME
        self.run_id = run_id or uuid.uuid4().hex[:12]

    def prepare(self, *, reset_events: bool = True) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if reset_events:
            self.events_path.write_text("")
        state_path = self.out_dir / "frontend_state.json"
        if reset_events:
            state_path.write_text(
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "latest_iteration": 0,
                        "latest_preview": None,
                        "iterations": [],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def emit(
        self,
        event: str,
        *,
        iteration: int,
        status: str | None = None,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
        estimates: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "iteration": int(iteration),
            "run_id": self.run_id,
        }
        if status is not None:
            payload["status"] = status
        if phase is not None:
            payload["phase"] = phase
        if details:
            payload["details"] = details
        if estimates is not None:
            payload["estimates"] = estimates
        if errors is not None:
            payload["errors"] = errors
        if warnings is not None:
            payload["warnings"] = warnings
        if notes is not None:
            payload["notes"] = notes
        with self.events_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(payload, sort_keys=True) + "\n")

    def emit_report(
        self,
        event: str,
        report: CandidateReport,
        *,
        iteration: int,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        combined_details = dict(details or {})
        combined_details.update(report.details)
        self.emit(
            event,
            iteration=iteration,
            status="passed" if report.passed else "failed",
            phase=phase,
            details=combined_details or None,
            estimates=[asdict(estimate) for estimate in report.estimates],
            errors=list(report.errors),
            warnings=list(report.warnings),
        )


def editable_region_sha256(candidate_path: Path) -> str:
    code = extract_editable_code(Path(candidate_path).read_text())
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def context_details(ctx: EstimateContext) -> dict[str, Any]:
    return {
        "object_id": ctx.object_id,
        "joint_count": len(ctx.joints),
        "joints": sorted(ctx.joints),
    }


def start_live_viewer_server(
    out_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    run_id: str | None = None,
) -> LiveViewerServer:
    try:
        from post_process.object_post_process.server import create_app
    except ModuleNotFoundError:
        from object_post_process.server import create_app
    from werkzeug.serving import make_server

    post_process_root = Path(__file__).resolve().parents[2]
    assets_root = post_process_root / "assets" / "object_assets"
    resolved_run_id = run_id or Path(out_dir).name
    app = create_app(
        assets_root=assets_root,
        kinematic_run_dir=Path(out_dir),
        kinematic_run_id=resolved_run_id,
    )
    httpd = make_server(host, port, app, threaded=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    actual_host, actual_port = httpd.server_address[:2]
    url_host = host if host != "0.0.0.0" else actual_host
    return LiveViewerServer(
        url=f"http://{url_host}:{actual_port}/kinematic-agent/{resolved_run_id}",
        _httpd=httpd,
        _thread=thread,
    )
