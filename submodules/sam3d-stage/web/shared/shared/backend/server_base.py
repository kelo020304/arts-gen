from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _gpu_info() -> dict[str, Any]:
    """Return GPU/VRAM info. Best-effort: works without torch installed."""
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "vram_used_mb": None, "vram_total_mb": None}
    if not torch.cuda.is_available():
        return {"device": "cpu", "vram_used_mb": None, "vram_total_mb": None}
    free, total = torch.cuda.mem_get_info(0)
    used = total - free
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_used_mb": round(used / 1024**2, 1),
        "vram_total_mb": round(total / 1024**2, 1),
    }


def _validate_filename(name: str) -> None:
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(400, f"invalid filename: {name!r}")


def create_app(
    title: str,
    *,
    data_dir: Path,
    health_extras: Callable[[], dict] | None = None,
    cors_origins: list[str] | None = None,
    app_frontend_dir: Path | None = None,
) -> FastAPI:
    """Build a FastAPI app pre-wired with health, output serving, and statics.

    Mounts:
      - ``/static``     -> ``shared/frontend/`` (this package's assets)
      - ``/static_app`` -> ``app_frontend_dir`` (caller's app-specific assets)

    Routes:
      - GET /api/health
      - GET /api/jobs/{sha}/{filename}  (serves data_dir/<sha>/output/<filename>)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title=title)
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    shared_frontend = resources.files("shared").joinpath("frontend")
    with resources.as_file(shared_frontend) as frontend_path:
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_path)),
            name="static",
        )

    if app_frontend_dir is not None:
        app_frontend_dir = Path(app_frontend_dir)
        if app_frontend_dir.is_dir():
            app.mount(
                "/static_app",
                StaticFiles(directory=str(app_frontend_dir)),
                name="static_app",
            )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        info: dict[str, Any] = {"status": "ok", **_gpu_info()}
        if health_extras is not None:
            info.update(health_extras())
        return info

    @app.get("/api/jobs/{sha}/{filename}")
    def serve_job_file(sha: str, filename: str) -> FileResponse:
        _validate_filename(sha)
        _validate_filename(filename)
        path = data_dir / sha / "output" / filename
        if not path.is_file():
            raise HTTPException(404, f"missing: {sha}/{filename}")
        return FileResponse(path)

    return app
