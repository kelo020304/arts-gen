"""IsaacLab Python discovery and app launch helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

_DISCOVERY_LOCK = threading.Lock()
_CACHED_PYTHON: Path | None = None
_CACHED_ERROR: RuntimeError | None = None


def find_isaaclab_python() -> Path:
    """Return the cached absolute path to ``env_isaaclab/bin/python``."""
    global _CACHED_ERROR, _CACHED_PYTHON

    if _CACHED_PYTHON is not None:
        return _CACHED_PYTHON
    if _CACHED_ERROR is not None:
        raise RuntimeError(str(_CACHED_ERROR))

    with _DISCOVERY_LOCK:
        if _CACHED_PYTHON is not None:
            return _CACHED_PYTHON
        if _CACHED_ERROR is not None:
            raise RuntimeError(str(_CACHED_ERROR))

        try:
            _CACHED_PYTHON = _discover_isaaclab_python()
        except RuntimeError as exc:
            _CACHED_ERROR = RuntimeError(str(exc))
            raise RuntimeError(str(exc)) from exc

        return _CACHED_PYTHON


def launch_isaaclab_app(*, headless: bool):
    """Launch Isaac Lab's ``AppLauncher`` with the Omniverse app enabled."""
    previous_launch_ov_app = os.environ.get("LAUNCH_OV_APP")
    os.environ["LAUNCH_OV_APP"] = "1"
    try:
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise RuntimeError("Failed to import Isaac Lab AppLauncher.") from exc

        try:
            return AppLauncher({"headless": headless})
        except Exception as exc:
            raise RuntimeError(f"Failed to launch Isaac Lab AppLauncher with headless={headless}.") from exc
    finally:
        if previous_launch_ov_app is None:
            os.environ.pop("LAUNCH_OV_APP", None)
        else:
            os.environ["LAUNCH_OV_APP"] = previous_launch_ov_app


def _discover_isaaclab_python() -> Path:
    conda_executable = _find_conda_executable()
    env_prefix = _discover_well_known_env_prefix(conda_executable, "env_isaaclab")
    if env_prefix is None:
        env_prefix = _find_conda_env_prefix(conda_executable, "env_isaaclab")
    python_path = env_prefix / "bin" / "python"
    if not python_path.is_file():
        raise RuntimeError(f"Discovered `env_isaaclab`, but `{python_path}` does not exist.")
    if not os.access(python_path, os.X_OK):
        raise RuntimeError(f"Discovered `env_isaaclab`, but `{python_path}` is not executable.")
    return python_path


def _find_conda_executable() -> Path:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidate = Path(conda_exe).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise RuntimeError(f"CONDA_EXE does not point to a file: {candidate}")

    which_conda = shutil.which("conda")
    if which_conda:
        return Path(which_conda).resolve()

    home_conda = Path.home() / "anaconda3" / "bin" / "conda"
    if home_conda.is_file():
        return home_conda.resolve()

    raise RuntimeError("Unable to locate the `conda` executable required to discover IsaacLab.")


def _find_conda_env_prefix(conda_executable: Path, env_name: str) -> Path:
    completed = _run_command([os.fspath(conda_executable), "--no-plugins", "env", "list"])
    output = f"{completed.stdout}{completed.stderr}"

    if completed.returncode != 0:
        raise RuntimeError(
            f"`conda env list` failed while discovering IsaacLab: {output.strip() or '<no output>'}"
        )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        columns = line.split()
        if columns[0] != env_name:
            continue

        env_prefix = Path(columns[-1]).expanduser()
        if not env_prefix.is_dir():
            raise RuntimeError(
                f"`conda env list` reported env `{env_name}` at a missing path: {env_prefix}"
            )
        return env_prefix.resolve()

    raise RuntimeError(f"`conda env list` did not report the `{env_name}` environment.")


def _discover_well_known_env_prefix(conda_executable: Path, env_name: str) -> Path | None:
    candidate_prefixes = []

    conda_root = conda_executable.resolve().parent.parent
    candidate_prefixes.append(conda_root / "envs" / env_name)
    candidate_prefixes.append(Path.home() / "anaconda3" / "envs" / env_name)
    candidate_prefixes.append(Path.home() / "miniconda3" / "envs" / env_name)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        current_prefix = Path(conda_prefix).expanduser()
        if current_prefix.name == env_name:
            candidate_prefixes.append(current_prefix)

    seen: set[Path] = set()
    for candidate in candidate_prefixes:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if resolved_candidate.is_dir():
            return resolved_candidate

    return None


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={
                **os.environ,
                "CONDA_NO_PLUGINS": "true",
            },
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to execute `{command[0]}` while discovering IsaacLab: {exc}") from exc
