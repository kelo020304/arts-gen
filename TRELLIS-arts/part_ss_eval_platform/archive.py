from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .metrics import (
    inspect_eval_report,
    inspect_test_export,
    load_eval_metrics,
    load_test_metrics,
)


def safe_artifact_path(experiment_root: Path, relative_path: str) -> Path:
    root = experiment_root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path outside experiment root: {relative_path}") from exc
    return target


def _experiment_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _latest_report_dir(run_dir: Path) -> Path | None:
    full_eval = run_dir / "full_eval"
    if not full_eval.is_dir():
        return None
    step_dirs = sorted(
        [path for path in full_eval.glob("step_*") if (path / "summary.json").is_file()],
        key=lambda path: path.name,
    )
    if step_dirs:
        return step_dirs[-1]
    if (full_eval / "summary.json").is_file():
        return full_eval
    return None


def _is_test_run(run_dir: Path) -> bool:
    return (
        (run_dir / "examples" / "index.json").is_file()
        or (run_dir / "test_export.log").is_file()
        or (run_dir / "inspections_eval").is_dir()
    )


def _scan_candidate_dirs(root: Path) -> list[Path]:
    candidates = [root]
    part_root = root / "part_ss_latent_flow"
    if part_root.is_dir():
        candidates.extend(path for path in part_root.iterdir() if path.is_dir())
    candidates.extend(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else None
    seen = set()
    unique = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


class ExperimentArchive:
    def __init__(self, roots: list[str | Path]):
        self.roots = [Path(root).expanduser() for root in roots]

    def list_experiments(self) -> list[dict[str, Any]]:
        experiments = []
        for root in self.roots:
            if not root.exists():
                continue
            for run_dir in _scan_candidate_dirs(root):
                report_dir = _latest_report_dir(run_dir)
                if report_dir is not None:
                    experiments.append(self._eval_record(run_dir, report_dir))
                elif _is_test_run(run_dir):
                    experiments.append(self._test_record(run_dir))
        return sorted(experiments, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        for item in self.list_experiments():
            if item["id"] == experiment_id:
                root = Path(item["root"])
                detail = dict(item)
                if item["kind"] == "eval":
                    report_dir = Path(item["report_dir_abs"])
                    try:
                        metrics = load_eval_metrics(report_dir, tolerant=True)
                    except Exception as exc:
                        metrics = _diagnostic_metrics(
                            "eval",
                            inspect_eval_report(report_dir),
                            f"指标解析失败: {exc}",
                        )
                    detail["metrics"] = metrics
                    detail["examples"] = metrics["examples"]
                    detail["artifacts"] = {
                        "report": "report.md" if (report_dir / "report.md").is_file() else "",
                        "plots": metrics["plots"],
                    }
                else:
                    export_dir = Path(item["export_dir_abs"])
                    try:
                        detail["metrics"] = load_test_metrics(export_dir, tolerant=True)
                    except Exception as exc:
                        detail["metrics"] = _diagnostic_metrics(
                            "test",
                            inspect_test_export(export_dir),
                            f"导出索引解析失败: {exc}",
                        )
                    detail["examples"] = []
                    detail["artifacts"] = {"export_root": str(export_dir.relative_to(root))}
                return detail
        raise KeyError(experiment_id)

    def delete_experiment(self, experiment_id: str) -> str:
        item = next(
            (item for item in self.list_experiments() if item["id"] == experiment_id),
            None,
        )
        if item is None:
            raise KeyError(experiment_id)

        run_dir = Path(item["root"]).resolve()
        resolved_roots = [Path(root).resolve() for root in self.roots]

        # SAFETY: run_dir must be STRICTLY inside one configured root. Never the
        # root itself, never outside it. is_relative_to alone would accept the
        # root, so we also require run_dir != root.
        inside_root = any(
            run_dir != root and run_dir.is_relative_to(root) for root in resolved_roots
        )
        if not inside_root:
            raise ValueError(
                f"refusing to delete outside configured roots: {run_dir}"
            )

        if not run_dir.is_dir():
            raise KeyError(experiment_id)

        shutil.rmtree(run_dir)
        return str(run_dir)

    def _eval_record(self, run_dir: Path, report_dir: Path) -> dict[str, Any]:
        return {
            "id": _experiment_id(run_dir),
            "name": run_dir.name,
            "kind": "eval",
            "root": str(run_dir),
            "report_dir": str(report_dir.relative_to(run_dir)),
            "report_dir_abs": str(report_dir),
            "log": "full_eval.log" if (run_dir / "full_eval.log").is_file() else "",
            "updated_at": _mtime_text(report_dir / "summary.json"),
        }

    def _test_record(self, run_dir: Path) -> dict[str, Any]:
        export_dir = run_dir / "examples"
        return {
            "id": _experiment_id(run_dir),
            "name": run_dir.name,
            "kind": "test",
            "root": str(run_dir),
            "export_dir": str(export_dir.relative_to(run_dir)) if export_dir.exists() else "",
            "export_dir_abs": str(export_dir),
            "log": "test_export.log" if (run_dir / "test_export.log").is_file() else "",
            "updated_at": _mtime_text(run_dir / "test_export.log"),
        }


def _mtime_text(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{path.stat().st_mtime:.6f}"


def _diagnostic_metrics(task_kind: str, diagnostics: dict[str, Any], message: str) -> dict[str, Any]:
    diagnostics = dict(diagnostics)
    diagnostics["status"] = "incomplete"
    diagnostics["message"] = message
    diagnostics.setdefault("errors", []).append({"path": "", "message": message})
    return {
        "task_kind": task_kind,
        "summary": {},
        "overall": {},
        "focused": {},
        "size_buckets": {},
        "metric_definitions": {},
        "examples": [],
        "plots": [],
        "diagnostics": diagnostics,
    }
