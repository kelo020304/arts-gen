"""Inference-run scanning + path-safety helpers + object inputs preview.

产物目录契约：``<root>/<object_id>-<angle_idx>/<run_id>/``，每个 run 含 ``meta.json``
（``mode``/``view``/``object_id``/``run_id``/``stage_status``）外加各 stage 落盘的
artifact（``voxel.npz``、``ss_latent.npy``、``parts/...`` 等）。

安全约定（与 v1 相同）：``_run_dir`` / ``safe_run_artifact`` 一律
``resolve()`` 后用 ``startswith(base + os.sep)`` 严格包含校验，越界即 ``ValueError``，
绝不返回基目录之外的路径。失败暴露——不存在的 run/manifest 抛 ``KeyError`` 由上层
映射成 404。
"""

import os
from pathlib import Path
import json
import re
import sys
import types

__all__ = [
    "list_roots",
    "list_runs",
    "latest_run",
    "run_container_name",
    "read_manifest",
    "safe_run_artifact",
    "stage_outputs",
    "part_labels",
    "part_voxels_combined",
    "object_inputs_preview",
    "list_objects",
    "list_configs",
    "list_checkpoints",
]


TRELLIS_PATH = Path(__file__).resolve().parents[1]


def _setup_trellis_imports() -> None:
    """Register trellis packages without executing trellis/__init__.py.

    The platform server imports dataset helpers for object preview/job
    validation. Importing trellis normally executes trellis/__init__.py, which
    pulls demo pipelines and requires rembg. The platform only needs datasets
    and model utilities here, so use the same lightweight package shell as the
    train/eval/infer entry points.
    """
    if str(TRELLIS_PATH) not in sys.path:
        sys.path.insert(0, str(TRELLIS_PATH))
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines", "renderers"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules.setdefault(f"trellis.{sp}", mod)


_setup_trellis_imports()


def _read_meta(run_dir: Path) -> dict:
    p = run_dir / "meta.json"
    if not p.is_file():
        raise KeyError(f"meta.json 不存在：{p}")
    return json.loads(p.read_text())


def list_roots(infer_base) -> list:
    """列出 infer_base 下的 root 容器目录（每个 root 内含若干 object/run）。"""
    base = Path(infer_base)
    roots = []
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                roots.append({"name": entry.name, "path": str(entry.resolve())})
    return roots


def _object_id_from_container(name: str) -> str:
    """Best-effort object_id from an object-angle container name."""
    head, sep, tail = str(name).rpartition("-")
    if sep and head and tail.isdigit():
        return head
    return str(name)


def _append_run_overview(runs: list[dict], seen: set[str], run_dir: Path, container: str) -> None:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return
    resolved = str(run_dir.resolve())
    if resolved in seen:
        return
    try:
        meta = _read_meta(run_dir)
    except (json.JSONDecodeError, OSError):
        return
    object_id = str(meta.get("object_id") or _object_id_from_container(container))
    runs.append({
        "object_id": object_id,
        "run_id": run_dir.name,
        "mode": meta.get("mode"),
        "view": meta.get("view"),
        "angle_idx": meta.get("angle_idx"),
        "stage_status": meta.get("stage_status", {}),
        "path": resolved,
        "container": str(container),
    })
    seen.add(resolved)


def list_runs(root) -> list:
    """扫描 root 下的 run 概览。

    新布局为 ``<root>/<object_id>-<angle_idx>/<run_id>/meta.json``。读取端同时兼容
    历史布局：``<root>/<object_id>/<run_id>``、``<root>/<object_id>/<object_id>/<run_id>``
    以及曾经短暂产生过的 ``<root>/<object_id>-<angle>/<object_id>/<run_id>``。
    缺 meta.json 的 run 目录跳过（断点续传时尚未初始化）。
    """
    root_path = Path(root)
    runs = []
    seen: set[str] = set()
    if not root_path.is_dir():
        return runs
    for container_dir in sorted(root_path.iterdir()):
        if not container_dir.is_dir():
            continue
        # Also support callers selecting an object/angle container itself, e.g.
        # .../full-stage/101940-0 where direct children are run directories.
        _append_run_overview(runs, seen, container_dir, root_path.name)
        for child_dir in sorted(container_dir.iterdir()):
            if not child_dir.is_dir():
                continue
            _append_run_overview(runs, seen, child_dir, container_dir.name)
            for run_dir in sorted(child_dir.iterdir()):
                if run_dir.is_dir():
                    _append_run_overview(runs, seen, run_dir, container_dir.name)
    return runs


def run_container_name(object_id, angle_idx: int | str | None = None) -> str:
    """Directory container for one object-angle under an inference root."""
    if angle_idx is None or str(angle_idx).strip() == "":
        return str(object_id)
    return f"{object_id}-{int(angle_idx)}"


def _run_dir_candidates(root, object_id, run_id, angle_idx: int | str | None = None) -> list[Path]:
    base = Path(root).resolve()
    object_id = str(object_id)
    run_id = str(run_id)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            candidates.append(resolved)
            seen.add(key)

    if angle_idx is not None and str(angle_idx).strip() != "":
        angle_container = run_container_name(object_id, angle_idx)
        # Preferred new layout.
        add(base / angle_container / run_id)
        # Root may already be the object-angle container selected from the UI.
        add(base / run_id)
        # Historical full-stage layouts produced during web iteration.
        add(base / angle_container / object_id / run_id)
        add(base / angle_container / angle_container / run_id)

    # Legacy non-angle layouts, and the shape used when root already points at
    # one object container (e.g. .../full-stage/102252).
    add(base / run_id)
    add(base / object_id / run_id)
    add(base / object_id / object_id / run_id)
    return candidates


def latest_run(
    root,
    object_id,
    *,
    mode: str | None = None,
    view: str | None = None,
    angle_idx: int | None = None,
) -> dict | None:
    """Return the newest matching run overview for one object under a root, or None."""
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    target = str(object_id)
    candidates = [run for run in list_runs(root_path) if str(run.get("object_id")) == target]
    if mode:
        candidates = [run for run in candidates if run.get("mode") == mode]
    if view:
        candidates = [run for run in candidates if run.get("view") == view]
    if angle_idx is not None:
        candidates = [
            run for run in candidates
            if run.get("angle_idx") is not None and int(run.get("angle_idx")) == int(angle_idx)
        ]
    if not candidates:
        return None
    def sort_key(run: dict) -> tuple[float, str]:
        meta_path = Path(run.get("path", "")) / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (mtime, str(run.get("run_id", "")))
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


_OBJECT_CACHE: dict[tuple[str, str], list[dict]] = {}


def list_objects(data_config: dict, *, limit: int = 5000) -> list[dict]:
    """List object ids and available angles from the configured manifest."""
    data_root = Path(data_config.get("data_root", "")).resolve()
    manifest_path = Path(str(data_config.get("manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path
    key = (str(data_root), str(manifest_path.resolve()))
    cached = _OBJECT_CACHE.get(key)
    if cached is not None:
        return cached[:limit]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest_path not found: {manifest_path}")

    objects: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            object_id = str(rec.get("object_id") or rec.get("obj_id") or "")
            if not object_id:
                continue
            item = objects.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "angles": set(),
                    "name": rec.get("name") or "",
                    "category": rec.get("category") or "",
                    "target_part_names": set(),
                },
            )
            item["angles"].add(int(rec.get("angle_idx", 0)))
            for part_name in rec.get("target_part_names") or []:
                item["target_part_names"].add(str(part_name))

    out = []
    for object_id, item in objects.items():
        out.append({
            "object_id": object_id,
            "angles": sorted(item["angles"]),
            "name": item["name"],
            "category": item["category"],
            "target_part_names": sorted(item["target_part_names"]),
        })
    out.sort(key=lambda item: item["object_id"])
    _OBJECT_CACHE[key] = out
    return out[:limit]


def _run_dir(root, object_id, run_id, angle_idx: int | str | None = None) -> Path:
    """解析 run 目录并做严格包含校验，越界抛 ``ValueError``。"""
    base = Path(root).resolve()
    candidates = _run_dir_candidates(base, object_id, run_id, angle_idx=angle_idx)
    rd = candidates[0]
    if not rd.exists():
        rd = next((candidate for candidate in candidates[1:] if candidate.exists()), rd)
    if not rd.exists() and base.is_dir():
        for run in list_runs(base):
            if str(run.get("object_id")) != str(object_id):
                continue
            if str(run.get("run_id")) != str(run_id):
                continue
            if angle_idx is not None and str(angle_idx).strip() != "":
                if run.get("angle_idx") is None or int(run.get("angle_idx")) != int(angle_idx):
                    continue
            rd = Path(run["path"]).resolve()
            break
    base_prefix = str(base) + os.sep
    if not (str(rd) + os.sep).startswith(base_prefix):
        raise ValueError(f"run 路径越界：{object_id}/{run_id}")
    return rd


def read_manifest(root, object_id, run_id, *, angle_idx: int | str | None = None) -> dict:
    """读取某 run 的 ``meta.json`` + 列出全部 artifact（相对路径→字节数）。"""
    rd = _run_dir(root, object_id, run_id, angle_idx=angle_idx)
    if not rd.is_dir():
        raise KeyError(f"run 不存在：{object_id}/{run_id}")
    meta = _read_meta(rd)
    artifacts = {}
    for f in sorted(rd.rglob("*")):
        if f.is_file():
            artifacts[str(f.relative_to(rd).as_posix())] = f.stat().st_size
    return {
        "object_id": str(object_id),
        "run_id": str(run_id),
        "meta": meta,
        "artifacts": artifacts,
    }


def safe_run_artifact(root, object_id, run_id, rel: str, *, angle_idx: int | str | None = None) -> Path:
    """解析 run 内的某个 artifact 相对路径，越界（``..`` 逃逸）抛 ``ValueError``。"""
    rd = _run_dir(root, object_id, run_id, angle_idx=angle_idx)
    target = (rd / str(rel)).resolve()
    rd_prefix = str(rd) + os.sep
    if not (str(target) + os.sep).startswith(rd_prefix):
        raise ValueError(f"artifact 路径越界：{rel}")
    return target


def _component_artifact_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort decoded component artifacts as overall, body, part_00, ..., other."""
    stem = path.stem
    if stem == "overall":
        return (-1, -1, path.name)
    if stem == "body":
        return (0, -1, path.name)
    if stem.startswith("part_"):
        try:
            return (1, int(stem.split("_", 1)[1]), path.name)
        except ValueError:
            return (1, 10**9, path.name)
    return (2, 10**9, path.name)


def _slat_artifacts(parts_dir: Path, suffix: str) -> list[Path]:
    if not parts_dir.is_dir():
        return []
    return sorted(
        (
            p for p in parts_dir.glob(f"*{suffix}")
            if p.stem in {"overall", "body"} or p.stem.startswith("part_")
        ),
        key=_component_artifact_sort_key,
    )


def stage_outputs(root, object_id, run_id, *, angle_idx: int | str | None = None) -> dict:
    """Return per-stage artifact presence for overwrite prompts and run reuse."""
    rd = _run_dir(root, object_id, run_id, angle_idx=angle_idx)
    if not rd.is_dir():
        raise KeyError(f"run 不存在：{object_id}/{run_id}")
    parts = rd / "parts"
    specs = {
        "ss": [rd / "voxel.npz", rd / "voxel.bin", rd / "ss_latent.npy"],
        "part": sorted(parts.glob("part_*_voxel.npz")) if parts.is_dir() else [],
        "slat": (
            _slat_artifacts(parts, ".glb") +
            _slat_artifacts(parts, ".ply")
            if parts.is_dir() else []
        ),
        "assemble": [rd / "complete.glb", rd / "complete.ply"],
    }
    out = {}
    for stage, paths in specs.items():
        artifacts = [
            str(p.relative_to(rd).as_posix())
            for p in paths
            if p.is_file()
        ]
        out[stage] = {"exists": bool(artifacts), "artifacts": artifacts}
    return out


def _read_part_label(parts_dir: Path, index: int) -> str:
    meta_path = parts_dir / f"part_{index:02d}_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            label = str(meta.get("target_part_name") or "").strip()
            if label:
                return label
        except (OSError, ValueError):
            pass

    voxel_path = parts_dir / f"part_{index:02d}_voxel.npz"
    if voxel_path.is_file():
        try:
            import numpy as np
            with np.load(voxel_path, allow_pickle=False) as data:
                if "target_part_name" in data.files:
                    value = data["target_part_name"]
                    label = str(value.item() if getattr(value, "shape", None) == () else value).strip()
                    if label:
                        return label
        except Exception:
            pass

    return f"part_{index:02d}"


def part_labels(root, object_id, run_id, *, angle_idx: int | str | None = None) -> dict:
    """Return display labels for decoded body/part artifacts in one run."""
    rd = _run_dir(root, object_id, run_id, angle_idx=angle_idx)
    if not rd.is_dir():
        raise KeyError(f"run 不存在：{object_id}/{run_id}")
    parts_dir = rd / "parts"
    if not parts_dir.is_dir():
        raise KeyError(f"parts 目录不存在：{object_id}/{run_id}")
    components = []
    if (parts_dir / "overall.glb").is_file() or (parts_dir / "overall.ply").is_file():
        components.append({"stem": "overall", "label": "overall", "body": False, "index": -2})
    if (parts_dir / "body.glb").is_file() or (parts_dir / "body.ply").is_file() or (parts_dir / "body_voxel.npz").is_file():
        components.append({"stem": "body", "label": "body", "body": True, "index": -1})
    indices: set[int] = set()
    for path in parts_dir.glob("part_*.*"):
        stem = path.stem
        if stem.endswith("_voxel"):
            stem = stem[: -len("_voxel")]
        if stem.endswith("_latent"):
            stem = stem[: -len("_latent")]
        if stem.endswith("_meta"):
            stem = stem[: -len("_meta")]
        if not stem.startswith("part_"):
            continue
        try:
            indices.add(int(stem.split("_", 1)[1]))
        except ValueError:
            continue
    for index in sorted(indices):
        stem = f"part_{index:02d}"
        components.append({
            "stem": stem,
            "label": _read_part_label(parts_dir, index),
            "body": False,
            "index": index,
        })
    return {"components": components}


# body 体素的标签哨兵（uint16）；part 标签是 0,1,2… 小值，65535 不会撞。前端据此把
# body 画成灰色半透明上下文、target part 画成彩色实心（对齐 eval 的 “灰 body + 彩 part”）。
BODY_VOXEL_LABEL = 65535


def part_voxels_combined(
    root,
    object_id,
    run_id,
    *,
    include_body: bool = True,
    angle_idx: int | str | None = None,
) -> bytes:
    """合并 ``parts/part_*_voxel.npz`` 成一段带标签的 LE uint16 字节流，每个体素 4 个值
    ``[x, y, z, label]``：part_label = part 文件顺序 0,1,2…（相邻 part 不同色）；
    ``include_body`` 时把整体 SS voxel（``voxel.npz``）**减去各 part** 作为 body 上下文，
    标签 ``BODY_VOXEL_LABEL``（前端画灰色半透明）。

    缺 parts 目录 / 无任何 part voxel → ``KeyError``（上层 404），不静默返回空。
    """
    import numpy as np

    rd = _run_dir(root, object_id, run_id, angle_idx=angle_idx)
    parts_dir = rd / "parts"
    if not parts_dir.is_dir():
        raise KeyError(f"parts 目录不存在：{object_id}/{run_id}")
    chunks = []
    part_coords_seen: set[tuple[int, int, int]] = set()
    for label, npz in enumerate(sorted(parts_dir.glob("part_*_voxel.npz"))):
        with np.load(npz) as data:
            coords = np.asarray(data["coords"]).astype(np.int32).reshape(-1, 3)
        if coords.size == 0:
            continue
        labels = np.full((coords.shape[0], 1), label, dtype=np.int32)
        chunks.append(np.concatenate([coords, labels], axis=1))
        part_coords_seen.update(map(tuple, coords.tolist()))
    if not chunks:
        raise KeyError(f"无 part voxel（先跑完 part 阶段）：{object_id}/{run_id}")
    # body = 整体 SS voxel 减去 part（不重叠，避免和彩色 part 抢同格 z-fighting）。
    body_npz = rd / "voxel.npz"
    if include_body and body_npz.is_file():
        with np.load(body_npz) as data:
            body = np.asarray(data["coords"]).astype(np.int32).reshape(-1, 3)
        if body.size:
            keep = np.array([tuple(c) not in part_coords_seen for c in body.tolist()], dtype=bool)
            body = body[keep]
        if body.size:
            blabels = np.full((body.shape[0], 1), BODY_VOXEL_LABEL, dtype=np.int32)
            chunks.insert(0, np.concatenate([body, blabels], axis=1))  # body 在前，先画
    return np.concatenate(chunks, axis=0).astype("<u2").tobytes()


# ---- /api/infer/options：configs + checkpoints 扫描 ----------------------------

def list_configs(repo_root) -> list:
    """扫描 ``<repo_root>/TRELLIS-arts/configs/arts/**/*.yaml``，返回训练/推理 config 列表。

    label = 相对 ``configs/arts`` 的路径（如 ``part_ss_latent_flow/part_ss_latent_flow.yaml``），
    按 label 排序。configs/arts 目录不存在则返回空列表（不报错）。
    """
    cfg_dir = (Path(repo_root) / "TRELLIS-arts" / "configs" / "arts").resolve()
    out = []
    if cfg_dir.is_dir():
        for f in cfg_dir.rglob("*.yaml"):
            if f.is_file():
                out.append({
                    "path": str(f.resolve()),
                    "label": str(f.resolve().relative_to(cfg_dir).as_posix()),
                })
    out.sort(key=lambda item: item["label"])
    return out


def _label_last_n(path: Path, n: int) -> str:
    """取路径最后 n 个组成部分作为 label（如 ``ckpts/foo.safetensors``）。"""
    parts = path.parts
    return "/".join(parts[-n:]) if len(parts) >= n else "/".join(parts)


def _checkpoint_label(path: Path, label2: dict[str, list[str]]) -> str:
    """Human-readable checkpoint label for dropdowns.

    Training outputs named denoiser_step*.pt / denoiser_ema*.pt are common
    across runs; always include the run directory so the UI shows e.g.
    tre_ss_flow_4_0605/ckpts/denoiser_step0070000.pt instead of only
    ckpts/denoiser_step0070000.pt.
    """
    lbl = _label_last_n(path, 2)
    if _is_preferred_checkpoint(path) and path.parent.name == "ckpts":
        return _label_last_n(path, 3)
    if len(label2[lbl]) > 1:
        return _label_last_n(path, 3)
    return lbl


# HuggingFace cache internals — `snapshots/*.safetensors` are symlinks that
# resolve into `blobs/<sha256>` (no extension, unreadable label), and the cache
# holds unrelated models (Hunyuan3D / dinov2 / ...). Exclude the whole cache so
# the dropdown only shows real named checkpoints.
_CKPT_SKIP_PARTS = {"blobs", "hf_cache", ".cache"}


def _collect_checkpoints(seen: dict, files_iter, cap: int) -> bool:
    """把 ``files_iter`` 里的 ckpt 路径并入 ``seen``（按 resolve 后路径去重）。

    跳过 HuggingFace 缓存（blobs/hf_cache）——那是别的模型的无名 hash blob。
    达到 cap 即返回 True（已满，调用方应停止继续收集），否则返回 False。
    """
    for f in files_iter:
        if len(seen) >= cap:
            return True
        if not f.is_file():
            continue
        rp = f.resolve()
        # Skip HF-cache blobs whether matched via the symlink path or its target.
        if _CKPT_SKIP_PARTS & set(f.parts) or _CKPT_SKIP_PARTS & set(rp.parts):
            continue
        seen[str(rp)] = rp
    return len(seen) >= cap

_PREFERRED_CKPT_RE = re.compile(
    r"(^|/)(denoiser(?:_ema[0-9.]+)?_step\d+\.pt|ss[-_]flow.*\.(?:safetensors|pt|ckpt)|part[-_]promptable[-_]seg[^/]*/ckpts/(?:latest|step_\d+)\.pt)$",
    re.IGNORECASE,
)


def _is_preferred_checkpoint(path: Path) -> bool:
    text = str(Path(path).as_posix())
    return bool(_PREFERRED_CKPT_RE.search(text))


def _iter_preferred(files_iter):
    for f in files_iter:
        if _is_preferred_checkpoint(f):
            yield f


# checkpoint extensions: TRELLIS uses .safetensors/.pt, sam3d uses .ckpt.
_CKPT_EXTS = ("*.safetensors", "*.pt", "*.ckpt")
# sam3d weights drop (ss_decoder.ckpt / ss_generator.ckpt / slat_*.ckpt + pipeline.yaml);
# overridable for non-dev layouts.
_SAM3D_WEIGHTS_DIR_DEFAULT = "/robot/data-lab/jzh/art-gen/weights"
# Third-party / TRELLIS pretrained weights live on VePFS (too big for the code
# dir): ss_dec_conv3d / slat_dec / dinov2 etc. Recursively scanned.
_THIRD_PARTY_DIR_DEFAULT = "/robot/data-lab/jzh/arts-gen/third-party-weights"
_EXPLICIT_CHECKPOINT_DEFAULTS = (
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.9999_step0020000.pt",
    "/robot/data-lab/jzh/art-gen-output/part_promptable_seg_full_M_0612-3/ckpts/latest.pt",
)


def _explicit_checkpoint_paths() -> list[Path]:
    """Operator-pinned inference ckpts that must survive scan caps."""
    text = os.environ.get("PART_SS_PLATFORM_EXTRA_CKPTS", "")
    include_defaults = os.environ.get("PART_SS_PLATFORM_DISABLE_DEFAULT_CKPTS", "").strip().lower()
    paths = [] if include_defaults in {"1", "true", "yes"} else [Path(p) for p in _EXPLICIT_CHECKPOINT_DEFAULTS]
    paths.extend(Path(piece.strip()) for piece in text.split(",") if piece.strip())
    return paths


def list_checkpoints(repo_root, scan_roots, *, cap: int = 400) -> list:
    """汇总可选 checkpoint，覆盖 TRELLIS 与 **sam3d** 两套权重：
      (a) ``<repo_root>/pretrained`` 递归 ``*.{safetensors,pt,ckpt}``（TRELLIS 预训练）；
      (b) sam3d 权重目录（env ``SAM3D_WEIGHTS_DIR``，默认 ``/robot/data-lab/jzh/art-gen/weights``）
          顶层 ``*.{ckpt,safetensors,pt}`` —— SS/SLat 的 latent 用 sam3d 权重，故
          ``ss_decoder.ckpt`` / ``ss_generator.ckpt`` / ``slat_*.ckpt`` 必须可选；
      (c) 每个 scan root 下 ``<root>/**/ckpts/*.{pt,safetensors,ckpt}``（训练输出，如 part flow）。

    label = 路径最后两段（如 ``weights/ss_decoder.ckpt``）；冲突则升级末三段。按 path
    resolve 去重，按 label 排序，HARD CAP 在 ``cap``（默认 400）。缺失目录静默跳过；
    用有界 ``rglob`` 收集，一旦达到 cap 立即停止。
    """
    seen: dict = {}  # str(resolved path) -> Path，天然去重

    # Operator-pinned ckpts are added before broad recursive scans so the UI can
    # always choose the current full-stage defaults even when scan roots contain
    # hundreds of old training checkpoints and cap out.
    for path in _explicit_checkpoint_paths():
        if len(seen) >= cap:
            break
        if path.is_file():
            rp = path.resolve()
            seen[str(rp)] = rp

    # (a) pretrained 递归（TRELLIS 预训练 .safetensors/.pt/.ckpt）。
    pretrained = (Path(repo_root) / "pretrained").resolve()
    if pretrained.is_dir():
        for pattern in _CKPT_EXTS:
            if _collect_checkpoints(seen, pretrained.rglob(pattern), cap):
                break

    # (b) sam3d 权重目录（顶层，不递归——里面就是一层 ckpt+yaml）。
    if len(seen) < cap:
        sam3d_dir = Path(os.environ.get("SAM3D_WEIGHTS_DIR") or _SAM3D_WEIGHTS_DIR_DEFAULT)
        if sam3d_dir.is_dir():
            for pattern in _CKPT_EXTS:
                if _collect_checkpoints(seen, sam3d_dir.glob(pattern), cap):
                    break

    # (b2) 第三方 / TRELLIS pretrained 权重目录（VePFS，递归）——ss_dec_conv3d 等。
    if len(seen) < cap:
        tp_dir = Path(os.environ.get("THIRD_PARTY_WEIGHTS_DIR") or _THIRD_PARTY_DIR_DEFAULT)
        if tp_dir.is_dir():
            for pattern in _CKPT_EXTS:
                if _collect_checkpoints(seen, tp_dir.rglob(pattern), cap):
                    break

    # (c) 每个 scan root 下的 <root>/**/ckpts/*.{pt,safetensors,ckpt}。
    if len(seen) < cap:
        for root in (scan_roots or []):
            if len(seen) >= cap:
                break
            root_path = Path(root)
            if not root_path.is_dir():
                continue  # 缺失目录（dev 才有的 /robot/data-lab）静默跳过
            for pattern in ("ckpts/*.pt", "ckpts/*.safetensors", "ckpts/*.ckpt"):
                if _collect_checkpoints(seen, _iter_preferred(root_path.rglob(pattern)), cap):
                    break
                if _collect_checkpoints(seen, root_path.rglob(pattern), cap):
                    break

    # label 计算：默认末两段，冲突则升级到末三段。
    label2 = {}
    for key, rp in seen.items():
        label2.setdefault(_label_last_n(rp, 2), []).append(key)
    out = []
    for key, rp in seen.items():
        lbl = _checkpoint_label(rp, label2)
        out.append({"path": key, "label": lbl})
    out.sort(key=lambda item: item["label"])
    return out


# 关键新增：object_inputs_preview —— /api/infer/inputs 的真实实现
def object_inputs_preview(state, object_id: str, *, angle_idx: int = 0, view_mode: str = "four") -> dict:
    """预览：该物体的输入 rgb 路径列表 + 是否有 GT voxel（模式A可用）。
    复用 eval dataset（object_inputs），失败暴露（manifest 无该物体 → KeyError → 上层 404/400）。"""
    from inference_pipeline.object_inputs import load_object_inputs
    from inference_pipeline.inputs_materialize import resolve_rgb_path
    dc = state.infer_data_config
    item = load_object_inputs(dc, object_id=object_id, angle_idx=angle_idx, view_mode=view_mode)
    rgb_rels = []
    for v in item["view_indices"]:
        try:
            rgb_rels.append(str(resolve_rgb_path(dc, object_id=object_id, angle_idx=angle_idx, view_index=v)))
        except FileNotFoundError:
            pass
    has_gt = item.get("raw_surface_coords") is not None
    return {"object_id": str(object_id), "angle_idx": angle_idx, "view_indices": list(item["view_indices"]),
            "rgb_paths": rgb_rels, "has_gt_voxel": bool(has_gt),
            "target_part_names": list(item["target_part_names"])}
