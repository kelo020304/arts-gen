from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_CHUNK = 1024 * 1024


async def save_upload(file: UploadFile, dest: Path) -> Path:
    """Stream-save an UploadFile to ``dest``. Returns ``dest``.

    Refuses (HTTP 413) if more than 50 MB is read.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await file.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"upload {file.filename!r} exceeds {MAX_UPLOAD_BYTES} bytes",
                )
            fh.write(chunk)
    return dest


async def read_upload_capped(
    file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES
) -> bytes:
    """Read an UploadFile fully into memory with a size cap. HTTP 413 if exceeded."""
    buf = await file.read(max_bytes + 1)
    if len(buf) > max_bytes:
        raise HTTPException(
            413,
            f"upload {file.filename!r} exceeds {max_bytes} bytes",
        )
    return buf


def read_image_rgb(path: Path) -> np.ndarray:
    """PNG/JPG/WebP -> (H, W, 3) uint8 RGB. Drops alpha if present."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def read_mask_bool(path: Path) -> np.ndarray:
    """PNG -> (H, W) bool.

    If single-channel: threshold > 127.
    If RGB(A): treat any non-zero channel as True.
    """
    img = Image.open(path)
    if img.mode in ("L", "1", "I", "I;16"):
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 127
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return arr.any(axis=-1)


def read_npy(path: Path) -> np.ndarray:
    """``np.load`` passthrough; raises HTTPException 400 on failure."""
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"cannot load .npy {path.name!r}: {exc}") from exc
