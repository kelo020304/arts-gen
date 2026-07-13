from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import base64
import io
import numpy as np
from PIL import Image


@dataclass
class BoxPrompt:
    """Normalized [0, 1] bounding box: (cx, cy, w, h) — center + size."""
    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self):
        for name, v in (("cx", self.cx), ("cy", self.cy), ("w", self.w), ("h", self.h)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"BoxPrompt.{name}={v} outside [0, 1]")
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"BoxPrompt has non-positive size: w={self.w}, h={self.h}")

    def as_list(self) -> list[float]:
        return [self.cx, self.cy, self.w, self.h]

    def to_xyxy_pixels(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert normalized (cx, cy, w, h) -> pixel XYXY for SAM-style box prompts."""
        x0 = (self.cx - self.w / 2) * img_w
        y0 = (self.cy - self.h / 2) * img_h
        x1 = (self.cx + self.w / 2) * img_w
        y1 = (self.cy + self.h / 2) * img_h
        return x0, y0, x1, y1

    @classmethod
    def from_pixel_box(cls, x0: float, y0: float, x1: float, y1: float,
                       img_w: int, img_h: int) -> "BoxPrompt":
        """Convert pixel corner box to normalized center+size."""
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"image dims must be positive, got {img_w}x{img_h}")
        lo_x, hi_x = sorted((float(x0), float(x1)))
        lo_y, hi_y = sorted((float(y0), float(y1)))
        if hi_x <= lo_x or hi_y <= lo_y:
            raise ValueError(
                f"degenerate box: ({x0},{y0})-({x1},{y1})"
            )
        if lo_x < 0 or lo_y < 0 or hi_x > img_w or hi_y > img_h:
            raise ValueError(
                f"box ({lo_x},{lo_y})-({hi_x},{hi_y}) outside image {img_w}x{img_h}"
            )
        w_px = hi_x - lo_x
        h_px = hi_y - lo_y
        cx_px = lo_x + w_px / 2.0
        cy_px = lo_y + h_px / 2.0
        return cls(
            cx=cx_px / img_w,
            cy=cy_px / img_h,
            w=w_px / img_w,
            h=h_px / img_h,
        )


@dataclass
class MaskOutput:
    mask: np.ndarray       # (H, W) bool
    score: float

    def save_png(self, path: Path) -> None:
        """Write canonical single-channel 0/255 PNG (used for the persisted mask file)."""
        path = Path(path)
        arr = (self.mask.astype(np.uint8) * 255)
        Image.fromarray(arr, mode="L").save(path)

    def to_base64_png(self) -> str:
        """Return a RED-on-TRANSPARENT RGBA data URL for direct browser overlay.

        Why RGBA, not grayscale: a grayscale PNG drawn to <canvas> has alpha=255
        everywhere (the L→RGBA promotion fills alpha), which breaks any "tint
        on the mask area" composite trick. Bake the tint here instead — drawImage
        on the result paints red where mask is on, transparent otherwise.
        """
        h, w = self.mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 255                                # R
        rgba[..., 3] = self.mask.astype(np.uint8) * 255   # A = mask
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
