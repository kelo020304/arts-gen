"""Wraps SAM3 image inference for SAM-style box-prompted segmentation.

Uses ``model.predict_inst(state, box=xyxy_pixels, ...)`` — the interactive
("SAM 1 task") path. Building requires ``enable_inst_interactivity=True``.

We deliberately avoid ``Sam3Processor.add_geometric_prompt`` even though it
looks similar: that API is open-vocabulary detection grounding (text + box),
not "given this box, segment THIS region". For ambiguous prompts on uniform
surfaces it returns oversized masks covering encompassing objects.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class MaskPipeline:
    def __init__(self,
                 ckpt_path: Path,
                 *,
                 device: str = "cuda"):
        """Load SAM 3 image model from a local checkpoint.

        ckpt_path: path to ``sam3.pt``.
        """
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._device = device
        self._model = build_sam3_image_model(
            checkpoint_path=str(ckpt_path),
            enable_inst_interactivity=True,
        )
        self._model = self._model.to(device).eval()
        self._processor = Sam3Processor(self._model, device=device)

    def embed(self, image: Image.Image) -> dict:
        """Run image backbone once; return state dict for repeated predictions.

        ``image`` MUST be a ``PIL.Image.Image`` — SAM 3's ``set_image`` reads
        ``np.ndarray.shape[-2:]`` as ``(H, W)`` but numpy is HWC, so passing a
        numpy array silently mangles dimensions.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"image must be PIL.Image.Image, got {type(image).__name__}. "
                "Pass PIL — SAM 3's set_image has a numpy axis bug."
            )
        return self._processor.set_image(image)

    def predict_box(self, state: dict,
                    x0: float, y0: float, x1: float, y1: float,
                    *,
                    pos_points: list[tuple[float, float]] | None = None,
                    neg_points: list[tuple[float, float]] | None = None,
                    ) -> "MaskOutput":
        """SAM-style box (+ optional positive / negative click points) → mask.

        All coords are pixel XYXY / XY in image space (matches what
        ``predict_inst`` expects with default ``normalize_coords=True``).
        SAM 3 labels: 1 = positive (include), 0 = negative (exclude).
        """
        from mask.types import MaskOutput

        kwargs: dict = {
            "box": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            "multimask_output": True,
        }
        coords: list[tuple[float, float]] = []
        labels: list[int] = []
        for p in pos_points or ():
            coords.append(p); labels.append(1)
        for p in neg_points or ():
            coords.append(p); labels.append(0)
        if coords:
            kwargs["point_coords"] = np.asarray(coords, dtype=np.float32)
            kwargs["point_labels"] = np.asarray(labels, dtype=np.int32)

        masks, scores, _ = self._model.predict_inst(state, **kwargs)

        if masks.shape[0] == 0:
            h, w = state["original_height"], state["original_width"]
            return MaskOutput(mask=np.zeros((h, w), dtype=bool), score=0.0)

        best = int(np.asarray(scores).argmax())
        return MaskOutput(
            mask=np.asarray(masks[best]).astype(bool),
            score=float(np.asarray(scores)[best]),
        )

    def unload(self) -> None:
        self._processor = None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
