"""Foreground point mask encoder for promptable part segmentation.

This module is intentionally independent from the current training path.  It
encodes only foreground mask pixels as point tokens, plus one centroid token per
visible view, and returns padded tokens with a key-padding mask for later
cross-attention integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


__all__ = [
    "PointMaskEncoder",
    "PointMaskEncoderOutput",
    "sample_mask_points",
]


TYPE_INTERIOR = 0
TYPE_BOUNDARY = 1
TYPE_CENTROID = 2
TYPE_NAMES = ("interior", "boundary", "centroid")


@dataclass(frozen=True)
class PointMaskSample:
    coords_xy: np.ndarray
    point_types: np.ndarray
    dist_to_boundary: np.ndarray
    log_area: float


@dataclass(frozen=True)
class PointMaskEncoderOutput:
    tokens: torch.Tensor
    key_padding_mask: torch.Tensor
    counts: torch.Tensor
    coords_uv: torch.Tensor
    point_types: torch.Tensor
    no_prompt_mask: torch.Tensor


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _erode3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool, copy=False), 1, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool, copy=False)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return [np.asarray(comp, dtype=np.int64) for comp in comps]


def _allocate_counts(sizes: list[int], total: int, *, min_each: int) -> list[int]:
    if total <= 0 or not sizes:
        return [0 for _ in sizes]
    active = [idx for idx, size in enumerate(sizes) if size > 0]
    if total >= len(active) * min_each:
        counts = [min_each if idx in active else 0 for idx in range(len(sizes))]
        remaining = total - sum(counts)
    else:
        counts = [0 for _ in sizes]
        order = sorted(active, key=lambda idx: sizes[idx], reverse=True)
        for idx in order[:total]:
            counts[idx] = 1
        return counts
    if remaining <= 0:
        return counts
    total_size = float(sum(sizes[idx] for idx in active))
    raw = [remaining * sizes[idx] / total_size if idx in active else 0.0 for idx in range(len(sizes))]
    floors = [int(math.floor(value)) for value in raw]
    for idx, extra in enumerate(floors):
        counts[idx] += extra
    leftover = remaining - sum(floors)
    order = sorted(active, key=lambda idx: (raw[idx] - floors[idx], sizes[idx]), reverse=True)
    for idx in order[:leftover]:
        counts[idx] += 1
    return counts


def _deterministic_order(points_yx: np.ndarray) -> np.ndarray:
    if points_yx.shape[0] == 0:
        return points_yx
    order = np.lexsort((points_yx[:, 1], points_yx[:, 0]))
    return points_yx[order]


def _sample_even(points_yx: np.ndarray, count: int, *, rng: np.random.Generator | None) -> np.ndarray:
    points_yx = _deterministic_order(points_yx)
    n_points = points_yx.shape[0]
    if count <= 0 or n_points == 0:
        return points_yx[:0]
    if n_points <= count:
        return points_yx
    if rng is not None:
        choice = rng.choice(n_points, size=count, replace=False)
        return points_yx[np.sort(choice)]
    idx = np.linspace(0, n_points - 1, num=count, dtype=np.int64)
    return points_yx[idx]


def _contour_order(points_yx: np.ndarray) -> np.ndarray:
    points_yx = _deterministic_order(points_yx)
    if points_yx.shape[0] <= 2:
        return points_yx
    center = points_yx.astype(np.float32).mean(axis=0)
    dy = points_yx[:, 0].astype(np.float32) - center[0]
    dx = points_yx[:, 1].astype(np.float32) - center[1]
    angle = np.arctan2(dy, dx)
    radius = dx * dx + dy * dy
    order = np.lexsort((points_yx[:, 1], points_yx[:, 0], radius, angle))
    return points_yx[order]


def _sample_even_ordered(points_yx: np.ndarray, count: int, *, rng: np.random.Generator | None) -> np.ndarray:
    n_points = points_yx.shape[0]
    if count <= 0 or n_points == 0:
        return points_yx[:0]
    if n_points <= count:
        return points_yx
    if rng is not None:
        choice = rng.choice(n_points, size=count, replace=False)
        return points_yx[np.sort(choice)]
    idx = np.linspace(0, n_points - 1, num=count, dtype=np.int64)
    return points_yx[idx]


def _fps(points_yx: np.ndarray, count: int, *, rng: np.random.Generator | None) -> np.ndarray:
    points_yx = _deterministic_order(points_yx)
    n_points = points_yx.shape[0]
    if count <= 0 or n_points == 0:
        return points_yx[:0]
    if n_points <= count:
        return points_yx
    coords = points_yx.astype(np.float32)
    if rng is None:
        start = int(np.lexsort((points_yx[:, 1], points_yx[:, 0]))[0])
    else:
        start = int(rng.integers(0, n_points))
    selected = [start]
    min_dist = np.sum((coords - coords[start]) ** 2, axis=1)
    for _ in range(1, count):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        dist = np.sum((coords - coords[nxt]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    selected_arr = np.asarray(selected, dtype=np.int64)
    order = np.lexsort((points_yx[selected_arr, 1], points_yx[selected_arr, 0]))
    return points_yx[selected_arr[order]]


def _distance_to_boundary(points_yx: np.ndarray, boundary_yx: np.ndarray, scale: float) -> np.ndarray:
    if points_yx.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if boundary_yx.shape[0] == 0:
        return np.zeros((points_yx.shape[0],), dtype=np.float32)
    p = points_yx.astype(np.float32)
    b = boundary_yx.astype(np.float32)
    out = np.empty((p.shape[0],), dtype=np.float32)
    chunk = 1024
    for start in range(0, p.shape[0], chunk):
        diff = p[start : start + chunk, None, :] - b[None, :, :]
        out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
    return out / max(float(scale), 1.0)


def sample_mask_points(
    mask: np.ndarray,
    *,
    k_boundary: int = 32,
    k_interior: int = 32,
    resample_points: bool = False,
    rng: np.random.Generator | None = None,
) -> PointMaskSample:
    """Sample foreground point prompts from a single 2D mask.

    Coordinates are returned as pixel-center ``(x, y)`` float32 values.
    """
    mask_bool = _as_bool_mask(mask)
    if k_boundary < 0 or k_interior < 0:
        raise ValueError("k_boundary and k_interior must be non-negative")
    if not bool(mask_bool.any()):
        return PointMaskSample(
            coords_xy=np.zeros((0, 2), dtype=np.float32),
            point_types=np.zeros((0,), dtype=np.int64),
            dist_to_boundary=np.zeros((0,), dtype=np.float32),
            log_area=0.0,
        )

    local_rng = rng if resample_points else None
    area_total = int(mask_bool.sum())
    eroded = _erode3(mask_bool)
    boundary_mask = mask_bool ^ eroded
    comps = _connected_components(mask_bool)
    sizes = [int(comp.shape[0]) for comp in comps]
    boundary_counts = _allocate_counts(sizes, int(k_boundary), min_each=1)
    interior_counts = _allocate_counts(sizes, int(k_interior), min_each=1)

    all_boundary_yx = np.argwhere(boundary_mask)
    scale = math.sqrt(float(area_total))
    coords: list[np.ndarray] = []
    types: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for comp, nb, ni in zip(comps, boundary_counts, interior_counts):
        comp_mask = np.zeros_like(mask_bool, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        comp_boundary = np.argwhere(comp_mask & boundary_mask)
        if comp_boundary.shape[0] == 0:
            comp_boundary = comp
        boundary = _sample_even_ordered(
            _contour_order(comp_boundary),
            min(int(nb), comp_boundary.shape[0]),
            rng=local_rng,
        )
        interior = _fps(comp, min(int(ni), comp.shape[0]), rng=local_rng)
        if boundary.shape[0] > 0:
            coords.append(boundary[:, ::-1].astype(np.float32))
            types.append(np.full((boundary.shape[0],), TYPE_BOUNDARY, dtype=np.int64))
            dists.append(np.zeros((boundary.shape[0],), dtype=np.float32))
        if interior.shape[0] > 0:
            coords.append(interior[:, ::-1].astype(np.float32))
            types.append(np.full((interior.shape[0],), TYPE_INTERIOR, dtype=np.int64))
            dists.append(_distance_to_boundary(interior, all_boundary_yx, scale))

    ys, xs = np.nonzero(mask_bool)
    centroid_xy = np.asarray([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
    coords.append(centroid_xy)
    types.append(np.asarray([TYPE_CENTROID], dtype=np.int64))
    dists.append(np.zeros((1,), dtype=np.float32))

    return PointMaskSample(
        coords_xy=np.concatenate(coords, axis=0),
        point_types=np.concatenate(types, axis=0),
        dist_to_boundary=np.concatenate(dists, axis=0).astype(np.float32, copy=False),
        log_area=float(math.log1p(area_total)),
    )


class PointMaskEncoder(nn.Module):
    """Encode foreground mask points into variable-length prompt tokens."""

    def __init__(
        self,
        *,
        dim: int = 256,
        num_views: int = 4,
        mask_size: int = 512,
        k_boundary: int = 32,
        k_interior: int = 32,
        fourier_bands: int = 10,
        resample_points: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
        self.k_boundary = int(k_boundary)
        self.k_interior = int(k_interior)
        self.fourier_bands = int(fourier_bands)
        self.resample_points = bool(resample_points)
        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")
        if self.mask_size < 2:
            raise ValueError(f"mask_size must be >= 2, got {self.mask_size}")
        if self.fourier_bands < 1:
            raise ValueError(f"fourier_bands must be >= 1, got {self.fourier_bands}")
        in_dim = 2 + 4 * self.fourier_bands + 2
        self.proj = nn.Linear(in_dim, self.dim)
        self.view_emb = nn.Parameter(torch.zeros(self.num_views, self.dim))
        self.type_emb = nn.Embedding(3, self.dim)
        self.register_buffer("no_prompt", torch.zeros(1, self.dim), persistent=True)
        self.norm = nn.LayerNorm(self.dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.no_prompt, std=0.02)

    @property
    def max_tokens_per_sample(self) -> int:
        return self.num_views * (self.k_boundary + self.k_interior + 1)

    def _encode_features(
        self,
        coords_uv: torch.Tensor,
        dist_to_boundary: torch.Tensor,
        log_area: torch.Tensor,
    ) -> torch.Tensor:
        bands = (2.0 ** torch.arange(self.fourier_bands, device=coords_uv.device, dtype=coords_uv.dtype)).view(1, -1)
        scaled = coords_uv.unsqueeze(-1) * bands.unsqueeze(0) * math.pi
        fourier = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(1)
        raw = torch.cat([coords_uv, fourier, dist_to_boundary[:, None], log_area[:, None]], dim=-1)
        return self.proj(raw)

    def _sample_batch(self, masks2d: torch.Tensor) -> list[list[PointMaskSample]]:
        masks_np = masks2d.detach().cpu().numpy()
        out: list[list[PointMaskSample]] = []
        rng = np.random.default_rng() if self.resample_points and self.training else None
        for b_idx in range(masks_np.shape[0]):
            views = []
            for v_idx in range(masks_np.shape[1]):
                views.append(
                    sample_mask_points(
                        masks_np[b_idx, v_idx],
                        k_boundary=self.k_boundary,
                        k_interior=self.k_interior,
                        resample_points=self.resample_points and self.training,
                        rng=rng,
                    )
                )
            out.append(views)
        return out

    def forward(self, masks2d: torch.Tensor) -> PointMaskEncoderOutput:
        if masks2d.dim() != 4:
            raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
        batch, views, height, width = masks2d.shape
        if views != self.num_views or height != self.mask_size or width != self.mask_size:
            raise ValueError(
                f"masks2d expected [B,{self.num_views},{self.mask_size},{self.mask_size}], "
                f"got {tuple(masks2d.shape)}"
            )

        sampled = self._sample_batch(masks2d)
        counts = torch.tensor(
            [sum(view.coords_xy.shape[0] for view in sample_views) for sample_views in sampled],
            dtype=torch.long,
            device=masks2d.device,
        )
        max_len = max(int(counts.max().item()) if counts.numel() else 0, 1)
        tokens = masks2d.new_zeros((batch, max_len, self.dim), dtype=torch.float32)
        coords_uv_out = masks2d.new_full((batch, max_len, 2), -1.0, dtype=torch.float32)
        type_out = torch.full((batch, max_len), -1, dtype=torch.long, device=masks2d.device)
        key_padding_mask = torch.ones((batch, max_len), dtype=torch.bool, device=masks2d.device)
        no_prompt_mask = torch.zeros((batch,), dtype=torch.bool, device=masks2d.device)

        denom = float(self.mask_size - 1)
        for b_idx, sample_views in enumerate(sampled):
            pieces = []
            view_ids = []
            for v_idx, view_sample in enumerate(sample_views):
                n_view = int(view_sample.coords_xy.shape[0])
                if n_view == 0:
                    continue
                coords_xy = torch.from_numpy(view_sample.coords_xy).to(device=masks2d.device, dtype=torch.float32)
                coords_uv = coords_xy / denom
                dist = torch.from_numpy(view_sample.dist_to_boundary).to(device=masks2d.device, dtype=torch.float32)
                area_norm = torch.full((n_view,), view_sample.log_area / math.log1p(self.mask_size * self.mask_size), device=masks2d.device)
                point_types = torch.from_numpy(view_sample.point_types).to(device=masks2d.device, dtype=torch.long)
                feat = self._encode_features(coords_uv, dist, area_norm)
                feat = feat + self.view_emb[v_idx].to(device=feat.device, dtype=feat.dtype)
                feat = feat + self.type_emb(point_types).to(dtype=feat.dtype)
                pieces.append((feat, coords_uv, point_types))
                view_ids.append(v_idx)
            if not pieces:
                tokens[b_idx, 0] = self.no_prompt[0].to(device=tokens.device, dtype=tokens.dtype)
                coords_uv_out[b_idx, 0] = -1.0
                type_out[b_idx, 0] = TYPE_CENTROID
                key_padding_mask[b_idx, 0] = False
                counts[b_idx] = 1
                no_prompt_mask[b_idx] = True
                continue
            feat_cat = torch.cat([piece[0] for piece in pieces], dim=0)
            coord_cat = torch.cat([piece[1] for piece in pieces], dim=0)
            type_cat = torch.cat([piece[2] for piece in pieces], dim=0)
            n = feat_cat.shape[0]
            tokens[b_idx, :n] = self.norm(feat_cat)
            coords_uv_out[b_idx, :n] = coord_cat
            type_out[b_idx, :n] = type_cat
            key_padding_mask[b_idx, :n] = False

        zero_dep = tokens.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                zero_dep = zero_dep + param.sum().to(dtype=tokens.dtype) * 0.0
        tokens = tokens + zero_dep
        return PointMaskEncoderOutput(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            counts=counts,
            coords_uv=coords_uv_out,
            point_types=type_out,
            no_prompt_mask=no_prompt_mask,
        )
