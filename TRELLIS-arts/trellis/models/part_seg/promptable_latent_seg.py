"""Promptable discriminative part SS-latent segmentation network."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .point_mask_encoder import PointMaskEncoder, PointMaskEncoderOutput

try:
    import spconv.pytorch as spconv
except Exception:  # pragma: no cover - token mode should not require spconv.
    spconv = None


def _nvtx_range(name: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return torch.autograd.profiler.record_function(name)

__all__ = [
    "MaskEncoder2D",
    "PointMaskEncoder",
    "PromptablePartLatentSegNet",
    "semantic_classes_from_ckpt",
    "semantic_classes_from_state",
    "voxel_embedding_dim_from_ckpt",
    "voxel_embedding_dim_from_state",
]


def semantic_classes_from_state(state: dict[str, torch.Tensor] | None) -> int:
    if not isinstance(state, dict):
        return 0
    weight = state.get("semantic_head.weight")
    if torch.is_tensor(weight) and weight.dim() == 2:
        return int(weight.shape[0])
    return 0


def semantic_classes_from_ckpt(ckpt: dict[str, Any]) -> int:
    return semantic_classes_from_state(ckpt.get("model") if isinstance(ckpt, dict) else None)


def voxel_embedding_dim_from_state(state: dict[str, torch.Tensor] | None) -> int:
    if not isinstance(state, dict):
        return 0
    weight = state.get("voxel_embed_out.weight")
    if torch.is_tensor(weight) and weight.dim() == 2:
        return int(weight.shape[0])
    return 0


def voxel_embedding_dim_from_ckpt(ckpt: dict[str, Any]) -> int:
    state_dim = voxel_embedding_dim_from_state(ckpt.get("model") if isinstance(ckpt, dict) else None)
    if state_dim > 0:
        return state_dim
    args = dict(ckpt.get("args") or {}) if isinstance(ckpt, dict) else {}
    metadata = dict(ckpt.get("metadata") or {}) if isinstance(ckpt, dict) else {}
    return max(0, int(args.get("voxel_embedding_dim", metadata.get("voxel_embedding_dim", 0)) or 0))


def _trunc_normal(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(child.weight, std=0.02)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _sincos_2d(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D sincos dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32) / float(dim // 4)
    omega = 1.0 / (10000.0 ** omega)
    enc = []
    for coord in (x.reshape(-1).float(), y.reshape(-1).float()):
        scaled = coord[:, None] * omega[None, :]
        enc.extend([scaled.sin(), scaled.cos()])
    return torch.cat(enc, dim=1)


class MaskEncoder2D(nn.Module):
    def __init__(self, *, dim: int = 256, num_views: int = 4, mask_size: int = 512) -> None:
        super().__init__()
        if mask_size != 512:
            raise ValueError("MaskEncoder2D is specified for 512x512 masks")
        self.dim = int(dim)
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )
        self.proj = nn.Conv2d(128, self.dim, 1)
        self.view_emb = nn.Parameter(torch.zeros(self.num_views, self.dim))
        self.register_buffer("pos2d", _sincos_2d(32, 32, self.dim), persistent=False)
        _trunc_normal(self)
        nn.init.trunc_normal_(self.view_emb, std=0.02)

    def forward(self, masks2d: torch.Tensor) -> torch.Tensor:
        if masks2d.dim() != 4:
            raise ValueError(f"masks2d expected [B,V,512,512], got {tuple(masks2d.shape)}")
        b, v, h, w = masks2d.shape
        if v != self.num_views or h != self.mask_size or w != self.mask_size:
            raise ValueError(f"masks2d expected [B,{self.num_views},{self.mask_size},{self.mask_size}], got {tuple(masks2d.shape)}")
        feat = self.net(masks2d.reshape(b * v, 1, h, w).float())
        feat = self.proj(feat).flatten(2).transpose(1, 2).contiguous()
        pos = self.pos2d.to(device=feat.device, dtype=feat.dtype).view(1, 1024, self.dim)
        view = self.view_emb.to(dtype=feat.dtype).view(1, v, 1, self.dim)
        feat = feat.view(b, v, 1024, self.dim) + pos.view(1, 1, 1024, self.dim) + view
        return feat.reshape(b, v * 1024, self.dim)


class LocalConv(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.channels_last_3d = False
        self.depthwise = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.pointwise = nn.Conv3d(dim, dim, 1)
        _trunc_normal(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        if n != 4096:
            raise ValueError(f"LocalConv expects 4096 cells, got {n}")
        y = x.transpose(1, 2).reshape(b, c, 16, 16, 16)
        if bool(self.channels_last_3d):
            y = y.contiguous(memory_format=torch.channels_last_3d)
        y = self.pointwise(self.depthwise(y))
        return y.reshape(b, c, n).transpose(1, 2).contiguous()


class TrunkBlock(nn.Module):
    def __init__(self, *, dim: int = 256, heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(dim)
        self.local = LocalConv(dim)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        _trunc_normal(self)

    def forward(
        self,
        x: torch.Tensor,
        mask_tokens: torch.Tensor,
        *,
        mask_token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.local(self.local_norm(x))
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + y
        kv = self.cross_kv_norm(mask_tokens)
        y, _ = self.cross_attn(
            self.cross_norm(x),
            kv,
            kv,
            key_padding_mask=mask_token_padding_mask,
            need_weights=False,
        )
        x = x + y
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SparseTokenBlock(nn.Module):
    def __init__(self, *, dim: int = 256, heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        _trunc_normal(self)

    def forward(
        self,
        x: torch.Tensor,
        mask_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        mask_token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = self.cross_kv_norm(mask_tokens)
        y, _ = self.cross_attn(
            self.cross_norm(x),
            kv,
            kv,
            key_padding_mask=mask_token_padding_mask,
            need_weights=False,
        )
        x = x + y
        x = x + self.mlp(self.mlp_norm(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class SpconvRefineBlock(nn.Module):
    def __init__(self, *, dim: int = 256, indice_key: str) -> None:
        super().__init__()
        if spconv is None:
            raise RuntimeError("refine_mode='spconv' requires spconv.pytorch")
        self.conv = spconv.SubMConv3d(dim, dim, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        if hasattr(self.conv, "weight"):
            nn.init.trunc_normal_(self.conv.weight, std=0.02)

    def forward(self, x):
        y = self.conv(x)
        feat = x.features + self.act(self.norm(y.features))
        return y.replace_feature(feat)


def _fourier_3d(coords: torch.Tensor, num_bands: int = 32) -> torch.Tensor:
    """Fourier encode normalized voxel coords in [-1, 1]."""
    if coords.numel() == 0:
        return coords.new_zeros((0, 3 * 2 * int(num_bands) + 3))
    bands = 2.0 ** torch.arange(int(num_bands), device=coords.device, dtype=coords.dtype)
    scaled = coords.unsqueeze(-1) * bands.view(1, 1, -1) * math.pi
    enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(1)
    return torch.cat([coords, enc], dim=-1)


class PromptablePartLatentSegNet(nn.Module):
    """Predict a prompt-selected part latent from global SS latent and 2D masks."""

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        dim: int = 256,
        num_views: int = 4,
        depth: int = 6,
        head_depth: int = 2,
        heads: int = 8,
        mask_size: int = 512,
        mask_prior: float = 0.01,
        use_xyz: bool = True,
        use_voxel_head: bool = False,
        voxel_depth: int = 3,
        mask_encoder: str = "cnn_grid",
        point_k_boundary: int = 32,
        point_k_interior: int = 32,
        point_resample_points: bool = False,
        semantic_classes: int = 0,
        voxel_embedding_dim: int = 0,
        refine_mode: str = "token",
        spconv_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.dim = int(dim)
        self.depth = int(depth)
        self.head_depth = int(head_depth)
        self.use_xyz = bool(use_xyz)
        self.use_voxel_head = bool(use_voxel_head)
        self.channels_last_3d = False
        self.refine_mode = str(refine_mode)
        if self.refine_mode not in {"token", "spconv"}:
            raise ValueError(f"unknown refine_mode={refine_mode!r}; expected 'token' or 'spconv'")
        self.semantic_classes = int(semantic_classes)
        self.voxel_embedding_dim = max(0, int(voxel_embedding_dim))
        self.mask_encoder_name = str(mask_encoder)
        if self.mask_encoder_name == "cnn_grid":
            self.mask_encoder = MaskEncoder2D(dim=self.dim, num_views=num_views, mask_size=mask_size)
        elif self.mask_encoder_name == "fg_points":
            self.mask_encoder = PointMaskEncoder(
                dim=self.dim,
                num_views=num_views,
                mask_size=mask_size,
                k_boundary=point_k_boundary,
                k_interior=point_k_interior,
                resample_points=point_resample_points,
            )
        else:
            raise ValueError(f"unknown mask_encoder {mask_encoder!r}; expected 'cnn_grid' or 'fg_points'")
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, 16),
                torch.linspace(-1.0, 1.0, 16),
                torch.linspace(-1.0, 1.0, 16),
                indexing="ij",
            ),
            dim=0,
        )
        self.register_buffer("coords3d", coords, persistent=False)
        self.stem = nn.Conv3d(self.latent_channels + (3 if self.use_xyz else 0), self.dim, 1)
        self.pos3d = nn.Parameter(torch.zeros(4096, self.dim))
        self.blocks = nn.ModuleList([TrunkBlock(dim=self.dim, heads=heads) for _ in range(self.depth)])
        self.head1_norm = nn.LayerNorm(self.dim)
        self.head1 = nn.Linear(self.dim, 1)
        prior = min(max(float(mask_prior), 1.0e-6), 1.0 - 1.0e-6)
        nn.init.constant_(self.head1.bias, math.log(prior / (1.0 - prior)))

        self.m_emb = nn.Linear(1, self.dim)
        self.head2_in = nn.Linear(self.dim * 2, self.dim)
        self.head2_blocks = nn.ModuleList([TrunkBlock(dim=self.dim, heads=heads) for _ in range(self.head_depth)])
        self.head2_norm = nn.LayerNorm(self.dim)
        self.delta = nn.Linear(self.dim, self.latent_channels)

        _trunc_normal(self.stem)
        nn.init.trunc_normal_(self.pos3d, std=0.02)
        _trunc_normal(self.m_emb)
        _trunc_normal(self.head2_in)
        _trunc_normal(self.delta)

        patch_offsets = torch.stack(
            torch.meshgrid(torch.arange(5), torch.arange(5), torch.arange(5), indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        self.register_buffer("patch5_offsets", patch_offsets.long(), persistent=False)
        if self.use_voxel_head:
            self.voxel_pos = nn.Linear(3 + 3 * 2 * 32, self.dim)
            if self.refine_mode == "token":
                self.voxel_patch = nn.Linear(125, self.dim)
                self.voxel_blocks = nn.ModuleList([SparseTokenBlock(dim=self.dim, heads=heads) for _ in range(int(voxel_depth))])
            else:
                if spconv is None:
                    raise RuntimeError("refine_mode='spconv' requires spconv.pytorch")
                self.spconv_refine = nn.ModuleList(
                    [
                        SpconvRefineBlock(dim=self.dim, indice_key=f"partseg_refine_{idx}")
                        for idx in range(int(spconv_depth))
                    ]
                )
            self.voxel_norm = nn.LayerNorm(self.dim)
            self.voxel_out = nn.Linear(self.dim, 1)
            _trunc_normal(self.voxel_pos)
            if self.refine_mode == "token":
                _trunc_normal(self.voxel_patch)
            _trunc_normal(self.voxel_out)
            if self.voxel_embedding_dim > 0:
                self.voxel_embed_out = nn.Linear(self.dim, self.voxel_embedding_dim)
                _trunc_normal(self.voxel_embed_out)
        if self.semantic_classes > 0:
            self.semantic_norm = nn.LayerNorm(self.dim)
            self.semantic_head = nn.Linear(self.dim, self.semantic_classes)
            _trunc_normal(self.semantic_head)

    def _encode_masks(self, masks2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encoded = self.mask_encoder(masks2d)
        if isinstance(encoded, PointMaskEncoderOutput):
            return encoded.tokens, encoded.key_padding_mask, encoded.no_prompt_mask
        return encoded, None, None

    def encode_cells(
        self,
        z_global: torch.Tensor,
        mask_tokens: torch.Tensor,
        *,
        mask_token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_global.dim() != 5 or tuple(z_global.shape[1:]) != (self.latent_channels, 16, 16, 16):
            raise ValueError(f"z_global expected [B,{self.latent_channels},16,16,16], got {tuple(z_global.shape)}")
        b = z_global.shape[0]
        if self.use_xyz:
            coords = self.coords3d.to(device=z_global.device, dtype=z_global.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
            stem_in = torch.cat([z_global.float(), coords.float()], dim=1)
        else:
            stem_in = z_global.float()
        # Keep the stem weight on the default layout; only the LocalConv
        # blocks opt into channels-last activation layout internally.
        x = self.stem(stem_in).flatten(2).transpose(1, 2).contiguous()
        x = x + self.pos3d.to(device=x.device, dtype=x.dtype).view(1, 4096, self.dim)
        for block in self.blocks:
            x = block(x, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
        if x.shape != (b, 4096, self.dim):
            raise RuntimeError(f"unexpected cell feature shape {tuple(x.shape)}")
        return x

    def forward(
        self,
        z_global: torch.Tensor,
        masks2d: torch.Tensor,
        empty_code: torch.Tensor | None = None,
        *,
        m_override: torch.Tensor | None = None,
        candidate_cells: torch.Tensor | None = None,
        full_occ: torch.Tensor | None = None,
        max_voxels_per_sample: int = 0,
    ) -> dict[str, torch.Tensor]:
        if candidate_cells is not None:
            if full_occ is None:
                raise ValueError("full_occ is required when candidate_cells is provided")
            return self.forward_voxels(
                z_global,
                masks2d,
                candidate_cells,
                full_occ,
                max_voxels_per_sample=max_voxels_per_sample,
            )
        if empty_code is None:
            raise ValueError("empty_code is required for latent forward")
        with _nvtx_range("partseg/mask_encode"):
            mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_masks(masks2d)
        with _nvtx_range("partseg/encode_cells"):
            feat = self.encode_cells(z_global, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
        m_logit = self.head1(self.head1_norm(feat)).squeeze(-1)
        semantic_logits = None
        if self.semantic_classes > 0:
            semantic_logits = self.semantic_head(self.semantic_norm(feat.mean(dim=1)))
        if m_override is None:
            m_flat = m_logit.sigmoid()
        else:
            if m_override.dim() == 4:
                m_flat = m_override.reshape(m_override.shape[0], 4096).to(device=z_global.device, dtype=z_global.dtype)
            elif m_override.dim() == 2:
                m_flat = m_override.to(device=z_global.device, dtype=z_global.dtype)
            else:
                raise ValueError(f"m_override expected [B,16,16,16] or [B,4096], got {tuple(m_override.shape)}")
            if m_flat.shape != m_logit.shape:
                raise ValueError(f"m_override shape {tuple(m_flat.shape)} does not match logits {tuple(m_logit.shape)}")

        m_embedding = self.m_emb(m_flat.unsqueeze(-1).float())
        h = self.head2_in(torch.cat([feat, m_embedding], dim=-1))
        for block in self.head2_blocks:
            h = block(h, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
        delta_flat = self.delta(self.head2_norm(h))
        delta = delta_flat.transpose(1, 2).reshape(z_global.shape[0], self.latent_channels, 16, 16, 16)
        m = m_flat.view(z_global.shape[0], 1, 16, 16, 16)
        empty = empty_code.to(device=z_global.device, dtype=z_global.dtype)
        if empty.dim() == 4:
            empty = empty.unsqueeze(0)
        if tuple(empty.shape[-4:]) != (self.latent_channels, 16, 16, 16):
            raise ValueError(f"empty_code expected [8,16,16,16] or [B,8,16,16,16], got {tuple(empty_code.shape)}")
        part_latent = m * (z_global + delta) + (1.0 - m) * empty
        return {
            "mask_tokens": mask_tokens,
            "no_prompt_mask": no_prompt_mask,
            "features": feat,
            "m_logit": m_logit,
            "semantic_logits": semantic_logits,
            "m_prob": m_logit.sigmoid(),
            "m_used": m_flat,
            "delta": delta,
            "part_latent": part_latent,
        }

    def forward_voxels(
        self,
        z_global: torch.Tensor,
        masks2d: torch.Tensor,
        candidate_cells: torch.Tensor,
        full_occ: torch.Tensor,
        *,
        max_voxels_per_sample: int = 0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if not self.use_voxel_head:
            raise RuntimeError("PromptablePartLatentSegNet was created without use_voxel_head=True")
        if candidate_cells.dim() != 4 or tuple(candidate_cells.shape[1:]) != (16, 16, 16):
            raise ValueError(f"candidate_cells expected [B,16,16,16], got {tuple(candidate_cells.shape)}")
        if full_occ.dim() != 5 or tuple(full_occ.shape[1:]) != (1, 64, 64, 64):
            raise ValueError(f"full_occ expected [B,1,64,64,64], got {tuple(full_occ.shape)}")

        mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_masks(masks2d)
        feat = self.encode_cells(z_global, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
        m_logit = self.head1(self.head1_norm(feat)).squeeze(-1)
        semantic_logits = None
        if self.semantic_classes > 0:
            semantic_logits = self.semantic_head(self.semantic_norm(feat.mean(dim=1)))

        bsz = z_global.shape[0]
        with _nvtx_range("partseg/voxel_candidates"):
            cell_mask64 = candidate_cells.bool().unsqueeze(1)
            cell_mask64 = cell_mask64.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)
            valid64 = (full_occ > 0.5) & cell_mask64

        coords_list: list[torch.Tensor] = []
        valid_list: list[torch.Tensor] = []
        if self.refine_mode == "token":
            padded_occ = F.pad(full_occ.float(), (2, 2, 2, 2, 2, 2))
            offsets = self.patch5_offsets.to(device=z_global.device)
        with _nvtx_range("partseg/voxel_token_pack"):
            packed_sample: list[torch.Tensor] = []
            packed_pos: list[torch.Tensor] = []
            packed_cell: list[torch.Tensor] = []
            packed_coords: list[torch.Tensor] = []
            for idx in range(bsz):
                coords = torch.nonzero(valid64[idx, 0], as_tuple=False).long()
                if coords.numel() == 0:
                    coords = torch.zeros((1, 3), dtype=torch.long, device=z_global.device)
                    valid_token = torch.zeros((1,), dtype=torch.bool, device=z_global.device)
                else:
                    if int(max_voxels_per_sample) > 0 and coords.shape[0] > int(max_voxels_per_sample):
                        # Deterministic uniform stride cap. This bounds the padded
                        # [B, max_len, D] token tensor without introducing per-step
                        # sampling noise or touching the full 64^3 occupancy target.
                        keep = int(max_voxels_per_sample)
                        ids = torch.linspace(0, coords.shape[0] - 1, keep, device=coords.device).round().long()
                        coords = coords.index_select(0, ids)
                    valid_token = torch.ones((coords.shape[0],), dtype=torch.bool, device=z_global.device)

                cell = torch.div(coords.clamp(0, 63), 4, rounding_mode="floor")
                flat_cell = cell[:, 0] * 256 + cell[:, 1] * 16 + cell[:, 2]
                valid_coords = coords[valid_token]
                if valid_coords.numel() > 0:
                    n_valid = int(valid_coords.shape[0])
                    packed_sample.append(torch.full((n_valid,), idx, dtype=torch.long, device=z_global.device))
                    packed_pos.append(torch.arange(n_valid, dtype=torch.long, device=z_global.device))
                    packed_cell.append(flat_cell[valid_token])
                    packed_coords.append(valid_coords)
                coords_list.append(coords)
                valid_list.append(valid_token)

            lengths = [int(coords.shape[0]) for coords in coords_list]
            max_len = max(lengths)
            token_dtype = feat.dtype
            tokens = torch.zeros((bsz, max_len, self.dim), device=z_global.device, dtype=token_dtype)
            pad_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=z_global.device)
            for idx, valid in enumerate(valid_list):
                pad_mask[idx, : lengths[idx]] = ~valid
                if not bool(valid.any()):
                    pad_mask[idx, 0] = False
            if packed_coords:
                sample_idx = torch.cat(packed_sample, dim=0)
                token_pos = torch.cat(packed_pos, dim=0)
                flat_cell = torch.cat(packed_cell, dim=0)
                coords = torch.cat(packed_coords, dim=0)
                cell_feat = feat[sample_idx, flat_cell]
                norm_coords = coords.to(dtype=z_global.dtype) / 63.0 * 2.0 - 1.0
                packed_tokens = cell_feat + self.voxel_pos(_fourier_3d(norm_coords.float()).to(dtype=cell_feat.dtype))
                if self.refine_mode == "token":
                    patch_idx = coords[:, None, :] + offsets[None, :, :]
                    patch = padded_occ[
                        sample_idx.view(-1, 1),
                        0,
                        patch_idx[..., 0],
                        patch_idx[..., 1],
                        patch_idx[..., 2],
                    ]
                    packed_tokens = packed_tokens + self.voxel_patch(patch.to(dtype=cell_feat.dtype))
                tokens[sample_idx, token_pos] = packed_tokens.to(dtype=tokens.dtype)
            token_list = [tokens[idx, : lengths[idx]] for idx in range(bsz)]

        if self.refine_mode == "token":
            with _nvtx_range("partseg/voxel_refine_token"):
                h = tokens
                for block in self.voxel_blocks:
                    h = block(
                        h,
                        mask_tokens,
                        key_padding_mask=pad_mask,
                        mask_token_padding_mask=mask_token_padding_mask,
                    )
        else:
            if spconv is None:
                raise RuntimeError("refine_mode='spconv' requires spconv.pytorch")
            flat_features: list[torch.Tensor] = []
            flat_indices: list[torch.Tensor] = []
            flat_slices: list[tuple[int, int, int]] = []
            cursor = 0
            for idx, token in enumerate(token_list):
                valid = valid_list[idx]
                if not bool(valid.any()):
                    flat_slices.append((idx, -1, -1))
                    continue
                n_valid = int(valid.sum().item())
                flat_features.append(token[valid].to(dtype=tokens.dtype))
                batch_col = torch.full((n_valid, 1), idx, dtype=torch.int32, device=z_global.device)
                flat_indices.append(torch.cat([batch_col, coords_list[idx][valid].to(dtype=torch.int32)], dim=1))
                flat_slices.append((idx, cursor, cursor + n_valid))
                cursor += n_valid
            if flat_features:
                features = torch.cat(flat_features, dim=0).contiguous()
                indices = torch.cat(flat_indices, dim=0).contiguous()
                sparse = spconv.SparseConvTensor(features, indices, spatial_shape=[64, 64, 64], batch_size=bsz)
                for block in self.spconv_refine:
                    sparse = block(sparse)
                refined = sparse.features
                h = tokens.new_zeros((bsz, max_len, self.dim))
                for idx, start, end in flat_slices:
                    if start < 0:
                        continue
                    h[idx, : end - start] = refined[start:end].to(dtype=h.dtype)
            else:
                h = tokens.new_zeros((bsz, max_len, self.dim))
        with _nvtx_range("partseg/voxel_output"):
            logits = self.voxel_out(self.voxel_norm(h)).squeeze(-1)
            logits = logits.masked_fill(pad_mask, -30.0)
        out = {
            "mask_tokens": mask_tokens,
            "no_prompt_mask": no_prompt_mask,
            "features": feat,
            "m_logit": m_logit,
            "semantic_logits": semantic_logits,
            "voxel_logits": logits,
            "voxel_pad_mask": pad_mask,
            "voxel_coords": coords_list,
        }
        if self.voxel_embedding_dim > 0:
            embeds = F.normalize(self.voxel_embed_out(self.voxel_norm(h)).float(), dim=-1, eps=1.0e-6)
            embeds = embeds.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            out["voxel_embeddings"] = embeds
        return out
