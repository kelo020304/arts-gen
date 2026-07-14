"""Promptable discriminative part SS-latent segmentation network."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .point_mask_encoder import PointMaskEncoder, PointMaskEncoderOutput

try:
    import spconv.pytorch as spconv
except Exception:  # pragma: no cover - token mode should not require spconv.
    spconv = None


def _nvtx_range(name: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return torch.autograd.profiler.record_function(name)


def _spconv_algo():
    if spconv is None:
        return None
    name = os.environ.get("SPCONV_ALGO", "auto").strip().lower()
    if name == "native":
        return spconv.ConvAlgo.Native
    if name == "implicit_gemm":
        return spconv.ConvAlgo.MaskImplicitGemm
    return None

__all__ = [
    "MaskEncoder2D",
    "PointMaskEncoder",
    "PromptablePartLatentSegNet",
    "semantic_classes_from_ckpt",
    "semantic_classes_from_state",
    "voxel_embedding_dim_from_ckpt",
    "voxel_embedding_dim_from_state",
    "joint_local_mode_from_ckpt",
    "joint_local_depth_from_ckpt",
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


def joint_local_mode_from_ckpt(ckpt: dict[str, Any]) -> str:
    args = dict(ckpt.get("args") or {}) if isinstance(ckpt, dict) else {}
    raw_state = ckpt.get("model") if isinstance(ckpt, dict) else None
    keys = [str(key).removeprefix("module.") for key in raw_state] if isinstance(raw_state, dict) else []
    if any(key.startswith("joint_local_graph.") for key in keys):
        return "edge_graph"
    if any(key.startswith("joint_local_post.") for key in keys):
        return "post_spconv"
    mode = str(args.get("joint_local_mode", "none"))
    return mode if mode in {"none", "post_spconv", "edge_graph"} else "none"


def joint_local_depth_from_ckpt(ckpt: dict[str, Any]) -> int:
    args = dict(ckpt.get("args") or {}) if isinstance(ckpt, dict) else {}
    raw_state = ckpt.get("model") if isinstance(ckpt, dict) else None
    keys = [str(key).removeprefix("module.") for key in raw_state] if isinstance(raw_state, dict) else []
    mode = joint_local_mode_from_ckpt(ckpt)
    prefix = "joint_local_graph." if mode == "edge_graph" else "joint_local_post."
    indices = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    if indices:
        return max(indices) + 1
    return max(0, int(args.get("joint_local_depth", 2) or 0))


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
        self.conv = spconv.SubMConv3d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
            algo=_spconv_algo(),
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        if hasattr(self.conv, "weight"):
            nn.init.trunc_normal_(self.conv.weight, std=0.02)

    def forward(self, x):
        y = self.conv(x)
        feat = x.features + self.act(self.norm(y.features))
        return y.replace_feature(feat)


class GatedSpconvRefineBlock(nn.Module):
    """Identity-initialized local sparse refinement for joint segmentation."""

    def __init__(self, *, dim: int = 256, indice_key: str) -> None:
        super().__init__()
        if spconv is None:
            raise RuntimeError("joint local refinement requires spconv.pytorch")
        self.conv = spconv.SubMConv3d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
            algo=_spconv_algo(),
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if hasattr(self.conv, "weight"):
            nn.init.trunc_normal_(self.conv.weight, std=0.02)

    def forward(self, x):
        y = self.conv(x)
        delta = self.act(self.norm(y.features))
        feat = x.features + self.gate.to(dtype=delta.dtype) * delta
        return x.replace_feature(feat)


class GatedEdgeGraphRefineBlock(nn.Module):
    """Identity-initialized six-neighbor feature-gated graph refinement."""

    def __init__(self, *, dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0), dtype=torch.float32))
        self.edge_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        _trunc_normal(self.value)
        _trunc_normal(self.out)

    def forward(self, h: torch.Tensor, edge_a: torch.Tensor, edge_b: torch.Tensor) -> torch.Tensor:
        if edge_a.numel() == 0:
            return h
        u = self.norm(h)
        unit = F.normalize(u.float(), dim=-1, eps=1.0e-6)
        scale = self.logit_scale.float().exp().clamp(0.25, 32.0)
        edge_weight = torch.sigmoid(
            scale * (unit[edge_a] * unit[edge_b]).sum(dim=-1) + self.edge_bias.float()
        ).to(dtype=h.dtype)
        value = self.value(u)
        aggregate = torch.zeros_like(value)
        degree = h.new_zeros((h.shape[0],))
        aggregate.index_add_(0, edge_a, edge_weight.unsqueeze(-1) * value[edge_b])
        aggregate.index_add_(0, edge_b, edge_weight.unsqueeze(-1) * value[edge_a])
        degree.index_add_(0, edge_a, edge_weight)
        degree.index_add_(0, edge_b, edge_weight)
        has_neighbor = degree > 0
        mean_neighbor = aggregate / degree.clamp_min(1.0e-6).unsqueeze(-1)
        delta = self.out(mean_neighbor - value)
        delta = torch.where(has_neighbor.unsqueeze(-1), delta, torch.zeros_like(delta))
        return h + self.gate.to(dtype=delta.dtype) * delta


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
        use_body_prompt: bool = False,
        negative_prompt_channel: bool = False,
        use_checkpoint: bool = False,
        joint_local_mode: str = "none",
        joint_local_depth: int = 2,
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
        self.use_body_prompt = bool(use_body_prompt)
        self.negative_prompt_channel = bool(negative_prompt_channel)
        self.use_checkpoint = bool(use_checkpoint)
        self.joint_local_mode = str(joint_local_mode)
        if self.joint_local_mode not in {"none", "post_spconv", "edge_graph"}:
            raise ValueError(
                f"unknown joint_local_mode={joint_local_mode!r}; expected none, post_spconv, or edge_graph"
            )
        self.joint_local_depth = max(0, int(joint_local_depth))
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
        if self.use_body_prompt:
            self.body_prompt = nn.Parameter(torch.zeros(1, self.dim))
        if self.negative_prompt_channel:
            self.negative_prompt_proj = nn.Linear(self.dim, self.dim, bias=False)
            nn.init.zeros_(self.negative_prompt_proj.weight)
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
        if self.use_body_prompt:
            nn.init.trunc_normal_(self.body_prompt, std=0.02)
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
            if self.use_body_prompt:
                self.joint_query_norm = nn.LayerNorm(self.dim)
                self.joint_query_self = nn.MultiheadAttention(self.dim, heads, batch_first=True)
                self.joint_query_mlp_norm = nn.LayerNorm(self.dim)
                self.joint_query_mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Linear(self.dim * 4, self.dim),
                )
                self.joint_summary = nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim),
                    nn.GELU(),
                )
                self.joint_voxel_norm = nn.LayerNorm(self.dim)
                self.joint_kv_norm = nn.LayerNorm(self.dim)
                self.joint_voxel_cross = nn.MultiheadAttention(self.dim, heads, batch_first=True)
                self.joint_voxel_mlp_norm = nn.LayerNorm(self.dim)
                self.joint_voxel_mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Linear(self.dim * 4, self.dim),
                )
                self.joint_score_voxel_norm = nn.LayerNorm(self.dim)
                self.joint_score_query_norm = nn.LayerNorm(self.dim)
                self.joint_score_voxel = nn.Linear(self.dim, self.dim, bias=False)
                self.joint_score_query = nn.Linear(self.dim, self.dim, bias=False)
                self.joint_logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
                _trunc_normal(self.joint_query_mlp)
                _trunc_normal(self.joint_summary)
                _trunc_normal(self.joint_voxel_mlp)
                _trunc_normal(self.joint_score_voxel)
                _trunc_normal(self.joint_score_query)
                if self.joint_local_mode == "post_spconv":
                    if spconv is None:
                        raise RuntimeError("joint local refinement requires spconv.pytorch")
                    if self.joint_local_depth <= 0:
                        raise ValueError("joint_local_depth must be > 0 when joint_local_mode is enabled")
                    self.joint_local_post = nn.ModuleList(
                        [
                            GatedSpconvRefineBlock(dim=self.dim, indice_key=f"partseg_joint_post_{idx}")
                            for idx in range(self.joint_local_depth)
                        ]
                    )
                elif self.joint_local_mode == "edge_graph":
                    if self.joint_local_depth <= 0:
                        raise ValueError("joint_local_depth must be > 0 when joint_local_mode is enabled")
                    self.joint_local_graph = nn.ModuleList(
                        [GatedEdgeGraphRefineBlock(dim=self.dim) for _ in range(self.joint_local_depth)]
                    )
        if self.semantic_classes > 0:
            self.semantic_norm = nn.LayerNorm(self.dim)
            self.semantic_head = nn.Linear(self.dim, self.semantic_classes)
            _trunc_normal(self.semantic_head)

    def _encode_masks(self, masks2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encoded = self.mask_encoder(masks2d)
        if isinstance(encoded, PointMaskEncoderOutput):
            return encoded.tokens, encoded.key_padding_mask, encoded.no_prompt_mask
        return encoded, None, None

    def _encode_body_prompt(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_body_prompt or not hasattr(self, "body_prompt"):
            raise RuntimeError("body prompt requested but model was created with use_body_prompt=False")
        token = self.body_prompt.to(device=device, dtype=dtype).view(1, 1, self.dim).expand(int(batch_size), -1, -1)
        return token, None, None

    def _encode_mixed_prompts(
        self,
        masks2d: torch.Tensor,
        *,
        negative_masks2d: torch.Tensor | None = None,
        use_body_prompt: bool = False,
        body_prompt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if bool(use_body_prompt):
            return self._encode_body_prompt(
                masks2d.shape[0],
                device=masks2d.device,
                dtype=masks2d.dtype,
            )
        mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_masks(masks2d)
        if negative_masks2d is not None:
            if not self.negative_prompt_channel or not hasattr(self, "negative_prompt_proj"):
                raise RuntimeError("negative prompt masks were provided but model was created with negative_prompt_channel=False")
            neg = torch.as_tensor(negative_masks2d, device=masks2d.device, dtype=masks2d.dtype)
            if tuple(neg.shape) != tuple(masks2d.shape):
                raise ValueError(f"negative_masks2d shape {tuple(neg.shape)} must match masks2d {tuple(masks2d.shape)}")
            neg_tokens, neg_padding_mask, _neg_no_prompt = self._encode_masks(neg)
            neg_summary = self._pool_prompt_queries(neg_tokens, neg_padding_mask)
            neg_context = self.negative_prompt_proj(neg_summary).to(dtype=mask_tokens.dtype)
            mask_tokens = mask_tokens + neg_context.unsqueeze(1)
        if self.use_body_prompt and hasattr(self, "body_prompt"):
            mask_tokens = mask_tokens + self.body_prompt.to(device=mask_tokens.device, dtype=mask_tokens.dtype).sum() * 0.0
        if body_prompt_mask is None:
            return mask_tokens, mask_token_padding_mask, no_prompt_mask
        body_mask = torch.as_tensor(body_prompt_mask, device=masks2d.device, dtype=torch.bool).flatten()
        if body_mask.numel() != masks2d.shape[0]:
            raise ValueError(f"body_prompt_mask expected [{masks2d.shape[0]}], got {tuple(body_mask.shape)}")
        if not bool(body_mask.any()):
            return mask_tokens, mask_token_padding_mask, no_prompt_mask
        if not self.use_body_prompt or not hasattr(self, "body_prompt"):
            raise RuntimeError("body prompt requested but model was created with use_body_prompt=False")
        mask_tokens = mask_tokens.clone()
        body_token = self.body_prompt.to(device=mask_tokens.device, dtype=mask_tokens.dtype).view(1, self.dim)
        mask_tokens[body_mask] = 0.0
        mask_tokens[body_mask, 0] = body_token
        if mask_token_padding_mask is None:
            mask_token_padding_mask = torch.zeros(
                (masks2d.shape[0], mask_tokens.shape[1]),
                dtype=torch.bool,
                device=masks2d.device,
            )
        else:
            mask_token_padding_mask = mask_token_padding_mask.clone()
        mask_token_padding_mask[body_mask] = True
        mask_token_padding_mask[body_mask, 0] = False
        if no_prompt_mask is not None:
            no_prompt_mask = no_prompt_mask.clone()
            no_prompt_mask[body_mask] = False
        return mask_tokens, mask_token_padding_mask, no_prompt_mask

    def _run_trunk_block(
        self,
        block: TrunkBlock,
        x: torch.Tensor,
        mask_tokens: torch.Tensor,
        *,
        mask_token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bool(self.use_checkpoint) and self.training:
            def fn(x_in: torch.Tensor, mask_tokens_in: torch.Tensor) -> torch.Tensor:
                return block(x_in, mask_tokens_in, mask_token_padding_mask=mask_token_padding_mask)

            return checkpoint(fn, x, mask_tokens, use_reentrant=False)
        return block(x, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)

    def _run_sparse_block(
        self,
        block: SparseTokenBlock,
        x: torch.Tensor,
        mask_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        mask_token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bool(self.use_checkpoint) and self.training:
            def fn(x_in: torch.Tensor, mask_tokens_in: torch.Tensor) -> torch.Tensor:
                return block(
                    x_in,
                    mask_tokens_in,
                    key_padding_mask=key_padding_mask,
                    mask_token_padding_mask=mask_token_padding_mask,
                )

            return checkpoint(fn, x, mask_tokens, use_reentrant=False)
        return block(
            x,
            mask_tokens,
            key_padding_mask=key_padding_mask,
            mask_token_padding_mask=mask_token_padding_mask,
        )

    def _pool_prompt_queries(
        self,
        mask_tokens: torch.Tensor,
        mask_token_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask_tokens.dim() != 3:
            raise ValueError(f"mask_tokens expected [K,T,D], got {tuple(mask_tokens.shape)}")
        if mask_token_padding_mask is None:
            return mask_tokens.mean(dim=1)
        valid = ~mask_token_padding_mask.to(device=mask_tokens.device, dtype=torch.bool)
        weights = valid.to(dtype=mask_tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (mask_tokens * weights).sum(dim=1) / denom

    def _refine_joint_queries(self, queries: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "joint_query_self"):
            raise RuntimeError("joint voxel forward requires use_body_prompt=True")
        y, _ = self.joint_query_self(
            self.joint_query_norm(queries),
            self.joint_query_norm(queries),
            self.joint_query_norm(queries),
            need_weights=False,
        )
        queries = queries + y
        queries = queries + self.joint_query_mlp(self.joint_query_mlp_norm(queries))
        return queries

    def _joint_queries_from_masks(
        self,
        part_masks2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if part_masks2d.dim() != 4:
            raise ValueError(f"part_masks2d expected [K,V,H,W], got {tuple(part_masks2d.shape)}")
        if part_masks2d.shape[0] <= 0:
            raise ValueError("forward_joint_voxels requires at least one prompted part")
        if not self.use_body_prompt or not hasattr(self, "body_prompt"):
            raise RuntimeError("forward_joint_voxels requires model created with use_body_prompt=True")
        mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_masks(part_masks2d)
        part_queries = self._pool_prompt_queries(mask_tokens, mask_token_padding_mask)
        body_query = self.body_prompt.to(device=part_queries.device, dtype=part_queries.dtype).view(1, self.dim)
        queries = torch.cat([body_query, part_queries], dim=0).unsqueeze(0)
        queries = self._refine_joint_queries(queries)
        return queries, mask_tokens, mask_token_padding_mask, no_prompt_mask

    def _joint_voxel_features(
        self,
        feat: torch.Tensor,
        full_occ: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if coords.dim() != 2 or coords.shape[-1] != 3:
            raise ValueError(f"coords expected [S,3], got {tuple(coords.shape)}")
        cell = torch.div(coords.clamp(0, 63), 4, rounding_mode="floor")
        flat_cell = cell[:, 0] * 256 + cell[:, 1] * 16 + cell[:, 2]
        cell_feat = feat[0, flat_cell]
        norm_coords = coords.to(dtype=full_occ.dtype) / 63.0 * 2.0 - 1.0
        voxel_feat = cell_feat + self.voxel_pos(_fourier_3d(norm_coords.float()).to(dtype=cell_feat.dtype))
        if hasattr(self, "voxel_patch"):
            padded_occ = F.pad(full_occ.float(), (2, 2, 2, 2, 2, 2))
            offsets = self.patch5_offsets.to(device=coords.device)
            patch_idx = coords[:, None, :] + offsets[None, :, :]
            patch = padded_occ[
                torch.zeros((coords.shape[0], 1), dtype=torch.long, device=coords.device),
                0,
                patch_idx[..., 0],
                patch_idx[..., 1],
                patch_idx[..., 2],
            ]
            voxel_feat = voxel_feat + self.voxel_patch(patch.to(dtype=cell_feat.dtype))
        return voxel_feat

    def _run_joint_local_refine(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        blocks: nn.ModuleList | None,
    ) -> torch.Tensor:
        if blocks is None or len(blocks) == 0:
            return h
        if spconv is None:
            raise RuntimeError("joint local refinement requires spconv.pytorch")
        if h.dim() != 2 or h.shape[0] != coords.shape[0]:
            raise ValueError(f"joint local refine expected h [S,D] aligned with coords, got {tuple(h.shape)}")
        batch_col = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
        indices = torch.cat([batch_col, coords.to(dtype=torch.int32)], dim=1).contiguous()
        sparse = spconv.SparseConvTensor(
            h.contiguous(),
            indices,
            spatial_shape=[64, 64, 64],
            batch_size=1,
        )
        for block in blocks:
            sparse = block(sparse)
        return sparse.features

    def _joint_six_neighbor_edges(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coord_keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
        source = torch.arange(coords.shape[0], device=coords.device)
        all_a: list[torch.Tensor] = []
        all_b: list[torch.Tensor] = []
        for delta_values in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            delta = torch.tensor(delta_values, dtype=torch.long, device=coords.device)
            query = coords + delta.view(1, 3)
            in_bounds = (query < 64).all(dim=1)
            if not bool(in_bounds.any().item()):
                continue
            a = source[in_bounds]
            q_keys = query[in_bounds, 0] * 4096 + query[in_bounds, 1] * 64 + query[in_bounds, 2]
            b = torch.searchsorted(coord_keys, q_keys)
            in_range = b < coord_keys.shape[0]
            a = a[in_range]
            q_keys = q_keys[in_range]
            b = b[in_range]
            matched = coord_keys[b] == q_keys
            if bool(matched.any().item()):
                all_a.append(a[matched])
                all_b.append(b[matched])
        if not all_a:
            empty = torch.empty((0,), dtype=torch.long, device=coords.device)
            return empty, empty
        return torch.cat(all_a), torch.cat(all_b)

    def _run_joint_edge_graph_refine(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        blocks = getattr(self, "joint_local_graph", None)
        if blocks is None or len(blocks) == 0:
            return h
        edge_a, edge_b = self._joint_six_neighbor_edges(coords)
        for block in blocks:
            h = block(h, edge_a, edge_b)
        return h

    def forward_joint_voxels(
        self,
        z_global: torch.Tensor,
        part_masks2d: torch.Tensor,
        candidate_cells: torch.Tensor,
        full_occ: torch.Tensor,
        *,
        max_voxels_per_sample: int = 0,
    ) -> dict[str, torch.Tensor]:
        if not self.use_voxel_head:
            raise RuntimeError("PromptablePartLatentSegNet was created without use_voxel_head=True")
        if z_global.dim() != 5 or z_global.shape[0] != 1:
            raise ValueError(f"forward_joint_voxels expects z_global [1,C,16,16,16], got {tuple(z_global.shape)}")
        if candidate_cells.dim() != 4 or tuple(candidate_cells.shape) != (1, 16, 16, 16):
            raise ValueError(f"candidate_cells expected [1,16,16,16], got {tuple(candidate_cells.shape)}")
        if full_occ.dim() != 5 or tuple(full_occ.shape) != (1, 1, 64, 64, 64):
            raise ValueError(f"full_occ expected [1,1,64,64,64], got {tuple(full_occ.shape)}")
        with _nvtx_range("partseg/joint_query_encode"):
            queries, mask_tokens, _mask_token_padding_mask, no_prompt_mask = self._joint_queries_from_masks(part_masks2d)
        with _nvtx_range("partseg/joint_encode_cells"):
            feat = self.encode_cells(z_global, queries)
        with _nvtx_range("partseg/joint_voxel_candidates"):
            cell_mask64 = candidate_cells.bool().unsqueeze(1)
            cell_mask64 = cell_mask64.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)
            valid64 = (full_occ > 0.5) & cell_mask64
            coords = torch.nonzero(valid64[0, 0], as_tuple=False).long()
            if coords.numel() == 0:
                raise ValueError("forward_joint_voxels got empty shared candidate voxels")
            if int(max_voxels_per_sample) > 0 and coords.shape[0] > int(max_voxels_per_sample):
                keep = int(max_voxels_per_sample)
                ids = torch.linspace(0, coords.shape[0] - 1, keep, device=coords.device).round().long()
                coords = coords.index_select(0, ids)
        with _nvtx_range("partseg/joint_voxel_cross"):
            h0 = self._joint_voxel_features(feat, full_occ, coords)
            h = h0.unsqueeze(0)
            cell_summary = self.joint_summary(feat.mean(dim=1, keepdim=True))
            kv = torch.cat([queries, cell_summary], dim=1)
            y, _ = self.joint_voxel_cross(
                self.joint_voxel_norm(h),
                self.joint_kv_norm(kv),
                self.joint_kv_norm(kv),
                need_weights=False,
            )
            h = h + y
            h = h + self.joint_voxel_mlp(self.joint_voxel_mlp_norm(h))
            h = self._run_joint_local_refine(h[0], coords, getattr(self, "joint_local_post", None)).unsqueeze(0)
            h = self._run_joint_edge_graph_refine(h[0], coords).unsqueeze(0)
        with _nvtx_range("partseg/joint_voxel_output"):
            voxel_score = self.joint_score_voxel(self.joint_score_voxel_norm(h[0]))
            query_score = self.joint_score_query(self.joint_score_query_norm(queries[0]))
            voxel_score = F.normalize(voxel_score.float(), dim=-1, eps=1.0e-6)
            query_score = F.normalize(query_score.float(), dim=-1, eps=1.0e-6)
            scale = self.joint_logit_scale.float().exp().clamp(1.0, 100.0)
            logits = torch.matmul(voxel_score, query_score.transpose(0, 1)) * scale
        return {
            "joint_logits": logits,
            "joint_coords": coords,
            "joint_queries": queries,
            "mask_tokens": mask_tokens,
            "no_prompt_mask": no_prompt_mask,
            "features": feat,
            "voxel_features": h[0],
        }

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
            x = self._run_trunk_block(block, x, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
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
        negative_masks2d: torch.Tensor | None = None,
        use_body_prompt: bool = False,
        body_prompt_mask: torch.Tensor | None = None,
        joint_voxels: bool = False,
    ) -> dict[str, torch.Tensor]:
        if bool(joint_voxels):
            if candidate_cells is None:
                raise ValueError("candidate_cells is required when joint_voxels=True")
            if full_occ is None:
                raise ValueError("full_occ is required when joint_voxels=True")
            return self.forward_joint_voxels(
                z_global,
                masks2d,
                candidate_cells,
                full_occ,
                max_voxels_per_sample=max_voxels_per_sample,
            )
        if candidate_cells is not None:
            if full_occ is None:
                raise ValueError("full_occ is required when candidate_cells is provided")
            return self.forward_voxels(
                z_global,
                masks2d,
                candidate_cells,
                full_occ,
                max_voxels_per_sample=max_voxels_per_sample,
                negative_masks2d=negative_masks2d,
                use_body_prompt=bool(use_body_prompt),
                body_prompt_mask=body_prompt_mask,
            )
        if empty_code is None:
            raise ValueError("empty_code is required for latent forward")
        with _nvtx_range("partseg/mask_encode"):
            mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_mixed_prompts(
                masks2d,
                negative_masks2d=negative_masks2d,
                use_body_prompt=bool(use_body_prompt),
                body_prompt_mask=body_prompt_mask,
            )
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
            h = self._run_trunk_block(block, h, mask_tokens, mask_token_padding_mask=mask_token_padding_mask)
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
        negative_masks2d: torch.Tensor | None = None,
        use_body_prompt: bool = False,
        body_prompt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if not self.use_voxel_head:
            raise RuntimeError("PromptablePartLatentSegNet was created without use_voxel_head=True")
        if candidate_cells.dim() != 4 or tuple(candidate_cells.shape[1:]) != (16, 16, 16):
            raise ValueError(f"candidate_cells expected [B,16,16,16], got {tuple(candidate_cells.shape)}")
        if full_occ.dim() != 5 or tuple(full_occ.shape[1:]) != (1, 64, 64, 64):
            raise ValueError(f"full_occ expected [B,1,64,64,64], got {tuple(full_occ.shape)}")

        mask_tokens, mask_token_padding_mask, no_prompt_mask = self._encode_mixed_prompts(
            masks2d,
            negative_masks2d=negative_masks2d,
            use_body_prompt=bool(use_body_prompt),
            body_prompt_mask=body_prompt_mask,
        )
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
                    h = self._run_sparse_block(
                        block,
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
            logits = logits + m_logit.sum() * 0.0
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
