from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_camera_pose: bool = True,
        use_view_id_embedding: bool = False,
        num_view_embeddings: int = 4,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.use_camera_pose = bool(use_camera_pose)
        self.use_view_id_embedding = bool(use_view_id_embedding)
        self.num_view_embeddings = int(num_view_embeddings)
        if self.use_view_id_embedding and self.num_view_embeddings <= 0:
            raise ValueError(f"num_view_embeddings must be positive, got {self.num_view_embeddings}")
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        if self.use_camera_pose:
            self.view_pose_proj = nn.Linear(4, cond_channels)
        if self.use_view_id_embedding:
            self.view_id_embedding = nn.Embedding(self.num_view_embeddings, cond_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        if self.use_camera_pose:
            self.view_pose_proj.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        if self.use_camera_pose:
            self.view_pose_proj.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        if self.use_camera_pose:
            nn.init.zeros_(self.view_pose_proj.weight)
            nn.init.zeros_(self.view_pose_proj.bias)
        if self.use_view_id_embedding:
            nn.init.normal_(self.view_id_embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _add_view_id_emb(
        self,
        cond: torch.Tensor,
        view_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond.ndim == 3:
            return cond
        if cond.ndim != 4:
            raise ValueError(f"cond expected [B,N,C] or [B,V,N,C], got {tuple(cond.shape)}")
        B, V, N, C = cond.shape
        if C != self.cond_channels:
            raise ValueError(f"cond channels {C} does not match model cond_channels={self.cond_channels}")
        if self.use_view_id_embedding:
            if view_ids is None:
                view_ids = torch.arange(V, device=cond.device, dtype=torch.long)
            else:
                view_ids = view_ids.to(device=cond.device, dtype=torch.long)
            if view_ids.ndim == 1:
                if view_ids.shape[0] != V:
                    raise ValueError(f"view_ids length {view_ids.shape[0]} does not match cond views V={V}")
                view_ids = view_ids.unsqueeze(0).expand(B, -1)
            elif view_ids.ndim == 2:
                if tuple(view_ids.shape) != (B, V):
                    raise ValueError(f"view_ids shape {tuple(view_ids.shape)} must be [B,V]=[{B},{V}]")
            else:
                raise ValueError(f"view_ids expected [V] or [B,V], got {tuple(view_ids.shape)}")
            # Production SS-flow uses four ordered views. Modulo keeps tests and
            # future non-4-view diagnostics shape-robust without adding pose.
            view_ids = view_ids.remainder(self.num_view_embeddings)
            view_emb = self.view_id_embedding(view_ids).to(dtype=cond.dtype).unsqueeze(2)
            cond = cond + view_emb
        return cond.reshape(B, V * N, C).contiguous()

    def _add_camera_emb(self, cond: torch.Tensor, cam_pose: torch.Tensor | None) -> torch.Tensor:
        if cam_pose is None:
            return cond
        if not self.use_camera_pose:
            return cond
        if cond.ndim != 3:
            raise ValueError(f"cond expected [B,N,C], got {tuple(cond.shape)}")
        if cam_pose.ndim != 3 or cam_pose.shape[-1] != 4:
            raise ValueError(f"cam_pose expected [B,V,4], got {tuple(cam_pose.shape)}")
        B, N, C = cond.shape
        if cam_pose.shape[0] != B:
            raise ValueError(f"cam_pose batch {cam_pose.shape[0]} does not match cond batch {B}")
        V = int(cam_pose.shape[1])
        if V <= 0 or N % V != 0:
            raise ValueError(f"cond token count N={N} must be divisible by cam_pose views V={V}")
        if C != self.cond_channels:
            raise ValueError(f"cond channels {C} does not match model cond_channels={self.cond_channels}")
        tokens_per_view = N // V
        cond_view = cond.view(B, V, tokens_per_view, C)
        pose_emb = self.view_pose_proj(cam_pose.to(device=cond.device, dtype=cond.dtype)).unsqueeze(2)
        return (cond_view + pose_emb).view(B, N, C)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        cam_pose: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        view_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass returning velocity prediction.

        Args:
            x: [B, C, 16, 16, 16] noised latent
            t: [B] timestep
            cond: [B, N, 1024] or [B, V, N, 1024] conditioning tokens.
                A 4-D tensor is flattened into one concat cross-attention context;
                optional learnable view-id embeddings are position/order based.
            cam_pose: optional [B, V, 4] per-view relative pose features.
            attn_bias: optional additive cross-attention logits bias broadcastable
                to [B, num_heads, query_tokens, condition_tokens]
            view_ids: optional [V] or [B,V] integer view ids for 4-D cond. If
                omitted, ids are the ordered positions 0..V-1.

        Returns:
            v_pred: [B, out_channels, 16, 16, 16] velocity prediction
        """
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        cond = self._add_view_id_emb(cond, view_ids=view_ids)
        cond = self._add_camera_emb(cond, cam_pose)
        if attn_bias is not None:
            attn_bias = attn_bias.to(device=h.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, attn_bias=attn_bias)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])

        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        v_pred = unpatchify(h, self.patch_size).contiguous()

        return v_pred
