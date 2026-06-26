"""Joint part-set sparse-structure latent Rectified Flow model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.modules.attention import MultiHeadAttention
from trellis.modules.norm import LayerNorm32
from trellis.modules.spatial import patchify, unpatchify
from trellis.modules.transformer import AbsolutePositionEmbedder


__all__ = ["PartSSLatentFlowModel"]


class PartSSLatentFlowModel(nn.Module):
    """Generate K absolute target-part SS latents conditioned on global SS latent."""

    def __init__(
        self,
        resolution: int,
        latent_channels: int,
        model_channels: int,
        cond_dim: int,
        num_blocks: int,
        num_heads: int,
        patch_size: int,
        num_views: int,
        require_part_token: bool,
        use_fp16: bool,
        use_checkpoint: bool,
        max_parts: int = 8,
        num_part_query_layers: int = 2,
        part_label_vocab_size: int = 64,
        global_cond_patch_size: int = 2,
        max_part_forward_batch: int = 0,
        # Ablation switch only. The main path keeps x_t_part as the sole RF
        # state input and sends z_global through global_z_tokens. If this is
        # enabled with latent_scale != 1, callers should account for the raw
        # z_global scale when interpreting concat-vs-no-concat results.
        concat_global: bool = False,
        # === Binding-fix flags (fixes 1, 2, 7, 8). Each defaults to the legacy
        # behavior so existing configs / checkpoints reproduce bit-for-bit. ===
        # Fix 1: process all valid parts of one object in a single joint backbone
        # pass where self-attention alternates even=global-cross-part /
        # odd=within-part by reshape. When False, the legacy per-part-independent
        # loop runs (kept for ablation).
        cross_part_attention: bool = False,
        # Fix 2: add a learnable per-part identity embedding (indexed by
        # target_slot) to EVERY latent token of that part.
        token_identity_embedding: bool = False,
        # Fix 7: accept previous-x0-hat self-conditioning concatenated on the
        # channel dim before patchify. Doubles the backbone input channels.
        self_conditioning: bool = False,
        # Fix 8: learnable null per-part condition for classifier-free guidance
        # (training conditioning-dropout + inference guidance). The shared global
        # z tokens are always kept; only the per-part query + role-marked cond
        # tokens are replaced with the null embedding when dropped.
        classifier_free_guidance: bool = False,
        # Complete the overlap-pooling that was only wired to the QUERY: make the
        # target/context ROLE marking in the cross-attn KEY/VALUE soft too, reusing
        # part_token_weights (gate = w / max(w)) instead of the hard min_fg=3
        # mask_token_labels vote. De-starves sub-patch parts (which the hard vote
        # marks all-context) in the decisive conditioning path. No new params, no
        # data change, no anchor.
        soft_role_marking: bool = False,
        # Summary-token (induced / Perceiver / ISAB) cross-part interaction for
        # the EVEN (global) joint blocks. Default OFF keeps the legacy full
        # [1,K*T,C] cross-part self-attention path bit-for-bit. When ON, each
        # even block pools every part's [T,C] tokens to n_summary_tokens learned
        # summaries via a spatially-structured attention pool, mixes the summaries
        # cross-part with the block's own self_attn (over K*m << K*T tokens), then
        # broadcasts the mixed summaries back to the full tokens via cross-attn.
        # Replaces O((K*T)^2) self-attention with O(K*m*T) + O((K*m)^2). Requires
        # cross_part_attention (the joint path); only affects even blocks.
        summary_cross_part_attention: bool = False,
        # Number of summary tokens m per part. Default 64 = a coarse 4x4x4 grid so
        # each summary keeps WHERE-in-the-part it pooled from (a coarse 3D pos emb
        # over the 4^3 grid is added to the learned summary queries).
        n_summary_tokens: int = 64,
        # Flat (cross-object batched) joint forward. The serial joint path runs
        # one object's K parts per `for b in range(B)` iteration; this packs ALL
        # objects' parts into one [N_total,T,C] batch so the expensive per-part
        # ops (within-part self-attn, cross-attn, MLP) are a single batched call,
        # while only the cheap summary-token mix/broadcast stays grouped per
        # object. Mathematically identical to the serial path (gated by the
        # equivalence tests); requires cross_part_attention.
        joint_flat_batch: bool = False,
        # Mask attention position bias (B). The nested config form
        # model.mask_attention_bias={enabled,lambda_bias,eps} is accepted here so
        # old flat model configs/checkpoints still build strictly.
        mask_attention_bias: dict | None = None,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.latent_channels = int(latent_channels)
        self.num_views = int(num_views)
        self.require_part_token = bool(require_part_token)
        self.concat_global = bool(concat_global)
        self.max_parts = int(max_parts)
        self.part_label_vocab_size = int(part_label_vocab_size)
        self.global_cond_patch_size = int(global_cond_patch_size)
        self.max_part_forward_batch = int(max_part_forward_batch)
        self.cross_part_attention = bool(cross_part_attention)
        self.token_identity_embedding = bool(token_identity_embedding)
        self.self_conditioning = bool(self_conditioning)
        self.classifier_free_guidance = bool(classifier_free_guidance)
        self.soft_role_marking = bool(soft_role_marking)
        self.summary_cross_part_attention = bool(summary_cross_part_attention)
        self.n_summary_tokens = int(n_summary_tokens)
        if self.summary_cross_part_attention and not self.cross_part_attention:
            raise ValueError(
                "summary_cross_part_attention requires cross_part_attention=true "
                "(it only replaces the EVEN-block self-attention scope of the joint path)"
            )
        if self.summary_cross_part_attention and self.n_summary_tokens <= 0:
            raise ValueError("n_summary_tokens must be positive when summary_cross_part_attention=true")
        self.joint_flat_batch = bool(joint_flat_batch)
        if self.joint_flat_batch and not self.cross_part_attention:
            raise ValueError(
                "joint_flat_batch requires cross_part_attention=true "
                "(it is a batched re-expression of the joint per-object forward)"
            )
        if self.max_part_forward_batch < 0:
            raise ValueError("max_part_forward_batch must be >= 0")
        mask_bias_cfg = dict(mask_attention_bias or {})
        self.mask_attention_bias_enabled = bool(mask_bias_cfg.get("enabled", False))
        self.mask_attention_bias_lambda = float(mask_bias_cfg.get("lambda_bias", 1.0))
        self.mask_attention_bias_eps = float(mask_bias_cfg.get("eps", 1.0e-3))
        if self.mask_attention_bias_eps <= 0:
            raise ValueError("mask_attention_bias.eps must be > 0")
        self._mask_attention_bias_warned_zero_weight = False
        if self.global_cond_patch_size <= 0:
            raise ValueError("global_cond_patch_size must be positive")
        if self.resolution % self.global_cond_patch_size != 0:
            raise ValueError(
                f"resolution={self.resolution} must be divisible by "
                f"global_cond_patch_size={self.global_cond_patch_size}"
            )
        model_channels = int(model_channels)
        self.model_channels = model_channels
        self.patch_size = int(patch_size)
        if self.resolution % self.patch_size != 0:
            raise ValueError(
                f"resolution={self.resolution} must be divisible by patch_size={self.patch_size}"
            )

        self.cond_proj = nn.Linear(int(cond_dim), model_channels)
        self.part_label_emb = nn.Embedding(self.part_label_vocab_size, model_channels)
        self.part_slot_emb = nn.Embedding(self.part_label_vocab_size, model_channels)
        self.part_type_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        self.target_token_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        self.context_token_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        self.target_query_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        self.context_query_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        self.global_token_type_emb = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        # Fix 8: learnable null per-part condition (one query token + one cond
        # token) used when the per-part condition is dropped for CFG. Created ONLY
        # under classifier_free_guidance so the embedded config FULLY determines the
        # parameter set: an ablation / legacy checkpoint rebuilt from its own config
        # (CFG off, e.g. the pre-CFG 0526 run) keeps the exact legacy params and
        # loads strictly — same gating discipline as summary_cross_part_attention
        # below. Without this guard these two unconditional params are the ONLY
        # mismatch that makes a pre-CFG checkpoint fail strict load.
        if self.classifier_free_guidance:
            self.null_part_query = nn.Parameter(torch.randn(1, model_channels) * 0.02)
            self.null_cond_token = nn.Parameter(torch.randn(1, model_channels) * 0.02)
        global_patch_dim = self.latent_channels * self.global_cond_patch_size ** 3
        self.global_cond_layer = nn.Linear(global_patch_dim, model_channels)
        self.global_cond_pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        global_grid = self.resolution // self.global_cond_patch_size
        coords = torch.meshgrid(
            torch.arange(global_grid, dtype=torch.float32),
            torch.arange(global_grid, dtype=torch.float32),
            torch.arange(global_grid, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        self.register_buffer("_global_voxel_coords", coords, persistent=False)
        if int(num_part_query_layers) > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_channels,
                nhead=int(num_heads),
                dim_feedforward=model_channels * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.part_query_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(num_part_query_layers),
            )
        else:
            self.part_query_encoder = nn.Identity()
        self.part_query_norm = nn.LayerNorm(model_channels)
        # Fix 7: doubling the input channel only when self-conditioning is on
        # (the previous x0_hat is concatenated on the channel dim). concat_global
        # also doubles the channel; the two are mutually exclusive feature
        # branches and may be combined (then the backbone sees 3x channels).
        in_channel_mult = 1
        if self.concat_global:
            in_channel_mult += 1
        if self.self_conditioning:
            in_channel_mult += 1
        self.in_channel_mult = in_channel_mult
        self.backbone = SparseStructureFlowModel(
            resolution=self.resolution,
            in_channels=self.latent_channels * in_channel_mult,
            model_channels=model_channels,
            cond_channels=model_channels,
            out_channels=self.latent_channels,
            num_blocks=int(num_blocks),
            num_heads=int(num_heads),
            patch_size=int(patch_size),
            pe_mode="ape",
            use_fp16=bool(use_fp16),
            use_checkpoint=bool(use_checkpoint),
            qk_rms_norm=True,
            qk_rms_norm_cross=False,
            use_camera_pose=False,
        )

        # === Summary-token cross-part interaction (ISAB / Perceiver pooling) ===
        # Built only when the flag is on so OFF configs/checkpoints keep the exact
        # legacy parameter set. These are model-level and SHARED across all even
        # blocks (each even block still uses its OWN block.self_attn for the
        # cross-part mix; only the pool/broadcast attentions and the summary
        # queries are shared here).
        if self.summary_cross_part_attention:
            m = self.n_summary_tokens
            # Learned summary queries arranged as a COARSE 3D grid so each query
            # owns a coarse spatial region of the part (preserves WHERE-in-part);
            # a coarse 3D positional embedding over the grid is added at use time.
            self.summary_queries = nn.Parameter(torch.randn(m, model_channels) * 0.02)
            grid = round(m ** (1.0 / 3.0))
            if grid ** 3 == m:
                coarse = torch.meshgrid(
                    torch.arange(grid, dtype=torch.float32),
                    torch.arange(grid, dtype=torch.float32),
                    torch.arange(grid, dtype=torch.float32),
                    indexing="ij",
                )
                coarse = torch.stack(coarse, dim=-1).reshape(-1, 3)
            else:
                # m is not a perfect cube: fall back to a 1D ordering so the
                # positional embedding stays well-defined (still spatial, just a
                # line). Replicated across all 3 coords keeps the embedder happy.
                line = torch.arange(m, dtype=torch.float32).unsqueeze(-1)
                coarse = line.repeat(1, 3)
            self.summary_query_pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            self.register_buffer("_summary_query_coords", coarse, persistent=False)
            # Pool: m summary queries cross-attend to a part's [T,C] tokens.
            self.summary_pool_attn = MultiHeadAttention(
                model_channels,
                ctx_channels=model_channels,
                num_heads=int(num_heads),
                type="cross",
                attn_mode="full",
                qk_rms_norm=False,
            )
            # Broadcast: a part's [T,C] tokens cross-attend to ALL parts' mixed
            # summaries [K*m, C].
            self.summary_broadcast_attn = MultiHeadAttention(
                model_channels,
                ctx_channels=model_channels,
                num_heads=int(num_heads),
                type="cross",
                attn_mode="full",
                qk_rms_norm=False,
            )
            # Pre-norms on the cross-attn key/value (the part tokens are the
            # adaLN-modulated self-attn input `x`; the summaries are pooled from
            # them). Keeps the pool/broadcast attention numerically stable without
            # touching the backbone block's own norms.
            self.summary_pool_kv_norm = LayerNorm32(model_channels, elementwise_affine=True, eps=1e-6)
            self.summary_mix_norm = LayerNorm32(model_channels, elementwise_affine=True, eps=1e-6)

    def build_condition_tokens(
        self,
        cond: torch.Tensor,
        mask_token_labels: torch.Tensor,
        part_valid: torch.Tensor,
        target_slots: torch.Tensor,
        part_token_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond.dim() != 3:
            raise ValueError(f"cond must be [B,V*T,D], got {tuple(cond.shape)}")
        if mask_token_labels.shape != cond.shape[:2]:
            raise ValueError(
                f"mask_token_labels shape {tuple(mask_token_labels.shape)} does not match "
                f"cond token shape {tuple(cond.shape[:2])}"
            )
        if part_valid.dim() != 2:
            raise ValueError(f"part_valid must be [B,K], got {tuple(part_valid.shape)}")
        if target_slots.shape != part_valid.shape:
            raise ValueError(
                f"target_slots shape {tuple(target_slots.shape)} does not match "
                f"part_valid shape {tuple(part_valid.shape)}"
            )
        if part_valid.shape[0] != cond.shape[0]:
            raise ValueError(
                f"part_valid batch {part_valid.shape[0]} does not match cond batch {cond.shape[0]}"
            )
        if part_valid.shape[1] > self.max_parts:
            raise ValueError(f"K={part_valid.shape[1]} exceeds model.max_parts={self.max_parts}")
        B, K = part_valid.shape
        if part_token_weights is not None and tuple(part_token_weights.shape) != (B, K, cond.shape[1]):
            raise ValueError(
                f"part_token_weights must be [B,K,T]={(B, K, cond.shape[1])}, "
                f"got {tuple(part_token_weights.shape)}"
            )

        label_ids = mask_token_labels.clamp(0, self.part_label_vocab_size - 1)
        proj_cond = self.cond_proj(cond)
        cond_tokens = proj_cond + self.part_label_emb(label_ids)
        slot_ids = target_slots.clamp(0, self.part_label_vocab_size - 1)
        if part_token_weights is not None:
            pooled_parts = torch.bmm(
                part_token_weights.to(device=proj_cond.device, dtype=proj_cond.dtype),
                proj_cond,
            )
            part_queries = (
                pooled_parts
                + self.part_type_emb.view(1, 1, -1).to(dtype=pooled_parts.dtype)
                + self.part_slot_emb(slot_ids)
            )
        else:
            part_queries = cond_tokens.new_zeros((B, K, cond_tokens.shape[-1]))
            for b in range(B):
                for k in range(K):
                    if not bool(part_valid[b, k]):
                        continue
                    slot = int(target_slots[b, k].item())
                    keep = mask_token_labels[b] == slot
                    if not bool(keep.any()):
                        if self.require_part_token:
                            raise ValueError(
                                f"target_slot={slot} has zero 2D mask token coverage; "
                                "set data.allow_missing_masks=true and model.require_part_token=false "
                                "to opt into the no-part-token ablation"
                            )
                        pooled = cond_tokens[b].mean(dim=0)
                    else:
                        pooled = cond_tokens[b, keep].mean(dim=0)
                    part_queries[b, k] = (
                        pooled
                        + self.part_type_emb.view(-1)
                        + self.part_slot_emb(slot_ids[b, k])
                    )
        if isinstance(self.part_query_encoder, nn.Identity):
            encoded = part_queries
        else:
            encoded = self.part_query_encoder(
                part_queries,
                src_key_padding_mask=~part_valid.bool(),
            )
        part_queries = self.part_query_norm(encoded)
        part_queries = part_queries * part_valid.to(dtype=part_queries.dtype).unsqueeze(-1)
        return cond_tokens, part_queries

    def _encode_global_condition(self, z_global: torch.Tensor) -> torch.Tensor:
        if z_global.dim() != 5:
            raise ValueError(f"z_global must be [B,C,R,R,R], got {tuple(z_global.shape)}")
        B, C, R, H, W = z_global.shape
        P = self.global_cond_patch_size
        if C != self.latent_channels:
            raise ValueError(f"z_global channels={C} must match latent_channels={self.latent_channels}")
        if (R, H, W) != (self.resolution, self.resolution, self.resolution):
            raise ValueError(
                f"z_global spatial shape {(R, H, W)} must match "
                f"model resolution {(self.resolution, self.resolution, self.resolution)}"
            )
        patches = z_global.reshape(B, C, R // P, P, H // P, P, W // P, P)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
        patches = patches.reshape(B, -1, C * P ** 3)
        tokens = self.global_cond_layer(patches)
        pos_emb = self.global_cond_pos_embedder(
            self._global_voxel_coords.to(device=z_global.device)
        ).to(dtype=tokens.dtype)
        return tokens + pos_emb.unsqueeze(0) + self.global_token_type_emb.to(dtype=tokens.dtype)

    def _build_target_part_queries(
        self,
        part_queries: torch.Tensor,
        part_valid: torch.Tensor,
        batch_idx: int,
        target_part_idx: int,
    ) -> torch.Tensor:
        query_roles = self.context_query_emb.to(dtype=part_queries.dtype).expand(
            part_queries.shape[1],
            -1,
        ).clone()
        query_roles[target_part_idx] = self.target_query_emb.to(dtype=part_queries.dtype).view(-1)
        target_part_queries = part_queries[batch_idx] + query_roles
        return target_part_queries * part_valid[batch_idx].to(dtype=part_queries.dtype).unsqueeze(-1)

    def _build_target_cond_tokens(
        self,
        cond_tokens: torch.Tensor,
        mask_token_labels: torch.Tensor,
        batch_idx: int,
        target_slot: torch.Tensor,
        weight_row: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_emb = self.target_token_emb.to(dtype=cond_tokens.dtype)
        context_emb = self.context_token_emb.to(dtype=cond_tokens.dtype)
        if self.soft_role_marking and weight_row is not None:
            # Soft role marking: a token's "target-ness" = its per-patch overlap
            # with this part (the SAME part_token_weights used for the pooled
            # query), rescaled to [0,1] by the part's own max. The hard min_fg=3
            # vote marks a sub-patch part's tokens ALL-context (degenerate, identical
            # across starved parts); this gives every part with >=1 mask pixel a
            # non-zero, per-(view,patch)-distinct role signal in the cross-attn
            # KEY/VALUE. g=1 -> exact target_emb, g=0 -> exact context_emb.
            w = weight_row.to(device=cond_tokens.device, dtype=cond_tokens.dtype)
            g = (w / w.max().clamp_min(1e-8)).clamp(0.0, 1.0).unsqueeze(-1)
            token_role_emb = g * target_emb + (1.0 - g) * context_emb
        else:
            target_mask = (mask_token_labels[batch_idx] == target_slot).unsqueeze(-1)
            token_role_emb = torch.where(target_mask, target_emb, context_emb)
        return cond_tokens[batch_idx] + token_role_emb

    def _build_cond_memory_batch(
        self,
        cond_tokens: torch.Tensor,
        mask_token_labels: torch.Tensor,
        part_queries: torch.Tensor,
        part_valid: torch.Tensor,
        target_slots: torch.Tensor,
        global_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        part_idx: torch.Tensor,
        drop_part_cond: torch.Tensor | None = None,
        part_token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memories = []
        for pos, (b, k) in enumerate(zip(batch_idx.tolist(), part_idx.tolist())):
            global_mem = global_tokens[b]
            weight_row = part_token_weights[b, k] if part_token_weights is not None else None
            target_cond_tokens = self._build_target_cond_tokens(
                cond_tokens,
                mask_token_labels,
                b,
                target_slots[b, k],
                weight_row=weight_row,
            )
            target_part_queries = self._build_target_part_queries(
                part_queries,
                part_valid,
                b,
                k,
            )
            # Fix 8 (CFG): when a part is dropped, replace its per-part query +
            # role-marked cond tokens with the learnable null embedding, keeping the
            # shared global z tokens (so CFG only nulls the part-identity cond).
            # Use an identity BLEND, not a python `if drop` branch: drop=0 -> exact
            # real, drop=1 -> exact null. The branch left null_part_query /
            # null_cond_token OUT of the autograd graph on no-drop steps, so under
            # DDP they were "unused" (and inconsistently so across ranks) -> the
            # gradient allreduce deadlocks. The blend keeps BOTH the null and the
            # real cond params in the graph every step on every rank (zero-weighted
            # when not selected), so DDP needs no find_unused_parameters.
            # Only under CFG do the null params exist (see __init__). With CFG off
            # there is no dropping, so the real per-part query / cond tokens pass
            # through unchanged — numerically identical to the drop=0 blend below,
            # but without touching (or requiring) null params the checkpoint lacks.
            if self.classifier_free_guidance:
                drop = float(drop_part_cond[pos]) if drop_part_cond is not None else 0.0
                null_query = self.null_part_query.to(dtype=global_mem.dtype).expand(
                    target_part_queries.shape[0], -1
                )
                null_cond = self.null_cond_token.to(dtype=global_mem.dtype).expand(
                    target_cond_tokens.shape[0], -1
                )
                part_q = (1.0 - drop) * target_part_queries + drop * null_query
                cond_t = (1.0 - drop) * target_cond_tokens + drop * null_cond
            else:
                part_q = target_part_queries
                cond_t = target_cond_tokens
            memories.append(torch.cat([part_q, cond_t, global_mem], dim=0))
        return torch.stack(memories, dim=0)

    def _build_mask_attention_bias_batch(
        self,
        *,
        cond_tokens: torch.Tensor,
        part_queries: torch.Tensor,
        global_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        part_idx: torch.Tensor,
        part_token_weights: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.mask_attention_bias_enabled:
            return None
        if part_token_weights is None:
            return None
        image_token_count = int(cond_tokens.shape[1])
        prefix_count = int(part_queries.shape[1])
        global_token_count = int(global_tokens.shape[1])
        total_tokens = prefix_count + image_token_count + global_token_count
        rows = []
        any_nonzero = False
        for b, k in zip(batch_idx.tolist(), part_idx.tolist()):
            weights = part_token_weights[b, k].to(device=cond_tokens.device, dtype=cond_tokens.dtype)
            if weights.shape != (image_token_count,):
                raise ValueError(
                    f"part_token_weights row must have {image_token_count} image tokens, "
                    f"got {tuple(weights.shape)}"
                )
            row = cond_tokens.new_zeros((total_tokens,))
            if bool((weights > 0).any()):
                bias = (
                    self.mask_attention_bias_lambda
                    * torch.log(weights.clamp_min(self.mask_attention_bias_eps))
                )
                row[prefix_count:prefix_count + image_token_count] = bias
                if bool((bias != 0).any()):
                    any_nonzero = True
            elif not self._mask_attention_bias_warned_zero_weight:
                print(
                    "[WARN] mask_attention_bias skipped for part with all-zero "
                    "part_token_weights",
                    flush=True,
                )
                self._mask_attention_bias_warned_zero_weight = True
            rows.append(row)
        if not rows or not any_nonzero:
            return None
        # [N, 1, 1, Lkv], broadcast over heads and query tokens.
        return torch.stack(rows, dim=0).unsqueeze(1).unsqueeze(1)

    def _patchify_tokens(self, x_in: torch.Tensor) -> torch.Tensor:
        """Patchify [N,C,R,R,R] -> tokens [N,T,model_channels] + backbone pos_emb."""
        bb = self.backbone
        h = patchify(x_in, bb.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = bb.input_layer(h)
        h = h + bb.pos_emb[None]
        return h

    def _unpatchify_tokens(self, h: torch.Tensor) -> torch.Tensor:
        """Tokens [N,T,model_channels] -> velocity [N,out_channels,R,R,R]."""
        bb = self.backbone
        h = F.layer_norm(h, h.shape[-1:])
        h = bb.out_layer(h)
        h = h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2], *[bb.resolution // bb.patch_size] * 3
        )
        return unpatchify(h, bb.patch_size).contiguous()

    @staticmethod
    def _block_modulation(block, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if block.share_mod:
            return t_emb.chunk(6, dim=1)
        return block.adaLN_modulation(t_emb).chunk(6, dim=1)

    def _summary_cross_part_attn(
        self,
        block,
        x: torch.Tensor,
        summary_identity: torch.Tensor,
    ) -> torch.Tensor:
        """ISAB/Perceiver cross-part interaction over learned summary tokens.

        x: [K, T, C] = the adaLN-modulated self-attn input (block.norm1(h) then
        scale/shift). summary_identity: [K, C] = self.part_slot_emb(slot) per part.
        Returns the even-block self-attn output [K, T, C] that REPLACES the full
        [1,K*T,C] block.self_attn output in the gated residual. Cost is
        O(K*m*T) (pool) + O((K*m)^2) (mix via block.self_attn) + O(K*T*K*m)
        (broadcast) instead of O((K*T)^2).
        """
        K, T, C = x.shape
        m = self.n_summary_tokens
        # 1) POOL each part to m summaries via learned, spatially-structured
        #    attention (NOT mean pooling — that washes out small-part footprints).
        pos = self.summary_query_pos_embedder(
            self._summary_query_coords.to(device=x.device)
        ).to(dtype=x.dtype)
        queries = (self.summary_queries.to(dtype=x.dtype) + pos).unsqueeze(0).expand(K, -1, -1)
        kv = self.summary_pool_kv_norm(x)
        summaries = self.summary_pool_attn(queries, kv)  # [K, m, C]
        # Per-part identity so summaries carry WHICH part (prevents co-located
        # parts collapsing to identical summaries).
        summaries = summaries + summary_identity.unsqueeze(1).to(dtype=summaries.dtype)
        # 2) CROSS-PART MIX over the concatenated summaries with the block's own
        #    self-attn (still self-attention, now over K*m << K*T tokens).
        mixed = block.self_attn(summaries.reshape(1, K * m, C)).reshape(K, m, C)
        mixed = self.summary_mix_norm(mixed)
        # 3) BROADCAST back: every part's full tokens cross-attend to ALL parts'
        #    mixed summaries, so cross-part context reaches the full resolution.
        all_summaries = mixed.reshape(1, K * m, C).expand(K, -1, -1)
        return self.summary_broadcast_attn(x, all_summaries)  # [K, T, C]

    def _joint_block_forward(
        self,
        block,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        cond_memory: torch.Tensor,
        attn_bias: torch.Tensor | None,
        global_self_attn: bool,
        summary_identity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reproduce ModulatedTransformerCrossBlock math for K parts of one object.

        h: [K, T, C] part tokens. t_emb: [K, C_mod] (identical across parts —
        all parts of an object share the object timestep). cond_memory: [K, N, C].
        global_self_attn True reshapes self-attention to [1, K*T, C] for
        cross-part mixing; False keeps it per-part [K, T, C]. When
        summary_cross_part_attention is on, the global (even) block instead runs
        the summary-token ISAB path (summary_identity required). Cross-attention
        is always per-part. The submodules (norm/self_attn/cross_attn/mlp) are the
        backbone block's own trained weights — read, never edited.
        """
        K, T, C = h.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._block_modulation(block, t_emb)

        # --- self attention (parity-controlled scope) ---
        x = block.norm1(h)
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        if global_self_attn and self.summary_cross_part_attention:
            # Even block: replace the O((K*T)^2) full cross-part self-attn with
            # the summary-token path. adaLN/gate/residual below stay identical.
            x = self._summary_cross_part_attn(block, x, summary_identity)
        elif global_self_attn:
            x = x.reshape(1, K * T, C)
            x = block.self_attn(x)
            x = x.reshape(K, T, C)
        else:
            x = block.self_attn(x)
        x = x * gate_msa.unsqueeze(1)
        h = h + x

        # --- cross attention (always per-part to per-part condition memory) ---
        x = block.norm2(h)
        x = block.cross_attn(x, cond_memory, attn_bias=attn_bias)
        h = h + x

        # --- feed forward ---
        x = block.norm3(h)
        x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = block.mlp(x)
        x = x * gate_mlp.unsqueeze(1)
        h = h + x
        return h

    def _forward_object_joint(
        self,
        x_in: torch.Tensor,
        t_obj: torch.Tensor,
        cond_memory: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        identity_slot_ids: torch.Tensor | None = None,
        summary_slot_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Joint forward for one object's K valid parts (fix 1).

        x_in: [K, in_channels, R, R, R]. t_obj: scalar timestep tensor shape [].
        cond_memory: [K, N, model_channels]. identity_slot_ids: optional [K] slot
        indices whose embedding is added to every latent token (fix 2).
        summary_slot_ids: [K] slot indices for the summary-token identity
        (required when summary_cross_part_attention is on; independent of fix 2).
        Returns [K, out_channels, R, R, R]. Self-attention alternates global (even
        block) vs within-part (odd block) by reshape; cross-attention stays
        per-part.
        """
        bb = self.backbone
        K = x_in.shape[0]
        h = self._patchify_tokens(x_in)
        if identity_slot_ids is not None:
            identity = self.part_slot_emb(identity_slot_ids).to(dtype=h.dtype).unsqueeze(1)
            h = h + identity
        summary_identity = None
        if self.summary_cross_part_attention:
            if summary_slot_ids is None:
                raise ValueError(
                    "summary_cross_part_attention requires summary_slot_ids in _forward_object_joint"
                )
            summary_identity = self.part_slot_emb(summary_slot_ids).to(dtype=bb.dtype)
        # All K parts share the object timestep, so t_emb is identical per part.
        t_emb = bb.t_embedder(t_obj.view(1)).expand(K, -1)
        if bb.share_mod:
            t_emb = bb.adaLN_modulation(t_emb)
        t_emb = t_emb.type(bb.dtype)
        h = h.type(bb.dtype)
        cond_memory = cond_memory.type(bb.dtype)
        for i, block in enumerate(bb.blocks):
            global_self_attn = (i % 2 == 0)
            if bb.use_checkpoint and torch.is_grad_enabled():
                h = checkpoint(
                    self._joint_block_forward,
                    block,
                    h,
                    t_emb,
                    cond_memory,
                    attn_bias,
                    global_self_attn,
                    summary_identity,
                    use_reentrant=False,
                )
            else:
                h = self._joint_block_forward(
                    block, h, t_emb, cond_memory, attn_bias, global_self_attn, summary_identity
                )
        h = h.type(x_in.dtype)
        return self._unpatchify_tokens(h)

    @staticmethod
    def _normalize_drop_part_cond(
        drop_part_cond: bool | torch.Tensor | None,
        part_valid: torch.Tensor,
    ) -> torch.Tensor | None:
        if drop_part_cond is None or drop_part_cond is False:
            return None
        B, K = part_valid.shape
        if drop_part_cond is True:
            return torch.ones((B, K), dtype=torch.bool, device=part_valid.device)
        drop = drop_part_cond
        if not isinstance(drop, torch.Tensor):
            drop = torch.as_tensor(drop, device=part_valid.device)
        drop = drop.to(device=part_valid.device)
        if drop.dim() == 1 and drop.shape[0] == B:
            drop = drop.view(B, 1).expand(B, K)
        if tuple(drop.shape) != (B, K):
            raise ValueError(
                f"drop_part_cond must broadcast to {(B, K)}, got {tuple(drop.shape)}"
            )
        return drop.bool()

    def forward(
        self,
        x_t_parts: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        cond: torch.Tensor,
        mask_token_labels: torch.Tensor,
        part_valid: torch.Tensor,
        target_slots: torch.Tensor,
        part_token_weights: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
        drop_part_cond: bool | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t_parts.dim() != 6:
            raise ValueError(f"x_t_parts must be [B,K,C,R,R,R], got {tuple(x_t_parts.shape)}")
        B, K, C, R, H, W = x_t_parts.shape
        if z_global.shape != (B, C, R, H, W):
            raise ValueError(
                f"z_global shape {tuple(z_global.shape)} does not match "
                f"x_t_parts object shape {(B, C, R, H, W)}"
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if part_valid.shape != (B, K):
            raise ValueError(f"part_valid must be {(B, K)}, got {tuple(part_valid.shape)}")
        if target_slots.shape != (B, K):
            raise ValueError(f"target_slots must be {(B, K)}, got {tuple(target_slots.shape)}")
        if x_self_cond is not None:
            if not self.self_conditioning:
                raise ValueError(
                    "x_self_cond provided but model.self_conditioning=false; "
                    "enable the flag to use self-conditioning"
                )
            if x_self_cond.shape != x_t_parts.shape:
                raise ValueError(
                    f"x_self_cond shape {tuple(x_self_cond.shape)} must match "
                    f"x_t_parts {tuple(x_t_parts.shape)}"
                )

        part_valid = part_valid.bool()
        if not bool(part_valid.any(dim=1).all()):
            raise ValueError("each object must contain at least one valid target part")
        valid_idx = part_valid.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            raise ValueError("part_valid contains no valid target parts")

        cond_tokens, part_queries = self.build_condition_tokens(
            cond,
            mask_token_labels,
            part_valid,
            target_slots,
            part_token_weights=part_token_weights,
        )
        global_tokens = self._encode_global_condition(z_global)
        drop_mask = self._normalize_drop_part_cond(drop_part_cond, part_valid)
        slot_ids = target_slots.clamp(0, self.part_label_vocab_size - 1)
        out = torch.zeros_like(x_t_parts)

        if self.cross_part_attention:
            joint_fn = self._forward_joint_flat if self.joint_flat_batch else self._forward_joint
            return joint_fn(
                x_t_parts=x_t_parts,
                t=t,
                z_global=z_global,
                cond_tokens=cond_tokens,
                mask_token_labels=mask_token_labels,
                part_queries=part_queries,
                part_valid=part_valid,
                target_slots=target_slots,
                slot_ids=slot_ids,
                global_tokens=global_tokens,
                x_self_cond=x_self_cond,
                drop_mask=drop_mask,
                part_token_weights=part_token_weights,
                out=out,
            )
        return self._forward_independent(
            x_t_parts=x_t_parts,
            t=t,
            z_global=z_global,
            cond_tokens=cond_tokens,
            mask_token_labels=mask_token_labels,
            part_queries=part_queries,
            part_valid=part_valid,
            target_slots=target_slots,
            slot_ids=slot_ids,
            global_tokens=global_tokens,
            valid_idx=valid_idx,
            x_self_cond=x_self_cond,
            drop_mask=drop_mask,
            part_token_weights=part_token_weights,
            out=out,
        )

    def _assemble_part_input(
        self,
        x_valid: torch.Tensor,
        z_valid: torch.Tensor,
        x_self_cond_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [x_valid]
        if self.concat_global:
            parts.append(z_valid)
        if self.self_conditioning:
            if x_self_cond_valid is None:
                # First self-conditioning pass: feed zeros for the previous x0_hat.
                x_self_cond_valid = torch.zeros_like(x_valid)
            parts.append(x_self_cond_valid)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _forward_independent(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        valid_idx,
        x_self_cond,
        drop_mask,
        part_token_weights,
        out,
    ) -> torch.Tensor:
        batch_idx = valid_idx[:, 0]
        part_idx = valid_idx[:, 1]
        chunk_size = self.max_part_forward_batch or int(valid_idx.shape[0])
        for start in range(0, int(valid_idx.shape[0]), chunk_size):
            end = min(start + chunk_size, int(valid_idx.shape[0]))
            chunk_batch_idx = batch_idx[start:end]
            chunk_part_idx = part_idx[start:end]
            x_valid = x_t_parts[chunk_batch_idx, chunk_part_idx]
            z_valid = z_global[chunk_batch_idx]
            sc_valid = x_self_cond[chunk_batch_idx, chunk_part_idx] if x_self_cond is not None else None
            x_in = self._assemble_part_input(x_valid, z_valid, sc_valid)
            t_valid = t[chunk_batch_idx]
            chunk_drop = (
                drop_mask[chunk_batch_idx, chunk_part_idx] if drop_mask is not None else None
            )
            cond_memory = self._build_cond_memory_batch(
                cond_tokens,
                mask_token_labels,
                part_queries,
                part_valid,
                target_slots,
                global_tokens,
                chunk_batch_idx,
                chunk_part_idx,
                drop_part_cond=chunk_drop,
                part_token_weights=part_token_weights,
            )
            attn_bias = self._build_mask_attention_bias_batch(
                cond_tokens=cond_tokens,
                part_queries=part_queries,
                global_tokens=global_tokens,
                batch_idx=chunk_batch_idx,
                part_idx=chunk_part_idx,
                part_token_weights=part_token_weights,
            )
            if self.token_identity_embedding:
                pred_valid = self._backbone_with_identity(
                    x_in, t_valid, cond_memory, slot_ids[chunk_batch_idx, chunk_part_idx], attn_bias=attn_bias
                )
            elif attn_bias is None:
                pred_valid = self.backbone(x_in, t_valid, cond_memory)
            else:
                pred_valid = self.backbone(x_in, t_valid, cond_memory, attn_bias=attn_bias)
            out[chunk_batch_idx, chunk_part_idx] = pred_valid.to(dtype=out.dtype)
        return out

    def _backbone_with_identity(
        self,
        x_in: torch.Tensor,
        t_valid: torch.Tensor,
        cond_memory: torch.Tensor,
        slot_ids: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the dense backbone but add the per-part identity embedding to every
        latent token (fix 2) in the independent path."""
        bb = self.backbone
        h = self._patchify_tokens(x_in)
        identity = self.part_slot_emb(slot_ids).to(dtype=h.dtype).unsqueeze(1)
        h = h + identity
        t_emb = bb.t_embedder(t_valid)
        if bb.share_mod:
            t_emb = bb.adaLN_modulation(t_emb)
        t_emb = t_emb.type(bb.dtype)
        h = h.type(bb.dtype)
        cond_memory = cond_memory.type(bb.dtype)
        if attn_bias is not None:
            attn_bias = attn_bias.to(device=h.device, dtype=bb.dtype)
        for block in bb.blocks:
            h = block(h, t_emb, cond_memory, attn_bias=attn_bias)
        h = h.type(x_in.dtype)
        return self._unpatchify_tokens(h)

    def _forward_joint(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        x_self_cond,
        drop_mask,
        part_token_weights,
        out,
    ) -> torch.Tensor:
        B = x_t_parts.shape[0]
        for b in range(B):
            valid_b = torch.nonzero(part_valid[b], as_tuple=False).flatten()
            if valid_b.numel() == 0:
                raise ValueError(f"object {b} has zero valid parts in joint forward")
            x_valid = x_t_parts[b, valid_b]
            z_valid = z_global[b : b + 1].expand(valid_b.numel(), -1, -1, -1, -1)
            sc_valid = x_self_cond[b, valid_b] if x_self_cond is not None else None
            x_in = self._assemble_part_input(x_valid, z_valid, sc_valid)
            chunk_batch_idx = torch.full_like(valid_b, b)
            chunk_drop = drop_mask[b, valid_b] if drop_mask is not None else None
            cond_memory = self._build_cond_memory_batch(
                cond_tokens,
                mask_token_labels,
                part_queries,
                part_valid,
                target_slots,
                global_tokens,
                chunk_batch_idx,
                valid_b,
                drop_part_cond=chunk_drop,
                part_token_weights=part_token_weights,
            )
            attn_bias = self._build_mask_attention_bias_batch(
                cond_tokens=cond_tokens,
                part_queries=part_queries,
                global_tokens=global_tokens,
                batch_idx=chunk_batch_idx,
                part_idx=valid_b,
                part_token_weights=part_token_weights,
            )
            identity_slot_ids = slot_ids[b, valid_b] if self.token_identity_embedding else None
            summary_slot_ids = slot_ids[b, valid_b] if self.summary_cross_part_attention else None
            pred_valid = self._forward_object_joint(
                x_in,
                t[b],
                cond_memory,
                attn_bias=attn_bias,
                identity_slot_ids=identity_slot_ids,
                summary_slot_ids=summary_slot_ids,
            )
            out[b, valid_b] = pred_valid.to(dtype=out.dtype)
        return out

    @staticmethod
    def _object_groups(batch_idx: torch.Tensor) -> list[tuple[int, int]]:
        """Contiguous (start, end) row ranges per object in a flat part batch.

        ``batch_idx`` is the object id of each flat part; it is non-decreasing
        because ``part_valid.nonzero()`` returns indices in row-major order, so
        every object's parts are contiguous. The ranges define the per-object
        attention scope for the cross-part (even) blocks.
        """
        if batch_idx.numel() == 0:
            return []
        if bool((batch_idx[1:] < batch_idx[:-1]).any()):
            raise ValueError("flat part batch_idx must be non-decreasing (objects contiguous)")
        counts = torch.unique_consecutive(batch_idx, return_counts=True)[1].tolist()
        groups, start = [], 0
        for c in counts:
            groups.append((start, start + int(c)))
            start += int(c)
        return groups

    def _summary_cross_part_attn_flat(
        self,
        block,
        x: torch.Tensor,
        summary_identity: torch.Tensor,
        groups: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Flat-batch summary-token cross-part attention (see _summary_cross_part_attn).

        The POOL step is per-part, so it runs flat over all N parts at once. Only
        the MIX (cross-part self-attn over K*m summaries) and BROADCAST stay
        grouped per object so parts never attend across objects. Each per-object
        sub-call is identical to the serial path."""
        N, T, C = x.shape
        m = self.n_summary_tokens
        pos = self.summary_query_pos_embedder(
            self._summary_query_coords.to(device=x.device)
        ).to(dtype=x.dtype)
        queries = (self.summary_queries.to(dtype=x.dtype) + pos).unsqueeze(0).expand(N, -1, -1)
        kv = self.summary_pool_kv_norm(x)
        summaries = self.summary_pool_attn(queries, kv)  # [N, m, C] (per-part)
        summaries = summaries + summary_identity.unsqueeze(1).to(dtype=summaries.dtype)
        outs = []
        for s, e in groups:
            Kb = e - s
            grp = summaries[s:e]
            mixed = block.self_attn(grp.reshape(1, Kb * m, C)).reshape(Kb, m, C)
            mixed = self.summary_mix_norm(mixed)
            all_summaries = mixed.reshape(1, Kb * m, C).expand(Kb, -1, -1)
            outs.append(self.summary_broadcast_attn(x[s:e], all_summaries))
        return torch.cat(outs, dim=0)

    def _joint_block_forward_flat(
        self,
        block,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        cond_memory: torch.Tensor,
        attn_bias: torch.Tensor | None,
        global_self_attn: bool,
        summary_identity: torch.Tensor | None,
        groups: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Flat-batch counterpart of _joint_block_forward over [N,T,C] parts.

        Per-part ops (modulation, within-part self-attn, cross-attn, MLP) run on
        the full flat batch; the cross-part (even) self-attn scope is grouped per
        object (summary path, or the [1,K*T,C] reshape when summary is off)."""
        N, T, C = h.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._block_modulation(block, t_emb)

        # --- self attention (parity-controlled scope, grouped per object) ---
        x = block.norm1(h)
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        if global_self_attn and self.summary_cross_part_attention:
            x = self._summary_cross_part_attn_flat(block, x, summary_identity, groups)
        elif global_self_attn:
            segs = []
            for s, e in groups:
                Kb = e - s
                seg = x[s:e].reshape(1, Kb * T, C)
                seg = block.self_attn(seg).reshape(Kb, T, C)
                segs.append(seg)
            x = torch.cat(segs, dim=0)
        else:
            x = block.self_attn(x)
        x = x * gate_msa.unsqueeze(1)
        h = h + x

        # --- cross attention (always per-part) ---
        x = block.norm2(h)
        x = block.cross_attn(x, cond_memory, attn_bias=attn_bias)
        h = h + x

        # --- feed forward ---
        x = block.norm3(h)
        x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = block.mlp(x)
        x = x * gate_mlp.unsqueeze(1)
        h = h + x
        return h

    def _run_flat_backbone(
        self,
        x_in: torch.Tensor,
        t_valid: torch.Tensor,
        cond_memory: torch.Tensor,
        attn_bias: torch.Tensor | None,
        groups: list[tuple[int, int]],
        identity_slot_ids: torch.Tensor | None,
        summary_slot_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the joint backbone over a flat [N,...] part batch (all objects)."""
        bb = self.backbone
        h = self._patchify_tokens(x_in)
        if identity_slot_ids is not None:
            h = h + self.part_slot_emb(identity_slot_ids).to(dtype=h.dtype).unsqueeze(1)
        summary_identity = None
        if self.summary_cross_part_attention:
            if summary_slot_ids is None:
                raise ValueError(
                    "summary_cross_part_attention requires summary_slot_ids in flat forward"
                )
            summary_identity = self.part_slot_emb(summary_slot_ids).to(dtype=bb.dtype)
        # t_valid already carries each part's object timestep (t[batch_idx]).
        t_emb = bb.t_embedder(t_valid)
        if bb.share_mod:
            t_emb = bb.adaLN_modulation(t_emb)
        t_emb = t_emb.type(bb.dtype)
        h = h.type(bb.dtype)
        cond_memory = cond_memory.type(bb.dtype)
        if attn_bias is not None:
            attn_bias = attn_bias.to(device=h.device, dtype=bb.dtype)
        for i, block in enumerate(bb.blocks):
            global_self_attn = (i % 2 == 0)
            if bb.use_checkpoint and torch.is_grad_enabled():
                h = checkpoint(
                    self._joint_block_forward_flat,
                    block,
                    h,
                    t_emb,
                    cond_memory,
                    attn_bias,
                    global_self_attn,
                    summary_identity,
                    groups,
                    use_reentrant=False,
                )
            else:
                h = self._joint_block_forward_flat(
                    block, h, t_emb, cond_memory, attn_bias, global_self_attn, summary_identity, groups
                )
        h = h.type(x_in.dtype)
        return self._unpatchify_tokens(h)

    def _forward_joint_flat(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        x_self_cond,
        drop_mask,
        part_token_weights,
        out,
    ) -> torch.Tensor:
        """Flat (cross-object batched) joint forward — numerically identical to
        _forward_joint but processes ALL objects' valid parts in one batch."""
        valid_idx = part_valid.nonzero(as_tuple=False)
        batch_idx = valid_idx[:, 0]
        part_idx = valid_idx[:, 1]
        x_valid = x_t_parts[batch_idx, part_idx]
        z_valid = z_global[batch_idx]
        sc_valid = x_self_cond[batch_idx, part_idx] if x_self_cond is not None else None
        x_in = self._assemble_part_input(x_valid, z_valid, sc_valid)
        t_valid = t[batch_idx]
        chunk_drop = drop_mask[batch_idx, part_idx] if drop_mask is not None else None
        cond_memory = self._build_cond_memory_batch(
            cond_tokens,
            mask_token_labels,
            part_queries,
            part_valid,
            target_slots,
            global_tokens,
            batch_idx,
            part_idx,
            drop_part_cond=chunk_drop,
            part_token_weights=part_token_weights,
        )
        attn_bias = self._build_mask_attention_bias_batch(
            cond_tokens=cond_tokens,
            part_queries=part_queries,
            global_tokens=global_tokens,
            batch_idx=batch_idx,
            part_idx=part_idx,
            part_token_weights=part_token_weights,
        )
        identity_slot_ids = slot_ids[batch_idx, part_idx] if self.token_identity_embedding else None
        summary_slot_ids = slot_ids[batch_idx, part_idx] if self.summary_cross_part_attention else None
        groups = self._object_groups(batch_idx)
        pred = self._run_flat_backbone(
            x_in, t_valid, cond_memory, attn_bias, groups, identity_slot_ids, summary_slot_ids
        )
        out[batch_idx, part_idx] = pred.to(dtype=out.dtype)
        return out
