from typing import Any

import torch
import torch.nn as nn

from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from .decoder_mesh import SLatMeshDecoder


class PartMaskedSLatMeshDecoder(SLatMeshDecoder):
    """Mesh decoder variant conditioned on a per-component mask channel.

    The input sparse feature at each occupied whole-object SLat coordinate is
    `[slat_feat_0..7, part_mask]`.  The mask channel can be zero-initialized
    from an 8-channel pretrained decoder so the starting behavior matches the
    original mesh decoder before finetuning.
    """

    def __init__(
        self,
        *args: Any,
        base_latent_channels: int = 8,
        mask_channels: int = 1,
        mask_modulation: str = "none",
        latent_channels: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.base_latent_channels = int(base_latent_channels)
        self.mask_channels = int(mask_channels)
        self.mask_modulation = str(mask_modulation)
        if self.mask_channels <= 0:
            raise ValueError(f"mask_channels must be positive, got {self.mask_channels}")
        if self.mask_modulation not in {
            "none",
            "per_block_add",
            "output_feature_add",
            "per_block_add_output_feature_add",
        }:
            raise ValueError(f"unsupported mask_modulation={self.mask_modulation!r}")
        effective_channels = int(latent_channels) if latent_channels is not None else self.base_latent_channels + self.mask_channels
        expected = self.base_latent_channels + self.mask_channels
        if effective_channels != expected:
            raise ValueError(
                f"PartMaskedSLatMeshDecoder expects latent_channels={expected} "
                f"(base={self.base_latent_channels}, mask={self.mask_channels}), got {effective_channels}"
            )
        super().__init__(*args, latent_channels=effective_channels, **kwargs)
        if self._use_block_modulation:
            self.mask_block_modulations = nn.ModuleList(
                [nn.Linear(self.mask_channels, self.model_channels) for _ in range(self.num_blocks)]
            )
            for module in self.mask_block_modulations:
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)
        else:
            self.mask_block_modulations = nn.ModuleList()
        if self._use_output_feature_modulation:
            self.mask_output_modulation = nn.Linear(self.mask_channels, self.out_channels)
            nn.init.constant_(self.mask_output_modulation.weight, 0)
            nn.init.constant_(self.mask_output_modulation.bias, 0)
        else:
            self.mask_output_modulation = None

    @property
    def _use_block_modulation(self) -> bool:
        return self.mask_modulation in {"per_block_add", "per_block_add_output_feature_add"}

    @property
    def _use_output_feature_modulation(self) -> bool:
        return self.mask_modulation in {"output_feature_add", "per_block_add_output_feature_add"}

    def load_partmasked_state_dict_from_base(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True):
        """Load an 8-channel mesh decoder checkpoint into this 9-channel variant.

        The pretrained `input_layer.weight[:, :8]` is copied and the appended
        mask channel is set to zero. All other parameters are loaded as-is.
        """
        state = dict(state_dict)
        key = "input_layer.weight"
        current = self.state_dict()[key]
        if key not in state:
            raise KeyError(f"base state_dict missing {key}")
        base_weight = state[key]
        if base_weight.ndim != 2 or current.ndim != 2:
            raise ValueError(f"{key}: expected rank-2 weights, got base={tuple(base_weight.shape)} current={tuple(current.shape)}")
        if base_weight.shape[0] != current.shape[0]:
            raise ValueError(f"{key}: output dim mismatch base={tuple(base_weight.shape)} current={tuple(current.shape)}")
        if base_weight.shape[1] != self.base_latent_channels:
            raise ValueError(
                f"{key}: base input dim must be {self.base_latent_channels}, got {base_weight.shape[1]}"
            )
        if current.shape[1] != self.base_latent_channels + self.mask_channels:
            raise ValueError(f"{key}: current input dim mismatch {tuple(current.shape)}")
        expanded = torch.zeros_like(current)
        expanded[:, : self.base_latent_channels] = base_weight.to(dtype=expanded.dtype)
        state[key] = expanded
        for key, value in self.state_dict().items():
            if (key.startswith("mask_block_modulations.") or key.startswith("mask_output_modulation.")) and key not in state:
                state[key] = value
        return self.load_state_dict(state, strict=strict)

    def _mask_values(self, x):
        mask = x.feats[:, self.base_latent_channels : self.base_latent_channels + self.mask_channels]
        if mask.shape[1] != self.mask_channels:
            raise ValueError(f"expected {self.mask_channels} mask channels, got {tuple(mask.shape)}")
        return mask

    def _forward_backbone_with_mask_modulation(self, x):
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        mask = self._mask_values(x).to(device=h.feats.device, dtype=h.feats.dtype)
        for idx, block in enumerate(self.blocks):
            if self._use_block_modulation:
                module = self.mask_block_modulations[idx]
                mod_mask = mask.to(dtype=module.weight.dtype)
                delta = module(mod_mask).to(dtype=h.feats.dtype)
                h = h.replace(h.feats + delta)
            h = block(h)
        return h

    def _mask_tensor_for_upsample(self, x):
        mask = self._mask_values(x).to(device=x.feats.device, dtype=x.feats.dtype)
        return x.replace(mask)

    def forward(self, x):
        if self.mask_modulation == "none":
            return super().forward(x)
        h = self._forward_backbone_with_mask_modulation(x)
        mask_h = self._mask_tensor_for_upsample(x) if self._use_output_feature_modulation else None
        for block in self.upsample:
            h = block(h)
            if mask_h is not None:
                mask_h = block.sub(mask_h)
        h = h.type(x.dtype)
        h = self.out_layer(h)
        if self._use_output_feature_modulation:
            if mask_h is None:
                raise RuntimeError("mask_h unexpectedly missing for output feature modulation")
            if not torch.equal(mask_h.coords, h.coords):
                raise RuntimeError("upsampled mask coords do not match decoder output coords")
            module = self.mask_output_modulation
            if module is None:
                raise RuntimeError("mask_output_modulation is not initialized")
            delta = module(mask_h.feats.to(dtype=module.weight.dtype)).to(dtype=h.feats.dtype)
            h = h.replace(h.feats + delta)
        return self.to_representation(h)


class ElasticPartMaskedSLatMeshDecoder(SparseTransformerElasticMixin, PartMaskedSLatMeshDecoder):
    pass
