"""Single-view path: model architecture is identical to the 4-view path."""
from __future__ import annotations

from trellis.models.part_flow.part_ss_latent_flow import (
    PartSSLatentFlowModel as PartSSLatentFlowSingleViewModel,
)


__all__ = ["PartSSLatentFlowSingleViewModel"]
