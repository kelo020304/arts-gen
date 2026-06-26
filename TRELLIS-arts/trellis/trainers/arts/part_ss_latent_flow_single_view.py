"""Single-view trainer entry for PartSSLatentFlow."""
from __future__ import annotations

from trellis.datasets.arts.part_ss_latent_flow_single_view import (
    PartSSLatentFlowSingleViewDataset,
)
from trellis.trainers.arts.part_ss_latent_flow import train as _origin_train


def train(config) -> None:
    return _origin_train(config, dataset_cls=PartSSLatentFlowSingleViewDataset)


__all__ = ["train"]
