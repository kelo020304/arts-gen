"""Lazy registry for arts stage datasets.

Each stage dataset module lives in this package; we only import the requested
module on attribute access so that pulling one stage doesn't drag in the
others' dependencies.
"""
import importlib

_STAGE_MODULES = {
    "ss_flow_art":    "trellis.datasets.arts.ss_flow_art",
    "ss_flow_global_z": "trellis.datasets.arts.ss_flow_global_z",
    "slat_flow_art":  "trellis.datasets.arts.slat_flow_art",
    "part_ss_latent_flow": "trellis.datasets.arts.part_ss_latent_flow",
    "part_ss_latent_flow_single_view": "trellis.datasets.arts.part_ss_latent_flow_single_view",
    "part_mmdit": "trellis.datasets.arts.part_mmdit",
    "part_predictor": "trellis.datasets.arts.part_predictor",
}


def __getattr__(name):
    if name in _STAGE_MODULES:
        return importlib.import_module(_STAGE_MODULES[name])
    raise AttributeError(f"module 'trellis.datasets.arts' has no attribute {name!r}")
