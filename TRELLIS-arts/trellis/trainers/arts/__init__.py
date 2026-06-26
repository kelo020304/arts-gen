"""Lazy registry for arts stage trainers + mixins.

Avoids importing all 4 stages eagerly (each stage trainer pulls ~200MB of
model classes). Stage modules are populated by Plan 09-03.
"""
import importlib

_STAGE_MODULES = {
    "ss_flow_art":    "trellis.trainers.arts.ss_flow_art",
    "slat_flow_art":  "trellis.trainers.arts.slat_flow_art",
    "part_ss_latent_flow": "trellis.trainers.arts.part_ss_latent_flow",
    "part_ss_latent_flow_single_view": "trellis.trainers.arts.part_ss_latent_flow_single_view",
    "part_predictor": "trellis.trainers.arts.part_predictor",
}


def __getattr__(name):
    if name in _STAGE_MODULES:
        return importlib.import_module(_STAGE_MODULES[name])
    raise AttributeError(f"module 'trellis.trainers.arts' has no attribute {name!r}")
