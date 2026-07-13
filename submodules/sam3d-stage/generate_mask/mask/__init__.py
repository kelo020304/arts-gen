from mask.types import BoxPrompt, MaskOutput
from mask.session import SessionManager, SessionEntry

__all__ = [
    "BoxPrompt", "MaskOutput",
    "MaskPipeline",
    "SessionManager", "SessionEntry",
]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy-load MaskPipeline so `python -m mask --help` (and any consumer that
    # only needs types/session) doesn't pay the cost of importing torch + sam3.
    if name == "MaskPipeline":
        from mask.pipeline import MaskPipeline
        return MaskPipeline
    raise AttributeError(f"module 'mask' has no attribute {name!r}")
