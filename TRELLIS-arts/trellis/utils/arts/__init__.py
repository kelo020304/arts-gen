"""Arts subpackage utilities (config / lora / anchor / mask / ddp / slat_render).

Most utilities are eager-imported (pure helpers, no heavy 3D deps).
`lora_utils` is lazy because it requires optional ``peft`` and eval paths that
only need config loading should not require LoRA dependencies. `slat_render_utils`
is lazy because it pulls trellis.renderers → nvdiffrast, which is only needed by
pipeline/03_final_decode.py.
"""
import importlib

# eager (no heavy deps)
from . import config_utils
from . import anchor_utils
from . import mask_utils
from . import ddp_utils

_LAZY = {
    "lora_utils": "trellis.utils.arts.lora_utils",
    "slat_render_utils": "trellis.utils.arts.slat_render_utils",
}


def __getattr__(name):
    if name in _LAZY:
        return importlib.import_module(_LAZY[name])
    raise AttributeError(f"module 'trellis.utils.arts' has no attribute {name!r}")
