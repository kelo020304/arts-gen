"""Inject TRELLIS-arts into sys.path so tests can `from trellis.X import Y`.

Replaces the `_setup_trellis_imports()` boilerplate previously duplicated
across 4 trainers + 1 export script. With trainer code now living inside
trellis/{trainers,datasets,utils}/arts/, only sys.path injection is needed.
"""
import os
import sys
import types
from pathlib import Path

_TRELLIS = Path(__file__).resolve().parents[2]   # TRELLIS-arts/  (conftest 在 tests/arts/ 下，2 级 up)
if str(_TRELLIS) not in sys.path:
    sys.path.insert(0, str(_TRELLIS))

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("TORCH_HOME", str(_TRELLIS.parent / "submodules" / "TRELLIS.1"))

# trellis.utils.arts.__init__.py eagerly imports lora_utils (peft) and other
# heavy deps. Dataset / model tests don't touch lora; stub the arts subpackage
# so the heavy import side-effect is bypassed and submodules can still be
# imported by their fully-qualified path (mask_utils etc.).
if 'trellis.utils.arts' not in sys.modules:
    _arts_pkg = types.ModuleType('trellis.utils.arts')
    _arts_pkg.__path__ = [str(_TRELLIS / 'trellis' / 'utils' / 'arts')]
    _arts_pkg.__package__ = 'trellis.utils.arts'
    sys.modules['trellis.utils.arts'] = _arts_pkg
