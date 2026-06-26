import sys
import types
from pathlib import Path


TRELLIS_ROOT = Path(__file__).resolve().parents[3]

if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

trellis_pkg = types.ModuleType("trellis")
trellis_pkg.__path__ = [str(TRELLIS_ROOT / "trellis")]
trellis_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", trellis_pkg)

for subpackage in ("datasets", "models", "modules", "trainers", "utils"):
    module = types.ModuleType(f"trellis.{subpackage}")
    module.__path__ = [str(TRELLIS_ROOT / "trellis" / subpackage)]
    module.__package__ = f"trellis.{subpackage}"
    sys.modules.setdefault(f"trellis.{subpackage}", module)
