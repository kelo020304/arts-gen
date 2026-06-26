"""Make ``scripts/tools/articulator/`` importable so tests can do ``from schema import ...``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
