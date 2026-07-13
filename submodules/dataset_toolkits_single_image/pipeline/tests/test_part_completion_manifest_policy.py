from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys


TOOLKIT_ROOT = Path(__file__).resolve().parents[2]


def _load_step10_module():
    module_path = TOOLKIT_ROOT / "pipeline" / "10_build_part_completion_manifest.py"
    spec = importlib.util.spec_from_file_location("step10_part_completion_manifest", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_part_completion_visibility_filters_are_not_fatal_by_default() -> None:
    module = _load_step10_module()

    skip_counter = Counter(
        {
            "no_visible_target_part": 15891,
            "no_manifest_valid_movable_parts": 48,
        }
    )

    assert module._has_fatal_skips(skip_counter, strict_zero_skips=False) is False


def test_part_completion_unexpected_skips_remain_fatal() -> None:
    module = _load_step10_module()

    skip_counter = Counter({"missing_part_complete_rgb": 1})

    assert module._has_fatal_skips(skip_counter, strict_zero_skips=False) is True


def test_part_completion_strict_zero_skips_preserves_old_contract() -> None:
    module = _load_step10_module()

    skip_counter = Counter({"no_visible_target_part": 1})

    assert module._has_fatal_skips(skip_counter, strict_zero_skips=True) is True


def test_step10_outputs_are_overwritten_on_launcher_rerun() -> None:
    script = (TOOLKIT_ROOT / "run_pipeline.sh").read_text(encoding="utf-8")

    assert 'local -a extra_args=()' in script
    assert 'cmd+=("${extra_args[@]}")' in script
    assert '10) run_step 10 "pipeline/10_build_part_completion_manifest.py" "no" "yes" "no" --overwrite ;;' in script


def test_step10_part_completion_manifest_uses_compact_local_labels() -> None:
    source = (TOOLKIT_ROOT / "pipeline" / "10_build_part_completion_manifest.py").read_text(encoding="utf-8")

    assert '"local_label": local_idx' in source
    assert '"local_label": part.label' not in source
