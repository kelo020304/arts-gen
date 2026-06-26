"""D-18: Dirichlet legacy files are deleted; importing them must fail."""

import importlib
import sys

import pytest


def test_dirichlet_flow_predictor_module_missing():
    sys.modules.pop('trellis.models.part_flow.dirichlet_flow_predictor', None)
    with pytest.raises((ModuleNotFoundError, ImportError)):
        importlib.import_module('trellis.models.part_flow.dirichlet_flow_predictor')


def test_dirichlet_path_module_missing():
    sys.modules.pop('trellis.models.part_flow.dirichlet_path', None)
    with pytest.raises((ModuleNotFoundError, ImportError)):
        importlib.import_module('trellis.models.part_flow.dirichlet_path')


def test_package_init_has_no_dirichlet_exports():
    import trellis.models.part_flow as pf
    names = set(getattr(pf, '__all__', []))
    assert not any('dirichlet' in name.lower() for name in names)


def test_part_flow_predictor_still_importable_without_dirichlet():
    from trellis.models.part_flow import PartFlowPredictor
    assert PartFlowPredictor.__name__ == 'PartFlowPredictor'


def test_legacy_dirichlet_bridge_removed():
    from trellis.models.part_flow import bridges
    assert not hasattr(bridges, '_LegacyDirichletBridge')
