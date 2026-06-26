"""Historical placeholder for the removed Dirichlet reference test."""

import pytest


pytest.skip(
    'Phase 8 D-18 deletes the Dirichlet implementation; the old c_factor '
    'reference test is intentionally disabled.',
    allow_module_level=True,
)
