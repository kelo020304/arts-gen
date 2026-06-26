"""Source-level contract tests for the browser box-select splitter.

The articulator UI is plain browser JavaScript without a node test harness.
These tests keep the important user-facing contract from regressing while
remaining cheap to run in the existing pytest suite.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = ROOT / "main.js"
INDEX_HTML = ROOT / "index.html"


def test_box_select_builds_independent_part_from_all_selected_clusters() -> None:
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "function createBoxSelectionPart(" in source
    assert "BOX_SELECT_WHOLE_CLUSTER_RATIO" in source
    assert "const sourceClusters = state.clusters.filter((c) => c.mesh.visible);" in source
    assert "createBoxSelectionPart(selectedClusterNames)" in source
    assert "bestCluster" not in source
    assert "split ONLY the cluster" not in source


def test_box_select_copy_promises_independent_part() -> None:
    source = INDEX_HTML.read_text(encoding="utf-8")

    assert "框选成独立 part" in source
    assert "框内内容会自动组成新 part" in source
