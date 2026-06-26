from datetime import timedelta
import importlib


def test_process_group_timeout_defaults_to_two_hours(monkeypatch):
    mod = importlib.import_module("trellis.utils.dist_utils")
    monkeypatch.delenv("ARTS_DDP_TIMEOUT_MINUTES", raising=False)

    assert mod.get_process_group_timeout() == timedelta(minutes=120)


def test_process_group_timeout_can_be_overridden(monkeypatch):
    mod = importlib.import_module("trellis.utils.dist_utils")
    monkeypatch.setenv("ARTS_DDP_TIMEOUT_MINUTES", "15")

    assert mod.get_process_group_timeout() == timedelta(minutes=15)
