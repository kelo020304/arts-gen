import importlib


def test_rank0_only_work_brackets_body_with_barriers(monkeypatch):
    mod = importlib.import_module("trellis.trainers.arts.part_ss_latent_flow")
    calls = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def barrier():
            calls.append("barrier")

    monkeypatch.setattr(mod, "dist", FakeDist)

    result = mod._run_rank0_only_work(
        label="eval",
        rank=0,
        work=lambda: calls.append("work") or "done",
    )

    assert result == "done"
    assert calls == ["barrier", "work", "barrier"]


def test_rank0_only_work_nonzero_rank_waits_without_body(monkeypatch):
    mod = importlib.import_module("trellis.trainers.arts.part_ss_latent_flow")
    calls = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def barrier():
            calls.append("barrier")

    monkeypatch.setattr(mod, "dist", FakeDist)

    result = mod._run_rank0_only_work(
        label="eval",
        rank=1,
        work=lambda: calls.append("work"),
    )

    assert result is None
    assert calls == ["barrier", "barrier"]


def test_ddp_wrapper_disables_static_buffer_broadcast(monkeypatch):
    mod = importlib.import_module("trellis.trainers.arts.part_ss_latent_flow")
    calls = {}

    class FakeDDP:
        def __init__(self, model, **kwargs):
            self.model = model
            calls.update(kwargs)

    monkeypatch.setattr(mod, "DDP", FakeDDP)

    model = object()
    wrapped = mod._wrap_ddp_model(model, local_rank=1)

    assert wrapped.model is model
    assert calls["device_ids"] == [1]
    assert calls["output_device"] == 1
    assert calls["broadcast_buffers"] is False
