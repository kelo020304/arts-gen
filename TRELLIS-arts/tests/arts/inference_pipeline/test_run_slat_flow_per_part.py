import torch
import inference as inf


def test_run_slat_flow_per_part_calls_tokens_version_per_part(monkeypatch):
    calls = []
    def fake_from_tokens(cond_tokens, coords, ckpt_path, num_steps=25, seed=None):
        calls.append({"n": coords.shape[0], "seed": seed, "steps": num_steps, "ckpt": ckpt_path})
        return ("SLAT", coords.shape[0], seed)
    monkeypatch.setattr(inf, "run_slat_flow_from_tokens", fake_from_tokens)

    cond = torch.zeros(4 * 1370, 1024)
    part_coords = {"wheel_0": torch.zeros(10, 3).long(), "wheel_1": torch.zeros(20, 3).long()}
    out = inf.run_slat_flow_per_part(cond, part_coords, "ckpt.pt", num_steps=7, base_seed=42, dataset_index=3)

    assert list(out.keys()) == ["wheel_0", "wheel_1"]          # order preserved
    assert [c["n"] for c in calls] == [10, 20]
    assert all(c["steps"] == 7 and c["ckpt"] == "ckpt.pt" for c in calls)
    assert calls[0]["seed"] != calls[1]["seed"]                # distinct deterministic seeds
    assert calls[0]["seed"] == (42 + 3 * 1_000_003 + 0 * 9_176) % (2**63 - 1)
    assert calls[1]["seed"] == (42 + 3 * 1_000_003 + 1 * 9_176) % (2**63 - 1)


def test_run_slat_flow_per_part_no_seed_passes_none(monkeypatch):
    seen = []
    monkeypatch.setattr(inf, "run_slat_flow_from_tokens",
                        lambda c, co, ck, num_steps=25, seed=None: seen.append(seed))
    inf.run_slat_flow_per_part(torch.zeros(2, 4), {"a": torch.zeros(1, 3).long()}, "ck", base_seed=None)
    assert seen == [None]
