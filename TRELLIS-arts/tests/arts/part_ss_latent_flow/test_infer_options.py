import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from part_ss_eval_platform import infer_runs


def _mk_repo(tmp_path):
    """构造一个最小 repo 布局，返回 (repo_root, scan_root)。"""
    repo = tmp_path / "repo"
    # config: TRELLIS-arts/configs/arts/x/y.yaml
    cfg = repo / "TRELLIS-arts" / "configs" / "arts" / "x" / "y.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("dummy: 1\n")
    # pretrained: pretrained/a/b.safetensors
    pre = repo / "pretrained" / "a" / "b.safetensors"
    pre.parent.mkdir(parents=True)
    pre.write_bytes(b"\x00")
    # scan root: <r>/exp/ckpts/step_1.pt
    scan_root = tmp_path / "r"
    ckpt = scan_root / "exp" / "ckpts" / "step_1.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"\x00")
    return repo, scan_root


def test_list_configs(tmp_path):
    repo, _ = _mk_repo(tmp_path)
    configs = infer_runs.list_configs(repo)
    labels = [c["label"] for c in configs]
    assert "x/y.yaml" in labels
    entry = next(c for c in configs if c["label"] == "x/y.yaml")
    assert entry["path"].endswith("/TRELLIS-arts/configs/arts/x/y.yaml")
    assert Path(entry["path"]).is_file()


def test_list_configs_missing_dir(tmp_path):
    # configs/arts 不存在不应崩溃，返回空列表。
    assert infer_runs.list_configs(tmp_path / "nope") == []


def test_list_checkpoints_finds_pretrained_and_ckpts(tmp_path):
    repo, scan_root = _mk_repo(tmp_path)
    cks = infer_runs.list_checkpoints(repo, [scan_root])
    labels = [c["label"] for c in cks]
    # pretrained 命中（label 末两段）
    assert "a/b.safetensors" in labels
    # <root>/**/ckpts/*.pt 命中（label 末两段）
    assert "ckpts/step_1.pt" in labels
    # 路径都是绝对且存在
    for c in cks:
        assert Path(c["path"]).is_absolute()
        assert Path(c["path"]).is_file()
    # 已排序
    assert labels == sorted(labels)


def test_list_checkpoints_missing_root_no_crash(tmp_path):
    repo, scan_root = _mk_repo(tmp_path)
    # 混入一个不存在的 root（如本机无 /robot/data-lab），不应崩溃。
    cks = infer_runs.list_checkpoints(repo, [tmp_path / "does_not_exist", scan_root])
    labels = [c["label"] for c in cks]
    assert "a/b.safetensors" in labels
    assert "ckpts/step_1.pt" in labels


def test_list_checkpoints_no_pretrained(tmp_path, monkeypatch):
    # 完全没有 pretrained 目录也不崩溃，仅返回 scan root 的 ckpt。
    monkeypatch.setenv("SAM3D_WEIGHTS_DIR", str(tmp_path / "missing_sam3d"))
    monkeypatch.setenv("THIRD_PARTY_WEIGHTS_DIR", str(tmp_path / "missing_third_party"))
    monkeypatch.setenv("PART_SS_PLATFORM_DISABLE_DEFAULT_CKPTS", "1")
    repo = tmp_path / "repo"
    scan_root = tmp_path / "r"
    ckpt = scan_root / "exp" / "ckpts" / "step_1.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"\x00")
    cks = infer_runs.list_checkpoints(repo, [scan_root])
    assert [c["label"] for c in cks] == ["ckpts/step_1.safetensors"]


def test_list_checkpoints_cap_honored(tmp_path, monkeypatch):
    monkeypatch.setenv("PART_SS_PLATFORM_DISABLE_DEFAULT_CKPTS", "1")
    monkeypatch.setenv("SAM3D_WEIGHTS_DIR", str(tmp_path / "missing_sam3d"))
    monkeypatch.setenv("THIRD_PARTY_WEIGHTS_DIR", str(tmp_path / "missing_third_party"))
    repo = tmp_path / "repo"
    pre_dir = repo / "pretrained" / "many"
    pre_dir.mkdir(parents=True)
    for i in range(50):
        (pre_dir / f"ck_{i:03d}.pt").write_bytes(b"\x00")
    cks = infer_runs.list_checkpoints(repo, [], cap=10)
    assert len(cks) == 10

def test_list_checkpoints_prefers_ss_flow_under_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("PART_SS_PLATFORM_DISABLE_DEFAULT_CKPTS", "1")
    monkeypatch.setenv("SAM3D_WEIGHTS_DIR", str(tmp_path / "missing_sam3d"))
    monkeypatch.setenv("THIRD_PARTY_WEIGHTS_DIR", str(tmp_path / "missing_third_party"))
    repo = tmp_path / "repo"
    scan_root = tmp_path / "r"
    ckpt_dir = scan_root / "many" / "ckpts"
    ckpt_dir.mkdir(parents=True)
    for i in range(20):
        (ckpt_dir / f"step_{i:03d}.pt").write_bytes(b"\x00")
    preferred = ckpt_dir / "denoiser_step0070000.pt"
    preferred.write_bytes(b"\x00")

    cks = infer_runs.list_checkpoints(repo, [scan_root], cap=10)
    paths = [c["path"] for c in cks]
    assert str(preferred.resolve()) in paths
    assert len(cks) == 10
    preferred_entry = next(c for c in cks if c["path"] == str(preferred.resolve()))
    assert preferred_entry["label"] == "many/ckpts/denoiser_step0070000.pt"


def test_list_checkpoints_adds_explicit_extra_ckpts_before_cap(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    scan_root = tmp_path / "r"
    ckpt_dir = scan_root / "many" / "ckpts"
    ckpt_dir.mkdir(parents=True)
    for i in range(20):
        (ckpt_dir / f"step_{i:03d}.pt").write_bytes(b"\x00")
    pinned = tmp_path / "part_promptable_seg_full_M_0612-3" / "ckpts" / "latest.pt"
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"\x00")
    monkeypatch.setenv("PART_SS_PLATFORM_EXTRA_CKPTS", str(pinned))

    cks = infer_runs.list_checkpoints(repo, [scan_root], cap=5)
    paths = [c["path"] for c in cks]
    assert str(pinned.resolve()) in paths
    assert len(cks) == 5
    entry = next(c for c in cks if c["path"] == str(pinned.resolve()))
    assert entry["label"] == "part_promptable_seg_full_M_0612-3/ckpts/latest.pt"


def test_list_checkpoints_dedup(tmp_path):
    # 同一文件即使被两条规则各命中一次，也只出现一次（按 resolve 路径去重）。
    repo = tmp_path / "repo"
    # 把 pretrained 放在 scan root 下，且 scan root 的 ckpts 与 pretrained 重叠
    scan_root = repo / "pretrained"
    ckpt = scan_root / "exp" / "ckpts" / "step_1.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"\x00")
    cks = infer_runs.list_checkpoints(repo, [scan_root])
    paths = [c["path"] for c in cks]
    assert len(paths) == len(set(paths))
    assert str(ckpt.resolve()) in paths
