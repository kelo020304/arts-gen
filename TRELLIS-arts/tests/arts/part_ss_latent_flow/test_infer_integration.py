import sys, os, json, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
CLI = REPO/"scripts"/"inference"/"infer_stage.py"; PY = "/home/mi/anaconda3/envs/arts-gen/bin/python"

def _run(stage, root):
    return subprocess.run([PY,str(CLI),"--stage",stage,"--object-id","o1","--root",str(root),
        "--run-id","r1","--mode","A","--view","four","--data-config","/x.yaml",
        "--part-flow-ckpt","/c","--ss-decoder-ckpt","/d"], capture_output=True, text=True,
        env={**os.environ,"INFER_DRY_RUN":"1"})

def test_gating_order_and_status(tmp_path):
    assert _run("part", tmp_path).returncode == 2            # part 在 ss 前 → 门控 exit 2
    r = _run("ss", tmp_path); assert r.returncode == 0, r.stderr   # ss dry → done
    meta = json.loads((tmp_path/"o1"/"r1"/"meta.json").read_text())
    assert meta["stage_status"]["ss"] == "done"
    assert meta["stage_status"]["part"] in ("pending","failed")    # part 那次失败前已置 running→failed
