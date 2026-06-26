import sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import inputs_materialize as im

def test_materialize_copies_rgb_and_full_mask(tmp_path):
    src = tmp_path/"renders"/"100075"/"angle_0"/"rgb"; src.mkdir(parents=True)
    Image.new("RGB", (64, 48), (10,20,30)).save(src/"view_2.png")
    run = tmp_path/"run"
    out = im.materialize_from_paths(run, rgb_src=src/"view_2.png", view_index=2)
    assert (run/"input_rgb"/"view_2.png").is_file()
    m = np.array(Image.open(run/"input_mask.png"))
    assert m.shape==(48,64) and m.min()==255 and m.max()==255   # 全幅
    assert out["view_index"]==2
