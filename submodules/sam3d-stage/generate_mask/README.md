# mask

Interactive box-prompted segmentation built on top of Meta's
[SAM 3](https://github.com/facebookresearch/sam3). Give it an image and a
bounding box; it gives you back a binary mask.

This is **not** a wrapper around any other segmentation library. It depends
only on `sam3`, which is expected to be installed editable in the same
environment (e.g. `sam3-process`) — `mask` imports `sam3.*` directly.

## Install

```bash
# inside the sam3-process env, with sam3 already editable-installed
pip install -e .
```

A SAM 3 image checkpoint (`sam3.pt`) must be available locally. By default
the CLI looks for `submodules/sam3/ckpt/sam3.pt` walking up from the
package; override with `--ckpt PATH` or `SAM3_CKPT_PATH=PATH`.

## CLI

```bash
python -m mask \
    --image path/to/photo.jpg \
    --box  "0.5,0.5,0.3,0.4" \
    -o     out/
# writes out/mask.png and out/prompt.json
```

Box coordinates are `cx,cy,w,h` normalized to `[0, 1]`.

## Python API

```python
from PIL import Image
from mask import MaskPipeline, BoxPrompt

pipeline = MaskPipeline(ckpt_path="submodules/sam3/ckpt/sam3.pt")
image = Image.open("photo.jpg").convert("RGB")     # MUST be PIL, not numpy
state = pipeline.embed(image)

box = BoxPrompt.from_pixel_box(120, 80, 400, 360,
                               img_w=image.width, img_h=image.height)
out = pipeline.predict_box(state, box)

out.save_png("mask.png")
print(out.score)

# Re-prompt the same image without re-embedding:
pipeline.reset_prompts(state)
out2 = pipeline.predict_box(state, BoxPrompt(0.6, 0.5, 0.2, 0.3))
```

### Sessions

`SessionManager` is a small in-memory LRU for serving multiple images
without re-running the backbone on each click. Up to 16 entries, 30-min
idle TTL by default.

```python
from mask import SessionManager

sessions = SessionManager()
sid = sessions.create(image=image, image_bytes=raw_bytes, state=state)
entry = sessions.get(sid)
out = pipeline.predict_box(entry.state, box)
```

## Note on numpy inputs

Upstream `Sam3Processor.set_image` has a bug where numpy arrays are read
with CHW assumptions (`shape[-2:]`) even though numpy images are HWC.
Passing numpy yields mask outputs with shape `(N, 1, W, 3)`. `mask` refuses
numpy at the boundary and requires `PIL.Image.Image`.
