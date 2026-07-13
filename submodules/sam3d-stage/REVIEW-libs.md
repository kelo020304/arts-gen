# Library Review: surface_voxel + texture

**Scope reviewed**
- `/tmp/arts-recon-build/generate_surface_voxel/` (Stage A)
- `/tmp/arts-recon-build/generate_texture/` (Stage B)

**Reference cross-checked**
- `sam3d_objects/pipeline/inference_pipeline_pointmap.py` (run() at L385, preprocess_image() at L173)
- `sam3d_objects/pipeline/inference_pipeline.py` (sample_slat() L723, decode_slat() L591, merge_image_and_mask() L571)
- `notebook/inference.py` (merge_mask_to_rgba L94)

## Summary

- **HIGH**: 0
- **MEDIUM**: 3
- **LOW**: 7
- **UNCERTAIN**: 1

The pipeline wiring is correct against the reference. No runtime-breaking bug found. Three medium-severity issues affect either correctness on edge cases or feature surface that is documented but never reachable.

---

## Findings

### MED-1 — `--with-layout-postprocess` is unreachable / always errors

**File**: `generate_texture/texture/pipeline.py:142-200`, surfaced via CLI `generate_texture/texture/cli.py:63-67`

In `_run_layout_postprocess` (L171-180) the code requires `slat_input_dict["rgb_pointmap_unnorm"]`. But `slat_input_dict` is produced by `pipe.preprocess_image(rgba, pipe.slat_preprocessor)` with `pointmap=None` (L124). Per `inference_pipeline_pointmap.py:205-218`, `rgb_pointmap_unnorm` is only inserted when `pointmap is not None`. Therefore every call with `with_layout_postprocess=True` deterministically raises `ValueError`. The CLI flag, the dataclass-level docs and the README all surface a feature the user cannot exercise.

Suggested fix: either (a) remove the flag from CLI + docstring + class, or (b) plumb `pointmap_unnorm.npy` from `voxel_dir`, run `ss_preprocessor` (which does emit `rgb_pointmap_unnorm`) instead of `slat_preprocessor`, and feed that to `run_post_optimization_GS`. Today it is misleading API surface.

### MED-2 — `VoxelOutput.save` crashes on empty coord set

**File**: `generate_surface_voxel/surface_voxel/types.py:29-33`

If `ss_return_dict["coords"]` is empty (e.g. degenerate input — model produced 0 surface voxels), `surface.min()` raises numpy's `ValueError: zero-size array to reduction operation minimum`. The current message would be a confusing `min()`-from-numpy traceback, not the friendly range message you wrote.

Suggested fix:
```python
if surface.size == 0:
    raise ValueError("surface voxel coords are empty (model produced 0 voxels)")
lo, hi = int(surface.min()), int(surface.max())
```

### MED-3 — `cli.py` `_load_mask` collapses RGB-without-alpha by `.max(axis=-1)`

**File**: `generate_surface_voxel/surface_voxel/cli.py:18-26` and identical block at `generate_texture/texture/cli.py:18-25`

For an RGB image (3 channels) the code does `arr.max(axis=-1)`. This works for typical "white-on-black" masks but silently does the wrong thing if the user passes a 3-channel mask where the foreground is mid-grey on a brighter background, or where the colour channels disagree. Reference `notebook/inference.py:351-356` documents the convention as `mask = mask > 0` then `mask[..., -1]` if 3D — i.e. "take the alpha if present, else assume the value is the mask". Your code branches on shape, falls back to max, then thresholds at >127. There is no error path if the user passes a 3-channel mask that doesn't actually contain mask data in the alpha; behaviour silently differs from the reference. Also: thresholding at `>127` is inconsistent with the reference (`> 0`). For a clean binary mask this never matters; for a soft-edged saved mask it produces a slightly tighter mask than the reference would have.

Suggested fix: document the convention explicitly in `--mask` help (`"PNG; binary; if 3- or 4-channel, the last channel is used"`) and choose a single threshold consistent across stages. Consider matching the reference's `> 0` to avoid drift if a downstream component re-uses the reference utility.

---

### LOW-1 — `__main__.py` executes at import time

**File**: `generate_surface_voxel/surface_voxel/__main__.py:1-3`, `generate_texture/texture/__main__.py:1-3`

```python
from surface_voxel.cli import main
main()
```

Lacks the `if __name__ == "__main__":` guard. For `python -m surface_voxel` it works (the module is executed as `__main__`), but any accidental `import surface_voxel.__main__` from another tool will run the CLI. Cheap fix:
```python
if __name__ == "__main__":
    main()
```

### LOW-2 — Style violations vs. the build spec

Multiple "what" comments instead of "why" comments — examples:
- `generate_surface_voxel/surface_voxel/pipeline.py:92` `# (1, 3, H, W) -> (H, W, 3); see preprocess_image line ~218` is half-what / half-why and is borderline OK; mentioning the source line is good.
- `generate_surface_voxel/surface_voxel/types.py:60` `# Re-add batch column (all zeros) and restore int32 dtype expected downstream.` describes WHAT, not WHY.
- `generate_texture/texture/pipeline.py:135-138` is a "WHY" comment — well done.

No defensive try/except around internal sam3d_objects calls (good). No `os.path` usage (good — uses `pathlib`).

### LOW-3 — Missing type hints on a few internal helpers

`generate_surface_voxel/surface_voxel/cli.py:_load_rgb / _load_mask` and the texture twin have return annotations but the latter could use `np.ndarray` everywhere. Acceptable; the public API (`SurfaceVoxelPipeline.__call__`, `TexturePipeline.__call__`) is fully typed.

`generate_texture/texture/pipeline.py:_load_voxel_dir` return type `tuple[torch.IntTensor, dict]` lacks key spec for the dict — minor.

### LOW-4 — `AppearanceOutput.gs / .mesh` typed as `Any`

**File**: `generate_texture/texture/types.py:12-13`

`gs: Any | None`, `mesh: Any | None`. Justified because `sam3d_objects` doesn't expose stable types, but a `TYPE_CHECKING` block importing `Gaussian` / `MeshExtractResult` for type-checker-only hints would be cheap and improve discoverability.

### LOW-5 — `pose.json` field ordering and clarity

**File**: `generate_surface_voxel/surface_voxel/types.py:37-44`

`intrinsics` and `seed` are stored as plain lists/ints. Fine, but the README documents shapes `(1,1,4)` / `(1,3)` / `(1,3)` / `(3,3)` whereas the code does not enforce or validate those shapes on `load()`. If a downstream consumer (e.g. the wave-2 web app referenced in the task) hand-crafts a `pose.json` with `(4,)` instead of `(1,1,4)` for rotation, `VoxelOutput.load` will return a tensor of the wrong rank without complaint and a downstream operation in sam3d_objects' postprocess will fail in an inscrutable way.

Suggested fix: validate `rotation.shape[-1] == 4`, `translation.shape[-1] == 3`, `scale.shape[-1] == 3`, `intrinsics.shape == (3, 3)` inside `VoxelOutput.load`.

### LOW-6 — Stage A `_merge_mask_to_rgba` ignores image alpha if present

**File**: `generate_surface_voxel/surface_voxel/pipeline.py:25-27` (identical in texture/pipeline.py:25-27)

`image[..., :3]` discards an existing alpha channel if the caller passes RGBA. Not a bug — Stage A CLI loads with `.convert("RGB")` — but the function silently truncates without validating `image.shape[-1] in (3, 4)`. A 1-channel grayscale image would crash deep inside `np.concatenate` with an unhelpful error. Cheap to add `assert image.ndim == 3 and image.shape[-1] in (3, 4)`.

### LOW-7 — Empty-output CLI prints `"wrote: "` with no payload

**File**: `generate_texture/texture/cli.py:112-117`

```python
print("wrote: " + ", ".join(msgs) if msgs else "no outputs produced")
```

This is Python operator precedence: `("wrote: " + ", ".join(msgs)) if msgs else "no outputs produced"` — that's actually correct, but it's easy to misread. Wrap for clarity:
```python
print("wrote: " + ", ".join(msgs) if msgs else "no outputs produced")
# -> 
if msgs:
    print("wrote: " + ", ".join(msgs))
else:
    print("no outputs produced")
```

---

### UNCERTAIN-1 — Seed offset semantics (`seed + 1`)

**File**: `generate_texture/texture/pipeline.py:118-126`

The spec says: `torch.manual_seed(seed + 1)` "to avoid bit-identical to monolithic". Code does this. The reference monolithic `run()` (inference_pipeline_pointmap.py:418-419) only calls `torch.manual_seed(seed)` ONCE at the top, before BOTH `sample_sparse_structure` and `sample_slat`. So in a monolithic run, by the time `sample_slat` is called, the RNG state has already advanced (consumed by `sample_sparse_structure`). The `+1` here therefore does NOT reproduce monolithic; it produces a different (but deterministic) result. This appears to be intentional ("decouple") and is documented in the README. I flag it for the test plan only: if someone later compares split vs monolithic output expecting bit-equivalence, they'll be confused. Recommend a regression test that explicitly does `seed+1` in a monolithic run and asserts equivalence, OR rename the doc to "split seed" so the intent is unambiguous.

**Test to run**: run `texture` with `seed=N` after running `surface_voxel` with `seed=N`, and compare with a monolithic run that does `torch.manual_seed(N)` before SS and `torch.manual_seed(N+1)` before SLAT. They should be bit-equal up to floating-point determinism.

---

## Wiring checklist (verified)

| Check | Status |
|---|---|
| Stage A: `compute_pointmap(image, pointmap=None)` | ✓ pipeline.py:64 |
| Stage A: `preprocess_image(..., pointmap=pointmap)` | ✓ pipeline.py:68-70 |
| Stage A: `sample_sparse_structure` | ✓ pipeline.py:73 |
| Stage A: `pose_decoder(...)` with `scene_scale` / `scene_shift` | ✓ pipeline.py:79-85 |
| Stage A: `scale *= downsample_factor` after pose_decoder | ✓ pipeline.py:86-88 |
| Stage A: under `with self._device:` | ✓ pipeline.py:63 |
| Stage A: `torch.manual_seed(seed)` before SS sampling | ✓ pipeline.py:72 |
| `LIDRA_SKIP_INIT` set before `import sam3d_objects` | ✓ pipeline.py:12 (both packages) |
| `surface.npy` → `(N, 3) int64`, batch col dropped | ✓ types.py:27 (`coords_cpu[:, 1:].to(torch.int64).numpy()`) |
| `[0, 63]` range validated | ✓ types.py:29-33 (Stage A) and pipeline.py:42-46 (Stage B re-validates on load — good) |
| `pose.json` contains rotation/translation/scale/intrinsics/downsample_factor/seed | ✓ types.py:37-44 |
| Stage B: re-add batch col → `(N, 4) int32` | ✓ pipeline.py:48-51 |
| Stage B: `preprocess_image(rgba, pipe.slat_preprocessor)` (no pointmap arg) | ✓ pipeline.py:124 |
| Stage B: `torch.manual_seed(seed+1)` before sample_slat | ✓ pipeline.py:118-126 |
| Stage B: `sample_slat(slat_input_dict, coords, ...)` | ✓ pipeline.py:127-132 |
| Stage B: `decode_slat(slat, formats)` | ✓ pipeline.py:134 |
| Stage B: `outputs["gaussian"][0]` and `outputs["mesh"][0]` | ✓ pipeline.py:139-140 |
| Stage B: GLB export reads `vertex_attrs[:, :3]`, `.faces`, `.vertices` | ✓ types.py:33-58 |
| `pyproject` `[project] name` = `"surface_voxel"` / `"texture"` | ✓ pyproject.toml both files |
| Setuptools build backend | ✓ both pyproject.toml |
| CLI `--image / --mask` loaded via PIL, threshold present | ✓ cli.py:18-26 (both) — see MED-3 about threshold value |
| CLI `--output-dir` created | ✓ surface_voxel/cli.py:79; texture writes via `AppearanceOutput.save` |
| Mesh tensors `.detach().cpu().numpy()` before downstream use | ✓ types.py:42-43 (no `.cpu()` missing-before-`.numpy()` cases found) |
| `pointmap_unnorm` `(1, 3, H, W)` → `(H, W, 3)` | ✓ pipeline.py:95 (`full[0].permute(1, 2, 0)`) |

---

## Verdict

**PASS** — ship it.

No HIGH-severity findings. The two libraries faithfully implement the reference pipeline at the correct seams. Three MED findings:

1. `--with-layout-postprocess` should be removed or made functional (MED-1).
2. Empty-coord crash message should be friendly (MED-2).
3. Mask loader convention should be aligned with the reference and documented (MED-3).

None of MED-1/2/3 block first-pass usage with valid inputs. They affect edge cases and a documented-but-broken optional flag. Fix them in a follow-up; do not gate Wave-2 web-app integration on them.

_Reviewed: 2026-05-11_
_Reviewer: Claude (Opus 4.7)_
_Depth: standard with cross-file verification against sam3d_objects reference_
