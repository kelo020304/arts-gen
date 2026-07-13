#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


def _install_torch_attention_compat() -> None:
    try:
        import contextlib
        import importlib.util
        import types

        import torch

        if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_dynamo_compiling"):
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
                torch.compiler.is_dynamo_compiling = torch._dynamo.is_compiling
            else:
                torch.compiler.is_dynamo_compiling = lambda: False

        try:
            from torch.utils import _pytree as pytree

            if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
                pytree.register_pytree_node = pytree._register_pytree_node
        except Exception:
            pass

        if "sam3.train.data.collator" not in sys.modules:
            @dataclass
            class BatchedDatapoint:
                pass

            collator = types.ModuleType("sam3.train.data.collator")
            collator.BatchedDatapoint = BatchedDatapoint
            sys.modules["sam3.train.data.collator"] = collator

        if importlib.util.find_spec("pycocotools") is None:
            pycoco = types.ModuleType("pycocotools")
            pycoco.__path__ = []
            mask_module = types.ModuleType("pycocotools.mask")

            def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("pycocotools is unavailable in this SAM3 image-only runtime")

            for name in ("area", "decode", "encode", "frPyObjects", "merge", "toBbox"):
                setattr(mask_module, name, _unavailable)
            pycoco.mask = mask_module
            sys.modules["pycocotools"] = pycoco
            sys.modules["pycocotools.mask"] = mask_module

        if "torch.nn.attention" in sys.modules:
            return
        if hasattr(torch.nn, "attention"):
            sys.modules["torch.nn.attention"] = torch.nn.attention
            return

        from torch.backends.cuda import SDPBackend
        from torch.backends.cuda import sdp_kernel as cuda_sdp_kernel

        def _backend_enabled(backends: Any, backend_name: str) -> bool:
            if backends is None:
                return True
            if not isinstance(backends, (list, tuple, set)):
                backends = [backends]
            return any(str(item).endswith(backend_name) for item in backends)

        @contextlib.contextmanager
        def sdpa_kernel(backends: Any = None):
            # PyTorch 2.1 exposes the same controls under torch.backends.cuda.
            # Keep math enabled as a fallback because SAM3 explicitly supports
            # pre-2.2 torch where Flash Attention v2 is unavailable.
            with cuda_sdp_kernel(
                enable_flash=_backend_enabled(backends, "FLASH_ATTENTION"),
                enable_math=True,
                enable_mem_efficient=_backend_enabled(backends, "EFFICIENT_ATTENTION"),
            ):
                yield {}

        module = types.ModuleType("torch.nn.attention")
        module.SDPBackend = SDPBackend
        module.sdpa_kernel = sdpa_kernel
        sys.modules["torch.nn.attention"] = module
        torch.nn.attention = module
    except Exception:
        return


def _load_sam3_safetensors_checkpoint(model: Any, checkpoint: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    ckpt = load_file(str(checkpoint), device="cpu")
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    used: set[str] = set()

    def add(dst: str, src: str, value: torch.Tensor | None = None) -> bool:
        if dst not in target or (value is None and src not in ckpt):
            return False
        tensor = ckpt[src] if value is None else value
        if tuple(tensor.shape) != tuple(target[dst].shape):
            return False
        if dst not in mapped:
            mapped[dst] = tensor
            used.add(src)
        return True

    def add_value(dst: str, srcs: list[str], value: torch.Tensor) -> bool:
        if dst not in target or tuple(value.shape) != tuple(target[dst].shape):
            return False
        if dst not in mapped:
            mapped[dst] = value
            used.update(srcs)
        return True

    def combine_qkv(dst: str, src: str) -> None:
        for suffix in ("weight", "bias"):
            keys = [f"{src}.{proj}_proj.{suffix}" for proj in "qkv"]
            if all(key in ckpt for key in keys):
                add_value(f"{dst}.in_proj_{suffix}", keys, torch.cat([ckpt[key] for key in keys], dim=0))
            for out_name in ("o_proj", "out_proj"):
                add(f"{dst}.out_proj.{suffix}", f"{src}.{out_name}.{suffix}")

    def direct_qkv(dst: str, src: str) -> None:
        for suffix in ("weight", "bias"):
            for proj in "qkv":
                add(f"{dst}.{proj}_proj.{suffix}", f"{src}.{proj}_proj.{suffix}")
            for out_name in ("o_proj", "out_proj"):
                add(f"{dst}.out_proj.{suffix}", f"{src}.{out_name}.{suffix}")

    def map_fpn(src_root: str, dst_root: str) -> None:
        for idx in range(4):
            src = f"{src_root}.fpn_layers.{idx}"
            dst = f"{dst_root}.{idx}"
            for suffix in ("weight", "bias"):
                add(f"{dst}.conv_1x1.{suffix}", f"{src}.proj1.{suffix}")
                add(f"{dst}.conv_3x3.{suffix}", f"{src}.proj2.{suffix}")
                if idx == 0:
                    add(f"{dst}.dconv_2x2_0.{suffix}", f"{src}.scale_layers.0.{suffix}")
                    add(f"{dst}.dconv_2x2_1.{suffix}", f"{src}.scale_layers.2.{suffix}")
                if idx == 1:
                    add(f"{dst}.dconv_2x2.{suffix}", f"{src}.scale_layers.0.{suffix}")

    def map_standard_layers(src_root: str, dst_root: str, count: int, *, text_decoder: bool = False, no_layers_dst: bool = False) -> None:
        for idx in range(count):
            src = f"{src_root}.{idx}" if src_root.endswith(".layers") else f"{src_root}.layers.{idx}"
            dst = f"{dst_root}.{idx}" if no_layers_dst else f"{dst_root}.layers.{idx}"
            combine_qkv(f"{dst}.self_attn", f"{src}.self_attn")
            if text_decoder:
                combine_qkv(f"{dst}.cross_attn", f"{src}.vision_cross_attn")
                combine_qkv(f"{dst}.ca_text", f"{src}.text_cross_attn")
                norms = {
                    "self_attn_layer_norm": "norm2",
                    "vision_cross_attn_layer_norm": "norm1",
                    "text_cross_attn_layer_norm": "catext_norm",
                    "mlp_layer_norm": "norm3",
                }
            else:
                combine_qkv(f"{dst}.cross_attn_image", f"{src}.cross_attn")
                norms = {"layer_norm1": "norm1", "layer_norm2": "norm2", "layer_norm3": "norm3"}
            for suffix in ("weight", "bias"):
                add(f"{dst}.linear1.{suffix}", f"{src}.mlp.fc1.{suffix}")
                add(f"{dst}.linear2.{suffix}", f"{src}.mlp.fc2.{suffix}")
                for src_norm, dst_norm in norms.items():
                    add(f"{dst}.{dst_norm}.{suffix}", f"{src}.{src_norm}.{suffix}")

    add("backbone.vision_backbone.trunk.patch_embed.proj.weight", "detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight")
    pos_src = "detector_model.vision_encoder.backbone.embeddings.position_embeddings"
    if pos_src in ckpt and "backbone.vision_backbone.trunk.pos_embed" in target:
        pos = target["backbone.vision_backbone.trunk.pos_embed"].clone()
        pos[:, 1:, :] = ckpt[pos_src]
        add_value("backbone.vision_backbone.trunk.pos_embed", [pos_src], pos)
    for suffix in ("weight", "bias"):
        add(f"backbone.vision_backbone.trunk.ln_pre.{suffix}", f"detector_model.vision_encoder.backbone.layer_norm.{suffix}")

    for idx in range(32):
        src = f"detector_model.vision_encoder.backbone.layers.{idx}"
        dst = f"backbone.vision_backbone.trunk.blocks.{idx}"
        for suffix in ("weight", "bias"):
            keys = [f"{src}.attention.{proj}_proj.{suffix}" for proj in "qkv"]
            if all(key in ckpt for key in keys):
                add_value(f"{dst}.attn.qkv.{suffix}", keys, torch.cat([ckpt[key] for key in keys], dim=0))
            add(f"{dst}.attn.proj.{suffix}", f"{src}.attention.o_proj.{suffix}")
            add(f"{dst}.norm1.{suffix}", f"{src}.layer_norm1.{suffix}")
            add(f"{dst}.norm2.{suffix}", f"{src}.layer_norm2.{suffix}")
            add(f"{dst}.mlp.fc1.{suffix}", f"{src}.mlp.fc1.{suffix}")
            add(f"{dst}.mlp.fc2.{suffix}", f"{src}.mlp.fc2.{suffix}")

    map_fpn("detector_model.vision_encoder.neck", "backbone.vision_backbone.convs")
    map_fpn("tracker_neck", "backbone.vision_backbone.sam2_convs")

    add("backbone.language_backbone.encoder.token_embedding.weight", "detector_model.text_encoder.text_model.embeddings.token_embedding.weight")
    add("backbone.language_backbone.encoder.positional_embedding", "detector_model.text_encoder.text_model.embeddings.position_embedding.weight")
    for suffix in ("weight", "bias"):
        add(f"backbone.language_backbone.encoder.ln_final.{suffix}", f"detector_model.text_encoder.text_model.final_layer_norm.{suffix}")
        add(f"backbone.language_backbone.resizer.{suffix}", f"detector_model.text_projection.{suffix}")
    if "detector_model.text_encoder.text_projection.weight" in ckpt:
        add_value(
            "backbone.language_backbone.encoder.text_projection",
            ["detector_model.text_encoder.text_projection.weight"],
            ckpt["detector_model.text_encoder.text_projection.weight"].t(),
        )
    for idx in range(24):
        src = f"detector_model.text_encoder.text_model.encoder.layers.{idx}"
        dst = f"backbone.language_backbone.encoder.transformer.resblocks.{idx}"
        combine_qkv(f"{dst}.attn", f"{src}.self_attn")
        for suffix in ("weight", "bias"):
            add(f"{dst}.ln_1.{suffix}", f"{src}.layer_norm1.{suffix}")
            add(f"{dst}.ln_2.{suffix}", f"{src}.layer_norm2.{suffix}")
            add(f"{dst}.mlp.c_fc.{suffix}", f"{src}.mlp.fc1.{suffix}")
            add(f"{dst}.mlp.c_proj.{suffix}", f"{src}.mlp.fc2.{suffix}")

    map_standard_layers("detector_model.detr_encoder", "transformer.encoder", 6)
    map_standard_layers("detector_model.detr_decoder", "transformer.decoder", 6, text_decoder=True)
    map_standard_layers("detector_model.geometry_encoder.layers", "geometry_encoder.encode", 3, no_layers_dst=True)
    for suffix in ("weight", "bias"):
        add(f"transformer.decoder.norm.{suffix}", f"detector_model.detr_decoder.output_layer_norm.{suffix}")
        add(f"transformer.decoder.presence_token_out_norm.{suffix}", f"detector_model.detr_decoder.presence_layer_norm.{suffix}")
        add(f"geometry_encoder.encode_norm.{suffix}", f"detector_model.geometry_encoder.output_layer_norm.{suffix}")
        add(f"geometry_encoder.norm.{suffix}", f"detector_model.geometry_encoder.prompt_layer_norm.{suffix}")
        add(f"geometry_encoder.img_pre_norm.{suffix}", f"detector_model.geometry_encoder.vision_layer_norm.{suffix}")

    for name in ("query_embed", "reference_points", "presence_token"):
        add(f"transformer.decoder.{name}.weight", f"detector_model.detr_decoder.{name}.weight")
    for layer_idx in range(3):
        for suffix in ("weight", "bias"):
            add(f"transformer.decoder.bbox_embed.layers.{layer_idx}.{suffix}", f"detector_model.detr_decoder.box_head.layer{layer_idx + 1}.{suffix}")
            add(f"transformer.decoder.presence_token_head.layers.{layer_idx}.{suffix}", f"detector_model.detr_decoder.presence_head.layer{layer_idx + 1}.{suffix}")
    for layer_idx in range(2):
        for suffix in ("weight", "bias"):
            add(f"transformer.decoder.ref_point_head.layers.{layer_idx}.{suffix}", f"detector_model.detr_decoder.ref_point_head.layer{layer_idx + 1}.{suffix}")
            add(f"transformer.decoder.boxRPB_embed_x.layers.{layer_idx}.{suffix}", f"detector_model.detr_decoder.box_rpb_embed_x.layer{layer_idx + 1}.{suffix}")
            add(f"transformer.decoder.boxRPB_embed_y.layers.{layer_idx}.{suffix}", f"detector_model.detr_decoder.box_rpb_embed_y.layer{layer_idx + 1}.{suffix}")

    for name in (
        "label_embed",
        "cls_embed",
        "points_direct_project",
        "points_pool_project",
        "points_pos_enc_project",
        "boxes_direct_project",
        "boxes_pool_project",
        "boxes_pos_enc_project",
        "final_proj",
    ):
        for suffix in ("weight", "bias"):
            add(f"geometry_encoder.{name}.{suffix}", f"detector_model.geometry_encoder.{name}.{suffix}")

    for suffix in ("weight", "bias"):
        add(f"dot_prod_scoring.hs_proj.{suffix}", f"detector_model.dot_product_scoring.query_proj.{suffix}")
        add(f"dot_prod_scoring.prompt_proj.{suffix}", f"detector_model.dot_product_scoring.text_proj.{suffix}")
        add(f"dot_prod_scoring.prompt_mlp.layers.0.{suffix}", f"detector_model.dot_product_scoring.text_mlp.layer1.{suffix}")
        add(f"dot_prod_scoring.prompt_mlp.layers.1.{suffix}", f"detector_model.dot_product_scoring.text_mlp.layer2.{suffix}")
        add(f"dot_prod_scoring.prompt_mlp.out_norm.{suffix}", f"detector_model.dot_product_scoring.text_mlp_out_norm.{suffix}")
        add(f"segmentation_head.cross_attn_norm.{suffix}", f"detector_model.mask_decoder.prompt_cross_attn_norm.{suffix}")
        add(f"segmentation_head.semantic_seg_head.{suffix}", f"detector_model.mask_decoder.semantic_projection.{suffix}")
        add(f"segmentation_head.instance_seg_head.{suffix}", f"detector_model.mask_decoder.instance_projection.{suffix}")
    combine_qkv("segmentation_head.cross_attend_prompt", "detector_model.mask_decoder.prompt_cross_attn")
    for idx in range(3):
        for suffix in ("weight", "bias"):
            add(f"segmentation_head.pixel_decoder.conv_layers.{idx}.{suffix}", f"detector_model.mask_decoder.pixel_decoder.conv_layers.{idx}.{suffix}")
            add(f"segmentation_head.pixel_decoder.norms.{idx}.{suffix}", f"detector_model.mask_decoder.pixel_decoder.norms.{idx}.{suffix}")
            add(f"segmentation_head.mask_predictor.mask_embed.layers.{idx}.{suffix}", f"detector_model.mask_decoder.mask_embedder.layers.{idx}.{suffix}")

    prefix = "inst_interactive_predictor.model"
    for src, dst in (
        ("memory_temporal_positional_encoding", "maskmem_tpos_enc"),
        ("no_memory_embedding", "no_mem_embed"),
        ("no_memory_positional_encoding", "no_mem_pos_enc"),
        ("no_object_pointer", "no_obj_ptr"),
        ("occlusion_spatial_embedding_parameter", "no_obj_embed_spatial"),
    ):
        add(f"{prefix}.{dst}", f"tracker_model.{src}")
    for suffix in ("weight", "bias"):
        add(f"{prefix}.mask_downsample.{suffix}", f"tracker_model.mask_downsample.{suffix}")
        add(f"{prefix}.obj_ptr_tpos_proj.{suffix}", f"tracker_model.temporal_positional_encoding_projection_layer.{suffix}")
        add(f"{prefix}.obj_ptr_proj.layers.0.{suffix}", f"tracker_model.object_pointer_proj.layers.0.{suffix}")
        add(f"{prefix}.obj_ptr_proj.layers.1.{suffix}", f"tracker_model.object_pointer_proj.proj_in.{suffix}")
        add(f"{prefix}.obj_ptr_proj.layers.2.{suffix}", f"tracker_model.object_pointer_proj.proj_out.{suffix}")
        add(f"{prefix}.maskmem_backbone.pix_feat_proj.{suffix}", f"tracker_model.memory_encoder.feature_projection.{suffix}")
        add(f"{prefix}.maskmem_backbone.out_proj.{suffix}", f"tracker_model.memory_encoder.projection.{suffix}")

    for idx in range(4):
        src = f"tracker_model.memory_attention.layers.{idx}"
        dst = f"{prefix}.transformer.encoder.layers.{idx}"
        direct_qkv(f"{dst}.self_attn", f"{src}.self_attn")
        direct_qkv(f"{dst}.cross_attn_image", f"{src}.cross_attn_image")
        for suffix in ("weight", "bias"):
            add(f"{dst}.linear1.{suffix}", f"{src}.linear1.{suffix}")
            add(f"{dst}.linear2.{suffix}", f"{src}.linear2.{suffix}")
            for src_norm, dst_norm in (("layer_norm1", "norm1"), ("layer_norm2", "norm2"), ("layer_norm3", "norm3")):
                add(f"{dst}.{dst_norm}.{suffix}", f"{src}.{src_norm}.{suffix}")
    for suffix in ("weight", "bias"):
        add(f"{prefix}.transformer.encoder.norm.{suffix}", f"tracker_model.memory_attention.layer_norm.{suffix}")

    for idx, base_idx in enumerate((0, 3, 6, 9)):
        for suffix in ("weight", "bias"):
            add(f"{prefix}.maskmem_backbone.mask_downsampler.encoder.{base_idx}.{suffix}", f"tracker_model.memory_encoder.mask_downsampler.layers.{idx}.conv.{suffix}")
            add(f"{prefix}.maskmem_backbone.mask_downsampler.encoder.{base_idx + 1}.{suffix}", f"tracker_model.memory_encoder.mask_downsampler.layers.{idx}.layer_norm.{suffix}")
    for suffix in ("weight", "bias"):
        add(f"{prefix}.maskmem_backbone.mask_downsampler.encoder.12.{suffix}", f"tracker_model.memory_encoder.mask_downsampler.final_conv.{suffix}")
    for idx in range(2):
        for suffix in ("weight", "bias"):
            add(f"{prefix}.maskmem_backbone.fuser.layers.{idx}.dwconv.{suffix}", f"tracker_model.memory_encoder.memory_fuser.layers.{idx}.depthwise_conv.{suffix}")
            add(f"{prefix}.maskmem_backbone.fuser.layers.{idx}.norm.{suffix}", f"tracker_model.memory_encoder.memory_fuser.layers.{idx}.layer_norm.{suffix}")
            add(f"{prefix}.maskmem_backbone.fuser.layers.{idx}.pwconv1.{suffix}", f"tracker_model.memory_encoder.memory_fuser.layers.{idx}.pointwise_conv1.{suffix}")
            add(f"{prefix}.maskmem_backbone.fuser.layers.{idx}.pwconv2.{suffix}", f"tracker_model.memory_encoder.memory_fuser.layers.{idx}.pointwise_conv2.{suffix}")
        add(f"{prefix}.maskmem_backbone.fuser.layers.{idx}.gamma", f"tracker_model.memory_encoder.memory_fuser.layers.{idx}.scale")

    add(f"{prefix}.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", "tracker_model.prompt_encoder.shared_embedding.positional_embedding")
    add(f"{prefix}.sam_prompt_encoder.no_mask_embed.weight", "tracker_model.prompt_encoder.no_mask_embed.weight")
    add(f"{prefix}.sam_prompt_encoder.not_a_point_embed.weight", "tracker_model.prompt_encoder.not_a_point_embed.weight")
    if "tracker_model.prompt_encoder.point_embed.weight" in ckpt:
        point_embed = ckpt["tracker_model.prompt_encoder.point_embed.weight"]
        for idx in range(min(4, int(point_embed.shape[0]))):
            add_value(f"{prefix}.sam_prompt_encoder.point_embeddings.{idx}.weight", ["tracker_model.prompt_encoder.point_embed.weight"], point_embed[idx : idx + 1])
    for suffix in ("weight", "bias"):
        add(f"{prefix}.sam_prompt_encoder.mask_downscaling.0.{suffix}", f"tracker_model.prompt_encoder.mask_embed.conv1.{suffix}")
        add(f"{prefix}.sam_prompt_encoder.mask_downscaling.1.{suffix}", f"tracker_model.prompt_encoder.mask_embed.layer_norm1.{suffix}")
        add(f"{prefix}.sam_prompt_encoder.mask_downscaling.3.{suffix}", f"tracker_model.prompt_encoder.mask_embed.conv2.{suffix}")
        add(f"{prefix}.sam_prompt_encoder.mask_downscaling.4.{suffix}", f"tracker_model.prompt_encoder.mask_embed.layer_norm2.{suffix}")
        add(f"{prefix}.sam_prompt_encoder.mask_downscaling.6.{suffix}", f"tracker_model.prompt_encoder.mask_embed.conv3.{suffix}")
        add(f"{prefix}.sam_mask_decoder.conv_s0.{suffix}", f"tracker_model.mask_decoder.conv_s0.{suffix}")
        add(f"{prefix}.sam_mask_decoder.conv_s1.{suffix}", f"tracker_model.mask_decoder.conv_s1.{suffix}")
        add(f"{prefix}.sam_mask_decoder.output_upscaling.0.{suffix}", f"tracker_model.mask_decoder.upscale_conv1.{suffix}")
        add(f"{prefix}.sam_mask_decoder.output_upscaling.1.{suffix}", f"tracker_model.mask_decoder.upscale_layer_norm.{suffix}")
        add(f"{prefix}.sam_mask_decoder.output_upscaling.3.{suffix}", f"tracker_model.mask_decoder.upscale_conv2.{suffix}")
    for name in ("iou_token", "mask_tokens", "obj_score_token"):
        add(f"{prefix}.sam_mask_decoder.{name}.weight", f"tracker_model.mask_decoder.{name}.weight")
    for idx in range(2):
        src = f"tracker_model.mask_decoder.transformer.layers.{idx}"
        dst = f"{prefix}.sam_mask_decoder.transformer.layers.{idx}"
        for attention in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            direct_qkv(f"{dst}.{attention}", f"{src}.{attention}")
        for suffix in ("weight", "bias"):
            add(f"{dst}.mlp.lin1.{suffix}", f"{src}.mlp.proj_in.{suffix}")
            add(f"{dst}.mlp.lin2.{suffix}", f"{src}.mlp.proj_out.{suffix}")
            for src_norm, dst_norm in (("layer_norm1", "norm1"), ("layer_norm2", "norm2"), ("layer_norm3", "norm3"), ("layer_norm4", "norm4")):
                add(f"{dst}.{dst_norm}.{suffix}", f"{src}.{src_norm}.{suffix}")
    direct_qkv(f"{prefix}.sam_mask_decoder.transformer.final_attn_token_to_image", "tracker_model.mask_decoder.transformer.final_attn_token_to_image")
    for suffix in ("weight", "bias"):
        add(f"{prefix}.sam_mask_decoder.transformer.norm_final_attn.{suffix}", f"tracker_model.mask_decoder.transformer.layer_norm_final_attn.{suffix}")
    for layer_idx, src_name in enumerate(("layers.0", "proj_in", "proj_out")):
        for suffix in ("weight", "bias"):
            add(f"{prefix}.sam_mask_decoder.iou_prediction_head.layers.{layer_idx}.{suffix}", f"tracker_model.mask_decoder.iou_prediction_head.{src_name}.{suffix}")
            add(f"{prefix}.sam_mask_decoder.pred_obj_score_head.layers.{layer_idx}.{suffix}", f"tracker_model.mask_decoder.pred_obj_score_head.{src_name}.{suffix}")
        for mlp_idx in range(4):
            for suffix in ("weight", "bias"):
                add(f"{prefix}.sam_mask_decoder.output_hypernetworks_mlps.{mlp_idx}.layers.{layer_idx}.{suffix}", f"tracker_model.mask_decoder.output_hypernetworks_mlps.{mlp_idx}.{src_name}.{suffix}")

    incompatible = model.load_state_dict(mapped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    return {
        "format": "safetensors_hf_sam3_compat",
        "source_keys": len(ckpt),
        "mapped_keys": len(mapped),
        "used_source_keys": len(used),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_sample": missing[:40],
        "unexpected_key_sample": unexpected[:40],
    }


class SegmentTextRequest(BaseModel):
    image_path: str
    prompt: str
    confidence_threshold: float | None = None


class SegmentPointsRequest(BaseModel):
    image_path: str
    points: list[list[float]] = Field(default_factory=list)
    point_labels: list[int] = Field(default_factory=list)
    multimask_output: bool = True


def _mask_png_data_url(mask: np.ndarray) -> str:
    arr = (mask.astype(bool) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _to_numpy(value: Any) -> np.ndarray:
    def tensor_to_numpy(tensor: Any) -> np.ndarray:
        tensor = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor.cpu()
        dtype_name = str(getattr(tensor, "dtype", ""))
        if "bfloat16" in dtype_name or "float16" in dtype_name:
            tensor = tensor.float()
        return tensor.numpy()

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return tensor_to_numpy(value)
    except Exception:
        pass
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return tensor_to_numpy(value)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return tensor_to_numpy(value)
    return np.asarray(value)


def _normalize_masks(value: Any) -> np.ndarray:
    masks = _to_numpy(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"expected masks [N,H,W], got {masks.shape}")
    return masks.astype(bool)


def _normalize_boxes(value: Any) -> list[list[float]]:
    boxes = _to_numpy(value)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape(-1, 4)
    return [[float(x) for x in row] for row in boxes]


def _normalize_scores(value: Any) -> list[float]:
    scores = _to_numpy(value)
    if scores.size == 0:
        return []
    return [float(x) for x in scores.reshape(-1)]


def _format_candidates(masks: Any, boxes: Any = None, scores: Any = None, *, limit: int = 12) -> dict[str, Any]:
    np_masks = _normalize_masks(masks)
    box_list = _normalize_boxes(boxes) if boxes is not None else [[] for _ in range(len(np_masks))]
    score_list = _normalize_scores(scores) if scores is not None else [0.0 for _ in range(len(np_masks))]
    order = np.argsort(np.asarray(score_list, dtype=np.float32))[::-1] if score_list else np.arange(len(np_masks))
    candidates = []
    for rank, idx in enumerate(order[:limit].tolist()):
        mask = np_masks[int(idx)]
        candidates.append(
            {
                "rank": rank,
                "source_index": int(idx),
                "score": float(score_list[int(idx)]) if int(idx) < len(score_list) else None,
                "box_xyxy": box_list[int(idx)] if int(idx) < len(box_list) else None,
                "area": int(mask.sum()),
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "mask_data_url": _mask_png_data_url(mask),
            }
        )
    return {"candidates": candidates, "count": len(candidates)}


class Sam3Runtime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.loaded = False
        self.load_error: str | None = None
        self.model = None
        self.processor = None
        self.device = args.device
        self.cache_key: tuple[str, float, int] | None = None
        self.cache_state: dict[str, Any] | None = None
        self.loaded_unix: float | None = None
        self.import_checked = False
        self.import_error: str | None = None
        self.load_stats: dict[str, Any] | None = None

    def probe_imports(self) -> bool:
        if self.import_checked:
            return self.import_error is None
        try:
            _install_torch_attention_compat()
            sam3_root = Path(self.args.sam3_root).resolve()
            if str(sam3_root) not in sys.path:
                sys.path.insert(0, str(sam3_root))
            from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor  # noqa: F401
            from sam3.model_builder import build_sam3_image_model as _build_sam3_image_model  # noqa: F401

            self.import_error = None
        except Exception as exc:
            self.import_error = repr(exc)
        self.import_checked = True
        return self.import_error is None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self.lock:
            if self.loaded:
                return
            try:
                if not self.probe_imports():
                    raise RuntimeError(f"SAM3 import probe failed: {self.import_error}")
                sam3_root = Path(self.args.sam3_root).resolve()
                if str(sam3_root) not in sys.path:
                    sys.path.insert(0, str(sam3_root))
                checkpoint = Path(self.args.checkpoint).resolve()
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
                bpe_path = sam3_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                if not bpe_path.is_file():
                    bpe_path = sam3_root / "assets/bpe_simple_vocab_16e6.txt.gz"
                if not bpe_path.is_file():
                    raise FileNotFoundError(f"SAM3 BPE vocab not found under {sam3_root}")

                _install_torch_attention_compat()
                import torch
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model

                device = self.args.device
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                self.device = device
                if checkpoint.suffix.lower() == ".safetensors":
                    self.model = build_sam3_image_model(
                        bpe_path=str(bpe_path),
                        device="cpu",
                        checkpoint_path=None,
                        load_from_HF=False,
                        enable_inst_interactivity=True,
                    )
                    self.load_stats = _load_sam3_safetensors_checkpoint(self.model, checkpoint)
                    if device == "cuda":
                        self.model = self.model.cuda()
                    self.model.eval()
                else:
                    self.model = build_sam3_image_model(
                        bpe_path=str(bpe_path),
                        device=device,
                        checkpoint_path=str(checkpoint),
                        load_from_HF=False,
                        enable_inst_interactivity=True,
                    )
                    self.load_stats = {"format": "torch_checkpoint", "checkpoint": str(checkpoint)}
                self.processor = Sam3Processor(
                    self.model,
                    device=device,
                    confidence_threshold=float(self.args.confidence_threshold),
                )
                self.loaded = True
                self.load_error = None
                self.loaded_unix = time.time()
            except Exception as exc:
                self.load_error = repr(exc)
                raise

    def _state_for_image(self, image_path: str) -> tuple[Image.Image, dict[str, Any]]:
        self.ensure_loaded()
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        stat = path.stat()
        key = (str(path), float(stat.st_mtime), int(stat.st_size))
        image = Image.open(path).convert("RGB")
        with self.lock:
            if self.cache_key != key or self.cache_state is None:
                self.cache_state = self.processor.set_image(image)
                self.cache_key = key
            state = dict(self.cache_state)
            if "backbone_out" in state:
                state["backbone_out"] = dict(state["backbone_out"])
        return image, state

    def segment_text(self, req: SegmentTextRequest) -> dict[str, Any]:
        image, state = self._state_for_image(req.image_path)
        if req.confidence_threshold is not None:
            self.processor.set_confidence_threshold(float(req.confidence_threshold))
        out = self.processor.set_text_prompt(prompt=req.prompt, state=state)
        formatted = _format_candidates(out["masks"], out.get("boxes"), out.get("scores"))
        return {
            "ok": True,
            "mode": "text",
            "prompt": req.prompt,
            "image_size": [image.width, image.height],
            "device": self.device,
            **formatted,
        }

    def segment_points(self, req: SegmentPointsRequest) -> dict[str, Any]:
        image, state = self._state_for_image(req.image_path)
        if len(req.points) != len(req.point_labels):
            raise ValueError("points and point_labels length mismatch")
        if not req.points:
            raise ValueError("at least one point is required")
        points = np.asarray(req.points, dtype=np.float32)
        labels = np.asarray(req.point_labels, dtype=np.int32)
        masks, scores, _logits = self.model.predict_inst(
            state,
            point_coords=points,
            point_labels=labels,
            multimask_output=bool(req.multimask_output),
        )
        formatted = _format_candidates(masks, scores=scores)
        return {
            "ok": True,
            "mode": "points",
            "points": points.tolist(),
            "point_labels": labels.tolist(),
            "image_size": [image.width, image.height],
            "device": self.device,
            **formatted,
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    runtime = Sam3Runtime(args)
    app = FastAPI(title="Fridge Workbench SAM3 Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        import_ok = runtime.probe_imports()
        return {
            "ok": True,
            "import_ok": import_ok,
            "import_error": runtime.import_error,
            "loaded": runtime.loaded,
            "load_error": runtime.load_error,
            "device": runtime.device,
            "sam3_root": str(Path(args.sam3_root).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "loaded_unix": runtime.loaded_unix,
            "load_stats": runtime.load_stats,
        }

    @app.post("/segment_text")
    def segment_text(req: SegmentTextRequest) -> dict[str, Any]:
        try:
            return runtime.segment_text(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.post("/segment_points")
    def segment_points(req: SegmentPointsRequest) -> dict[str, Any]:
        try:
            return runtime.segment_points(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

    @app.exception_handler(HTTPException)
    def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})

    @app.exception_handler(Exception)
    def exception_handler(_request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"ok": False, "detail": repr(exc)})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(Path(args.sam3_root).resolve()))
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
