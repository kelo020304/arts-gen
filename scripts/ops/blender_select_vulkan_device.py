#!/usr/bin/env python3
"""Configure Blender's Vulkan preferred device in the active user config."""

from __future__ import annotations

import argparse
import os
import sys

import bpy


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set Blender Vulkan preferred GPU device.")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print Blender's Vulkan preference state and exit without saving preferences.",
    )
    parser.add_argument(
        "--setup-backend-only",
        action="store_true",
        help="Only set the GPU backend to Vulkan and save preferences.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help=(
            "Zero-based index into Blender's non-AUTO gpu_preferred_device enum. "
            "This should match the Vulkan physical device order after excluding AUTO."
        ),
    )
    parser.add_argument(
        "--fallback-first",
        action="store_true",
        help="If --device-index is out of range but devices exist, select device index 0.",
    )
    parser.add_argument(
        "--allow-missing-device",
        action="store_true",
        help="If Blender exposes no non-AUTO devices, save backend preferences and exit 0.",
    )
    if "--" not in argv:
        return parser.parse_args([])
    return parser.parse_args(argv[argv.index("--") + 1 :])


def enum_items() -> list[tuple[str, str]]:
    prop = bpy.context.preferences.system.bl_rna.properties["gpu_preferred_device"]
    return [(item.identifier, item.name) for item in prop.enum_items]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    system = bpy.context.preferences.system
    try:
        system.gpu_backend = "VULKAN"
    except TypeError as exc:
        print(f"[vulkan-pref] could not set gpu_backend=VULKAN: {exc}", flush=True)

    items = enum_items()
    print(f"[vulkan-pref] backend={system.gpu_backend}", flush=True)
    print(f"[vulkan-pref] preferred_device={system.gpu_preferred_device}", flush=True)
    print(f"[vulkan-pref] enum_items={items}", flush=True)
    try:
        import gpu

        print(f"[vulkan-pref] gpu.platform.backend={gpu.platform.backend_type_get()}", flush=True)
        print(f"[vulkan-pref] gpu.platform.vendor={gpu.platform.vendor_get()}", flush=True)
        print(f"[vulkan-pref] gpu.platform.renderer={gpu.platform.renderer_get()}", flush=True)
        print(f"[vulkan-pref] gpu.platform.device_type={gpu.platform.device_type_get()}", flush=True)
    except Exception as exc:
        print(f"[vulkan-pref] gpu.platform unavailable: {exc}", flush=True)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "DRI_PRIME",
        "MESA_VK_DEVICE_SELECT",
        "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE",
        "MESA_VK_DEVICE_SELECT_DEBUG",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
        "VK_LOADER_LAYERS_ENABLE",
        "BLENDER_USER_CONFIG",
        "DISPLAY",
    ):
        print(f"[vulkan-pref] env {name}={os.environ.get(name, '')}", flush=True)

    if args.list_only:
        return 0

    if args.setup_backend_only:
        bpy.ops.wm.save_userpref()
        print("[vulkan-pref] saved Vulkan backend preference", flush=True)
        return 0

    if args.device_index is None:
        raise ValueError("--device-index is required unless --setup-backend-only is set")
    if args.device_index < 0:
        raise ValueError(f"--device-index must be >= 0, got {args.device_index}")

    device_items = [item for item in items if item[0] != "AUTO"]
    if not device_items:
        if args.allow_missing_device:
            bpy.ops.wm.save_userpref()
            print("[vulkan-pref][warn] no non-AUTO devices exposed; saved backend only", flush=True)
            return 0
        raise RuntimeError("Blender exposes no non-AUTO gpu_preferred_device entries")
    if args.device_index >= len(device_items):
        if args.fallback_first:
            print(
                f"[vulkan-pref][warn] requested device_index={args.device_index}, "
                f"but only {len(device_items)} non-AUTO device(s) are exposed; "
                "falling back to device_index=0",
                flush=True,
            )
            args.device_index = 0
        else:
            raise RuntimeError(
                f"Requested Vulkan device index {args.device_index}, but Blender exposes "
                f"{len(device_items)} non-AUTO device(s): {device_items}"
            )

    identifier, name = device_items[args.device_index]
    system.gpu_preferred_device = identifier
    bpy.ops.wm.save_userpref()
    print(
        f"[vulkan-pref] selected device_index={args.device_index} "
        f"identifier={identifier!r} name={name!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
