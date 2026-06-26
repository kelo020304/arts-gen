"""USD exporter backed by Isaac Sim's MJCF importer."""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from utils.isaaclab_launcher import launch_isaaclab_app


class IsaacUsdExportSession:
    """Persistent Isaac Sim session for repeated MJCF -> USD exports."""

    def __init__(self, *, headless: bool = True):
        self._headless = bool(headless)
        self._closed = False
        self._app_launcher = None
        self._simulation_app = None
        self._mjcf_converter_cls = None
        self._mjcf_converter_cfg_cls = None

        app_launcher = launch_isaaclab_app(headless=self._headless)
        simulation_app = getattr(app_launcher, "app", None)
        if simulation_app is None:
            raise RuntimeError("Isaac Sim SimulationApp failed to launch.")

        try:
            _ensure_mjcf_extension_enabled()
            MjcfConverter, MjcfConverterCfg = _import_mjcf_converter_modules()
        except Exception:
            simulation_app.close()
            raise

        self._app_launcher = app_launcher
        self._simulation_app = simulation_app
        self._mjcf_converter_cls = _build_mjcf_converter_cls(MjcfConverter)
        self._mjcf_converter_cfg_cls = MjcfConverterCfg

    def __enter__(self) -> "IsaacUsdExportSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def simulation_app(self):
        if self._closed or self._simulation_app is None:
            raise RuntimeError("Isaac USD export session is already closed.")
        return self._simulation_app

    def export_asset_to_usd(self, asset_name: str, assets_root: Path) -> dict[str, Any]:
        """Export ``assets/object_assets/<asset>/mjcf/<asset>.xml`` to binary USD."""
        if self._closed:
            raise RuntimeError("Isaac USD export session is already closed.")

        assets_root = Path(assets_root)
        asset_dir = assets_root / asset_name
        xml_path = asset_dir / "mjcf" / f"{asset_name}.xml"
        output_dir = asset_dir / "usd"
        output_path = output_dir / f"{asset_name}.usd"

        if not asset_dir.is_dir():
            return _error_result(
                message=f"Asset directory not found: {asset_dir}",
                usd_path=output_path,
                asset_name=asset_name,
            )
        if not xml_path.is_file():
            return _error_result(
                message=f"MJCF file not found: {xml_path}",
                usd_path=output_path,
                asset_name=asset_name,
            )

        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            converter_cfg = self._mjcf_converter_cfg_cls(
                asset_path=str(xml_path),
                usd_dir=str(output_dir),
                usd_file_name=f"{asset_name}.usd",
                fix_base=True,
                import_sites=True,
                force_usd_conversion=True,
                make_instanceable=False,
            )
            converter = self._mjcf_converter_cls(converter_cfg)

            resolved_output_path = Path(converter.usd_path).resolve()
            if not resolved_output_path.is_file():
                raise RuntimeError(f"Isaac Sim MJCF importer did not generate USD: {resolved_output_path}")

            relative_xml_path = _relative_to_asset_dir(xml_path, asset_dir)
            relative_output_path = _relative_to_asset_dir(resolved_output_path, asset_dir)
            return {
                "status": "ok",
                "usd_path": str(resolved_output_path),
                "message": f"Exported USD to {resolved_output_path}",
                "asset": asset_name,
                "xml_path": relative_xml_path,
                "output_path": relative_output_path,
            }
        except Exception as exc:
            return _error_result(
                message=str(exc),
                usd_path=output_path,
                asset_name=asset_name,
                xml_path=_relative_to_asset_dir(xml_path, asset_dir),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None


def export_asset_to_usd(asset_name: str, assets_root: Path) -> dict[str, Any]:
    """Compatibility helper for one-shot exports in non-service contexts."""
    with IsaacUsdExportSession(headless=True) as session:
        return session.export_asset_to_usd(asset_name, assets_root)


def export_asset_to_usda(asset_name: str, assets_root: Path) -> dict[str, Any]:
    """Compatibility shim for existing CLI/server imports."""
    return export_asset_to_usd(asset_name, assets_root)


def _import_mjcf_converter_modules():
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    except Exception as exc:  # pragma: no cover - depends on Isaac Sim environment
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(
            "Isaac Sim MJCF exporter could not import MjcfConverter after AppLauncher startup."
        ) from exc
    return MjcfConverter, MjcfConverterCfg


def _ensure_mjcf_extension_enabled() -> None:
    import omni.kit.app
    from isaacsim.core.utils.extensions import enable_extension

    extension_name = "isaacsim.asset.importer.mjcf"
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(extension_name):
        enable_extension(extension_name)


def _build_mjcf_converter_cls(base_cls):
    class _ConfiguredMjcfConverter(base_cls):
        """MjcfConverter with project-specific importer settings."""

        def _get_mjcf_import_config(self):
            import omni.kit.commands

            _, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
            import_config.set_import_sites(self.cfg.import_sites)
            import_config.set_make_instanceable(self.cfg.make_instanceable)
            import_config.set_instanceable_usd_path(self.usd_instanceable_meshes_path)
            import_config.set_density(self.cfg.link_density)
            import_config.set_import_inertia_tensor(self.cfg.import_inertia_tensor)
            import_config.set_fix_base(self.cfg.fix_base)
            import_config.set_self_collision(self.cfg.self_collision)
            import_config.set_convex_decomp(False)
            return import_config

    return _ConfiguredMjcfConverter


def _relative_to_asset_dir(path: Path, asset_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=asset_dir.resolve())).as_posix()


def _error_result(
    *,
    message: str,
    usd_path: Path,
    asset_name: str,
    xml_path: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": "error",
        "usd_path": str(usd_path.resolve()),
        "message": message,
        "asset": asset_name,
        "output_path": f"usd/{asset_name}.usd",
    }
    if xml_path is not None:
        result["xml_path"] = xml_path
    else:
        result["xml_path"] = f"mjcf/{asset_name}.xml"
    return result
