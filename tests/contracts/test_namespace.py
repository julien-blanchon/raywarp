"""The public namespace: `raywarp`'s top level is a small fixed set; the settings groups live in
`rw.settings` (with `RenderSettings`, `FAST` and `Outputs`), the scene's enums and flags in
`rw.scene`."""

from __future__ import annotations

import dataclasses
import types

import raywarp as rw

TOP_LEVEL = {
  "FAST",
  "CameraSpec",
  "GpuScene",
  "Headlight",
  "Light",
  "Lighting",
  "Material",
  "Outputs",
  "PerWorld",
  "RenderGraph",
  "RenderOutputs",
  "RenderSettings",
  "Renderer",
  "Scene",
  "SceneBuilder",
  "SceneOptions",
  "StaleBinding",
  "upload",
}
SUBMODULES = {
  "cache",
  "randomize",
  "scene",
  "settings",
  "core",
  "integrations",
  "formats",
  "data",
  "viewer",
  "kernels",
  "cli",  # imported by the tests of this session (`raywarp.cli`, `raywarp.video`), not by `import raywarp`
  "video",
}


def test_top_level_names() -> None:
  for name in rw._LAZY:  # imported on first use
    getattr(rw, name)
  public = {n for n, v in vars(rw).items() if not n.startswith("_") and not isinstance(v, types.ModuleType)}
  assert public == TOP_LEVEL
  modules = {n for n, v in vars(rw).items() if not n.startswith("_") and isinstance(v, types.ModuleType)}
  assert {"randomize", "scene", "settings"} <= modules <= SUBMODULES


def test_settings_module_holds_every_group_and_spec() -> None:
  groups = {type(rw.RenderSettings().group(f.name)).__name__ for f in dataclasses.fields(rw.RenderSettings)}
  specs = {
    "ColorOutput",
    "DepthOutput",
    "FloatOutput",
    "IdOutput",
    "LuminanceOutput",
    "MotionOutput",
    "NormalOutput",
    "ShadowOutput",
  }
  public = {n: v for n, v in vars(rw.settings).items() if not n.startswith("_") and n != "annotations"}
  exported = {n for n, v in public.items() if not isinstance(v, types.ModuleType)}
  assert exported == groups | specs | {"RenderSettings", "FAST"}
