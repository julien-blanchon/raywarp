"""MDL materials without the bundled MDL SDK (an install without the build hook's output: its
import made to fail): the read raises."""

from __future__ import annotations

import functools
import sys

import pytest
from rw_helpers import bind, new_stage, quad
from rw_helpers import mdl_material as _shader

pytest.importorskip("pxr")

from pxr import Sdf

import raywarp as rw
import raywarp.formats.usd
from raywarp.formats.usd.materials.mdl import sdk


@pytest.fixture
def no_wheel(monkeypatch):
  """`import raywarp._native.mdl` fails (None entries in sys.modules) and the start goes through
  a fresh memo; both restored after. The process's real SDK (and its loaded library, which cannot
  be loaded twice) is left untouched for the tests that follow."""
  for name in [n for n in sys.modules if n.startswith("raywarp._native.mdl")]:
    monkeypatch.setitem(sys.modules, name, None)
  monkeypatch.setitem(sys.modules, "raywarp._native.mdl", None)
  monkeypatch.setattr(sdk, "_start_once", functools.cache(sdk._start_once.__wrapped__))


def test_without_the_sdk_an_mdl_stage_does_not_load(tmp_path, no_wheel):
  """No bundled MDL SDK: reading a stage with MDL materials raises with the cause (a silent grey
  read once passed for a renderer bug: the Isaac Hospital, dark)."""
  with pytest.raises(sdk.MdlSdkError, match="MDL SDK"):
    sdk.api()
  stage = new_stage(tmp_path)
  shader = _shader(stage, "/Looks/P0", "OmniPBR.mdl", "OmniPBR", metallic_constant=(Sdf.ValueTypeNames.Float, 0.0))
  bind(quad(stage, "/Q0"), shader)
  with pytest.raises(sdk.MdlSdkError) as caught:
    rw.formats.usd.read(stage)
  assert isinstance(caught.value.__cause__, ImportError)


def test_a_stage_without_mdl_needs_no_sdk(tmp_path, no_wheel):
  """The SDK starts only for a stage's first MDL material."""
  stage = new_stage(tmp_path)
  quad(stage, "/Q0")
  assert rw.formats.usd.read(stage).report.dropped == {}


def test_the_cache_key_names_the_mdl_sdk_and_the_mdl_paths(tmp_path, monkeypatch):
  """A .wpbr read with another bundled MDL SDK, or with other MDL search paths, is not reused."""
  stage = tmp_path / "s.usda"
  monkeypatch.delenv("MDL_USER_PATH", raising=False)
  base = rw.formats.usd.cache_key(stage)
  monkeypatch.setenv("MDL_USER_PATH", str(tmp_path))
  assert rw.formats.usd.cache_key(stage) != base
  monkeypatch.delenv("MDL_USER_PATH")

  monkeypatch.setattr(sdk, "version", lambda: "other")  # another bundled MDL SDK release
  assert rw.formats.usd.cache_key(stage) != base


def test_code_modules_are_not_dependencies(tmp_path, monkeypatch):
  """raywarp's MDL library and the bundled SDK's modules are code (versioned by the cache key), at
  another path on every machine: never `.wpbr` dependencies (a prepared scene went stale on every
  other machine); the stage's own modules and textures are."""
  from types import SimpleNamespace

  from raywarp.formats.usd.materials.mdl import load
  from raywarp.formats.usd.materials.mdl.baker import SdkBaker

  wheel = tmp_path / "venv" / "raywarp" / "_native" / "mdl"
  monkeypatch.setattr(sdk, "api", lambda: SimpleNamespace(system_paths=(str(wheel),)))
  baker = SdkBaker(None)
  own = [str(tmp_path / "stage" / "M.mdl"), str(tmp_path / "stage" / "tex.png")]
  baker.sources |= {str(load.LIBRARY / "Base" / "OmniPBR.mdl"), str(wheel / "nvidia" / "core_definitions.mdl"), *own}
  assert baker.dependencies() == sorted(own)


def test_a_material_with_mdl_and_preview_surfaces_raises_without_the_sdk(tmp_path, no_wheel):
  """The default material contexts choose a material's MDL output over its preview surface: without
  the SDK that is an error, never the preview (or grey) in its place."""
  from pxr import UsdShade

  stage = new_stage(tmp_path)
  mat = _shader(stage, "/Looks/Both", "OmniPBR.mdl", "OmniPBR")
  preview = UsdShade.Shader.Define(stage, "/Looks/Both/Preview")
  preview.CreateIdAttr("UsdPreviewSurface")
  mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
  bind(quad(stage, "/Q0"), mat)
  with pytest.raises(sdk.MdlSdkError):
    rw.formats.usd.read(stage)
