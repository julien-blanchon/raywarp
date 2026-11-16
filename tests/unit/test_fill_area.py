"""The bundle fill lists the soft area shadows' pass (`shadows.area_rays`): the OptiX build's area
program beside its render and GI programs, the software build's tracing kernel and the filter
kernel (CPU, no compile), so that turning the setting on after a refill loads, never compiles."""

from __future__ import annotations

import dataclasses

import pytest
from conftest import recipe
from fixtures.optix_codegen import preset_features, scene_flags

from raywarp.core.post.display import deferred
from raywarp.core.shade.shadows import area as area_shadows
from raywarp.core.trace.optix.device import programs
from tools.prebuilt import fill


def _build(*, area_lights: bool, supersampled: bool = False) -> fill.Build:
  """A default-recipe OptiX build of the parity scene, with or without LTC area lights."""
  f = dataclasses.replace(preset_features(recipe("default")), scene_free=True, area_lights=area_lights)
  flags = programs.program_flags(f, scene_flags())
  key = programs.ptx_key(f, flags, supersampled=supersampled, arch=89)
  return fill.Build(key=key, features=f, flags=flags, supersampled=supersampled, registers=(0, 0), arch=89)


def test_an_ltc_build_fills_its_area_program() -> None:
  """A build with LTC area lights has one area unit, keyed as the renderer looks it up
  (`launch.frame_units`); without them, or supersampled, none."""
  build = _build(area_lights=True)
  units = fill._units(build, {}, lens=False)  # the fill's unit list
  area = [u for u in units if u.area]
  assert len(area) == 1
  assert area[0].key == programs.ptx_key(build.features, build.flags, supersampled=False, arch=89, area=True)
  assert area[0].key not in {u.key for u in units if not u.area}
  assert not [u for u in fill._units(_build(area_lights=False), {}, lens=False) if u.area]
  assert fill.area_key(_build(area_lights=True, supersampled=True)) is None


def test_the_software_fill_lists_the_pass_kernels() -> None:
  """A software build with LTC area lights lists its tracing kernel's module; the filter is among
  the display kernels every shaded fill point lists (`deferred.reachable_kernels`)."""
  f = dataclasses.replace(preset_features(recipe("default")), traversal="software", area_lights=True)
  assert fill._area_kernel(f) == (area_shadows.software_kernel(f).module,)
  assert fill._area_kernel(dataclasses.replace(f, area_lights=False)) == ()
  assert area_shadows.apply_kernel in deferred.reachable_kernels()


@pytest.mark.parametrize("spp", [1, 4])
def test_an_area_key_build_admits_area_rays(spp: int) -> None:
  """Every build `area_key` gives a key for is one whose settings admit `shadows.area_rays`
  (`RenderSettings._check_area_rays`): a supersampled build has none (BUG-773: the refill
  compiled area keys of builds no renderer could use)."""
  settings = recipe("default").replace({"lighting.area_lights": "ltc", "aa.spp": spp})
  f = dataclasses.replace(preset_features(settings), scene_free=True, area_lights=True)
  supersampled = f.spp > 1 and f.any_rgb
  flags = programs.program_flags(f, scene_flags())
  key = programs.ptx_key(f, flags, supersampled=supersampled, arch=89)
  build = fill.Build(key=key, features=f, flags=flags, supersampled=supersampled, registers=(0, 0), arch=89)
  admits = True
  try:
    settings.replace({"shadows.area_rays": 4})
  except ValueError:
    admits = False
  assert (fill.area_key(build) is not None) == admits
