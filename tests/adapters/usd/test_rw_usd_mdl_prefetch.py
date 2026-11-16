"""The MDL prefetch (parallel lanes, lazy bake images), the bake keys and the failure paths."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from rw_helpers import bind, new_stage, quad, write_png
from rw_helpers import material_named as _material
from rw_helpers import mdl_material as _shader

pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdShade

import raywarp.formats.usd
from raywarp.cache import paths as cache_paths
from raywarp.formats.usd.materials.mdl import bake, baker, sdk, steps
from raywarp.formats.usd.materials.mdl.baker import ERROR
from raywarp.formats.usd.materials.mdl.values import UnsupportedError

# A baked roughness (sin is not affine in the texel) scaled by `k`, through a module function.
SIN_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
float pattern(float x) {{ return math::sin(x * {k}); }}
export material {name}(uniform texture_2d T = texture_2d("./t.png", ::tex::gamma_linear))
 = let {{
  float R = pattern(tex::lookup_float4(T, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)).x);
}} in material(surface: material_surface(scattering: df::custom_curve_layer(normal_reflectivity: 0.04,
  layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: R * R, roughness_v: R * R), base: df::diffuse_reflection_bsdf())));
"""


def _stage(tmp_path, n: int = 3, bound: bool = True):
  """A stage of `n` quads, each bound (or not) to its own baked material."""
  stage = new_stage(tmp_path)
  img = (np.arange(16).astype(np.uint8).reshape(4, 4) * 16).astype(np.uint8)
  write_png(tmp_path / "t.png", img)
  for i in range(n):
    (tmp_path / f"M{i}.mdl").write_text(SIN_BODY.format(name=f"M{i}", k=1.0 + i))
    mat = _shader(stage, f"/Looks/M{i}", f"M{i}.mdl", f"M{i}")
    if bound:
      bind(quad(stage, f"/Q{i}"), mat)
  stage.Save()
  return stage


def _images(tmp_path, cache=None):
  """The built scene's images (level 0) and the read's report."""
  b = raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  s = b.build()
  return [np.asarray(s.images.level(i)) for i in range(len(s.images))], b


def test_prefetch_lanes_are_bit_identical_to_the_serial_path(tmp_path, monkeypatch):
  """Many lanes, read repeatedly, give the serial conversion's images and materials bit for bit
  (each lane works in its own transactions)."""
  _stage(tmp_path, n=12)
  monkeypatch.setattr(baker, "_lanes", lambda: 8)
  runs = [_images(tmp_path) for _ in range(3)]
  monkeypatch.setattr(baker.SdkBaker, "prepare", lambda self, ctx, stage: None)
  serial, sb = _images(tmp_path)
  for images, b in runs:
    assert len(images) == len(serial) == 12
    for a, c in zip(images, serial, strict=True):
      np.testing.assert_array_equal(a, c)
    assert [_material(b, "/Looks/" + f"M{i}") for i in range(12)] == [_material(sb, "/Looks/" + f"M{i}") for i in range(12)]


def test_unbound_shaders_are_not_prefetched_and_add_no_image(tmp_path):
  """A material no prim binds is neither baked nor imaged (the prefetch follows bindings; images
  are added when a slot binds a bake)."""
  _stage(tmp_path, n=2, bound=False)
  stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
  bind(quad(stage, "/Q0"), UsdShade.Material(stage.GetPrimAtPath("/Looks/M0")))
  stage.Save()
  cache = tmp_path / "cache"
  images, _ = _images(tmp_path, cache)
  assert len(images) == 1 and len(list((cache / "mdl").glob("*.npy"))) == 1


def test_instances_convert_their_prototype_shader_once(tmp_path):
  """Instance proxies of one prototype share its shader's conversion (memoised at the
  prototype's path)."""
  _stage(tmp_path, n=1)
  stage = Usd.Stage.Open(str(tmp_path / "scene.usda"))
  for i in range(3):
    prim = stage.DefinePrim(f"/I{i}")
    prim.GetReferences().AddInternalReference("/Q0")
    prim.SetInstanceable(True)
  stage.Save()
  b = raywarp.formats.usd.read(tmp_path / "scene.usda")
  assert b.report.counts["mdl.distilled"] == 1


def test_an_unsupported_step_is_a_reported_error_material_and_a_bug_raises(tmp_path, monkeypatch):
  """A step failing with a material error (an MDL construct the evaluators cannot represent)
  leaves the stage read going: the material is the magenta ERROR with
  `mdl.unsupported:<message>(error)`. Any other exception (a programming error) raises."""
  _stage(tmp_path, n=2)

  def unsupported(*args, **kwargs):
    raise UnsupportedError("boom")

  monkeypatch.setattr(steps, "readers", unsupported)
  b = raywarp.formats.usd.read(tmp_path / "scene.usda")
  assert b.report.dropped["mdl.unsupported:boom(error)"] == 2
  assert _material(b, "/Looks/M0") == dataclasses.replace(ERROR, name="/Looks/M0")

  def bug(*args, **kwargs):
    raise RuntimeError("bug")

  monkeypatch.setattr(steps, "readers", bug)
  with pytest.raises(RuntimeError, match="bug"):
    raywarp.formats.usd.read(tmp_path / "scene.usda")


def test_prefetch_lanes_need_no_linux_affinity(monkeypatch):
  """The prefetch's lane count reads the process's cores on every platform the bundled SDK
  ships for: macOS and Windows have no `os.sched_getaffinity`."""
  import os
  import sys

  monkeypatch.delattr(os, "sched_getaffinity", raising=False)
  monkeypatch.setattr(sys, "platform", "darwin")
  assert 1 <= baker._lanes() <= 8


def test_material_error_keys_are_one_format_on_every_path():
  """A MaterialX error reports its own key (`mtlx.<key>`), whichever step raised it; an MDL
  construct its message, cut."""
  from raywarp.formats.usd.materials.context import MaterialXError

  assert baker.unsupported_key(MaterialXError("unknown_node:ND_x")) == "mtlx.unknown_node:ND_x"
  assert baker.unsupported_key(UnsupportedError("x" * 100)) == "mdl.unsupported:" + "x" * 60


def test_a_failed_or_corrupt_bake_is_reported_not_an_abort(tmp_path, monkeypatch):
  """A corrupt cached bake is a miss (baked again); a bake whose texels the pipeline rejects is
  `bake_failed` (the field unchanged), once."""
  cache = tmp_path / "cache"
  _stage(tmp_path, n=1)
  first, _ = _images(tmp_path, cache)
  for f in (cache / "mdl").rglob("*.*"):
    f.write_bytes(b"garbage") if f.suffix == ".npy" else f.unlink()
  again, _ = _images(tmp_path, cache)
  np.testing.assert_array_equal(first[0], again[0])
  calls = []

  def reject(job, data):
    calls.append(job.key)
    raise ValueError("non-finite texels")

  monkeypatch.setattr(bake, "_finish", reject)
  b = raywarp.formats.usd.read(tmp_path / "scene.usda")
  assert b.report.dropped["mdl.bake_failed.specular_roughness(unchanged)"] == 1 and len(calls) == 1


def test_a_cached_nonfinite_bake_is_still_reported(tmp_path, monkeypatch):
  """A bake with NaN / inf texels is reported `bake_nonfinite` on the read that bakes it and on
  every read of its cache file (the flag is stored with the entry)."""
  cache = tmp_path / "cache"
  _stage(tmp_path, n=1)
  monkeypatch.setattr(bake, "_finite", lambda texels: (texels, True))
  _, first = _images(tmp_path, cache)
  key = next(k for k in first.report.approximated if k.startswith("mdl.bake_nonfinite."))
  monkeypatch.undo()
  for f in (cache / "mdl").rglob("*.*"):
    if f.suffix not in (".npy", ".nonfinite"):
      f.unlink()  # the conversion entries: the read bakes again, from the cached texels
  _, again = _images(tmp_path, cache)
  assert again.report.approximated[key] == first.report.approximated[key]


def test_bake_keys_follow_function_bodies_roots_and_the_sdk(tmp_path, monkeypatch):
  """A module function's body edited, the same module name in two roots with different bodies,
  or another SDK version: each is a new bake, not a stale or wrong one."""
  cache = tmp_path / "cache"
  _stage(tmp_path, n=1)
  first, _ = _images(tmp_path, cache)
  assert len(first) == 1
  # This is an MDL identifier, distinct from the bound USD material's /Looks/M0 path.
  (tmp_path / "M0.mdl").write_text(SIN_BODY.format(name="M0", k=3.0))
  edited, _ = _images(tmp_path, cache)
  assert len(edited) == 1, "the edited MDL fixture must compile and bake"
  assert len(list((cache / "mdl").glob("*.npy"))) == 2 and not np.array_equal(first[0], edited[0])
  monkeypatch.setattr(sdk, "version", lambda: "0.0-other")
  _images(tmp_path, cache)
  assert len(list((cache / "mdl").glob("*.npy"))) == 3


def test_one_module_name_in_two_roots_bakes_twice(tmp_path):
  """Two roots shipping `::Plastic` with different function bodies in one read: two bakes."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", (np.arange(16).astype(np.uint8).reshape(4, 4) * 16).astype(np.uint8))
  for d, k in (("a", 1.0), ("b", 2.0)):
    (tmp_path / d).mkdir()
    write_png(tmp_path / d / "t.png", (np.arange(16).astype(np.uint8).reshape(4, 4) * 16).astype(np.uint8))
    (tmp_path / d / "Plastic.mdl").write_text(SIN_BODY.format(name="Plastic", k=k))
    bind(quad(stage, f"/Q{d}"), _shader(stage, f"/Looks/{d}", f"./{d}/Plastic.mdl", "Plastic"))
  stage.Save()
  cache = tmp_path / "cache"
  images, _ = _images(tmp_path, cache)
  assert len(images) == 2 and not np.array_equal(images[0], images[1])


def test_a_wide_gamut_input_bakes_with_its_transfer_and_is_reported(tmp_path):
  """An input texture authored in g22_ap1 reaches the SDK with gamma 2.2 (from the color space
  registry); the bake keeps AP1 primaries: reported."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.full((4, 4), 128, np.uint8))
  (tmp_path / "G.mdl").write_text(
    SIN_BODY.format(name="G", k=1.0).replace('texture_2d("./t.png", ::tex::gamma_linear)', "texture_2d()")
  )
  mat = _shader(stage, "/Looks/G", "G.mdl", "G", T=(Sdf.ValueTypeNames.Asset, Sdf.AssetPath("t.png")))
  UsdShade.Shader(stage.GetPrimAtPath("/Looks/G/Shader")).GetInput("T").GetAttr().SetColorSpace("g22_ap1_scene")
  bind(quad(stage, "/A"), mat)
  stage.Save()
  images, b = _images(tmp_path)
  assert "mdl.bake_gamut:g22_ap1_scene(primaries as Rec.709)" in b.report.approximated
  # Gamma 2.2: the baked roughness r = sin((128 / 255)^2.2) (stored as r: the sqrt of alpha r^2).
  assert abs(int(images[0][0, 0, 0]) - round(255 * np.sin((128 / 255) ** 2.2))) <= 1


def test_an_unwritable_conversion_cache_warns_and_keeps_the_material(tmp_path, monkeypatch):
  """A cache directory the conversion cache cannot write to is warned about; the material is
  the cold read's."""
  _stage(tmp_path, n=1)
  cold = _material(raywarp.formats.usd.read(tmp_path / "scene.usda"), "/Looks/M0")

  def refuse(*args, **kwargs):
    raise PermissionError("read-only")

  monkeypatch.setattr(cache_paths, "remove_stale", refuse)  # every disk cache write (`write_atomic`)
  with pytest.warns(RuntimeWarning, match="conversion cache not written"):
    b = raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=tmp_path / "cache")
  assert _material(b, "/Looks/M0") == cold


def test_a_conversion_entry_that_does_not_replay_is_a_miss_and_removed(tmp_path):
  """An entry of another schema (or with a bake file gone) converts live, never grey, and is
  removed (then written again)."""
  cache = tmp_path / "cache"
  _stage(tmp_path, n=1)
  first = _material(raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache), "/Looks/M0")
  (entry,) = (cache / "mdl").rglob("conv-*/*.json")
  entry.write_text(entry.read_text().replace('"material"', '"materiel"'))
  b = raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  assert _material(b, "/Looks/M0") == first and "mdl.disk_cached" not in b.report.counts
  assert '"material"' in entry.read_text()  # written again by the live conversion


def test_conversion_entries_follow_the_converter_version(tmp_path, monkeypatch):
  """The conversion cache is keyed on the `.wpbr` key's converter version
  (`formats.usd.version.CONVERTER`): a bumped converter misses."""
  from raywarp.formats.usd import version
  from raywarp.formats.usd.materials.mdl import memo

  cache = tmp_path / "cache"
  _stage(tmp_path, n=1)
  raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  assert "mdl.disk_cached" in raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache).report.counts
  memo._code.cache_clear()
  monkeypatch.setattr(version, "CONVERTER", version.CONVERTER + 1)
  b = raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  memo._code.cache_clear()
  assert "mdl.disk_cached" not in b.report.counts


def test_a_prefetch_error_is_reported_and_the_read_converts_on_demand(tmp_path, monkeypatch):
  """A prefetch that trips on a material error is `mdl.prefetch_error:<type>`; the material is
  still converted. A programming error in the prefetch raises."""
  _stage(tmp_path, n=1)
  cold = _material(raywarp.formats.usd.read(tmp_path / "scene.usda"), "/Looks/M0")

  def unsupported(*args, **kwargs):
    raise UnsupportedError("boom")

  monkeypatch.setattr(baker.SdkBaker, "_load_all", unsupported)
  b = raywarp.formats.usd.read(tmp_path / "scene.usda")
  assert b.report.dropped["mdl.prefetch_error:UnsupportedError"] == 1 and _material(b, "/Looks/M0") == cold

  def bug(*args, **kwargs):
    raise RuntimeError("bug")

  monkeypatch.setattr(baker.SdkBaker, "_load_all", bug)
  with pytest.raises(RuntimeError, match="bug"):
    raywarp.formats.usd.read(tmp_path / "scene.usda")


LOOP_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
float pattern() {
  float s = 0.0;
  for (int i = 1; i <= 4; ++i)
    s += math::sin(state::texture_coordinate(0).x * 6.0 * float(i));
  return s;
}
export material Loop() = let {
  float R = math::saturate(pattern() * 0.2 + 0.5);
} in material(surface: material_surface(scattering: df::custom_curve_layer(normal_reflectivity: 0.04,
  layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: R * R, roughness_v: R * R), base: df::diffuse_reflection_bsdf())));
"""


def test_a_function_reading_the_state_in_its_body_bakes_a_pattern(tmp_path):
  """A non-inlined function (a loop) reading texture_coordinate in its body varies: baked over
  the surface, not one texel."""
  stage = new_stage(tmp_path)
  (tmp_path / "Loop.mdl").write_text(LOOP_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/L", "Loop.mdl", "Loop"))
  stage.Save()
  images, b = _images(tmp_path)
  assert b.report.counts["mdl.baked"] == 1 and len(images) == 1 and np.ptp(images[0][..., 0]) > 50


def test_a_nvidia_module_function_reading_the_state_bakes_a_pattern(tmp_path):
  """The varying check reads the SDK's semantics, not module names: a plain MDL function of a
  `::nvidia::` module (as vMaterials' helpers are) that reads the state bakes its pattern."""
  stage = new_stage(tmp_path)
  (tmp_path / "nvidia").mkdir()
  body = LOOP_BODY.replace("import ::state::*;", "import ::state::*;\nimport ::nvidia::core_definitions::*;")
  (tmp_path / "nvidia" / "loops.mdl").write_text(body.replace("export material Loop()", "export material Loop()"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/L", "./nvidia/loops.mdl", "Loop"))
  stage.Save()
  images, b = _images(tmp_path)
  assert b.report.counts["mdl.baked"] == 1 and len(images) == 1 and np.ptp(images[0][..., 0]) > 50


def test_a_prefetch_failing_after_the_loads_raises(tmp_path, monkeypatch):
  """A prefetch whose SDK commit raises is no material error: the read raises (never a material
  silently converted another way)."""
  from raywarp.formats.usd.materials.mdl import load

  _stage(tmp_path, n=2)

  def fail(self):
    raise RuntimeError("commit")

  monkeypatch.setattr(load.Session, "commit", fail)
  with pytest.raises(RuntimeError, match="commit"):
    raywarp.formats.usd.read(tmp_path / "scene.usda")


def test_a_second_read_replays_every_conversion(tmp_path):
  """Two reads with the disk cache: the second replays every material (the entries keep the
  search paths of their loads)."""
  cache = tmp_path / "cache"
  _stage(tmp_path, n=3)
  raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  assert raywarp.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache).report.counts["mdl.disk_cached"] == 3
