"""The derived GPU material table (`core/pipeline/upload/materials.py`) and the canonical material's
validation (`checks.material`): exactness properties of canonical_design.md §2.2 / §2.4."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

import raywarp as rw
from raywarp._base.named import replace_named
from raywarp.core.pipeline.derive import materials
from raywarp.scene.model import checks, tables
from raywarp.scene.model.material import Material, OpenPbrExtras, RaywarpExt

F = rw.scene.MaterialFlags
f32s = st.floats(0.0, 2.0, width=32)
units = st.floats(0.0, 1.0, width=32)


def _derive(*mats: Material) -> tuple[materials.DerivedMaterials, rw.scene.LoadReport]:
  """The device values of `mats` and the derivation's report."""
  report = rw.scene.LoadReport()
  out = materials.derive(
    tables.Materials.from_materials(mats), tables.MaterialExtras.from_materials(mats), report, volume=np.zeros(len(mats), bool)
  )
  return out, report


@given(s=f32s, ior=st.floats(1.0078125, 3.0, width=32))
def test_reflectance_is_exact_for_a_level(s, ior):
  """w = 4 s^2 with s a float32 level: r = s x 5 |ior - 1| / (ior + 1), the legacy reflectance,
  exactly (4 s^2 / 4 = s^2 and sqrt(fl(s^2)) = s in IEEE arithmetic)."""
  d, _ = _derive(Material(specular_weight=4.0 * s * s, specular_ior=ior))
  ior64 = float(np.asarray(ior, np.float32))
  assert d.reflectance[0] == np.asarray(min(s * (5.0 * abs(ior64 - 1.0) / (ior64 + 1.0)), 2.5), np.float32)


@given(s=f32s)
def test_reflectance_of_the_unreal_curve_is_exact(s):
  """SPECULAR_CURVE with w = 2 s at ior 1.5: r = sqrt(s / 2), the Unreal curve, exactly."""
  d, _ = _derive(Material(specular_weight=2.0 * s, ext=RaywarpExt(flags=F.SPECULAR_CURVE)))
  assert d.reflectance[0] == np.asarray(min(math.sqrt(s / 2.0), 2.5), np.float32)


@given(base=st.tuples(units, units, units), tint=st.tuples(units, units, units))
def test_full_transmission_takes_the_transmission_color_exactly(base, tint):
  """t == 1: the derived base color is `transmission_color` exactly (not a lerp)."""
  d, report = _derive(Material(base_color=base, transmission_weight=1.0, transmission_color=tint))
  assert tuple(d.base[0, :3]) == tuple(np.asarray(tint, np.float32))
  assert d.base[0, 3] == np.asarray(2.0**-10, np.float32) and report.approximated["gpu.material.transmission(alpha)"] == 1


@given(base=st.tuples(units, units, units), color=st.tuples(units, units, units))
def test_full_subsurface_takes_the_subsurface_color_exactly(base, color):
  """s == 1: the derived base color is `subsurface_color` exactly (not a lerp)."""
  d, _ = _derive(Material(base_color=base, extras=OpenPbrExtras(subsurface_weight=1.0, subsurface_color=color)))
  assert tuple(d.base[0, :3]) == tuple(np.asarray(color, np.float32))


@given(weight=units, roughness=units, coat_roughness=units)
def test_gltf_coat_is_unchanged_at_ior_1_5(weight, roughness, coat_roughness):
  """A COAT_GLTF coat at coat_ior 1.5 (q = 1): the device coat weight is `coat_weight` and the
  base keeps its roughness and F0."""
  mat = Material(
    specular_roughness=roughness,
    coat_weight=weight,
    coat_roughness=coat_roughness,
    coat_ior=1.5,
    ext=RaywarpExt(flags=F.COAT_GLTF),
  )
  d, report = _derive(mat)
  assert d.coat[0] == float(np.asarray(weight, np.float32)) and d.specular_roughness[0] == np.asarray(roughness, np.float32)
  assert d.reflectance[0] == 0.5 and not report.approximated


@given(weight=st.floats(0.015625, 1.0, width=32), roughness=units, coat_roughness=units, ior=st.floats(1.25, 2.0, width=32))
def test_openpbr_coat_roughens_the_base(weight, roughness, coat_roughness, ior):
  """A coupled coat roughens an untextured base (min(1, 2 c^4 + r^4)^(1/4), mixed by the weight);
  its weight stays the authored one (its IOR is its own: the device's F0((n - 1) / (n + 1))²)."""
  d, _ = _derive(Material(specular_roughness=roughness, coat_weight=weight, coat_roughness=coat_roughness, coat_ior=ior))
  r, c, w = (float(np.asarray(x, np.float32)) for x in (roughness, coat_roughness, weight))
  under = min(1.0, 2.0 * c**4 + r**4) ** 0.25
  assert d.specular_roughness[0] == pytest.approx(r + (under - r) * w, rel=1e-6, abs=1e-7)
  assert d.coat[0] == w


def test_property_tests_have_no_deadline():
  """The suite's Hypothesis profile has no per-example deadline: one stall of a busy worker
  (a full collection of its heap) must not fail a property test as flaky."""
  assert settings().deadline is None


def test_textured_roughness_is_not_roughened():
  ref = rw.scene.TextureRef(rw.scene.model.types.TextureId(0))
  d, report = _derive(
    Material(specular_roughness=1.0, coat_weight=1.0, coat_roughness=0.5).with_texture(rw.scene.Slot.SPECULAR_ROUGHNESS, ref)
  )
  assert d.specular_roughness[0] == 1.0
  assert report.approximated["gpu.material.coat_roughening(textured roughness)"] == 1


def test_subsurface_lod_and_gated_extras():
  """The subsurface color replaces the base color at weight 1 (the spec's LOD rule, reported);
  extras of a lobe that is off (fuzz color at fuzz weight 0) are not reported."""
  extras = OpenPbrExtras(
    subsurface_weight=1.0, subsurface_color=(0.1, 0.2, 0.3), subsurface_radius=2.0, fuzz_color=(0.5, 0.5, 0.5)
  )
  d, report = _derive(Material(base_color=(0.9, 0.9, 0.9), extras=extras), Material())
  np.testing.assert_allclose(d.base[0, :3], [0.1, 0.2, 0.3], rtol=1e-7)
  assert tuple(d.base[1, :3]) == tuple(np.asarray((0.8, 0.8, 0.8), np.float32))
  assert report.approximated == {"gpu.material.subsurface_weight(subsurface_color as the diffuse color)": 1}
  assert report.dropped == {"gpu.material.subsurface_radius": 1}


def test_emission_peak_and_tint():
  d, _ = _derive(Material(emission_color=(1.0, 0.5, 0.25), emission_luminance=4.0), Material())
  assert d.emission[0] == 4.0 and tuple(d.emission_tint[0]) == (1.0, 0.5, 0.25)
  assert d.emission[1] == 0.0 and tuple(d.emission_tint[1]) == (0.0, 0.0, 0.0)


def test_mask_cutoff_on_the_opacity_map():
  ref = rw.scene.TextureRef(rw.scene.model.types.TextureId(0), channel=3)
  mask = Material(geometry_opacity=0.5, ext=RaywarpExt(alpha_mode=rw.scene.AlphaMode.MASK, alpha_cutoff=0.25))
  d, _ = _derive(mask.with_texture(rw.scene.Slot.GEOMETRY_OPACITY, ref), mask)
  assert d.alpha[0] == materials.Alpha(opacity=1.0, cutoff=0.5, translucent=False)  # cutoff / factor on the texel
  assert d.alpha[1] == materials.Alpha(opacity=1.0, cutoff=0.0, translucent=False)  # 0.5 >= 0.25: the surface passes


# -- validation ------------------------------------------------------------------------------------


def _accepted(name: str, value: float) -> bool:
  """Whether float input `name` = `value` is in its range (> 0 for an IOR)."""
  low, high = checks.FLOAT_RANGES[name]
  return math.isfinite(value) and (value > 0.0 if name in checks.POSITIVE else low <= value <= high)


def _check_float(name: str, value: float | np.ndarray, mat: Material) -> None:
  """`checks.material(mat)` passes iff `value` (a float or a NumPy float32 scalar) of input `name`
  is in range."""
  if _accepted(name, float(np.asarray(value))):
    checks.material(mat, 0)
  else:
    with pytest.raises(ValueError, match=name):
      checks.material(mat, 0)


_CORE_FLOATS = sorted(checks.FLOAT_RANGES.keys() & set(Material.__dataclass_fields__))
_EXTRA_FLOATS = sorted(checks.FLOAT_RANGES.keys() & set(OpenPbrExtras.__dataclass_fields__))
_BLEND = RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND)  # a partial opacity needs a blend mode


@given(name=st.sampled_from(_CORE_FLOATS), value=st.floats(), f32=st.booleans())
def test_material_floats_accepted_exactly_in_their_range(name, value, f32):
  """`checks.material` accepts a float input iff it is finite and in its range, a Python float
  or a numpy float32 alike (a numpy scalar is not skipped)."""
  v = np.asarray(value, np.float32)[()] if f32 else value  # [()]: the NumPy scalar
  _check_float(name, v, replace_named(Material(ext=_BLEND), {name: v}))


@given(name=st.sampled_from(_EXTRA_FLOATS), value=st.floats(), f32=st.booleans())
def test_extras_floats_accepted_exactly_in_their_range(name, value, f32):
  v = np.asarray(value, np.float32)[()] if f32 else value  # [()]: the NumPy scalar
  _check_float(name, v, Material(extras=replace_named(OpenPbrExtras(), {name: v})))


def test_numpy_scalars_and_arrays_are_checked():
  """A numpy scalar out of range fails; a color that is not a tuple fails (it would not hash)."""
  f32 = np.asarray([7.0, -5.0, np.nan], np.float32)  # its items are NumPy scalars
  bad = {
    "specular_roughness": f32[0],
    "emission_luminance": f32[1],
    "transmission_weight": f32[2],
    "base_color": np.array([0.5, 0.5, 0.5]),
  }
  for name, value in bad.items():
    mat = replace_named(Material(), {name: value})  # NumPy values where floats / tuples go
    with pytest.raises(ValueError, match=r"specular_roughness|emission_luminance|transmission_weight|RGB tuple"):
      checks.material(mat, 0)


def test_default_extras_are_no_extras():
  """`extras` at every default is `extras=None`: one material, no MaterialExtras row."""
  b = rw.SceneBuilder()
  assert b.add_material(Material(extras=OpenPbrExtras())) == b.add_material(Material()) == 0
  assert len(b.build().material_extras) == 0


def test_no_coat_clamp_report_at_ior_1_5():
  """F0(1.5) is 0.04000000000000001: q = 1, so a full coat is not reported as clamped."""
  _, report = _derive(Material(coat_weight=1.0, coat_ior=1.5))
  assert not [k for k in report.approximated if k.startswith("gpu.material.coat_")]


def test_abbe_number_reported_only_with_transmission():
  extras = OpenPbrExtras(transmission_dispersion_scale=1.0, transmission_dispersion_abbe_number=30.0)
  _, off = _derive(Material(extras=extras))
  _, on = _derive(Material(transmission_weight=1.0, extras=extras))
  assert "gpu.material.transmission_dispersion_abbe_number" not in off.dropped
  assert on.dropped["gpu.material.transmission_dispersion_abbe_number"] == 1


@given(color=st.tuples(st.floats(), st.floats(), st.floats()))
def test_reflectance_colors_accepted_exactly_in_unit_range(color):
  ok = all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in color)
  if ok:
    checks.material(Material(base_color=color), 0)
  else:
    with pytest.raises(ValueError, match="base_color"):
      checks.material(Material(base_color=color), 0)
