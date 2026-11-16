"""MuJoCo materials as canonical OpenPBR materials (canonical_design.md §3.1): the mapping, the
alias identity with the device tables."""

from __future__ import annotations

import os
from pathlib import Path

import hypothesis
import hypothesis.strategies as st
import mujoco
import numpy as np
import pytest
from conftest import REPO

from raywarp.core.pipeline.derive.materials import derive
from raywarp.integrations.mujoco.materials import openpbr_materials
from raywarp.scene.model.normal_map import NORMAL_UNPACK
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import MaterialExtras, Materials
from raywarp.scene.model.types import CHANNEL_RGB, AlphaMode, MaterialFlags, Slot

_ROOT = REPO

_XML = """
<mujoco>
  <asset>
    <texture name="c" type="2d" builtin="checker" width="8" height="8" rgb1="1 1 1" rgb2="0 0 0"/>
    <material name="plain" rgba="0.2 0.4 0.6 1" specular="0.3" shininess="0.25" reflectance="0.4"/>
    <material name="glow" rgba="0.9 0.5 0.1 0.5" metallic="0.7" roughness="0.1" emission="2" texture="c"/>
    <material name="layered" texrepeat="2 3" texuniform="true">
      <layer texture="c" role="rgb"/>
      <layer texture="c" role="orm"/>
      <layer texture="c" role="opacity"/>
      <layer texture="c" role="normal"/>
      <layer texture="c" role="emissive"/>
    </material>
    <material name="cut"><layer texture="c" role="rgba"/></material>
  </asset>
  <custom><numeric name="raywarp:alpha_cutoff:cut" data="0.25"/></custom>
  <worldbody><geom type="sphere" size="0.1" material="plain"/></worldbody>
</mujoco>
"""


def _materials(xml: str = _XML):
  """The model of `xml` and its canonical materials."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  return mjm, openpbr_materials(mjm, LoadReport())


def test_constants_follow_the_mapping_table():
  """rgba -> base color / opacity (BLEND below 1), specular s -> w = 4 s^2 at ior 1.5, shininess
  -> roughness, reflectance -> ext.mirror, emission -> luminance with the rgb as color."""
  mjm, (plain, glow, *_) = _materials()
  assert plain.name == "plain"
  np.testing.assert_allclose(plain.base_color, (0.2, 0.4, 0.6), rtol=1e-6)
  assert plain.geometry_opacity == 1.0 and plain.ext.alpha_mode == AlphaMode.OPAQUE
  assert plain.specular_weight == 4.0 * float(mjm.mat_specular[0]) ** 2 and plain.specular_ior == 1.5
  assert plain.ext.flags & MaterialFlags.DIELECTRIC_SPECULAR_LEVEL
  assert glow.ext.flags & MaterialFlags.DIELECTRIC_SPECULAR_LEVEL
  np.testing.assert_allclose(plain.specular_roughness, 0.75, rtol=1e-6)  # 1 - shininess
  assert plain.base_metalness == 0.0
  assert plain.ext.mirror == float(mjm.mat_reflectance[0])
  assert plain.emission_luminance == 0.0
  np.testing.assert_allclose((glow.base_metalness, glow.specular_roughness), (0.7, 0.1), rtol=1e-6)
  assert glow.geometry_opacity == 0.5 and glow.ext.alpha_mode == AlphaMode.BLEND
  assert glow.emission_luminance == 2.0
  assert glow.emission_color == glow.base_color


def test_texture_roles_bind_the_slots_the_device_reads():
  """`texture=` is an rgb layer (BASE_COLOR; untextured emission follows the textured base:
  EMISSION_COLOR); orm reads red / green / blue into occlusion / roughness / metalness; opacity its
  red channel; texrepeat / texuniform reach the UV transform and UV_WORLD; a cut-out ignores the
  rgba alpha (the device tests the opacity layer only)."""
  _, (_, glow, layered, cut) = _materials()
  assert glow.texture(Slot.BASE_COLOR) is not None
  assert glow.texture(Slot.EMISSION_COLOR) == glow.texture(Slot.BASE_COLOR)
  channels = {slot: ref.channel for slot, ref in layered.textures}
  assert channels == {
    Slot.BASE_COLOR: CHANNEL_RGB,
    Slot.BASE_METALNESS: 2,
    Slot.SPECULAR_ROUGHNESS: 1,
    Slot.OCCLUSION: 0,
    Slot.GEOMETRY_NORMAL: CHANNEL_RGB,
    Slot.EMISSION_COLOR: CHANNEL_RGB,
    Slot.GEOMETRY_OPACITY: 0,
  }
  assert layered.emission_color == (1.0, 1.0, 1.0)  # the emissive layer is the color
  normal = layered.texture(Slot.GEOMETRY_NORMAL)
  assert normal is not None and (normal.scale, normal.bias) == NORMAL_UNPACK  # the device's 2 t - 1
  assert all(ref.uv_xform == ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0)) for _, ref in layered.textures)
  assert layered.ext.flags == int(MaterialFlags.UV_WORLD | MaterialFlags.DIELECTRIC_SPECULAR_LEVEL)
  assert layered.ext.alpha_mode == AlphaMode.BLEND  # an opacity map needs a blend mode
  assert cut.texture(Slot.GEOMETRY_OPACITY) is None and cut.geometry_opacity == 1.0
  assert cut.ext.alpha_mode == AlphaMode.MASK and cut.ext.alpha_cutoff == 0.25


def test_unset_values_only_fill_unset_fields():
  """`unset_metallic` / `unset_roughness` replace MuJoCo's -1 only."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  plain, glow, *_ = openpbr_materials(mjm, LoadReport(), unset_metallic=1.0, unset_roughness=0.2)
  np.testing.assert_allclose((plain.base_metalness, plain.specular_roughness), (1.0, 0.2), rtol=1e-6)
  np.testing.assert_allclose((glow.base_metalness, glow.specular_roughness), (0.7, 0.1), rtol=1e-6)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan"])
def test_alpha_cutoff_outside_the_unit_range_is_rejected(value):
  """A cutoff numeric outside [0, 1] fails at the adapter, naming it (0: no cutoff)."""
  xml = _XML.replace('data="0.25"', f'data="{value}"')
  with pytest.raises(ValueError, match="raywarp:alpha_cutoff:cut"):
    _materials(xml)
  _, (*_, cut) = _materials(_XML.replace('data="0.25"', 'data="0"'))
  assert cut.ext.alpha_mode == AlphaMode.BLEND
  with pytest.raises(ValueError, match="raywarp:alpha_cutoff:nothing"):
    _materials(_XML.replace("alpha_cutoff:cut", "alpha_cutoff:nothing"))


def _derived(mats):
  """`derive` of canonical materials `mats` (no transmission)."""
  table, extras = Materials.from_materials(mats), MaterialExtras.from_materials(mats)
  return derive(table, extras, LoadReport(), volume=np.zeros(len(mats), bool))


@hypothesis.given(st.floats(0.0, 1.0, width=32, allow_subnormal=False))  # MJCF rejects subnormals
def test_specular_weight_derives_the_reflectance_exactly(s):
  """w = 4 s^2 of a float32 level s at ior 1.5: `derive` gives the device reflectance s back, bit
  for bit (the alias identity of `m.mat_specular`)."""
  mjm, mats = _materials(_XML.replace('specular="0.3"', f'specular="{s!r}"'))
  assert _derived(mats).reflectance[0] == mjm.mat_specular[0]


_ASSETS = Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "raywarp"))
_MODELS = (
  _ROOT / "tests/data/scenes/geoms.xml",
  _ROOT / "tests/data/scenes/randomize.xml",
  _ASSETS / "menagerie/anybotics_anymal_c/anymal_c.xml",
)


@pytest.mark.parametrize("path", _MODELS, ids=lambda p: p.stem)
def test_alias_identity_of_the_device_tables(path):
  """The aliased device tables hold the derived quantities of the canonical materials
  (`core/pipeline/derive/materials.py`): reflectance = m.mat_specular and base = m.mat_rgba exactly;
  the emitted radiance (peak x tint) is MuJoCo's m.mat_emission x the rgb, which the core computes
  by scaling the base color with the aliased m.mat_emission. Cut-outs (MASK) keep only the rgb."""
  if not path.exists():
    pytest.skip(f"{path} missing: run internal/tools/fetch_assets.py")
  _check_alias_identity(mujoco.MjModel.from_xml_path(str(path)))


def test_alias_identity_of_emissive_and_specular_materials():
  """The same identities on authored emission and specular (the shipped models author neither)."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  assert (mjm.mat_emission > 0).any() and (mjm.mat_specular != 0.5).any()
  _check_alias_identity(mjm)


def _check_alias_identity(mjm: mujoco.MjModel) -> None:
  """`test_alias_identity_of_the_device_tables`'s checks on `mjm`."""
  mats = openpbr_materials(mjm, LoadReport())
  derived = _derived(mats)
  np.testing.assert_array_equal(derived.reflectance, mjm.mat_specular)
  blended = np.array([m.ext.alpha_mode != AlphaMode.MASK for m in mats], bool)  # a cut-out ignores rgba alpha
  np.testing.assert_array_equal(derived.base[blended], mjm.mat_rgba[blended])
  radiance = derived.emission[:, None] * derived.emission_tint
  np.testing.assert_allclose(radiance, mjm.mat_emission[:, None] * mjm.mat_rgba[:, :3], rtol=1e-6)


_CUTOUTS = """
<mujoco>
  <asset>
    <texture name="c" type="2d" builtin="checker" width="8" height="8" rgb1="1 1 1" rgb2="0 0 0"/>
    <material name="faint" rgba="1 1 1 0.2"><layer texture="c" role="rgb"/><layer texture="c" role="opacity"/></material>
    <material name="rgba_only"><layer texture="c" role="rgba"/></material>
    <material name="bare" rgba="1 1 1 0.1"/>
  </asset>
  <custom>
    <numeric name="raywarp:alpha_cutoff:{name}" data="0.5"/>
  </custom>
  <worldbody>
    <geom type="sphere" size="0.1" material="faint"/>
    <geom type="sphere" size="0.1" material="rgba_only" pos="1 0 0"/>
    <geom type="sphere" size="0.1" material="bare" pos="2 0 0"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(
  ("name", "reports", "cutout"),
  [
    ("faint", {"mujoco.alpha_cutoff(rgba alpha ignored)"}, True),
    ("rgba_only", {"mujoco.alpha_cutoff(rgba alpha ignored)", "mujoco.alpha_cutoff(no opacity layer: opaque)"}, False),
    ("bare", {"mujoco.alpha_cutoff(rgba alpha ignored)", "mujoco.alpha_cutoff(no opacity layer: opaque)"}, False),
  ],
)
def test_cutouts_follow_the_device(name, reports, cutout):
  """A cut-out as the device draws it (it tests the opacity layer's texel against the cutoff and
  draws what passes opaque): (A) rgba alpha 0.2 with an opacity layer cuts on the layer alone;
  (B) an rgba layer without an opacity layer and (C) a < cutoff without any layer are opaque, not
  cut. The device cutoff table keeps the authored value (pixel-neutral); the reports say what is
  ignored."""
  import mujoco_warp as mjw
  import warp as wp

  import raywarp as rw
  import raywarp.integrations.mujoco

  mjm = mujoco.MjModel.from_xml_string(_CUTOUTS.format(name=name))
  report = LoadReport()
  mats = openpbr_materials(mjm, report)
  mat = next(m for m in mats if m.name == name)
  assert mat.ext.alpha_mode == AlphaMode.MASK and mat.geometry_opacity == 1.0
  opacity = mat.texture(Slot.GEOMETRY_OPACITY)
  assert (opacity is not None) == cutout and (opacity is None or opacity.channel == 0)
  assert set(report.approximated) == reports
  i = mats.index(mat)
  alpha = _derived(mats).alpha[i]
  assert alpha.opacity == 1.0 and (alpha.cutoff > 0.0) == cutout  # C: opaque, not invisible
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  assert scene.capabilities.alpha_cutout == cutout
  assert scene.tables["material.alpha_cutoff"].numpy()[0, i] == np.asarray(0.5, np.float32)
  assert set(reports) <= set(scene.report.approximated)


def test_rgba_alpha_times_an_opacity_layer_is_reported():
  """An rgba layer and an opacity layer: the device multiplies both alphas, OpenPBR has one."""
  xml = _XML.replace('<layer texture="c" role="rgba"/>', '<layer texture="c" role="rgba"/><layer texture="c" role="opacity"/>')
  report = LoadReport()
  openpbr_materials(mujoco.MjModel.from_xml_string(xml.replace('data="0.25"', 'data="0"')), report)
  assert report.approximated["mujoco.texture.rgba_alpha(times the opacity layer)"] == 1


def test_a_model_without_materials_builds():
  """nmat = 0: empty material tables, no mirror, no cut-out."""
  import mujoco_warp as mjw
  import warp as wp

  import raywarp as rw
  import raywarp.integrations.mujoco

  mjm = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  assert scene.tables["material.base_metalness"].shape == (1, 0)
  assert not scene.capabilities.has_mirror and not scene.capabilities.alpha_cutout
  assert "material.alpha_cutoff" not in scene.tables
