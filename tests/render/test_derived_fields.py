"""Derived per-world fields (core/pipeline/worlds/fields.py): `specular_weight`,
`specular_ior`, the coat inputs and `emission_luminance` declared per world derive the device
reflectance, roughness, coat bytes and emission every frame: equal to a fresh upload at float32
values, within 1 ulp of it for float64-authored weights (and the coat roughening's 4th root)."""

import dataclasses
from typing import Final

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.derive import fields

_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
_DEPTH = rw.RenderSettings().replace(outputs=rw.Outputs.of("depth"))
_DERIVED = rw.PerWorld(material=["specular_weight", "emission_luminance"])
# Per material: a plain dielectric, a high IOR, a coupled (OpenPBR) coat over IOR 1.4, a glTF coat,
# a partly transmissive thin-walled pane (its two interfaces: `derive.thin_wall_scale`).
_MATERIALS: Final = (
  rw.Material(),
  rw.Material(specular_ior=2.4),
  rw.Material(specular_ior=1.4, coat_weight=0.5, coat_ior=1.6),
  rw.Material(coat_weight=1.0, coat_ior=1.5, ext=rw.scene.RaywarpExt(flags=int(rw.scene.MaterialFlags.COAT_GLTF))),
  rw.Material(specular_ior=1.7, transmission_weight=0.6, geometry_thin_walled=True),
)
_COLORS = ((1.0, 0.5, 0.25), (0.3, 0.9, 0.7), (1.0, 1.0, 1.0), (0.123, 0.456, 0.789), (0.5, 0.5, 0.5))
# (nworld, nmat) user values: float32 values, w > 4 caps the reflectance at 2.5.
_WEIGHTS = np.array([[0.0, 0.25, 1.0, 1.7, 0.8], [3.3, 12.0, 0.01, 0.5, 2.0]], np.float32)
_LUMINANCES = np.array([[0.0, 1.0, 3.5, 1e-3, 2.0], [250.0, 0.7, 0.0, 9.0, 0.0]], np.float32)


def _host(weights: np.ndarray, luminances: np.ndarray) -> rw.Scene:
  """The test materials at `weights` / `luminances`, each on a sphere."""
  b = rw.SceneBuilder()
  for i, (base, color) in enumerate(zip(_MATERIALS, _COLORS, strict=True)):
    mat = dataclasses.replace(
      base, specular_weight=float(weights[i]), emission_luminance=float(luminances[i]), emission_color=color
    )
    b.add_instance("sphere", size=(0.2, 0.0, 0.0), material=b.add_material(mat), xform=np.c_[np.eye(3), [i, 0.0, 0.0]])
  return b.build()


def _renderer(scene: rw.GpuScene, nworld: int, per_world: rw.PerWorld | None) -> rw.Renderer:
  return rw.Renderer(scene, nworld=nworld, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=_DEPTH, per_world=per_world)


@pytest.mark.parametrize("device", _DEVICES)
def test_derived_slots_equal_a_fresh_upload(device):
  """Per world, the derived reflectance / emission equal those of a scene uploaded with the
  world's float32 values (bit for bit: the reflectance is derived in float64, the emission is one
  product of float32 values), coupled coats and the reflectance cap included."""
  scene = rw.upload(_host(_WEIGHTS[0], _LUMINANCES[0]), device=device)
  r = _renderer(scene, 2, _DERIVED)
  r.per_world["material.specular_weight"].assign(_WEIGHTS)
  r.per_world["material.emission_luminance"].assign(_LUMINANCES)
  r.render()
  frame = r.debug.frame()
  for w in range(2):
    fresh = rw.upload(_host(_WEIGHTS[w], _LUMINANCES[w]), device=device).tables
    np.testing.assert_array_equal(frame.material_reflectance.numpy()[w], fresh["material.reflectance"].numpy()[0])
    np.testing.assert_array_equal(frame.material_emission.numpy()[w], fresh["material.emission"].numpy()[0])
  assert frame.material_reflectance.numpy()[1, 1] == np.asarray(2.5, np.float32)  # sqrt(12 / 4) x 5 x 1.4 / 3.4 capped


def test_declared_tables_start_at_the_scene_values():
  """The user tables start at the scene's (float32) OpenPBR values and the first frame keeps the
  scene's device values; the derive kernels are first-frame warm-up entries; undeclared renderers
  have no derive launch."""
  scene = rw.upload(_host(_WEIGHTS[0], _LUMINANCES[0]), device="cpu")
  r = _renderer(scene, 3, _DERIVED)
  np.testing.assert_array_equal(r.per_world["material.specular_weight"].numpy(), np.tile(_WEIGHTS[:1], (3, 1)))
  np.testing.assert_array_equal(r.per_world["material.emission_luminance"].numpy(), np.tile(_LUMINANCES[:1], (3, 1)))
  r.render()
  frame = r.debug.frame()
  np.testing.assert_array_equal(
    frame.material_reflectance.numpy(), np.tile(scene.tables["material.reflectance"].numpy(), (3, 1))
  )
  np.testing.assert_array_equal(frame.material_emission.numpy(), np.tile(scene.tables["material.emission"].numpy(), (3, 1)))
  plain = _renderer(scene, 1, None)
  assert plain.debug.frame().material_reflectance is scene.tables["material.reflectance"]
  assert not plain._per_world.derived._launches and not plain._per_world.derived.slots
  derive = fields.kernels(("material.specular_weight", "material.emission_luminance"))
  assert all(k in warmup.first_frame(r.context(), r._backend).default for k in derive)
  assert not any(k in warmup.first_frame(plain.context(), plain._backend).default for k in derive)


def test_float64_weights_derive_within_one_ulp():
  """A float64-authored weight (0.9, 0.55: not float32 values) is rounded into the float32 user
  table: the first frame's reflectance is within 1 ulp of the upload's."""
  b = rw.SceneBuilder()
  for i, w in enumerate((0.9, 0.55)):
    mat = b.add_material(rw.Material(specular_weight=w, specular_ior=1.7))
    b.add_instance("sphere", size=(0.2, 0.0, 0.0), material=mat, xform=np.c_[np.eye(3), [i, 0.0, 0.0]])
  scene = rw.upload(b.build(), device="cpu")
  r = _renderer(scene, 1, rw.PerWorld(material=["specular_weight"]))
  r.render()
  np.testing.assert_array_max_ulp(r.debug.frame().material_reflectance.numpy(), scene.tables["material.reflectance"].numpy(), 1)


def test_luminance_without_an_upload_emission_shows_the_emission_color():
  """A material emitting nothing at upload had no emission map: a per-world luminance lit its base
  color (green), not its `emission_color` (red). The declaration now gives it the tint map, so
  both spheres show their emission color, as a fresh upload does."""
  b = rw.SceneBuilder()
  for x, lum in ((-0.6, 1.0), (0.6, 0.0)):
    mat = rw.Material(base_color=(0.0, 1.0, 0.0), emission_color=(1.0, 0.0, 0.0), emission_luminance=lum)
    b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=b.add_material(mat), xform=np.c_[np.eye(3), [x, 0.0, 0.0]])
  scene = rw.upload(b.build(), device="cpu")
  settings = rw.RenderSettings().replace(outputs=rw.Outputs.of("emission"))
  r = rw.Renderer(
    scene,
    nworld=1,
    cameras=[rw.CameraSpec(fovy=40.0)],
    resolution=(32, 16),
    settings=settings,
    per_world=rw.PerWorld(material=["emission_luminance"]),
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.per_world["material.emission_luminance"].assign(np.array([[2.0, 2.0]], np.float32))
  e = required(r.render().emission).numpy()[0, 0].reshape(-1, 3)
  red, green = e[:, 0] > 2.0 * e[:, 1], e[:, 1] > 2.0 * e[:, 0]
  assert red.any() and not green.any()  # both spheres red (the second one was green)


def test_derived_slots_are_not_declarable():
  """The device slots a derived field writes are neither declared nor read per world directly."""
  scene = rw.upload(_host(_WEIGHTS[0], _LUMINANCES[0]), device="cpu")
  for name in ("reflectance", "emission"):
    with pytest.raises(ValueError, match="derived per frame from material"):
      _renderer(scene, 1, rw.PerWorld(material=[name]))
  r = _renderer(scene, 1, _DERIVED)
  with pytest.raises(KeyError, match=r"derived per frame from material\.specular_weight"):
    r.per_world["material.reflectance"]
  with pytest.raises(KeyError, match="not declared"):
    _renderer(scene, 1, None).per_world["material.specular_weight"]


_MJCF = """
<mujoco>
  <asset>
    <material name="a" rgba="0.8 0.3 0.2 1" specular="0.3" emission="0.5"/>
    <material name="b" rgba="0.2 0.7 0.9 1" specular="0.7"/>
  </asset>
  <worldbody>
    <geom type="sphere" size="0.2" material="a"/>
    <geom type="sphere" size="0.2" pos="1 0 0" material="b"/>
  </worldbody>
</mujoco>
"""


def test_mujoco_tables_start_at_the_live_model():
  """MuJoCo: the user tables start from the live `m.mat_specular` / `m.mat_emission` (edited after
  the scene was built): w = 4 s^2, L = e; the first frame keeps them (within 1 ulp)."""
  mjm = mujoco.MjModel.from_xml_string(_MJCF)
  with wp.ScopedDevice("cpu"):
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mujoco.MjData(mjm), nworld=2)
    scene = rw.integrations.mujoco.scene(mjm, m)
    levels, emission = np.array([0.45, 0.2], np.float32), np.array([0.0, 1.25], np.float32)
    m.mat_specular.assign(levels)
    m.mat_emission.assign(emission)
    r = _renderer(scene, 2, _DERIVED)
    np.testing.assert_allclose(r.per_world["material.specular_weight"].numpy(), np.tile(4.0 * levels**2, (2, 1)), rtol=2e-7)
    np.testing.assert_array_equal(r.per_world["material.emission_luminance"].numpy(), np.tile(emission, (2, 1)))
    r.render(d)
    frame = r.debug.frame(d)
  np.testing.assert_array_max_ulp(frame.material_reflectance.numpy(), np.tile(levels, (2, 1)), 1)
  np.testing.assert_array_equal(frame.material_emission.numpy(), np.tile(emission, (2, 1)))


def test_mujoco_specular_weight_is_the_level_squared():
  """MuJoCo: w = 4 s^2 gives the device reflectance s back, the luminance is `mat_emission`
  (IOR scale and peak color 1), starting at the model's values."""
  mjm = mujoco.MjModel.from_xml_string(_MJCF)
  with wp.ScopedDevice("cpu"):
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mujoco.MjData(mjm))
    scene = rw.integrations.mujoco.scene(mjm, m)
    r = _renderer(scene, 1, _DERIVED)
    np.testing.assert_array_equal(r.per_world["material.emission_luminance"].numpy()[0], mjm.mat_emission)
    levels = np.array([0.1, 0.9], np.float32)
    r.per_world["material.specular_weight"].assign((4.0 * levels.astype(np.float64) ** 2).astype(np.float32)[None])
    r.per_world["material.emission_luminance"].assign(np.array([[2.0, 0.25]], np.float32))
    r.render(d)
    frame = r.debug.frame(d)
  np.testing.assert_allclose(frame.material_reflectance.numpy()[0], levels, rtol=1e-7)
  np.testing.assert_array_equal(frame.material_emission.numpy()[0], [2.0, 0.25])


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
def test_captured_graph_derives_the_written_values():
  """A captured graph runs the derive kernels: values written after the capture reach the slot."""
  scene = rw.upload(_host(_WEIGHTS[0], _LUMINANCES[0]), device="cuda:0")
  r = _renderer(scene, 2, _DERIVED)
  g = r.capture()
  r.per_world["material.specular_weight"].assign(_WEIGHTS)
  g.launch()
  wp.synchronize()
  fresh = rw.upload(_host(_WEIGHTS[1], _LUMINANCES[0]), device="cuda:0").tables["material.reflectance"].numpy()[0]
  np.testing.assert_array_equal(r.debug.frame().material_reflectance.numpy()[1], fresh)


# Per material: (base color, emission color, upload luminance); material 1 emits nothing at upload.
_EMITTERS = (((0.2, 0.2, 0.2), (1.0, 0.5, 0.25), 3.0), ((0.0, 1.0, 0.0), (0.9, 0.3, 0.6), 0.0))
_EMISSION_SETTINGS = rw.RenderSettings().replace(outputs=rw.Outputs.of("emission"))


def _emitters(luminances: tuple[float, float]) -> rw.Scene:
  """Two emissive spheres side by side, at `luminances`."""
  b = rw.SceneBuilder()
  for x, (base, color, _), lum in zip((-0.6, 0.6), _EMITTERS, luminances, strict=True):
    mat = rw.Material(base_color=base, emission_color=color, emission_luminance=lum)
    b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=b.add_material(mat), xform=np.c_[np.eye(3), [x, 0.0, 0.0]])
  return b.build()


def _emission_image(scene: rw.GpuScene, per_world: rw.PerWorld | None, luminances: np.ndarray | None) -> np.ndarray:
  """The emission output of `scene` seen from +z (`luminances`: the per-world ones written)."""
  r = rw.Renderer(
    scene, nworld=1, cameras=[rw.CameraSpec(fovy=40.0)], resolution=(32, 16), settings=_EMISSION_SETTINGS, per_world=per_world
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  if luminances is not None:
    r.per_world["material.emission_luminance"].assign(luminances)
  return required(r.render().emission).numpy()[0, 0]


def test_luminance_without_an_upload_emission_equals_a_fresh_upload():
  """A per-world luminance on a material that emitted nothing at upload lit base color x L x
  max(emission_color) (L4's caveat): it now renders as a scene uploaded with that luminance, and
  the scene's tables (codegen, memory) are unchanged without the declaration."""
  luminances = np.array([[3.0, 2.5]], np.float32)
  scene = rw.upload(_emitters((3.0, 0.0)), device="cpu")
  fresh = rw.upload(_emitters((3.0, 2.5)), device="cpu")
  declared = rw.PerWorld(material=["emission_luminance"])
  np.testing.assert_allclose(
    _emission_image(scene, declared, luminances), _emission_image(fresh, None, None), rtol=1e-6, atol=1e-6
  )
  plain = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=_EMISSION_SETTINGS)
  assert plain.debug.frame().material_texture is scene.tables["material.texture"]
  assert plain._per_world.derived.emission_maps is None


_COAT_FIELDS = rw.PerWorld(material=["specular_ior", "coat_weight", "coat_ior", "coat_roughness"])
# (nworld, nmat) per-world coat inputs of the `_MATERIALS` (coupled, glTF, none, a coat at IOR 1.5,
# a thin-walled pane).
_IORS = np.array([[1.5, 2.4, 1.4, 1.5, 1.7], [1.3, 1.7, 2.0, 1.45, 1.2]], np.float32)
_COAT_WEIGHTS = np.array([[0.0, 0.0, 0.5, 1.0, 0.0], [0.25, 1.0, 0.0, 0.75, 0.5]], np.float32)
_COAT_IORS = np.array([[1.5, 1.5, 1.6, 1.5, 1.5], [1.8, 1.25, 1.5, 1.5, 1.6]], np.float32)
_COAT_ROUGHNESS = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 0.9, 0.5, 0.125, 0.2]], np.float32)


def _coated(w: int) -> rw.Scene:
  """The `_MATERIALS` at world w's IOR and coat inputs (world -1: the scene's)."""
  b = rw.SceneBuilder()
  for i, base in enumerate(_MATERIALS):
    mat = dataclasses.replace(base, specular_roughness=0.4, specular_weight=float(_WEIGHTS[0, i]))
    if w >= 0:
      mat = dataclasses.replace(
        mat,
        specular_ior=float(_IORS[w, i]),
        coat_weight=float(_COAT_WEIGHTS[w, i]),
        coat_ior=float(_COAT_IORS[w, i]),
        coat_roughness=float(_COAT_ROUGHNESS[w, i]),
      )
    b.add_instance("sphere", size=(0.2, 0.0, 0.0), material=b.add_material(mat), xform=np.c_[np.eye(3), [i, 0.0, 0.0]])
  return b.build()


@pytest.mark.parametrize("device", _DEVICES)
def test_per_world_ior_and_coat_equal_a_fresh_upload(device):
  """Per world, `specular_ior` and the coat inputs derive the reflectance (bit for bit), the coat-
  roughened roughness (within 1 ulp: the 4th root) and the instances' coat bytes of a scene
  uploaded with the world's values; they were not declarable (design §4.1)."""
  scene = rw.upload(_coated(-1), device=device)
  r = _renderer(scene, 2, _COAT_FIELDS)
  for name, values in (
    ("specular_ior", _IORS),
    ("coat_weight", _COAT_WEIGHTS),
    ("coat_ior", _COAT_IORS),
    ("coat_roughness", _COAT_ROUGHNESS),
  ):
    r.per_world[f"material.{name}"].assign(values)
  r.render()
  frame = r.debug.frame()
  for w in range(2):
    fresh = rw.upload(_coated(w), device=device).tables
    np.testing.assert_array_equal(frame.material_reflectance.numpy()[w], fresh["material.reflectance"].numpy()[0])
    np.testing.assert_array_max_ulp(
      frame.material_specular_roughness.numpy()[w], fresh["material.specular_roughness"].numpy()[0], 1
    )
    np.testing.assert_array_equal(frame.instance_shading.numpy()[w], fresh["instance.shading"].numpy()[0])


def test_coat_fields_and_the_slots_they_derive_are_exclusive():
  """A coat field derives `specular_roughness` / `shading`: declaring both raises, and the derived
  slot is not readable per world."""
  scene = rw.upload(_coated(-1), device="cpu")
  with pytest.raises(ValueError, match="derived per frame from the declared coat fields"):
    _renderer(scene, 1, rw.PerWorld(material=["coat_weight", "specular_roughness"]))
  r = _renderer(scene, 1, rw.PerWorld(material=["coat_ior"]))
  with pytest.raises(KeyError, match="not declared"):
    r.per_world["instance.shading"]


_ALL_FIELDS = rw.PerWorld(material=[f.removeprefix("material.") for f in fields.DERIVED_FIELDS])
# World 1's invalid entries of each field, one per material: negative, zero, NaN, inf (`RANGES`).
_INVALID = {
  "specular_weight": (-1.0, np.nan, np.inf, -0.5, np.nan),
  "specular_ior": (-1.0, 0.0, np.nan, np.inf, -2.0),
  "coat_weight": (2.0, -0.1, np.nan, np.inf, 1.5),
  "coat_ior": (0.0, -1.5, np.nan, -np.inf, 0.0),
  "coat_roughness": (1.5, -1.0, np.nan, np.inf, -0.5),
  "emission_luminance": (-1.0, np.nan, 3.0e38, np.inf, -2.0),  # 3e38 x peak overflowed to inf
}


def test_invalid_per_world_inputs_derive_from_the_scene_and_raise():
  """A per-world IOR of -1 made eta + 1 = 0 (inf / NaN reflectance), a NaN coat IOR or roughness
  passed through, a negative weight took sqrt(< 0): such entries now derive from the scene's
  value, and the next frame's validation raises naming the ranges."""
  scene = rw.upload(_coated(-1), device="cpu")
  r = _renderer(scene, 2, _ALL_FIELDS)
  for name, bad in _INVALID.items():
    table = r.per_world[f"material.{name}"]
    values = table.numpy().copy()
    values[1] = bad
    table.assign(values)
  r.render()
  frame = r.debug.frame()
  for field in ("material_reflectance", "material_specular_roughness", "material_emission", "instance_shading"):
    got = getattr(frame, field).numpy()
    assert np.isfinite(got).all(), field
    np.testing.assert_array_equal(got[1], got[0], err_msg=field)  # world 0 holds the scene's values
  with pytest.raises(ValueError, match="per-world values must be numbers in range"):
    r.render()


def test_mujoco_coat_weight_is_declarable():
  """MuJoCo scenes had no `instance.shading` table, so the coat fields raised "not available":
  a per-world coat weight now writes the coat bytes of each geom's material, compiles the
  clearcoat in; world 0 (no coat) keeps the live reflectance (`m.mat_specular`), world 1's coupled
  coat changes the base IOR the reflectance derives with."""
  from raywarp._base.types import COAT_WEIGHT_SHIFT

  mjm = mujoco.MjModel.from_xml_string(_MJCF)
  with wp.ScopedDevice("cpu"):
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mujoco.MjData(mjm), nworld=2)
    scene = rw.integrations.mujoco.scene(mjm, m)
    assert "instance.shading" not in scene.tables  # the declaration allocates it
    rgb = rw.RenderSettings().replace(outputs=rw.Outputs.of("rgb"))
    r = rw.Renderer(
      scene,
      nworld=2,
      cameras=[rw.CameraSpec()],
      resolution=(8, 8),
      settings=rgb,
      per_world=rw.PerWorld(material=["coat_weight"]),
    )
    r.per_world["material.coat_weight"].assign(np.array([[0.0, 0.0], [0.5, 1.0]], np.float32))
    r.render(d)
    frame = r.debug.frame(d)
  assert r.features.clearcoat
  weights = np.bitwise_and(np.right_shift(frame.instance_shading.numpy(), COAT_WEIGHT_SHIFT), 255)
  # The device weight is the coat weight (its IOR is the layered coat's own Fresnel law), as bytes.
  expected = np.rint(np.array([0.5, 1.0]) * 255.0)
  np.testing.assert_array_equal(weights, [[0, 0], expected])
  np.testing.assert_array_max_ulp(frame.material_reflectance.numpy()[0], mjm.mat_specular, 1)


def test_mujoco_specular_edits_reach_the_render_with_a_coat_field_declared():
  """With a coat field declared the reflectance is derived; its specular weight came from
  `m.mat_specular` once, at the declaration, so a later edit of the live table was overwritten.
  The weight now follows the live table every frame (world 0 has no coat: its reflectance is the
  edited `m.mat_specular`)."""
  mjm = mujoco.MjModel.from_xml_string(_MJCF)
  with wp.ScopedDevice("cpu"):
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mujoco.MjData(mjm), nworld=2)
    scene = rw.integrations.mujoco.scene(mjm, m)
    r = _renderer(scene, 2, rw.PerWorld(material=["coat_weight"]))
    r.render(d)
    np.testing.assert_array_max_ulp(r.debug.frame(d).material_reflectance.numpy()[0], mjm.mat_specular, 1)
    edited = np.array([0.05, 0.8], np.float32)
    m.mat_specular.assign(edited)
    r.render(d)
    frame = r.debug.frame(d)
  np.testing.assert_array_max_ulp(frame.material_reflectance.numpy()[0], edited, 1)


def test_thin_walled_pane_reflects_both_interfaces():
  """A thin-walled transmissive pane's F0 is the two interfaces' 2 F / (1 + F) (OpenPBR's thin
  wall) over its transmission share; an opaque or solid material keeps one interface's F."""
  b = rw.SceneBuilder()
  mats = (
    rw.Material(transmission_weight=1.0, geometry_thin_walled=True),
    rw.Material(transmission_weight=0.5, geometry_thin_walled=True),
    rw.Material(transmission_weight=1.0),
    rw.Material(geometry_thin_walled=True),
  )
  for i, mat in enumerate(mats):
    b.add_instance("sphere", size=(0.2, 0.0, 0.0), material=b.add_material(mat), xform=np.c_[np.eye(3), [i, 0.0, 0.0]])
  r = rw.upload(b.build(), device="cpu").tables["material.reflectance"].numpy()[0]
  f0 = 0.16 * r.astype(np.float64) ** 2
  one = 0.04  # ((1.5 - 1) / (1.5 + 1))^2
  np.testing.assert_allclose(f0, [2 * one / (1 + one), one * (1 + 0.5 * (1 - one) / (1 + one)), one, one], rtol=1e-6)
