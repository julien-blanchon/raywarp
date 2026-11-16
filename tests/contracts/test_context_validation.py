"""Argument validation of RenderSettings, Renderer, rw.integrations.mujoco.scene, env.id and
read_hdr."""

import dataclasses
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
from conftest import LIBRARY_GI, recipe, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp._base import types as core_types
from raywarp._base.types import CameraSlot
from raywarp.scene.images.hdr import read_hdr
from raywarp.settings import fields

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def mjm() -> mujoco.MjModel:
  """The test model."""
  return mujoco.MjModel.from_xml_string(XML)


BAD_SETTINGS = [
  ({"post.tone_mapper": "filmic"}, r"post\.tone_mapper must be one of"),
  ({"performance.render_order": "row"}, r"performance\.render_order must be one of"),
  ({"aa.spp": 0}, r"aa\.spp must be at least 1"),
  ({"shadows.samples": 0}, r"shadows\.samples must be at least 1"),
  ({"transparency.max_layers": 0}, r"transparency\.max_layers must be at least 1"),
  ({"shadows.max_casters": 0}, r"shadows\.max_casters must be in \[1, 16\]"),
  ({"lighting.max_exact": 0}, r"lighting\.max_exact must be in \[1, 16\]"),
  ({"lighting.light_grid_cell": 0.0}, r"lighting\.light_grid_cell must be positive"),
  ({"lighting.light_grid_cell": float("inf")}, r"lighting\.light_grid_cell must be positive and finite"),
  ({"gi.spacing": float("inf")}, r"gi\.spacing must be positive and finite"),
  ({"exposure.ev100": float("nan")}, r"exposure\.ev100 must be finite"),
  ({"exposure.mode": "metered"}, r"exposure\.mode must be one of"),
  ({"performance.traversal": "hardware"}, r"performance\.traversal must be"),
  ({"outputs.rgb": None}, r"outputs: request at least one output"),
  ({"transparency.enabled": "yes"}, r'transparency\.enabled must be True, False or "auto"'),
  ({"reflections.max_roughness": 1.5}, r"reflections\.max_roughness must be in \[0, 1\]"),
  ({"environment.background_color": (0, 0)}, r"environment\.background_color must be an sRGB triple"),
  ({"performance.rebuild_every": -1}, r"performance\.rebuild_every must be at least 0"),
  ({"atmosphere.start": -1.0}, r"atmosphere\.start must be at least 0"),
  ({"atmosphere.end": float("nan")}, r"atmosphere\.end must be at least 0"),
  # Types from the annotations: a truthy string, a float count, a bool count are not accepted.
  ({"shadows.enabled": "yes"}, r"shadows\.enabled must be True or False"),
  ({"aa.spp": 1.5}, r"aa\.spp must be an integer, got 1\.5"),
  ({"shadows.samples": True}, r"shadows\.samples must be an integer"),
  ({"reflections.max_roughness": "8"}, r"reflections\.max_roughness must be a number"),
  ({"environment.background_color": (0.0, float("nan"), 0.0)}, r"environment\.background_color must be an sRGB triple"),
  ({"atmosphere.color": (0.5, 0.5, "x")}, r"atmosphere\.color must be an sRGB triple"),
  ({"shadow_samples": 4}, "unknown setting 'shadow_samples'"),
  ({"shadows.count": 4}, "unknown setting 'shadows.count'"),
]


@pytest.mark.parametrize("kwargs,match", BAD_SETTINGS, ids=[m for _, m in BAD_SETTINGS])
def test_invalid_settings(kwargs: dict[str, object], match: str) -> None:
  with pytest.raises(ValueError, match=match):
    rw.RenderSettings().replace(**kwargs)


def test_every_group_validates_under_its_settings_name() -> None:
  """Each group's messages name it as `RenderSettings` does ("<group>.<field>"), and numbers from
  NumPy pass the type checks."""
  for f in dataclasses.fields(rw.RenderSettings):
    group = rw.RenderSettings().group(f.name)
    checked = next(g.name for g in dataclasses.fields(group) if g.name not in ("kinds", "bounds", "color", "background_color"))
    with pytest.raises(ValueError, match=rf"^{f.name}\.{checked} must be"):
      rw.RenderSettings().replace(**{f"{f.name}.{checked}": object()})
  ev100 = np.float32(10.0)  # pyrefly: ignore[bad-argument-count]  # np.float32(x): untyped
  s = rw.RenderSettings().replace(**{"aa.spp": np.int64(2), "exposure.ev100": ev100, "shadows.enabled": np.True_})
  assert s.aa.spp == 2 and s.shadows.enabled is True


def test_choice_ids_follow_the_literals() -> None:
  """The Features / Params id tables list exactly the choices of their settings' annotations,
  and the device enums agree with them."""
  assert {m.name.lower(): int(m) for m in core_types.ToneMapper} == fields.TONE_MAPPERS
  assert {m.name.lower(): int(m) for m in core_types.RenderOrder} == fields.RENDER_ORDERS
  assert rw.settings.outputs.KINDS == fields.choices(fields.OutputKind)


def test_settings_replace_groups_and_dotted_keys() -> None:
  """A whole group and dotted fields in one call: the dotted ones apply over the group given."""
  s = recipe("fast").replace(
    shadows=rw.settings.Shadows(samples=4, max_casters=2),
    **{"shadows.samples": 8, "outputs": rw.Outputs.of("rgb", "depth"), "performance.rebuild_every": 32},
  )
  assert s.shadows == rw.settings.Shadows(samples=8, max_casters=2)
  assert s.lighting.light_grid and s.outputs.kinds == frozenset({"rgb", "depth"}) and s.performance.rebuild_every == 32
  assert recipe("fast").shadows.samples == 1  # frozen: the recipe is unchanged
  with pytest.raises(TypeError, match="shadows must be a Shadows"):
    rw.RenderSettings(shadows=4)  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(TypeError, match="shadows must be a settings group"):
    rw.RenderSettings().replace(shadows=4, **{"shadows.samples": 2})


def test_settings_flat_and_diff_round_trip() -> None:
  a = recipe("default")
  b = recipe("rtx_look")
  d = a.diff(b)
  assert d and all("." in k for k in d) and d["post.tone_mapper"] == "aces_rtx"
  assert a.replace(**d) == b
  assert a.diff(a) == {}
  assert rw.RenderSettings().replace(**b.flat()) == b


def test_fields_table_metadata() -> None:
  table = {f.name: f for f in rw.RenderSettings.fields_table()}
  assert len(table) == len(rw.RenderSettings().flat())
  assert table["aa.spp"].runtime is False and table["aa.spp"].rebuild is None and table["aa.spp"].default == 1
  assert table["performance.rebuild_every"].runtime and table["performance.rebuild_every"].rebuild is None
  assert table["gi.spacing"].runtime and table["gi.spacing"].rebuild == "gi"
  assert not table["gi.enabled"].runtime and table["gi.enabled"].rebuild == "gi"
  assert table["lighting.light_grid_cell"].runtime and table["lighting.light_grid_cell"].rebuild == "light_grid"
  assert not table["aa.spp"].runtime and table["exposure.ev100"].runtime


# B65: the field defaults before they became `balanced` (now the default), and each named
# configuration's values over them then (recorded at a39f3e1; the presets were removed in
# W9-PRESET-IMPL: `tools/recipes.py`), adjusted below for later approved decisions.
OLD_DEFAULTS = {
  "shadows.max_casters": None,
  "shadows.translucent": "off",
  "lighting.area_lights": "point",
  "lighting.light_grid": False,
  "reflections.enabled": False,
  "surfaces.anisotropy": 1,
  "lighting.max_exact": None,
  "reflections.max_exact": None,
  "gi.enabled": False,
  "gi.spacing": None,  # W4-A: derived from the scene (scale-free)
  "gi.rays": 256,
  "gi.bake_updates": 8,
  "post.tone_mapper": "pbr_neutral",
  # The old field defaults of the fields training_fast changes (W3-1: training_fast became the
  # field defaults, user decision Q-15).
  "transparency.max_layers": 4,
}
# Post-release decisions (user): balanced had RTX's ACES approximation for a while; since
# 2026-10-28 (robotics defaults) every preset but omniverse tone maps with PBR Neutral.
# Phase A decision 1 (user): coarse probe GI in balanced; quality / omniverse / reference keep the
# 0.5 m / 128 rays / 16 updates grid, training_fast / fastest stayed without GI.
# Robotics defaults item 7 (user, 2026-10-28): balanced keeps that grid but GI off (opt-in).
# W2-3 / D3 (autonomous, 2026-11-03): probe GI on in balanced; one grid for every preset with GI
# and for training_fast's sky occlusion: 0.5 m / 256 rays / 8 updates (the OLD_DEFAULTS above).
_COARSE_GI = {"gi.enabled": True}
_FINE_GI = {"gi.enabled": True}
_BALANCED_OLD = {
  "shadows.max_casters": 4,
  "shadows.translucent": "transmittance",
  "lighting.area_lights": "ltc",
  "lighting.light_grid": "auto",  # robotics defaults: on from 8 lights
  "reflections.enabled": True,
  "surfaces.anisotropy": 8,  # B66 follow-up (user): anisotropic filtering in balanced and above
  "lighting.max_exact": 4,  # M5 lane 1: the light caps in balanced and quality
  "reflections.max_exact": 2,
  **_COARSE_GI,
}
_QUALITY_OLD = {
  **_BALANCED_OLD,
  "aa.spp": 2,
  "shadows.samples": 8,
  "shadows.max_casters": 16,
  "lighting.max_exact": 8,
  "reflections.max_exact": 4,
  "reflections.shadows": "sun",
  "transparency.max_layers": 8,
  **_FINE_GI,
  "post.tone_mapper": "pbr_neutral",
}
# Post-release decision (user, 2026-10-28): quality's shadow samples 8 -> 2 (same FLIP vs
# reference, up to 44 % less frame time); reference keeps the values it inherited from the old
# quality.
_QUALITY = {**_QUALITY_OLD, "shadows.samples": 2}
OLD_PRESETS = {
  "fast": {
    "shadows.max_casters": 2,
    "lighting.light_grid": "auto",  # robotics defaults: on from 8 lights
    "reflections.enabled": False,
    "transparency.max_layers": 2,
    "lighting.visible_lights": False,  # W10-SPEED, user-approved: FAST draws no light surfaces
    **_COARSE_GI,  # W3-1: probe GI in training_fast (no measurable frame cost on static scenes)
  },
  "default": _BALANCED_OLD,
  "quality": _QUALITY,
  "rtx_look": {
    "shadows.max_casters": 8,
    "shadows.translucent": "clear",
    "lighting.area_lights": "ltc",
    "lighting.light_grid": "auto",  # robotics defaults: on from 8 lights
    "lighting.light_grid_cell": 0.5,
    "reflections.enabled": True,
    "reflections.shadows": "sun",
    "gi.enabled": True,
    "gi.normal_bias": 0.3,
    "post.tone_mapper": "aces_rtx",
    "surfaces.anisotropy": 8,
  },
  "reference": {
    **_QUALITY_OLD,
    "lighting.max_exact": None,
    "reflections.max_exact": None,
    "reflections.enabled": True,
  },
}


def test_default_and_recipes_keep_their_values() -> None:
  """One default (W9-PRESET-IMPL): `RenderSettings()` is the old balanced; `FAST` and the recipes
  keep the old training_fast / quality / omniverse / reference values; approved changes are
  pinned."""
  # The test session's GI default is off (conftest.py); `recipe` keeps the library's.
  assert rw.RenderSettings(gi=LIBRARY_GI) == recipe("default")
  assert recipe("quality").aa.spp == 2
  assert recipe("reference").aa.spp == 2
  old = rw.RenderSettings().replace(**OLD_DEFAULTS)
  for name, values in OLD_PRESETS.items():
    assert recipe(name) == old.replace(**values), name


BAD_LIGHTING = [
  ({"envmap": np.zeros((8, 16), np.float32)}, r"envmap\[0\] must be a path or a \(height, width, 3\)"),
  ({"envmap": [np.zeros((8, 16, 3), np.float32), np.zeros((8, 16, 5), np.float32)]}, r"envmap\[1\]"),
  ({"ambient": (0.1, float("nan"), 0.1)}, "ambient must be finite"),
  ({"headlight": rw.Headlight(ambient=(0.1, 0.1, 0.1), diffuse=(0.5, -0.1, 0.5))}, "headlight.diffuse must be non-neg"),
  ({"envmap_intensity": -1.0}, r"envmap_intensity .* must be in \[0.0, inf\]"),
  ({"ev100": float("inf")}, "ev100 must be finite"),
]


@pytest.mark.parametrize("kwargs,match", BAD_LIGHTING, ids=[m for _, m in BAD_LIGHTING])
def test_invalid_scene_lighting(kwargs, match):
  """`rw.Lighting` rejects bad values at construction (every adapter takes it)."""
  with pytest.raises(ValueError, match=match):
    rw.Lighting(**kwargs)


def test_invalid_scene_arguments(mjm: mujoco.MjModel) -> None:
  with pytest.raises(ValueError, match=r"unset_roughness must be in \[0, 1\]"):
    rw.integrations.mujoco.Options(unset_roughness=2.0)
  with pytest.raises(ValueError, match="no geoms or flexes in the enabled geom groups"):
    rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), adapter=rw.integrations.mujoco.Options(geom_groups=(5,)))


def test_invalid_renderer_arguments(mjm: mujoco.MjModel) -> None:
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  with pytest.raises(ValueError, match="nworld must be at least 1"):
    rw.Renderer(scene, nworld=0, resolution=(8, 8))
  with pytest.raises(ValueError, match="resolution must be"):
    rw.Renderer(scene, nworld=1, resolution=(0, 8))
  with pytest.raises(ValueError, match="camera 'nope' not found"):
    rw.Renderer(scene, nworld=1, cameras=["nope"], resolution=(8, 8))
  with pytest.raises(ValueError, match="camera index 3 out of range"):
    rw.Renderer(scene, nworld=1, cameras=[3], resolution=(8, 8))
  with pytest.raises(ValueError, match="lens distortion needs a perspective camera"):
    cams = [rw.CameraSpec(projection="orthographic", distortion=(0.1, 0.0, 0.0, 0.0))]
    rw.Renderer(scene, nworld=1, cameras=cams, resolution=(8, 8))
  with pytest.raises(ValueError, match=r"CameraSpec\.resolution must be"):
    rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(resolution=(0, 4))], resolution=(8, 8))
  r = rw.Renderer(scene, nworld=2, resolution=(8, 8))
  with pytest.raises(KeyError, match="unknown per-world field"):
    r.per_world["instance.sparkle"]
  with pytest.raises(KeyError, match="is a pose"):
    r.per_world["camera.pos"]
  with pytest.raises(KeyError, match="shared by every world"):
    r.per_world["instance.pos"]
  with pytest.raises(
    KeyError, match=r"not declared; declare it: Renderer\(\.\.\., per_world=rw\.PerWorld\(material=\['color'\]\)\)"
  ):
    r.per_world["instance.color"]
  with pytest.raises(KeyError, match="not declared"):
    r.per_world["exposure"]
  with pytest.raises(KeyError, match=r"rw\.PerWorld\(env=\['sensor\.noise'\]\)"):
    r.per_world["sensor.noise"]
  m = mjw.put_model(mjm)
  with pytest.raises(ValueError, match="the simulator state has 1 worlds, the renderer 2"):
    r.render(mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1))
  with pytest.raises(ValueError, match="pass it to render"):
    r.render()
  del m


def test_renderer_boundary_types(mjm: mujoco.MjModel) -> None:
  """Review C O1: the Renderer's public entry checks its arguments' types.

  Before, a bool nworld (True: 1 world) and a float one passed the `< 1` check (1.5 failed deep in
  an allocation), a 3-tuple resolution failed with an unpacking error, `PerWorld(alpha="no")`
  compiled translucency in ("no" is truthy), an instance pattern True selected instance 1, a seed
  of 2**32 + 5 rendered as seed 5 (Warp wraps a kernel int to 32 bits) and `update_gi(updates=0)`
  ran a periodic step instead of none."""
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  for nworld in (True, 1.5, np.float32(2.0)):  # pyrefly: ignore[bad-argument-count]  # np.float32(x): untyped
    with pytest.raises(ValueError, match="nworld must be at least 1 \\(an integer\\)"):
      rw.Renderer(scene, nworld=nworld, resolution=(8, 8))  # pyrefly: ignore[bad-argument-type]
  for resolution in ((8, 8, 8), (8.0, 8), (True, 8), "88"):
    with pytest.raises(ValueError, match="resolution must be"):
      rw.Renderer(scene, nworld=1, resolution=resolution)  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(TypeError, match="settings must be a RenderSettings"):
    rw.Renderer(scene, nworld=1, resolution=(8, 8), settings={"exposure.mode": "manual", "exposure.ev100": 2.0})  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(TypeError, match=r"PerWorld\.alpha must be a bool"):
    rw.PerWorld(material=["base_color"], alpha="no")  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(TypeError, match="an instance pattern is a name glob"):
    rw.PerWorld(pose=[True])
  with pytest.raises(TypeError, match="takes field names"):
    rw.PerWorld(light=[3])  # pyrefly: ignore[bad-argument-type]
  r = rw.Renderer(scene, nworld=1, resolution=(8, 8), per_world=rw.PerWorld(material=["color"]))
  with pytest.raises(TypeError, match="an instance pattern is a name glob"):
    r.per_world.columns("instance.color", [1.0])  # pyrefly: ignore[bad-argument-type]  # a value where a name goes, on purpose
  d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1)
  for seed in (2**32 + 5, -(2**31) - 1, True, 1.5):
    with pytest.raises(ValueError, match="seed must be in"):
      r.render(d, seed=seed)  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(ValueError, match="updates must be at least 1"):
    r.update_gi(d, updates=0)
  with pytest.raises(TypeError):
    r.set_clip(0.1)  # pyrefly: ignore[bad-argument-count]  # keyword-only
  r.render(d, seed=np.int64(2**31 - 1))  # pyrefly: ignore[bad-argument-type]  # NumPy integers are integers
  with rw.Renderer(scene, nworld=np.int64(1), resolution=[np.int32(8), 8]) as closed:  # pyrefly: ignore[bad-argument-type]
    assert closed.nworld == 1 and type(closed.nworld) is int and closed.width == 8


def test_renderer_boundary_values(mjm: mujoco.MjModel) -> None:
  """Review C O1: `set_clip` accepted a negative or NaN znear and znear > zfar (written into the
  lens table), and a bool camera (True: camera 1 with two cameras); `update_gi` skipped render's
  world-count check (it refit the BVH from mismatched poses); a NumPy world-skip mask failed with
  "must be a (2,) bool array, got bool (2,)"."""
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  r = rw.Renderer(
    scene, nworld=2, cameras=[0, 0], resolution=(8, 8), settings=rw.RenderSettings(gi=rw.settings.GI(enabled=True))
  )
  for kwargs, match in (
    ({"znear": -1.0}, "znear must be a finite distance"),
    ({"znear": float("nan")}, "znear must be a finite distance"),
    ({"zfar": float("inf")}, "zfar must be a finite distance"),
    ({"znear": True}, "znear must be a finite distance"),
    ({"znear": 5.0, "zfar": 1.0}, r"set_clip: znear \(5.0\) must be below zfar"),
    ({"zfar": 0.001, "camera": 1}, "camera 1: znear"),
    ({"znear": 0.1, "camera": True}, "camera must be in"),
    ({"znear": 0.1, "camera": 2}, "camera must be in"),
  ):
    with pytest.raises(ValueError, match=match):
      r.set_clip(**kwargs)  # pyrefly: ignore[bad-argument-type]
  r.set_clip(znear=5.0, zfar=0.0, camera=CameraSlot(1))  # zfar 0: unbounded
  r.set_clip(zfar=10.0)
  with pytest.raises(ValueError, match=r"znear \(5.0\) must be below zfar \(1.0"):
    r.set_clip(zfar=1.0)  # camera 1 keeps znear 5
  d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=3)
  with pytest.raises(ValueError, match="the simulator state has 3 worlds, the renderer 2"):
    r.update_gi(d)


def test_renderer_rejects_a_host_scene_and_takes_one_camera(mjm: mujoco.MjModel) -> None:
  """A host Scene failed with "'Scene' object has no attribute 'device'"; cameras="cam" was read as
  the cameras "c", "a", "m" ("camera 'c' not found") and one CameraSpec failed as not iterable:
  one name, spec or index is that camera alone, like a one-element list."""
  builder = rw.SceneBuilder()
  builder.add_instance("sphere", size=(0.5, 0.0, 0.0))
  with pytest.raises(TypeError, match=r"scene must be a GpuScene, got Scene: upload it first"):
    rw.Renderer(builder.build(), nworld=1, resolution=(8, 8))  # pyrefly: ignore[bad-argument-type]  # on purpose
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  for cameras in ("cam", np.int64(0), 0):
    with rw.Renderer(scene, nworld=1, cameras=cameras, resolution=(8, 8)) as r:  # pyrefly: ignore[bad-argument-type]  # a NumPy index
      assert r.ncam == 1 and r.cameras[0] == scene.cameras["cam"]
  with rw.Renderer(scene, nworld=1, cameras=rw.CameraSpec(fovy=40.0), resolution=(8, 8)) as r:
    assert r.cameras == [rw.CameraSpec(fovy=40.0)]
  with pytest.raises(ValueError, match="camera 'ca' not found"):
    rw.Renderer(scene, nworld=1, cameras="ca", resolution=(8, 8))


def test_mujoco_geom_groups_are_mujoco_groups(mjm: mujoco.MjModel) -> None:
  """A group outside [0, mjNGROUP) (no geom can be in it) was kept and rendered nothing of it,
  silently; a bare int failed with "'int' object is not iterable"."""
  for groups in ((0, 7), (-1,), (True,), (0.0,), "0"):
    with pytest.raises(ValueError, match=r"geom_groups must be integers in \[0, 6\)"):
      rw.integrations.mujoco.Options(geom_groups=groups)  # pyrefly: ignore[bad-argument-type]  # on purpose
  assert rw.integrations.mujoco.Options(geom_groups=[np.int64(3), 0]).geom_groups == (3, 0)  # pyrefly: ignore[bad-argument-type]


def test_env_id_range(mjm: mujoco.MjModel) -> None:
  maps = [np.full((8, 16, 3), 0.5, np.float32)] * 2
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), lighting=rw.Lighting(envmap=maps))
  r = rw.Renderer(scene, nworld=2, resolution=(8, 8), per_world=rw.PerWorld(env=["id"]))
  r.per_world["env.id"].assign(np.array([0, 2], dtype=np.int32))
  d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=2)
  r.render(d)  # checked on the device (clamped meanwhile), reported by the next render
  with pytest.raises(ValueError, match=r"env.id must be in \[0, 2\)"):
    r.render(d)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_per_world_exposure_range(mjm: mujoco.MjModel, bad: float) -> None:
  """Regression (sweep S3): a per-world exposure of 0 (or negative / NaN / inf) made that world's
  HDR NaN / inf. Checked on the device (exposure 1 meanwhile: finite), reported by the next
  render."""
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  settings = rw.RenderSettings().replace(outputs=rw.Outputs.of("rgb", "hdr"))
  r = rw.Renderer(scene, nworld=3, resolution=(8, 8), per_world=rw.PerWorld(env=["exposure"]), settings=settings)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  d = mjw.put_data(mjm, mjd, nworld=3)
  exposure = r.per_world["exposure"]
  good = float(exposure.numpy()[0])
  before = required(r.render(d).hdr).numpy().copy()
  exposure.assign(np.array([good, bad, 1.0], dtype=np.float32))
  out = r.render(d)
  hdr = required(out.hdr).numpy()
  assert np.all(np.isfinite(hdr)) and np.all(np.isfinite(required(out.rgb).numpy()))
  np.testing.assert_array_equal(hdr[0], before[0])  # the other worlds are unaffected
  np.testing.assert_array_equal(hdr[1], hdr[2])  # world 1 renders with exposure 1 meanwhile
  with pytest.raises(ValueError, match="exposure must be finite and > 0"):
    r.render(d)
  exposure.assign(np.full(3, good, dtype=np.float32))
  r.render(d)  # fixed: no error (the word was consumed)
  r.render(d)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_per_world_color_finite(mjm: mujoco.MjModel, bad: float) -> None:
  """Regression (sweep S4): a NaN per-world color made that world's HDR NaN. Checked on the
  device (read as black meanwhile: finite, other worlds unaffected), reported by the next render."""
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  settings = rw.RenderSettings().replace(outputs=rw.Outputs.of("rgb", "hdr"))
  r = rw.Renderer(scene, nworld=3, resolution=(8, 8), per_world=rw.PerWorld(material=["color"]), settings=settings)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  d = mjw.put_data(mjm, mjd, nworld=3)
  color = r.per_world["instance.color"]
  good = color.numpy().copy()
  before = required(r.render(d).hdr).numpy().copy()
  values = good.copy()
  values[1, :, 0] = bad
  color.assign(values)
  out = r.render(d)
  hdr = required(out.hdr).numpy()
  assert np.all(np.isfinite(hdr)) and np.all(np.isfinite(required(out.rgb).numpy()))
  np.testing.assert_array_equal(hdr[0], before[0])  # the other worlds are unaffected
  np.testing.assert_array_equal(hdr[2], before[2])
  with pytest.raises(ValueError, match="must be finite"):
    r.render(d)
  color.assign(good)
  r.render(d)  # fixed: no error (the word was consumed)
  r.render(d)


def _rgbe(image: np.ndarray) -> np.ndarray:
  peak = image.max(axis=-1)
  exponent = np.where(peak > 1e-32, np.ceil(np.log2(np.maximum(peak, 1e-32))), -128).astype(np.int32)
  mantissa = np.clip(image / np.ldexp(1.0, exponent)[..., None] * 256.0, 0, 255).astype(np.uint8)
  return np.concatenate([mantissa, (exponent + 128).clip(0, 255).astype(np.uint8)[..., None]], axis=-1)


def _rle_channel(values: np.ndarray[[int], np.dtype[np.uint8]]) -> bytes:
  """Adaptive run-length encoding of one scanline channel (runs >= 3, else literals)."""
  out = bytearray()
  i, n = 0, len(values)
  while i < n:
    run = 1
    while i + run < n and run < 127 and values[i + run] == values[i]:
      run += 1
    if run >= 3:
      out += bytes([128 + run, int(values[i])])
      i += run
      continue
    j = i
    while j < n and j - i < 128 and not (j + 2 < n and values[j] == values[j + 1] == values[j + 2]):
      j += 1
    out += bytes([j - i]) + bytes(values[i:j])
    i = j
  return bytes(out)


def _write_rle_hdr(path: Path, image: np.ndarray) -> None:
  h, w, _ = image.shape
  rgbe = _rgbe(image)
  body = bytearray()
  for y in range(h):
    body += bytes([2, 2, w >> 8, w & 0xFF])
    for c in range(4):
      body += _rle_channel(rgbe[y, :, c])
  path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode() + bytes(body))


def test_read_hdr_flat(tmp_path: Path) -> None:
  """An uncompressed (flat) RGBE file decodes to its pixels."""
  rng = np.random.default_rng(0)
  image = rng.uniform(0.0, 50.0, size=(4, 6, 3)).astype(np.float32)
  header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {image.shape[0]} +X {image.shape[1]}\n".encode()
  (tmp_path / "env.hdr").write_bytes(header + _rgbe(image).tobytes())
  # RGBE shares one exponent per pixel: each channel is exact to 1/256 of the pixel's peak.
  tolerance = image.max(axis=-1, keepdims=True) / 128.0
  assert np.all(np.abs(read_hdr(tmp_path / "env.hdr") - image) <= tolerance)


def test_read_hdr_run_length(tmp_path: Path) -> None:
  rng = np.random.default_rng(0)
  image = rng.uniform(0.0, 4.0, (5, 40, 3)).astype(np.float32)
  image[:, 10:30] = 1.5  # long runs
  _write_rle_hdr(tmp_path / "env.hdr", image)
  decoded = read_hdr(tmp_path / "env.hdr")
  expected = _rgbe(image)
  scale = np.ldexp(1.0, expected[..., 3].astype(np.int32) - 136)
  np.testing.assert_array_equal(decoded, (expected[..., :3] * scale[..., None]).astype(np.float32))


@pytest.mark.parametrize(
  "data,match",
  [
    (b"P6\n", "not a Radiance .hdr file"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n", "truncated .hdr header"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n+Y 2 +X 2\n" + bytes(16), "only '-Y H \\+X W'"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n" + bytes(10), "truncated pixel data"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 8, 200, 1]), "bad run-length data"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 9]), "width does not match"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n" + bytes([2, 2, 0]), "truncated pixel data"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 8, 136]), "bad run-length data"),
  ],
)
def test_read_hdr_malformed(tmp_path: Path, data: bytes, match: str) -> None:
  (tmp_path / "bad.hdr").write_bytes(data)
  with pytest.raises(ValueError, match=match):
    read_hdr(tmp_path / "bad.hdr")


def test_settings_take_numpy_scalars_and_reject_lookalikes(tmp_path: Path) -> None:
  """NumPy integers are accepted where integers are and stored as Python ints; a float equal to an
  integer choice, a bare item where a sequence is taken and a scalar for a vector field are
  rejected with the field's name; a non-glTF file given to `gltf.load` says so (not a JSON
  error)."""
  from raywarp.formats.gltf import api as gltf_api

  shadow = rw.settings.ShadowOutput(light=np.int64(1))  # pyrefly: ignore[bad-argument-type]  # a NumPy integer
  assert shadow.light == 1 and type(shadow.light) is int
  color = rw.settings.ColorOutput(channels=np.int64(3))  # pyrefly: ignore[bad-argument-type]  # a NumPy integer
  assert type(color.channels) is int
  for bad, match in (
    (lambda: rw.settings.ColorOutput(channels=3.0), r"outputs\.rgb\.channels must be one of"),  # pyrefly: ignore[bad-argument-type]
    (lambda: rw.settings.ShadowOutput(light=1.0), r"outputs\.shadow\.light must be a light index"),  # pyrefly: ignore[bad-argument-type]
    (lambda: rw.settings.ShadowOutput(light=True), r"outputs\.shadow\.light must be a light index"),
    (lambda: rw.settings.Sensor(vignette=0.5), r"sensor\.vignette needs 3 finite numbers"),  # pyrefly: ignore[bad-argument-type]
  ):
    with pytest.raises(ValueError, match=match):
      bad()
  with pytest.raises(TypeError, match=r"Lighting\.lights takes a sequence"):
    rw.Lighting(lights=rw.Light.sun((0.0, 0.0, -1.0)))  # pyrefly: ignore[bad-argument-type]
  obj = tmp_path / "mesh.obj"
  obj.write_text("v 0 0 0\n")
  with pytest.raises(ValueError, match="not a glTF file"):
    gltf_api.load(obj, cache=None)
