"""The environment's sun (`SceneOptions.env_sun`, on by default) on MuJoCo scenes: a shadowed
directional light after the model's lights, following each world's env.id / env.yaw /
env.intensity as run-time data (no compile), none (and no change) for a map without one."""

import math
from collections.abc import Sequence

import mujoco
import mujoco_warp as mjw
import numpy as np
from conftest import default_settings, required
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.build.variants.resolve import GRID_MIN_LIGHTS
from raywarp.scene.images import latlong
from raywarp.scene.lighting.rig import Envmap

# A grey floor and a box hovering over its middle, seen from straight above (image right = +x),
# lit by the environment only.
_XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom name="box" type="box" pos="0 0 0.5" size="0.25 0.25 0.1" rgba="0.7 0.7 0.7 1"/>
    <camera name="top" pos="0 0 4" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <camera name="sky" pos="0 0 1" xyaxes="0 1 0 0.7071068 0 -0.7071068" fovy="20"/>
  </worldbody>
</mujoco>
"""
RES = (32, 32)


def _sky(*, sun: bool) -> np.ndarray:
  """A clear sky over a grey ground (64 x 128), with a sun 45 degrees up towards +x (the map's
  centre column) holding most of the light; without `sun` the same sky, overcast-smooth."""
  h, w = 64, 128
  dirs = latlong.latlong_directions(h, w)
  z = dirs[..., 2]
  sky = np.where((z > 0.0)[..., None], (0.6 + 0.4 * z)[..., None] * np.array([0.55, 0.75, 1.0]), 0.25).astype(np.float32)
  if sun:
    toward = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    texel = np.unravel_index(np.argmax(dirs @ toward), z.shape)
    sky[texel] += 30.0 / latlong.texel_solid_angles(h, w)[texel]
  return sky


def _scene(
  envmap: Envmap | Sequence[Envmap],
  *,
  nworld: int = 1,
  per_world: rw.PerWorld | None = None,
  env_sun: bool = True,
  outputs: rw.Outputs | None = None,
  camera: str = "top",
  xml: str = _XML,
):
  """(MuJoCo Warp data, GpuScene, Renderer) of `xml` lit by `envmap` (a map or a bank), seen by
  `camera`."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  lighting = rw.Lighting(envmap=envmap, envmap_intensity=1.0)
  gpu = rw.integrations.mujoco.scene(mjm, m, d, lighting=lighting, options=rw.SceneOptions(env_sun=env_sun))
  settings = default_settings(
    outputs=outputs or rw.Outputs.of("hdr", "seg"), **{"exposure.mode": "manual", "exposure.ev100": 0.0}
  )
  r = rw.Renderer(gpu, nworld=nworld, cameras=[camera], resolution=RES, per_world=per_world, settings=settings)
  return d, gpu, r


def _floor(r: rw.Renderer) -> tuple[np.ndarray, np.ndarray]:
  """((nworld, h, w) floor brightness (mean RGB), (nworld, h, w) floor mask) of the last render."""
  hdr = required(r.outputs.hdr).numpy()[:, 0, ..., :3].mean(axis=-1)
  seg = required(r.outputs.seg).numpy()[:, 0]
  return hdr, seg == 0


def _shadow_x(hdr: np.ndarray, floor: np.ndarray) -> float:
  """Mean image column of the darkest floor pixels (the box's sun shadow), centred on 0."""
  values = np.where(floor, hdr, np.inf)
  dark = values < np.quantile(values[floor], 0.05)
  return float(np.argwhere(dark)[:, 1].mean() - (hdr.shape[1] - 1) / 2)


def test_the_sun_is_a_light_after_the_models():
  """The sunny map gives one sun: a light shaded after the model's (none here), no light-table
  column; the box's shadow falls away from the sun (-x); the map keeps the sky."""
  d, gpu, r = _scene(_sky(sun=True))
  assert len(gpu.environment.suns) == 1
  assert gpu.nlight == 1 and gpu.source_lights == 0
  assert gpu.capabilities.sun_disc
  r.render(d)
  hdr, floor = _floor(r)
  assert _shadow_x(hdr[0], floor[0]) < -3.0
  assert hdr[0][floor[0]].min() < 0.6 * np.median(hdr[0][floor[0]])
  assert any("gpu.environment.sun" in k for k in gpu.report.approximated)


def test_the_sky_shows_the_sun_disc():
  """A camera looking at the sun sees its disc drawn over the refilled map (`background.sun_disc`):
  where the map had its sun, holding the light the map's sun texel held (the excess over the sky
  times the pixels' solid angle; env_sun off shows the map's own texel), and nothing past it."""
  views = []
  for env_sun in (True, False):
    d, _, r = _scene(_sky(sun=True), env_sun=env_sun, outputs=rw.Outputs.of("hdr"), camera="sky")
    r.render(d)
    hdr = required(r.outputs.hdr).numpy()[0, 0, ..., :3].mean(axis=-1).astype(np.float64)
    sky = float(np.median(hdr))
    views.append(np.where(hdr > 5.0 * sky, hdr - sky, 0.0))  # the sun: far over the smooth sky
  disc = views[0]
  power = [float(v.sum()) for v in views]
  assert 0.8 < power[0] / power[1] < 1.25
  centre = [np.average(np.argwhere(v > 0.0), axis=0, weights=v[v > 0.0]) for v in views]
  assert (np.linalg.norm(centre[0] - centre[1], keepdims=True) < 1.5).all()
  assert 0 < (disc > 0.0).sum() < 0.2 * disc.size  # a disc, not a glow over the sky


def test_the_sun_follows_each_worlds_yaw_and_intensity():
  """Two worlds turned half a turn apart: their shadows fall on opposite sides; a world at
  intensity 0 is black (no sun left in it)."""
  per_world = rw.PerWorld(env=["yaw", "intensity"])
  d, _, r = _scene(_sky(sun=True), nworld=3, per_world=per_world)
  r.per_world["env.yaw"].assign(np.array([0.0, math.pi, 0.0], np.float32))
  r.per_world["env.intensity"].assign(np.array([1.0, 1.0, 0.0], np.float32))
  r.render(d)
  hdr, floor = _floor(r)
  assert _shadow_x(hdr[0], floor[0]) < -3.0
  assert _shadow_x(hdr[1], floor[1]) > 3.0
  assert hdr[2].max() == 0.0


def test_a_sun_turned_per_world_keeps_the_light_grid():
  """BUG-737: worlds turning the map (env.yaw) turn its sun, a directional light listed in every
  grid cell: every world keeps its culled light lists (a turned sun made the world evaluate every
  light: Office 4x slower under per-world yaw)."""
  points = "".join(f'<light type="point" pos="{k * 0.3 - 1.0} 0 2" castshadow="false"/>' for k in range(GRID_MIN_LIGHTS))
  xml = _XML.replace("<worldbody>", f"<worldbody>{points}")
  d, _, r = _scene(_sky(sun=True), nworld=3, per_world=rw.PerWorld(env=["yaw"]), xml=xml)
  assert r.features.culling
  r.per_world["env.yaw"].assign(np.array([0.0, 1.0, 2.0], np.float32))
  r.render(d)
  grid = r.context().light_grid
  assert grid is not None
  np.testing.assert_array_equal(grid.valid.numpy(), [1] * 6)  # reach, then far fields, per world


def test_switching_sunny_and_overcast_maps_never_compiles():
  """A bank (sunny, overcast) over two worlds: switching which world shows which, turning and
  dimming them loads no Warp module and keeps the Features; the sun lights only the world that
  shows the sunny map."""
  per_world = rw.PerWorld(env=["id", "yaw", "intensity"])
  d, gpu, r = _scene([_sky(sun=True), _sky(sun=False)], nworld=2, per_world=per_world)
  assert gpu.environment.suns.env.tolist() == [0]
  r.per_world["env.id"].assign(np.array([0, 1], np.int32))
  r.render(d)
  features = r.features
  first, floor = _floor(r)
  with module_loads() as loads:
    r.per_world["env.id"].assign(np.array([1, 0], np.int32))
    r.per_world["env.yaw"].assign(np.array([0.3, 0.0], np.float32))
    r.per_world["env.intensity"].assign(np.array([2.0, 1.0], np.float32))
    r.render(d)
  assert loads == []
  assert r.features == features
  swapped, _ = _floor(r)
  np.testing.assert_allclose(swapped[1], first[0], rtol=1e-5)  # the sunny world, moved
  for img, mask in ((first[1], floor[1]), (swapped[0], floor[0])):  # the overcast one: no sharp shadow
    assert img[mask].min() > 0.6 * np.median(img[mask])


def test_a_map_without_a_sun_changes_nothing():
  """An overcast map: no sun, the model's light count, and the same image as with env_sun off."""
  images = []
  for env_sun in (True, False):
    d, gpu, r = _scene(_sky(sun=False), env_sun=env_sun)
    assert len(gpu.environment.suns) == 0 and gpu.nlight == 0
    r.render(d)
    images.append(required(r.outputs.hdr).numpy().copy())
  np.testing.assert_array_equal(images[0], images[1])


def test_env_sun_off_keeps_the_sun_in_the_map():
  """SceneOptions(env_sun=False): no light; the sun lights the scene through the map (its
  irradiance kept, no sharp shadow)."""
  d, gpu, r = _scene(_sky(sun=True), env_sun=False)
  assert len(gpu.environment.suns) == 0 and gpu.nlight == 0
  r.render(d)
  hdr, floor = _floor(r)
  assert hdr[0][floor[0]].min() > 0.6 * np.median(hdr[0][floor[0]])


def test_floor_brightness_is_kept_by_the_split():
  """The lit floor far from the box gets the same light whether the sun is in the map (SH
  irradiance) or a directional light (within the SH's own error)."""
  lit = []
  for env_sun in (True, False):
    d, _, r = _scene(_sky(sun=True), env_sun=env_sun)
    r.render(d)
    hdr, _ = _floor(r)
    lit.append(float(np.median(hdr[0, :, -6:])))  # +x edge: away from the shadow
  assert 0.85 < lit[0] / lit[1] < 1.15


# A floor and a free ball, no light at all: the ambient fill (the headlight's ambient) only.
_DARK = """
<mujoco>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0 0 0" specular="0 0 0"/></visual>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.7 0.7 0.7 1"/>
    <body pos="0 0 0.2"><freejoint/><geom type="sphere" size="0.1" rgba="0.8 0.8 0.8 1"/></body>
    <camera name="top" pos="0 0 2.2" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
</mujoco>
"""


def test_gi_bakes_over_worlds_without_lights(scene):
  """A scene without lights over two worlds bakes GI (its (nworld, 0) light rows are no per-world
  GI input; they once failed the bake: `Invalid indexing in slice 0:0:1`)."""
  s = scene(_DARK, nworld=2, cam_res=(16, 16), outputs=rw.Outputs.of("hdr"), **{"gi.enabled": True}).render()
  hdr = s.hdr()
  assert np.isfinite(hdr).all()
  assert (hdr[..., 0].reshape(2, -1).max(axis=1) > 0.0).all()


def _host_with_panel() -> rw.Scene:
  """A host scene: a floor, an emissive panel (fitted as a rect light) and a sunny sky."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6)))
  glow = b.add_material(rw.Material(emission_luminance=50.0, emission_color=(1.0, 1.0, 1.0)))
  b.add_instance("box", size=(2.0, 2.0, 0.05), material=grey, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
  b.add_instance("box", size=(0.3, 0.3, 0.005), material=glow, xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]])
  sky = b.add_texture(_sky(sun=True), colorspace="lin_rec709_scene")
  b.add_environment(rw.scene.EnvironmentLight(texture=sky, path="/sky"))
  return b.build()


def test_a_host_scene_with_panels_and_a_sun_renders():
  """Emissive panels (table lights) and an environment sun (after them): the per-instance light
  tables are the tables' width; the Renderer builds and renders."""
  gpu = rw.upload(_host_with_panel(), "cpu")
  assert len(gpu.environment.suns) == 1 and gpu.source_lights >= 1
  assert len(gpu.light_instance) == gpu.source_lights and (gpu.light_instance >= 0).any()
  r = rw.Renderer(
    gpu,
    nworld=1,
    cameras=[rw.CameraSpec(fovy=60.0)],
    resolution=(8, 8),
    settings=default_settings(outputs=rw.Outputs.of("hdr")),
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  assert np.isfinite(required(r.outputs.hdr).numpy()).all()


def test_the_shadow_output_takes_the_sun():
  """outputs.shadow on a scene lit by a sunny map only: "sun" and "strongest" are the env sun
  (light `source_lights`), its shadow under the box."""
  for light in ("sun", "strongest"):
    outputs = rw.Outputs.of("hdr", shadow=rw.settings.ShadowOutput(light=light))
    d, _, r = _scene(_sky(sun=True), outputs=outputs)
    r.render(d)
    shadow = required(r.outputs.shadow).numpy()[0, 0]
    assert shadow.min() == 0.0 and shadow.max() == 1.0


def test_environment_off_turns_the_sun_off():
  """environment.enabled=False: the map no longer lights the scene, nor does its sun."""
  d, _, r = _scene(_sky(sun=True))
  r.update_settings(**{"environment.enabled": False})
  r.render(d)
  packed = r._tables.light_pack.lights.numpy()  # the packed sun
  assert not record_field(packed, "active").any()
