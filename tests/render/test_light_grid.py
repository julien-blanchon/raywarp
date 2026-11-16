"""Packed lights (P27, P7) and environment suns that follow their world's environment (B16)."""

import dataclasses
import math
from typing import cast

import numpy as np
import pytest
import warp as wp
from conftest import internal_module, required

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.build.variants.resolve import GRID_MIN_LIGHTS
from raywarp.core.kernels.abi.rays import SoftwareGeometry
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.worlds.sharing import PerWorld
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights import pack, sources
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.scene.model.lights import EnvironmentLight
from raywarp.settings.fields import Tri

_HDR = {"outputs": rw.Outputs.of("rgb", "hdr")}


def _dynamic_ids(grid: light_grid.Grid) -> np.ndarray[[int], np.dtype[np.int32]]:
  """The movable light IDs of `grid` (a host read)."""
  start = host_copy(grid.start)
  return host_copy(grid.lights)[start[-2] : start[-1]].astype(np.int32)


def _packed(r: rw.Renderer) -> np.ndarray:
  """The renderer's packed light table (rows, nlight) as a structured array."""
  return r._tables.light_pack.lights.numpy()


def test_spot_cone_constants_are_packed(scene):
  """P7: the cone's outer cosine and smooth-start angle are computed once per light and frame,
  not per shading point."""
  xml = """
    <mujoco>
      <worldbody>
        <light type="spot" pos="0 0 2" dir="0 0 -1" cutoff="40" intensity="5" castshadow="false"/>
        <light type="point" pos="1 0 2" intensity="5" castshadow="false"/>
        <geom type="plane" size="2 2 0.1"/>
        <camera name="c" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
      </worldbody>
    </mujoco>"""
  s = scene(xml).render()
  rows = _packed(s.r)
  outer = math.radians(40.0)
  assert rows.shape[0] == 1
  np.testing.assert_allclose(record_field(rows, "cos_outer")[0, 0], math.cos(outer), rtol=1e-5)
  inner = float(s.r.debug.frame().light_cone_inner.numpy()[0, 0])
  np.testing.assert_allclose(record_field(rows, "cone_start")[0, 0], inner * outer, rtol=1e-5)
  np.testing.assert_allclose(record_field(rows, "cone_scale")[0, 0], 1.0 / (outer - inner * outer), rtol=1e-4)
  assert record_field(rows, "cos_outer")[0, 1] < -2.0  # no cone


def test_packed_rows_follow_per_world_light_tables(scene):
  """A per-world light table packs one row per world; shared tables pack one row."""
  xml = """
    <mujoco>
      <visual><headlight active="0" ambient="0 0 0"/></visual>
      <worldbody>
        <light type="point" pos="0 0 2" intensity="5" castshadow="false"/>
        <geom type="plane" size="2 2 0.1"/>
        <camera name="c" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
      </worldbody>
    </mujoco>"""
  s = scene(xml, nworld=3, per_world=rw.PerWorld(light=["strength"]), outputs=rw.Outputs.of("rgb", "hdr")).render()
  assert _packed(s.r).shape[0] == 3  # MuJoCo's light positions are per world (d.light_xpos)
  intensity = s.r.per_world["light.strength"]
  table = intensity.numpy()
  table[1] *= 0.0
  intensity.assign(table)
  s.render()
  rows = _packed(s.r)
  assert record_field(rows, "intensity")[1, 0] == 0.0 and record_field(rows, "intensity")[0, 0] > 0.0
  hdr = s.hdr()
  assert hdr[1].max() == 0.0 and hdr[0].max() > 0.0


_ATTENUATION_XML = """
  <mujoco>
    <visual><headlight active="0" ambient="0 0 0"/></visual>
    <worldbody>
      <light type="{kind}" pos="0 0 2" dir="0 0 -1" cutoff="40"
             diffuse="1 1 1" intensity="{intensity}"
             attenuation="1 0 0.5" castshadow="false" range="10"/>
      <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
      <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="20"/>
    </worldbody>
  </mujoco>"""


@pytest.mark.parametrize("kind", ["point", "spot"])
def test_classic_attenuation_is_live_and_isolated_per_world(scene, kind):
  """At the centre d=2 m, (1, 0, .5) gives one third of default direct light.

  A batched model coefficient table is read again each frame and changes only its own world.
  """
  s = scene(_ATTENUATION_XML.format(kind=kind, intensity=0), nworld=2, cam_res=(5, 5), **_HDR)
  s.m.light_attenuation = wp.array(np.array([[[1.0, 0.0, 0.5]], [[1.0, 0.0, 0.0]]], np.float32), dtype=wp.vec3, device="cpu")
  s.render()
  a = s.hdr()[:, 2, 2, :].copy()
  np.testing.assert_allclose(a[0], a[1] / 3.0, rtol=1e-5, atol=1e-6)
  rows = _packed(s.r)
  np.testing.assert_array_equal(record_field(rows, "attenuation")[:, 0], [[1, 0, 0.5], [1, 0, 0]])
  s.m.light_attenuation.assign(np.array([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.5]]], np.float32))
  s.render()
  b = s.hdr()[:, 2, 2, :]
  np.testing.assert_allclose(b[0], a[1], rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(b[1], a[0], rtol=1e-5, atol=1e-6)


def test_photometric_light_ignores_classic_attenuation(scene):
  """MuJoCo's photometric intensity/range remains the core's physical light model."""
  s = scene(_ATTENUATION_XML.format(kind="point", intensity=20), cam_res=(5, 5), **_HDR).render()
  identity = s.r.debug.frame(s.d).light_attenuation
  assert identity.shape == (1, s.mjm.nlight)  # shared placeholder, not one table per world
  before = s.hdr().copy()
  np.testing.assert_array_equal(record_field(_packed(s.r), "attenuation")[0, 0], [1, 0, 0])
  s.m.light_attenuation.assign(np.array([[[0.0, 0.0, 1.0]]], np.float32))
  s.render()
  np.testing.assert_array_equal(s.hdr(), before)
  assert s.r.debug.frame(s.d).light_attenuation is identity  # no per-frame allocation


def test_classic_directional_light_ignores_attenuation(scene):
  """A sun has no point-to-light distance and keeps its existing intensity."""
  s = scene(_ATTENUATION_XML.format(kind="directional", intensity=0), cam_res=(5, 5), **_HDR).render()
  before = s.hdr().copy()
  s.m.light_attenuation.assign(np.array([[[0.0, 0.0, 1.0]]], np.float32))
  s.render()
  np.testing.assert_array_equal(s.hdr(), before)


def test_env_sun_follows_its_world_environment():
  """B16: the dome's sun (env_sun) turns with the world's env.yaw, scales with env.intensity and
  is off in a world showing no map of its own (env.id); it used to light every world as world 0."""
  fs = internal_module("feature_scenes")
  per_world = rw.PerWorld(env=["yaw", "intensity"])
  rig = fs.load("environment", settings={**_HDR, "gi.enabled": False}, res=(16, 16), nworld=3, per_world=per_world)
  r = rig.r
  yaw = r.per_world["env.yaw"]
  base = float(yaw.numpy()[0])
  yaw.assign(np.array([base, base + math.pi, base], np.float32))
  scale = r.per_world["env.intensity"]
  scale.assign(np.array([1.0, 1.0, 0.0], np.float32))
  rig.render()
  assert len(rig.gpu.environment.suns) == 1
  rows = _packed(r)[:, rig.gpu.source_lights]  # the sun: after the dome's lights
  # A half turn about z flips the sun's horizontal direction.
  np.testing.assert_allclose(record_field(rows, "dir")[1, :2], -record_field(rows, "dir")[0, :2], atol=1e-5)
  np.testing.assert_allclose(record_field(rows, "dir")[1, 2], record_field(rows, "dir")[0, 2], atol=1e-6)
  assert record_field(rows, "intensity")[2] == 0.0 and record_field(rows, "intensity")[0] > 0.0
  # The dark world sees no sun (and no sky): black.
  assert rig.hdr()[2].max() == 0.0 and rig.hdr()[0].max() > 0.0


# ---------------------------------------------------------------------------------------------
# Light scaling (W32, M5 lane L): the light-cost counters, the light caps, packed lights.


def _room_xml(n: int, cast: bool = True) -> str:
  """A grey plane and a box under `n` point lights of decreasing intensity, seen from above."""
  shadow = "true" if cast else "false"
  lights = "".join(
    f'<light type="point" pos="{0.4 * np.cos(k)} {0.4 * np.sin(k)} {1.2 + 0.2 * k}" intensity="{10.0 / (k + 1)}" '
    f'range="1000" castshadow="{shadow}"/>'
    for k in range(n)
  )
  return f"""
    <mujoco>
      <visual><headlight active="0" ambient="0 0 0"/></visual>
      <asset><material name="mat" rgba="0.8 0.8 0.8 1" roughness="0.6"/></asset>
      <worldbody>
        {lights}
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <geom type="box" pos="0.1 0 0.1" size="0.1 0.1 0.1" material="mat"/>
        <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="30"/>
      </worldbody>
    </mujoco>
  """


# Every light shaded at every point with its own shadow ray: the counters are exact.
_EXACT = {
  **_HDR,
  "lighting.light_grid": False,
  "shadows.max_casters": None,
  "reflections.enabled": False,
  "lighting.area_lights": "point",
}


def test_counters_count_the_light_loop(scene):
  """P23: one sample and one shaded layer per pixel, every light evaluated at it, one shadow ray
  per lit caster; the counters change no pixel."""
  plain = scene(_room_xml(4), **_EXACT).render()
  with tuning.override(counters=True):
    counted = scene(_room_xml(4), **_EXACT).render()
  assert counted.r.features.counters and not plain.r.features.counters
  np.testing.assert_array_equal(counted.hdr(), plain.hdr())
  c = counted.r.counters()
  assert c["samples"] == 1.0 and c["layers"] == 1.0
  assert c["lights"] == 4.0 and c["cheap"] == 0.0
  assert 3.0 <= c["shadow_rays"] <= 4.0  # a light below a box side's horizon needs no ray
  assert c["closest_rays"] == 1.0 and c["reflection_rays"] == 0.0
  with pytest.raises(ValueError, match=r"counters build"):
    plain.r.counters()
  # A camera with its own resolution: the means are over its pixels (they were over ncam x the
  # Renderer's resolution: 1/4 of the samples here).
  spec = dataclasses.replace(counted.scene.cameras["top"], resolution=(counted.r.width, counted.r.height))
  with tuning.override(counters=True):
    r = rw.Renderer(counted.scene, nworld=1, cameras=[spec], resolution=(64, 64), settings=counted.r.settings)
    r.render(counted.d)
  assert r.counters() == c


def test_budget_remainder_is_never_pitch_black(scene):
  """A floor point under a small box: the strongest light (straight above) is blocked, four side
  lights reach it. With a budget of one shadow ray the untraced side lights used to take the
  blocked light's visibility (0): pitch black where most of the light arrives (the Warehouse floor
  at fast). They now keep at least half of their light."""
  side = "".join(
    f'<light type="point" pos="{1.5 * np.cos(a)} {1.5 * np.sin(a)} 1.5" intensity="2" range="1000" castshadow="true"/>'
    for a in (0.3, 1.9, 3.5, 5.1)
  )
  xml = f"""
    <mujoco>
      <visual><headlight active="0" ambient="0 0 0"/></visual>
      <worldbody>
        <light type="point" pos="0 0 2" intensity="20" range="1000" castshadow="true"/>
        {side}
        <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        <geom type="box" pos="0 0 1" size="0.1 0.1 0.02" rgba="0.8 0.8 0.8 1"/>
        <camera name="top" pos="0 0 0.6" xyaxes="1 0 0 0 1 0" fovy="2"/>
      </worldbody>
    </mujoco>
  """
  base = {**_EXACT}
  exact = scene(xml, cam_res=(4, 4), **base).render().hdr()[0].mean()
  budget = scene(xml, cam_res=(4, 4), **{**base, "shadows.max_casters": 1}).render().hdr()[0].mean()
  assert exact > 0.0
  assert budget >= 0.45 * exact


_TWO_ROOMS = """
  <mujoco>
    <visual><headlight active="0" ambient="0 0 0"/></visual>
    <worldbody>
      <light type="point" pos="-1 0 1" intensity="5" range="1000" castshadow="true" active="{active}"/>
      <light type="point" pos="-3.9 3.9 1.9" intensity="0.05" range="1000" castshadow="true"/>
      <geom type="plane" size="4 4 0.1" rgba="0.8 0.8 0.8 1"/>
      <geom type="box" pos="0 0 1" size="0.05 4 1"/>
      <geom type="box" pos="0 0 2.05" size="4 4 0.05"/>
      <camera name="left" pos="-1.5 0 1.5" xyaxes="1 0 0 0 1 0" fovy="30"/>
    </worldbody>
  </mujoco>
"""


def test_light_grid_lists_lights_on_in_other_worlds(scene):
  """R: the light grid is built from world 0. A light off there but on in world 1 (light.active
  per world) must still be listed: the left room's cells listed only the dim corner light, and
  world 1 rendered the room as dim as world 0."""
  s = scene(
    _TWO_ROOMS.format(active="false"),
    nworld=2,
    cam_res=(8, 8),
    per_world=rw.PerWorld(light=["active"]),
    lighting=rw.settings.Lights(light_grid=True),
    **_HDR,
  )
  table = s.r.per_world["light.active"]
  values = table.numpy()
  values[1, 0] = True
  table.assign(values)
  hdr = s.render().hdr()
  assert s.r.features.culling and s.r.context().light_grid is not None
  assert hdr[1].mean() > 10.0 * hdr[0].mean()


def _dark_lamp_renderer(per_world: rw.PerWorld) -> rw.Renderer:
  """Two rooms split by a wall, a lamp of intensity 0 in the left one and a dim one in the right
  one, 2 worlds, the light grid on, the camera in the left room."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  b.add_instance("box", material=m, size=(0.05, 4.0, 1.0), xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]])
  b.add_instance("box", material=m, size=(4.0, 4.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, 2.05]])
  for x, power in ((-1.0, 0.0), (1.0, 50.0)):
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=power, radius=0.005, xform=np.c_[np.eye(3), [x, 0.0, 1.0]]))
  settings = rw.RenderSettings().replace(
    **{"outputs": rw.Outputs.of("hdr"), "environment.enabled": False, "lighting.light_grid": True}
  )
  r = rw.Renderer(
    rw.upload(b.build()),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=30.0)],
    resolution=(8, 8),
    settings=settings,
    per_world=per_world,
  )
  r.camera_pose.assign(np.array([[[-1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]]] * 2, np.float32))
  return r


def _static_lights(grid: light_grid.Grid) -> set[int]:
  """The light IDs some cell of `grid` lists."""
  start = host_copy(grid.start)
  return set(host_copy(grid.lights)[: start[-2]].tolist())


def test_light_grid_lists_no_dark_light():
  """BUG-744: a light of zero intensity whose emission nothing writes later (street lamps
  authored off) is flooded by no bake and listed in no cell."""
  r = _dark_lamp_renderer(rw.PerWorld())
  r.render()
  grid = r.context().light_grid
  assert grid is not None and _static_lights(grid) == {1}


def test_light_grid_floods_dark_lights_whose_emission_varies():
  """A dark light whose strength and color are declared per world is flooded like the others: a
  world that lights it keeps its culled lists (valid reach) and shows it."""
  r = _dark_lamp_renderer(rw.PerWorld(light=["strength", "color"]))
  for field, value in (("strength", 5.0e4), ("color", (1.0, 1.0, 1.0))):  # a dark light's color is 0
    table = r.per_world[f"light.{field}"]
    values = table.numpy()
    values[1, 0] = value
    table.assign(values)
  r.render()
  grid = r.context().light_grid
  hdr = required(r.outputs.hdr).numpy()[:, 0]
  assert grid is not None and _static_lights(grid) == {0, 1}
  # The reach rows stay valid (culled lists); a far field holding the lit lamp drops (`same_power`).
  np.testing.assert_array_equal(grid.valid.numpy()[:2], [1, 1])
  assert hdr[1].mean() > 10.0 * hdr[0].mean()


def test_light_grid_never_culls_lights_moved_per_world():
  """R: a light whose position is declared per world reaches different cells in every world: the
  grid (world 0's) cannot cull it, so every light is evaluated everywhere. Moved into the right
  room in world 1, it lights world 1's right room (it was culled there)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  b.add_instance("box", material=m, size=(0.05, 4.0, 1.0), xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]])
  b.add_instance("box", material=m, size=(4.0, 4.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, 2.05]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.005, xform=np.c_[np.eye(3), [-1.0, 0.0, 1.0]]))
  settings = rw.RenderSettings().replace(
    **{"outputs": rw.Outputs.of("hdr"), "environment.enabled": False, "lighting.light_grid": True}
  )
  r = rw.Renderer(
    rw.upload(b.build()),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=30.0)],
    resolution=(8, 8),
    settings=settings,
    per_world=rw.PerWorld(light=["pos"]),
  )
  r.camera_pose.assign(np.array([[[1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]]] * 2, np.float32))
  pos = r.per_world["light.pos"]
  values = pos.numpy()
  values[1, 0] = (1.0, 0.0, 1.0)
  pos.assign(values)
  r.render()
  hdr = required(r.outputs.hdr).numpy()[:, 0]
  assert r.features.culling and r.context().light_grid is None
  assert hdr[0].max() == 0.0 and hdr[1].mean() > 0.0


def test_movable_directional_light_is_listed_once():
  """A directional light declared movable (`PerWorld.dynamic_lights`) is evaluated from the movable
  list only: the grid listed every directional light in every cell after removing the movable
  ones, so a movable sun was shaded twice."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, xform=np.c_[np.eye(3), [0.0, 0.0, 3.0]]))
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.005, xform=np.c_[np.eye(3), [-1.0, 0.0, 1.0]]))
  settings = rw.RenderSettings().replace(
    **{"outputs": rw.Outputs.of("hdr"), "environment.enabled": False, "lighting.light_grid": True}
  )
  r = rw.Renderer(
    rw.upload(b.build()),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=30.0)],
    resolution=(8, 8),
    settings=settings,
    per_world=rw.PerWorld(dynamic_lights=[0]),
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  grid = r.context().light_grid
  assert grid is not None
  start, ids = grid.start.numpy(), grid.lights.numpy()
  assert (ids[: start[-2]] != 0).all()
  np.testing.assert_array_equal(_dynamic_ids(grid), [0])


def test_light_grid_bake_cost_is_bounded():
  """Sweep S5 / BUG-744: a tiny `lighting.light_grid_cell` baked for minutes, and Rivermark's
  640 m town over 228 lit lights for 2-5 min on an RTX PRO 6000. The cell grows until the bake
  needs at most MAX_TESTS lattice x light tests; the default cell of an interior is kept."""
  from raywarp.core.shade.lights import grid as light_grid

  def tests(extent: list[float], cell: float, n: int) -> tuple[float, int]:
    c = light_grid._cell_size(np.asarray(extent), cell, n)
    return c, math.prod(2 * math.ceil(e / c) + 1 for e in extent) * n

  cell, n = tests([16.0, 16.0, 4.0], 0.05, 256)  # many_lights at cell 0.05
  assert cell > 0.05 and n <= light_grid.MAX_TESTS
  cell, n = tests([640.0, 600.0, 17.0], 1.0, 228)  # Rivermark's lit lights
  assert n <= light_grid.MAX_TESTS and cell > 1.25
  assert tests([14.4, 21.8, 7.4], 1.0, 4) == (1.0, 31 * 45 * 17 * 4)  # Office: unchanged


def test_light_grid_sees_through_instances_hidden_per_world():
  """The grid is traced in world 0 only: an instance hidden (or moved) per world must not cull the
  lights behind it in the other worlds. The wall between the rooms is hidden in world 1: the
  left room's light reaches world 1's right room as without the grid."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  b.add_instance("box", material=m, size=(0.05, 4.0, 1.0), xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]], path="wall")
  b.add_instance("box", material=m, size=(4.0, 4.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, 2.05]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.005, xform=np.c_[np.eye(3), [-1.0, 0.0, 1.0]]))
  gpu = rw.upload(b.build())

  def render(grid: bool) -> np.ndarray:
    settings = rw.RenderSettings().replace(
      **{
        "outputs": rw.Outputs.of("hdr"),
        "environment.enabled": False,
        "lighting.light_grid": grid,
        "lighting.light_grid_cell": 0.25,
      }
    )
    r = rw.Renderer(
      gpu, nworld=2, cameras=[rw.CameraSpec(fovy=30.0)], resolution=(8, 8), settings=settings,
      per_world=rw.PerWorld(visible=["wall"]),
    )  # fmt: skip
    r.camera_pose.assign(np.array([[[1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]]] * 2, np.float32))
    vis = r.per_world["instance.visible"]
    vis.assign(np.array([[1], [0]], np.int32))
    r.render()
    assert not grid or r.context().light_grid is not None
    return required(r.outputs.hdr).numpy()[:, 0]

  culled, full = render(True), render(False)
  assert culled[0].max() == 0.0 and full[1].mean() > 0.0
  np.testing.assert_allclose(culled[1], full[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gi", [False, True])
def test_per_world_lighting_keeps_per_world_gi(scene, gi):
  """Per-world light strengths keep GI (two distinct worlds: one shared volume tinted per world,
  `gi.Relight`) without the world-0 baked-lighting warning; direct-light culling needs no delayed
  warning either."""
  import warnings

  kw = {"gi": rw.settings.GI(enabled=gi, bounds=((-2.0, -2.0, -0.1), (2.0, 2.0, 2.0)), rays=16, bake_updates=1)}
  s = scene(_TWO_ROOMS.format(active="true"), nworld=2, cam_res=(4, 4), per_world=rw.PerWorld(light=["strength"]), **_HDR, **kw)
  table = s.r.per_world["light.strength"]
  values = table.numpy()
  values[1] *= 0.5
  table.assign(values)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(3):
      s.render()
  baked = [w for w in caught if "baked" in str(w.message)]
  assert not baked, [str(w.message) for w in caught]
  assert s.r.features.gi == gi
  if gi:
    vol = s.r.debug.gi_volume
    assert vol is not None and vol.relight is not None
  assert "gpu.gi.per_world_inputs" not in s.r.scene.report.dropped


def _sheet(transmission: float, specular: float = 0.0) -> np.ndarray:
  """HDR of a rough white sheet (alpha 0.5, or transmission 0.5) seen from above under a point
  light, nothing behind."""
  b = rw.SceneBuilder()
  alpha = 1.0 if transmission > 0.0 else 0.5
  m = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_weight=4.0 * specular**2,
      specular_roughness=1.0,
      transmission_weight=transmission,
      geometry_opacity=alpha,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  b.add_instance("box", material=m, size=(1.0, 1.0, 0.01))
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=1.0e5, radius=0.005, xform=np.c_[np.eye(3), [0.0, 0.0, 2.0]]))
  settings = rw.RenderSettings().replace(**{"outputs": rw.Outputs.of("hdr"), "environment.enabled": False})
  r = rw.Renderer(
    rw.upload(b.build()),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=20.0)],
    resolution=(8, 8),
    settings=settings,
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  assert r.features.transparency and r.features.transmissive == (transmission > 0.0)
  return required(r.outputs.hdr).numpy()[0, 0]


def test_transmissive_layers_keep_the_untransmitted_diffuse():
  """P28 / BUG-029: a translucent layer of a transmissive material keeps its Fresnel-weighted
  specular, its diffuse lobe weighted by 1 - t (OpenPBR mixes diffuse and transmission by t),
  and transmits the rest: a rough sheet without specular at t = 0.5 is lit like an alpha-blended
  one of coverage 0.5, and clear (t = 1) it is black."""
  half, blended = _sheet(0.5), _sheet(0.0)
  assert blended.mean() > 0.0
  np.testing.assert_allclose(half.mean(), blended.mean(), rtol=0.02)
  assert _sheet(1.0).max() < 1.0e-2 * blended.mean()


def test_area_lights_that_do_not_reach_take_no_shadow_ray():
  """Found with the P23 counters (Isaac Hospital: 7.3 shadow rays per pixel, no light reaching):
  an area light behind the point (a one-sided rect facing away) was a 'caster' with zero
  radiance, so the shadow budget traced it. Two such rects and a point light, a budget of 2: one
  shadow ray per pixel (the point light's), not two. The rects sit outside the frame: their
  emitting faces look up at the camera, and a pixel that sees a (camera-visible) light takes no
  shadow ray."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  up = np.diag(np.array([1.0, -1.0, -1.0]))  # local -Z (the emission) along world +Z: away from the floor
  for x in (-1.0, 1.0):  # the frame spans |x| <= 0.35 m at their height (fovy 20 deg, 2 m away)
    b.add_light(
      rw.Light(
        kind=rw.scene.LightKind.RECT,
        intensity=100.0,
        width=0.5,
        height=0.5,
        shadow_enable=True,
        xform=np.c_[up, [x, 0.0, 1.0]],
      )
    )
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.SPHERE, intensity=1.0e4, radius=0.01, shadow_enable=True, xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]]
    )
  )
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "environment.enabled": False,
      "shadows.max_casters": 2,
      "lighting.max_exact": None,
      "lighting.light_grid": False,
    }
  )
  with tuning.override(counters=True):
    r = rw.Renderer(rw.upload(b.build()), nworld=1, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(8, 8), settings=settings)
    r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
  # Three lights take the small form: the budget selects nothing (every caster that reaches is
  # traced, the others take no ray).
  assert r.features.area_lights and r.features.budget == 0
  assert r.counters()["shadow_rays"] == 1.0
  assert r.counters()["lights"] == 3.0
  assert r.counters()["light_terms"] == r.counters()["light_casters"] == 1.0


def test_light_cap_leaves_a_small_area_light_table_whole():
  """lighting.max_exact=2 under the four LTC lights of lights_area: a table the small light form
  takes is shaded whole whatever the cap (ranking four lights costs more than it saves), so the
  capped render is the uncapped one, with no cheap estimate."""
  fs = internal_module("feature_scenes")
  base = {"outputs": rw.Outputs.of("rgb", "hdr")}
  with tuning.override(counters=True):
    full = fs.load("lights_area", settings=base, res=(48, 36))
    capped = fs.load("lights_area", settings={**base, "lighting.max_exact": 2}, res=(48, 36))
    assert capped.r.features.area_lights and capped.r.features.light_cap == 0
    assert capped.r.features == full.r.features
    full.render()
    capped.render()
  assert capped.r.counters()["cheap"] == 0.0
  np.testing.assert_array_equal(capped.hdr(), full.hdr())


def _lit_scene(nlight: int, grid: Tri) -> rw.Renderer:
  """A host scene lit by one sun and `nlight - 1` point lights, lighting.light_grid=`grid`."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(4.0, 4.0, 0.1))
  b.add_instance("box", material=m, size=(0.3, 0.3, 0.3), xform=np.c_[np.eye(3), [0.0, 0.0, 0.3]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  for k in range(nlight - 1):
    b.add_light(
      rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.005, xform=np.c_[np.eye(3), [1.0, 0.1 * k, 1.0]])
    )
  settings = rw.RenderSettings(lighting=rw.settings.Lights(light_grid=grid))
  return rw.Renderer(rw.upload(b.build()), nworld=1, cameras=[rw.CameraSpec(fovy=40.0)], resolution=(8, 8), settings=settings)


@pytest.mark.parametrize(
  ("nlight", "grid", "culling"),
  [
    (1, "auto", False),
    (2, "auto", False),
    (GRID_MIN_LIGHTS - 1, "auto", False),
    (GRID_MIN_LIGHTS, "auto", True),
    (1, True, True),
    (GRID_MIN_LIGHTS, False, False),
  ],
)
def test_light_grid_auto_from_a_light_count(nlight, grid, culling):
  """P (robotics defaults; g1 regression before): below GRID_MIN_LIGHTS lights the grid's cell
  lookup and dynamic light loop cost more than what it culls (a single sun is listed in every
  cell: fast on the g1 Newton scene was 15 % slower on an L4), so "auto" resolves it off;
  True / False force it."""
  assert _lit_scene(nlight, grid).features.culling == culling


@pytest.mark.parametrize(("nlight", "culling"), [(1, False), (GRID_MIN_LIGHTS, True)])
def test_light_grid_auto_on_mujoco_models(scene, nlight, culling):
  """P: the same rule on MuJoCo models (m.nlight)."""
  lights = '<light type="directional" dir="0 0 -1"/>' + "".join(
    f'<light type="point" pos="{0.1 * k} 0 2"/>' for k in range(nlight - 1)
  )
  xml = f'<mujoco><worldbody>{lights}<geom type="plane" size="2 2 0.1"/><camera name="c" pos="0 0 2"/></worldbody></mujoco>'
  assert scene(xml, cam_res=(8, 8)).r.features.culling == culling


# ---------------------------------------------------------------------------------------------
# Movable light hints keep static culling and retain conservative reach fallback.


_cell, _light = sources.make_cell(4, culling=True)


@wp.kernel
def _candidates(
  # Model:
  lit: Lights,
  # In:
  point: wp.vec3,
  # Out:
  ids: wp.array2d[int],
  counts: wp.array[int],
):
  """Collect the exact light candidates of one point in each world."""
  world = wp.tid()
  cell, first, last, _far = _cell(lit, world, point, wp.vec3(0.0, 0.0, 1.0))
  counts[world] = last - first
  for j in range(first, last):
    ids[world, j - first] = _light(lit, cell, j)


def _grid() -> light_grid.Grid:
  """Two sparse cells with light 1 movable and absent from both static lists."""
  return _assemble(_four_lights(), start=[0, 1, 2], ids=[0, 2], dynamic=[1])


def _four_lights() -> list[Light]:
  """Four active white lights of unit intensity at x = 0..3."""
  packed = [Light() for _ in range(4)]
  for i, light in enumerate(packed):
    light.pos, light.active = wp.vec3(float(i), 0.0, 1.0), True
    light.color, light.intensity = wp.vec3(1.0), 1.0
  return packed


def _assemble(
  packed: list[Light],
  *,
  start: list[int],
  ids: list[int],
  dynamic: list[int],
  far_members: tuple[int, ...] = (),
  unflooded: tuple[int, ...] = (),
) -> light_grid.Grid:
  """A two-cell grid over `packed` (one row): these lists, movable lights and far-field lights."""
  return light_grid.assemble(
    origin=wp.vec3(0.0),
    cell=1.0,
    dims=wp.vec3i(2, 1, 1),
    start=np.asarray(start, np.int32),
    lights=np.asarray(ids, np.int32),
    dynamic=np.asarray(dynamic, np.int32),
    snapshot=wp.array(packed, dtype=Light, device="cpu").reshape((1, len(packed))),
    far=np.zeros((0, 3), np.float32),
    far_members=np.asarray(far_members, np.int32),
    nlight=len(packed),
    unflooded=np.asarray(unflooded, np.int32),
  )


def _collect(grid: light_grid.Grid, point: wp.vec3) -> tuple[np.ndarray, np.ndarray]:
  """Read candidates without compiling a renderer or loading a scene."""
  lit = Lights()
  light_grid.wire(lit, grid)
  ids = wp.full((2, 4), -1, dtype=int, device="cpu")
  counts = wp.empty(2, dtype=int, device="cpu")
  wp.launch(_candidates, dim=2, inputs=[lit, point], outputs=[ids, counts], device="cpu")
  return ids.numpy(), counts.numpy()


def test_declared_motion_preserves_sparse_candidates_and_undeclared_fallback() -> None:
  """One changed world remains sparse for hinted motion, falls back otherwise, and resets."""
  grid = _grid()
  current = np.repeat(grid.snapshot.numpy(), 2, axis=0)
  record_field(current, "pos")[1, 1] = (100.0, 0.0, 1.0)
  grid = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(grid.valid.numpy(), [1, 1, 1, 1])
  ids, counts = _collect(grid, wp.vec3(0.5))
  np.testing.assert_array_equal(counts, [2, 2])
  np.testing.assert_array_equal(ids[:, :2], [[0, 1], [0, 1]])
  ids, _ = _collect(grid, wp.vec3(1.5, 0.5, 0.5))
  np.testing.assert_array_equal(ids[:, :2], [[2, 1], [2, 1]])
  record_field(current, "pos")[1, 3] = (100.0, 0.0, 1.0)
  grid = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  ids, counts = _collect(grid, wp.vec3(0.5))
  np.testing.assert_array_equal(counts, [2, 4])
  np.testing.assert_array_equal(ids[1], [0, 1, 2, 3])
  current[1] = grid.snapshot.numpy()[0]
  grid = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(grid.valid.numpy(), [1, 1, 1, 1])
  _, counts = _collect(grid, wp.vec3(4.0))
  np.testing.assert_array_equal(counts, [4, 4])


def test_grid_cache_preserves_movable_hint() -> None:
  """Cache reload retains the movable list and its reach-check mask."""
  original = _grid()
  restored = light_grid.from_arrays(light_grid.to_arrays(original), device=wp.get_device("cpu"))
  np.testing.assert_array_equal(_dynamic_ids(restored), [1])
  np.testing.assert_array_equal(restored.dynamic_mask.numpy(), [0, 1, 0, 0])


@pytest.mark.parametrize("ids, error", [([1, 1], ValueError), ([-1], ValueError), ([4], ValueError), ([True], TypeError)])
def test_movable_light_indices_are_validated(ids: list[int], error: type[Exception]) -> None:
  """A hint cannot silently omit an invalid or duplicated light."""
  with pytest.raises(error):
    PerWorld(dynamic_lights=ids).light_ids(4)


def test_static_and_all_dynamic_cell_ranges() -> None:
  """Cell-local offsets preserve empty-hint behavior and support an entirely movable table."""
  packed = _four_lights()
  static = _assemble(packed, start=[0, 1, 2], ids=[0, 2], dynamic=[])
  ids, counts = _collect(static, wp.vec3(1.5, 0.5, 0.5))
  np.testing.assert_array_equal(counts, [1, 1])
  np.testing.assert_array_equal(ids[:, 0], [2, 2])
  movable = _assemble(packed, start=[0, 0, 0], ids=[], dynamic=[0, 1, 2, 3])
  ids, counts = _collect(movable, wp.vec3(0.5))
  np.testing.assert_array_equal(counts, [4, 4])
  np.testing.assert_array_equal(ids, [[0, 1, 2, 3], [0, 1, 2, 3]])


def test_color_intensity_and_off_keep_static_reach_valid() -> None:
  """Changing radiance and disabling static lights cannot enlarge their baked footprint; only a
  far-field light's change drops the row's far fields."""
  grid = _assemble(_four_lights(), start=[0, 1, 2], ids=[0, 2], dynamic=[1], far_members=(3,))
  current = np.repeat(grid.snapshot.numpy(), 2, axis=0)
  record_field(current, "active")[1, 0] = False
  record_field(current, "color")[1, 2] = (0.1, 0.3, 1.0)
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 1, 1, 1])  # reach, then far fields
  record_field(current, "intensity")[1, 3] = 1000.0
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 1, 1, 0])


def test_dark_light_lighting_up_invalidates_its_row() -> None:
  """BUG-744: a light that emitted nothing at the bake is listed in no cell: a world where it
  emits (intensity and color above 0, on) evaluates every light; off, it keeps the lists."""
  packed = _four_lights()
  packed[2].intensity = 0.0
  grid = _assemble(packed, start=[0, 1, 2], ids=[0, 3], dynamic=[1], unflooded=(2,))
  current = np.repeat(grid.snapshot.numpy(), 2, axis=0)
  record_field(current, "intensity")[1, 2] = 3.0
  record_field(current, "active")[1, 2] = False
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 1, 1, 1])
  record_field(current, "active")[1, 2] = True
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 0, 1, 1])


def test_turned_directional_light_keeps_static_reach_valid() -> None:
  """BUG-737: a directional light is listed in every cell, so a world turning it (an environment
  sun under a per-world env.yaw) keeps the row's culled lists; one turned into a point light
  drops them (every light evaluated)."""
  packed = _four_lights()
  packed[0].type, packed[0].dir = lights.LIGHT_DIRECTIONAL, wp.vec3(0.0, 0.0, -1.0)
  grid = _assemble(packed, start=[0, 1, 2], ids=[0, 2], dynamic=[1])
  current = np.repeat(grid.snapshot.numpy(), 2, axis=0)
  record_field(current, "dir")[1, 0] = (0.6, 0.0, -0.8)
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 1, 1, 1])
  record_field(current, "type")[1, 0] = lights.LIGHT_POINT
  checked = light_grid.checked(grid, wp.array(current, dtype=Light, device="cpu"), 4)
  np.testing.assert_array_equal(checked.valid.numpy(), [1, 0, 1, 1])


# cap 1 < 4 lights: slotted sources.
_capped = cast(
  "sources.Source[wp.Vector[int, int]]",
  sources.make_source(nlight=4, culling=True, cap=1, has_spot=False, columns=(-1, -1)),
)


_full = cast(
  "sources.Source[wp.Vector[int, int]]",
  sources.make_source(nlight=4, culling=False, cap=1, has_spot=False, columns=(-1, -1)),
)


@wp.kernel
def _capped_candidates(
  # Model:
  geo: SoftwareGeometry,
  lit: Lights,
  # Out:
  ids: wp.array2d[int],
  rest: wp.array2d[wp.vec3],
):
  """Compare capped selection and remainder for the same positive light candidates."""
  world = wp.tid()
  slots, first, _last, remainder = _capped.begin(geo, lit, world, wp.vec3(0.5), wp.vec3(0.0, 0.0, 1.0))
  ids[world, 0] = _capped.light_of(lit, slots, first)
  rest[world, 0] = remainder
  full, ffirst, _flast, fremainder = _full.begin(geo, lit, world, wp.vec3(0.5), wp.vec3(0.0, 0.0, 1.0))
  ids[world, 1] = _full.light_of(lit, full, ffirst)
  rest[world, 1] = fremainder


def test_capped_selection_and_remainder_include_movable_light_once() -> None:
  """The strongest movable light and diffuse remainder agree with full-table capped shading."""
  grid = _grid()
  packed = np.repeat(grid.snapshot.numpy(), 2, axis=0)
  record_field(packed, "type")[:] = lights.LIGHT_POINT
  record_field(packed, "color")[:] = 1.0
  record_field(packed, "intensity")[:] = [1.0, 10.0, 0.0, 0.0]
  record_field(packed, "range")[:] = 100.0
  record_field(packed, "attenuation")[:] = (1.0, 0.0, 0.0)
  lit = Lights()
  lit.lights = wp.array(packed, dtype=Light, device="cpu")
  lit.hot = pack.hot_records(lit.lights)
  lit.nlight = 4
  light_grid.wire(lit, grid)
  ids = wp.empty((2, 2), dtype=int, device="cpu")
  rest = wp.empty((2, 2), dtype=wp.vec3, device="cpu")
  wp.launch(_capped_candidates, dim=2, inputs=[SoftwareGeometry(), lit], outputs=[ids, rest], device="cpu")
  np.testing.assert_array_equal(ids.numpy(), [[1, 1], [1, 1]])
  np.testing.assert_allclose(rest.numpy()[:, 0], rest.numpy()[:, 1], rtol=0, atol=1e-6)


def test_env_sun_of_every_map():
  """BUG-009: each environment map's sun becomes a directional light of its own, on only in the
  worlds that show its map (`env.id`); it used to be cut out of map 0 only, so a per-world HDRI
  lost its sun and sun shadows. The suns are shaded after the scene's lights, no light-table
  column."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  b.add_instance("box", size=(2.0, 2.0, 0.05), material=m, xform=np.eye(4)[:3])
  for col in (4, 20):  # two skies, their suns 180 degrees apart in azimuth
    sky = np.full((32, 64, 3), 0.5, np.float32)
    sky[6:8, col : col + 2] = 5.0e4
    tex = b.add_texture(sky, colorspace="lin_rec709_scene")
    b.add_environment(EnvironmentLight(texture=tex, path=f"/sky{col}"))
  gpu = rw.upload(b.build())
  assert gpu.environment.suns.env.tolist() == [0, 1]
  assert gpu.source_lights == 0 and gpu.nlight == 2
  assert gpu.tables["light.active"].shape[1] == 0
  r = rw.Renderer(gpu, nworld=2, cameras=[rw.CameraSpec()], resolution=(8, 8), per_world=rw.PerWorld(env=["id"]))
  r.per_world["env.id"].assign(np.array([0, 1], np.int32))
  r.render()
  rows = _packed(r)
  for w in range(2):
    on = [int(gpu.environment.suns.env[i]) for i in range(2) if record_field(rows, "active")[w, i]]
    assert on == [w], (w, on)


def test_light_grid_keeps_the_sun_through_a_slit():
  """BUG-018: a hard sun reaching the floor through a 4 cm slit in a ceiling, a point light in
  the room. The cells under the slit see the point light but none of their lattice points sees
  the sun, which the grid used to cull there (the slit's sunlit stripe vanished); directional
  lights are now listed in every cell."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(3.0, 3.0, 0.1))
  b.add_instance("box", material=m, size=(1.605, 3.0, 0.05), xform=np.c_[np.eye(3), [-1.395, 0.0, 2.05]])
  b.add_instance("box", material=m, size=(1.375, 3.0, 0.05), xform=np.c_[np.eye(3), [1.625, 0.0, 2.05]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=50.0, radius=0.01, xform=np.c_[np.eye(3), [1.0, 0.0, 1.5]]))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=1.0e4, angle=0.0))
  host = rw.upload(b.build())
  rows = []
  for grid in (False, True):
    settings = rw.RenderSettings().replace(
      **{
        "outputs": rw.Outputs.of("hdr"),
        "environment.enabled": False,
        "lighting.light_grid": grid,
        "lighting.light_grid_cell": 1.0,
      }
    )
    r = rw.Renderer(host, nworld=1, cameras=[rw.CameraSpec(fovy=10.0)], resolution=(32, 32), settings=settings)
    r.camera_pose.assign(np.array([[[0.23, 0.0, 1.9, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
    assert r.features.culling is grid
    rows.append(required(r.outputs.hdr).numpy()[0, 0, 16, :, :3].mean(axis=-1))
  off, on = rows
  assert off.max() > 20.0 * np.median(off)  # the sunlit stripe
  np.testing.assert_allclose(on, off, rtol=1e-4, atol=1e-6)


def test_environment_upload_is_cached(tmp_path, monkeypatch):
  """The processed environment maps and their suns come from the disk cache on a second upload
  (same content: identical maps, suns and report); another map content misses."""
  from raywarp.core.pipeline.upload import environment as env_upload
  from raywarp.scene.model.report import LoadReport

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  calls = []
  real = env_upload._processed
  monkeypatch.setattr(env_upload, "_processed", lambda *a, **k: calls.append(1) or real(*a, **k))

  def host(peak: float) -> rw.Scene:
    b = rw.SceneBuilder()
    sky = np.full((32, 64, 3), 0.5, np.float32)
    sky[6:8, 4:6] = peak
    b.add_environment(EnvironmentLight(texture=b.add_texture(sky, colorspace="lin_rec709_scene"), path="/sky"))
    return b.build()

  dev = wp.get_device("cpu")
  runs = []
  for peak in (5.0e4, 5.0e4, 4.0e4):
    report = LoadReport()
    env = env_upload.environment(host(peak), report, env_sun=True, device=dev)
    runs.append((env.maps.sh.numpy(), env.suns.direction, dict(report.approximated)))
  assert len(calls) == 2  # the second upload hit the cache
  np.testing.assert_array_equal(runs[0][0], runs[1][0])
  np.testing.assert_array_equal(runs[0][1], runs[1][1])
  assert runs[0][2] == runs[1][2] and runs[0][2]


@pytest.mark.parametrize(("intensity", "height"), [(10.0, 4.0), (0.157, 1.8)])
def test_host_lights_have_no_absolute_cutoff(intensity, height):
  """A point light above a white floor below 1 lux there (10 cd at 4 m; a 0.157 cd bulb at 1.8 m,
  the dim bulb that left a Cornell room black): the old 1 lux window (at least 1 m) lit nothing;
  host lights follow the inverse square law at any distance (range inf)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, specular_weight=0.0))
  b.add_instance("plane", material=m, size=(20.0, 20.0, 0.1))
  b.add_light(
    rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=intensity, radius=0.0, xform=np.c_[np.eye(3), [0.0, 0.0, height]])
  )
  gpu = rw.upload(b.build())
  assert np.isinf(gpu.tables["light.range"].numpy()).all()
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "environment.enabled": False,
      "exposure.mode": "manual",
      "exposure.ev100": 0.0,
      "lighting.light_grid": False,
    }
  )
  r = rw.Renderer(gpu, nworld=1, cameras=[rw.CameraSpec(fovy=2.0)], resolution=(4, 4), settings=settings)
  r.camera_pose.assign(np.array([[[0.0, 0.0, height - 0.5, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  hdr = required(r.outputs.hdr).numpy()[0, 0, 1:3, 1:3].mean()
  # White Lambertian floor straight below the light: L = E / pi = I / (d^2 pi).
  np.testing.assert_allclose(hdr, intensity / (height**2 * math.pi), rtol=0.02)


def _point_lights(rows: list[tuple[float, float, float, float]]) -> Lights:
  """A one-row packed table of white unbounded point lights (x, y, z, intensity)."""
  out = []
  for x, y, z, intensity in rows:
    light = Light()
    light.type, light.active, light.range = lights.LIGHT_POINT, True, math.inf
    light.pos, light.color, light.intensity = wp.vec3(x, y, z), wp.vec3(1.0, 1.0, 1.0), intensity
    light.attenuation, light.cos_outer = wp.vec3(1.0, 0.0, 0.0), lights.NO_CONE
    out.append(light)
  lit = Lights()
  lit.lights, lit.nlight = wp.array(out, dtype=Light, device="cpu").reshape((1, len(out))), len(out)
  return lit


def test_far_field_holds_the_weak_distant_lights():
  """A cell keeps its strong and its near lights; distant lights leave it, the weakest first, while
  together at most FAR_FRACTION of its estimate, into its far field: their irradiance at the
  centre (sum) and on the six axis normals (an ambient cube)."""
  # One cell [0, 1]^3 (centre 0.5): a strong light 1.5 m above, a weak one 4 m away along +x, a
  # strong one 4 m away along -x (too strong to leave), a near weak one (closer than FAR_CELLS).
  lit = _point_lights([(0.5, 0.5, 2.0, 100.0), (4.5, 0.5, 0.5, 0.5), (-3.5, 0.5, 0.5, 100.0), (1.2, 0.5, 0.5, 0.01)])
  mask = np.ones((1, 4), bool)
  pairs = light_grid._pairs(mask, lit, lo=np.zeros(3), cell=1.0, dims=[1, 1, 1])
  np.testing.assert_array_equal(pairs.farable, [False, True, True, False])
  drop = light_grid._far_drop(pairs, 1)
  np.testing.assert_array_equal(drop, [False, True, False, False])
  rgb, pos, neg = light_grid._far_fields(pairs, drop, 1)
  np.testing.assert_allclose(rgb[0], [0.5 / 16.0] * 3, rtol=1e-6)
  np.testing.assert_allclose(pos[0], [0.5 / 16.0, 0.0, 0.0], rtol=1e-5)
  np.testing.assert_allclose(neg[0], 0.0, atol=1e-9)
  np.testing.assert_array_equal(light_grid._strongest_first(pairs, ~drop), [0, 2, 3])


def test_far_field_keeps_a_many_light_row_close_to_every_light_exact():
  """A row of 24 unbounded point lights over a long floor: the light grid (far fields on) renders
  within 2 % of every light evaluated exactly (no grid), with shorter cell lists."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0, specular_weight=0.0))
  b.add_instance("plane", material=m, size=(30.0, 3.0, 0.1))
  for i in range(24):
    x = -23.0 + 2.0 * i
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=50.0, radius=0.0, xform=np.c_[np.eye(3), [x, 0.0, 1.5]]))
  gpu = rw.upload(b.build())
  images = []
  for grid in (False, True):
    settings = rw.RenderSettings().replace(
      **{
        "outputs": rw.Outputs.of("hdr"),
        "environment.enabled": False,
        "exposure.mode": "manual",
        "exposure.ev100": 0.0,
        "lighting.light_grid": grid,
        "lighting.max_exact": None,
        "shadows.enabled": False,
      }
    )
    r = rw.Renderer(gpu, nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(48, 16), settings=settings)
    r.camera_pose.assign(np.array([[[0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
    images.append(required(r.outputs.hdr).numpy()[0, 0].mean(-1))
    if grid:
      baked = required(r._baked.grid)
      start = baked.start.numpy()[:-1]
      assert np.diff(start).max() < 24
      assert baked.far.shape[0] > 0
  exact, gridded = images
  floor = exact > 0.05 * exact.max()
  np.testing.assert_allclose(gridded[floor], exact[floor], rtol=0.02)
