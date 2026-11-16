"""Domain randomisation end to end, in the scenarios training pipelines randomise (CPU, small).

Every scenario randomises per world through the public API (`PerWorld` tables, MuJoCo Warp's
batched model tables, Isaac Lab's material writers) and checks:

* per-world correctness: world w of the batch renders exactly what a 1-world renderer given world
  w's values renders (no leakage between worlds, no stale cache);
* no recompile and no module load after the first frame, whatever is written;
* no allocation in a steady frame (writes from device arrays allocate nothing either);
* the update cost (printed: `pytest -s`).

  (a) tabletop manipulation: textures, colours, roughness / metalness, lights, a distractor, the
      camera jittered through the model (pose and field of view);
  (b) legged robot outdoors: environment map swap, yaw and intensity, sun direction, ground
      texture swap and placement;
  (c) indoor room: lamps on / off, colour, intensity and position, wall materials (host scene);
  (d) Isaac Lab: per-environment visual-material writes through the adapter, every step;
  (e) MuJoCo Warp: batched model tables (geom / material colours, textures, lights, cameras),
      written every step, K-row tables tiled as world % K;
  (f) extremes: every world different, every table randomised every step for 100 steps, edge
      values, invalid values reported.
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Callable, Mapping
from typing import Final

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import UNLIT_FAST_CHANGES, recipe, required
from fixtures import conformance

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.scene.images.mips import linear_to_srgb_u8
from raywarp.scene.model.types import MaterialId, Slot, TextureId, TextureRef
from raywarp.settings.fields import exposure_from_ev100

# The test resolution (internal/benchmarks/specs/w9_dr.py renders the scenarios larger: not Final).
RES: tuple[int, int] = (32, 24)
NWORLD: Final = 4
# A per-world value table: (nworld, ...) for `per_world[field]`, (nworld, ncolumn) for a
# `columns(field, pattern)` key "field:pattern", (nworld, ...) for a batched model table "m.<name>".
type Values = Mapping[str, np.ndarray]
# Values to write: host or device tables.
type Writes = Mapping[str, np.ndarray | wp.array]

TABLETOP: Final = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.6 0.4 0.2" rgb2="0.3 0.2 0.1" width="32" height="32"/>
    <texture name="tiles" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.1 0.1 0.1" width="32" height="32"/>
    <material name="table" texture="wood" texrepeat="4 4"/>
    <material name="floor" texture="tiles" texrepeat="8 8"/>
    <material name="cube" rgba="0.8 0.1 0.1 1"/>
    <material name="can" rgba="0.1 0.6 0.9 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.5 -0.5 2" dir="-0.2 0.2 -1" type="spot" cutoff="60" castshadow="true"/>
    <light name="fill" dir="0.3 0.2 -1" directional="true" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="table" type="box" size="0.6 0.4 0.02" pos="0 0 0.5" material="table"/>
    <body name="cube" pos="0.1 0 0.56"><freejoint/><geom name="cube" type="box" size="0.04 0.04 0.04" material="cube"/></body>
    <body name="can" pos="-0.15 0.1 0.57"><freejoint/><geom name="can" type="cylinder" size="0.03 0.05" material="can"/></body>
    <geom name="distractor" type="sphere" size="0.05" pos="0.2 0.2 0.57" rgba="0.9 0.9 0.1 1"/>
    <camera name="cam" pos="0 -1.0 1.1" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
  </worldbody>
</mujoco>
"""

QUADRUPED: Final = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.2 0.5 0.1" rgb2="0.3 0.4 0.1" width="32" height="32"/>
    <texture name="gravel" type="2d" builtin="checker" rgb1="0.5 0.5 0.45" rgb2="0.3 0.3 0.28" width="32" height="32"/>
    <material name="ground" texture="grass" texrepeat="6 6"/>
    <material name="rock" texture="gravel"/>
    <material name="body" rgba="0.9 0.5 0.1 1" metallic="0.2" roughness="0.4"/>
  </asset>
  <worldbody>
    <light name="sun" dir="0.3 0.4 -0.8" directional="true" castshadow="true"/>
    <geom name="ground" type="plane" size="10 10 0.1" material="ground"/>
    <geom name="boulder" type="sphere" size="0.3" pos="1.2 0.8 0.1" material="rock"/>
    <body name="torso" pos="0 0 0.45">
      <freejoint/>
      <geom name="torso" type="box" size="0.3 0.12 0.06" material="body"/>
      <body pos="0.25 0.12 0"><joint type="hinge" axis="0 1 0"/><geom name="fl" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.03" material="body"/></body>
      <body pos="0.25 -0.12 0"><joint type="hinge" axis="0 1 0"/><geom name="fr" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.03" material="body"/></body>
      <body pos="-0.25 0.12 0"><joint type="hinge" axis="0 1 0"/><geom name="hl" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.03" material="body"/></body>
      <body pos="-0.25 -0.12 0"><joint type="hinge" axis="0 1 0"/><geom name="hr" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.03" material="body"/></body>
    </body>
    <camera name="chase" pos="-1.6 -1.2 0.9" xyaxes="0.6 -0.8 0 0.3 0.22 0.93" fovy="60"/>
  </worldbody>
</mujoco>
"""


def _sky(zenith: tuple[float, float, float], horizon: tuple[float, float, float], sun: float) -> np.ndarray:
  """A (16, 32, 3) equirectangular sky: a zenith-to-horizon gradient, a dark ground and a sun
  spot of radiance `sun` (0: overcast)."""
  rows = np.linspace(0.0, 1.0, 8)[:, None, None]
  sky = (1.0 - rows) * np.asarray(zenith, np.float32) + rows * np.asarray(horizon, np.float32)
  img = np.concatenate([np.broadcast_to(sky, (8, 32, 3)), np.full((8, 32, 3), 0.05, np.float32)]).astype(np.float32)
  img[2, 5] += sun
  return img


SKIES: Final = (
  _sky((0.2, 0.4, 1.0), (0.7, 0.8, 1.0), 40.0),  # clear noon
  _sky((0.3, 0.2, 0.4), (1.0, 0.5, 0.2), 15.0),  # sunset
  _sky((0.5, 0.5, 0.5), (0.6, 0.6, 0.6), 0.0),  # overcast
)


def _mujoco(
  xml: str, nworld: int, batched: Mapping[str, np.ndarray] | None = None
) -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """The model of `xml` with `nworld` worlds; `batched`: "m.<name>" -> (nworld, ...) model rows
  (MuJoCo Warp's domain randomisation: a batched model table, read per world)."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mjw.put_model(mjm)
  for key, rows in (batched or {}).items():
    name = key.removeprefix("m.")
    table = getattr(m, name)
    setattr(m, name, wp.array(np.ascontiguousarray(rows), dtype=table.dtype, device=table.device))
  d = mjw.make_data(mjm, nworld=nworld)
  mjw.forward(m, d)
  return mjm, m, d


def _batched_rows(m: mjw.Model, name: str, nworld: int) -> np.ndarray:
  """(nworld, ...) rows of model table `name` (its first row repeated)."""
  return np.repeat(getattr(m, name).numpy()[:1], nworld, axis=0)


def _write(r: rw.Renderer, values: Writes) -> None:
  """Writes the renderer-side tables of `values` ("m.*" keys are model rows: not written here)."""
  for key, v in values.items():
    if key.startswith("m."):
      continue
    field, _, pattern = key.partition(":")
    table = r.per_world.columns(field, pattern) if pattern else r.per_world[field]
    table.assign(v)


def _row(values: Values, w: int) -> dict[str, np.ndarray]:
  """World w's values as a 1-world table set."""
  return {k: v[w : w + 1] for k, v in values.items()}


def _check_worlds_render_alone(make: Callable[[Values], np.ndarray], values: Values, worlds: tuple[int, ...]) -> None:
  """World w of the batch renders exactly what a 1-world renderer given world w's values does."""
  batch = make(values)
  assert batch.shape[0] == next(iter(values.values())).shape[0]
  for w in worlds:
    alone = make(_row(values, w))
    np.testing.assert_array_equal(batch[w], alone[0], err_msg=f"world {w}")
  pairs = [(a, b) for a in range(batch.shape[0]) for b in range(a)]
  assert all(np.any(batch[a] != batch[b]) for a, b in pairs), "randomised worlds should all differ"


def _steady(render: Callable[[], object], write: Callable[[], None], steps: int) -> tuple[list[str], list[int], float]:
  """(modules loaded, bytes allocated by the renders, mean write time [ms]) over `steps` write +
  render steps after a warm frame."""
  render()
  loads: list[str] = []
  allocated: list[int] = []
  spent = 0.0
  with conformance.module_loads() as loaded:
    for _ in range(steps):
      t0 = time.perf_counter()
      write()
      wp.synchronize()
      spent += time.perf_counter() - t0
      with conformance.allocations() as sizes:
        render()
      allocated += sizes
  loads += loaded
  return loads, allocated, 1e3 * spent / steps


# (a) Tabletop manipulation ----------------------------------------------------------------------

TABLETOP_WORLD: Final = rw.PerWorld(
  material=["base_color", "specular_roughness", "base_metalness", "tex_variation", "color"],
  light=["strength", "color", "active"],
  env=["exposure"],
  visible=["distractor"],
)


def _tabletop_values(r: rw.Renderer, m: mjw.Model, rng: np.random.Generator, nworld: int) -> dict[str, np.ndarray]:
  """Random per-world values of every tabletop table, the camera jittered through the model
  (scales of the scene's light strengths and exposure: `_scene_values`)."""
  pw = r.per_world
  base = _scene_values(r)
  colors = pw["material.base_color"].numpy()
  colors[..., :3] = rng.uniform(0.05, 1.0, colors[..., :3].shape)
  tint = pw["instance.color"].numpy()
  tint[..., :3] = rng.uniform(0.05, 1.0, tint[..., :3].shape)
  strength = base["light.strength"]
  active = np.ones(strength.shape, bool)
  active[:, 0] = rng.integers(0, 2, nworld).astype(bool)
  cam_pos = _batched_rows(m, "cam_pos", nworld) + rng.uniform(-0.05, 0.05, (nworld, 1, 3)).astype(np.float32)
  return {
    "material.base_color": colors,
    "instance.color": tint,
    "material.specular_roughness": rng.uniform(0.05, 1.0, colors.shape[:2]).astype(np.float32),
    "material.base_metalness": rng.uniform(0.0, 1.0, colors.shape[:2]).astype(np.float32),
    "material.tex_variation": rw.randomize.jitter(
      pw["material.tex_variation"].numpy(),
      rng,
      uv_offset=1.0,
      uv_rotation=np.pi,
      hue=0.5,
    ),
    "light.strength": (strength * rng.uniform(0.3, 2.0, strength.shape)).astype(np.float32),
    "light.color": rng.uniform(0.3, 1.0, (*strength.shape, 3)).astype(np.float32),
    "light.active": active,
    "exposure": (base["exposure"] * rng.uniform(0.6, 1.6, nworld)).astype(np.float32),
    "instance.visible:distractor": rng.integers(0, 2, (nworld, 1)).astype(np.int32),
    "m.cam_pos": cam_pos.astype(np.float32),
    "m.cam_fovy": rng.uniform(40.0, 60.0, (nworld, 1)).astype(np.float32),
  }


def _scene_values(r: rw.Renderer) -> dict[str, np.ndarray]:
  """The scene's light strengths (nworld, nlight) and the renderer's exposure (nworld,): random
  draws scale these, never the previous draw (no random walk)."""
  strength = r.scene.tables["light.strength"].numpy()
  exposure = exposure_from_ev100(r.settings.exposure.render_ev100())
  return {"light.strength": np.repeat(strength[:1], r.nworld, axis=0), "exposure": np.full(r.nworld, exposure, np.float32)}


def _tabletop(values: Values | None, nworld: int) -> tuple[rw.Renderer, mjw.Model, mjw.Data]:
  """The tabletop renderer (GI off: worlds compare exactly; GI per world: test_gi_groups.py)."""
  batched = {k: v for k, v in (values or {}).items() if k.startswith("m.")}
  mjm, m, d = _mujoco(TABLETOP, nworld, batched)
  settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
  r = rw.Renderer(
    rw.integrations.mujoco.scene(mjm, m, d),
    nworld=nworld,
    cameras=["cam"],
    resolution=RES,
    settings=settings,
    per_world=TABLETOP_WORLD,
  )
  return r, m, d


def _render_tabletop(values: Values) -> np.ndarray:
  """(nworld, H, W, 3) HDR of the tabletop with `values`."""
  nworld = next(iter(values.values())).shape[0]
  r, _, d = _tabletop(values, nworld)
  with r:
    r.render(d)
    _write(r, values)
    return required(r.render(d).hdr).numpy()[:, 0, ..., :3]


def test_tabletop_worlds_render_their_own_values() -> None:
  """(a) Textures, colours, roughness / metalness, lights, a distractor and a jittered camera:
  every world renders its own values, as alone."""
  with wp.ScopedDevice("cpu"):
    r, m, _ = _tabletop(None, NWORLD)
    values = _tabletop_values(r, m, np.random.default_rng(0), NWORLD)
    r.close()
    _check_worlds_render_alone(_render_tabletop, values, worlds=(0, 3))


def test_tabletop_randomised_every_step_never_compiles_or_allocates() -> None:
  """(a) Every table re-randomised every step from device arrays (as a training loop writes them):
  no module loads after the first frame, no allocation in a frame or a write."""
  with wp.ScopedDevice("cpu"):
    r, m, d = _tabletop(None, NWORLD)
    rng = np.random.default_rng(1)
    draws = [_tabletop_values(r, m, rng, NWORLD) for _ in range(3)]
    device = {k: [wp.array(v[k], dtype=_dtype(r, k), device="cpu") for v in draws] for k in draws[0] if not k.startswith("m.")}
    step = [0]

    def write() -> None:
      k = step[0] % len(draws)
      step[0] += 1
      _write(r, {key: arrays[k] for key, arrays in device.items()})

    with conformance.allocations() as written:
      write()
    assert not written, f"a write from device arrays allocated {written}"
    features = r.features
    loads, allocated, ms = _steady(lambda: r.render(d), write, steps=4)
    print(f"tabletop: {len(device)} tables written in {ms:.2f} ms per step")
    assert not loads, f"modules loaded after the first frame: {loads}"
    assert not allocated, f"steady frames allocated {allocated} bytes"
    assert r.features is features
    r.close()


def test_column_writes_after_the_first_frame_load_nothing_and_reuse_their_view() -> None:
  """(a) The first `columns(...).assign` after the first frame loads no module (the warm-up loaded
  the scatter's), and asking for the same columns again (every step) gives the same view: no new
  device column ids, no allocation."""
  with wp.ScopedDevice("cpu"):
    r, _, d = _tabletop(None, 2)
    r.render(d)
    shown = wp.zeros((2, 1), dtype=int)
    with conformance.module_loads() as loads, conformance.allocations() as allocated:
      view = r.per_world.columns("instance.visible", "distractor")
      view.assign(shown)
      assert r.per_world.columns("instance.visible", "distractor") is view
      assert r.per_world.columns("instance.visible", ["distractor"]) is view
    assert not loads, loads
    assert len(allocated) <= 1, allocated  # the first view's column ids, once
    r.close()


@pytest.mark.slow  # a GI bake and its relit re-bake on the CPU (~20 s)
def test_tabletop_gi_rebake_after_randomisation_loads_no_module() -> None:
  """(a) With GI, the first randomisation re-bakes the volume as a relit one: its modules (the
  relight, the per-light slots, the re-bake bookkeeping) were loaded with the first frame."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _mujoco(TABLETOP, NWORLD)
    settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"))
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d),
      nworld=NWORLD,
      cameras=["cam"],
      resolution=RES,
      settings=settings,
      per_world=TABLETOP_WORLD,
    )
    r.render(d)
    values = _tabletop_values(r, m, np.random.default_rng(9), NWORLD)
    with conformance.module_loads() as loads:
      _write(r, values)
      for _ in range(3):
        r.render(d)
    assert len(r.debug.gi_rebakes().bakes) == 2  # the first bake, then the relit one
    assert not loads, loads
    r.close()


def _dtype(r: rw.Renderer, key: str) -> type:
  """The dtype of the table `key` writes."""
  field, _, pattern = key.partition(":")
  table = r.per_world.columns(field, pattern).array if pattern else r.per_world[field]
  assert isinstance(table, wp.array)
  return table.dtype


# (b) Legged robot outdoors ----------------------------------------------------------------------

OUTDOOR_WORLD: Final = rw.PerWorld(material=["texture", "tex_variation"], env=["id", "yaw", "intensity"])


def _outdoor(values: Values | None, nworld: int) -> tuple[rw.Renderer, mjw.Data]:
  """The quadruped on open ground under one of SKIES per world (GI off: an open scene)."""
  batched = {k: v for k, v in (values or {}).items() if k.startswith("m.")}
  mjm, m, d = _mujoco(QUADRUPED, nworld, batched)
  lighting = rw.Lighting(envmap=list(SKIES), envmap_intensity=1.0)
  settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
  scene = rw.integrations.mujoco.scene(mjm, m, d, lighting=lighting)
  r = rw.Renderer(scene, nworld=nworld, cameras=["chase"], resolution=RES, settings=settings, per_world=OUTDOOR_WORLD)
  return r, d


def _outdoor_values(r: rw.Renderer, rng: np.random.Generator, nworld: int) -> dict[str, np.ndarray]:
  """Sky, its yaw and intensity, the sun's direction and the ground's texture per world."""
  texture = r.per_world["material.texture"].numpy()
  ground, rock = 0, 1  # material ids in XML order
  swap = rng.integers(0, 2, nworld).astype(bool)
  texture[swap, ground], texture[swap, rock] = texture[swap, rock].copy(), texture[swap, ground].copy()
  elevation = rng.uniform(np.radians(15.0), np.radians(75.0), nworld)
  azimuth = rng.uniform(0.0, 2.0 * np.pi, nworld)
  sun = -np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], -1)
  return {
    "env.id": (np.arange(nworld) % len(SKIES)).astype(np.int32),
    "env.yaw": rng.uniform(-np.pi, np.pi, nworld).astype(np.float32),
    "env.intensity": rng.uniform(0.5, 2.0, nworld).astype(np.float32),
    "material.texture": texture,
    "material.tex_variation": rw.randomize.jitter(
      r.per_world["material.tex_variation"].numpy(),
      rng,
      uv_offset=1.0,
      uv_rotation=np.pi,
    ),
    "m.light_dir": sun.reshape(nworld, 1, 3).astype(np.float32),
  }


def _render_outdoor(values: Values) -> np.ndarray:
  """(nworld, H, W, 3) HDR of the quadruped scene with `values`."""
  nworld = next(iter(values.values())).shape[0]
  r, d = _outdoor(values, nworld)
  with r:
    r.render(d)
    _write(r, values)
    return required(r.render(d).hdr).numpy()[:, 0, ..., :3]


def test_outdoor_sky_sun_and_ground_per_world() -> None:
  """(b) Environment map swap, yaw and intensity, the sun's direction (batched m.light_dir) and the
  ground texture: every world renders its own, as alone."""
  with wp.ScopedDevice("cpu"):
    r, _ = _outdoor(None, NWORLD)
    values = _outdoor_values(r, np.random.default_rng(2), NWORLD)
    r.close()
    _check_worlds_render_alone(_render_outdoor, values, worlds=(1, 2))


# (c) Indoor room (host scene) ---------------------------------------------------------------------


def _room() -> rw.Scene:
  """A room: textured floor and walls, a crate and a ball, two lamps and a window sun."""
  b = rw.SceneBuilder()
  y, x = np.mgrid[0:16, 0:16]
  tiles = np.zeros((16, 16, 4), np.float32)
  tiles[..., :3] = (((x // 4 + y // 4) % 2) * 0.7 + 0.15)[..., None]
  tiles[..., 3] = 1.0
  paper = np.zeros((16, 16, 4), np.float32)
  paper[..., 0], paper[..., 1], paper[..., 2], paper[..., 3] = (x // 2) % 2, 0.3, 1 - (x // 2) % 2, 1.0
  t_tiles, t_paper = b.add_texture(tiles, colorspace="linear"), b.add_texture(paper, colorspace="linear")

  def textured(texture: TextureId, roughness: float) -> MaterialId:
    material = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=roughness)
    return b.add_material(material.with_texture(Slot.BASE_COLOR, TextureRef(texture)))

  floor, walls = textured(t_tiles, 0.7), textured(t_paper, 0.9)
  crate = b.add_material(rw.Material(base_color=(0.8, 0.1, 0.1), specular_roughness=0.4))
  ball = b.add_material(rw.Material(base_color=(0.1, 0.2, 0.8), specular_roughness=0.3, base_metalness=1.0))
  eye = np.eye(3)
  b.add_instance("box", material=floor, size=(2, 2, 0.05), xform=np.c_[eye, [0, 0, -0.05]], path="floor")
  b.add_instance("box", material=walls, size=(2, 0.05, 1.2), xform=np.c_[eye, [0, 2, 1.2]], path="wall_n")
  b.add_instance("box", material=walls, size=(0.05, 2, 1.2), xform=np.c_[eye, [2, 0, 1.2]], path="wall_e")
  b.add_instance("box", material=crate, size=(0.3, 0.3, 0.3), xform=np.c_[eye, [0.3, 0.5, 0.3]], path="crate")
  b.add_instance("sphere", material=ball, size=(0.25, 0.25, 0.25), xform=np.c_[eye, [-0.6, 0.2, 0.25]], path="ball")
  b.add_light(rw.Light.point((0.0, 0.0, 2.2), intensity=80.0, radius=0.05))
  b.add_light(rw.Light.point((1.2, 1.2, 1.5), intensity=40.0, radius=0.05, color=(1.0, 0.8, 0.6)))
  b.add_light(rw.Light.sun((0.3, 0.4, -0.8), illuminance=3.0))
  return b.build()


ROOM_WORLD: Final = rw.PerWorld(
  material=["base_color", "specular_roughness", "texture"], light=["active", "color", "strength", "pos"], env=["exposure"]
)
# Look into the room from the south wall, 1.3 m up, pitched down (x, y, z, then a quaternion).
ROOM_CAMERA: Final = np.array([0.0, -1.9, 1.3, 0.45, 0.0, 0.0, 0.893], np.float32)


def _room_renderer(scene: rw.GpuScene, nworld: int, *, gi: bool) -> rw.Renderer:
  """The room renderer (one free camera)."""
  settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": gi})
  r = rw.Renderer(
    scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=70.0)], resolution=RES, settings=settings, per_world=ROOM_WORLD
  )
  r.camera_pose.assign(np.tile(ROOM_CAMERA, (nworld, 1, 1)))
  return r


def _room_values(r: rw.Renderer, rng: np.random.Generator, nworld: int) -> dict[str, np.ndarray]:
  """Lamps on / off, colour, intensity and position; wall paper swapped and tinted per world."""
  pw = r.per_world
  colors = pw["material.base_color"].numpy()
  colors[:, 1, :3] = rng.uniform(0.2, 1.0, (nworld, 3))  # the walls' tint
  texture = pw["material.texture"].numpy()
  swap = np.arange(nworld) % 2 == 1
  texture[swap, 1], texture[swap, 0] = texture[swap, 0].copy(), texture[swap, 1].copy()
  active = np.ones(pw["light.active"].shape, bool)
  active[:, :2] = rng.integers(0, 2, (nworld, 2)).astype(bool)
  strength = pw["light.strength"].numpy()
  return {
    "material.base_color": colors,
    "material.specular_roughness": rng.uniform(0.1, 1.0, colors.shape[:2]).astype(np.float32),
    "material.texture": texture,
    "light.active": active,
    "light.color": rng.uniform(0.3, 1.0, (*strength.shape, 3)).astype(np.float32),
    "light.strength": (strength * rng.uniform(0.3, 2.0, strength.shape)).astype(np.float32),
    "light.pos": (pw["light.pos"].numpy() + rng.uniform(-0.5, 0.5, (*strength.shape, 3))).astype(np.float32),
    "exposure": (pw["exposure"].numpy() * rng.uniform(0.7, 1.4, nworld)).astype(np.float32),
  }


def test_indoor_room_lamps_and_walls_per_world() -> None:
  """(c) Lamps switched, coloured, dimmed and moved, wall paper swapped and tinted per world (a host
  scene, as USD loads one): every world renders its own, as alone."""
  with wp.ScopedDevice("cpu"):
    scene = rw.upload(_room())
    r = _room_renderer(scene, NWORLD, gi=False)
    values = _room_values(r, np.random.default_rng(3), NWORLD)
    r.close()

    def render(v: Values) -> np.ndarray:
      with _room_renderer(scene, next(iter(v.values())).shape[0], gi=False) as rr:
        rr.render()
        _write(rr, v)
        return required(rr.render().hdr).numpy()[:, 0, ..., :3]

    _check_worlds_render_alone(render, values, worlds=(0, 1))


@pytest.mark.slow  # two GI bakes on the CPU (~40 s)
def test_indoor_room_gi_follows_randomisation_after_the_first_frame() -> None:
  """(c) With GI, worlds randomised after the first frame render (from the next frame on) what
  worlds randomised before it render: no stale probe radiance, light grid or bake."""
  with wp.ScopedDevice("cpu"):
    scene = rw.upload(_room())
    late = _room_renderer(scene, NWORLD, gi=True)
    values = _room_values(late, np.random.default_rng(4), NWORLD)
    late.render()
    _write(late, values)
    late.render()  # a new variant: the re-bake comes a frame later (`shade/gi/groups.py`)
    after = required(late.render().hdr).numpy()[:, 0]
    early = _room_renderer(scene, NWORLD, gi=True)
    _write(early, values)
    before = required(early.render().hdr).numpy()[:, 0]
    for w in range(NWORLD):
      np.testing.assert_allclose(after[w].mean(), before[w].mean(), rtol=0.02, err_msg=f"world {w}")
    late.close()
    early.close()


# (d) Isaac Lab per-environment material writes ---------------------------------------------------


def test_isaaclab_material_writes_every_step_per_environment() -> None:
  """(d) Isaac Lab's visual-material randomisation: every step a new colour per environment
  through the adapter's writer; each environment shows its own, without recompiling."""
  pytest.importorskip("newton")
  from fixtures import isaaclab_fake as fk

  from raywarp.integrations.isaaclab import Cameras

  n, w, h = 4, 16, 12
  model, state = fk.boxes(n)
  assert model.shape_world is not None
  paths = [
    f"/World/envs/env_{e}/Looks/Paint" if "/Box/" in label else ""
    for e, label in zip(model.shape_world.numpy(), model.shape_label, strict=True)
  ]
  cams = Cameras(fk.Config(settings=UNLIT_FAST_CHANGES))
  cams.initialize(fk.Simulation(model, state, materials=paths))
  colors: wp.array[wp.vec3, [int]] = wp.array(np.full((n, 3), 0.5, np.float32), dtype=wp.vec3, device="cpu")
  writer = cams.material_writer([f"/World/envs/env_{e}/Looks/Paint" for e in range(n)], colors)
  writer()
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(w, h, ["albedo"]), n))
  data = fk.CameraData.allocate(["albedo"], n, w, h)
  cams.set_outputs(session, data.output)
  pos, rot = fk.poses([(-3.0, 0.0, 0.5)] * n)
  cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(n, w, h)))
  cams.render(session)
  features = session.renderer.features
  rng = np.random.default_rng(5)
  with conformance.module_loads() as loads:
    for _ in range(5):
      linear = rng.uniform(0.05, 0.95, (n, 3)).astype(np.float32)
      colors.assign(linear)
      writer()
      cams.render(session)
      centre = data.output["albedo"].warp.numpy()[:, h // 2, w // 2, :3]
      np.testing.assert_allclose(centre, linear_to_srgb_u8(linear), atol=1)
  assert not loads, loads
  assert session.renderer.features is features
  cams.close()


# (e) MuJoCo Warp batched model tables -------------------------------------------------------------


def _model_values(m: mjw.Model, mjm: mujoco.MjModel, rng: np.random.Generator, nworld: int) -> dict[str, np.ndarray]:
  """Per-world rows of the model tables MuJoCo Warp randomises (and its renderer reads)."""
  geom_rgba = _batched_rows(m, "geom_rgba", nworld)
  geom_rgba[:, mjm.geom("distractor").id, :3] = rng.uniform(0.1, 1.0, (nworld, 3))
  mat_rgba = _batched_rows(m, "mat_rgba", nworld)
  mat_rgba[:, :, :3] = rng.uniform(0.1, 1.0, mat_rgba[:, :, :3].shape)
  mat_texid = _batched_rows(m, "mat_texid", nworld)
  table, floor = mjm.material("table").id, mjm.material("floor").id
  swap = np.arange(nworld) % 2 == 1
  mat_texid[swap, table], mat_texid[swap, floor] = mat_texid[swap, floor].copy(), mat_texid[swap, table].copy()
  diffuse = rng.uniform(0.2, 1.0, (nworld, mjm.nlight, 3)).astype(np.float32)
  active = np.ones((nworld, mjm.nlight), bool)
  active[:, 0] = np.arange(nworld) % 3 != 2
  return {
    "m.geom_rgba": geom_rgba,
    "m.mat_rgba": mat_rgba,
    "m.mat_texid": mat_texid,
    "m.light_diffuse": diffuse,
    "m.light_active": active,
    "m.cam_fovy": rng.uniform(35.0, 65.0, (nworld, 1)).astype(np.float32),
  }


def _render_model(values: Values) -> np.ndarray:
  """(nworld, H, W, 3) HDR of the tabletop whose model tables are `values` (no PerWorld)."""
  nworld = next(iter(values.values())).shape[0]
  mjm, m, d = _mujoco(TABLETOP, nworld, values)
  settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
  with rw.Renderer(
    rw.integrations.mujoco.scene(mjm, m, d), nworld=nworld, cameras=["cam"], resolution=RES, settings=settings
  ) as r:
    return required(r.render(d).hdr).numpy()[:, 0, ..., :3]


def test_mujoco_warp_batched_model_tables_per_world() -> None:
  """(e) Batched geom / material colours, textures, lights and the camera's field of view: world w
  renders its rows of the model, as alone."""
  with wp.ScopedDevice("cpu"):
    mjm, m, _ = _mujoco(TABLETOP, NWORLD)
    values = _model_values(m, mjm, np.random.default_rng(6), NWORLD)
    _check_worlds_render_alone(_render_model, values, worlds=(0, 1, 2))


def test_mujoco_warp_model_written_every_step_and_k_rows() -> None:
  """(e) The batched tables written in place every step (as mjlab's reset events do) follow without
  a compile or an allocation; a K-row table (1 < K < nworld) is read as world % K."""
  with wp.ScopedDevice("cpu"):
    mjm, m, _ = _mujoco(TABLETOP, NWORLD)
    rng = np.random.default_rng(7)
    draws = [_model_values(m, mjm, rng, NWORLD) for _ in range(3)]
    k_rows = {"m.light_diffuse": draws[0]["m.light_diffuse"][:2]}  # K = 2
    mjm, m, d = _mujoco(TABLETOP, NWORLD, {**draws[0], **k_rows})
    settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
    r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m, d), nworld=NWORLD, cameras=["cam"], resolution=RES, settings=settings)
    sources = {key: [wp.array(v[key], dtype=getattr(m, key[2:]).dtype, device="cpu") for v in draws] for key in draws[0]}
    sources["m.light_diffuse"] = [wp.array(v["m.light_diffuse"][:2], dtype=wp.vec3, device="cpu") for v in draws]
    step = [0]

    def write() -> None:
      k = step[0] % len(draws)
      step[0] += 1
      for key, arrays in sources.items():
        wp.copy(getattr(m, key[2:]), arrays[k])
      mjw.forward(m, d)  # MuJoCo Warp's own temporaries are its business

    loads, allocated, ms = _steady(lambda: r.render(d), write, steps=4)
    print(f"mujoco warp: {len(sources)} model tables written in {ms:.2f} ms per step")
    assert not loads, loads
    assert not allocated, f"steady frames allocated {allocated} bytes"
    # World 2 reads light row 0, world 3 row 1: the same image as a model whose rows say so.
    last = draws[(step[0] - 1) % len(draws)]
    expected = _render_model({**last, "m.light_diffuse": last["m.light_diffuse"][:2][np.arange(NWORLD) % 2]})
    np.testing.assert_array_equal(required(r.render(d).hdr).numpy()[:, 0, ..., :3], expected)
    r.close()


def test_mujoco_warp_camera_field_of_view_per_world() -> None:
  """(e) A batched m.cam_fovy (MuJoCo Warp's camera randomisation) changes each world's view, and
  a per-world table declared over a K-row model table starts from its rows tiled as world % K."""
  with wp.ScopedDevice("cpu"):
    rgba = np.zeros((2, 5, 4), np.float32)  # K = 2 rows of geom colours
    mjm, m, d = _mujoco(TABLETOP, NWORLD, {"m.cam_fovy": np.array([[50.0], [20.0], [50.0], [20.0]], np.float32)})
    rgba[:] = m.geom_rgba.numpy()[:1]
    rgba[1, :, :3] = 0.25
    m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
    settings = recipe("unlit_fast").replace(outputs=rw.Outputs.of("depth"))
    decl = rw.PerWorld(material=["color"])
    with rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d),
      nworld=NWORLD,
      cameras=["cam"],
      resolution=RES,
      settings=settings,
      per_world=decl,
    ) as r:
      depth = required(r.render(d).depth).numpy()[:, 0]
      np.testing.assert_array_equal(depth[0], depth[2])
      assert np.abs(depth[0] - depth[1]).max() > 0.01  # 20 degrees: a narrower view
      np.testing.assert_array_equal(r.per_world["instance.color"].numpy(), rgba[np.arange(NWORLD) % 2])
      # The camera export reads the live fields of view the render used: fy = H / 2 / tan(fovy / 2).
      fy = r.camera_info().k.numpy()[:, 0, 1, 1]
      fovy = np.array([50.0, 20.0, 50.0, 20.0])[:NWORLD]
      np.testing.assert_allclose(fy, RES[1] / 2 / np.tan(np.radians(fovy) / 2), rtol=1e-5)


def _shininess_hdr(shininess: np.ndarray) -> np.ndarray:
  """(nworld, H, W, 3) HDR of the tabletop, `shininess` (nworld, nmat): a batched
  `m.mat_shininess` when nworld > 1, else the model's own (the scene built from it)."""
  mjm = mujoco.MjModel.from_xml_string(TABLETOP)
  nworld = shininess.shape[0]
  if nworld == 1:
    mjm.mat_shininess[:] = shininess[0]
  m = mjw.put_model(mjm)
  if nworld > 1:
    m.mat_shininess = wp.array(shininess, dtype=float)
  d = mjw.make_data(mjm, nworld=nworld)
  mjw.forward(m, d)
  settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
  with warnings.catch_warnings():
    warnings.simplefilter("error")  # read, not warned about
    with rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d), nworld=nworld, cameras=["cam"], resolution=RES, settings=settings
    ) as r:
      r.render(d)
      return required(r.render(d).hdr).numpy()[:, 0, ..., :3]


def test_mujoco_warp_batched_shininess_is_the_roughness_per_world() -> None:
  """(e) A batched `m.mat_shininess` (MuJoCo Warp's per-world randomisation) is read: each world
  renders the roughness 1 - shininess of its row for the materials that leave roughness unset,
  as a scene built with that shininess (BUG-481: it was only warned about)."""
  with wp.ScopedDevice("cpu"):
    nmat = mujoco.MjModel.from_xml_string(TABLETOP).nmat
    rows = np.stack([np.full(nmat, 0.5, np.float32), np.linspace(0.95, 0.05, nmat, dtype=np.float32)])
    batch = _shininess_hdr(rows)
    assert not np.array_equal(batch[0], batch[1])
    for w in range(2):
      np.testing.assert_array_equal(batch[w], _shininess_hdr(rows[w : w + 1])[0], err_msg=f"world {w}")


def test_a_shininess_batched_after_the_renderer_warns() -> None:
  """(e) `m.mat_shininess` batched after the Renderer was created is not read (the roughness slot
  is chosen at creation): warned once, naming the fix."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _mujoco(TABLETOP, 2)
    settings = recipe("unlit_fast").replace(outputs=rw.Outputs.of("depth"))
    with rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d), nworld=2, cameras=["cam"], resolution=(8, 6), settings=settings
    ) as r:
      r.render(d)
      m.mat_shininess = wp.array(_batched_rows(m, "mat_shininess", 2), dtype=float, device="cpu")
      with pytest.warns(UserWarning, match=r"m\.mat_shininess.*after the Renderer"):
        r.render(d)


# (f) Extremes -------------------------------------------------------------------------------------


def _edge_values(r: rw.Renderer, m: mjw.Model, rng: np.random.Generator, nworld: int) -> dict[str, np.ndarray]:
  """`_tabletop_values` with a third of the entries at their range's edges (0, 1, very bright)."""
  values = _tabletop_values(r, m, rng, nworld)
  for key in ("material.specular_roughness", "material.base_metalness"):
    v = values[key]
    edge = rng.random(v.shape) < 1.0 / 3.0
    v[edge] = rng.integers(0, 2, int(edge.sum()))
  colors = values["material.base_color"]
  colors[rng.random(colors.shape[:2]) < 1.0 / 3.0, :3] = 0.0
  values["light.strength"][rng.random(values["light.strength"].shape) < 1.0 / 3.0] *= 1.0e4
  values["exposure"][0] *= 1.0e-4
  return values


@pytest.mark.slow  # 100 frames on the CPU (~15 s)
def test_every_world_every_table_every_step_for_100_steps() -> None:
  """(f) 8 worlds, all different, every table and the camera re-randomised every step for 100
  steps with edge values: finite images, no compile, no steady allocation, and the last step's
  worlds still render as alone."""
  nworld, steps = 8, 100
  with wp.ScopedDevice("cpu"):
    r, m, d = _tabletop(None, nworld)
    rng = np.random.default_rng(8)
    first = _edge_values(r, m, rng, nworld)
    r.close()
    r, m, d = _tabletop(first, nworld)  # the camera's model tables batched per world
    features = r.features
    r.render(d)
    spent = []
    values = first
    with conformance.module_loads() as loads:
      for _ in range(steps):
        values = _edge_values(r, m, rng, nworld)
        t0 = time.perf_counter()
        _write(r, values)
        m.cam_pos.assign(values["m.cam_pos"])
        m.cam_fovy.assign(values["m.cam_fovy"])
        spent.append(time.perf_counter() - t0)
        mjw.forward(m, d)
        hdr = required(r.render(d).hdr).numpy()
        assert np.isfinite(hdr).all() and hdr.min() >= 0.0
    print(f"extremes: write {1e3 * float(np.mean(np.asarray(spent))):.2f} ms per step (host values, uploads included)")
    assert not loads, loads
    assert r.features is features
    final = required(r.render(d).hdr).numpy()[:, 0, ..., :3]
    r.close()
    for w in (0, nworld - 1):
      np.testing.assert_array_equal(final[w], _render_tabletop(_row(values, w))[0], err_msg=f"world {w}")


@pytest.mark.parametrize(
  ("field", "value", "names"),
  [
    ("material.base_metalness", np.nan, "material.base_metalness in [0, 1]"),
    ("material.base_metalness", 2.0, "material.base_metalness in [0, 1]"),
    ("light.strength", np.nan, "light.strength in [0, inf]"),
    ("light.color", -1.0, "light.color in [0, inf]"),
    ("material.base_color", -0.5, "material.base_color in [0, inf]"),
  ],
)
def test_invalid_per_world_values_raise_at_the_next_render(field: str, value: float, names: str) -> None:
  """(f) A NaN, negative or out-of-range value in a declared table (which would render NaN or
  negative radiance) raises at the next render, naming the field and its range."""
  with wp.ScopedDevice("cpu"):
    r, _, d = _tabletop(None, 2)
    r.render(d)
    table = r.per_world[field]
    v = table.numpy()
    v[1] = value
    table.assign(v)
    r.render(d)  # checked on the device, read a frame later
    with pytest.raises(ValueError, match=re.escape(names)):
      r.render(d)
    r.close()


@pytest.mark.parametrize(("field", "value"), [("env.intensity", np.nan), ("env.yaw", np.inf), ("camera.fovy", 0.0)])
def test_invalid_environment_and_camera_values_raise(field: str, value: float) -> None:
  """(f) The environment's and the camera's per-world tables are checked too."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _mujoco(QUADRUPED, 2)
    group, _, name = field.partition(".")
    per_world = rw.PerWorld(env=[name]) if group == "env" else rw.PerWorld(camera=[name])
    scene = rw.integrations.mujoco.scene(mjm, m, d, lighting=rw.Lighting(envmap=SKIES[0], envmap_intensity=1.0))
    settings = recipe("fast").replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
    r = rw.Renderer(scene, nworld=2, cameras=["chase"], resolution=(8, 6), settings=settings, per_world=per_world)
    r.render(d)
    v = r.per_world[field].numpy()
    v[0] = value
    r.per_world[field].assign(v)
    r.render(d)
    with pytest.raises(ValueError, match=re.escape(field)):
      r.render(d)
    r.close()


# Glass in or out per world --------------------------------------------------------------------


def _glass_scene(box_glass: bool) -> rw.Scene:
  """A floor, a box (opaque or glass), a glass sphere beside it (both materials in use), a sun
  casting shadows through the glass. The glass's volume depth is authored: without one it is
  derived from the authored scene's instances of the material (`pipeline/upload/optics.py`), which a
  per-world choice does not redefine."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_roughness=0.8))
  opaque = b.add_material(rw.Material(base_color=(0.8, 0.2, 0.1), specular_roughness=0.5))
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      transmission_weight=1.0,
      transmission_color=(0.6, 0.9, 0.6),
      specular_ior=1.5,
      specular_roughness=0.0,
      extras=rw.scene.OpenPbrExtras(transmission_depth=0.35),
    )
  )
  eye = np.eye(3)
  b.add_instance("box", material=floor, size=(2.0, 2.0, 0.05), xform=np.c_[eye, [0.0, 0.0, -0.05]], path="floor")
  b.add_instance(
    "box", material=glass if box_glass else opaque, size=(0.3, 0.3, 0.3), xform=np.c_[eye, [-0.4, 0.0, 0.3]], path="box"
  )
  b.add_instance("sphere", material=glass, size=(0.25, 0.25, 0.25), xform=np.c_[eye, [0.5, 0.0, 0.25]], path="ball")
  b.add_instance("box", material=opaque, size=(0.1, 0.1, 0.1), xform=np.c_[eye, [0.5, 0.6, 0.1]], path="cube")
  b.add_light(rw.Light.sun((0.4, 0.2, -0.9), illuminance=5.0))
  b.add_environment(rw.scene.EnvironmentLight(intensity=0.5))
  return b.build()


def _glass_renderer(box_glass: bool, nworld: int, recipe_name: str) -> rw.Renderer:
  """The glass scene seen from above at an angle, materials selectable per world."""
  settings = recipe(recipe_name).replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
  r = rw.Renderer(
    rw.upload(_glass_scene(box_glass)),
    nworld=nworld,
    cameras=[rw.CameraSpec(fovy=60.0)],
    resolution=RES,
    settings=settings,
    per_world=rw.PerWorld(material=["instance.material", "specular_ior"]),
  )
  r.camera_pose.assign(np.tile(np.array([0.0, -2.0, 1.6, 0.383, 0.0, 0.0, 0.924], np.float32), (nworld, 1, 1)))
  return r


@pytest.mark.parametrize(
  ("recipe_name", "to_glass"),
  [("fast", True), ("default", True), ("default", False), ("fast", False)],
)
def test_material_switched_into_or_out_of_glass_per_world(recipe_name: str, to_glass: bool) -> None:
  """A world whose box is switched opaque -> glass (or back) after the first frame renders what a
  scene authored that way renders: no stale transmission, IOR or shadow-transmission data, and the
  other world is untouched."""
  with wp.ScopedDevice("cpu"):
    r = _glass_renderer(not to_glass, 2, recipe_name)
    before = required(r.render().hdr).numpy()[:, 0].copy()
    table = r.per_world["instance.material"]
    ids = table.numpy()
    box = 1  # instance order: floor, box, ball, cube; materials: floor 0, opaque 1, glass 2
    ids[1, box] = 2 if to_glass else 1
    table.assign(ids)
    r.render()
    after = required(r.render().hdr).numpy()[:, 0]
    r.close()
    with _glass_renderer(to_glass, 1, recipe_name) as fresh:
      expected = required(fresh.render().hdr).numpy()[0, 0]
    np.testing.assert_array_equal(after[0], before[0])
    assert np.abs(expected - before[1]).max() > 1e-3  # the switch is visible
    np.testing.assert_allclose(after[1], expected, rtol=1e-4, atol=1e-4)
