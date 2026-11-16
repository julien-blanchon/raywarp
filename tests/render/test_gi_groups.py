"""Per-world GI groups (`core/shade/gi/groups.py`) and the relit volume of continuous
randomisation (`relight.py`) on a red / green Cornell-style room.

A white room, a red wall on -x and a green wall on +x, a ceiling light. The floor's indirect light
is tinted by the walls: redder near the red wall. Worlds randomise the wall colours, the light and
a moved box; with K groups each world's GI equals the GI of a renderer of that world alone (no
leak between worlds), identical worlds collapse to one volume (byte-identical to an unrandomised
renderer), and a world reset to other inputs follows them. A relit volume approximates each
world's GI (its tint follows the world's walls: the bleed flips sides with them) and is exact for
world 0, whose inputs it was baked with.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.shade.gi import groups
from raywarp.core.shade.gi import groups as gi_groups
from raywarp.core.shade.gi.rebakes import Reason
from raywarp.scene.model.types import MaterialId

RED: tuple[float, float, float] = (0.9, 0.05, 0.05)


GREEN: tuple[float, float, float] = (0.05, 0.9, 0.05)


WHITE: tuple[float, float, float] = (0.8, 0.8, 0.8)


_HALF = 1.0  # room half size [m]


# Material rows of the room (SceneBuilder order).
_LEFT, _RIGHT = 1, 2


def _room(left: tuple[float, float, float] = RED, right: tuple[float, float, float] = GREEN, *, lamp: bool = False) -> rw.Scene:
  """The room: white floor / ceiling / back wall, `left` wall on -x, `right` on +x, a box and a
  light; `lamp`: a second light (index 1) low in the room."""
  b = rw.SceneBuilder()
  white = b.add_material(rw.Material(base_color=WHITE, specular_roughness=1.0))
  lm = b.add_material(rw.Material(base_color=left, specular_roughness=1.0))
  rm = b.add_material(rw.Material(base_color=right, specular_roughness=1.0))
  h, t = _HALF, 0.05

  def wall(material: MaterialId, size: tuple[float, float, float], at: tuple[float, float, float], path: str) -> None:
    """A box of half sizes `size` centred at `at`."""
    b.add_instance("box", material=material, size=size, xform=np.c_[np.eye(3), at], path=path)

  wall(white, (h, h, t), (0.0, 0.0, -t), "floor")
  wall(white, (h, h, t), (0.0, 0.0, 2 * h + t), "ceiling")
  wall(white, (h, t, h), (0.0, h + t, h), "back")
  wall(lm, (t, h, h), (-h - t, 0.0, h), "left")
  wall(rm, (t, h, h), (h + t, 0.0, h), "right")
  wall(white, (0.15, 0.15, 0.15), (0.3, 0.2, 0.15), "box")
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=20.0, radius=0.05, xform=np.c_[np.eye(3), [0.0, 0.0, 1.8]]))
  if lamp:
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=40.0, radius=0.05, xform=np.c_[np.eye(3), [-0.5, 0.3, 0.6]]))
  return b.build()


def _settings() -> rw.RenderSettings:
  """HDR, software traversal, a coarse fast probe bake, no stochastic effects."""
  return rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "performance.traversal": "software",
      "gi.enabled": True,
      "gi.spacing": 0.5,
      "gi.rays": 32,
      "gi.bake_updates": 4,
      "reflections.enabled": False,
      "lighting.light_grid": False,
      "shadows.samples": 1,
    }
  )


def _renderer(nworld: int, scene: rw.Scene | None = None, per_world: rw.PerWorld | None = None) -> rw.Renderer[None]:
  """The room seen from the front (looking +y at the back wall and the floor)."""
  r = rw.Renderer(
    rw.upload(scene if scene is not None else _room(), device="cpu"),
    nworld=nworld,
    cameras=[rw.CameraSpec(fovy=80.0)],
    resolution=(16, 16),
    settings=_settings(),
    per_world=per_world or rw.PerWorld(),
  )
  # Camera at the front, looking along +y (rotation of -z onto +y), slightly down.
  rot = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
  rot[:] = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))  # x by +90 deg: -z -> +y
  pose = np.concatenate([[0.0, -0.95, 0.9], rot]).astype(np.float32)
  r.camera_pose.assign(np.tile(pose, (nworld, 1, 1)))
  return r


def _walls(r: rw.Renderer[None], colors: Sequence[tuple[tuple[float, ...], tuple[float, ...]]]) -> None:
  """World w's (left, right) wall colours."""
  table = r.per_world["material.base_color"]
  values = table.numpy()
  for w, (left, right) in enumerate(colors):
    values[w, _LEFT, :3] = left
    values[w, _RIGHT, :3] = right
  table.assign(values)


def _hdr(r: rw.Renderer[None]) -> np.ndarray:
  """(nworld, H, W, 3) HDR of a render."""
  out = r.render().hdr
  assert out is not None
  return out.numpy().reshape(r.nworld, 16, 16, -1)[..., :3].astype(np.float64)


def _tint(image: np.ndarray) -> float:
  """(R / G of the floor's left half) / (R / G of its right half): > 1 when red bleeds from the
  left wall."""
  floor = image[12:]
  left, right = floor[:, :8].mean((0, 1)), floor[:, 8:].mean((0, 1))
  return float(left[0] / left[1] / (right[0] / right[1]))


def _alone(left: tuple[float, float, float], right: tuple[float, float, float]) -> np.ndarray:
  """The HDR of a one-world renderer of the room with these walls."""
  with _renderer(1, _room(left, right)) as r:
    return _hdr(r)[0]


def test_classify_orders_groups_by_first_world() -> None:
  """Equal digests share a group; groups are numbered by their lowest world."""
  d = np.array([[5, 1], [3, 3], [5, 1], [7, 0], [3, 3]], np.uint32)
  world_group, digests, first = gi_groups.classify(d)
  np.testing.assert_array_equal(world_group, [0, 1, 0, 2, 1])
  np.testing.assert_array_equal(digests, d[[0, 1, 3]])
  np.testing.assert_array_equal(first, [0, 1, 3])


def test_identical_worlds_collapse_to_one_group_byte_identical() -> None:
  """Declared per-world walls with equal rows: one group, the unrandomised renderer's image."""
  with _renderer(3) as plain:
    reference = _hdr(plain)
    assert plain.debug.gi_volume is not None and not plain.debug.gi_volume.groups.tracked
  with _renderer(3, per_world=rw.PerWorld(material=["base_color"])) as r:
    image = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.groups.tracked and vol.ngroup == 1
  np.testing.assert_array_equal(image, reference)


def _error(image: np.ndarray, reference: np.ndarray) -> float:
  """Mean relative error of `image` against `reference` (luminance)."""
  lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
  return float(np.mean(np.abs(lum(image) - lum(reference))) / np.mean(lum(reference)))


def test_each_world_gets_its_own_bounce() -> None:
  """K groups (two variants over four worlds): each world's image is the one-world render of its
  walls (no leak)."""
  a, b = (RED, GREEN), (GREEN, RED)
  colors = [a, b, a, b]
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, colors)
    image = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.relight is None and vol.ngroup == 2
  for w, (left, right) in enumerate(colors):
    np.testing.assert_allclose(image[w], _alone(left, right), rtol=1e-4, atol=1e-6)
  # The physics: red bleeds onto the floor's left half in world 0, green in world 1.
  assert _tint(image[0]) > 1.0 > _tint(image[1])


def test_relit_worlds_follow_their_walls() -> None:
  """Mostly distinct worlds: one shared volume (world 0's walls) tinted per world. World 0 renders
  as alone; the others approximate their one-world render (the swapped walls' bleed flips sides),
  much closer than the shared volume untinted."""
  a, b, c = (RED, GREEN), (GREEN, RED), (WHITE, WHITE)
  colors = [a, b, c, a]
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, colors)
    image = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.relight is not None and vol.ngroup == 1
    vol.relight.texels.zero_()  # the shared volume untinted
    vol.relight.field.texture.copy_from(vol.relight.texels)
    r.render()  # (no input changed: the ratios stay as set)
    untinted = _hdr(r)
  np.testing.assert_allclose(image[0], _alone(*a), rtol=1e-4, atol=1e-6)
  np.testing.assert_array_equal(image[3], image[0])
  for w in (1, 2):
    alone = _alone(*colors[w])
    assert _error(image[w], alone) < 0.5 * _error(untinted[w], alone), w
  assert _tint(image[0]) > 1.0 > _tint(image[1])


def test_light_and_box_inputs() -> None:
  """Per-world light colour splits the worlds' GI (two distinct worlds: a relit volume, world 1
  tinted warm); a per-world box pose (hidden from the probes) does not."""
  per_world = rw.PerWorld(light=["color"], pose=["box"])
  with _renderer(2, per_world=per_world) as r:
    pos = r.per_world.columns("instance.pos", "box").numpy()
    pos[1, 0] += np.array([-0.4, 0.0, 0.0])
    r.per_world.columns("instance.pos", "box").assign(pos)
    _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.groups.tracked and vol.ngroup == 1
  with _renderer(2, per_world=per_world) as r:
    color = r.per_world["light.color"].numpy()
    color[1] = (1.0, 0.4, 0.2)
    r.per_world["light.color"].assign(color)
    image = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.relight is not None  # both worlds distinct: relit
  warm = image[1].mean((0, 1))
  assert warm[0] / warm[2] > 2.0 * image[0].mean((0, 1))[0] / image[0].mean((0, 1))[2]


def test_relit_tint_ignores_light_motion_below_the_probe_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
  """A light that follows a body moves a little every step: below half a probe spacing the relit
  tint is not re-traced (`groups.POSITION_QUANTUM`); a move across a cell is."""
  from raywarp.core.shade.gi.relight import Relight

  changed: list[int] = []
  original = Relight.track

  def track(self: Relight, frame: object, dynamic: object) -> None:
    original(self, frame, dynamic)  # pyrefly: ignore[bad-argument-type]  # the frame and mask as passed
    changed.append(int(self.changed.numpy().sum()))

  monkeypatch.setattr(Relight, "track", track)
  with _renderer(2, per_world=rw.PerWorld(light=["pos", "color"])) as r:
    color = r.per_world["light.color"].numpy()
    color[1] = (1.0, 0.4, 0.2)
    r.per_world["light.color"].assign(color)
    _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.relight is not None
    pos = r.per_world["light.pos"]
    start = pos.numpy().copy()
    quantum = vol.relight.quantum
    # From the middle of the light's quantisation cell: no boundary within the small moves.
    centre = (np.floor(start / quantum) + 0.5) * quantum
    pos.assign(centre.astype(np.float32))
    _hdr(r)
    changed.clear()
    for k in range(1, 4):
      pos.assign((centre + 0.02 * quantum * k).astype(np.float32))
      _hdr(r)
    small = list(changed)
    pos.assign((centre + quantum).astype(np.float32))
    _hdr(r)
    assert r.debug.gi_volume is vol
  assert small == [0, 0, 0], small
  assert changed[-1] == 2, changed


def test_one_world_light_written_in_place_re_bakes() -> None:
  """A per-world light table of one world has one row, like the shared tables: written in place,
  its content (not only its identity) is checked against the bake's, and the volume re-bakes to
  a fresh renderer's with the new light (diag gi_light_change)."""
  per_world = rw.PerWorld(light=["strength"])
  with _renderer(1, per_world=per_world) as r:
    _hdr(r)
    vol = r.debug.gi_volume
    strength = r.per_world["light.strength"]
    doubled = strength.numpy() * 2.0
    strength.assign(doubled)
    _hdr(r)  # the change is read a frame later (pinned, no synchronisation)
    after = _hdr(r)
    assert r.debug.gi_volume is not vol
  with _renderer(1, per_world=per_world) as fresh:
    fresh.per_world["light.strength"].assign(doubled)
    np.testing.assert_allclose(after, _hdr(fresh), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pattern", ["groups", "relit"])
def test_reset_follows_the_new_inputs(pattern: str) -> None:
  """A world reset to other walls: its GI follows them in the reset frame, other worlds keep
  theirs; reset back, it renders as before (no state carried). Grouped volumes switch to a known
  variant; a relit world recomputes its tint."""
  a, b, c = (RED, GREEN), (GREEN, RED), (WHITE, WHITE)
  start = [a, b, a, b] if pattern == "groups" else [a, b, c, a]
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, start)
    first = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and (vol.relight is not None) == (pattern == "relit")
    after = [b, b, a, b] if pattern == "groups" else [b, b, c, a]
    _walls(r, after)
    frames = [_hdr(r) for _ in range(3)]
    assert r.debug.gi_volume is vol  # no re-bake: a known variant / a relit world
    _walls(r, start)
    back = _hdr(r)
  np.testing.assert_array_equal(frames[0], frames[-1])
  # The bleed flips sides like the walls (red on the right now), as in the one-world render.
  assert _tint(first[0]) > 1.0 > _tint(frames[0][0])
  assert _tint(_alone(*b)) < 1.0
  for w in range(1, 4):
    np.testing.assert_array_equal(frames[0][w], first[w])
  np.testing.assert_array_equal(back, first)


def test_new_variant_rebakes_grouped_volumes() -> None:
  """Grouped volumes meeting a variant no group holds re-classify (a frame later), once: the next
  frame keeps the re-baked volume (the overflow used to store an empty key: a second bake)."""
  a, b = (RED, GREEN), (GREEN, RED)
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, [a, b, a, b])
    _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.ngroup == 2 and vol.relight is None
    _walls(r, [a, b, (WHITE, WHITE), b])
    _hdr(r)  # tracked: overflow raised
    image = _hdr(r)  # re-classified: three variants over four worlds -> a relit volume
    rebaked = r.debug.gi_volume
    assert rebaked is not vol and rebaked is not None and rebaked.relight is not None
    _hdr(r)
    assert r.debug.gi_volume is rebaked
  np.testing.assert_allclose(image[0], _alone(a[0], a[1]), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pattern", ["one", "groups"])
def test_frozen_shared_groups_do_no_probe_work_after_the_bake(monkeypatch: pytest.MonkeyPatch, pattern: str) -> None:
  """Tracked worlds over one or K baked groups, frozen bake: later frames only track the groups (a
  reset to a baked variant moves the world, nothing warms up), no probe update runs; the image
  is unchanged."""
  from raywarp.core.shade.gi.volume import ProbeVolume

  a, b = (RED, GREEN), (GREEN, RED)
  colors = [a, a, a, a] if pattern == "one" else [a, b, a, b]
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, colors)
    first = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.groups.tracked and vol.relight is None

    def no_update(*_args: object, **_kw: object) -> None:
      raise AssertionError("a probe update ran on a frozen shared bake")

    monkeypatch.setattr(ProbeVolume, "update", no_update)
    np.testing.assert_array_equal(_hdr(r), first)
    if pattern == "groups":
      _walls(r, [b, a, a, b])  # worlds 0 and 1 reset to each other's baked variant
      image = _hdr(r)
      np.testing.assert_array_equal(image[0], first[1])
      np.testing.assert_array_equal(image[1], first[0])
      np.testing.assert_array_equal(image[2:], first[2:])


def test_inactive_light_motion_keeps_bake_but_activation_and_reset_follow_world() -> None:
  """Disabled light motion is irrelevant; enabling it refreshes GI only where it contributes."""
  with _renderer(4, per_world=rw.PerWorld(light=["pos", "active"])) as r:
    active = r.per_world["light.active"]
    active.assign(np.zeros(active.shape, dtype=bool))
    first = _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.ngroup == 1
    position = r.per_world["light.pos"]
    moved = position.numpy().copy()
    moved[1, 0, 0] += 0.25
    position.assign(moved)
    for _ in range(3):
      np.testing.assert_array_equal(_hdr(r), first)
      assert r.debug.gi_volume is vol
    enabled = np.zeros(active.shape, dtype=bool)
    enabled[1, 0] = True
    active.assign(enabled)
    _hdr(r)
    lit = _hdr(r)
    active_volume = r.debug.gi_volume
    assert active_volume is not vol
    assert lit[1].mean() > first[1].mean()
    np.testing.assert_array_equal(lit[[0, 2, 3]], first[[0, 2, 3]])
    active.assign(np.zeros(active.shape, dtype=bool))
    for _ in range(2):
      np.testing.assert_array_equal(_hdr(r), first)


# ---------------------------------------------------------------------------------------------
# Radiance-input hashes cover logical bytes and preserve world/reset isolation.


def test_byte_view_excludes_reserved_capacity() -> None:
  """Reserved storage beyond a contiguous table must not change logical row strides."""
  owner = wp.array(np.arange(64).astype(np.float32), dtype=float, device="cpu")
  table = wp.array(ptr=owner.ptr, shape=(2, 3), dtype=float, capacity=owner.capacity, device="cpu", copy=False)
  view = groups._bytes(table, 2)
  assert view.shape == (2, 12)
  np.testing.assert_array_equal(view.numpy(), np.arange(6).astype(np.float32).view(np.uint8).reshape(2, 12))


def test_relit_update_reads_the_authored_maps_untinted() -> None:
  """An update of a relit volume's shared maps (`Renderer.update_gi`, periodic updates) traces
  the authored inputs (world 0's at the first bake) untinted, then recomputes the reference and
  every tint against the new maps: a world at the authored inputs has tint exactly 1, whatever
  world 0 holds now (no state of earlier episodes)."""
  a, b, c = (RED, GREEN), (GREEN, RED), (WHITE, WHITE)
  with _renderer(4, per_world=rw.PerWorld(material=["base_color"])) as r:
    _walls(r, [a, b, c, a])
    _hdr(r)
    vol = r.debug.gi_volume
    assert vol is not None and vol.relight is not None
    _walls(r, [b, b, c, a])  # world 0 reset away from the authored walls
    _hdr(r)
    relight = vol.relight
    field = relight.field
    ny, nz = field.counts[1], field.counts[2]

    def tint(w: int) -> np.ndarray:
      """World w's six slabs of ratio - 1."""
      t = relight.texels.numpy()
      out = []
      for s in range(6 * w, 6 * w + 6):
        y, z = (s // field.tiles) * ny, (s % field.tiles) * nz
        out.append(t[z : z + nz, y : y + ny, :, :3].astype(np.float32))
      return np.stack(out)

    assert np.abs(tint(0)).max() > 0.0  # tinted: away from the authored walls
    np.testing.assert_array_equal(tint(3), 0.0)  # the authored walls
    r.update_gi()
    image = _hdr(r)
    assert r.debug.gi_volume is vol
    np.testing.assert_array_equal(tint(3), 0.0)
    assert np.abs(tint(0)).max() > 0.0
    np.testing.assert_array_equal(image[1], image[0])  # equal inputs, equal images


def test_relit_worlds_depend_on_their_own_inputs_only() -> None:
  """BUG-E15 / I16: a relit world's image is a function of its own inputs: the shared volume is
  baked with the scene's authored inputs, whatever world 0 holds or the renderer met before (a
  warm renderer after re-randomisations equals a fresh one in the same state; changing world 0
  leaves the others unchanged)."""
  a, b, c = (RED, GREEN), (GREEN, RED), (WHITE, WHITE)
  decl = rw.PerWorld(material=["base_color"])
  with _renderer(4, per_world=decl) as r:
    _hdr(r)
    for colors in ([a, b, c, a], [c, a, b, b], [b, c, a, c]):
      _walls(r, colors)
      _hdr(r)
    _walls(r, [b, a, c, a])
    _hdr(r)
    warm = _hdr(r)
  with _renderer(4, per_world=decl) as r:
    _walls(r, [b, a, c, a])
    _hdr(r)
    fresh = _hdr(r)
  with _renderer(4, per_world=decl) as r:
    _walls(r, [(RED, WHITE), a, c, a])  # world 0 changed, the worlds still mostly distinct (relit)
    _hdr(r)
    other = _hdr(r)
  # Float rounding only (the warm renderer re-baked through another path: 1e-6 relative).
  np.testing.assert_allclose(warm, fresh, rtol=1e-4, atol=1e-7)
  np.testing.assert_allclose(other[1:], fresh[1:], rtol=1e-4, atol=1e-7)


def _move_lamp_and_box(r: rw.Renderer[None], step: int) -> None:
  """World w's lamp (light 1) moved and recoloured, its box moved: a robot step."""
  pos = r.per_world["light.pos"].numpy()
  pos[:, 1] += np.array([0.3 * step, -0.2, 0.1], np.float32)
  r.per_world["light.pos"].assign(pos)
  color = r.per_world["light.color"].numpy()
  color[:, 1] = (1.0, 0.5 + 0.1 * step, 0.3)
  r.per_world["light.color"].assign(color)
  box = r.per_world.columns("instance.pos", "box").numpy()
  box[:, 0] += np.array([0.05 * step, 0.0, 0.0], np.float32)
  r.per_world.columns("instance.pos", "box").assign(box)


def test_dynamic_lights_and_moving_instances_never_re_bake() -> None:
  """The robot scenario's GI contract: a dynamic light (`PerWorld.dynamic_lights`) moving and
  changing colour every frame, and a moving instance, add nothing to the re-bake log (one bake,
  no per-world re-light); the probes never see the dynamic light (the maps of a renderer whose
  lamp sits elsewhere are identical). The same lamp undeclared re-lights the worlds it moves in."""
  per_world = rw.PerWorld(light=["pos", "color"], pose=["box"], dynamic_lights=[1])
  with _renderer(2, scene=_room(lamp=True), per_world=per_world) as r:
    _hdr(r)
    for step in range(1, 4):
      _move_lamp_and_box(r, step)
      _hdr(r)
    log = r.debug.gi_rebakes()
    vol = r.debug.gi_volume
    assert vol is not None
    maps = vol.struct.irradiance.numpy()
  assert len(log.bakes) == 1 and log.bakes[0].startswith(str(Reason.FIRST)), log
  assert log.relights.tolist() == [0, 0] and log.updates == 0
  with _renderer(2, scene=_room(lamp=True), per_world=per_world) as moved:
    _move_lamp_and_box(moved, 3)
    _hdr(moved)
    other = moved.debug.gi_volume
    assert other is not None
    np.testing.assert_array_equal(other.struct.irradiance.numpy(), maps)
  undeclared = rw.PerWorld(light=["pos", "color"], pose=["box"])
  with _renderer(2, scene=_room(lamp=True), per_world=undeclared) as r:
    _hdr(r)
    for step in range(1, 3):
      _move_lamp_and_box(r, step)
      _hdr(r)
    log = r.debug.gi_rebakes()
  assert len(log.bakes) > 1 or log.relights.sum() > 0, log


def test_moving_light_not_dynamic_warns(monkeypatch: pytest.MonkeyPatch) -> None:
  """A light the simulator moves (`Binding.moving_lights`) that is not dynamic warns once at the
  bake; declared dynamic, it does not."""
  from raywarp.core.pipeline.gpu_scene import binding

  monkeypatch.setattr(binding.StaticPoses, "moving_lights", lambda _self: (0,))
  with _renderer(1) as r, pytest.warns(RuntimeWarning, match=r"lights \[0\] move with the simulation"):
    _hdr(r)
  with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    with _renderer(1, per_world=rw.PerWorld(dynamic_lights=[0])) as r:
      _hdr(r)


def _lamps(r: rw.Renderer[None], colors: np.ndarray, strengths: np.ndarray) -> None:
  """World w's light colours (nworld, 2, 3) and strengths (nworld, 2)."""
  color = r.per_world["light.color"].numpy()
  color[:] = colors
  r.per_world["light.color"].assign(color)
  strength = r.per_world["light.strength"].numpy()
  strength[:] = strengths
  r.per_world["light.strength"].assign(strength)


def _indirect(r: rw.Renderer[None]) -> np.ndarray:
  """(nworld, H, W, 3) indirect radiance (GI, environment, ambient) of a render."""
  out = r.render().indirect
  assert out is not None
  return out.numpy().reshape(r.nworld, 16, 16, -1)[..., :3].astype(np.float64)


def test_light_colour_and_intensity_relight_without_rays() -> None:
  """Linear re-light (`shade.gi.slots`): worlds that change only their lights' colours, intensities
  or switch one off are re-tinted from the baked per-light shares with no rays (no re-light in the
  log), and their indirect light matches a full re-bake of each world alone (a one-world renderer
  baked with its own lights) much closer than the authored lights' (no tint)."""
  per_world = rw.PerWorld(light=["color", "strength"])
  outputs = rw.Outputs.of("hdr", "indirect")
  with _renderer(3, scene=_room(lamp=True), per_world=per_world) as r:
    r.update_settings(outputs=outputs)
    base_c, base_s = r.per_world["light.color"].numpy().copy(), r.per_world["light.strength"].numpy().copy()
    colors, strengths = base_c.copy(), base_s.copy()
    colors[1, 1] = (1.0, 0.25, 0.1)  # world 1: a red lamp, the ceiling light dimmed
    strengths[1, 0] *= 0.4
    strengths[2, 1] = 0.0  # world 2: the lamp off, the ceiling light doubled
    strengths[2, 0] *= 2.0
    warm = colors.copy()
    warm[1, 1] = (1.0, 0.8, 0.6)
    _lamps(r, warm, strengths)
    _indirect(r)  # the bake: distinct worlds, a relit volume
    vol = r.debug.gi_volume
    _lamps(r, colors, strengths)  # world 1's lamp turns red: recombined, no rays
    image = _indirect(r)
    authored = image[0]  # world 0 keeps the authored lights: the untinted shared volume
    log = r.debug.gi_rebakes()
    assert vol is not None and r.debug.gi_volume is vol and vol.relight is not None
  assert log.relights.tolist() == [0, 0, 0], log
  errors = {}
  for w in (1, 2):
    with _renderer(1, scene=_room(lamp=True), per_world=per_world) as alone:
      alone.update_settings(outputs=outputs)
      _lamps(alone, colors[w : w + 1], strengths[w : w + 1])
      ref = _indirect(alone)[0]
    errors[w] = (round(_error(image[w], ref), 4), round(_error(authored, ref), 4))
  print({"indirect error vs re-bake (linear, untinted)": errors})
  for linear, plain in errors.values():
    assert linear < 0.5 * plain, errors


def test_a_light_off_at_the_bake_switched_on_relights_by_rays() -> None:
  """A light off in the authored frame has no baked share to scale: a world that switches it on
  is re-lit by rays (one re-light in the log), and its indirect light matches a re-bake of the
  world alone; the world that keeps it off stays recombined."""
  per_world = rw.PerWorld(light=["color", "strength"])
  outputs = rw.Outputs.of("hdr", "indirect")
  with _renderer(2, scene=_room(lamp=True), per_world=per_world) as r:
    r.update_settings(outputs=outputs)
    colors, on = r.per_world["light.color"].numpy().copy(), r.per_world["light.strength"].numpy().copy()
    off = on.copy()
    off[:, 1] = 0.0  # the lamp off in every world at the bake
    off[1, 0] *= 0.5  # world 1's ceiling light dimmed: distinct worlds, a relit volume
    _lamps(r, colors, off)
    _indirect(r)
    lit = off.copy()
    lit[1, 1] = on[1, 1]  # world 1 switches the lamp on
    _lamps(r, colors, lit)
    image = _indirect(r)
    log = r.debug.gi_rebakes()
    assert r.debug.gi_volume is not None and r.debug.gi_volume.relight is not None
  assert log.relights.tolist() == [0, 1], log
  with _renderer(1, scene=_room(lamp=True), per_world=per_world) as alone:
    alone.update_settings(outputs=outputs)
    _lamps(alone, colors[1:], lit[1:])
    ref = _indirect(alone)[0]
  relit, untinted = _error(image[1], ref), _error(image[0], ref)
  print({"indirect error vs re-bake (relit, lamp-off world)": (round(relit, 4), round(untinted, 4))})
  assert relit < 0.5 * untinted, (relit, untinted)


def test_an_implicit_re_bake_makes_captured_graphs_stale() -> None:
  """A re-bake an eager frame triggers by itself (here a one-world light written in place, which
  keeps captured graphs valid) replaces the volume a graph's kernels read: the graph version must
  change, or a replay reads the released volume (no StaleBinding was raised)."""
  with _renderer(1, per_world=rw.PerWorld(light=["strength"])) as r:
    _hdr(r)
    vol, version = r.debug.gi_volume, r.debug.graph_version()
    strength = r.per_world["light.strength"]
    strength.assign(strength.numpy() * 2.0)
    _hdr(r)
    assert r.debug.graph_version() == version  # an in-place write: the graph stays valid
    _hdr(r)  # the re-bake
    assert r.debug.gi_volume is not vol
    assert r.debug.graph_version() != version


def test_shared_digest_ignores_world_0_light_activity() -> None:
  """Regression (BUG-489): the shared inputs' digest (one row) masked its light properties by
  world 0's row of a per-world activity table, so a reset toggling one of world 0's lights changed
  it and re-baked the shared volume every other step; a shared activity table still masks."""
  import types

  nlight, nworld = 2, 4
  cutoff = wp.array([[60.0, 45.0]], dtype=float)
  skips = groups.Skips(wp.zeros(0, dtype=wp.int32), wp.zeros(nlight, dtype=wp.int32), wp.zeros(0, dtype=wp.int32))

  def shared(active: np.ndarray) -> tuple[int, ...]:
    """The shared digest of the light cutoffs under activity `active`."""
    frame = types.SimpleNamespace(light_cutoff=cutoff, light_active=wp.array(active, dtype=bool))
    out = wp.zeros((1, 2), dtype=wp.uint32)
    groups.shared_digest(frame, frozenset({"light_cutoff"}), skips, out)  # pyrefly: ignore[bad-argument-type]
    return tuple(out.numpy().ravel().tolist())

  on = np.ones((nworld, nlight), bool)
  toggled = on.copy()
  toggled[0, 0] = False
  assert shared(on) == shared(toggled)
  # A shared activity table (one row) masks the properties of its inactive lights.
  assert shared(np.array([[True, True]])) != shared(np.array([[False, True]]))


def _exposed(r: rw.Renderer[None], scale: Sequence[float]) -> None:
  """World w's exposure: the renderer's times `scale[w]`."""
  table = r.per_world["exposure"]
  table.assign(table.numpy() * np.asarray(scale, np.float32))


def test_exposure_is_no_gi_input() -> None:
  """A per-world exposure scales the world's image after its GI (the probes light display-relative
  emission at the renderer's exposure): randomising it re-lights nothing, and the HDR (radiance,
  before exposure) stays the same."""
  per_world = rw.PerWorld(material=["base_color"], env=["exposure"])
  with _renderer(3, per_world=per_world) as r:
    _walls(r, [(RED, GREEN), (GREEN, RED), (WHITE, RED)])  # distinct worlds: a relit volume
    before = _hdr(r)
    _exposed(r, [1.0, 4.0, 0.25])
    after = _hdr(r)
    log = r.debug.gi_rebakes()
    assert r.debug.gi_volume is not None and r.debug.gi_volume.relight is not None
  assert log.relights.tolist() == [0, 0, 0] and len(log.bakes) == 1, log
  np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-7)


def test_light_changes_after_a_relight_by_rays_recombine() -> None:
  """A world re-lit by rays for its own walls (materials randomised per episode) then changing only
  its lights' colours / intensities (per frame) is rescaled from its traced tint, no rays: it was
  re-lit by rays for every light change once its inputs left the authored ones. Its indirect light
  stays close to a re-bake of the world alone (much closer than the tint before the change)."""
  per_world = rw.PerWorld(light=["color", "strength"], material=["base_color"])
  outputs = rw.Outputs.of("hdr", "indirect")
  walls = [(RED, GREEN), (RED, RED), (WHITE, RED)]
  with _renderer(3, scene=_room(lamp=True), per_world=per_world) as r:
    r.update_settings(outputs=outputs)
    colors, strengths = r.per_world["light.color"].numpy().copy(), r.per_world["light.strength"].numpy().copy()
    _walls(r, [(RED, GREEN), (GREEN, RED), (WHITE, RED)])
    _indirect(r)  # the bake: distinct worlds, a relit volume
    _walls(r, walls)  # world 1's walls: one re-light by rays
    stale = _indirect(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 1, 0]
    colors[1, 1] = (1.0, 0.25, 0.1)  # world 1: a red lamp, the ceiling light dimmed
    strengths[1, 0] *= 0.4
    strengths[2, 1] = 0.0  # world 2: the lamp off, the ceiling light doubled
    strengths[2, 0] *= 2.0
    _lamps(r, colors, strengths)
    image = _indirect(r)
    log = r.debug.gi_rebakes()
  assert log.relights.tolist() == [0, 1, 0], log
  errors = {}
  for w in (1, 2):
    with _renderer(1, scene=_room(*walls[w], lamp=True), per_world=rw.PerWorld(light=["color", "strength"])) as alone:
      alone.update_settings(outputs=outputs)
      _lamps(alone, colors[w : w + 1], strengths[w : w + 1])
      ref = _indirect(alone)[0]
    errors[w] = (round(_error(image[w], ref), 4), round(_error(stale[w], ref), 4))
  print({"indirect error vs re-bake (rescaled, before the light change)": errors})
  for rescaled, before in errors.values():
    assert rescaled < 0.5 * before, errors


def test_switching_lights_off_recombines() -> None:
  """A light switched off scales its slot to zero: no re-light by rays (the inactive light's other
  properties no longer hashed made its non-linear inputs look changed)."""
  per_world = rw.PerWorld(light=["active", "strength"])
  with _renderer(2, scene=_room(lamp=True), per_world=per_world) as r:
    strength = r.per_world["light.strength"]
    s = strength.numpy()
    s[1, 0] *= 0.5  # distinct worlds: a relit volume
    strength.assign(s)
    _hdr(r)
    active = r.per_world["light.active"]
    on = active.numpy()
    on[1, 1] = False
    active.assign(on)
    _hdr(r)
    log = r.debug.gi_rebakes()
    assert r.debug.gi_volume is not None and r.debug.gi_volume.relight is not None
  assert log.relights.tolist() == [0, 0], log


def test_on_request_relights_wait_for_their_request() -> None:
  """gi.relight="on_request": a world whose walls change keeps its tint (no rays) until it is
  requested, then re-lights once and renders as under "on_change"; light changes still recombine
  meanwhile; a request without a change re-lights nothing."""
  per_world = rw.PerWorld(material=["base_color"], light=["strength"])
  first = [(RED, GREEN), (GREEN, RED), (WHITE, RED)]
  later = [(RED, GREEN), (GREEN, GREEN), (WHITE, RED)]
  with _renderer(3, per_world=per_world) as eager:
    _walls(eager, first)
    _hdr(eager)
    _walls(eager, later)
    reference = _hdr(eager)
  with _renderer(3, per_world=per_world) as r:
    r.update_settings(**{"gi.relight": "on_request"})
    _walls(r, first)
    _hdr(r)
    _walls(r, later)
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 0, 0]
    strength = r.per_world["light.strength"]
    strength.assign(strength.numpy() * np.array([[1.0], [1.0], [0.5]], np.float32))
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 0, 0]  # world 2: recombined
    r.request_gi_relight(wp.array([True, True, False], dtype=bool, device=r.device))
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 1, 0]
    r.request_gi_relight(wp.array([True, True, True], dtype=bool, device=r.device))
    image = _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 1, 0]
  np.testing.assert_allclose(image[1], reference[1], rtol=1e-4, atol=1e-6)


def test_on_request_a_request_waits_for_the_change() -> None:
  """gi.relight="on_request": a request made the frame before the world's table write (an episode
  reset requested, then its tables written) re-lights at the write's frame; it is served once (a
  later change waits for a new request)."""
  per_world = rw.PerWorld(material=["base_color"])
  first = [(RED, GREEN), (GREEN, RED), (WHITE, RED)]
  later = [(RED, GREEN), (GREEN, GREEN), (WHITE, RED)]
  with _renderer(3, per_world=per_world) as r:
    r.update_settings(**{"gi.relight": "on_request"})
    _walls(r, first)
    _hdr(r)
    r.request_gi_relight(wp.array([False, True, False], dtype=bool, device=r.device))
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 0, 0]  # nothing changed yet: still pending
    _walls(r, later)
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 1, 0]
    _walls(r, first)
    _hdr(r)
    assert r.debug.gi_rebakes().relights.tolist() == [0, 1, 0]  # served: held until requested again
