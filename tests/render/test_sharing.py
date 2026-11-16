"""Shared scenery (core/pipeline/worlds/sharing.py): what is shared, shared == per-world copies,
safety checks."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline.worlds.tables import InstanceColumns
from raywarp.core.pipeline.worlds.watch import pose_hint
from raywarp.settings.fields import DEFAULT_EV100

NWORLD = 3
# A table welded to the world body, a free ball, a mocap target and a floor.
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <geom name="post" type="box" size="0.05 0.05 0.3" pos="-0.7 0.6 0.3" rgba="0.3 0.3 0.8 1"/>
    <body name="table" pos="0 0 0.4">
      <geom name="top" type="box" size="0.5 0.5 0.02" rgba="0.6 0.4 0.2 1"/>
      <geom name="knob" type="sphere" size="0.06" pos="-0.4 0.4 0.06" rgba="0.2 0.8 0.2 1"/>
      <geom name="rail" type="capsule" size="0.02 0.3" pos="0 -0.45 0.05" euler="0 90 0"/>
      <geom name="egg" type="ellipsoid" size="0.05 0.08 0.04" pos="0.3 -0.3 0.06"/>
      <geom name="can" type="cylinder" size="0.04 0.06" pos="0.3 0.3 0.08"/>
    </body>
    <body name="ball" pos="0 0 0.6">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.08" rgba="0.8 0.1 0.1 1"/>
    </body>
    <body name="target" mocap="true" pos="0.2 0.2 0.5">
      <geom name="target" type="box" size="0.04 0.04 0.04" rgba="0.1 0.1 0.9 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="cam" pos="1.4 -1.4 1.3" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""
SETTINGS = rw.RenderSettings(
  gi=NO_GI,
  outputs=rw.Outputs.of("rgb", "depth", "seg"),
  performance=rw.settings.Performance(traversal="software"),
  shadows=rw.settings.Shadows(enabled=True),
  exposure=rw.settings.Exposure(mode="manual", ev100=DEFAULT_EV100),  # renders are compared across worlds and runs
)
# The per-world appearance `_randomise` writes.
LOOKS = {"material": ["color"], "light": ["strength"]}


@pytest.fixture
def mj() -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of `XML` with NWORLD worlds (the mocap target moved in world 1)."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=NWORLD)
  q = d.mocap_pos.numpy()
  q[1, 0] += (0.1, 0.0, 0.0)  # the mocap target differs in world 1
  d.mocap_pos.assign(q)
  mjw.forward(m, d)
  return mjm, m, d


def _ids(mjm: mujoco.MjModel, *names: str) -> list[int]:
  """The sorted geom ids of `names`."""
  return sorted(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names)


def _renderer[S](scene: rw.GpuScene[S], per_world: rw.PerWorld | None = None, traversal: str = "software") -> rw.Renderer[S]:
  """A Renderer of `scene` (NWORLD worlds, camera "cam", SETTINGS with `traversal`)."""
  settings = SETTINGS.replace(**{"performance.traversal": traversal})
  return rw.Renderer(scene, nworld=NWORLD, cameras=["cam"], resolution=(32, 24), settings=settings, per_world=per_world)


def _out[S](r: rw.Renderer[S], sim: S | None = None) -> np.ndarray:
  """(NWORLD, pixels) rgba, depth and seg of a render of `sim`."""
  r.render(sim)
  o = r.outputs
  return np.concatenate(
    [
      required(o.rgb).numpy().reshape(NWORLD, -1).astype(np.float64),
      required(o.depth).numpy().reshape(NWORLD, -1),
      required(o.seg).numpy().reshape(NWORLD, -1),
    ],
    axis=1,
  )


def _randomise[S](r: rw.Renderer[S]) -> None:
  """Per-world appearance and lights (never affects sharing)."""
  c = r.per_world["instance.color"]
  v = c.numpy()
  v[1, :, :3] *= 0.5
  c.assign(v)
  li = r.per_world["light.strength"]
  v = li.numpy()
  v[2] *= 0.3
  li.assign(v)


def test_mujoco_static_geoms_are_shared_and_render_as_copies(mj):
  mjm, m, d = mj
  # Batched mjw fields: identical rows keep sharing; a differing geom_pos row makes "post" per
  # world; per-world colors (geom_rgba) do not affect sharing.
  pos = np.tile(m.geom_pos.numpy()[:1], (NWORLD, 1, 1))
  post = _ids(mjm, "post")[0]
  pos[2, post, 1] += 0.2
  m.geom_pos = wp.array(pos, dtype=wp.vec3)
  rgba = np.tile(m.geom_rgba.numpy()[:1], (NWORLD, 1, 1))
  rgba[1, _ids(mjm, "top")[0], :3] = (0.9, 0.9, 0.1)
  m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  mjw.forward(m, d)
  scene = rw.integrations.mujoco.scene(mjm, m, d)
  shared = _renderer(scene, rw.PerWorld(**LOOKS))
  np.testing.assert_array_equal(shared.layout.shared_ids, _ids(mjm, "top", "knob", "rail", "egg", "can"))
  assert shared.features.shared and not shared.features.compact_poses
  copies = _renderer(scene, rw.PerWorld(pose=["*"], **LOOKS))
  assert copies.layout.nshared == 0 and not copies.features.shared
  base = _out(shared, d)
  np.testing.assert_array_equal(base, _out(copies, d))
  assert np.any(base[1] != base[0])  # color and mocap differ
  assert shared.layout.per_world[post] and shared.layout.dynamic[post]  # a differing batched geom_pos
  for r in (shared, copies):
    _randomise(r)
  np.testing.assert_array_equal(_out(shared, d), _out(copies, d))


def test_mujoco_per_world_declarations(mj):
  mjm, m, d = mj
  scene = rw.integrations.mujoco.scene(mjm, m, d)
  r = _renderer(scene)
  with pytest.raises(ValueError, match=r"'top' is shared by every world.*PerWorld\(size=\['top'\]\)"):
    r.per_world.columns("instance.size", "top")
  with pytest.raises(ValueError, match="matches no instance"):
    _renderer(scene, rw.PerWorld(visible=["nope*"]))
  r = _renderer(scene, rw.PerWorld(size=["top"], visible=["eg*"]))
  assert _ids(mjm, "top")[0] not in r.layout.shared_ids and _ids(mjm, "egg")[0] in r.layout.shared_ids
  with pytest.raises(ValueError, match=r"'can' is shown in every world.*PerWorld\(visible=\['can'\]\)"):
    r.per_world.columns("instance.visible", "can")
  with pytest.raises(ValueError, match="come from the simulator"):
    r.per_world.columns("instance.pos", "top")
  base = _out(r, d)
  vis = r.per_world.columns("instance.visible", "eg*")
  assert vis.shape == (NWORLD, 1)
  vis.assign(np.array([[1], [0], [1]]))
  size = r.per_world.columns("instance.size", "top")
  s = size.numpy()
  s[2, 0, 2] *= 4.0
  size.assign(s)
  out = _out(r, d)
  np.testing.assert_array_equal(out[0], base[0])
  egg = _ids(mjm, "egg")[0]
  seg = out[:, -32 * 24 :]
  assert np.any(seg[0] == egg) and not np.any(seg[1] == egg)
  assert np.any(out[2] != base[2])


# Static geoms of every shape type (and the floor plane) hidden per world.
HIDE = ["floor", "post", "top", "knob", "rail", "egg", "can"]


def _hide[S](r: rw.Renderer[S]) -> np.ndarray:
  """Hides alternate declared geoms in worlds 1 and 2 (world 0 shows all); returns the table."""
  vis = r.per_world["instance.visible"]
  shown = np.ones(vis.shape, np.int32)
  shown[1, ::2] = 0
  shown[2, 1::2] = 0
  vis.assign(shown)
  return shown


def test_per_world_visibility_keeps_shared_instances_shared(mj):
  """W41 (a): `PerWorld.visible` is a per-world mask on shared instances: the same pixels as the
  same instances as per-world copies, world 0 (all shown) as without the declaration."""
  mjm, m, d = mj
  scene = rw.integrations.mujoco.scene(mjm, m, d)
  masked = _renderer(scene, rw.PerWorld(visible=HIDE))
  static = _ids(mjm, *HIDE[1:])
  assert set(static) <= set(masked.layout.shared_ids.tolist()) and masked.features.visibility
  assert masked.layout.nvisible == len(HIDE)
  copies = _renderer(scene, rw.PerWorld(pose=["*"], visible=HIDE))
  assert copies.layout.nshared == 0
  shown = _hide(masked)
  _hide(copies)
  out = _out(masked, d)
  np.testing.assert_array_equal(out, _out(copies, d))
  np.testing.assert_array_equal(out[0], _out(_renderer(scene), d)[0])
  seg = out[:, -32 * 24 :]
  visible = masked.per_world["instance.visible"]
  assert isinstance(visible, InstanceColumns)  # the declared instances' columns
  names = visible.names
  seen = [k for k, name in enumerate(names) if np.any(seg[0] == _ids(mjm, name)[0])]
  assert len(seen) >= 5
  for w in (1, 2):
    for k in seen:
      assert np.any(seg[w] == _ids(mjm, names[k])[0]) == bool(shown[w, k]), (w, names[k])


@pytest.mark.parametrize("worlds", ["all", "one"])
def test_moved_shared_scenery_raises(mj, worlds):
  """Shared scenery is static: the table top moved alike in every world or in one world (MuJoCo
  Warp computes static geom poses once, in make_data / put_data: write them) is detected one frame
  later and raises with the declaration to add; once restored, renders go on."""
  mjm, m, d = mj
  r = _renderer(rw.integrations.mujoco.scene(mjm, m, d))
  _out(r, d)
  top = _ids(mjm, "top")[0]
  pos = d.geom_xpos.numpy().copy()
  moved = pos.copy()
  moved[slice(None) if worlds == "all" else 1, top, 0] += 0.15
  d.geom_xpos.assign(moved)
  _out(r, d)  # checks the moved poses
  with pytest.raises(ValueError, match=r"shared scenery 'top' moved or changed.*PerWorld\(pose=\['top'\]\)"):
    _out(r, d)
  d.geom_xpos.assign(pos)
  _out(r, d)
  _out(r, d)


# A room on a mocap body (as mjlab wraps a fixed-base entity), a free ball and a camera.
ROOM = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <body name="room" mocap="true">
      <geom name="room/floor" type="box" size="1 1 0.02" rgba="0.6 0.6 0.6 1"/>
      <geom name="room/wall" type="box" size="0.02 1 0.5" pos="-1 0 0.5" rgba="0.8 0.3 0.2 1"/>
      <geom name="room/desk" type="box" size="0.3 0.2 0.02" pos="0.2 0.3 0.4" rgba="0.5 0.4 0.2 1"/>
    </body>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.08" rgba="0.8 0.1 0.1 1"/>
    </body>
    <camera name="cam" pos="1.6 -1.6 1.4" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""
_ROOM = ["room/floor", "room/wall", "room/desk"]


def _room(move: tuple[float, float, float] | None = None) -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of `ROOM` with NWORLD worlds; `move`: the room's mocap offset in world 1."""
  mjm = mujoco.MjModel.from_xml_string(ROOM)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=NWORLD)
  if move is not None:
    _move_room(m, d, {1: move})
  return mjm, m, d


def _move_room(m: mjw.Model, d: mjw.Data, moves: dict[int, tuple[float, float, float]]) -> None:
  """Offsets the room's mocap body in the given worlds, then runs the kinematics."""
  q = d.mocap_pos.numpy()
  for w, offset in moves.items():
    q[w, 0] += offset
  d.mocap_pos.assign(q)
  mjw.forward(m, d)


def _same(traversal: str, a: np.ndarray, b: np.ndarray, *, cpu: bool) -> None:
  """`_out` rows equal: exactly in software on the CPU. OptiX's per-world cell transforms round
  hits differently from the shared IAS (about 1e-5 m; at a silhouette a pixel may flip), and so
  does the CUDA software traversal's shared (compact) vs per-world pose path (FMA contraction:
  2 of 768 depths 2e-5 m apart on the RTX PRO 6000): there at most 1 % of pixels differ in seg or
  by more than 1 in rgba, depth within 1e-4 m elsewhere."""
  if traversal == "software" and cpu:
    np.testing.assert_array_equal(a, b)
    return
  npix = 32 * 24
  rgba = np.abs(a[:, : 4 * npix] - b[:, : 4 * npix]).reshape(len(a), npix, 4).max(-1) > 1
  seg = a[:, 5 * npix :] != b[:, 5 * npix :]
  assert rgba.mean() <= 0.01 and seg.mean() <= 0.01, (rgba.mean(), seg.mean())
  depth = np.abs(a[:, 4 * npix : 5 * npix] - b[:, 4 * npix : 5 * npix])
  assert depth[~seg].max() <= 1e-4


_TRAVERSALS = [
  "software",
  pytest.param("optix", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="OptiX needs CUDA")),
]


@pytest.mark.parametrize("traversal", _TRAVERSALS)
def test_mocap_body_posed_alike_in_every_world_is_shared(traversal):
  mjm, m, d = _room()
  scene = rw.integrations.mujoco.scene(mjm, m, d)
  r = _renderer(scene, traversal=traversal)
  assert r.features.traversal == traversal
  room = _ids(mjm, *_ROOM)
  np.testing.assert_array_equal(r.layout.shared_ids, room)
  assert not r.layout.dynamic[room].any() and r.layout.dynamic[_ids(mjm, "ball")].all()
  copies = _renderer(scene, rw.PerWorld(pose=["*"]), traversal=traversal)
  _same(traversal, _out(r, d), _out(copies, d), cpu=r.device.is_cpu)
  # The room moves (the same in every world): raises at the next render, with a one-line
  # declaration of the whole body (shared scenery is static).
  _move_room(m, d, dict.fromkeys(range(NWORLD), (0.1, 0.05, 0.0)))
  _out(r, d)
  with pytest.raises(ValueError, match=r"'room/floor'.* moved or changed.*PerWorld\(pose=\['room/\*'\]\)"):
    _out(r, d)
  # The declared renderer renders the per-world rooms as copies do.
  declared = _renderer(scene, rw.PerWorld(pose=["room/*"]), traversal=traversal)
  assert declared.layout.nshared == 0
  _same(traversal, _out(declared, d), _out(copies, d), cpu=r.device.is_cpu)


def test_mocap_body_posed_per_world_at_build_is_per_world():
  mjm, m, d = _room(move=(0.0, 0.3, 0.0))
  r = _renderer(rw.integrations.mujoco.scene(mjm, m, d))
  assert r.layout.nshared == 0 and r.layout.dynamic[_ids(mjm, *_ROOM)].all()
  base = _out(r, d)
  assert np.any(base[1] != base[0]) and np.array_equal(base[2], base[0])
  copies = _renderer(rw.integrations.mujoco.scene(mjm, m, d), rw.PerWorld(pose=["*"]))
  np.testing.assert_array_equal(base, _out(copies, d))
  # Without the data the room is posed by m (the same in every world): shared, then raises.
  shared = _renderer(rw.integrations.mujoco.scene(mjm, m))
  assert shared.layout.nshared == len(_ROOM)
  _out(shared, d)
  with pytest.raises(ValueError, match=r"moved or changed after the first frame"):
    _out(shared, d)


def test_pose_hint_globs_a_whole_path_prefix():
  every = ["room/floor", "room/wall", "room/desk", "robot/pelvis", "geom4"]
  assert pose_hint(["room/floor", "room/wall", "room/desk"], every) == "'room/*'"
  assert pose_hint(["room/floor", "room/wall"], every) == "'room/floor', 'room/wall'"  # the glob would match room/desk
  assert pose_hint(["room/floor"], every) == "'room/floor'"
  assert pose_hint(["room/floor", "robot/pelvis"], every) == "'room/floor', 'robot/pelvis'"


def test_host_scene_shared_matches_copies():
  builder = rw.SceneBuilder()
  mat = builder.add_material(rw.Material(base_color=(0.8, 0.7, 0.6), specular_roughness=0.5))
  for i, kind in enumerate(("box", "sphere", "capsule", "cylinder", "ellipsoid")):
    xf = np.eye(4)[:3]
    xf[:, 3] = (0.5 * i - 1.0, 0.0, 0.2)
    builder.add_instance(kind, material=mat, size=(0.15, 0.1, 0.12), xform=xf, path=f"/World/prop{i}")
  floor = np.eye(4)[:3]
  builder.add_instance("plane", material=mat, size=(3.0, 3.0, 0.0), xform=floor, path="/World/floor")
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0, xform=np.eye(4)[:3]))
  scene = rw.upload(builder.build(), device="cpu")
  assert scene.instance_names[0] == "/World/prop0"
  cams = [rw.CameraSpec(fovy=50.0)]
  settings = rw.RenderSettings(
    gi=NO_GI,
    outputs=rw.Outputs.of("rgb", "depth"),
    performance=rw.settings.Performance(traversal="software"),
    exposure=rw.settings.Exposure(mode="manual", ev100=DEFAULT_EV100),
  )
  out = []
  for pw in (None, rw.PerWorld(pose=["/World/*"])):
    r = rw.Renderer(scene, nworld=2, cameras=cams, resolution=(24, 16), settings=settings, per_world=pw)
    pose = np.zeros((2, 1, 7), np.float32)
    pose[..., 1:3] = (-3.0, 2.0)
    pose[..., 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.95))
    r.camera_pose.assign(pose)
    out.append(required(r.render().depth).numpy())
    assert r.layout.compact == (pw is None) and r.layout.nshared == (6 if pw is None else 0)  # the floor too
  np.testing.assert_array_equal(out[0], out[1])
