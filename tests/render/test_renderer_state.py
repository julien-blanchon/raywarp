"""Renderer state across frames and setting changes (pass F bugs B1-B4, B6, B7, P1, W21 checks).

Each test renders the same thing two ways (changed in place vs built fresh) or checks the
per-frame validation (`core/pipeline/worlds/validator.py`): moved / reshaped shared scenery,
out-of-range index tables, a late pinned flag word.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, internal_module, required, whole_table

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp._base.records import record_field
from raywarp.settings.fields import DEFAULT_EV100

NWORLD = 2
# A table top shared by every world (welded to the world body), a free ball, a floor, a mesh.
XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="table" pos="0 0 0.4">
      <geom name="top" type="box" size="0.2 0.2 0.02" rgba="0.6 0.4 0.2 1"/>
    </body>
    <geom name="tet" type="mesh" mesh="tet" pos="0.5 0.3 0.05" rgba="0.2 0.7 0.3 1"/>
    <body name="ball" pos="0 0 0.8">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.08" rgba="0.8 0.1 0.1 1"/>
    </body>
    <camera name="cam" pos="1.4 -1.4 1.3" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""
SETTINGS = rw.RenderSettings(
  gi=NO_GI,
  outputs=rw.Outputs.of("rgb", "depth", "seg"),
  performance=rw.settings.Performance(traversal="software"),
  exposure=rw.settings.Exposure(mode="manual", ev100=DEFAULT_EV100),  # renders are compared across worlds and runs
)


def _mj() -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of XML with NWORLD worlds, forward kinematics run."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=NWORLD)
  mjw.forward(m, d)
  return mjm, m, d


def _geom(mjm: mujoco.MjModel, name: str) -> int:
  """The id of geom `name`."""
  return mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name)


def _renderer[S](scene: rw.GpuScene[S], per_world: rw.PerWorld | None = None, **settings: object) -> rw.Renderer[S]:
  """A Renderer of `scene` (NWORLD worlds, camera "cam") with `settings` over SETTINGS."""
  return rw.Renderer(
    scene, nworld=NWORLD, cameras=["cam"], resolution=(32, 24), settings=SETTINGS.replace(**settings), per_world=per_world
  )


def _images[S](r: rw.Renderer[S], sim: S | None = None) -> np.ndarray:
  """(nworld, -1) rgb, depth and seg of one render."""
  out = r.render(sim)
  return np.concatenate(
    [
      required(out.rgb).numpy().reshape(NWORLD, -1).astype(np.float64),
      required(out.depth).numpy().reshape(NWORLD, -1),
      required(out.seg).numpy().reshape(NWORLD, -1),
    ],
    axis=1,
  )


def test_b4_translucent_shadow_mode_changes_with_update_settings():
  """B4: switching shadows.translucent between "transmittance" and "clear" in place renders as a
  Renderer built with the new mode (the colored pane's shadow is untinted with "clear")."""
  fs = internal_module("feature_scenes")
  views = [((0.5, 0.05 - 1e-3, 0.3), (0.5, 0.05, 0.0), 12.0)]
  base = {"outputs": rw.Outputs.of("hdr"), "performance.traversal": "software", "gi.enabled": False}
  rig = fs.load("glass", settings={**base, "shadows.translucent": "transmittance"}, res=(8, 8), views=views, device="cpu")
  tinted = rig.render().hdr.numpy().copy()
  rig.r.update_settings(**{"shadows.translucent": "clear"})
  changed = rig.render().hdr.numpy()
  assert not np.array_equal(changed, tinted)  # the view sees the pane's shadow
  fresh = fs.load("glass", settings={**base, "shadows.translucent": "clear"}, res=(8, 8), views=views, device="cpu")
  np.testing.assert_array_equal(changed, fresh.render().hdr.numpy())


def _panel_scene() -> rw.GpuScene[None]:
  """A floor under an emissive panel (made a two-sided rect light by the upload) and a sun."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=0.6))
  glow = b.add_material(
    rw.Material(base_color=(0.0, 0.0, 0.0), specular_roughness=1.0, emission_luminance=50.0, emission_color=(1, 1, 1))
  )
  b.add_instance("plane", size=(5, 5, 0), material=floor)
  b.add_instance("box", size=(1.0, 1.0, 0.004), xform=np.c_[np.eye(3), [0, 0, 2]], material=glow)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=1.0, angle=0.0, xform=np.eye(4)[:3]))
  return rw.upload(b.build(), device="cpu")


def _panel_renderer(scene: rw.GpuScene[None], panels: bool) -> rw.Renderer[None]:
  """A light-grid Renderer of `_panel_scene` looking down at the floor."""
  s = rw.RenderSettings(
    gi=NO_GI,
    outputs=rw.Outputs.of("hdr"),
    lighting=rw.settings.Lights(light_grid=True, emissive_lights=panels),
    environment=rw.settings.Environment(enabled=False),
    performance=rw.settings.Performance(traversal="software"),
  )
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=40.0)], resolution=(8, 8), settings=s)
  r.camera_pose.assign(np.array([[[0, 0, 1, 0, 0, 0, 1]]], np.float32))
  return r


def test_b6_light_grid_follows_the_settings_it_was_built_with():
  """B6: the light grid lists the lights its build saw: one built with the emissive-panel lights
  off is rebuilt when they are switched on (else the panel never lights the floor)."""
  scene = _panel_scene()
  r = _panel_renderer(scene, panels=False)
  unlit = required(r.render().hdr).numpy().copy()
  r.update_settings(**{"lighting.emissive_lights": True})
  changed = required(r.render().hdr).numpy()
  assert changed.mean() > unlit.mean()  # the panel lights the floor
  np.testing.assert_array_equal(changed, required(_panel_renderer(scene, panels=True).render().hdr).numpy())


def test_b1_a_size_change_of_shared_scenery_raises():
  """B1: a common change of a shared instance's size (an aliased m.geom_size) is detected by the
  per-frame validation and raises (shared scenery is static), naming the instance."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene)
  top = _geom(mjm, "top")
  assert top in r.layout.shared_ids.tolist()
  _images(r, d)
  size = m.geom_size.numpy()
  size[:, top, :2] *= 2.5  # a much wider table top
  m.geom_size.assign(size)
  _images(r, d)  # checks: the shared top changed
  with pytest.raises(ValueError, match=r"shared scenery 'top' moved or changed after the first frame"):
    _images(r, d)  # reads the flag word


def test_b3_moved_error_clears_once_the_poses_are_restored():
  """B3: the "shared scenery moved" error is raised once per detection: after the poses are
  restored, renders succeed again."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene)
  _images(r, d)
  top = _geom(mjm, "top")
  pos = d.geom_xpos.numpy().copy()
  moved = pos.copy()
  moved[1, top, 1] += 0.1
  d.geom_xpos.assign(moved)
  _images(r, d)
  with pytest.raises(ValueError, match=r"'top' moved or changed.*PerWorld\(pose=\['top'\]\)"):
    _images(r, d)
  d.geom_xpos.assign(pos)
  _images(r, d)
  _images(r, d)
  _images(r, d)


def test_b7_a_stale_flag_word_does_not_raise_twice():
  """B7: a flag word copied before an error was reported (a late pinned copy on CUDA) carries the
  epoch it was checked against and is ignored after the report: one error per detection."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene)
  _images(r, d)
  pos = d.geom_xpos.numpy().copy()
  moved = pos.copy()
  moved[:, _geom(mjm, "top"), 0] += 0.1
  d.geom_xpos.assign(moved)
  _images(r, d)
  validator = r._watch.validator
  stale = validator.host.numpy().copy()  # the word that reports the move
  with pytest.raises(ValueError, match="shared scenery 'top' moved"):
    _images(r, d)
  d.geom_xpos.assign(pos)
  validator.host.numpy()[:] = stale  # a late copy of that old word lands now
  _images(r, d)


def test_validation_without_shared_scenery_allocates_once(monkeypatch):
  """GPU-4: a scene without shared scenery never snapshots; its frames check against one empty
  snapshot allocated once (a fresh one per frame cost two host -> device copies per frame, also
  inside a captured CUDA graph: the Newton residency failure)."""
  from raywarp.core.pipeline.worlds import validator as validate

  allocated = []
  allocate = validate.Validator._allocate
  monkeypatch.setattr(validate.Validator, "_allocate", lambda self, source: allocated.append(1) or allocate(self, source))
  mjm = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3" dir="0 0 -1"/><camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>'
    '<body pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.5"/></body></worldbody></mujoco>'
  )
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  r = _renderer(rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm)))
  d = mjw.put_data(mjm, mjd, nworld=NWORLD)
  assert r.layout.nshared == 0 and r._watch.validator.active()
  for _ in range(3):
    r.render(d)
  assert len(allocated) == 1
  assert r._watch.validator.snapshot is None  # still nothing shared to check


def test_env_id_written_in_place_is_validated_every_frame():
  """W21 / W36: an out-of-range "env.id" written in place after the first render is reported by
  the per-frame validation (one frame later, no host read on the frame path) and clamped
  meanwhile."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene, rw.PerWorld(env=["id"]))
  _images(r, d)
  r.per_world["env.id"].assign(np.array([0, 7], np.int32))
  _images(r, d)  # clamped in the kernels; the validation word says out of range
  with pytest.raises(ValueError, match=r"env\.id"):
    _images(r, d)
  whole_table(r.per_world["env.id"]).zero_()
  _images(r, d)
  _images(r, d)


def test_mesh_id_out_of_range_is_reported_and_clamped():
  """W21: an out-of-range per-world "instance.mesh" id renders (clamped in-kernel, never an
  out-of-bounds read) and raises on the next frame."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene, rw.PerWorld(mesh=["tet"]))
  base = _images(r, d)
  mesh = r.per_world.columns("instance.mesh", "tet")
  mesh.assign(np.full((NWORLD, 1), 5, np.int32))
  np.testing.assert_array_equal(_images(r, d), base)  # clamped to the only mesh
  with pytest.raises(ValueError, match=r"instance\.mesh"):
    _images(r, d)


def _watch_launches[**P, R](launch: Callable[P, R], running: list[int], early: list[str]) -> Callable[P, R]:
  """`launch` recording in `early` the kernels launched from the main thread while `running`."""

  def watched(*args: P.args, **kwargs: P.kwargs) -> R:
    """`launch`, recording an early launch's kernel key."""
    if threading.current_thread() is threading.main_thread() and running:
      early.append(str(getattr(args[0] if args else kwargs.get("kernel"), "key", "?")))
    return launch(*args, **kwargs)

  return watched


def test_p1_first_render_waits_for_the_warmup_before_launching(monkeypatch):
  """P1: the first render launches nothing before the background compile of its modules is done
  (the frame's compile errors are raised before any launch)."""
  from raywarp.core.pipeline import warmup

  running = []  # one entry per compile thread still running
  run = warmup.Warmup._run

  def slow_run(device: wp.DeviceLike, mods: list[wp.Module], block_dim: int | None) -> None:
    running.append(1)
    time.sleep(0.3)
    run(device, mods, block_dim)
    running.pop()

  monkeypatch.setattr(warmup.Warmup, "_run", staticmethod(slow_run))
  early: list[str] = []
  watched = _watch_launches(wp.launch, running, early)
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene)
  monkeypatch.setattr(wp, "launch", watched)
  r.render(d)
  assert not early, f"launched before the warmup finished: {early}"


def test_first_render_waits_for_a_replaced_warmup(monkeypatch):
  """P1 after a settings change: the constructor's warm-up still compiling when the Features
  change (`update_settings` before the first render) is waited for too (the render waited only
  for the new one, then launched while the old threads still loaded modules)."""
  from raywarp.core.pipeline import warmup

  running = []  # one entry per compile thread still running
  calls = []
  run = warmup.Warmup._run

  def slow_run(device: wp.DeviceLike, mods: list[wp.Module], block_dim: int | None) -> None:
    calls.append(1)
    running.append(1)
    time.sleep(1.0 if len(calls) <= 2 else 0.05)  # the constructor's two threads are the slow ones
    run(device, mods, block_dim)
    running.pop()

  monkeypatch.setattr(warmup.Warmup, "_run", staticmethod(slow_run))
  mjm, m, d = _mj()
  r = _renderer(rw.integrations.mujoco.scene(mjm, m), None, **rw.FAST)  # reflections off: turning them on compiles
  r.update_settings(**{"reflections.enabled": not r.settings.reflections.enabled})
  early: list[str] = []
  watched = _watch_launches(wp.launch, running, early)
  monkeypatch.setattr(wp, "launch", watched)
  r.render(d)
  assert len(calls) > 2  # a second warm-up ran
  assert not early, f"launched before the first warm-up finished: {early}"


def test_instance_columns_assign_is_a_device_scatter(monkeypatch):
  """W10: `InstanceColumns.assign` of a Warp array writes the matched columns with one kernel (no
  host round trip of the whole table: graph-capturable); NumPy values broadcast as before."""
  mjm, m, _d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene, rw.PerWorld(material=["color"]))
  table = r.per_world["instance.color"]
  before = table.numpy().copy()
  cols = r.per_world.columns("instance.color", ["top", "ball"])
  values = wp.array(np.full((NWORLD, 2, 4), 0.25, np.float32), dtype=wp.vec4)
  reads = []
  monkeypatch.setattr(type(table), "numpy", lambda self: reads.append(1) or np.zeros(0))
  cols.assign(values)
  monkeypatch.undo()
  assert not reads
  after = table.numpy()
  idx = sorted([_geom(mjm, "top"), _geom(mjm, "ball")])
  np.testing.assert_array_equal(after[:, idx], 0.25)
  rest = np.setdiff1d(np.arange(after.shape[1]), idx)
  np.testing.assert_array_equal(after[:, rest], before[:, rest])
  cols.assign(np.array([0.5, 0.5, 0.5, 1.0], np.float32))  # broadcast to every world and column
  np.testing.assert_array_equal(cols.numpy(), np.broadcast_to([0.5, 0.5, 0.5, 1.0], (NWORLD, 2, 4)))


def test_per_world_writes_never_reconfigure():
  """W10: a declared per-world field compiles its feature at construction ("auto" translucency for
  instance colors with a per-world alpha): writing translucent values keeps the same Features and
  kernel."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene, rw.PerWorld(material=["color"], alpha=True))
  features = r.features
  assert features.transparency
  _images(r, d)
  color = r.per_world["instance.color"]
  rgba = color.numpy().copy()
  rgba[..., 3] = 0.5
  color.assign(rgba)
  _images(r, d)
  assert r.features is features


def test_live_material_id_out_of_range_is_reported_and_clamped():
  """W21: an out-of-range live material id (MuJoCo `m.geom_matid`, here 3 in a model without
  materials) renders clamped (no material; never an out-of-bounds read: it crashed the CPU
  renderer before) and raises on the next frame."""
  mjm, m, d = _mj()
  scene = rw.integrations.mujoco.scene(mjm, m)
  r = _renderer(scene)
  base = _images(r, d)
  ids = m.geom_matid.numpy()
  ids[..., _geom(mjm, "top")] = 3
  m.geom_matid.assign(ids)
  np.testing.assert_array_equal(_images(r, d), base)
  with pytest.raises(ValueError, match=r"instance\.material"):
    _images(r, d)


def test_instance_columns_assign_rejects_a_wrong_world_count():
  """`InstanceColumns.assign` of a Warp array takes 1 or nworld rows: 3 rows for 2 worlds raise
  (they were read modulo 3, silently)."""
  mjm, m, _d = _mj()
  r = _renderer(rw.integrations.mujoco.scene(mjm, m), rw.PerWorld(material=["color"]))
  cols = r.per_world.columns("instance.color", ["top", "ball"])
  with pytest.raises(ValueError, match="nworld or 1"):
    cols.assign(wp.array(np.zeros((NWORLD + 1, 2, 4), np.float32), dtype=wp.vec4))
  cols.assign(wp.array(np.zeros((1, 2, 4), np.float32), dtype=wp.vec4))


def test_set_clip_updates_the_supersampled_lens_tables_in_place():
  """`set_clip` writes the supersampled (spp n, OptiX) lens tables in place, like the main one: a
  captured graph keeps reading the same arrays (they were dropped and rebuilt: a replay read the
  old clip planes from a freed table)."""
  mjm, m, _d = _mj()
  r = _renderer(rw.integrations.mujoco.scene(mjm, m))
  table = r._rig.scaled_lens(2)
  r.set_clip(znear=0.25)
  assert r._rig.scaled_lens(2) is table
  np.testing.assert_allclose(record_field(table.numpy(), "znear"), 0.25)


_TEXTURED = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0 0 0" width="16" height="16"/>
    <material name="m" texture="grid"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" material="m"/>
    <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


def test_live_texture_id_out_of_range_is_reported_and_clamped():
  """N1 (M10): an out-of-range live texture id (MuJoCo `m.mat_texid`, aliased by the texture slot,
  here 99 with one texture) read the texture table out of bounds: a segfault on the CPU device on
  the first frame. It now renders with the last texture and raises on the next frame."""
  mjm = mujoco.MjModel.from_xml_string(_TEXTURED)
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=NWORLD)
  mjw.forward(m, d)
  r = _renderer(rw.integrations.mujoco.scene(mjm, m))
  base = _images(r, d)
  tex = m.mat_texid.numpy()
  tex[tex >= 0] = 99
  m.mat_texid.assign(tex)
  np.testing.assert_array_equal(_images(r, d), base)
  with pytest.raises(ValueError, match=r"material\.texture"):
    _images(r, d)


def test_reset_history_rejects_a_wrong_mask_without_temporal_state():
  """A reset mask that is not a bool (nworld,) array raises, also when the renderer keeps no
  temporal state (it was checked only inside the history)."""
  mjm, m, _d = _mj()
  with _renderer(rw.integrations.mujoco.scene(mjm, m)) as r:
    r.reset_history(wp.ones(NWORLD, dtype=wp.bool, device=r.device))
    for bad in (wp.ones(NWORLD + 1, dtype=wp.bool, device=r.device), wp.ones(NWORLD, dtype=wp.int32, device=r.device)):
      with pytest.raises(ValueError, match="reset_history"):
        r.reset_history(bad)  # pyrefly: ignore[bad-argument-type]  # under test: wrong shape, wrong dtype


def test_render_after_close_raises():
  """A closed renderer refuses to render (it rendered silently), to change settings, to run GI or
  to reset its history (those re-made the released OptiX build); closing again is a no-op."""
  mjm, m, d = _mj()
  r = _renderer(rw.integrations.mujoco.scene(mjm, m))
  r.render(d)
  r.close()
  for call in (lambda: r.render(d), lambda: r.update_settings(**{"shadows.samples": 2}), r.update_gi, r.reset_history):
    with pytest.raises(RuntimeError, match="after close"):
      call()
  r.close()


def test_setters_after_close_raise():
  """Every setter of a closed renderer raises before changing anything (they changed the released
  renderer's tables silently): per-world binding, camera rays, clip, outputs, semantics, segment
  table, world skip."""
  mjm, m, _d = _mj()
  r = _renderer(rw.integrations.mujoco.scene(mjm, m))
  r.close()
  calls = {
    "bind_per_world": lambda: r.bind_per_world("material.base_color", wp.zeros((1, 1), dtype=wp.vec4)),
    "set_camera_rays": lambda: r.set_camera_rays([]),
    "set_clip": lambda: r.set_clip(znear=0.01),
    "bind_outputs": r.bind_outputs,
    "set_semantics": lambda: r.set_semantics({}),
    "set_segment_table": lambda: r.set_segment_table(None),
  }
  for name, call in calls.items():
    with pytest.raises(RuntimeError, match=f"Renderer.{name} after close"):
      call()


def test_a_failed_construction_releases_what_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
  """A constructor failing after the backend was made (settings the scene cannot render, checked
  after it) releases it: nothing else can (the Renderer is never returned)."""
  from raywarp.core.pipeline import renderer as renderer_module

  closed: list[str] = []
  close = renderer_module.SoftwareBackend.close

  def recording(backend: object) -> None:
    closed.append("backend")
    close(backend)  # pyrefly: ignore[bad-argument-type]

  monkeypatch.setattr(renderer_module.SoftwareBackend, "close", recording)
  mjm, m, _d = _mj()
  bad = {"outputs": rw.Outputs.of("direct"), "surfaces.shading": "unlit"}
  with pytest.raises(ValueError, match="unlit"):
    _renderer(rw.integrations.mujoco.scene(mjm, m), **bad)
  assert closed == ["backend"]
