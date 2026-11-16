"""The per-frame validation word (`core/pipeline/worlds/validator.py`,
`core/pipeline/worlds/watch.py`): late pinned copies, undone moves, the baked-lighting check's
fields; the graph replay's array fingerprint; backend names."""

import warnings
from collections.abc import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, make_scene, whole_table

import raywarp as rw
from raywarp.core.pipeline import graph
from raywarp.core.pipeline.gpu_scene.binding import StaleBinding
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.worlds.validator import Flags
from raywarp.core.pipeline.worlds.validator_kernels import EPOCH_MASK, Word
from raywarp.core.trace import backend

NWORLD = 2

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom name="top" type="box" pos="0 0 0.3" size="0.2 0.2 0.2" rgba="0.8 0.3 0.2 1"/>
    <body pos="0.8 0 0.3"><freejoint/><geom type="sphere" size="0.2"/></body>
  </worldbody>
</mujoco>
"""

_SETTINGS = rw.RenderSettings(gi=NO_GI, outputs=rw.Outputs.of("rgb"), performance=rw.settings.Performance(traversal="software"))


def _mj() -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of `_XML` with NWORLD worlds."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=NWORLD)
  return mjm, m, d


def _renderer(mjm: mujoco.MjModel, m: mjw.Model, per_world: rw.PerWorld | None = None) -> rw.Renderer[mjw.Data]:
  """A Renderer of `_XML` (`NWORLD` worlds, `_SETTINGS`)."""
  return rw.Renderer(
    rw.integrations.mujoco.scene(mjm, m),
    nworld=NWORLD,
    cameras=["cam"],
    resolution=(16, 12),
    settings=_SETTINGS,
    per_world=per_world,
  )


def test_a_late_copy_of_a_reported_error_does_not_raise_again():
  """An error word whose pinned copy lands after the error was reported and the data fixed (a
  late copy on CUDA) is not reported again: `poll` zeroed the host word, and the late copy then
  raised a second time for data fixed meanwhile."""
  mjm, m, d = _mj()
  r = _renderer(mjm, m, rw.PerWorld(env=["id"]))
  r.render(d)
  r.per_world["env.id"].assign(np.array([0, 7], np.int32))
  r.render(d)  # its check flags env.id
  host = r._watch.validator.host
  reported = host.numpy().copy()
  with pytest.raises(ValueError, match=r"env\.id"):
    r.render(d)
  whole_table(r.per_world["env.id"]).zero_()
  host.numpy()[:] = reported  # the reported frame's copy lands only now
  r.render(d)
  r.render(d)


def _counting[**P](fn: Callable[P, None], calls: list[int]) -> Callable[P, None]:
  """`fn`, appending 1 to `calls` on each call."""

  def counted(*args: P.args, **kwargs: P.kwargs) -> None:
    calls.append(1)
    fn(*args, **kwargs)

  return counted


def test_validate_every_checks_one_frame_in_n():
  """`tuning.validate_every=3`: frames 0, 3, 6, ... check; an error enqueued between them is
  found by the next checked frame and reported by the frame after it."""
  mjm, m, d = _mj()
  r = _renderer(mjm, m, rw.PerWorld(env=["id"]))
  checks: list[int] = []
  check = r._watch.validator.check
  r._watch.validator.check = _counting(check, checks)
  with tuning.override(validate_every=3):
    for _ in range(6):
      r.render(d)
  assert len(checks) == 2
  r.per_world["env.id"].assign(np.array([0, 7], np.int32))
  r.render(d)  # frame 6: checks, flags env.id
  with pytest.raises(ValueError, match=r"env\.id"):
    r.render(d)


def test_poll_does_not_write_the_pinned_word():
  """`poll` only reads the pinned word (a DMA copy may be landing in it on CUDA: a host write mixed
  or lost words)."""
  mjm, m, d = _mj()
  r = _renderer(mjm, m, rw.PerWorld(env=["id"]))
  r.render(d)
  r.per_world["env.id"].assign(np.array([0, 7], np.int32))
  r.render(d)
  host = r._watch.validator.host
  before = host.numpy().copy()
  with pytest.raises(ValueError, match=r"env\.id"):
    r.render(d)
  np.testing.assert_array_equal(host.numpy(), before)


def test_a_move_undone_before_the_poll_still_raises():
  """A shared instance moved for one frame and moved back before the next poll: that frame traced
  stale shared trees, so the word's move raises (with no name: none is away by now), once."""
  mjm, m, d = _mj()
  r = _renderer(mjm, m)
  r.render(d)
  assert r.layout.nshared
  pos = d.geom_xpos.numpy().copy()
  top = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "top")
  moved = pos.copy()
  moved[:, top, 0] += 0.1
  d.geom_xpos.assign(moved)
  r.render(d)  # its check flags the move
  d.geom_xpos.assign(pos)
  with pytest.raises(ValueError, match=r"shared scenery \(back in place by now\) moved or changed"):
    r.render(d)
  r.render(d)
  r.render(d)


_SPOTS = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light type="spot" pos="0 0 1.5" dir="0 0 -1" cutoff="40" intensity="5" range="1000" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" pos="0.3 0 0.2" size="0.2 0.2 0.2"/>
    <camera name="top" pos="0 0 1.8" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("gi", [False, True])
def test_per_world_spot_cone_keeps_per_world_gi(gi: bool) -> None:
  """Declared cone changes keep GI per world (the changed world's tint is recomputed: a relit
  volume), no load-report omission and no stale world-0 probe warning."""
  kw = {"gi": rw.settings.GI(enabled=gi, bounds=((-2.0, -2.0, -0.1), (2.0, 2.0, 1.6)), rays=16, bake_updates=1)}
  s = make_scene(
    _SPOTS,
    nworld=2,
    cam_res=(4, 4),
    per_world=rw.PerWorld(light=["cutoff"]),
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    **kw,
  )
  assert s.r.features.gi is gi
  assert "gpu.gi.per_world_inputs" not in s.scene.report.dropped
  table = s.r.per_world["light.cutoff"]
  values = table.numpy()
  values[1] = 20.0
  table.assign(values)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(3):
      s.render()
  baked = [w for w in caught if "baked" in str(w.message)]
  assert not baked, [str(w.message) for w in caught]
  vol = s.r.debug.gi_volume
  assert (vol is not None and vol.relight is not None) is gi


class _FakeRenderer:
  """A `graph.Replayable` whose graph array is swapped by the test."""

  def __init__(self, array: wp.array) -> None:
    self.device = wp.get_device("cpu")
    self.array = array

  def _graph_version(self) -> int:
    return 0

  def _graph_arrays(self, sim: None) -> dict[str, wp.array]:
    return {"instance.pos": self.array}

  def _before_replay(self, sim: None) -> None:
    pass

  def _replayed(self) -> None:
    pass


class _FakeSegments:
  """Counts replays."""

  def __init__(self) -> None:
    self.launches = 0

  def launch(self) -> None:
    self.launches += 1


def test_graph_replay_rejects_an_array_of_another_shape_at_the_captured_address():
  """A graph bakes in each array's address, shape and strides: an array of another shape at the
  captured address (caching allocators reuse addresses) is stale (only addresses were compared)."""
  base = wp.zeros(64, dtype=wp.vec3, device="cpu")
  r = _FakeRenderer(base)
  segments = _FakeSegments()
  g = graph.RenderGraph(r, segments, None)  # pyrefly: ignore[bad-argument-type]  # a stand-in for the @final CUDA-graph Segments
  g.launch()
  assert segments.launches == 1
  r.array = wp.array(ptr=base.ptr, shape=(32,), dtype=wp.vec3, device="cpu")  # same address, other shape
  with pytest.raises(StaleBinding, match=r"instance\.pos"):
    g.launch()
  assert segments.launches == 1


def test_an_unknown_backend_name_raises_value_error():
  """`backend.named` of an unregistered name raises ValueError naming the registered ones
  (it raised StopIteration)."""
  with pytest.raises(ValueError, match="registered"):
    backend.named("embree")


def test_flags_follow_the_word_bits():
  """`Validator.poll` reads bit `Word.<NAME>` into `Flags.<name>` by position: the two orders stay
  the same."""
  assert Flags._fields == tuple(bit.name.lower() for bit in Word)


def test_a_poll_during_a_landing_copy_reads_one_whole_word(monkeypatch):
  """The pinned copy of the word can land while the host polls it (CUDA: asynchronous, read one
  frame late). A poll in the middle of a copy read the new word's epoch and first slots with the
  old word's last ones: an error reported and fixed meanwhile (a NaN color) came back with the new
  one (an env.id out of range). The word is now one int32, read in one load."""
  from raywarp.core.pipeline.worlds import validator as validate

  mjm, m, d = _mj()
  r = _renderer(mjm, m, rw.PerWorld(material=["color"], env=["id"]))
  r.render(d)
  color = r.per_world["instance.color"]
  good = color.numpy().copy()
  bad = good.copy()
  bad[1, 1, 0] = np.nan
  color.assign(bad)
  r.render(d)  # its check flags the color
  with pytest.raises(ValueError, match="finite"):
    r.render(d)
  color.assign(good)
  r.per_world["env.id"].assign(np.array([0, 7], np.int32))
  validator = r._watch.validator
  polled: list[Flags] = []
  copy = wp.copy

  def landing(dest: wp.array, src: wp.array, *args: object, **kwargs: object) -> None:
    """The word's copy landing in two halves (DMA order), polled in between."""
    if dest is not validator.host:
      copy(dest, src, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]  # forwarded as received
      return
    half = dest.shape[0] // 2
    dest.numpy()[:half] = src.numpy()[:half]
    polled.append(validator.poll())
    dest.numpy()[half:] = src.numpy()[half:]

  monkeypatch.setattr(wp, "copy", landing)
  r.render(d)  # its check flags env.id; its copy lands while polled
  assert polled == [validate.NO_FLAGS] or polled == [validate.NO_FLAGS._replace(env_id=True)]
  with pytest.raises(ValueError, match=r"env\.id"):
    r.render(d)


def test_the_validation_inputs_are_named_as_the_kernel_parameters():
  """`Validator._inputs` names every input of the validation kernel (a positional list of 33, where
  same-typed counts and switches could swap silently); the launch orders them by the kernel's
  parameters (`VALIDATE_INPUTS`)."""
  from raywarp.core.pipeline.worlds import validator as validate

  mjm, m, d = _mj()
  r = _renderer(mjm, m)
  r.render(d)
  validator = r._watch.validator
  snap = validator.snapshot if validator.snapshot is not None else validator._unshared
  named = validator._inputs(None, r.debug.frame(d), snap=snap, npose=0, instance_type=r.scene.instance_type)
  assert set(named) == set(validate.VALIDATE_INPUTS) and len(named) == len(validate.VALIDATE_INPUTS)
  assert validate.VALIDATE_INPUTS[-1] == "epoch" and "flags" not in validate.VALIDATE_INPUTS


def _word(epoch: int, *bits: Word) -> int:
  """A host validation word: `epoch` (wrapped to EPOCH_MASK) above the flag `bits`."""
  from raywarp.core.pipeline.worlds.validator_kernels import EPOCH_MASK, EPOCH_SHIFT

  return ((epoch & EPOCH_MASK) << EPOCH_SHIFT) | sum(int(b) for b in bits)


@pytest.mark.parametrize("start", [5, EPOCH_MASK])
def test_poll_reads_crafted_words(start: int):
  """A word of another epoch is stale (NO_FLAGS, epoch kept); the current epoch's word gives its
  flags and starts a new epoch; the epoch is compared modulo 2^20 (the word's 20 bits wrap)."""
  from raywarp.core.pipeline.worlds import validator as validate

  mjm, m, d = _mj()
  r = _renderer(mjm, m)
  r.render(d)
  validator = r._watch.validator
  validator.epoch = start
  host = validator.host.numpy()
  host[0] = _word(start - 1, Word.ENV_ID, Word.COLOR)
  assert validator.poll() == validate.NO_FLAGS and validator.epoch == start
  host[0] = _word(start, Word.MESH_ID, Word.CASTERS)
  assert validator.poll() == validate.NO_FLAGS._replace(mesh_id=True, casters=True)
  assert validator.epoch == start + 1  # wraps in the word: 2^20 reads as 0
  host[0] = _word(start + 1, Word.EXPOSURE)
  assert validator.poll() == validate.NO_FLAGS._replace(exposure=True)
  host[0] = _word(start + 2)  # current epoch, no flag
  assert validator.poll() == validate.NO_FLAGS and validator.epoch == start + 2


def _host_renderer(per_world: rw.PerWorld) -> rw.Renderer[None]:
  """Two worlds of a host scene: a floor, a box, a point light without a falloff window."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6)))
  b.add_instance("box", material=grey, size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], path="floor")
  b.add_instance("box", material=grey, size=(0.2, 0.2, 0.2), xform=np.c_[np.eye(3), [0.0, 0.0, 0.2]], path="box")
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=10.0, radius=0.05, xform=np.c_[np.eye(3), [0, 0, 2.0]]))
  r = rw.Renderer(
    rw.upload(b.build(), device="cpu"),
    nworld=NWORLD,
    cameras=[rw.CameraSpec(fovy=60.0)],
    resolution=(16, 12),
    settings=_SETTINGS,
    per_world=per_world,
  )
  pose = np.array([0.0, -3.0, 1.0, np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)], np.float32)
  r.camera_pose.assign(np.tile(pose, (NWORLD, 1, 1)))
  return r


def test_an_unbounded_light_range_is_a_valid_per_world_value():
  """A host light without a falloff window has light.range inf (the inverse square law at every
  distance): declaring the field per world rendered it once, then the validation rejected the
  scene's own value. inf and finite windows render; NaN still raises, naming light.range."""
  with _host_renderer(rw.PerWorld(light=["range"])) as r:
    table = r.per_world["light.range"]
    assert np.isinf(table.numpy()).all()
    r.render()
    r.render()
    windows = table.numpy().copy()
    windows[1] = 1.5
    table.assign(windows)
    image = r.render().rgb
    r.render()
    assert image is not None
    rgb = image.numpy()
    assert not np.array_equal(rgb[0], rgb[1])  # world 1's window cuts the floor's far light
    windows[1] = np.nan
    table.assign(windows)
    r.render()
    with pytest.raises(ValueError, match=r"light\.range in \[0, inf\]: world 1 has nan"):
      r.render()


def test_a_range_error_names_the_failing_field_only():
  """An out-of-range per-world value names its field, world, value and count, not every declared
  table's range."""
  with _host_renderer(rw.PerWorld(light=["cutoff", "strength"], material=["base_metalness"])) as r:
    r.render()
    table = r.per_world["light.cutoff"]
    cutoff = table.numpy().copy()
    cutoff[1] = 200.0
    table.assign(cutoff)
    r.render()
    with pytest.raises(ValueError, match=r"light\.cutoff in \[0, 180\]: world 1 has 200 \(1 values\)") as err:
      r.render()
  assert "light.strength" not in str(err.value) and "base_metalness" not in str(err.value)
