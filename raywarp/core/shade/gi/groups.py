"""Which worlds share GI: the world -> group table of a probe volume, found from the data.

A world's GI depends on its *radiance inputs*: the frame's light, material and instance-appearance
rows and its environment (`gi_input`), not on the poses or shapes of the per-world instances
(the probes trace the static scenery only: `baked._bake_dynamic`). Worlds whose inputs are equal
share one set of irradiance maps (a *group*); the static geometry, the probe placement and the
distance maps are shared by every group.

* `digest`: a 64-bit hash per world of its rows of every GI input that has one row per world
  (tables with one row are common to every world and hash to nothing), on the device. The
  instance tables skip the columns of per-world instances (hidden from the probes). Inactive
  lights contribute only their activation flags; their other properties cannot affect GI. Light
  positions hash quantised to a `quantum` of the probe spacing (`POSITION_QUANTUM`) where the
  caller knows the volume (the shared check, the relit tint): a light that follows a body moves a
  little every physics step, below what the probes resolve, and re-tracing every world each
  frame for it cost a third of a manipulation step.
* `classify` (on the bake, one host read): the distinct digests -> K groups. K = 1: one volume,
  as without randomisation. K <= RELIGHT_FRACTION x nworld: K stacked volumes, each baked with its
  inputs. More (`relit`: continuous randomisation, where new values appear on every reset): one
  shared volume tinted per world (`relight.Relight`, which tracks the worlds itself).
* `Groups.track` (every frame, a few tiny launches, no host read): the frame's digests; a world
  whose digest matches another group moves to it (a switch between known variants costs
  nothing). A digest no group holds raises the sticky `overflow` flag, copied to pinned memory
  and read a frame later (`overflowed`): new inputs appear at run time, the worlds are randomised
  continuously, so the renderer re-bakes once as a relit volume (`baked.Baked`).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import device_caps, host_copy

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.pipeline.frame import Frame

wp.set_module_options({"enable_backward": False})

# Frame fields of the probes' radiance inputs besides the material_* / light_* tables. Not the
# exposure: the probes light display-relative emission at the renderer's
# (`frame_structs.probe_environment`), a world's exposure scales its image after them.
_ENVIRONMENT: Final = frozenset({"ambient", "env_id", "env_intensity", "env_yaw"})
# Instance tables among them: (rows, ninstance), their per-world columns skipped.
INSTANCE_INPUTS: Final = frozenset({"instance_color", "instance_material", "instance_shading"})
# Bytes hashed per thread (`hash_rows`): a world's rows hash in parallel chunks (one thread per
# world over a per-world table of thousands of materials took 20 ms a frame at 512 worlds, RTX).
HASH_CHUNK: Final = 256
# The two 32-bit hash lanes' seeds (FNV-1a offset basis, and another odd constant).
_SEED_A: Final = 2166136261
_SEED_B: Final = 0x9E3779B9
# The `Groups.overflow` codes: a world's inputs match no baked group (new variants appear at run
# time), or the inputs every world shares changed since the bake (a re-bake, the same groups).
NEW_VARIANT: Final = 1
SHARED_CHANGED: Final = 2
# One shared volume tinted per world (`relight`) when more than this fraction of the worlds are
# distinct: K stacked volumes cost K maps and cube slabs, and a new variant re-bakes all of them.
RELIGHT_FRACTION: Final = 0.5
# Light positions hash in cells of this fraction of the probe spacing (`hash_positions`): half the
# spacing the probes sample the irradiance at, so a light moving less than that (a light tracking a
# robot's centre of mass) does not re-trace; a move across a cell boundary does (at most half a
# spacing late).
POSITION_QUANTUM: Final = 0.5

type Digests[NWorld: IntVar] = wp.array[wp.uint32, [NWorld, 2]]
type Ids[N: IntVar] = wp.array[wp.int32, [N]]


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Skips[NInstance: IntVar]:
  """The table entries a digest skips: `instances` (ninstance,) 1 for the instances the probes
  ignore (moving / per-world); `lights` (nlight or 0,) 1 for the dynamic lights
  (`shade.gi.dynamic`); `none` an empty table."""

  instances: Ids[NInstance]
  lights: Ids[int]
  none: Ids[int]


def gi_input(name: str) -> bool:
  """Whether Frame field `name` is a radiance input of the probes (see the module doc)."""
  return name.startswith(("material_", "light_")) or name in INSTANCE_INPUTS or name in _ENVIRONMENT


def gi_slot(name: str) -> bool:
  """Whether slot / PerWorld field `name` ("light.color", "env.id", ...) is a radiance input."""
  return gi_input(name.replace(".", "_"))


def shared_inputs(frame: Frame) -> list[tuple[str, AnyDeviceArray]]:
  """(name, array) of the frame's radiance inputs shared by every world (one row)."""
  out = []
  for name, value in vars(frame).items():
    if isinstance(value, wp.array) and gi_input(name) and value.ndim >= 1 and value.shape[0] == 1 and value.size:
      out.append((name, value))
  return sorted(out)


def per_world_inputs(frame: Frame) -> list[tuple[str, AnyDeviceArray]]:
  """(name, array) of the frame's radiance inputs with one row per world (more than one row; an
  empty table, e.g. the (nworld, 0) light rows of a scene without lights, is no input)."""
  out = []
  for name, value in vars(frame).items():
    if isinstance(value, wp.array) and gi_input(name) and value.ndim >= 1 and value.shape[0] > 1 and value.size > 0:
      out.append((name, value))
  return sorted(out)


@wp.func
def _mix(
  # In:
  h: wp.uint32,
  byte: wp.uint32,
  prime: wp.uint32,
) -> wp.uint32:
  """One FNV-1a step."""
  return (h ^ byte) * prime


@wp.func
def _finish(
  # In:
  h: wp.uint32,
) -> wp.uint32:
  """MurmurHash3's finaliser: every input bit flips about half the output bits."""
  h = h ^ (h >> wp.uint32(16))
  h = h * wp.uint32(0x85EBCA6B)
  h = h ^ (h >> wp.uint32(13))
  h = h * wp.uint32(0xC2B2AE35)
  return h ^ (h >> wp.uint32(16))


@wp.kernel
def hash_rows(
  # In:
  data: wp.array2d[wp.uint8],
  element: int,
  skip: wp.array1d[wp.int32],
  active: wp.array2d[bool],
  light_property: bool,
  salt: int,
  # Out:
  digests: wp.array2d[wp.uint32],
):
  """Adds the hash of chunk k (HASH_CHUNK bytes) of world w's row of `data` (`element` bytes per
  entry) to its digest, skipping the entries e with skip[e] != 0 (an empty `skip`: none). One
  thread per (world, chunk): the chunks hash in parallel; their finalised hashes, seeded by the
  chunk and the table (`salt`), are summed (mod 2^32: the same digest whatever the order)."""
  w, k = wp.tid()
  start = k * wp.static(HASH_CHUNK)
  end = wp.min(start + wp.static(HASH_CHUNK), data.shape[1])
  seed = wp.uint32(salt * 65599 + k)
  a = _mix(wp.uint32(_SEED_A), seed, wp.uint32(16777619))
  b = _mix(wp.uint32(_SEED_B), seed, wp.uint32(0x01000193 ^ 0x5BD1E995))
  nskip = skip.shape[0]
  for i in range(start, end):
    e = i // element
    if e < nskip:
      if skip[e] != 0:
        continue
    # A disabled light cannot affect GI. Hash its activation flag separately,
    # so enabling it still invalidates the group and includes all its properties.
    if light_property and e < active.shape[1]:
      if not active[w % active.shape[0], e]:
        continue
    byte = wp.uint32(data[w, i])
    a = _mix(a, byte, wp.uint32(16777619))
    b = _mix(b, byte + wp.uint32(i), wp.uint32(0x01000193 ^ 0x5BD1E995))
  wp.atomic_add(digests, w, 0, _finish(a))
  wp.atomic_add(digests, w, 1, _finish(b))


@wp.kernel
def hash_positions(
  # In:
  pos: wp.array2d[wp.vec3],
  active: wp.array2d[bool],
  skip: wp.array1d[wp.int32],
  inv_quantum: float,
  salt: int,
  # Out:
  digests: wp.array2d[wp.uint32],
):
  """Adds the hash of light e's position in row w, as integer cells of 1 / `inv_quantum`, to the
  row's digest (inactive lights and those with skip[e] != 0: nothing, as `hash_rows`); one thread
  per (row, light), summed."""
  w, e = wp.tid()
  if not active[w % active.shape[0], e]:
    return
  if e < skip.shape[0]:
    if skip[e] != 0:
      return
  q = pos[w, e] * inv_quantum
  a = _mix(wp.uint32(_SEED_A), wp.uint32(salt * 65599 + e), wp.uint32(16777619))
  b = _mix(wp.uint32(_SEED_B), wp.uint32(salt * 65599 + e), wp.uint32(0x01000193 ^ 0x5BD1E995))
  for k in range(3):
    c = wp.uint32(wp.int32(wp.floor(q[k])))
    a = _mix(a, c, wp.uint32(16777619))
    b = _mix(b, c + wp.uint32(k), wp.uint32(0x01000193 ^ 0x5BD1E995))
  wp.atomic_add(digests, w, 0, _finish(a))
  wp.atomic_add(digests, w, 1, _finish(b))


@wp.kernel
def seed_digests(
  # Out:
  digests: wp.array2d[wp.uint32],
):
  """The hash lanes' seeds."""
  w = wp.tid()
  digests[w, 0] = wp.uint32(_SEED_A)
  digests[w, 1] = wp.uint32(_SEED_B)


def _bytes(a: AnyDeviceArray, rows: int) -> wp.array[wp.uint8, [int, int]]:
  """A logical (rows, row bytes) view of contiguous `a`, excluding reserved allocation bytes.

  No copy: the frame holds `a`. Capacity may exceed the table size and is not a row stride.
  """
  if not a.is_contiguous:
    raise ValueError("GI digests read contiguous tables only")
  row = a.size * wp.types.type_size_in_bytes(a.dtype) // rows
  return wp.array(ptr=a.ptr, dtype=wp.uint8, shape=(rows, row), device=a.device, copy=False)


def digest[NWorld: IntVar, NInstance: IntVar](
  frame: Frame,
  nworld: int,
  skips: Skips[NInstance],
  out: Digests[NWorld],
  quantum: float = 0.0,
  *,
  names: frozenset[str] | None = None,
) -> Digests[NWorld]:
  """Writes the (nworld, 2) digests of the worlds' radiance inputs into `out` (device launches
  only). `skips`: what the probes ignore (`Skips`); `quantum` > 0: light positions in cells of
  that size [m] (module doc), 0: exact; `names`: only these Frame fields (their `nworld` rows,
  whatever their tables' role: e.g. an authored frame's one-row copies of per-world tables), with
  every light's properties hashed whatever its activity (the relight's non-linear inputs: a light
  switched off keeps them, `slots` scales it to zero)."""
  _run(digest_launches(frame, nworld, skips, out, quantum, names=names))
  return out


def digest_launches[NWorld: IntVar, NInstance: IntVar](
  frame: Frame,
  nworld: int,
  skips: Skips[NInstance],
  out: Digests[NWorld],
  quantum: float = 0.0,
  *,
  names: frozenset[str] | None = None,
) -> list[wp.Launch]:
  """`digest`'s launches, recorded and not run (`Replay`)."""
  if names is None:
    tables = [(name, a) for name, a in per_world_inputs(frame) if a.shape[0] == nworld and a.size]
    return _hash_launches(frame, tables, nworld, skips, out, quantum)
  tables = [(name, getattr(frame, name)) for name in sorted(names)]
  tables = [(name, a) for name, a in tables if a.shape[0] == nworld and a.size]
  return _hash_launches(frame, tables, nworld, skips, out, quantum, every_light=True)


def shared_digest[NInstance: IntVar](
  frame: Frame, live: frozenset[str], skips: Skips[NInstance], out: Digests[int], quantum: float = 0.0
) -> Digests[int]:
  """Writes the (1, 2) digest of the radiance inputs shared by every world (`shared_inputs`)
  that are `live` (Frame field names of arrays written in place: a binding's model tables) into
  `out` (device launches only; `quantum`: `digest`'s)."""
  _run(_hash_launches(frame, [(n, a) for n, a in shared_inputs(frame) if n in live], 1, skips, out, quantum))
  return out


def record(
  kernel: wp.Kernel | Callable[..., None],
  dim: int | tuple[int, ...],
  inputs: Sequence[object],
  outputs: Sequence[object],
  device: wp.Device | None,
) -> wp.Launch:
  """The recorded launch of `kernel` (not run)."""
  cmd = wp.launch(kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), device=device, record_cmd=True)
  if cmd is None:
    raise RuntimeError("wp.launch(record_cmd=True) returned no recorded launch")
  return cmd


def _run(launches: list[wp.Launch]) -> None:
  """Runs recorded launches in order."""
  for cmd in launches:
    cmd.launch()


def _hash_launches[NInstance: IntVar](
  frame: Frame,
  tables: list[tuple[str, AnyDeviceArray]],
  rows: int,
  skips: Skips[NInstance],
  out: Digests[int],
  quantum: float,
  *,
  every_light: bool = False,
) -> list[wp.Launch]:
  """The recorded launches writing the (rows, 2) digests of `tables` (each `rows` rows) into `out`
  (`hash_rows`; light positions `hash_positions` with `quantum` > 0; the inactive lights'
  properties skipped unless `every_light`). A launch holds the address of a byte view of its
  table (`_bytes`): valid while the frame holds the table."""
  dev = out.device
  out_launches = [record(seed_digests, rows, [], [out], dev)]
  nlight = frame.light_active.shape[1]
  active = _every_light(nlight, str(frame.light_active.device)) if every_light else _activity(frame.light_active, rows)
  for salt, (name, a) in enumerate(tables):
    lights = name.startswith("light_")
    if name == "light_pos" and quantum > 0.0:
      inputs = [a, active, skips.lights, 1.0 / quantum, salt]
      out_launches.append(record(hash_positions, a.shape, inputs, [out], dev))
      continue
    element = wp.types.type_size_in_bytes(a.dtype)
    skip = skips.instances if name in INSTANCE_INPUTS else skips.lights if lights else skips.none
    light_property = lights and name != "light_active"
    view = _bytes(a, rows)
    dim = (rows, max(1, -(-view.shape[1] // HASH_CHUNK)))
    out_launches.append(record(hash_rows, dim, [view, element, skip, active, light_property, salt], [out], dev))
  return out_launches


def _activity(light_active: wp.array[bool, [int, int]], rows: int) -> wp.array[bool, [int, int]]:
  """The light activity that masks the light properties of `rows`-row digests: the frame's when
  it has one row or one per digest row, else every light active. A digest of the shared inputs
  (one row) over a per-world activity table would read world 0's: its shared light tables would
  then change digest with world 0's activity (a re-bake of the shared volume whenever a reset
  toggled one of world 0's lights); each world's activity is its own digest's input."""
  if light_active.shape[0] in (1, rows):
    return light_active
  return _every_light(light_active.shape[1], str(light_active.device))


@functools.cache
def _every_light(nlight: int, device: str) -> wp.array[bool, [int, int]]:
  """A (1, nlight) activity table of every light active (`_activity`)."""
  return wp.ones((1, nlight), dtype=bool, device=device)


@final
class Replay:
  """The recorded launches of a per-frame check (`Groups.track`, `relight.Relight.track`),
  replayed while the frame holds the same arrays and the skips are the same: packing a launch's
  arguments is most of its host cost, and these checks run every eager frame (64 worlds x 128^2:
  0.74 ms of a 1.70 ms frame before, RTX PRO 6000). Same launches, same order: same digests."""

  def __init__(self) -> None:
    """Nothing recorded."""
    self._frame: Frame | None = None
    self._skips: object = None
    self._launches: list[wp.Launch] = []

  def run(self, frame: Frame, skips: object, record: Callable[[], list[wp.Launch]]) -> bool:
    """Runs the launches of `frame` (`record`ed again when its arrays or `skips` changed: then
    True)."""
    fresh = self._skips is not skips or not frame.same(self._frame)
    if fresh:
      self._launches = record()
      self._frame, self._skips = frame, skips
    _run(self._launches)
    return fresh


@wp.kernel
def compare_shared(
  # In:
  now: wp.array2d[wp.uint32],
  baked: wp.array2d[wp.uint32],
  # Out:
  overflow: wp.array1d[wp.int32],
):
  """Raises SHARED_CHANGED when the shared inputs' digest differs from the bake's."""
  if overflow[0] == 0 and (now[0, 0] != baked[0, 0] or now[0, 1] != baked[0, 1]):
    overflow[0] = wp.static(SHARED_CHANGED)


@wp.kernel
def clear_reps(
  # In:
  nworld: int,
  # Out:
  rep: wp.array1d[wp.int32],
):
  """No representative world yet (nworld: none)."""
  rep[wp.tid()] = nworld


@wp.kernel
def assign(
  # In:
  digests: wp.array2d[wp.uint32],
  group_digest: wp.array2d[wp.uint32],
  # Out:
  world_group: wp.array1d[wp.int32],
  rep: wp.array1d[wp.int32],
  overflow: wp.array1d[wp.int32],
):
  """World w's group this frame: the group holding its digest (none: the sticky overflow flag,
  the world keeps its group)."""
  w = wp.tid()
  a = digests[w, 0]
  b = digests[w, 1]
  for g in range(group_digest.shape[0]):
    if group_digest[g, 0] == a and group_digest[g, 1] == b:
      world_group[w] = g
      wp.atomic_min(rep, g, w)
      return
  overflow[0] = wp.static(NEW_VARIANT)


type HostIds = np.ndarray[[int], np.dtype[np.int32]]
type HostDigests = np.ndarray[[int, 2], np.dtype[np.uint32]]


def relit(digests: HostDigests) -> bool:
  """Whether worlds of these (nworld, 2) digests get one shared volume tinted per world
  (`relight.Relight`): more than RELIGHT_FRACTION of them distinct."""
  return len(np.unique(digests, axis=0)) > RELIGHT_FRACTION * len(digests)


def classify(digests: HostDigests) -> tuple[HostIds, HostDigests, HostIds]:
  """(world -> group (nworld,), group digests (K, 2), lowest world of each group (K,)) of host
  digests (nworld, 2): groups in order of their lowest world (world 0 is in group 0)."""
  keys = np.left_shift(digests[:, 0].astype(np.uint64), 32) | digests[:, 1].astype(np.uint64)
  _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
  order = np.argsort(first)
  rank = np.empty_like(order)
  rank[order] = np.arange(len(order))
  firsts = first[order].astype(np.int32)
  return rank[inverse.reshape(-1)].astype(np.int32), digests[firsts], firsts


@final
class Groups:
  """The world -> group table of a probe volume and its tracking state (see the module doc).

  Attributes:
    ngroup: K, the stacked irradiance volumes.
    tracked: the inputs can differ per world (digests are tracked every frame); False: one row
      for every world, never re-checked (K = 1).
    world_group: (nworld,) group of each world ((1,) [0] untracked).
    group_digest: (K, 2) the inputs digest of each group.
    rep: (K,) the lowest world of each group this frame (nworld: none; the probe rays read its
      inputs).
    overflow: (1,) sticky: NEW_VARIANT (a world's inputs match no group) or SHARED_CHANGED (the
      inputs every world shares changed since the bake: `bake_shared`); `host_overflow` its
      pinned copy.
  """

  def __init__(self, nworld: int, *, digests: HostDigests | None, device: wp.Device) -> None:
    """Groups of host `digests` (nworld, 2) (`digest`); None: untracked, one group."""
    self.nworld = nworld
    self.device = device
    self.tracked = digests is not None
    if digests is None:
      world_group, group_digest, rep = np.zeros(1, np.int32), np.zeros((1, 2), np.uint32), np.zeros(1, np.int32)
    else:
      world_group, group_digest, rep = classify(digests)
    self.ngroup = len(group_digest)
    self.world_group = wp.array(world_group, dtype=wp.int32, device=device)
    self.group_digest = wp.array(group_digest, dtype=wp.uint32, device=device)
    self.rep = wp.array(rep, dtype=wp.int32, device=device)
    self.overflow = wp.zeros(1, dtype=wp.int32, device=device)
    self.host_overflow = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=device_caps(device).pinned)
    self.digests = wp.zeros((nworld, 2), dtype=wp.uint32, device=device)
    self.shared = wp.zeros((1, 2), dtype=wp.uint32, device=device)
    self.shared_baked = wp.zeros((1, 2), dtype=wp.uint32, device=device)
    # Whether a frame was checked (`track`): no pinned read before (periodic untracked volumes).
    self._checked = False
    # The shared inputs at the bake (`bake_shared`): the live names, the arrays' identities, and
    # whether one was replaced since.
    self._live: frozenset[str] = frozenset()
    self._shared_arrays: dict[str, int | None] = {}
    self._replaced = False
    self._quantum = 0.0
    # `track`'s recorded launches, without and with the shared check.
    self._replays = {False: Replay(), True: Replay()}

  @property
  def nbytes(self) -> int:
    """Device memory of the tables."""
    arrays = (self.world_group, self.group_digest, self.rep, self.digests)
    return int(sum(a.capacity for a in arrays))

  def bake_shared[NInstance: IntVar](
    self, frame: Frame, live: frozenset[str], skips: Skips[NInstance], *, quantum: float
  ) -> None:
    """Records the inputs every world shares at the bake: the identities of their arrays, and the
    digest of the `live` ones (written in place: `shared_digest`, light positions in cells of
    `quantum` [m]; device launches)."""
    self._live = live
    self._quantum = quantum
    self._replays = {False: Replay(), True: Replay()}
    self._shared_arrays = {name: a.ptr for name, a in shared_inputs(frame)}
    shared_digest(frame, live, skips, self.shared_baked, quantum)

  def track[NInstance: IntVar](self, frame: Frame, skips: Skips[NInstance], *, shared: bool) -> None:
    """The frame's groups (see the module doc) and, with `shared` (a frozen bake: periodic
    updates trace the current shared inputs anyway), the check of the shared inputs against the
    bake's: device launches (`Replay`) and one copy to pinned memory."""
    if not (shared or self.tracked):
      return
    self._checked = True
    replay = self._replays[shared]
    if replay.run(frame, skips, lambda: self._launches(frame, skips, shared=shared)) and shared:
      # An array replaced (another identity) is seen on the host; one written in place by its
      # digest (only the live ones: an upload's tables are not written in place).
      if {name: a.ptr for name, a in shared_inputs(frame)} != self._shared_arrays:
        self._replaced = True
    wp.copy(self.host_overflow, self.overflow)

  def _launches[NInstance: IntVar](self, frame: Frame, skips: Skips[NInstance], *, shared: bool) -> list[wp.Launch]:
    """`track`'s launches of `frame` (recorded, not run)."""
    dev = self.device
    out: list[wp.Launch] = []
    if shared:
      live = [(n, a) for n, a in shared_inputs(frame) if n in self._live]
      out += _hash_launches(frame, live, 1, skips, self.shared, self._quantum)
      out.append(record(compare_shared, 1, [self.shared, self.shared_baked], [self.overflow], dev))
    if self.tracked:
      out += digest_launches(frame, self.nworld, skips, self.digests)
      out.append(record(clear_reps, self.ngroup, [self.nworld], [self.rep], dev))
      groups = [self.world_group, self.rep, self.overflow]
      out.append(record(assign, self.nworld, [self.digests, self.group_digest], groups, dev))
    return out

  def overflow_code(self) -> int:
    """An earlier frame's overflow code (0: none; NEW_VARIANT, SHARED_CHANGED; a pinned read, no
    synchronisation)."""
    if self._replaced:
      return SHARED_CHANGED
    return int(host_copy(self.host_overflow)[0]) if self._checked else 0


@wp.func
def group_used(
  # In:
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  nworld: int,
  g: int,
) -> bool:
  """Whether group g updates now: the gate is open and a world uses it."""
  return gate[0] != 0 and rep[g] < nworld
