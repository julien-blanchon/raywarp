"""Per-world relighting of one shared GI volume: the GI of continuous domain randomisation.

When the worlds' radiance inputs (materials, lights, environment) are mostly distinct, one probe
volume per world would cost its maps and ambient cube per world (0.8-2.7 GB at 512 worlds) and a
full re-bake per reset world (about a frame per world). Instead every world reads the shared
volume, baked once with world 0's inputs, times its own tint (`probes.RelightField`): an RGB
ratio per coarse cell of the probe box (cells of at least CELL_PROBES probe spacings, at most
MAX_CELLS of them) and axis direction (an ambient cube of ratios, blended by the squared shading
normal), interpolated trilinearly at the shading point.

A world's ratio in a cell and direction is the sum of its radiance along SAMPLES x RAYS fixed
probe rays of the cell (SAMPLES probes at the cell's 2 x 2 x 2 sub-positions, RAYS fixed
spherical Fibonacci directions, fixed seeds), weighted by the direction's cosine lobe, over the
same sum of the shared bake's (the authored inputs: world 0's at the first GI bake, stored as the
reference). Directional: a surface facing away from a wall whose colour changed is not tinted by
it (an isotropic ratio has twice the error of no tint on a swapped-walls room). The rays are
shaded like the probe updates (direct light, emission, the environment on a miss, and the bounce
read from the shared cube times the world's own tint at the hit, so PASSES passes converge the
tint's own inter-reflection). The same rays in both sums cancel most of their noise: the ratio
carries how the world's inputs change the indirect light (its wall colours' bleed, its lights'
strengths, its sky), the shared volume the geometry (occlusion, walls, the spatial falloff). The
shared volume is baked with the authored inputs whenever it is (re-)baked
(`baked.Baked._authored_frame`), so a world back at them has ratio 1 and renders as before: no
state of earlier episodes.

Change detection: the worlds' input digests (`groups.digest`) every frame; a world whose digest
changed (a reset) gets its ratio recomputed in that frame (PASSES passes of its rays, from ratio
1: a world reset to earlier inputs renders as before), unless only its lights' colours,
intensities, activity or its environment intensity changed since the bake or since its last
re-light by rays (`slots`: a recombination, no rays). gi.relight="on_request" holds the re-lights
by rays until the world is requested (`Relight.request`, e.g. at an episode reset): the world
keeps its tint meanwhile; a request waits for its world's next change. Every launch is gated on
the device (no host read): graph-capturable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.shade.gi import cube as cube_lib
from raywarp.core.shade.gi import slots as slots_lib
from raywarp.core.shade.gi.groups import (
  POSITION_QUANTUM,
  HostDigests,
  Replay,
  Skips,
  digest,
  digest_launches,
  per_world_inputs,
  record,
)
from raywarp.core.shade.gi.probes import RelightField, relight_texel
from raywarp.core.shade.gi.rebakes import count_relights
from raywarp.core.shade.gi.slots import AUTHORED, TRACED
from raywarp.core.shade.lights.punctual import Light, LightHot

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.shade.gi.volume import ProbeInputs, ProbeVolume
  from raywarp.core.shade.lights.punctual import Lights

wp.set_module_options({"enable_backward": False})

# Smallest cell side, in probe spacings: a cell holds the 2 x 2 x 2 sample probes.
CELL_PROBES: Final = 2
# Most cells per world: 6 x 16 B of tint (staging + texture) and 6 SAMPLES x 12 B of sums each
# (44 MB / 512 worlds at 128 cells), and SAMPLES x RAYS x PASSES rays per cell for a reset world
# (65k rays).
MAX_CELLS: Final = 128
# Sample probes per cell (its 2 x 2 x 2 sub-positions) and fixed rays per sample probe.
SAMPLES: Final = 8
RAYS: Final = 32
# Passes of a world's ratio: the first bounces the shared volume's light off the world's
# surfaces, the second its own tint once more.
PASSES: Final = 2
# Largest ratio: a cell the reference barely lights cannot multiply the shared light without bound.
RATIO_MAX: Final = 16.0
# Axis directions of the ratios (+x, -x, +y, -y, +z, -z).
FACES: Final = 6
# A reference channel below this fraction of its cell's luminance keeps ratio 1 (nothing to tint).
_DARK_FRACTION: Final = 1.0e-4


def cell_grid(counts: tuple[int, int, int]) -> tuple[tuple[int, int, int], int]:
  """(cells per axis, probes per cell side) over a probe grid of `counts` probes per axis:
  cells of CELL_PROBES probes, grown until at most MAX_CELLS."""
  side = CELL_PROBES
  while True:
    cells = tuple(max(1, math.ceil(c / side)) for c in counts)
    if math.prod(cells) <= MAX_CELLS:
      return (cells[0], cells[1], cells[2]), side
    side += 1


def sample_probes(
  counts: tuple[int, int, int], cells: tuple[int, int, int], side: int
) -> np.ndarray[[int, int], np.dtype[np.int32]]:
  """(ncell, SAMPLES) probe indices: per cell, the probes nearest its 2 x 2 x 2 sub-positions
  (a quarter and three quarters of the cell side), clamped to the grid without its outer layer
  where it has more than two probes: the default probe box is padded by one spacing beyond the
  scenery (`baked._probe_grid`), so that layer lies outside a room's walls, where the sun and
  the sky light the walls' outsides (a cell along a wall would take their ratio)."""
  out = np.empty((math.prod(cells), SAMPLES), np.int32)
  nx, ny, _ = counts
  lo = [1 if n > 2 else 0 for n in counts]
  hi = [n - 2 if n > 2 else n - 1 for n in counts]
  for cz in range(cells[2]):
    for cy in range(cells[1]):
      for cx in range(cells[0]):
        c = (cz * cells[1] + cy) * cells[0] + cx
        for s in range(SAMPLES):
          sub = (s & 1, (s >> 1) & 1, (s >> 2) & 1)
          x, y, z = (
            min(max(cc * side + ((2 * k + 1) * side) // 4, lo[a]), hi[a])
            for a, (cc, k) in enumerate(zip((cx, cy, cz), sub, strict=True))
          )
          out[c, s] = (z * ny + y) * nx + x
  return out


def fibonacci(n: int) -> np.ndarray[[int, 3], np.dtype[np.float32]]:
  """(n, 3) spherical Fibonacci directions (`probes.fibonacci_direction`)."""
  i = np.arange(n).astype(np.float64)
  golden = 0.6180339887498949
  phi = 2.0 * np.pi * (i * golden - np.floor(i * golden))
  cos_t = 1.0 - (2.0 * i + 1.0) / n
  sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
  return np.stack([np.cos(phi) * sin_t, np.sin(phi) * sin_t, cos_t], axis=1).astype(np.float32)


@wp.kernel
def mark_changed_linear(
  # In:
  digests: wp.array2d[wp.uint32],
  nonlinear: wp.array2d[wp.uint32],
  authored: wp.array2d[wp.uint32],
  base_nonlinear: wp.array2d[wp.uint32],
  hold: wp.array1d[wp.int32],
  requests: wp.array1d[wp.int32],
  # Out:
  world_digest: wp.array2d[wp.uint32],
  changed: wp.array1d[wp.int32],
  linear: wp.array1d[wp.int32],
):
  """World w: `linear` AUTHORED when its inputs but `slots.LINEAR_INPUTS` are the authored ones,
  TRACED when they are the ones of its last ray re-light (its tint may be a recombination of the
  slots' shares, no rays: `slots.split_linear` decides), else 0; `changed` when its inputs
  digest differs from its last one (the digest stored), unless a re-light by rays waits for a
  request (`hold`: gi.relight="on_request", and no `requests`[w]): the world keeps its tint and
  its old digest until then."""
  w = wp.tid()
  lin = 0
  if nonlinear[w, 0] == authored[0, 0] and nonlinear[w, 1] == authored[0, 1]:
    lin = wp.static(AUTHORED)
  elif nonlinear[w, 0] == base_nonlinear[w, 0] and nonlinear[w, 1] == base_nonlinear[w, 1]:
    lin = wp.static(TRACED)
  linear[w] = lin
  if world_digest[w, 0] != digests[w, 0] or world_digest[w, 1] != digests[w, 1]:
    if lin != 0 or hold[0] == 0 or requests[w] != 0:
      world_digest[w, 0] = digests[w, 0]
      world_digest[w, 1] = digests[w, 1]
      changed[w] = 1


@wp.func
def _store(
  # Model:
  field: RelightField,
  # In:
  w: int,
  i: int,
  deviation: wp.vec3,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """Writes ratio - 1 `deviation` of world w's entry i (cell i / FACES, face i % FACES) into the
  tint's staging texels."""
  c = i // wp.static(FACES)
  n = field.counts
  x = c % n[0]
  y = (c // n[0]) % n[1]
  z = c // (n[0] * n[1])
  at = relight_texel(field, w * wp.static(FACES) + i % wp.static(FACES))
  for k in range(3):
    texels[at[1] + z, at[0] + y, x, k] = wp.float16(deviation[k])


@wp.kernel
def restart(
  # Model:
  field: RelightField,
  # In:
  gate: wp.array1d[int],
  changed: wp.array1d[wp.int32],
  # Out:
  texels: wp.array4d[wp.float16],
):
  """Ratio 1 for the worlds that recompute theirs (the first pass bounces the shared light)."""
  w, i = wp.tid()
  if gate[0] != 0 or changed[w] != 0:
    _store(field, w, i, wp.vec3(0.0), texels)


@wp.kernel
def resolve(
  # Model:
  field: RelightField,
  # In:
  sums: wp.array3d[wp.vec3],
  reference: wp.array3d[wp.vec3],
  gate: wp.array1d[int],
  changed: wp.array1d[wp.int32],
  # Out:
  texels: wp.array4d[wp.float16],
  traced: wp.array2d[wp.vec3],
):
  """World w's ratio i (cell i / FACES, face i % FACES): its sample sums over the reference's,
  per channel (fixed order: bit-identical for equal inputs), into the tint and `traced` (what a
  TRACED recombination scales)."""
  w, i = wp.tid()
  if gate[0] == 0 and changed[w] == 0:
    return
  c = i // wp.static(FACES)
  f = i % wp.static(FACES)
  mine = wp.vec3(0.0)
  ref = wp.vec3(0.0)
  for s in range(sums.shape[2] // wp.static(FACES)):
    mine += sums[w, c, s * wp.static(FACES) + f]
    ref += reference[0, c, s * wp.static(FACES) + f]
  dark = wp.static(_DARK_FRACTION) * (ref[0] + ref[1] + ref[2]) / 3.0
  out = wp.vec3(0.0)
  for k in range(3):
    if ref[k] > dark and ref[k] > 0.0:
      out[k] = wp.clamp(mine[k] / ref[k], 0.0, wp.static(RATIO_MAX)) - 1.0
  traced[w, i] = out + wp.vec3(1.0)
  _store(field, w, i, out, texels)


@wp.kernel
def store_linear(
  # Model:
  field: RelightField,
  # In:
  ratio: wp.array2d[wp.vec3],
  linear: wp.array1d[wp.int32],
  gate: wp.array1d[int],
  # Out:
  texels: wp.array4d[wp.float16],
):
  """World w's recombined ratio of entry i into the tint (the worlds `linear` only; TRACED ones
  not with the gate open: their rays were just traced against new shared maps)."""
  w, i = wp.tid()
  lin = linear[w]
  if lin == wp.static(AUTHORED) or (lin == wp.static(TRACED) and gate[0] == 0):
    _store(field, w, i, ratio[w, i] - wp.vec3(1.0), texels)


@wp.kernel
def remember(
  # In:
  gate: wp.array1d[int],
  changed: wp.array1d[wp.int32],
  nonlinear: wp.array2d[wp.uint32],
  power: wp.array2d[wp.vec3],
  # Out:
  base_nonlinear: wp.array2d[wp.uint32],
  base_power: wp.array2d[wp.vec3],
):
  """The inputs of world w's ratio when it was just traced by rays (`changed`, or the gate open):
  its non-linear digest and slot powers, what a TRACED recombination starts from."""
  w = wp.tid()
  if gate[0] == 0 and changed[w] == 0:
    return
  base_nonlinear[w, 0] = nonlinear[w, 0]
  base_nonlinear[w, 1] = nonlinear[w, 1]
  for k in range(power.shape[1]):
    base_power[w, k] = power[w, k]


@wp.kernel
def settle(
  # Out:
  changed: wp.array1d[wp.int32],
  requests: wp.array1d[wp.int32],
):
  """The worlds' ratios are current; the requests of the worlds that re-lit are served (a request
  waits for its world's next change: one made the frame before a table write still counts)."""
  w = wp.tid()
  if changed[w] != 0:
    requests[w] = 0
  changed[w] = 0


@wp.kernel
def _request(
  # In:
  worlds: wp.array1d[bool],
  # Out:
  requests: wp.array1d[wp.int32],
):
  """World w may re-light by rays on the next frame (`Relight.request`)."""
  w = wp.tid()
  if worlds[w]:
    requests[w] = 1


@final
class Relight:
  """The per-world tint of a shared volume and its change tracking (see the module doc).

  Args:
    vol: the shared, placed volume (its `struct.relight` is set to this tint's field).
    digests: (nworld, 2) the worlds' input digests at the bake (`groups.digest`).
    relights: (nworld,) the re-bake log's per-world counter (`rebakes.RebakeLog`), counting the
      re-lights by rays of changed worlds.
    slot_of: (nlight,) each light's slot (`slots.assign`); nslot: the light slots.
  """

  def __init__(
    self,
    vol: ProbeVolume,
    digests: HostDigests,
    relights: wp.array[wp.int32, [int]],
    slot_of: np.ndarray[[int], np.dtype[np.int32]],
    nslot: int,
  ) -> None:
    """Sizes the cells over the volume's probe box and allocates every buffer (every world
    pending: `bake` computes them)."""
    dev = vol.device
    self.device = dev
    self.nworld = len(digests)
    counts = vol.counts
    self.cells, side = cell_grid(counts)
    self.ncell = math.prod(self.cells)
    self.samples = wp.array(sample_probes(counts, self.cells, side), dtype=wp.int32, device=dev)
    self.dirs = wp.array(fibonacci(RAYS), dtype=wp.vec3, device=dev)
    self.spread = math.sqrt(4.0 * math.pi / RAYS)
    field = RelightField()
    tiles, height, depth = cube_lib.layout(self.cells, self.nworld * FACES)
    self.texels = wp.zeros((depth, height, self.cells[0], 4), dtype=wp.float16, device=dev)
    # Derived from the bake and the worlds' inputs (hashed as arrays by the bake keys): named by
    # its layout only.
    field.texture = bake_cache.register_texture(cube_lib.texture3d(self.texels, linear=True), f"gi-relight:{self.texels.shape}")
    field.rows = self.nworld
    # The cells tile the probe cells' box: probe (0, 0, 0) sits half a spacing inside the box.
    step = np.asarray(vol.spacing, np.float64)
    field.origin = wp.vec3(*vol.box[0].tolist())
    field.inv_cell = wp.vec3(*(1.0 / (step * side)).tolist())
    field.counts = wp.vec3i(*self.cells)
    field.tiles = tiles
    self.field = field
    # Light positions hash in cells of this size [m] (`groups.POSITION_QUANTUM`).
    self.quantum = POSITION_QUANTUM * float(np.min(step))
    vol.struct.relight = field
    self.reference = wp.zeros((1, self.ncell, SAMPLES * FACES), dtype=wp.vec3, device=dev)
    self.sums = wp.zeros((self.nworld, self.ncell, SAMPLES * FACES), dtype=wp.vec3, device=dev)
    self.world_digest = wp.array(digests, dtype=wp.uint32, device=dev)
    self.digests = wp.zeros((self.nworld, 2), dtype=wp.uint32, device=dev)
    self.changed = wp.ones(self.nworld, dtype=wp.int32, device=dev)
    # gi.relight: 1 = re-lights by rays wait for `request` (`set_policy`); the requested worlds.
    self.hold = wp.zeros(1, dtype=wp.int32, device=dev)
    self._held = False
    self.requests = wp.zeros(self.nworld, dtype=wp.int32, device=dev)
    self.relights = relights
    self.closed = wp.zeros(1, dtype=int, device=dev)
    self.always = wp.ones(1, dtype=int, device=dev)
    self._slots(slot_of, nslot)

  def _slots(self, slot_of: np.ndarray[[int], np.dtype[np.int32]], nslot: int) -> None:
    """The linear re-light's buffers (`slots`): the slot of each light, per slot a one-row tint
    field of its shares (what its own bounces read), the slots' sample sums and shares, the
    worlds' and the authored slot powers, the recombined ratios and the worlds' linear flags."""
    dev, field = self.device, self.field
    self.slot_of = wp.array(np.asarray(slot_of if len(slot_of) else [slots_lib.NO_SLOT], np.int32), dtype=wp.int32, device=dev)
    self.nslot = nslot + 1  # + the rest slot
    tiles, height, depth = cube_lib.layout(self.cells, FACES)
    self.slot_texels = [
      wp.full((depth, height, self.cells[0], 4), -1.0, dtype=wp.float16, device=dev) for _ in range(self.nslot)
    ]
    self.slot_fields: list[RelightField] = []
    for k, texels in enumerate(self.slot_texels):
      f = RelightField()
      f.texture = bake_cache.register_texture(cube_lib.texture3d(texels, linear=True), f"gi-slot{k}:{texels.shape}")
      f.rows, f.origin, f.inv_cell, f.counts, f.tiles = 1, field.origin, field.inv_cell, field.counts, tiles
      self.slot_fields.append(f)
    entries = self.ncell * FACES
    self.slot_sums = wp.zeros((self.nslot, self.ncell, SAMPLES * FACES), dtype=wp.vec3, device=dev)
    self.shares = wp.zeros((self.nslot, entries), dtype=wp.vec3, device=dev)
    self.power = wp.zeros((self.nworld, self.nslot), dtype=wp.vec3, device=dev)
    self.authored_power = wp.zeros((1, self.nslot), dtype=wp.vec3, device=dev)
    self.ratio = wp.zeros((self.nworld, entries), dtype=wp.vec3, device=dev)
    self.linear = wp.zeros(self.nworld, dtype=wp.int32, device=dev)
    self.nonlinear = wp.zeros((self.nworld, 2), dtype=wp.uint32, device=dev)
    self.authored_nonlinear = wp.zeros((1, 2), dtype=wp.uint32, device=dev)
    # Each world's last re-light by rays (`remember`): its non-linear digest, slot powers and ratio.
    self.base_nonlinear = wp.zeros((self.nworld, 2), dtype=wp.uint32, device=dev)
    self.base_power = wp.zeros((self.nworld, self.nslot), dtype=wp.vec3, device=dev)
    self.traced = wp.ones((self.nworld, entries), dtype=wp.vec3, device=dev)
    # The per-world GI tables hashed for the linear flag (`adopt`): all but LINEAR_INPUTS.
    self.nonlinear_names: frozenset[str] = frozenset()
    self._replay = Replay()  # `track`'s recorded launches
    self._masked: tuple[wp.array[Light, [int, int]], wp.array[LightHot, [int, int]]] | None = None

  @property
  def nbytes(self) -> int:
    """Device memory of the tint and its buffers."""
    arrays = (
      self.texels,
      self.reference,
      self.sums,
      self.samples,
      self.dirs,
      self.world_digest,
      self.digests,
      self.changed,
      self.slot_sums,
      self.shares,
      self.power,
      self.authored_power,
      self.ratio,
      self.linear,
      self.nonlinear,
      self.authored_nonlinear,
      self.base_nonlinear,
      self.base_power,
      self.traced,
      self.hold,
      self.requests,
      self.slot_of,
      *self.slot_texels,
    )
    # Each texture (the tint's, one per slot) holds a copy of its staging texels.
    textures = self.texels.capacity + sum(t.capacity for t in self.slot_texels)
    return int(sum(a.capacity for a in arrays) + textures)

  def adopt[NInstance: IntVar](self, frame: Frame, skips: Skips[NInstance], authored: Frame) -> None:
    """Takes the frame's (quantised) digests as the worlds' current ones: the bake's tint is theirs
    (the bake's host digests are exact, for the groups); and the `authored` frame's digest of the
    per-world tables but `slots.LINEAR_INPUTS` (the inputs the slots' shares were baked with); and
    marks the worlds at those inputs as recombined (`track`)."""
    digest(frame, self.nworld, skips, self.world_digest, self.quantum)
    names = {name for name, a in per_world_inputs(frame) if a.shape[0] == self.nworld}
    self.nonlinear_names = frozenset(names - slots_lib.LINEAR_INPUTS)
    self._replay = Replay()
    digest(authored, 1, skips, self.authored_nonlinear, self.quantum, names=self.nonlinear_names)
    # Which worlds take the recombined tint, already for the bake: else the bake's frame showed
    # their traced ratio and the next frame the recombined one (an image change without an input
    # change). Their digests are the ones just adopted: no world is marked changed.
    self.track(frame, skips)

  def track[NInstance: IntVar](self, frame: Frame, skips: Skips[NInstance]) -> None:
    """Marks the worlds whose inputs changed since their ratio was computed (device launches;
    light positions quantised: `groups.POSITION_QUANTUM`; recorded and replayed: `Replay`)."""
    self._replay.run(frame, skips, lambda: self._launches(frame, skips))

  def _launches[NInstance: IntVar](self, frame: Frame, skips: Skips[NInstance]) -> list[wp.Launch]:
    """`track`'s launches of `frame` (recorded, not run)."""
    out = digest_launches(frame, self.nworld, skips, self.digests, self.quantum)
    out += digest_launches(frame, self.nworld, skips, self.nonlinear, self.quantum, names=self.nonlinear_names)
    inputs = [self.digests, self.nonlinear, self.authored_nonlinear, self.base_nonlinear, self.hold, self.requests]
    outputs = [self.world_digest, self.changed, self.linear]
    return [*out, record(mark_changed_linear, self.nworld, inputs, outputs, self.device)]

  def set_policy(self, *, on_request: bool) -> None:
    """gi.relight: re-light a world by rays when its inputs change (False), or only once it is
    requested (`request`; True). A device write when it changes (none otherwise)."""
    if on_request != self._held:
      self._held = on_request
      self.hold.fill_(int(on_request))

  def request(self, worlds: wp.array[bool, [int]]) -> None:
    """Lets the `worlds` (nworld,) re-light by rays at their next input change, this frame or a
    later one (gi.relight="on_request"; device launch, no host read)."""
    wp.launch(_request, dim=self.nworld, inputs=[worlds], outputs=[self.requests], device=self.device)

  def bake(self, kernel: wp.Kernel, inputs: ProbeInputs) -> None:
    """The reference (world 0's sums with every ratio 1), then every world's ratio (part of the
    bake: not counted as re-lights; the open gate computes every world), after the slots' shares
    (`slots`)."""
    self.changed.zero_()
    self._bake_slots(kernel, inputs)
    self.step(kernel, inputs, self.always)

  def _bake_slots(self, kernel: wp.Kernel, inputs: ProbeInputs) -> None:
    """The slots' shares (PASSES passes of the fixed rays per slot, world 0's authored inputs, each
    slot's bounce read through its own share field) and the authored slot powers."""
    dev = self.device
    env, mats = inputs.env, inputs.mats
    dark_env = _copy(env, env_intensity=wp.zeros_like(env.env_intensity), ambient_radiance=wp.zeros_like(env.ambient_radiance))
    dark_mats = _copy(mats, material_emission=wp.zeros_like(mats.material_emission))
    entries = self.ncell * FACES
    for _ in range(PASSES):
      for k in range(self.nslot):
        rest = k == self.nslot - 1
        lit = self._slot_lights(inputs.lights(), slots_lib.REST if rest else k)
        e = env if rest else dark_env
        e = _copy(e, gi=_copy(env.gi, relight=self.slot_fields[k]))
        slot_inputs = inputs._replace(env=e, mats=mats if rest else dark_mats)
        self._trace(kernel, slot_inputs, lit, self.always, rows=1, fixed_world=0, out=self.slot_sums[k : k + 1])
      wp.launch(slots_lib.shares_of, dim=(self.nslot, entries), inputs=[self.slot_sums], outputs=[self.shares], device=dev)
      for k, (field, texels) in enumerate(zip(self.slot_fields, self.slot_texels, strict=True)):
        wp.launch(slots_lib.store_shares, dim=entries, inputs=[field, self.shares, k], outputs=[texels], device=dev)
        field.texture.copy_from(texels)
    lit = inputs.lights()
    wp.launch(
      slots_lib.slot_power,
      dim=1,
      inputs=[lit.lights, self.slot_of, env.env_intensity, 1],
      outputs=[self.authored_power],
      device=dev,
    )

  def _slot_lights(self, lit: Lights, slot: int) -> Lights:
    """`lit` (a fresh struct, modified) with only slot `slot`'s lights on (REST: none)."""
    packed = lit.lights
    if self._masked is None or self._masked[0].shape != packed.shape:
      self._masked = (
        wp.empty(packed.shape, dtype=Light, device=self.device),
        wp.empty(packed.shape, dtype=LightHot, device=self.device),
      )
    out, hot = self._masked
    wp.launch(
      slots_lib.slot_lights, dim=packed.shape, inputs=[packed, self.slot_of, slot], outputs=[out, hot], device=self.device
    )
    lit.lights, lit.hot = out, hot
    return lit

  def restart(self, gate: wp.array[int, [1]]) -> None:
    """Ratio 1 for the changed worlds (every world when `gate` is open: before an update of the
    shared maps, whose probe rays then read them untinted)."""
    wp.launch(
      restart,
      dim=(self.nworld, self.ncell * FACES),
      inputs=[self.field, gate, self.changed],
      outputs=[self.texels],
      device=self.device,
    )
    self.field.texture.copy_from(self.texels)

  def step(self, kernel: wp.Kernel, inputs: ProbeInputs, gate: wp.array[int, [1]] | None = None) -> None:
    """Recomputes the ratios of the changed worlds: ratio 1, then PASSES passes of their rays,
    each resolved into the ratios the next one bounces. With `gate` open (the bake, an update of
    the shared maps): first the reference, world 0's sums untinted, then every world."""
    gate = self.closed if gate is None else gate
    dev, n = self.device, self.nworld
    worlds = inputs.worlds if inputs.worlds is not None else inputs
    field, texels = self.field, self.texels
    lit = worlds.lights()
    self._split_linear(worlds, lit, gate)
    self.restart(gate)
    self._trace(kernel, inputs, inputs.lights(), gate, rows=1, fixed_world=0, out=self.reference)
    for _ in range(PASSES):
      self._trace(kernel, worlds, lit, gate, rows=n, fixed_world=-1, out=self.sums)
      wp.launch(
        resolve,
        dim=(n, self.ncell * FACES),
        inputs=[field, self.sums, self.reference, gate, self.changed],
        outputs=[texels, self.traced],
        device=dev,
      )
      field.texture.copy_from(texels)
    self._recombine(gate)
    wp.launch(count_relights, dim=n, inputs=[self.changed], outputs=[self.relights], device=dev)
    remembered = [self.base_nonlinear, self.base_power]
    wp.launch(remember, dim=n, inputs=[gate, self.changed, self.nonlinear, self.power], outputs=remembered, device=dev)
    wp.launch(settle, dim=n, outputs=[self.changed, self.requests], device=dev)

  def _split_linear(self, worlds: ProbeInputs, lit: Lights, gate: wp.array[int, [1]]) -> None:
    """The worlds' slot powers, and which of the worlds at the authored inputs but
    `slots.LINEAR_INPUTS` take the recombined tint (`slots.split_linear`): the others are re-lit by
    rays when their inputs changed."""
    dev, n = self.device, self.nworld
    wp.launch(
      slots_lib.slot_power,
      dim=n,
      inputs=[lit.lights, self.slot_of, worlds.env.env_intensity, n],
      outputs=[self.power],
      device=dev,
    )
    wp.launch(
      slots_lib.split_linear,
      dim=n,
      inputs=[self.power, self.authored_power, self.hold, self.requests, gate],
      outputs=[self.linear, self.changed, self.world_digest],
      device=dev,
    )

  def _recombine(self, gate: wp.array[int, [1]]) -> None:
    """The tint of the `linear` worlds: their slot powers over the authored ones (or their traced
    ones), weighted by the slots' shares (no rays)."""
    dev, n, entries = self.device, self.nworld, self.ncell * FACES
    wp.launch(
      slots_lib.recombine,
      dim=(n, entries),
      inputs=[self.shares, self.power, self.authored_power, self.base_power, self.traced, self.linear, RATIO_MAX],
      outputs=[self.ratio],
      device=dev,
    )
    inputs = [self.field, self.ratio, self.linear, gate]
    wp.launch(store_linear, dim=(n, entries), inputs=inputs, outputs=[self.texels], device=dev)
    self.field.texture.copy_from(self.texels)

  def _trace(
    self,
    kernel: wp.Kernel,
    inputs: ProbeInputs,
    lit: Lights,
    gate: wp.array[int, [1]],
    *,
    rows: int,
    fixed_world: int,
    out: wp.array[wp.vec3, [int, int, int]],
  ) -> None:
    """One launch of the relight kernel (`kernels.make_relight_kernel`) over `rows` rows."""
    wp.launch(
      kernel,
      dim=(rows, self.ncell, SAMPLES),
      inputs=[inputs.geo, inputs.mats, lit, inputs.env, self.dirs, self.samples, gate, self.changed, fixed_world, self.spread],
      outputs=[out],
      device=self.device,
    )


def _copy[S](s: S, **changes: object) -> S:
  """A shallow copy of Warp struct `s` with fields `changes` replaced."""
  out = type(s)()
  for name, value in bake_cache.fields(s).items():
    setattr(out, name, changes.get(name, value))
  return out
