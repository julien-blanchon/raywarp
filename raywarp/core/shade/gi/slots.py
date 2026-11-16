"""Linear re-light of a relit volume: per-light-slot shares of the indirect light, recombined per
world with its lights' colours and intensities and its environment intensity, without rays.

Light transport is linear in emission: the indirect light of a cell and direction is the sum of
each source's contribution, and scaling a source scales its contribution. The relit volume's
coarse cells (`relight.py`) therefore get, at the bake, the **share** of each slot in each cell and
axis direction: K light slots (one per light, the lights beyond MAX_LIGHT_SLOTS - 1 by baked power
sharing the last one) and one *rest* slot (the environment, emissive surfaces and the ambient
fill). A slot's share comes from the relit volume's fixed probe rays shaded with that slot's
sources only, PASSES passes: the first direct only, the next ones bouncing the shared cube times the
slot's own share at the hit (the share of the slot in the light arriving there), the classic
iteration of a light's own inter-reflection at the cells' resolution.

A world whose other GI inputs (materials, light positions and shapes, environment map and yaw:
everything but `LINEAR_INPUTS`) are the authored ones (AUTHORED) gets its tint as

  ratio(cell, dir) = sum_k  weight_k(world) * share_k(cell, dir)

with weight_k the per-channel ratio of the world's slot power (sum over its lights of colour x
intensity x active, the packed lights') to the authored one, and the rest slot's weight the ratio
of its environment intensity: exact at the cells' resolution for any colour, intensity or on / off
change of a slot lit at the bake, at the cost of a few multiply-adds per cell. A slot dark at the
bake (its lights off or of zero power in the authored frame, or a dark environment) has no share to
scale: a world that lights it is re-lit by rays (`split_linear`). A world whose other inputs differ
is re-lit by rays (`relight.Relight`) once; while they stay those of that re-light (TRACED: a
material randomised per episode, the lights per frame), its traced ratio is rescaled by the
recombination at its slot powers now over the one at its traced powers (`recombine`; the shares
are the authored scene's: an approximation when its materials moved the light between slots).
Light positions follow the relight's quantised digest. Aggregated lights (beyond
MAX_LIGHT_SLOTS - 1) are exact when they scale together, otherwise weighted by their summed power.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import warp as wp

from raywarp.core.shade.gi.probes import RelightField, relight_texel
from raywarp.core.shade.lights import punctual as lights_lib
from raywarp.core.shade.lights.punctual import Light, LightHot

wp.set_module_options({"enable_backward": False})

# Light slots at most (the last one aggregates the weakest lights); plus the rest slot.
MAX_LIGHT_SLOTS: Final = 8
# The Frame fields a recombination handles (the rest re-light by rays).
LINEAR_INPUTS: Final = frozenset({"light_color", "light_strength", "light_active", "env_intensity"})
# The rest slot's index in the slot-of-light table: no light is in it.
REST: Final = -2
# Not in any slot (a dynamic light: `dynamic.StaticLights`).
NO_SLOT: Final = -1
# A world's linear mode (`relight.mark_changed_linear`): its other inputs are the authored ones
# (recombined from the shares), or the ones of its last re-light by rays (that ratio rescaled).
AUTHORED: Final = 1
TRACED: Final = 2
# A TRACED world's recombination at its traced slot powers below this (per channel): no light to
# rescale from; it takes the AUTHORED recombination there.
_DARK_RATIO: Final = 1.0e-6


def assign(
  power: np.ndarray[[int], np.dtype[np.float32]], dynamic: tuple[int, ...]
) -> tuple[np.ndarray[[int], np.dtype[np.int32]], int]:
  """(slot of each light (NO_SLOT for the dynamic ones), light slots) from the lights' baked
  luminous power `power` (nlight,): the strongest MAX_LIGHT_SLOTS - 1 get their own slot when
  there are more lights than MAX_LIGHT_SLOTS, the others share the last one."""
  n = len(power)
  slot = np.full(n, NO_SLOT, np.int32)
  skipped = set(dynamic)
  live = np.array([i for i in range(n) if i not in skipped], np.int64)
  if not len(live):
    return slot, 0
  order = live[np.argsort(-power[live], kind="stable")]
  nslot = min(len(order), MAX_LIGHT_SLOTS)
  for rank, light in enumerate(order):
    slot[light] = min(rank, nslot - 1)
  return slot, nslot


@wp.kernel
def slot_lights(
  # In:
  packed: wp.array2d[Light],
  slot_of: wp.array1d[wp.int32],
  slot: int,
  # Out:
  out: wp.array2d[Light],
  hot: wp.array2d[LightHot],
):
  """Light `li` of row `row` with only the lights of slot `slot` on (REST: none on)."""
  row, li = wp.tid()
  light = packed[row, li]
  if li >= slot_of.shape[0] or slot_of[li] != slot:
    light.active = False
  out[row, li] = light
  hot[row, li] = lights_lib.hot_of(light)


@wp.kernel
def slot_power(
  # In:
  packed: wp.array2d[Light],
  slot_of: wp.array1d[wp.int32],
  env_intensity: wp.array1d[float],
  nworld: int,
  # Out:
  power: wp.array2d[wp.vec3],
):
  """World w's power per slot (colour x intensity of its active lights, summed per slot; the rest
  slot, last: its environment intensity, grey). One thread per world."""
  w = wp.tid()
  nslot = power.shape[1] - 1
  for k in range(nslot):
    power[w, k] = wp.vec3(0.0)
  row = w % packed.shape[0]
  for li in range(slot_of.shape[0]):
    k = slot_of[li]
    if k >= 0 and li < packed.shape[1]:
      light = packed[row, li]
      if light.active:
        power[w, k] = power[w, k] + light.color * light.intensity
  e = env_intensity[w % env_intensity.shape[0]]
  power[w, nslot] = wp.vec3(e)
  _ = nworld


@wp.kernel
def split_linear(
  # In:
  power: wp.array2d[wp.vec3],
  authored: wp.array2d[wp.vec3],
  hold: wp.array1d[wp.int32],
  requests: wp.array1d[wp.int32],
  gate: wp.array1d[int],
  # Out:
  linear: wp.array1d[wp.int32],
  changed: wp.array1d[wp.int32],
  world_digest: wp.array2d[wp.uint32],
):
  """World w, `linear` (AUTHORED or TRACED): nothing to recombine when its inputs did not change
  and the gate is closed (`linear` cleared: its tint is current); else takes the recombined tint
  (`changed` cleared: no rays) unless a slot dark at the bake (zero authored power in a channel:
  its share holds none of the slot's light) has power in it; such a world is re-lit by rays when
  its inputs changed (`linear` cleared), or, while re-lights wait for a request (`hold`) and none
  came, keeps its tint (its digest cleared: checked again next frame). One thread per world."""
  w = wp.tid()
  if linear[w] == 0:
    return
  if changed[w] == 0 and gate[0] == 0:
    linear[w] = 0
    return
  for k in range(power.shape[1]):
    a = authored[0, k]
    p = power[w, k]
    for ch in range(3):
      if a[ch] <= 0.0 and p[ch] > 0.0:
        linear[w] = 0
        if hold[0] != 0 and requests[w] == 0 and changed[w] != 0:
          changed[w] = 0
          world_digest[w, 0] = wp.uint32(0)
          world_digest[w, 1] = wp.uint32(0)
        return
  changed[w] = 0


@wp.kernel
def shares_of(
  # In:
  sums: wp.array3d[wp.vec3],
  # Out:
  shares: wp.array2d[wp.vec3],
):
  """Slot k's share of entry i (cell i // FACES, face i % FACES) of the slots' summed sample
  sums: its sum over all slots' (per channel; a dark entry: all of it to the rest slot)."""
  k, i = wp.tid()
  nslot = sums.shape[0]
  c = i // 6
  f = i % 6
  mine = wp.vec3(0.0)
  total = wp.vec3(0.0)
  for s in range(sums.shape[2] // 6):
    mine += sums[k, c, s * 6 + f]
    for j in range(nslot):
      total += sums[j, c, s * 6 + f]
  out = wp.vec3(0.0)
  for ch in range(3):
    if total[ch] > 0.0:
      out[ch] = wp.max(mine[ch], 0.0) / total[ch]
    elif k == nslot - 1:
      out[ch] = 1.0
  shares[k, i] = out


@wp.kernel
def store_shares(
  # Model:
  field: RelightField,
  # In:
  shares: wp.array2d[wp.vec3],
  slot: int,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """Slot `slot`'s shares as a one-row tint (share - 1: `probes.relit` reads 1 + it), entry i."""
  i = wp.tid()
  c = i // 6
  n = field.counts
  x = c % n[0]
  y = (c // n[0]) % n[1]
  z = c // (n[0] * n[1])
  at = relight_texel(field, i % 6)
  v = shares[slot, i]
  for ch in range(3):
    texels[at[1] + z, at[0] + y, x, ch] = wp.float16(v[ch] - 1.0)


@wp.func
def _recombined(
  # In:
  shares: wp.array2d[wp.vec3],
  power: wp.array2d[wp.vec3],
  authored: wp.array2d[wp.vec3],
  w: int,
  i: int,
) -> wp.vec3:
  """Entry i's ratio of slot powers row w over the authored ones, weighted by the slots' shares."""
  out = wp.vec3(0.0)
  for k in range(shares.shape[0]):
    a = authored[0, k]
    p = power[w, k]
    weight = wp.vec3(1.0)
    for ch in range(3):
      if a[ch] > 0.0:
        weight[ch] = p[ch] / a[ch]
    out += wp.cw_mul(weight, shares[k, i])
  return out


@wp.kernel
def recombine(
  # In:
  shares: wp.array2d[wp.vec3],
  power: wp.array2d[wp.vec3],
  authored: wp.array2d[wp.vec3],
  base_power: wp.array2d[wp.vec3],
  traced: wp.array2d[wp.vec3],
  linear: wp.array1d[wp.int32],
  ratio_max: float,
  # Out:
  ratio: wp.array2d[wp.vec3],
):
  """World w's ratio of entry i when its other inputs are the authored ones (AUTHORED: its slots'
  power ratios weighted by their shares) or those of its last re-light by rays (TRACED: that
  traced ratio times the recombination at its powers now over the one at its traced powers; the
  shares are the authored scene's, the rescale moves the light between slots as they would)."""
  w, i = wp.tid()
  lin = linear[w]
  if lin == 0:
    return
  out = _recombined(shares, power, authored, w, i)
  if lin == wp.static(TRACED):
    base = _recombined(shares, base_power, authored, w, i)
    t = traced[w, i]
    for ch in range(3):
      if base[ch] > wp.static(_DARK_RATIO):
        out[ch] = t[ch] * out[ch] / base[ch]
  for ch in range(3):
    out[ch] = wp.clamp(out[ch], 0.0, ratio_max)
  ratio[w, i] = out
