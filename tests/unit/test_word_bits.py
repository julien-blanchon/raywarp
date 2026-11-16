"""The build word's and the switches' bit tables (`kernels/abi/toggles.py`): one bit per name, no
two names on one bit (but the ellipsoid and cylinder, which share the custom-primitive bit), every
word bit inside the int32 copies' 31 value bits (each half: `bits`, `bits2`), `DEFINED` outside
them, and the groups the gates and the copies' masks use (`BOUND_ALL`,
`trace/optix/device/params._COPIES`) the union of their bits."""

from __future__ import annotations

import functools
import operator

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi import toggles


def _union(bits: list[int]) -> int:
  """The bits of `bits` or-ed."""
  return functools.reduce(operator.or_, bits, 0)


def _word_bits() -> dict[str, int]:
  """Every named bit of the build word (the shape types by their bit: ellipsoid and cylinder
  once)."""
  named = {f"class {n}": b for n, b in toggles.CLASS_BITS.items()}
  named |= {"FLEX": toggles.FLEX, "FLEX_MATERIALS": toggles.FLEX_MATERIALS}
  for prefix in ("GEO_", "SCENE_", "SETTING_"):
    named |= {n: getattr(toggles, n) for n in vars(toggles) if n.startswith(prefix) and not n.endswith("_ALL")}
  named |= {f"shape {bit:#x}": bit for bit in set(toggles.SHAPE_BITS.values())}
  return named


def _high_bits() -> dict[str, int]:
  """Every named bit of the word's second half, in-half (`toggles.high` moves them up)."""
  names = ("MAT_", "MAP_", "LAYER_", "SURF_", "GLASS_")
  named = {n: getattr(toggles, n) for n in vars(toggles) if n.startswith(names)}
  return named


def test_every_word_bit_is_one_bit_of_its_own() -> None:
  """Each name holds one bit, no bit has two names, all below bit 31 (the copies are int32)."""
  bits, high = _word_bits(), _high_bits()
  for half in (bits, high):
    assert all(b > 0 and b & (b - 1) == 0 for b in half.values()), half
    assert len(set(half.values())) == len(half), half
    assert max(half.values()) < 1 << 31
  # The compile-time settings read at run time (`tiers.SETTING_FIELDS`, the run-time sampling): bits
  # of their own, the same value in `Toggles.flags`.
  from raywarp.core.build.variants import tiers

  settings = [*tiers.SETTING_FIELDS.values(), toggles.SOFT_SHADOWS, toggles.ANISOTROPY, toggles.RUN_COUNTS]
  assert len(set(settings)) == len(settings) and not _union(settings) & _union(list(high.values()))
  assert not _union(settings) & _union([b for n, b in toggles.FLAGS.items() if n not in tiers.SETTING_FIELDS])
  assert toggles.HIGH_ALL == _union([*high.values(), *settings])
  assert toggles.BOUND_ALL == _union(list(bits.values())) | toggles.high(toggles.HIGH_ALL)
  assert not toggles.DEFINED & toggles.BOUND_ALL


def test_the_field_bits_are_the_second_half_s() -> None:
  """`tiers.FIELD_BITS`: every field on a second-half bit (`toggles.high`), one field per bit."""
  from raywarp.core.build.variants import tiers

  values = list(tiers.FIELD_BITS.values())
  assert len(set(values)) == len(values)
  assert {b >> toggles.HIGH for b in values} == set(_high_bits().values())


def test_the_group_masks_are_their_bits() -> None:
  """`CLASS_ALL`, `GEO_ALL`, `SHAPE_ALL`, `SCENE_ALL`, `SETTING_ALL`: the union of their names'
  bits; the ellipsoid and the cylinder share the custom-primitive bit, the other shape types one
  each."""
  for group in ("GEO_", "SCENE_", "SETTING_"):
    names = [getattr(toggles, n) for n in vars(toggles) if n.startswith(group) and not n.endswith("_ALL")]
    assert getattr(toggles, group + "ALL") == _union(names), group
  assert toggles.CLASS_ALL == _union(list(toggles.CLASS_BITS.values()))
  assert toggles.SHAPE_ALL == _union(list(toggles.SHAPE_BITS.values()))
  shapes = toggles.SHAPE_BITS
  assert shapes[int(ShapeType.ELLIPSOID)] == shapes[int(ShapeType.CYLINDER)] == toggles.SHAPE_CUSTOM
  others = [b for t, b in shapes.items() if t not in (int(ShapeType.ELLIPSOID), int(ShapeType.CYLINDER))]
  assert len(set(others)) == len(others)


def test_switch_flags_are_distinct_bits() -> None:
  """The `Toggles.flags` switches: one bit each, `ALL` their union."""
  flags = list(toggles.FLAGS.values())
  assert all(b > 0 and b & (b - 1) == 0 for b in flags) and len(set(flags)) == len(flags)
  assert toggles.ALL == _union(flags)
