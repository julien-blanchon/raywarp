"""Checks of the values given to a `SceneBuilder`: bad input fails there, with a clear message.

Each helper raises ValueError naming the value, what it got and the range it expects, so a
negative radius or a NaN color fails at `add_light` / `add_material`, not deep in a kernel.
`where` names the call and the object ("add_light 'rect' /World/Lamp").
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
import operator
from collections.abc import Sequence
from typing import Final, SupportsIndex

import numpy as np
import numpy.typing as npt

from raywarp.scene.model import normal_map
from raywarp.scene.model.material import GEOMETRY_INPUTS, INPUTS, TEXTURE_SLOTS, Material, OpenPbrExtras, RaywarpExt
from raywarp.scene.model.types import (
  CHANNEL_A,
  CHANNEL_RGB,
  UV_SET_WORLD,
  AlphaMode,
  Array,
  Camera,
  MaterialFlags,
  Projection,
  Slot,
  TextureRef,
)

# Arrays up to this many entries are shown in full in an error message.
_SHOWN: Final = 16


def finite_array(name: str, value: npt.ArrayLike, where: str, *, shape: tuple[int, ...] | None = None) -> Array:
  """`value` as a float64 array of `shape` (None: any) with every entry finite.

  Raises:
    ValueError: for another shape or a NaN / inf entry.
  """
  a = np.asarray(value, np.float64)
  if shape is not None and a.shape != shape:
    raise ValueError(f"{where}: {name} must have shape {shape}, got {a.shape}")
  if not np.all(np.isfinite(a)):
    raise ValueError(f"{where}: {name} must be finite, got {a.tolist() if a.size <= _SHOWN else 'NaN / inf entries'}")
  return a


def color(name: str, value: Sequence[float] | Array, where: str, *, size: int = 3) -> Array:
  """A finite, non-negative color / radiance of `size` components (float64).

  Raises:
    ValueError: for another size, or a NaN / inf / negative component.
  """
  a = finite_array(name, value, where, shape=(size,))
  if np.any(a < 0.0):
    raise ValueError(f"{where}: {name} must be non-negative, got {a.tolist()}")
  return a


def in_range(name: str, value: float, where: str, *, low: float, high: float = math.inf) -> float:
  """`value` when finite and in [low, high].

  Raises:
    ValueError: for a NaN / inf or a value outside [low, high].
  """
  v = float(value)
  if not (math.isfinite(v) and low <= v <= high):
    raise ValueError(f"{where}: {name} must be in [{low}, {high}], got {v}")
  return v


def _integer(name: str, value: float, where: str, *, low: int, high: int) -> None:
  """Checks an integer in [low, high].

  Raises:
    ValueError: for a NaN / inf, a fractional value or one outside [low, high].
  """
  if not in_range(name, value, where, low=low, high=high).is_integer():
    raise ValueError(f"{where}: {name} must be an integer, got {value}")


def finite(name: str, value: float, where: str) -> float:
  """`value` when finite.

  Raises:
    ValueError: for a NaN or an inf.
  """
  v = float(value)
  if not math.isfinite(v):
    raise ValueError(f"{where}: {name} must be finite, got {v}")
  return v


def positive(name: str, value: float, where: str) -> float:
  """`value` when finite and positive.

  Raises:
    ValueError: for a NaN / inf, zero or negative value.
  """
  v = float(value)
  if not 0.0 < v < math.inf:
    raise ValueError(f"{where}: {name} must be positive and finite, got {v}")
  return v


def index(name: str, value: SupportsIndex, count: int, *, where: str, optional: bool = True) -> int:
  """A row id into a table of `count` rows: in [0, count), or -1 (none) when `optional`.

  Raises:
    ValueError: for an id out of that range.
  """
  i = operator.index(value)
  if (optional and i == -1) or 0 <= i < count:
    return i
  none = " or -1 (none)" if optional else ""
  raise ValueError(f"{where}: {name} {i} is not one of the {count} added so far (ids 0..{count - 1}{none})")


def member[E: enum.IntEnum](name: str, value: SupportsIndex, kind: type[E], *, where: str) -> E:
  """`value` as a member of `kind`.

  Raises:
    ValueError: for a value that is not one (listing the valid ones).
  """
  try:
    return kind(operator.index(value))
  except (ValueError, TypeError):
    valid = ", ".join(f"{m.name.lower()}={int(m)}" for m in kind)
    raise ValueError(f"{where}: {name} {value} is not a {kind.__name__} ({valid})") from None


def named[E: enum.IntEnum](
  name: str, value: SupportsIndex | str, kind: type[E], *, where: str, aliases: dict[str, E] | None = None
) -> E:
  """`value` as a member of `kind`, a string naming one (any case; or one of `aliases`) or a value.

  Raises:
    ValueError: for an unknown name or value.
  """
  if not isinstance(value, str):
    return member(name, value, kind, where=where)
  names = {m.name.lower(): m for m in kind} | ({} if aliases is None else aliases)
  found = names.get(value.lower())
  if found is None:
    raise ValueError(f"{where}: {name} {value!r} is not one of {sorted(names)}")
  return found


def flags[F: enum.IntFlag](name: str, value: SupportsIndex, kind: type[F], *, where: str) -> int:
  """`value` when it sets only bits of flag type `kind`.

  Raises:
    ValueError: for a negative value or unknown bits.
  """
  known = functools.reduce(operator.or_, (int(m) for m in kind), 0)
  v = operator.index(value)
  if v < 0 or v & ~known:
    raise ValueError(f"{where}: {name} {v:#x} sets bits outside {kind.__name__} ({known:#x})")
  return v


def texture_ref(ref: TextureRef, *, ntexture: int, where: str) -> None:
  """Checks a texture binding: a texture added before (one of `ntexture`), its channel, UV set
  and finite scale / bias / UV transform.

  Raises:
    ValueError: for a value out of range.
  """
  index("texture", ref.texture, ntexture, where=where, optional=False)
  _integer("channel", ref.channel, where, low=CHANNEL_RGB, high=CHANNEL_A)
  _integer("uv_set", ref.uv_set, where, low=0, high=UV_SET_WORLD)
  finite_array("scale", ref.scale, where, shape=(4,))
  finite_array("bias", ref.bias, where, shape=(4,))
  finite_array("uv_xform", ref.uv_xform, where, shape=(2, 3))


# The OpenPBR normal inputs without a Slot: their textures are tangent-space normal maps too.
NORMAL_INPUTS: Final = tuple(n for n in GEOMETRY_INPUTS if n.endswith("_normal") and n not in TEXTURE_SLOTS)


def normal_ref(ref: TextureRef, *, where: str) -> None:
  """Checks a normal-map binding (scene/model/normal_map.py): its decoded z must point out of the
  surface for some texel (finite scale / bias: `texture_ref`).

  Raises:
    ValueError: for a binding whose every normal lies in or below the tangent plane.
  """
  if not normal_map.outward(ref.scale, ref.bias):
    raise ValueError(
      f"{where}: a normal map's z = scale[2] x texel + bias[2] must be > 0 for some texel in [0, 1], got scale "
      f"{ref.scale[2]}, bias {ref.bias[2]} (the plain unpack is scale 2, bias -1)"
    )


# The closed range [low, high] of a material input (`POSITIVE` inputs exclude 0).
_UNIT: Final = (0.0, 1.0)
_SIGNED_UNIT: Final = (-1.0, 1.0)
_NON_NEGATIVE: Final = (0.0, math.inf)
# Strictly positive inputs (an IOR, an Abbe number).
POSITIVE: Final = frozenset({"specular_ior", "coat_ior", "thin_film_ior", "transmission_dispersion_abbe_number"})
# The range of every float OpenPBR input of `Material` / `OpenPbrExtras`.
FLOAT_RANGES: Final = {
  **dict.fromkeys(
    ("base_weight", "base_metalness", "specular_roughness", "transmission_weight", "coat_weight", "coat_roughness",
     "geometry_opacity", "base_diffuse_roughness", "specular_roughness_anisotropy", "subsurface_weight", "fuzz_weight",
     "fuzz_roughness", "coat_darkening", "coat_roughness_anisotropy", "thin_film_weight"),
    _UNIT,
  ),
  **dict.fromkeys(("transmission_scatter_anisotropy", "subsurface_scatter_anisotropy"), _SIGNED_UNIT),
  **dict.fromkeys(
    ("specular_weight", "emission_luminance", "transmission_depth", "subsurface_radius", "thin_film_thickness",
     "transmission_dispersion_scale", *POSITIVE),
    _NON_NEGATIVE,
  ),
}  # fmt: skip
# The range of every color input: reflectance-like colors in [0, 1], the others >= 0.
COLOR_RANGES: Final = {
  **dict.fromkeys(
    ("base_color", "specular_color", "transmission_color", "coat_color", "subsurface_color", "fuzz_color"), _UNIT
  ),
  **dict.fromkeys(("emission_color", "transmission_scatter", "subsurface_radius_scale"), _NON_NEGATIVE),
}
# The index of refraction the Unreal specular curve (SPECULAR_CURVE) and a COAT_GLTF coat assume.
FIXED_IOR: Final = 1.5


def _value(name: str, value: float, where: str) -> float:
  """`value` of float OpenPBR input `name` when in its range (`FLOAT_RANGES`, `POSITIVE`).

  Raises:
    ValueError: for a NaN / inf or a value out of range.
  """
  if name in POSITIVE:
    return positive(name, value, where)
  low, high = FLOAT_RANGES[name]
  return in_range(name, value, where, low=low, high=high)


def _color(name: str, value: object, where: str) -> Array:
  """`value` of color OpenPBR input `name` when an RGB tuple, finite and in its range
  (`COLOR_RANGES`).

  Raises:
    ValueError: for a value that is not a tuple of 3, a NaN / inf or a component out of range.
  """
  if not isinstance(value, tuple):
    raise ValueError(f"{where}: {name} must be an RGB tuple, got {type(value).__name__} {value!r}")
  low, high = COLOR_RANGES[name]
  a = finite_array(name, value, where, shape=(3,))
  if np.any(a < low) or np.any(a > high):
    raise ValueError(f"{where}: {name} components must be in [{low}, {high}], got {a.tolist()}")
  return a


def _ext(ext: RaywarpExt, where: str) -> None:
  """Checks the raywarp extensions of a material.

  Raises:
    ValueError: for a value out of range, an unknown alpha mode or flag.
  """
  member("ext.alpha_mode", ext.alpha_mode, AlphaMode, where=where)
  if not (math.isfinite(ext.alpha_cutoff) and 0.0 < ext.alpha_cutoff <= 1.0):
    raise ValueError(f"{where}: ext.alpha_cutoff must be in (0, 1], got {ext.alpha_cutoff}")
  in_range("ext.occlusion_strength", ext.occlusion_strength, where, low=0.0, high=1.0)
  in_range("ext.mirror", ext.mirror, where, low=0.0, high=1.0)
  in_range("ext.wrap", ext.wrap, where, low=0.0, high=1.0)
  finite("ext.anisotropy_rotation", ext.anisotropy_rotation, where)
  flags("ext.flags", ext.flags, MaterialFlags, where=where)


def _inputs(record: type[Material | OpenPbrExtras]) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """(float inputs, color inputs) of a record type: its fields with a range."""
  names = [f.name for f in dataclasses.fields(record)]
  return tuple(n for n in names if n in FLOAT_RANGES), tuple(n for n in names if n in COLOR_RANGES)


# The ranged inputs of each record (checked unconditionally, whatever the value's type).
MATERIAL_INPUTS: Final = _inputs(Material)
EXTRAS_INPUTS: Final = _inputs(OpenPbrExtras)


def _ranges(record: Material | OpenPbrExtras, inputs: tuple[tuple[str, ...], tuple[str, ...]], where: str) -> None:
  """Checks every float and color input of `record` (`inputs`: `_inputs` of its type) against
  `FLOAT_RANGES` / `COLOR_RANGES` (numpy scalars included).

  Raises:
    ValueError: for a value out of range or of the wrong kind.
  """
  floats, colors = inputs
  for name in floats:
    _value(name, operator.attrgetter(name)(record), where)
  for name in colors:
    _color(name, operator.attrgetter(name)(record), where)


def _extras(extras: OpenPbrExtras, ntexture: int, where: str) -> None:
  """Checks the OpenPBR extras of a material: ranges, and textures keyed by an OpenPBR input that
  no `Slot` drives, sorted, one per input.

  Raises:
    ValueError: for a value out of range or a bad texture key.
  """
  _ranges(extras, EXTRAS_INPUTS, where)
  keys = [k for k, _ in extras.textures]
  if keys != sorted(set(keys)):
    raise ValueError(f"{where}: extras.textures must be sorted by input, one per input, got {keys}")
  for key, ref in extras.textures:
    if key not in INPUTS or key in TEXTURE_SLOTS:
      raise ValueError(f"{where}: extras.textures key {key!r} is not an OpenPBR input without a Slot")
    texture_ref(ref, ntexture=ntexture, where=f"{where} extras texture {key}")
    if key in NORMAL_INPUTS:
      normal_ref(ref, where=f"{where} extras texture {key}")


def material(mat: Material, ntexture: int) -> None:
  """Checks a canonical material: every OpenPBR input in its range, finite colors, the
  combinations the derivation relies on, the extensions and the textures (one per slot, added
  before).

  Raises:
    ValueError: naming the material, the field, its value and the expected range.
  """
  where = f"material {mat.name!r}"
  _ranges(mat, MATERIAL_INPUTS, where)
  _ext(mat.ext, where)
  if mat.displacement is not None:
    amount = mat.displacement.amount
    if isinstance(amount, TextureRef):
      texture_ref(amount, ntexture=ntexture, where=f"{where} displacement")
      if amount.channel not in (0, 1, 2, 3) or amount.uv_set not in (0, 1):
        raise ValueError(f"{where}: displacement requires a scalar channel and mesh UV set 0 or 1")
    elif isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
      raise ValueError(f"{where}: displacement amount must be a finite scalar or TextureRef")
  if mat.extras is not None:
    _extras(mat.extras, ntexture, where)
  opacity_map = mat.texture(Slot.GEOMETRY_OPACITY) is not None
  if (mat.geometry_opacity < 1.0 or opacity_map) and mat.ext.alpha_mode == AlphaMode.OPAQUE:
    raise ValueError(f"{where}: geometry_opacity < 1 or an opacity texture needs ext.alpha_mode BLEND or MASK")
  for flag, ior, field in (
    (MaterialFlags.SPECULAR_CURVE, mat.specular_ior, "specular_ior"),
    (MaterialFlags.COAT_GLTF, mat.coat_ior, "coat_ior"),
  ):
    if mat.ext.flags & flag and ior != FIXED_IOR:
      raise ValueError(f"{where}: {flag.name} needs {field} {FIXED_IOR}, got {ior}")
  slots = [s for s, _ in mat.textures]
  if slots != sorted(set(slots)):
    raise ValueError(f"{where}: textures must be sorted by slot, one per slot, got {slots}")
  for slot, ref in mat.textures:
    member("slot", slot, Slot, where=where)
    texture_ref(ref, ntexture=ntexture, where=f"{where} slot {Slot(slot).name.lower()}")
    if slot == Slot.GEOMETRY_NORMAL:
      normal_ref(ref, where=f"{where} slot geometry_normal")


def camera(cam: Camera, where: str) -> None:
  """Checks a camera: a finite transform, a known projection, a positive focal length and
  apertures, finite offsets and exposure, near >= 0, far 0 (unbounded) or beyond near, and a
  non-negative f-stop and focus distance.

  Raises:
    ValueError: for a value out of range.
  """
  finite_array("xform", cam.xform, where, shape=(3, 4))
  member("projection", cam.projection, Projection, where=where)
  for name, value in (
    ("focal_length", cam.focal_length),
    ("horizontal_aperture", cam.horizontal_aperture),
    ("vertical_aperture", cam.vertical_aperture),
  ):
    positive(name, value, where)
  for name, value in (
    ("horizontal_offset", cam.horizontal_offset),
    ("vertical_offset", cam.vertical_offset),
    ("exposure [stops]", cam.exposure),
  ):
    finite(name, value, where)
  near = in_range("near [m]", cam.near, where, low=0.0)
  far = in_range("far [m]", cam.far, where, low=0.0)
  if far != 0.0 and far <= near:
    raise ValueError(f"{where}: far must be 0 (unbounded) or beyond near ({near}), got {far}")
  in_range("fstop", cam.fstop, where, low=0.0)
  in_range("focus_distance [m]", cam.focus_distance, where, low=0.0)
