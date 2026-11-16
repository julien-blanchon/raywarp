"""`Writer`: the canonical Material of one distilled material, field by field: constants, the
texture a value reads (bound as is), else an SDK bake; values clamped to their OpenPBR ranges."""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Mapping
from typing import Final, final

from raywarp._base.named import replace_named
from raywarp.formats.usd.materials.context import MeshNeeds, ShadeContext, UvRows, Where
from raywarp.formats.usd.materials.mdl import memo, ops
from raywarp.formats.usd.materials.mdl.bake import Baked, Bakes, Kind, Request, bake, encoding, rows_of
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.values import (
  RGB,
  Lin,
  NeedsBakeError,
  SlotValue,
  UnsupportedError,
  analyse,
  comps,
  pad4,
)
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model import normal_map
from raywarp.scene.model.checks import COLOR_RANGES, FLOAT_RANGES
from raywarp.scene.model.material import GEOMETRY_INPUTS, Material, OpenPbrExtras
from raywarp.scene.model.normal_map import NORMAL_UNPACK
from raywarp.scene.model.types import CHANNEL_RGB, UV_SET_WORLD, Slot, TextureId, TextureRef, Wrap

# A Material / OpenPbrExtras field value.
type Field = float | tuple[float, ...] | bool
# Out-of-range values within it are rounding (clamped, not reported).
_CLAMP_TOL: Final = 1e-3
_TOL: Final = 1e-6
_COLORS: Final = ("base_color", "transmission_color", "emission_color", "specular_color")
# The scalar inputs of `OpenPbrExtras` (a field of these is set there).
_EXTRA_SCALARS: Final = frozenset(f.name for f in dataclasses.fields(OpenPbrExtras) if f.type in ("float", float))


@final
class Writer:
  """Accumulates the Material of one distilled material from its expressions."""

  def __init__(self, ctx: ShadeContext, dag: Dag, prim: Where, bakes: Bakes) -> None:
    """A writer of `dag`'s material for `prim` (reports), its bakes kept in `bakes`."""
    self.ctx = ctx
    self.dag = dag
    self.prim = prim
    self.bakes = bakes
    self.mat = Material(specular_roughness=0.5)
    self.uv_sets: set[int] = set()
    self.binary_opacity = False
    self.rec = memo.Recorder()  # the texture ops, for the conversion's disk cache
    # A planning pass (`plan`): the bakes the mapping asks for, made in parallel before the pass
    # that binds them.
    self.requests: list[Request] | None = None

  def report(self, what: str, *, approx: bool = False) -> None:
    """A report entry `mdl.<what>` at the shader."""
    (self.ctx.report.approx if approx else self.ctx.report.drop)(f"mdl.{what}", self.prim)

  def set(self, **fields: Field) -> None:
    """Sets Material fields, clamped to their ranges (reported beyond rounding; NaN: 0)."""
    self.mat = replace_named(self.mat, self._clamped(fields))

  def set_extras(self, **fields: Field) -> None:
    """Sets OpenPbrExtras fields, clamped as `set`'s."""
    extras = replace_named(self.mat.extras or OpenPbrExtras(), self._clamped(fields))
    self.mat = dataclasses.replace(self.mat, extras=extras)

  def _clamped(self, fields: Mapping[str, Field]) -> dict[str, Field]:
    """`fields` in their ranges (`checks.FLOAT_RANGES` / `COLOR_RANGES`; reported beyond
    rounding)."""
    out: dict[str, Field] = {}
    for name, v in fields.items():
      low, high = FLOAT_RANGES.get(name) or COLOR_RANGES.get(name) or (-math.inf, math.inf)
      xs = () if isinstance(v, bool) else v if isinstance(v, tuple) else (v,)
      if any(not low - _CLAMP_TOL <= x <= high + _CLAMP_TOL for x in xs):
        self.report(f"clamped.{name}(to [{low}, {high}])", approx=True)
      clamped = tuple(min(max(x, low), high) if math.isfinite(x) else max(low, 0.0) for x in xs)
      out[name] = v if isinstance(v, bool) else clamped if isinstance(v, tuple) else clamped[0]
    return out

  def const(self, path: str, n: int) -> tuple[float, ...] | None:
    """The value at `path` when it does not vary over the surface (None: it varies)."""
    sv = self._direct(path, n, direction=False)
    return None if sv is None else sv.const

  def mean(self, path: str, n: int, default: float) -> tuple[float, ...]:
    """The value at `path`, its mean when it varies (reported), `default` when the SDK cannot
    bake it (reported)."""
    value = self.const(path, n)
    if value is not None:
      return value
    kind: Kind = "color" if n > 1 else "scalar"
    if self.requests is not None:
      self.requests.append((path, kind, True))
      return (default,) * n
    if self.dag.varies(path):
      self.report(f"{path.rpartition('.')[2]}(texture mean)", approx=True)
    baked = bake(self.ctx, self.dag, path, kind=kind, bakes=self.bakes, mean=True)
    self._nonfinite(baked, path)
    if baked is None or baked.const is None:
      self.report(f"bake_failed.{path.rpartition('.')[2]}(default {default})")
      return (default,) * n
    return baked.const

  def _nonfinite(self, baked: Baked | None, path: str) -> None:
    """Reports a bake whose texels held NaN / inf (stored as 0)."""
    if baked is not None and baked.key in self.bakes.nonfinite:
      self.report(f"bake_nonfinite.{path.rpartition('.')[2]}(read as 0)", approx=True)

  def field(  # noqa: PLR0913  # the field, its slot and expression, then keyword-only options
    self, field: str, slot: Slot | None, path: str, *, kind: Kind, factor: float = 1.0, extra: str | None = None
  ) -> float | None:
    """Sets `field` from the expression at `path` (times `factor`): a constant, a texture of
    `slot` (the field then holds `factor`), baked or not; the mean without a slot. `extra`: the
    texture binds `OpenPbrExtras.textures` input `extra` instead (`slot`: its role, e.g.
    GEOMETRY_NORMAL for `geometry_coat_normal`). Returns the gain of an emission texture (its
    largest scale, divided out of the texture); None, and the field unchanged, when the SDK
    cannot bake it."""
    n = 3 if kind in ("color", "emission", "normal") else 1
    value = self._root(path) if kind == "roughness" else self._direct(path, n, direction=kind == "normal")
    gain = max(value.scale, default=1.0) if value is not None and kind == "emission" and not any(value.bias) else 1.0
    if value is not None and gain > 0.0 and gain != 1.0:
      value = dataclasses.replace(value, scale=tuple(x / gain for x in value.scale))
    if value is not None and kind == "normal" and value.tex is not None and not self._outward(value):
      value = None  # a degenerate normal map (no texel points outward): baked
    if value is not None and (value.const is not None or (slot is not None and self.bind(slot, value, extra))):
      self._set(field, value.const or (1.0,) * n, factor)
      return gain
    if slot is None:
      self._set(field, self.mean(path, n, 0.0), factor)
      return 1.0
    if self.requests is not None:
      self.requests.append((path, kind, False))
      return 1.0
    baked = bake(self.ctx, self.dag, path, kind=kind, bakes=self.bakes)
    self._nonfinite(baked, path)
    if baked is None or (baked.const is None and not self._bind_baked(slot, baked, kind, extra)):
      self.report(f"bake_failed.{field}(unchanged)")
      return None
    for space in baked.gamuts:
      self.report(f"bake_gamut:{space}(primaries as Rec.709)", approx=True)
    self._set(field, baked.const or (1.0,) * n, factor)
    return baked.peak

  def _direct(self, path: str, n: int, *, direction: bool) -> SlotValue | None:
    """The slot value at `path`: a constant or one texture's affine channels (a normal up to
    its length), else None."""
    try:
      return analyse(self.dag.value(path, direction=direction), n)
    except (NeedsBakeError, UnsupportedError):
      return None

  @staticmethod
  def _outward(sv: SlotValue) -> bool:
    """Whether a normal map's affine read (its signs, strength and offset as the expression
    holds them: `scene.model.normal_map`'s convention, the baker's tangent frame) points outward."""
    return normal_map.outward(pad4(sv.scale, 1.0), pad4(sv.bias, 0.0))

  def _root(self, path: str) -> SlotValue | None:
    """The perceptual roughness r of a GGX alpha r x r at `path`: a constant or one texture
    channel's affine function, else None."""
    try:
      alpha = comps(self.dag.value(path))[0]
    except UnsupportedError:
      return None
    if isinstance(alpha, Lin) and alpha.const:
      return SlotValue(const=(math.sqrt(max(alpha.c, 0.0)),))
    root = ops.square_of(alpha, frozenset(self.dag.floats))
    try:
      return None if root is None else analyse(root, 1)
    except (NeedsBakeError, UnsupportedError):
      return None

  def _set(self, field: str, value: tuple[float, ...], factor: float) -> None:
    """`field` = value x factor (a scalar, or RGB; a constant normal is the geometry's)."""
    if field in _COLORS:
      self.set(**{field: tuple(factor * x for x in (value * RGB if len(value) == 1 else value)[:RGB])})
    elif field in _EXTRA_SCALARS:
      self.set_extras(**{field: factor * value[0]})
    elif field not in GEOMETRY_INPUTS:  # a normal map's field is its texture
      self.set(**{field: factor * value[0]})

  def bind(self, slot: Slot, sv: SlotValue, extra: str | None = None) -> bool:
    """Binds `slot` (`extra`: that `OpenPbrExtras.textures` input, `slot` its role) to the
    texture of a slot value (False when unusable; planning: whether it is usable)."""
    read = None if sv.tex is None or sv.uv is None or not sv.tex.path else rows_of(sv.uv, self.ctx.frame)
    if self.requests is not None:  # planning: no builder texture
      return read is not None
    image = -1 if read is None or sv.tex is None else self.ctx.texture(sv.tex.path, slot, sv.tex.colorspace, sv.wrap)
    if read is None or image < 0 or sv.tex is None:
      return False
    space = "" if sv.tex.colorspace is None else sv.tex.colorspace.value
    self.rec.add(TextureId(image), ("file", sv.tex.path, int(slot), space, sv.wrap[0], sv.wrap[1]))
    ref = TextureRef(TextureId(image), channel=sv.channel, scale=pad4(sv.scale, 1.0), bias=pad4(sv.bias, 0.0))
    self._ref(slot, ref, read, extra)
    return True

  def _bind_baked(self, slot: Slot, baked: Baked, kind: Kind, extra: str | None) -> bool:
    """Binds `slot` (`extra`: that extras input) to a baked image read through its domain (False:
    the builder rejects its texels)."""
    image = self.bakes.image(self.ctx.builder, baked, kind)
    if image is None:
      return False
    space = ColorSpace.SRGB_REC709 if encoding(kind) == "srgb" else ColorSpace.LIN_REC709
    texture = self.ctx.builder.add_texture(image, colorspace=space, wrap=(Wrap.REPEAT, Wrap.REPEAT))
    transfer = "srgb" if encoding(kind) == "srgb" else "linear"
    self.rec.add(
      TextureId(texture), None if baked.file is None else ("baked", str(baked.file), transfer, space.value, baked.peak)
    )
    scale, bias = NORMAL_UNPACK if kind == "normal" else ((1.0,) * 4, (0.0,) * 4)
    channel = 0 if kind in ("scalar", "roughness") else CHANNEL_RGB
    ref = TextureRef(TextureId(texture), channel=channel, scale=scale, bias=bias)
    self._ref(slot, ref, (baked.domain.uv_set, baked.domain.rows), extra)
    self.ctx.report.count("mdl.baked")
    self.binary_opacity = self.binary_opacity or (slot == Slot.GEOMETRY_OPACITY and baked.binary)
    if baked.domain.position:
      self.report(f"baked_position.{extra or slot.name.lower()}(the position read as the mesh UVs (u, v, 0))", approx=True)
    elif not baked.domain.exact and baked.domain.uv_set == UV_SET_WORLD:
      what = "the finest world tiling's window; coarser lookups repeat with it"
      self.report(f"baked_world_window.{extra or slot.name.lower()}({what})", approx=True)
    elif not baked.domain.exact:
      self.report(f"baked_uv_grid.{extra or slot.name.lower()}(exact for mesh UVs in [0, 1])", approx=True)
    return True

  def _ref(self, slot: Slot, ref: TextureRef, read: tuple[int, UvRows], extra: str | None) -> None:
    """Binds `slot` (`extra`: that `OpenPbrExtras.textures` input) to `ref` read through
    (uv_set, rows)."""
    self.uv_sets.add(read[0])
    ref = dataclasses.replace(ref, uv_xform=read[1], uv_set=read[0])
    if extra is None:
      self.mat = self.mat.with_texture(slot, ref)
      return
    extras = self.mat.extras or OpenPbrExtras()
    textures = tuple(sorted((*((k, r) for k, r in extras.textures if k != extra), (extra, ref)), key=operator.itemgetter(0)))
    self.mat = dataclasses.replace(self.mat, extras=dataclasses.replace(extras, textures=textures))

  def needs(self) -> MeshNeeds:
    """UV primvars the bound textures read: MDL texture_coordinate(i) is primvars:st / st<i>."""
    return MeshNeeds(("st", "st1") if 1 in self.uv_sets else ("st",))
