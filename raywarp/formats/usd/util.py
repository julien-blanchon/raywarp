"""Small pxr helpers: typed attribute values, matrices, asset paths, authored color spaces.

Shaped host types of the adapter (`Matrix4`, `Matrices4`, `Vec3`) are defined here; the
matrices are float64, column-vector convention (`world @ point`).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import warnings
from collections.abc import Generator, Sequence
from typing import final

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdShade

from raywarp.scene.images import colorspace
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model.report import LoadReport

type Matrix4 = np.ndarray[[4, 4], np.dtype[np.float64]]
type Matrices4 = np.ndarray[[int, 4, 4], np.dtype[np.float64]]
type Matrix3 = np.ndarray[[3, 3], np.dtype[np.float64]]
type Matrices3 = np.ndarray[[int, 3, 3], np.dtype[np.float64]]
type Vec3 = np.ndarray[[3], np.dtype[np.float64]]
type Color = tuple[float, float, float]
# The time code of a stage read (Usd.TimeCode; EarliestTime when the options name none).
type Time = Usd.TimeCode
# A value read from USD: Python scalars, strings, Gf vectors / matrices, asset paths, arrays.
type UsdValue = object


def open_stage(path: str | os.PathLike[str]) -> Usd.Stage:
  """The USD stage of file `path`.

  Raises:
    FileNotFoundError: for a file that does not open (missing or unreadable: pxr raises its own
      Tf error, or returns None).
  """
  try:
    stage = Usd.Stage.Open(os.fspath(path))
  except Tf.ErrorException as e:
    raise FileNotFoundError(f"cannot open USD stage {os.fspath(path)}: {e}") from None
  if stage is None:
    raise FileNotFoundError(f"cannot open USD stage {os.fspath(path)}")
  return stage


def _authored_first(prim: Usd.Prim, names: Sequence[str], time: Time) -> UsdValue:
  """The first of `names` with a value on `prim`: authored ones first, then schema fallbacks."""
  for authored_only in (True, False):
    for name in names:
      attr = prim.GetAttribute(name)
      if attr and (attr.HasAuthoredValue() or not authored_only):
        v = attr.Get(time)
        if v is not None:
          return v
  return None


def _names(names: str | Sequence[str]) -> Sequence[str]:
  """`names` as a sequence."""
  return (names,) if isinstance(names, str) else names


def as_float(v: UsdValue) -> float:
  """A USD number (bool, int, float) as a float."""
  if isinstance(v, (bool, int, float)):
    return float(v)
  if isinstance(v, np.generic):
    return float(np.asarray(v, np.float64))
  raise TypeError(f"not a number: {v!r}")


def float_value(prim: Usd.Prim, names: str | Sequence[str], default: float, time: Time) -> float:
  """A float attribute (the first of `names` with a value), else `default`."""
  v = _authored_first(prim, _names(names), time)
  return default if v is None else as_float(v)


def bool_value(prim: Usd.Prim, names: str | Sequence[str], *, default: bool, time: Time) -> bool:
  """A bool attribute (the first of `names` with a value), else `default`."""
  v = _authored_first(prim, _names(names), time)
  return default if v is None else bool(v)


def token_value(prim: Usd.Prim, names: str | Sequence[str], default: str, time: Time) -> str:
  """A token / string attribute (the first of `names` with a value), else `default`."""
  v = _authored_first(prim, _names(names), time)
  return default if v is None else str(v)


def color_value(prim: Usd.Prim, names: str | Sequence[str], default: Color, time: Time) -> Color:
  """A color / float3 attribute (the first of `names` with a value), else `default`."""
  v = _authored_first(prim, _names(names), time)
  return default if v is None else vec3(v)


def asset_value(prim: Usd.Prim, names: str | Sequence[str], time: Time) -> str:
  """Resolved path of an asset attribute ('' when unset or unresolved)."""
  return asset_path(_authored_first(prim, _names(names), time))


def floats(v: UsdValue) -> tuple[float, ...]:
  """A Gf vector / sequence of numbers as a float tuple."""
  return tuple(float(c) for c in np.asarray(v, np.float64).reshape(-1))


def vec3(v: UsdValue) -> Color:
  """A Gf vector / sequence of three numbers as a float tuple."""
  x, y, z = floats(v)[:3]
  return x, y, z


def describe(where: object) -> str:
  """A load-report example: a USD object (prim, property) or schema as its path, else `str`."""
  if isinstance(where, (Usd.Object, Usd.SchemaBase)):
    return str(where.GetPath())
  return str(where)


def valued_attributes(prim: Usd.Prim) -> list[Usd.Attribute]:
  """`prim`'s attributes with an authored value or connection (a declaration without either, as
  Kit writes for every input of an applied schema, authors nothing)."""
  return [a for a in prim.GetAuthoredAttributes() if a.HasAuthoredValue() or a.HasAuthoredConnections()]


def authored(prim: Usd.Prim, name: str) -> bool:
  """Whether `prim` has an authored value for attribute `name`."""
  attr = prim.GetAttribute(name)
  return bool(attr) and attr.HasAuthoredValue()


def matrix(m: Gf.Matrix4d) -> Matrix4:
  """A Gf.Matrix4d (row-vector convention) as a (4, 4) column-vector float64 matrix."""
  return np.array(m, dtype=np.float64).T


def asset_path(v: UsdValue) -> str:
  """Resolved path of an Sdf.AssetPath value ('' if unresolved or not an asset path)."""
  if isinstance(v, Sdf.AssetPath):
    return v.resolvedPath or ""
  return ""


@final
class _AssetCache:
  """Asset-path resolutions of one stage read, memoised by (anchor layer, authored path).

  USD resolves an asset-valued attribute on every Get (file-system lookups: slow on a network
  mount, and a large interior's ~2100 texture inputs name 340 distinct files). The first Get of each
  (layer, path) is USD's own resolution; the others reuse it. Paths whose resolution can depend
  on more than the layer (time samples, variable expressions, anonymous layers) are not cached.

  Attributes:
    missing: anchored paths of the file assets USD could not resolve (cache dependencies: the
      scene changes when one appears).
  """

  def __init__(self) -> None:
    """Creates an empty cache."""
    self.resolved: dict[tuple[str, str], str] = {}
    self.missing: set[str] = set()

  def get(self, attr: Usd.Attribute) -> UsdValue:
    """`attr.Get()` of an asset-valued attribute, through the memo when its value is static."""
    if attr.GetNumTimeSamples():
      return attr.Get()
    spec = next((s for s in attr.GetPropertyStack(Usd.TimeCode.Default()) if s.HasDefaultValue()), None)
    v = spec.default if spec is not None else None
    if spec is None or not isinstance(v, Sdf.AssetPath) or not v.path:
      return attr.Get()
    path, layer = v.path, spec.layer
    if layer.anonymous or any(c in path for c in "<>$`"):
      return attr.Get()
    key = (layer.identifier, path)
    resolved = self.resolved.get(key)
    if resolved is None:
      value = attr.Get()
      resolved = value.resolvedPath if isinstance(value, Sdf.AssetPath) else ""
      if not resolved and "://" not in path:
        # Windows exports (3ds Max: `@textures\wall.png@`) author backslash separators, which USD
        # does not resolve elsewhere: the same path with `/`, anchored at the authoring layer.
        anchored = os.path.normpath(layer.ComputeAbsolutePath(path.replace("\\", "/")))
        if "\\" in path and os.path.isfile(anchored):
          resolved = anchored
        else:
          self.missing.add(anchored)
      self.resolved[key] = resolved
      return Sdf.AssetPath(path, resolved) if resolved else value
    return Sdf.AssetPath(path, resolved)


# The resolutions of the current stage read (`asset_cache`); a context variable: reads in other
# threads or contexts do not share it.
_asset_cache: contextvars.ContextVar[_AssetCache | None] = contextvars.ContextVar("asset_cache", default=None)


@contextlib.contextmanager
def asset_cache() -> Generator[_AssetCache]:
  """Caches asset-path resolutions meanwhile (one stage read; see `_AssetCache`)."""
  cache = _AssetCache()
  token = _asset_cache.set(cache)
  try:
    yield cache
  finally:
    _asset_cache.reset(token)


def get(attr: Usd.Attribute) -> UsdValue:
  """`attr.Get()`, with asset paths resolved through the active `asset_cache`."""
  cache = _asset_cache.get()
  if cache is not None and attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
    return cache.get(attr)
  return attr.Get()


def producing(inp: UsdShade.Input) -> tuple[UsdValue, Usd.Attribute | None]:
  """(value, attribute) that produces a shader input's value, following connections.

  When the input is connected to a shader output, the value is None and the attribute is that
  output (`outputs:...`).
  """
  attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
  if attrs:
    attr = attrs[0]
    if attr.GetName().startswith("outputs:"):
      return None, attr
    return get(attr), attr
  attr = inp.GetAttr()
  return get(attr), attr


def shader_inputs(prim: Usd.Prim) -> dict[str, UsdShade.Input]:
  """A shader prim's inputs by base name."""
  return {i.GetBaseName(): i for i in UsdShade.Shader(prim).GetInputs()}


def input_value(inputs: dict[str, UsdShade.Input], name: str) -> UsdValue:
  """The value producing input `name` (None when absent or connected to a shader output)."""
  inp = inputs.get(name)
  return None if inp is None else producing(inp)[0]


def input_floats(inputs: dict[str, UsdShade.Input], name: str, default: tuple[float, ...]) -> tuple[float, ...]:
  """The value of vector input `name` as floats, else `default`."""
  v = input_value(inputs, name)
  return default if v is None else floats(v)


def connected_output(inp: UsdShade.Input) -> Usd.Attribute | None:
  """The shader output connected to `inp` (None: it has a value or none)."""
  _, attr = producing(inp)
  return attr if attr is not None and attr.GetName().startswith("outputs:") else None


def colorspace_name(attr: Usd.Attribute | None, unauthored: str = "") -> str:
  """An attribute's colorSpace metadata, `unauthored` when it has none."""
  return str(attr.GetColorSpace()) if attr is not None and attr.HasColorSpace() else unauthored


def colorspace_of(attr: Usd.Attribute | None, token: str = "", *, report: LoadReport, where: object) -> ColorSpace | None:
  """The color space of a `sourceColorSpace` token, else of the asset attribute's colorSpace
  metadata (`colorspace.parse`: interop IDs, OCIO / MaterialX / MDL names and OCIO's ACES config
  names); None when neither is authored ("" / "auto": the caller's context default,
  `auto_colorspace`).

  An unknown name is reported (`texture.colorspace.unknown:<name>` at `where`) and warned once per
  name, then read by its prefix: "lin_*" names as LIN_REC709, others as unauthored (None: sRGB
  for 8-bit color images).
  """
  name = token.strip()
  if not name or name.lower() == "auto":
    name = colorspace_name(attr).strip()
  return colorspace_named(name, report=report, where=where)


def colorspace_named(name: str, *, report: LoadReport, where: object) -> ColorSpace | None:
  """The color space of a color space name (`colorspace.parse`); None for "" / "auto" (the
  caller's context default). An unknown name is reported and warned once per name: "lin_*" names
  read as LIN_REC709, others as unauthored (None). One rule for every material path (UsdShade,
  MDL, MaterialX)."""
  name = name.strip()
  if not name or name.lower() == "auto":
    return None
  space = colorspace.parse(name)
  if space is not None:
    return space
  linear = name.lower().startswith("lin_")
  read = "linear Rec.709" if linear else "as unauthored"
  report.approx(f"texture.colorspace.unknown:{name}(read {read})", where)
  # The message names the space, so Python's default filter shows it once per name.
  warnings.warn(f"USD color space {name!r} is unknown: read {read}", UserWarning, stacklevel=2)
  return ColorSpace.LIN_REC709 if linear else None
