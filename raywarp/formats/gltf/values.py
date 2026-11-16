"""glTF JSON values as pygltflib keeps them (extensions, extras: plain dicts), read with their types
checked: the USD and MJCF converters' accessors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pygltflib

# A glTF JSON object (an extension's or extras' values; pygltflib keeps them as dicts).
type Object = Mapping[str, object]


def obj(d: Object, key: str) -> Object:
  """`d[key]` as a JSON object ({} when absent).

  Raises:
    ValueError: for another value.
  """
  v = d.get(key)
  if v is None:
    return {}
  if not isinstance(v, Mapping):
    raise ValueError(f"glTF: {key} must be an object, got {v!r}")
  return cast("Object", v)


def num(d: Object, key: str, default: float) -> float:
  """`d[key]` as a number (`default` when absent).

  Raises:
    ValueError: for another value.
  """
  v = d.get(key, default)
  if not isinstance(v, int | float):
    raise ValueError(f"glTF: {key} must be a number, got {v!r}")
  return float(v)


def nums(d: Object, key: str, default: Sequence[float]) -> list[float]:
  """`d[key]` as a list of numbers (`default` when absent).

  Raises:
    ValueError: for another value.
  """
  v = d.get(key, default)
  if not isinstance(v, list | tuple) or not all(isinstance(x, int | float) for x in v):
    raise ValueError(f"glTF: {key} must be a list of numbers, got {v!r}")
  return [float(x) for x in v]


# A material's textureInfo (glTF 2.0 §5.21; the normal and occlusion ones add a scale / strength).
type TextureInfo = pygltflib.TextureInfo | pygltflib.NormalMaterialTexture | pygltflib.OcclusionTextureInfo
