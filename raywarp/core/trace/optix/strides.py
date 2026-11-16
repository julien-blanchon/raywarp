"""The innermost-stride check of OptiX launch parameters, vectorised (host code).

`params.bound_words` binds each array's innermost stride to its element size, so every launch
checks that no non-empty array of its parameters is a strided view. The check reads, for every
array descriptor of the struct, its data pointer, shape and innermost stride from the struct's
bytes in one gather (`misread`), instead of one ctypes descriptor per array.
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
from typing import Final, final

import numpy as np

# Bytes of a descriptor's data pointer and of one shape or stride entry (int32).
_POINTER_BYTES: Final = 8
_ENTRY_BYTES: Final = 4

# (launch-parameter byte offset of an array descriptor, its innermost stride's index, element
# bytes, field path): `params._Array`.
type ArrayField = tuple[int, int, int, str]


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _Gather:
  """Where `misread` reads each array descriptor of a launch-parameter struct.

  Attributes:
    index: (narray, 8 + 4 ndim + 4) byte offsets of each descriptor's data pointer, shape and
      innermost stride.
    dims: (narray, ndim) the shape entries up to the innermost one.
    size: (narray,) element bytes.
  """

  index: np.ndarray[tuple[int, int], np.dtype[np.intp]]
  dims: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
  size: np.ndarray[tuple[int], np.dtype[np.int32]]


@functools.cache
def _gather(arrays: tuple[ArrayField, ...], array_t: type[ctypes.Structure]) -> _Gather:
  """The `_Gather` of descriptors `arrays` laid out as Warp's ctypes descriptor `array_t`."""
  ndim = getattr(array_t, "shape").size // _ENTRY_BYTES  # noqa: B009  # a ctypes field descriptor (bytes)
  base = np.array([a[0] for a in arrays], np.intp).reshape(-1, 1)
  inner = np.array([a[1] for a in arrays], np.intp).reshape(-1, 1)
  data, shape, strides = (getattr(array_t, name).offset for name in ("data", "shape", "strides"))
  index = np.concatenate(
    [
      base + data + np.arange(_POINTER_BYTES),
      base + shape + np.arange(_ENTRY_BYTES * ndim),
      base + strides + _ENTRY_BYTES * inner + np.arange(_ENTRY_BYTES),
    ],
    axis=1,
  )
  return _Gather(index=index, dims=np.arange(ndim) <= inner, size=np.array([a[2] for a in arrays], np.int32))


def misread(raw: bytes, arrays: tuple[ArrayField, ...], array_t: type[ctypes.Structure]) -> tuple[int, int] | None:
  """(index in `arrays`, innermost stride) of the first array of launch-parameter bytes `raw` that
  is read (a data pointer and no empty dimension up to its innermost one) and whose innermost
  stride is not its element size; None when there is none. `array_t`: Warp's ctypes array
  descriptor (`params._array_type`)."""
  g = _gather(arrays, array_t)
  fields = np.frombuffer(raw, np.uint8)[g.index]
  data = fields[:, :_POINTER_BYTES].copy().view(np.uint64)[:, 0]
  shape = fields[:, _POINTER_BYTES:-_ENTRY_BYTES].copy().view(np.int32)
  stride = fields[:, -_ENTRY_BYTES:].copy().view(np.int32)[:, 0]
  read = (data != 0) & np.all((shape > 0) | ~g.dims, axis=1)
  bad = np.flatnonzero(read & (stride != g.size))
  return (int(bad[0]), int(stride[bad[0]])) if len(bad) else None
