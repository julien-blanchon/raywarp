"""OpenEXR images (linear HDR texels), read by the `OpenEXR` package (a dependency of the usd
extra): every layout and compression it reads (scan-line, tiled, multi-part: the first part).

The image is the data window. Channels: R, G, B (A) -> RGB(A), Y -> grey, else the first
channels in the file's (alphabetical) order. A file whose decoded channels are all HALF is stored
as float16 (it holds them exactly), else float32. Without the package an EXR is rejected at
probe, reported `texture.exr.openexr_missing`.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
from typing import TYPE_CHECKING, Final, final

import numpy as np

from raywarp.scene.images import files

if TYPE_CHECKING:
  import OpenEXR

# Channel name orders picked, in preference order (RGB(A), then grey).
_WANTED: Final = (("R", "G", "B", "A"), ("R", "G", "B"), ("Y", "A"), ("Y",))
_MAX_CHANNELS: Final = 4
_HALF: Final = "HALF"


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Header:
  """What a probe needs from an EXR header: the data window's size, the decoded channels in
  output order and whether they are all HALF."""

  width: int
  height: int
  channels: tuple[str, ...]
  half: bool


def _selected(names: list[str]) -> list[str]:
  """The channels decoded, in output order (layered names such as "diffuse.R": the first layer,
  in file order, holding every wanted component; never components of two layers)."""
  for wanted in _WANTED:
    if all(w in names for w in wanted):
      return list(wanted)
  layers: dict[str, dict[str, str]] = {}
  for n in names:
    layer, _, component = n.rpartition(".")
    layers.setdefault(layer, {})[component.upper()] = n
  for wanted in _WANTED[:2]:
    for components in layers.values():
      if all(w in components for w in wanted):
        return [components[w] for w in wanted]
  return names[:_MAX_CHANNELS]


def _require() -> None:
  """Raises without the OpenEXR package.

  Raises:
    ValueError: `texture.exr.openexr_missing: ...`.
  """
  if importlib.util.find_spec("OpenEXR") is None:
    raise ValueError("texture.exr.openexr_missing: EXR images are decoded by the OpenEXR package (pip install openexr)")


def _source(path: str) -> str | io.BytesIO:
  """What the package opens for `path`: the path, or a packaged file's bytes."""
  return io.BytesIO(files.read(path)) if files.is_packaged(path) else path


def read_header(path: str) -> Header:
  """The header of EXR file `path` (its first part).

  Raises:
    ValueError: without the OpenEXR package, for a sub-sampled channel or an empty data window.
  """
  _require()
  import OpenEXR  # noqa: PLC0415 (optional dependency: the usd extra)

  f = OpenEXR.InputFile(_source(path))  # typed channels (`OpenEXR.File`'s header-only read has no types)
  try:
    header = f.header()
  finally:
    f.close()
  box = header["dataWindow"]
  width, height = box.max.x - box.min.x + 1, box.max.y - box.min.y + 1
  if width <= 0 or height <= 0:
    raise ValueError(f"{path}: dataWindow must have positive width and height, got {width} x {height}")
  channels = header["channels"]
  selected = _selected(sorted(channels))
  for name in selected:
    if (channels[name].xSampling, channels[name].ySampling) != (1, 1):
      raise ValueError(f"{path}: sub-sampled channel {name!r} (unsupported)")
  half = all(str(channels[n].type) == _HALF for n in selected)
  return Header(width=width, height=height, channels=tuple(selected), half=half)


def decode(path: str) -> np.ndarray[[int, int, int], np.dtype[np.float32]]:
  """(height, width, channels) float32 linear texels of EXR file `path` (see the module
  docstring)."""
  _require()
  import OpenEXR  # noqa: PLC0415 (optional dependency: the usd extra)

  with OpenEXR.File(_source(path), separate_channels=True) as f:
    found = f.parts[0].channels
    selected = _selected(sorted(found))
    planes = [_plane(found, name) for name in selected]
  return np.stack([np.asarray(p, np.float32) for p in planes], axis=2)


def _plane(channels: dict[str, OpenEXR.Channel], name: str) -> np.ndarray[[int, int]]:
  """The pixels of channel `name`.

  Raises:
    ValueError: when the file has no such channel.
  """
  ch = channels.get(name)
  if ch is None:
    raise ValueError(f"OpenEXR: no channel {name!r}")
  return np.asarray(ch.pixels)
