"""The MDL CPU bake detaches borrowed SDK tile storage before releasing its handles."""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

pytest.importorskip("pxr")

from raywarp.formats.usd.materials.context import ShadeContext
from raywarp.formats.usd.materials.mdl import bake, sdk
from raywarp.formats.usd.materials.mdl.dag import Dag

_WINDOW = (0.0, 1.0, 0.0, 1.0)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Tile:
  """A raw pointer view, matching the SDK's ctypes/frombuffer ownership behavior."""

  data: bake.Texels

  def get_data_numpy(self) -> bake.Texels:
    """Borrow the buffer without retaining this tile or the SDK canvas."""
    pointer = ctypes.cast(self.data.ctypes.data, ctypes.POINTER(ctypes.c_uint8 * self.data.nbytes))
    return np.frombuffer(pointer.contents, dtype=np.float32).reshape(self.data.shape)


@dataclasses.dataclass(slots=True, weakref_slot=True, kw_only=True)
class _Canvas:
  """Invalidate data at canvas destruction, keeping the allocation safe to inspect."""

  data: bake.Texels

  def get_tile(self) -> _Tile:
    """Return a temporary tile handle."""
    return _Tile(data=self.data)

  def __del__(self) -> None:
    """Model native storage reuse without dereferencing actually freed memory."""
    self.data.fill(np.nan)


@dataclasses.dataclass(slots=True, kw_only=True)
class _Images:
  """Keep allocations, but only weak references to their SDK canvas owners."""

  channels: int
  buffers: list[bake.Texels] = dataclasses.field(default_factory=list)
  owners: list[weakref.ReferenceType[_Canvas]] = dataclasses.field(default_factory=list)

  def create_canvas(self, pixel_type: str, width: int, height: int) -> _Canvas:
    """Make distinguishable rows so vertical orientation is observable."""
    data = np.arange(width * height * self.channels).astype(np.float32).reshape(height, width, self.channels) / 16
    owner = _Canvas(data=data)
    self.buffers.append(data)
    self.owners.append(weakref.ref(owner))
    return owner


class _Baker:
  """A successful CPU bake into the fake canvas's already initialized buffer."""

  def get_pixel_type(self) -> str:
    """The scalar type (the fake image API controls channel count)."""
    return "Float32"

  def bake_texture(self, *args: object, **kwargs: object) -> int:
    """Keep the canvas contents and report success."""
    return 0

  def is_valid_interface(self) -> bool:
    """Match the SDK interface validity check."""
    return True


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Distiller:
  """Count CPU baker creation to witness a disk-cache hit."""

  calls: list[str]

  def create_baker(self, material: object, path: str, resource: object) -> _Baker:
    """Return a constant-expression baker."""
    self.calls.append(path)
    return _Baker()


def _api(images: _Images, calls: list[str]) -> sdk.Api:
  """Supply only the SDK interfaces exercised here at the fake binding boundary."""
  return cast(sdk.Api, SimpleNamespace(images=images, distiller=_Distiller(calls=calls), bake_on_cpu=None))


@pytest.mark.parametrize("channels", [1, 3], ids=["scalar", "float3"])
@pytest.mark.parametrize("side", [1, 2], ids=["constant", "varying-control"])
def test_bake_owns_texels_after_canvas_release(channels: int, side: int) -> None:
  """Owner release and repeated allocation reuse cannot change a returned bake."""
  images = _Images(channels=channels)
  api = _api(images, [])
  expected = np.arange(side * side * channels).astype(np.float32).reshape(side, side, channels)[::-1] / 16
  result = bake._bake(api, _Baker(), _WINDOW, side)  # Test the native ownership boundary.
  assert result is not None
  gc.collect()
  assert images.owners[0]() is None
  assert np.isnan(images.buffers[0]).all()  # the fake owner actually invalidated storage
  assert result.dtype == np.float32 and result.flags["C_CONTIGUOUS"] and result.flags["OWNDATA"]
  assert not np.shares_memory(result, images.buffers[0])
  for _ in range(32):
    bake._bake(api, _Baker(), _WINDOW, side)  # Exercise repeated canvas release.
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("channels", [1, 3], ids=["scalar", "float3"])
def test_constant_serial_prefetch_and_cache_agree(channels: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Delayed serial/prefetch consumption and repeat-cache reads preserve constants."""
  images, calls = _Images(channels=channels), []
  api = _api(images, calls)
  dag = cast(Dag, SimpleNamespace(api=api, material=None, lane=0))
  ctx = cast(ShadeContext, SimpleNamespace())
  kind: bake.Kind = "scalar" if channels == 1 else "emission"
  cached = tmp_path / "constant.npy"
  job = bake._Job(  # Isolate storage ownership from MDL expression planning.
    path="constant",
    domain=bake.Domain(window=_WINDOW, uv_set=0, rows=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), size=2, exact=True),
    kind=kind,
    mean=False,
    varies=False,
    key="constant",
    cached=cached,
    gamuts=(),
  )
  monkeypatch.setattr(bake, "_job", lambda *args, **kwargs: job)
  serial = bake.bake(ctx, dag, "constant", kind=kind, bakes=bake.Bakes(None))
  assert serial is not None
  cached.unlink()
  prefetched = bake.Bakes(None)
  with ThreadPoolExecutor(max_workers=1) as pool:
    bake.prefetch(ctx, [(dag, [("constant", kind, False)])], prefetched, pool)
  assert prefetched.made["constant"] == serial
  assert bake.bake(ctx, dag, "constant", kind=kind, bakes=bake.Bakes(None)) == serial
  assert calls == ["constant", "constant"]
  assert serial.const == tuple(float(x) / 16 for x in range(channels))
  assert all(owner() is None for owner in images.owners)
