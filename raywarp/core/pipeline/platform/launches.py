"""`CachedLaunch`: one eager per-frame launch site's recorded `wp.Launch`, replayed while its
arguments are unchanged (packing a launch's arguments is most of its host cost); `BandedLaunch`:
its CPU form over every core (one recorded launch per band, `device.launch_bands`); `recorded`:
the checked result of a recording `wp.launch`.

Arguments are compared by identity for arrays (the cache holds them: an id is not reused while
cached), by raw bytes for structs (a struct is passed by value and may be edited in place), by
value otherwise.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from typing import final

import warp as wp

from raywarp._base.warp_structs import struct_mirror
from raywarp.core.pipeline.platform.device import launch_bands


def struct_bytes(s: object) -> bytes:
  """Raw bytes of a Warp struct instance's ctypes mirror (array fields hold pointers and shapes)."""
  c = struct_mirror(s)
  return ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c))


def recorded(cmd: wp.Launch | None) -> wp.Launch:
  """The launch `wp.launch(..., record_cmd=True)` returned.

  Raises:
    RuntimeError: when it returned none.
  """
  if cmd is None:
    raise RuntimeError("wp.launch(record_cmd=True) returned no recorded launch")
  return cmd


def _key(arg: object) -> object:
  """The comparison key of one launch argument (see the module doc)."""
  if isinstance(arg, wp.array):
    return id(arg)
  if wp.types.is_struct(arg):
    return struct_bytes(arg)
  return arg


@final
class CachedLaunch:
  """The recorded launch of one call site (see the module doc)."""

  def __init__(self) -> None:
    """Nothing recorded yet."""
    self._cached: tuple[tuple[object, ...], tuple[object, ...], wp.Launch] | None = None

  def launch(
    self,
    kernel: wp.Kernel | Callable[..., None],
    dim: tuple[int, ...],
    *,
    inputs: Sequence[object],
    outputs: Sequence[object] = (),
    device: wp.Device | str | None,
    block_dim: int = 256,
  ) -> None:
    """`wp.launch(kernel, dim, inputs, outputs, device, block_dim)`, replaying the recorded launch
    when nothing changed since the last call."""
    args = (*inputs, *outputs)
    key = (kernel, tuple(dim), block_dim, str(device), *(_key(a) for a in args))
    cached = self._cached
    if cached is not None and cached[0] == key:
      cached[2].launch()
      return
    cmd = recorded(
      wp.launch(
        kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), device=device, block_dim=block_dim, record_cmd=True
      )
    )
    cmd.launch()
    self._cached = (key, args, cmd)


@final
class BandedLaunch:
  """`CachedLaunch` of a kernel launched in bands along its launch's last axis
  (`device.launch_bands`: one band per CPU core): the kernel takes the band's first index as its
  last input, and each band's launch is recorded and replayed (see the module doc)."""

  def __init__(self) -> None:
    """Nothing recorded yet."""
    self._bands: dict[int, CachedLaunch] = {}  # by the band's first index
    self._dim: tuple[int, ...] = ()  # the launch the bands split (another one drops them)

  def launch(
    self,
    kernel: wp.Kernel,
    dim: tuple[int, ...],
    *,
    inputs: Sequence[object],
    outputs: Sequence[object] = (),
    device: wp.Device | str | None,
    min_band: int,
    per_core: int,
  ) -> None:
    """`kernel` over `dim`, in `per_core` bands per core of at least `min_band` along its last
    axis (`device.launch_bands`): band [a, b) is the launch (*dim[:-1], b - a) with the inputs
    `inputs` + [a]."""
    head = tuple(dim[:-1])
    if tuple(dim) != self._dim:
      self._bands, self._dim = {}, tuple(dim)

    def run(a: int, b: int) -> None:
      """Band [a, b)."""
      band = self._bands.setdefault(a, CachedLaunch())
      band.launch(kernel, (*head, b - a), inputs=[*inputs, a], outputs=outputs, device=device)

    launch_bands(kernel.module, device=device, n=dim[-1], launch=run, min_band=min_band, per_core=per_core)
