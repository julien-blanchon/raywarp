"""`ActiveWorlds`: the worlds a render renders (`Renderer.render(active=...)`), turned on the device
into the render launch's world list and the per-world skip mask, with no host synchronisation.

The caller's (k,) int32 array of world ids (-1: a hole, nothing rendered for that entry) is read
by three small kernels each frame, all device work (graph-capturable, its content written in place
between frames):

1. `_clear`: the per-world "listed" counts and the list's entries (-1), the list length.
2. `_list`: each valid id claims a slot of the compact list `units` (an atomic counter: the
   listed worlds first, the holes after them). An id out of [0, nworld) is not rendered, a
   repeated id only once (one of its entries); either sets `invalid`, which stays set until the
   validation reads it (`Word.WORLD_IDS`, raised at the next validated render, also with
   `tuning.validate_every` > 1).
3. `_mask`: the skip mask (not listed: skipped) and the worlds rendered again after a skip
   (`resumed`, whose temporal history the Renderer restarts).

The render launch iterates `units` (k units: thread -> unit -> world, `camera.launch_world`), so a
short list launches fewer threads and the listed worlds share warps whatever the holes. The
deferred display, its metering and exposure solve launch over the same units
(`exposure.source.Source.worlds`), and RenderOrder.AUTO compares the cameras of the listed worlds
only (`cameras._camera_order`); the other per-world passes read the skip mask
(`structs.skipped`). Without an active array nothing of this runs: the launch maps units to
worlds one to one (an empty `units`) and nothing is skipped.
"""

from __future__ import annotations

from typing import final

import warp as wp

from raywarp.core.pipeline.platform.launches import CachedLaunch

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _clear(
  # Out:
  listed: wp.array1d[wp.int32],
  units: wp.array1d[wp.int32],
  count: wp.array1d[wp.int32],
):
  """No world listed, every list entry a hole (one thread per world or entry)."""
  i = wp.tid()
  if i < listed.shape[0]:
    listed[i] = 0
  if i < units.shape[0]:
    units[i] = -1
  if i == 0:
    count[0] = 0


@wp.kernel
def _list(
  # In:
  ids: wp.array1d[wp.int32],
  # Out:
  listed: wp.array1d[wp.int32],
  units: wp.array1d[wp.int32],
  count: wp.array1d[wp.int32],
  invalid: wp.array1d[wp.int32],
):
  """Entry i of the caller's ids: a valid world first listed here takes the next list slot; an
  invalid or repeated one sets `invalid` (cleared by the validation that reports it)."""
  i = wp.tid()
  w = ids[i]
  if w < 0:
    return
  if w >= listed.shape[0] or wp.atomic_add(listed, w, 1) != 0:
    invalid[0] = 1
    return
  units[wp.atomic_add(count, 0, 1)] = w


@wp.kernel
def _mask(
  # In:
  listed: wp.array1d[wp.int32],
  # In / Out:
  skip: wp.array1d[bool],
  # Out:
  resumed: wp.array1d[bool],
):
  """World w's skip flag for the frame (not listed); `resumed`: skipped the frame before, rendered
  now."""
  w = wp.tid()
  now = listed[w] == 0
  resumed[w] = skip[w] and not now
  skip[w] = now


@final
class ActiveWorlds:
  """The active-world state of one Renderer (see the module doc).

  Attributes:
    ids: the caller's world-id array of the last render (None: every world).
    units: (k,) the render launch's worlds, the listed ones first (-1 after them); empty: every
      world in order.
    skip: (nworld,) the frame's skipped worlds, or the Renderer's (1,) "none" placeholder.
    resumed: (nworld,) the worlds rendered this frame and skipped the frame before.
    invalid: (1,) nonzero when the ids of a frame since the last validation held one out of
      [0, nworld) or a repeat (the validation clears it: `validator_kernels.take_count`).
  """

  def __init__(self, nworld: int, none: wp.array[bool, [int]]) -> None:
    """Every world active (`none`: the (1,) all-false skip placeholder)."""
    self.nworld = nworld
    self.device = none.device
    self._none = none
    self.ids: wp.array[wp.int32, [int]] | None = None
    self.units = wp.zeros(0, dtype=wp.int32, device=self.device)
    self.skip = none
    self.resumed = wp.zeros(nworld, dtype=bool, device=self.device)
    self.invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
    self._listed = wp.zeros(nworld, dtype=wp.int32, device=self.device)
    self._count = wp.zeros(1, dtype=wp.int32, device=self.device)
    self._launches = (CachedLaunch(), CachedLaunch(), CachedLaunch())

  def use(self, ids: wp.array[wp.int32, [int]] | None) -> bool:
    """Takes the render's ids (None: every world); returns whether the launch's arrays changed
    (a list length or the skip table: the kernel structs and graphs are stale).

    Raises:
      TypeError: for something else than a Warp array.
      ValueError: for an array that is not a 1-D int32 one of 1 to nworld entries on the device.
    """
    if ids is not None:
      self._check(ids)
    self.ids = ids
    if ids is None:
      changed = self.units.shape[0] != 0 or self.skip is not self._none
      self.units = wp.zeros(0, dtype=wp.int32, device=self.device) if self.units.shape[0] else self.units
      self.skip = self._none
      return changed
    changed = self.units.shape[0] != ids.shape[0] or self.skip is self._none
    if self.units.shape[0] != ids.shape[0]:
      self.units = wp.zeros(ids.shape[0], dtype=wp.int32, device=self.device)
    if self.skip is self._none:
      self.skip = wp.zeros(self.nworld, dtype=bool, device=self.device)  # every world rendered before
    return changed

  def _check(self, ids: object) -> None:
    """Raises for render ids that are not a (k,) int32 device array, 1 <= k <= nworld."""
    if not isinstance(ids, wp.array):
      raise TypeError(f"active must be a warp int32 array of world ids, got {type(ids).__name__}")
    if ids.dtype != wp.int32 or ids.ndim != 1 or not 1 <= ids.shape[0] <= self.nworld:
      raise ValueError(
        f"active must be a 1-D int32 array of 1 to {self.nworld} world ids, got {wp.types.type_repr(ids.dtype)} {ids.shape}"
      )
    if ids.device != self.device:
      raise ValueError(f"the active array is on {ids.device}, the renderer renders on {self.device}")

  def update(self) -> None:
    """The frame's list, skip mask and resumed worlds from the ids' current content (device work;
    nothing without ids)."""
    if self.ids is None:
      return
    clear, listing, mask = self._launches
    outputs, dev = [self._listed, self.units, self._count, self.invalid], self.device
    clear.launch(_clear, (max(self.nworld, self.units.shape[0]),), inputs=[], outputs=outputs[:3], device=dev)
    listing.launch(_list, (self.ids.shape[0],), inputs=[self.ids], outputs=outputs, device=dev)
    mask.launch(_mask, (self.nworld,), inputs=[self._listed, self.skip], outputs=[self.resumed], device=dev)
