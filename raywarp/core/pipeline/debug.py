"""`RendererDebug` (`Renderer.debug`): the read-only views of a renderer's internal state, for
diagnostics, tests and tooling: one frame's kernel-field arrays, the temporal history length, the
GI probe volume and its re-bake log, the captured-graph version. None of it is needed to render."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
  import warp as wp

  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.renderer import Renderer
  from raywarp.core.shade import gi as gi_lib


@final
class RendererDebug[S]:
  """The debug views of one renderer (`Renderer.debug`); each reads the renderer's current state."""

  def __init__(self, renderer: Renderer[S]) -> None:
    """The views of `renderer`."""
    self._r = renderer

  def frame(self, sim: S | None = None) -> Frame:
    """The kernel-field arrays of a call for `sim` (what `render(sim)` reads), without rendering."""
    return self._r._frame(sim)  # noqa: SLF001  # the renderer's debug view

  @property
  def history_length(self) -> wp.array[int, [int, int]] | None:
    """(nworld, ncam) int frames of history (the motion output), None without history."""
    return self._r._history_length()  # noqa: SLF001  # the renderer's debug view

  @property
  def gi_volume(self) -> gi_lib.ProbeVolume | None:
    """The GI probe volume (None before the first render or without GI)."""
    return self._r._gi_volume()  # noqa: SLF001  # the renderer's debug view

  def gi_rebakes(self) -> gi_lib.GIRebakes:
    """The GI's re-bake log so far (`gi.GIRebakes`: each bake with its reason, the probe updates
    of `update_gi`, and per world the re-lights of its tint; a host read). Moving instances,
    dynamic lights (`PerWorld.dynamic_lights`) and cameras never add to it."""
    return self._r._gi_rebakes()  # noqa: SLF001  # the renderer's debug view

  def graph_version(self) -> int:
    """Changes whenever a captured graph becomes stale (`graph.Replayable`): a change of the
    renderer, or a light grid / GI volume an eager frame replaced by itself."""
    return self._r._graph_version()  # noqa: SLF001  # the renderer's debug view
