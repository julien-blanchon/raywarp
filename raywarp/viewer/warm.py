"""The viewer's background warm: after a scene opens (and after a change of compile-time settings),
`Renderer.warm(..., nearby=N, nearby_outputs=...)` prepares the settings one change away and every
output view, so that the next click finds its modules ready (OptiX modules created, or the
software traversal's Warp modules compiled).

One warm runs at a time, on a thread of its own; a newer request replaces one not yet started
(the latest settings win, none queue). A warm's device work takes turns with other warms only,
never with the frames: the stream keeps going (OptiX's module creation is CPU work, on every
core). Its outcome is a status for the page (`Warmer.status`), a failure included.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import final

import raywarp as rw
from raywarp.viewer.scenes import Loaded


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Request:
  """A configuration to warm.

  Attributes:
    loaded: the scene (and the simulator state its frame renders).
    size: the image (width, height).
    fov: the camera's vertical field of view [degrees].
    settings: the settings, with the current view's outputs.
    outputs: the output set of every view (planned first).
  """

  loaded: Loaded
  size: tuple[int, int]
  fov: float
  settings: rw.RenderSettings
  outputs: tuple[rw.Outputs, ...]


@final
class Warmer:
  """Warms the latest `Request` in the background (`limit`: the modules one change away at most,
  `Renderer.warm`'s `nearby`; 0: no warm)."""

  def __init__(self, limit: int) -> None:
    """Idle; nothing warmed yet."""
    self.limit = limit
    self._lock = threading.Lock()
    self._pending: Request | None = None
    self._thread: threading.Thread | None = None
    self._status: dict[str, object] = {"running": False}  # the last warm (`status`)
    self._start = 0.0  # the running warm's start (`time.monotonic`)

  def request(self, request: Request) -> None:
    """Warms `request` next (replacing one not yet started); returns at once."""
    if self.limit <= 0:
      return
    with self._lock:
      self._pending = request
      if self._thread is None:
        self._thread = threading.Thread(target=self._run, name="rw_warm_viewer", daemon=True)
        self._thread.start()

  def status(self) -> dict[str, object]:
    """The last warm, as JSON: running, scene, seconds (so far), and when done the plan (modules
    created, their estimate) or the error."""
    with self._lock:
      status, start = dict(self._status), self._start
    if status.get("running"):
      status["seconds"] = time.monotonic() - start
    return status

  def _run(self) -> None:
    """The warm thread: the pending requests, one at a time, until none is left."""
    while True:
      with self._lock:
        request, self._pending = self._pending, None
        if request is None:
          self._thread = None
          return
        self._status = {"running": True, "scene": request.loaded.entry.name}
        self._start = time.monotonic()
      self._finish(request.loaded.entry.name, *self._warm(request))

  def _warm(self, request: Request) -> tuple[float, dict[str, object]]:
    """Warms `request` and waits: (seconds, the outcome's JSON)."""
    start = time.monotonic()
    try:
      warmed = rw.Renderer.warm(
        request.loaded.gpu,
        nworld=1,
        cameras=[rw.CameraSpec(fovy=request.fov)],
        resolution=request.size,
        settings=request.settings,
        sim=request.loaded.sim,
        nearby=self.limit,
        nearby_outputs=request.outputs,
      ).result()
    except Exception as e:  # noqa: BLE001  # shown in the page: a failed warm leaves frames as they are
      return time.monotonic() - start, {"error": f"{type(e).__name__}: {e}"}
    plan = warmed.nearby
    found: dict[str, object] = {"summary": str(warmed)}
    if plan is not None:
      found |= {"modules": plan.count, "outside": len(plan.outside), "skipped": plan.skipped}
    return time.monotonic() - start, found

  def _finish(self, scene: str, seconds: float, outcome: dict[str, object]) -> None:
    """Notes a finished warm."""
    with self._lock:
      self._status = {"running": False, "scene": scene, "seconds": seconds, **outcome}
