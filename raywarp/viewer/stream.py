"""The frame stream: one WebSocket per page (`/api/stream`) carries the frames to the page and the
camera, the live switch and the frame acknowledgements back.

The server pushes frames without waiting for a request: the frame rate is bound by the renderer
and the link's bandwidth, not by its round trip. The camera is the latest one received (cameras
are never queued: each frame renders the newest), and a credit window (`Window`) caps the frames
sent and not yet acknowledged, so that frames never pile up in the link: the window grows while
the round trip shows no queue and shrinks when it does (a delay-based control, TCP Vegas's rule
in frames), so a camera move shows after about one round trip at any bandwidth.

Page -> server (text, JSON; any subset of the keys): `camera` (the fly camera; `seq`, its number,
echoed by the frame rendered from it), `live` (render continuously, else only for a new camera or
`refresh`), `refresh` (one frame: after a command), `ack` (the last frame shown: every frame up to
it is acknowledged). Server -> page: a frame is a binary message, a 4-byte big-endian length, the
frame's info (JSON) and the JPEG; a status is a text message `{"type": "status", ...}`: what the
render thread is busy with, the compiles and module creations running (`watch.running`) and the
background warm, sent every `STATUS_S` while any of it runs (a render thread blocked by a compile
or a module creation is never a black screen); an error is `{"type": "error", "error": ...}`.
"""

from __future__ import annotations

import collections
import json
import math
import struct
import threading
import time
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np

from raywarp.viewer import display, websocket
from raywarp.viewer.watch import WATCH, Kind

if TYPE_CHECKING:
  from raywarp.viewer.server import RenderThread

# The JPEG quality of the streamed frames.
JPEG_QUALITY: Final = 88
# Seconds between two status messages while something runs (and the stream's longest sleep).
STATUS_S: Final = 0.25
# A compile or module creation is shown once it ran this long [s] (a cache hit takes milliseconds).
SHOW_AFTER_S: Final = 0.3


class Status(TypedDict):
  """A status message (`status_json`).

  Attributes:
    type: "status".
    busy: what the render thread runs and for how many seconds (None: idle, or for less than
      `SHOW_AFTER_S`).
    running: the compiles and module creations running (`CompileWatch.running`).
    compiles: the compile counts per kind so far.
    events: the compile events so far.
    warm: the background warm (`Warmer.status`).
  """

  type: str
  busy: dict[str, object] | None
  running: list[dict[str, object]]
  compiles: dict[Kind, int]
  events: int
  warm: dict[str, object]


class Window:
  """The frames a stream may have sent and not yet seen acknowledged (`size`, in frames).

  Once per round trip, the frames queued in the link are estimated from the delivery rate and
  the round trip's excess over the smallest one seen (Little's law): fewer than `LOW` while the
  window was full grows the window by one, more than `HIGH` shrinks it by one; between `MIN` and
  `MAX`. `MAX` bounds the frames in flight when the round trip is long (16 frames over 250 ms: 64
  frames/s).
  """

  START: Final = 2
  MIN: Final = 1
  MAX: Final = 16
  LOW: Final = 1.0
  HIGH: Final = 2.0

  def __init__(self) -> None:
    """A window of `START` frames, nothing in flight."""
    self.size = self.START
    self.rtt_min = math.inf
    self.rtt = 0.0
    self._sent: dict[int, float] = {}
    self._acks: collections.deque[tuple[float, int]] = collections.deque()  # (time, frames acknowledged)
    self._samples: list[float] = []
    self._full = False
    self._next = 0.0

  @property
  def inflight(self) -> int:
    """Frames sent and not acknowledged."""
    return len(self._sent)

  def open(self) -> bool:
    """Whether one more frame may be sent."""
    return len(self._sent) < self.size

  def sent(self, frame: int, now: float) -> None:
    """Notes frame `frame` sent at `now` [s]."""
    self._sent[frame] = now
    self._full = self._full or len(self._sent) >= self.size

  def acked(self, frame: int, now: float) -> None:
    """Notes every frame up to `frame` acknowledged at `now` [s] (an unknown frame: ignored)."""
    start = self._sent.get(frame)
    if start is None:
      return
    done = [f for f in self._sent if f <= frame]
    for f in done:
      del self._sent[f]
    rtt = now - start
    self.rtt_min = min(self.rtt_min, rtt)
    self._samples.append(rtt)
    self._acks.append((now, len(done)))
    while len(self._acks) > 2 and self._acks[0][0] < now - 1.0:
      self._acks.popleft()
    if now >= self._next:
      self._adapt(now)

  def rate(self) -> float:
    """Frames acknowledged per second over the last second (an acknowledgement covers the frames
    the page skipped; 0: fewer than two acknowledgements)."""
    if len(self._acks) < 2:
      return 0.0
    frames = sum(n for _, n in list(self._acks)[1:])
    return frames / max(self._acks[-1][0] - self._acks[0][0], 1e-6)

  def _adapt(self, now: float) -> None:
    """Grows or shrinks the window by the frames queued over the last round trip."""
    self.rtt = min(self._samples)
    queued = self.rate() * (self.rtt - self.rtt_min)
    if queued > self.HIGH:
      self.size = max(self.MIN, self.size - 1)
    elif queued < self.LOW and self._full:
      self.size = min(self.MAX, self.size + 1)
    self._samples, self._full, self._next = [], len(self._sent) >= self.size, now + self.rtt


def frame_message(info: dict[str, object], jpeg: bytes) -> bytes:
  """A frame's binary message: the info's length (4 bytes, big-endian), its JSON, the JPEG."""
  head = json.dumps(info).encode()
  return struct.pack(">I", len(head)) + head + jpeg


def parse_frame(message: bytes) -> tuple[dict[str, object], bytes]:
  """(info, JPEG) of a frame message (`frame_message`)."""
  n = struct.unpack(">I", message[:4])[0]
  return json.loads(message[4 : 4 + n]), message[4 + n :]


class Stream:
  """One page's stream (see the module doc): a reader thread for the page's messages, the
  caller's thread rendering and sending (`run`)."""

  def __init__(
    self, rfile: websocket.ByteStream, wfile: websocket.ByteStream, render: RenderThread, stop: threading.Event
  ) -> None:
    """A stream over an opened WebSocket (`rfile` / `wfile`)."""
    self._rfile, self._wfile, self._render, self._stop = rfile, wfile, render, stop
    self._cond = threading.Condition()
    self._send_lock = threading.Lock()
    self.window = Window()
    self._camera: object = None
    self._seq = 0
    self._live = False
    self._wanted = False
    self._closed = False
    self._frames = 0
    self._status_at = 0.0
    self._status_shown = False

  # Page -> server --------------------------------------------------------------------------------

  def _read(self) -> None:
    """The reader thread: the page's messages until it closes."""
    try:
      while True:
        opcode, payload = websocket.read(self._rfile)
        if opcode == websocket.CLOSE:
          self._send(websocket.CLOSE, payload[:2])
          break
        if opcode == websocket.PING:
          self._send(websocket.PONG, payload)
        elif opcode == websocket.TEXT:
          self._message(json.loads(payload))
    except (OSError, ValueError, TypeError):  # the socket ended or the page sent garbage: the stream ends
      pass
    finally:
      with self._cond:
        self._closed = True
        self._cond.notify_all()

  def _message(self, message: object) -> None:
    """One page message (see the module doc; unknown keys ignored, a non-object ends the stream)."""
    if not isinstance(message, dict):
      raise ValueError("a stream message is a JSON object")
    with self._cond:
      if "camera" in message:
        self._camera, self._seq, self._wanted = message["camera"], int(message.get("seq") or 0), True
      if "live" in message:
        self._live = bool(message["live"])
        self._wanted = self._wanted or self._live
      if message.get("refresh"):
        self._wanted = True
      if "ack" in message:
        self.window.acked(int(message["ack"]), time.monotonic())
      self._cond.notify_all()

  # Server -> page --------------------------------------------------------------------------------

  def _send(self, opcode: int, payload: bytes) -> None:
    """One message to the page (from either thread)."""
    with self._send_lock:
      websocket.write(self._wfile, opcode, payload, masked=False)

  def _status(self, force: bool = False) -> None:
    """A status message when something runs (every `STATUS_S`), and one more when it stopped."""
    now = time.monotonic()
    if now - self._status_at < STATUS_S and not force:
      return
    status = status_json(self._render)
    shown = bool(status["busy"] or status["running"] or status["warm"].get("running"))
    if shown or self._status_shown or force:
      self._send(websocket.TEXT, json.dumps(status).encode())
      self._status_at = now
    self._status_shown = shown

  def _next(self) -> tuple[object, int] | None:
    """Waits (at most `STATUS_S`) for a frame to render: (camera, its seq), or None."""
    with self._cond:
      self._cond.wait_for(lambda: self._ending() or self._due(), STATUS_S)
      if self._ending() or not self._due():
        return None
      self._wanted = False
      return self._camera, self._seq

  def _ending(self) -> bool:
    """Whether the page closed or the server stops."""
    return self._closed or self._stop.is_set()

  def _due(self) -> bool:
    """Whether a frame is wanted and the window lets it go (under `_cond`)."""
    return self.window.open() and (self._live or self._wanted)

  def _frame(self, camera: object, seq: int) -> None:
    """Renders one frame from `camera` and sends it (statuses meanwhile); a failed frame is an
    error message and waits for the next camera or refresh (no retry loop)."""
    future = self._render.submit(lambda s: s.render(camera), "frame")
    while True:
      try:
        image, info = future.result(STATUS_S)
        break
      except TimeoutError:
        self._status()
      except Exception as e:  # noqa: BLE001  # the page shows it; the stream keeps serving
        with self._cond:
          self._live = False
        self._send(websocket.TEXT, json.dumps({"type": "error", "error": f"frame failed: {type(e).__name__}: {e}"}).encode())
        return
    start = time.perf_counter()
    jpeg = display.encode(np.ascontiguousarray(image), "jpeg", JPEG_QUALITY)
    self._frames += 1
    w = self.window
    info |= {"id": self._frames, "seq": seq, "encode_ms": (time.perf_counter() - start) * 1e3, "bytes": len(jpeg)}
    info |= {"window": w.size, "inflight": w.inflight + 1, "rtt_ms": w.rtt * 1e3, "rate": w.rate()}
    with self._cond:
      w.sent(self._frames, time.monotonic())
    self._send(websocket.BINARY, frame_message(info, jpeg))

  def run(self) -> None:
    """Streams until the page closes or the server stops."""
    reader = threading.Thread(target=self._read, name="rw_viewer_stream_read", daemon=True)
    reader.start()
    try:
      self._status(force=True)
      while not self._ending():
        found = self._next()
        if found is not None:
          self._frame(*found)
        self._status()
    except OSError:  # the page went away mid-send
      pass


def status_json(render: RenderThread) -> Status:
  """The status message of `render`'s thread, its compiles and its background warm (`Status`)."""
  busy = render.busy()
  return {
    "type": "status",
    "busy": None if busy is None or busy[1] < SHOW_AFTER_S else {"what": busy[0], "seconds": busy[1]},
    "running": WATCH.running(SHOW_AFTER_S),
    "compiles": WATCH.counts(),
    "events": WATCH.total(),
    "warm": render.session.warmer.status(),
  }
