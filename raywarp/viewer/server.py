"""The viewer's HTTP server (standard library only): the page, a JSON API and the frame stream.

Frames are pushed over one WebSocket per page (`/api/stream`, `stream.py`): the page sends its
camera, the server renders the newest one and sends the JPEG, within a credit window of frames
not yet acknowledged; status messages tell what a busy render thread waits for (a scene load, a
kernel compile, an OptiX module creation). Commands (`/api/command`) and frames run on one render
thread, in arrival order: a command lands between two frames. Everything is plain HTTP on one
port: an SSH port forward carries it.

Routes: GET `/` (the page) and its static files; GET `/api/state`; GET `/api/events?since=N` (the
compile events after event N); GET `/api/stream` (the WebSocket); POST `/api/command` {type, ...}.
The API needs the session token (`X-Viewer-Token` header, or `?token=` on the page URL, which the
page passes on; the WebSocket, which cannot carry a header from a browser, takes `?token=`),
unless the server was started without one.
"""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import secrets
import socket
import socketserver
import threading
import time
import urllib.parse
from collections.abc import Callable
from importlib import resources
from typing import Final

from raywarp.viewer import websocket
from raywarp.viewer.session import Session
from raywarp.viewer.stream import Stream
from raywarp.viewer.watch import WATCH

# Static files served, name -> content type (the page has no build step).
STATIC: Final = {
  "index.html": "text/html; charset=utf-8",
  "app.js": "text/javascript; charset=utf-8",
  "style.css": "text/css; charset=utf-8",
}
# Largest request body accepted [bytes] (a settings patch or a camera).
MAX_BODY: Final = 1 << 20
TOKEN_HEADER: Final = "X-Viewer-Token"


class RenderThread:
  """Runs the session's work on one thread, in arrival order (one CUDA context, one renderer), and
  tells what it is busy with (`busy`, from any thread)."""

  def __init__(self, session: Session) -> None:
    """Starts the thread."""
    self.session = session
    self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="rw_viewer_render")
    self._lock = threading.Lock()
    self._busy: tuple[str, float] | None = None

  def submit[T](self, work: Callable[[Session], T], what: str) -> concurrent.futures.Future[T]:
    """Queues `work(session)` (`what`: what `busy` calls it); its future."""

    def run(session: Session) -> T:
      """`work`, noted as the thread's business while it runs."""
      with self._lock:
        self._busy = (what, time.monotonic())
      try:
        return work(session)
      finally:
        with self._lock:
          self._busy = None

    return self._pool.submit(run, self.session)

  def run[T](self, work: Callable[[Session], T], what: str) -> T:
    """`work(session)` on the render thread (waits; re-raises its exception)."""
    return self.submit(work, what).result()

  def busy(self) -> tuple[str, float] | None:
    """(what the thread runs, for how many seconds), None when idle."""
    with self._lock:
      busy = self._busy
    return None if busy is None else (busy[0], time.monotonic() - busy[1])

  def close(self) -> None:
    """Releases the renderer and stops the thread."""
    self.run(Session.close, "close")
    self._pool.shutdown(wait=True)


def _static(name: str) -> bytes:
  """The packaged static file `name`."""
  return resources.files("raywarp.viewer").joinpath("static", name).read_bytes()


def _events(since: int) -> list[dict[str, object]]:
  """The compile events after event `since`, as JSON objects."""
  return [e.json() for e in WATCH.since(since)]


class Handler(http.server.BaseHTTPRequestHandler):
  """One request (the server's `render` and `token` attributes)."""

  protocol_version = "HTTP/1.1"

  def _viewer(self) -> ViewerServer:
    """The server (a `ViewerServer`)."""
    if not isinstance(self.server, ViewerServer):
      raise TypeError("Handler serves a ViewerServer")
    return self.server

  def log_message(self, format: str, *args: object) -> None:  # the base class's signature
    """Quiet: the frames would flood the terminal (errors go to the browser)."""

  def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
    """One complete response. An error closes the connection: its request body may be unread."""
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    if status != 204:  # a 204 has no body and no Content-Length (RFC 9110 8.6)
      self.send_header("Content-Length", str(len(body)))
    if status >= 400:
      self.send_header("Connection", "close")
      self.close_connection = True
    self.send_header("Cache-Control", "no-store")
    for k, v in (headers or {}).items():
      self.send_header(k, v)
    self.end_headers()
    self.wfile.write(body)

  def _json(self, status: int, value: object) -> None:
    """A JSON response."""
    self._send(status, json.dumps(value).encode(), "application/json")

  def _authorized(self, query: str = "") -> bool:
    """Whether the request carries the session token, in its header or (`query`: the URL's query,
    the WebSocket's) `?token=` (always, without one)."""
    token = self._viewer().token
    if not token:
      return True
    given = self.headers.get(TOKEN_HEADER) or urllib.parse.parse_qs(query).get("token", [""])[0]
    if secrets.compare_digest(given.encode(), token.encode()):
      return True
    self._json(403, {"error": "missing or wrong viewer token: open the URL the server printed"})
    return False

  def _body(self) -> object:
    """The request's JSON body.

    Raises:
      ValueError: for a missing, oversized or malformed body.
    """
    length = int(self.headers.get("Content-Length") or 0)
    if not 0 < length <= MAX_BODY:
      raise ValueError(f"a JSON body of 1 to {MAX_BODY} bytes is required")
    return json.loads(self.rfile.read(length))

  def do_GET(self) -> None:  # the base class's dispatch name
    """The page, its files, the state and the events."""
    url = urllib.parse.urlsplit(self.path)
    name = "index.html" if url.path == "/" else url.path.lstrip("/")
    if name in STATIC:
      self._send(200, _static(name), STATIC[name])
      return
    if name == "favicon.ico":
      self._send(204, b"", "image/x-icon")
      return
    if not url.path.startswith("/api/"):
      self._json(404, {"error": f"no such page {url.path!r}"})
      return
    if not self._authorized(url.query if url.path == "/api/stream" else ""):
      return
    if url.path == "/api/stream":
      self._stream()
    elif url.path == "/api/state":
      self._reply(lambda s: s.state(), "state")
    elif url.path == "/api/events":
      query = urllib.parse.parse_qs(url.query)
      try:
        since = int(query.get("since", ["0"])[0])
      except ValueError:
        self._json(400, {"error": "since must be an integer"})
        return
      self._json(200, {"events": _events(since), "compiles": WATCH.counts()})
    else:
      self._json(404, {"error": f"no such endpoint {url.path!r}"})

  def do_POST(self) -> None:  # the base class's dispatch name
    """A command."""
    path = urllib.parse.urlsplit(self.path).path
    if not self._authorized():
      return
    try:
      body = self._body()
    except ValueError as e:
      self._json(400, {"error": str(e)})
      return
    if path == "/api/command":
      what = body.get("type") if isinstance(body, dict) else None
      self._reply(lambda s: s.command(body), str(what))
    else:
      self._json(404, {"error": f"no such endpoint {path!r}"})

  def _reply(self, work: Callable[[Session], dict[str, object]], what: str) -> None:
    """`work`'s JSON on the render thread (`what`: its name); a ValueError is the user's error
    (400), anything else a server error (500) that leaves the server running."""
    try:
      self._json(200, self._viewer().render.run(work, what))
    except ValueError as e:
      self._json(400, {"error": str(e)})
    except Exception as e:  # noqa: BLE001  # reported to the browser, the server keeps serving
      self._json(500, {"error": f"{type(e).__name__}: {e}"})

  def _stream(self) -> None:
    """The WebSocket handshake, then the frame stream (`stream.Stream`) until the page closes."""
    key = self.headers.get("Sec-WebSocket-Key", "")
    if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
      self._json(400, {"error": "/api/stream is a WebSocket"})
      return
    self.send_response(101)
    self.send_header("Upgrade", "websocket")
    self.send_header("Connection", "Upgrade")
    self.send_header("Sec-WebSocket-Accept", websocket.accept_key(key))
    self.end_headers()
    self.wfile.flush()
    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # acks and small frames go at once
    viewer = self._viewer()
    Stream(self.rfile, self.wfile, viewer.render, viewer.stopping).run()
    self.close_connection = True


class ViewerServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
  """The HTTP server: a thread per connection, the device work on `render`'s single thread."""

  daemon_threads = True
  allow_reuse_address = True

  def __init__(self, address: tuple[str, int], render: RenderThread, token: str) -> None:
    """Binds `address` (port 0: any free port)."""
    super().__init__(address, Handler)
    self.render = render
    self.token = token
    self.stopping = threading.Event()  # ends the streams


def serve(server: ViewerServer, stop: threading.Event | None = None) -> None:
  """Serves until `stop` is set or the process is interrupted, then releases the renderer."""
  thread = threading.Thread(target=server.serve_forever, name="rw_viewer_http", daemon=True)
  thread.start()
  try:
    (stop or threading.Event()).wait()
  except KeyboardInterrupt:
    pass
  finally:
    server.stopping.set()
    server.shutdown()
    server.server_close()
    server.render.close()
