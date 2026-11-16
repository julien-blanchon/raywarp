"""A standard-library client of the viewer's frame stream (`raywarp/viewer/stream.py`), for the
drive script, the stream benchmark and the tests: the WebSocket handshake over a raw socket, then
the page's messages (camera, live, refresh, ack) and the server's (frames, statuses, errors).

  with StreamClient.connect("http://127.0.0.1:8080", token) as s:
    s.send(camera=camera, seq=1, refresh=True)
    info, jpeg = s.frame()  # statuses on the way are kept in s.statuses
    s.send(ack=info["id"])
"""

from __future__ import annotations

import json
import select
import socket
import time
import urllib.parse
from typing import BinaryIO, Self, cast

from raywarp.viewer import websocket
from raywarp.viewer.stream import parse_frame

type Json = dict[str, object]


class StreamClient:
  """One open stream (see the module doc)."""

  def __init__(self, sock: socket.socket) -> None:
    """A stream over `sock`, its handshake done."""
    self.sock = sock
    # unbuffered: `quiet` sees every byte not read (a raw SocketIO, read like the server's stream)
    self._rfile = cast("BinaryIO", sock.makefile("rb", buffering=0))
    self._wfile = sock.makefile("wb")
    self.statuses: list[Json] = []

  @classmethod
  def connect(cls, base: str, token: str, timeout: float = 600.0) -> Self:
    """Opens `base`'s `/api/stream` with `token`.

    Raises:
      ConnectionError: when the server refuses the upgrade (its status line in the message).
    """
    url = urllib.parse.urlsplit(base)
    sock = socket.create_connection((url.hostname or "127.0.0.1", url.port or 80), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = websocket.new_key()
    query = f"?token={urllib.parse.quote(token)}" if token else ""
    request = (
      f"GET /api/stream{query} HTTP/1.1\r\nHost: {url.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
      f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(request.encode())
    client = cls(sock)
    status = client._rfile.readline().decode()
    headers = {}
    while (line := client._rfile.readline().decode().strip()) != "":
      name, _, value = line.partition(":")
      headers[name.strip().lower()] = value.strip()
    if " 101 " not in status or headers.get("sec-websocket-accept") != websocket.accept_key(key):
      sock.close()
      raise ConnectionError(f"the stream was refused: {status.strip()}")
    return client

  def send(self, **message: object) -> None:
    """One page message (`camera`, `seq`, `live`, `refresh`, `ack`)."""
    websocket.write(self._wfile, websocket.TEXT, json.dumps(message).encode(), masked=True)

  def next(self) -> tuple[str, Json, bytes]:
    """The next server message: ("frame", info, JPEG), ("status", status, b"") or ("error", ...)."""
    while True:
      opcode, payload = websocket.read(self._rfile)
      if opcode == websocket.BINARY:
        info, jpeg = parse_frame(payload)
        return "frame", info, jpeg
      if opcode == websocket.TEXT:
        message = json.loads(payload)
        return str(message["type"]), message, b""
      if opcode == websocket.CLOSE:
        raise websocket.ClosedError("the server closed the stream")

  def quiet(self, seconds: float) -> bool:
    """Whether the server sends nothing for `seconds` (nothing is read)."""
    return not select.select([self.sock], [], [], seconds)[0]

  def frame(self, timeout: float = 600.0) -> tuple[Json, bytes]:
    """The next frame (statuses kept in `statuses`).

    Raises:
      RuntimeError: for an error message (a failed frame).
      TimeoutError: when none came within `timeout` seconds.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      kind, message, jpeg = self.next()
      if kind == "frame":
        return message, jpeg
      if kind == "error":
        raise RuntimeError(str(message["error"]))
      self.statuses.append(message)
    raise TimeoutError(f"no frame within {timeout} s")

  def close(self) -> None:
    """Closes the stream (a close message, then the socket)."""
    try:
      websocket.write(self._wfile, websocket.CLOSE, b"\x03\xe8", masked=True)
    except OSError:
      pass
    self.sock.close()

  def __enter__(self) -> Self:
    """The client (`close` on exit)."""
    return self

  def __exit__(self, *_: object) -> None:
    """Closes the stream."""
    self.close()
