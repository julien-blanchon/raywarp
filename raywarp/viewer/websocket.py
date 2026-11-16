"""The WebSocket protocol (RFC 6455), as much as the viewer's stream needs, on the standard library:
the opening handshake's accept key and the message framing of both ends (a client masks what it
sends, a server does not). Messages are text (JSON) or binary; a fragmented message is joined;
ping is answered by the reader's caller (`PING` -> `PONG`); no extension is negotiated.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import struct
from typing import BinaryIO, Final

# The handshake's GUID (RFC 6455 section 1.3).
_GUID: Final = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
# Opcodes (section 5.2).
CONTINUATION: Final = 0x0
TEXT: Final = 0x1
BINARY: Final = 0x2
CLOSE: Final = 0x8
PING: Final = 0x9
PONG: Final = 0xA
# A byte stream of a connection: a socket file (`socket.makefile`, an HTTP handler's rfile / wfile).
type ByteStream = BinaryIO | io.BufferedIOBase
# Largest message read [bytes] (the viewer's client messages are small JSON; a client reads frames).
MAX_MESSAGE: Final = 64 << 20


class ClosedError(ConnectionError):
  """The peer closed the connection (a close message, or the socket's end)."""


def accept_key(key: str) -> str:
  """The `Sec-WebSocket-Accept` value answering the client's `Sec-WebSocket-Key` `key`."""
  return base64.b64encode(hashlib.sha1(key.encode() + _GUID).digest()).decode()


def new_key() -> str:
  """A client's random `Sec-WebSocket-Key`."""
  return base64.b64encode(os.urandom(16)).decode()


def _exactly(stream: ByteStream, n: int) -> bytes:
  """`n` bytes of `stream`, buffered or not (ClosedError at its end)."""
  data = b""
  while len(data) < n:
    more = stream.read(n - len(data))
    if not more:
      raise ClosedError("the connection ended")
    data += more
  return data


def _unmask(payload: bytes, mask: bytes) -> bytes:
  """`payload` XOR the repeated 4-byte `mask` (in one big-integer XOR: no per-byte loop)."""
  if not payload:
    return payload
  n = len(payload)
  key = (mask * (n // 4 + 1))[:n]
  return (int.from_bytes(payload, "little") ^ int.from_bytes(key, "little")).to_bytes(n, "little")


def _frame(stream: ByteStream) -> tuple[bool, int, bytes]:
  """One frame of `stream`: (final, opcode, unmasked payload).

  Raises:
    ClosedError: at the stream's end.
    ValueError: for a frame over `MAX_MESSAGE` or with reserved bits set.
  """
  head = _exactly(stream, 2)
  if head[0] & 0x70:
    raise ValueError("a WebSocket frame with reserved bits (no extension was negotiated)")
  length = head[1] & 0x7F
  if length == 126:
    length = struct.unpack(">H", _exactly(stream, 2))[0]
  elif length == 127:
    length = struct.unpack(">Q", _exactly(stream, 8))[0]
  if length > MAX_MESSAGE:
    raise ValueError(f"a WebSocket frame of {length} bytes (at most {MAX_MESSAGE})")
  mask = _exactly(stream, 4) if head[1] & 0x80 else b""
  payload = _exactly(stream, length)
  return bool(head[0] & 0x80), head[0] & 0x0F, _unmask(payload, mask) if mask else payload


def read(stream: ByteStream) -> tuple[int, bytes]:
  """The next message of `stream`: (opcode, payload); fragments joined, control frames (close,
  ping, pong) returned as they come.

  Raises:
    ClosedError: at the stream's end.
    ValueError: for a malformed or oversized message.
  """
  final, opcode, payload = _frame(stream)
  if opcode >= CLOSE or final:
    return opcode, payload
  parts = [payload]
  while not final:
    final, more, payload = _frame(stream)
    if more != CONTINUATION:
      raise ValueError("a control frame or new message inside a fragmented message")
    parts.append(payload)
    if sum(len(p) for p in parts) > MAX_MESSAGE:
      raise ValueError(f"a WebSocket message over {MAX_MESSAGE} bytes")
  return opcode, b"".join(parts)


def write(stream: ByteStream, opcode: int, payload: bytes, *, masked: bool) -> None:
  """Writes one unfragmented message to `stream` and flushes (`masked`: a client's)."""
  n = len(payload)
  head = bytes([0x80 | opcode])
  bit = 0x80 if masked else 0
  if n < 126:
    head += bytes([bit | n])
  elif n < 1 << 16:
    head += bytes([bit | 126]) + struct.pack(">H", n)
  else:
    head += bytes([bit | 127]) + struct.pack(">Q", n)
  if masked:
    mask = os.urandom(4)
    head, payload = head + mask, _unmask(payload, mask)
  stream.write(head + payload)
  stream.flush()
