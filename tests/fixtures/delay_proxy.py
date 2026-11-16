"""A TCP proxy that adds a fixed delay each way and optionally caps the bandwidth: a long, thin
link (an SSH tunnel across an ocean) on one machine, for measuring the viewer's stream.

  python tests/fixtures/delay_proxy.py --listen 9000 --to 127.0.0.1:8080 --delay-ms 70 [--mbps 20]

Each chunk read on one side is written to the other `delay` seconds later, in order; with a cap,
a direction's writes are paced to that many megabits per second (the queue they wait in is the
link's buffer).
"""

from __future__ import annotations

import argparse
import queue
import socket
import threading
import time
from typing import Final

# Bytes read at once per direction.
_CHUNK: Final = 64 << 10


def _pipe(src: socket.socket, dst: socket.socket, delay: float, rate: float) -> None:
  """Copies `src` to `dst`, each chunk `delay` seconds late, at most `rate` bytes/s (0: no cap)."""
  pending: queue.Queue[tuple[float, bytes]] = queue.Queue()

  def reader() -> None:
    """Timestamps what `src` sends."""
    try:
      while data := src.recv(_CHUNK):
        pending.put((time.monotonic() + delay, data))
    except OSError:
      pass
    pending.put((0.0, b""))

  threading.Thread(target=reader, daemon=True).start()
  free = time.monotonic()  # when the capped link is free again
  try:
    while True:
      due, data = pending.get()
      if not data:
        break
      if rate > 0:
        due = max(due, free)
        free = due + len(data) / rate
      time.sleep(max(0.0, due - time.monotonic()))
      dst.sendall(data)
  except OSError:
    pass
  for s in (src, dst):
    try:
      s.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass


class DelayProxy:
  """A running proxy (`port`: where it listens) to `target`."""

  def __init__(self, target: tuple[str, int], delay_s: float, mbps: float = 0.0, port: int = 0) -> None:
    """Listens on 127.0.0.1:`port` (0: any free port) and serves in the background."""
    self.target, self.delay, self.rate = target, delay_s, mbps * 1e6 / 8
    self._server = socket.create_server(("127.0.0.1", port))
    self.port = self._server.getsockname()[1]
    threading.Thread(target=self._serve, daemon=True).start()

  def _serve(self) -> None:
    """Accepts connections until closed."""
    while True:
      try:
        client, _ = self._server.accept()
      except OSError:
        return
      upstream = socket.create_connection(self.target)
      for s in (client, upstream):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      threading.Thread(target=_pipe, args=(client, upstream, self.delay, self.rate), daemon=True).start()
      threading.Thread(target=_pipe, args=(upstream, client, self.delay, self.rate), daemon=True).start()

  def close(self) -> None:
    """Stops accepting (open connections end with their peers)."""
    self._server.close()


def main() -> None:
  """Runs a proxy until interrupted."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--listen", type=int, required=True)
  p.add_argument("--to", required=True, help="HOST:PORT")
  p.add_argument("--delay-ms", type=float, default=70.0, help="added each way")
  p.add_argument("--mbps", type=float, default=0.0, help="bandwidth cap per direction (0: none)")
  args = p.parse_args()
  host, port = args.to.rsplit(":", 1)
  DelayProxy((host, int(port)), args.delay_ms / 1e3, args.mbps, args.listen)
  print(f"proxy 127.0.0.1:{args.listen} -> {args.to}, +{args.delay_ms} ms each way", flush=True)
  threading.Event().wait()


if __name__ == "__main__":
  main()
