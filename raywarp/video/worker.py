"""An encoder process of `Recorder` (`python -m raywarp.video.worker`, started by the recorder).

Encoding runs in its own processes so that it never holds the render process's GIL: an NVENC
`Encode` call or a 64-session loop in a thread of the render process slowed a host-bound render
loop by 10-20 % (RTX PRO 6000, showcase room). The process maps the recorder's NV12 ring and
its "ready" events through CUDA IPC (a CPU-device ring: a memory-mapped file) and encodes the output
frames it owns.

Its messages: `raywarp/video/protocol.py`.
"""

from __future__ import annotations

import pickle
import sys
import traceback

import numpy as np
import warp as wp

from raywarp.video.encoders import Encoder, NvencEncoder, Ring, X264Encoder
from raywarp.video.protocol import DONE, FINISH, READY, Job, read_exact, read_int, send_error, send_int


def open_ring(job: Job) -> Ring:
  """The recorder's ring in this process."""
  if isinstance(job.ring, str):
    host = np.memmap(job.ring, np.uint8, "r+", shape=job.shape)
    return wp.array(host, dtype=wp.uint8, device="cpu", copy=False)
  ring = wp.from_ipc_handle(job.ring, wp.uint8, job.shape, device=job.device)
  if not ring.ptr:
    raise RuntimeError("could not open the recorder's ring (CUDA IPC)")
  return ring


def make_encoder(job: Job, ring: Ring) -> Encoder:
  """The job's back end."""
  if job.backend == "nvenc":
    return NvencEncoder(ring, job.frames, job.paths, job.enc)
  return X264Encoder(ring, job.frames, job.paths, job.enc, job.threads)


def serve(job: Job, fd_in: int, fd_out: int) -> None:
  """Opens the ring and the encoder, then encodes each slot read from `fd_in` until FINISH."""
  ring = open_ring(job)
  events = [wp.event_from_ipc_handle(h, device=job.device) for h in job.events]
  encoder = make_encoder(job, ring)
  send_int(fd_out, READY)
  while (slot := read_int(fd_in)) != FINISH:
    encoder.encode(slot, events[slot] if events else None)
    send_int(fd_out, slot)
  encoder.close()
  send_int(fd_out, DONE)


def main() -> None:
  """Reads the job and serves it; any failure is reported on stdout and ends the process."""
  fd_in, fd_out = sys.stdin.fileno(), sys.stdout.fileno()
  sys.stdout = sys.stderr  # prints go to stderr: stdout is the reply channel
  try:
    job = pickle.loads(read_exact(fd_in, read_int(fd_in)))
    with wp.ScopedDevice(job.device):
      serve(job, fd_in, fd_out)
  except BaseException:  # noqa: BLE001  (reported to the recorder, which raises it)
    send_error(fd_out, traceback.format_exc())


if __name__ == "__main__":
  main()
