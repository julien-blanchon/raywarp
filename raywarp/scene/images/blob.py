"""The texel blob of a scene: where each image goes in it, and the threaded decode that fills it.

An image is read and probed as soon as it is registered (`Ahead`, on a thread pool, while the
adapter still walks its source), and decoded there too while the decoded top levels fit in
AHEAD_BYTES. At build, the remaining images are decoded largest first on a thread pool; each
worker hands its image's levels to a `Sink` that copies them into the in-memory blob or writes
them to the cache file (`os.pwrite`), both of which release the GIL.

Memory: a `ByteBudget` (a byte-weighted semaphore) bounds what workers hold, not only what is
retained. `Ahead` partitions AHEAD_BYTES between encoded read/probe buffers (at most 64 MiB)
and decoded working sets / retained top levels, so a full retained cache cannot deadlock a
reader. With read-ahead decoding disabled, only that bounded read/probe allowance remains.
The build charges retained top levels to DECODE_BYTES and drops only those that leave too
little room for the largest pending working set (`decode_bytes`: top level, mip chain, float32
filtering copies). An oversized read or decode runs alone in its respective budget; its own
size can exceed that limit. Source arrays and the final scene blob are not temporary workspace.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import os
import threading
from collections.abc import Callable, Generator, Sequence
from typing import Final, final

import numpy as np

from raywarp.scene.images.decode import ImageInfo, ImageSource, decode_base, image_nbytes, probe, probe_bytes, read_file
from raywarp.scene.images.mips import mip_chain, mip_count
from raywarp.scene.model.rayw import Dependency, align
from raywarp.scene.model.tables import DIGEST_BYTES, Images
from raywarp.scene.model.types import Texels

# Receives image i's decoded levels (on a worker thread): `sink(i, levels)`.
type Sink = Callable[[int, list[Texels]], None]
# `Ahead` read/probe buffers plus decoded working sets / retained top levels [bytes].
# Read when an `Ahead` is made (tests can lower it); 0 disables ahead decoding.
AHEAD_BYTES: Final = 4 << 30
# Read/probe portion of AHEAD_BYTES. Independent admission avoids waiting on retained images
# that cannot be released until all probes finish. Oversized files are read one at a time.
_READ_BYTES: Final = 64 << 20
# Working memory of the build's concurrent decodes [bytes] (`decode_all`).
DECODE_BYTES: Final = 4 << 30
# Worker threads by default: the CPU count, at most this.
MAX_THREADS: Final = 32
# Bytes of a float32 component: mip filtering works on float32 copies of a level.
_F32_BYTES: Final = 4


def worker_count(threads: int | None) -> int:
  """Worker count: `threads` (at least 1), else (None) the CPU count, at most MAX_THREADS.

  Raises:
    ValueError: for a `threads` below 1.
  """
  if threads is None:
    return min(MAX_THREADS, os.cpu_count() or 1)
  if threads < 1:
    raise ValueError(f"worker threads must be at least 1 (None: the CPU count), got {threads}")
  return threads


@final
class ByteBudget:
  """A byte-weighted semaphore: holders of `hold(n)` together hold at most `total` bytes.

  A request larger than `total` is clamped to it: it waits for an empty budget and then runs
  alone. `peak` is the most bytes held at once (tests and reports).
  """

  def __init__(self, total: int) -> None:
    """A budget of `total` bytes (at least 1).

    Raises:
      ValueError: for a `total` below 1.
    """
    if total < 1:
      raise ValueError(f"a byte budget must be at least 1 byte, got {total}")
    self.total = total
    self.peak = 0
    self._held = 0
    self._cond = threading.Condition()

  @property
  def held(self) -> int:
    """Bytes held now."""
    with self._cond:
      return self._held

  def _take(self, n: int) -> None:
    """Records `n` more bytes held (the caller holds the lock)."""
    self._held += n
    self.peak = max(self.peak, self._held)

  def acquire(self, n: int) -> int:
    """Waits until `n` bytes (clamped to [0, total]) are free and takes them; the bytes taken."""
    n = min(max(n, 0), self.total)
    with self._cond:
      self._cond.wait_for(lambda: self._held + n <= self.total)
      self._take(n)
    return n

  def try_acquire(self, n: int) -> bool:
    """Takes `n` bytes when they are free now (never clamped, never waits); whether it did."""
    with self._cond:
      if n < 0 or self._held + n > self.total:
        return False
      self._take(n)
    return True

  def release(self, n: int) -> None:
    """Gives back `n` bytes taken by `acquire` / `try_acquire`."""
    with self._cond:
      self._held -= n
      self._cond.notify_all()

  @contextlib.contextmanager
  def hold(self, n: int) -> Generator[None]:
    """Holds `n` bytes (clamped to [0, total]) for the `with` block."""
    taken = self.acquire(n)
    try:
      yield
    finally:
      self.release(taken)


def decode_bytes(info: ImageInfo, levels: int) -> int:
  """Estimated peak working memory of decoding an image and building its `levels` mip levels.

  The decode's own work at the source resolution first (`ImageInfo.decode_work`: the file's
  bytes, the full-size decoder output and its conversions, before any downscale). Then without
  mips: twice the top level (a decoder's output and its conversion to the stored layout).
  With mips: the top level, the mip chain, and the float32 copies `mips.downsample` filters
  with: the linear top level, its alpha in float64 and the filtered half, 2.1 float32 top levels
  measured on an sRGB RGBA8 image (tracemalloc): 2.5 are reserved.
  """
  top = image_nbytes(info, 1)
  if levels <= 1:
    return info.decode_work + 2 * top
  components = info.height * info.width * info.channels
  return info.decode_work + top + image_nbytes(info, levels) + (5 * components * _F32_BYTES) // 2


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Probed:
  """An image read ahead: its layout and, when it was decoded ahead, its top level.

  Attributes:
    info: the decoded layout (`images.probe`).
    base: the stored top level, None when left to the build (over AHEAD_BYTES).
    error: the decode error ("": none).
    state: the file's state when read (fstat of the open file; None: not read here).
  """

  info: ImageInfo
  base: Texels | None = None
  error: str = ""
  state: Dependency | None = None


def _resolved(item: concurrent.futures.Future[Probed] | ImageSource | Probed) -> Probed:
  """The result of an `Ahead` entry: a read's, an array probed now, or one already resolved."""
  if isinstance(item, Probed):
    return item
  if isinstance(item, ImageSource):
    return Probed(info=probe(item))
  return item.result()


@final
class Ahead:
  """Reads, probes and decodes image files on a thread pool as they are registered (`submit`).

  Arrays are probed at `results`, on the caller's thread: an array registered without a copy may
  still change until the build. Derived images (computed from files) are read ahead too.

  Attributes:
    budget: decoded top levels kept and decodes in flight (the decoded portion of AHEAD_BYTES).
  """

  def __init__(self) -> None:
    """Workers sharing AHEAD_BYTES between file reads and decoded pixels (0: probe only)."""
    self._pool: concurrent.futures.ThreadPoolExecutor | None = None
    self._items: list[concurrent.futures.Future[Probed] | ImageSource | Probed] = []
    read_bytes = min(_READ_BYTES, AHEAD_BYTES) if AHEAD_BYTES > 0 else _READ_BYTES
    self._reads = ByteBudget(read_bytes)
    self._decode_bytes = max(AHEAD_BYTES - read_bytes, 0)
    self.budget = ByteBudget(max(self._decode_bytes, 1))

  def submit(self, source: ImageSource) -> None:
    """Starts reading the next image (a file; an array waits for `results`)."""
    if source.array is not None:
      self._items.append(source)
      return
    if self._pool is None:
      self._pool = concurrent.futures.ThreadPoolExecutor(worker_count(None), thread_name_prefix="raywarp-image")
    self._items.append(self._pool.submit(self._work, source))

  def _work(self, source: ImageSource) -> Probed:
    """Reads `source` once, probes it, and decodes its top level when its working memory fits the
    budget now (the top level then stays reserved: it is kept until the build)."""
    with self._reads.hold(probe_bytes(source)):
      data, state = read_file(source)
      info = probe(source, data)
      working, kept = decode_bytes(info, 1), image_nbytes(info, 1)
      if info.error or working > self._decode_bytes or not self.budget.try_acquire(working):
        del data
        return Probed(info=info, state=state)
      try:
        base, error = decode_base(source, info, data)
      except BaseException:
        del data
        self.budget.release(working)  # nothing kept
        raise
      del data
      self.budget.release(working - kept)
      return Probed(info=info, base=base, error=error, state=state)

  def results(self) -> list[Probed]:
    """Every submitted image's result (in order), once all are done (arrays probed now; kept
    until `close`)."""
    self._items = [_resolved(x) for x in self._items]
    return [x for x in self._items if isinstance(x, Probed)]

  def abort(self) -> None:
    """Drops the reads and decodes not started yet without waiting for the running ones (a build
    that will not run: its layout failed, or its builder was dropped)."""
    if self._pool is not None:
      self._pool.shutdown(wait=False, cancel_futures=True)
      self._pool = None
    self._items = []

  def close(self) -> None:
    """Shuts the pool down and releases the decoded top levels (the caller's `results` list keeps
    them until written; a later build decodes again)."""
    if self._pool is not None:
      self._pool.shutdown()
      self._pool = None
    results = self.results()
    # The kept top levels leave the budget with them (the build's decode budget covers them next).
    self.budget.release(sum(image_nbytes(p.info, 1) for p in results if p.base is not None))
    self._items = [Probed(info=p.info, state=p.state) for p in results]


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class BlobLayout:
  """Where each image's levels go in the texel blob (each image ALIGN-aligned).

  Attributes:
    levels: mip levels per image (1 without mips).
    offsets: byte offset of each image in the blob (Python ints, stored as int64 in
      `Images.offset`: blobs over 2 GiB address correctly).
    total: blob size [bytes].
  """

  levels: list[int]
  offsets: list[int]
  total: int

  @staticmethod
  def of(infos: Sequence[ImageInfo], *, mips: bool) -> BlobLayout:
    """Layout of images `infos`, with full mip chains when `mips`."""
    levels = [mip_count(i.width, i.height) if mips else 1 for i in infos]
    offsets: list[int] = []
    total = 0
    for info, nlevel in zip(infos, levels, strict=True):
      offsets.append(total)
      total = align(total + image_nbytes(info, nlevel))
    return BlobLayout(levels=levels, offsets=offsets, total=total)

  def table(self, infos: Sequence[ImageInfo], names: Sequence[str]) -> Images:
    """The `Images` table of these images (its `data` blob still empty)."""
    columns = {
      "width": [i.width for i in infos],
      "height": [i.height for i in infos],
      "channels": [i.channels for i in infos],
      "pixel": [int(i.pixel) for i in infos],
      "levels": self.levels,
      "offset": self.offsets,
      "digest": [[0] * DIGEST_BYTES for _ in infos],  # set as each image is decoded
    }
    return Images.from_columns(columns, {"paths": names})

  def sink(self, put: Callable[[int, Texels], None]):
    """A `Sink` handing each decoded level to `put(byte offset in the blob, level)`."""

    def levels(i: int, arrays: list[Texels]) -> None:
      """Hands image `i`'s levels to `put` at their blob offsets."""
      offset = self.offsets[i]
      for level in arrays:
        put(offset, level)
        offset += level.nbytes

    return levels


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Compaction:
  """The texel blob with each image content stored once (`compaction`): images of equal content
  digests (`Images.digest`) share the first one's bytes, the others move down over the gaps.

  Attributes:
    offsets: each image's byte offset in the compacted blob.
    moves: (source offset, destination offset, bytes) of the kept images that move, in blob
      order: every destination lies below its source and above the previous move's end, so the
      moves run in place one after another (`move` forward).
    total: the compacted blob's size [bytes].
    shared: images stored as another one's bytes.
  """

  offsets: list[int]
  moves: list[tuple[int, int, int]]
  total: int
  shared: int


def compaction(layout: BlobLayout, digests: np.ndarray[[int, int], np.dtype[np.uint8]]) -> Compaction:
  """The compaction of `layout`'s blob once its images' content `digests` are known (each image
  ALIGN-aligned, as laid out)."""
  ends = [*layout.offsets[1:], layout.total][: len(layout.offsets)]
  first: dict[bytes, int] = {}
  offsets: list[int] = []
  moves: list[tuple[int, int, int]] = []
  total = 0
  for i, (start, end) in enumerate(zip(layout.offsets, ends, strict=True)):
    kept = first.setdefault(digests[i].tobytes(), i)
    if kept != i:
      offsets.append(offsets[kept])
      continue
    offsets.append(total)
    if total != start:
      moves.append((start, total, end - start))
    total += end - start
  return Compaction(offsets=offsets, moves=moves, total=total, shared=len(offsets) - len(first))


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T], threads: int | None) -> list[R]:
  """`[fn(x) for x in items]` on `worker_count(threads)` threads (in order)."""
  if not items:
    return []
  workers = worker_count(threads)
  if workers == 1 or len(items) == 1:
    return [fn(x) for x in items]
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    return list(pool.map(fn, items))


def _reserve_retained(probed: list[Probed], levels: Sequence[int], memory: ByteBudget) -> None:
  """Charge retained images without starving the largest pending decode; evict only excess.

  Reserving every retained image blindly can deadlock: no worker can get its extra workspace
  while only those workers can release retained pixels. Leave room for one full working set.
  """
  largest = max((decode_bytes(p.info, n) for p, n in zip(probed, levels, strict=True)), default=0)
  available = max(memory.total - largest, 0)
  kept = 0
  for i, p in enumerate(probed):
    if p.base is not None:
      if kept + p.base.nbytes <= available:
        kept += p.base.nbytes
      else:
        probed[i] = dataclasses.replace(p, base=None)
  memory.acquire(kept)


def decode_all(  # noqa: PLR0913  # image inputs, then keyword-only sink and execution limits
  sources: Sequence[ImageSource],
  probed: list[Probed],
  levels: Sequence[int],
  *,
  sink: Sink,
  threads: int | None = None,
  budget: ByteBudget | None = None,
) -> list[str]:
  """Decodes every image not decoded ahead (largest first), adds its mips and hands its levels to
  `sink(i, levels)`; returns each image's error ("": none).

  `sink` runs on the worker thread (it copies the pixels into the blob or writes them to the
  cache file with `os.pwrite`, both of which release the GIL). Each worker holds its image's
  `decode_bytes` in `budget` (default: a new DECODE_BYTES one) while it works. Decoded top
  levels are released from `probed` once handed over.
  """
  order = sorted(range(len(sources)), key=lambda i: -probed[i].info.width * probed[i].info.height)
  memory = ByteBudget(DECODE_BYTES) if budget is None else budget
  _reserve_retained(probed, levels, memory)

  def work(i: int) -> str:
    """Decodes image `i` into the sink; its error message ("": none)."""
    p = probed[i]
    with memory.hold(decode_bytes(p.info, levels[i])):
      # The full working-set reservation now covers this image's retained base too.
      probed[i] = dataclasses.replace(p, base=None)
      if p.base is not None:
        memory.release(p.base.nbytes)
      base, error = (p.base, p.error) if p.base is not None else decode_base(sources[i], p.info)
      probed[i] = Probed(info=p.info, error=error)
      sink(i, mip_chain(base, levels[i], transfer=sources[i].transfer))
      del base, p  # release image references before returning their workspace reservation
    return error

  try:
    errors = parallel_map(work, order, threads)
  finally:
    # Sequential failure can leave later bases unprocessed. Return their reservations too.
    for i, p in enumerate(probed):
      if p.base is not None:
        probed[i] = dataclasses.replace(p, base=None)
        memory.release(p.base.nbytes)
  result = [""] * len(sources)
  for i, error in zip(order, errors, strict=True):
    result[i] = error
  return result
