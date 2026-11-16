"""The images of a `SceneBuilder`: registered once per source, read ahead, decoded at build.

Images are deduplicated by source, size limit and 8-bit transfer (files by absolute path, arrays
by identity, derived images by their key). Registration starts the read ahead (`blob.Ahead`);
`ImagePool.decode` lays out the texel blob and fills it through a `blob.Sink`.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Hashable
from typing import final

import numpy as np

from raywarp.scene.images import blob, files
from raywarp.scene.images import decode as imaging
from raywarp.scene.images.mips import Transfer
from raywarp.scene.model import wpbr
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import Images
from raywarp.scene.model.types import NO_IMAGE, Array, ImageId

# Image dedup key: (path, max size, transfer) for files, ("array", id, max size, transfer),
# ("derived", key).
type ImageKey = tuple[str, int | None, Transfer] | tuple[str, int, int | None, Transfer] | tuple[str, tuple[Hashable, ...]]


@final
class ImagePool:
  """The images registered so far (image id = index), read ahead on a thread pool.

  Attributes:
    report: where missing / unsupported files and decode failures are reported.
    dependencies: the image files (missing ones included), shared with the builder's.
  """

  def __init__(self, report: LoadReport, dependencies: set[str]) -> None:
    """An empty pool reporting into `report`, adding the files it reads to `dependencies`."""
    self.report = report
    self.dependencies = dependencies
    self.sources: list[imaging.ImageSource] = []
    self._keys: dict[ImageKey, ImageId] = {}
    self._ahead = blob.Ahead()
    # A builder dropped before its build (an adapter's error) stops the reads ahead.
    weakref.finalize(self, self._ahead.abort)

  def __len__(self) -> int:
    """Number of images."""
    return len(self.sources)

  def _register(self, key: ImageKey, source: imaging.ImageSource) -> ImageId:
    """Id of the image of `key`, registering `source` (and starting its read) when new."""
    if key not in self._keys:
      self._keys[key] = ImageId(len(self.sources))
      self.sources.append(source)
      self._ahead.submit(source)
    return self._keys[key]

  def _usable(self, path: str) -> bool:
    """Whether image file `path` exists and has a supported format (else reported); a dependency."""
    self.dependencies.add(files.container(path))
    if not files.is_file(path):
      self.report.drop("texture.missing_file", path)
      return False
    if not imaging.supported(path):
      self.report.drop(f"texture.unsupported_format{files.suffix(path)}", path)
      return False
    return True

  def add(self, source: str | os.PathLike[str] | Array, *, max_size: int | None, transfer: Transfer) -> ImageId:
    """`SceneBuilder.add_image`."""
    imaging.check_max_size(max_size, "add_image")
    if isinstance(source, np.ndarray):
      imaging.check_array(source, "add_image")
      array = imaging.ImageSource(array=source, max_size=max_size, transfer=transfer)
      return self._register(("array", id(source), max_size, transfer), array)
    path = files.absolute(os.fspath(source))
    key = (path, max_size, transfer)
    if key in self._keys:
      return self._keys[key]
    if not self._usable(path):
      return NO_IMAGE
    return self._register(key, imaging.ImageSource(path=path, max_size=max_size, transfer=transfer))

  def add_derived(self, derived: imaging.Derived) -> ImageId:
    """`SceneBuilder.add_derived_image`.

    Raises:
      ValueError: for a source that is not an image file (an array or a derived image).
    """
    if any(not src.path for src in derived.sources):
      raise ValueError(f"add_derived_image {derived.key!r}: every source must be an image file (ImageSource.path)")
    key = ("derived", derived.key)
    if key in self._keys:
      return self._keys[key]
    if not all(self._usable(src.path) for src in derived.sources):
      return NO_IMAGE
    self.report.count("images.derived")
    return self._register(
      key, imaging.ImageSource(transfer="srgb" if derived.encoding == "srgb" else "linear", derived=derived)
    )

  def read_states(self) -> dict[str, wpbr.Dependency]:
    """`SceneBuilder.read_states`."""
    return {p.state.path: p.state for p in self._ahead.results() if p.state is not None}

  def layout(self, *, mips: bool) -> tuple[blob.BlobLayout, Images, list[blob.Probed]]:
    """(the blob layout, the `Images` table (its blob still empty), the probed images), once the
    reads ahead are done; counts and downscales reported."""
    try:
      with self.report.timer("build.image_ahead"):
        probed = self._ahead.results()
    except BaseException:
      self._ahead.abort()
      raise
    self._ahead.close()
    infos = [p.info for p in probed]
    layout = blob.BlobLayout.of(infos, mips=mips)
    names = [source.name for source in self.sources]
    self.report.count("images", len(infos))
    self.report.count("texels", sum(i.width * i.height for i in infos))
    for info, name in zip(infos, names, strict=True):
      if info.factor > 1:
        self.report.approx("texture.downscaled", name)
    return layout, layout.table(infos, names), probed

  def decode(self, layout: blob.BlobLayout, probed: list[blob.Probed], sink: blob.Sink, *, threads: int | None) -> None:
    """Decodes every image into `sink` (`blob.decode_all`), reporting the failures."""
    with self.report.timer("build.image_decode"):
      errors = blob.decode_all(self.sources, probed, layout.levels, sink=sink, threads=threads)
    for error, source in zip(errors, self.sources, strict=True):
      if error:
        self.report.drop(imaging.report_key(error), f"{source.name}: {error}")
