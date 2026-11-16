"""A glTF 2.0 document (.gltf / .glb) and its binary data: accessors as arrays, images, node
transforms, and the files it references (`Gltf`)."""

from __future__ import annotations

import io
import urllib.parse
from pathlib import Path
from typing import Final, cast, final

import numpy as np
import pygltflib
from PIL import Image

from raywarp.scene.model.types import itemsize
from raywarp.scene.model.xform import quat_to_matrix

type Matrix4 = np.ndarray[[4, 4], np.dtype[np.float64]]
type Accessor = np.ndarray[tuple[int, ...], np.dtype[np.number]]

# glTF accessor component types and element sizes (glTF 2.0 §3.6.2).
_COMPONENT_DTYPE: Final = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
# The integer component types' largest value (`normalized` accessors; glTF 2.0 3.6.2.3).
_COMPONENT_MAX: Final = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0, 5125: 4294967295.0}
_TYPE_SIZE: Final = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# Primitive mode TRIANGLES (glTF 2.0 §3.7.2.1): the only one rendered.
TRIANGLES: Final = 4


@final
class Gltf:
  """A loaded glTF document and its buffers.

  Attributes:
    path: the .gltf / .glb file.
    doc: the pygltflib document.
  """

  def __init__(self, path: Path) -> None:
    """Loads the document (buffers are read on first use).

    Raises:
      ValueError: for a binary glTF without a JSON chunk.
    """
    self.path = path
    doc = pygltflib.GLTF2().load(str(path))
    if doc is None:
      raise ValueError(f"{path}: no glTF document found")
    self.doc = doc
    self._buffers: dict[int, bytes] = {}

  def uri_path(self, uri: str | None) -> Path | None:
    """The file an external (not data:) URI names, relative to the document; None otherwise."""
    if not uri or uri.startswith("data:"):
      return None
    return self.path.parent / urllib.parse.unquote(uri)

  def files(self) -> list[Path]:
    """The document and every external file it references (buffers, images), existing or not."""
    uris = [b.uri for b in self.doc.buffers] + [i.uri for i in self.doc.images]
    return [self.path, *(p for p in (self.uri_path(u) for u in uris) if p is not None)]

  def _buffer(self, index: int) -> bytes:
    """Buffer `index`'s bytes (GLB chunk, data URI or file)."""
    if index not in self._buffers:
      buf = self.doc.buffers[index]
      external = self.uri_path(buf.uri)
      if buf.uri is None:
        blob = self.doc.binary_blob()
        if blob is None:
          raise ValueError(f"{self.path}: buffer {index} has no URI and the file no binary chunk")
        self._buffers[index] = blob
      elif external is None:
        self._buffers[index] = self.doc.decode_data_uri(buf.uri)
      else:
        self._buffers[index] = external.read_bytes()
    return self._buffers[index]

  def view_bytes(self, view_index: int) -> bytes:
    """The bytes of buffer view `view_index`."""
    view = self.doc.bufferViews[view_index]
    data = self._buffer(view.buffer)
    return data[view.byteOffset or 0 : (view.byteOffset or 0) + view.byteLength]

  def accessor(self, index: int) -> Accessor:
    """Accessor `index` as (count,) or (count, n) values (normalised integers as floats: unsigned
    in [0, 1], signed in [-1, 1], glTF 2.0 §3.11).

    Raises:
      NotImplementedError: for sparse accessors and accessors without a buffer view.
      ValueError: for an accessor whose elements do not fit in its buffer view.
    """
    acc = self.doc.accessors[index]
    if acc.sparse is not None:
      raise NotImplementedError("sparse accessors are not supported")
    if acc.bufferView is None:
      raise NotImplementedError("accessors without a bufferView (all zeros) are not supported")
    dtype = np.dtype(_COMPONENT_DTYPE[acc.componentType])
    ncomp = _TYPE_SIZE[acc.type]
    view = self.doc.bufferViews[acc.bufferView]
    data = self.view_bytes(acc.bufferView)
    element = itemsize(dtype) * ncomp
    stride, offset = view.byteStride or element, acc.byteOffset or 0
    # `as_strided` does not check bounds: a malformed accessor would read past the view's bytes.
    if acc.count and offset + (acc.count - 1) * stride + element > len(data):
      raise ValueError(f"{self.path}: accessor {index} ({acc.count} x {element} B) overruns its buffer view ({len(data)} B)")
    rows = np.lib.stride_tricks.as_strided(
      np.frombuffer(data, dtype=np.uint8, offset=offset), shape=(acc.count, element), strides=(stride, 1)
    )
    out = np.ascontiguousarray(rows).view(dtype).reshape(acc.count, ncomp)
    if acc.normalized:
      # Signed: max(c / max, -1) (the most negative value is -1 too).
      top = _COMPONENT_MAX.get(acc.componentType)
      if top is None:
        raise ValueError(f"{self.path}: accessor {index} normalizes {dtype} components (integers only)")
      out = np.maximum(out.astype(np.float32) / top, -1.0)
    return cast("Accessor", out if ncomp > 1 else out[:, 0])

  def image_bytes(self, source: int) -> bytes:
    """The encoded bytes of image `source` (buffer view, data URI or file)."""
    image = self.doc.images[source]
    if image.bufferView is not None:
      return self.view_bytes(image.bufferView)
    external = self.uri_path(image.uri)
    return self.doc.decode_data_uri(image.uri) if external is None else external.read_bytes()

  def image(self, texture_index: int | None) -> Image.Image:
    """The decoded image of texture `texture_index`.

    Raises:
      ValueError: for a missing index or a texture without an image source.
    """
    if texture_index is None:
      raise ValueError(f"{self.path}: texture reference without an index")
    source = self.doc.textures[texture_index].source
    if source is None:
      raise ValueError(f"{self.path}: texture {texture_index} without an image source")
    return Image.open(io.BytesIO(self.image_bytes(source)))

  @staticmethod
  def node_matrix(node: pygltflib.Node) -> Matrix4:
    """The node's local (4, 4) column-vector matrix (its matrix, else T R S)."""
    if node.matrix is not None:
      return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # glTF is column-major.
    t = np.eye(4)
    t[:3, 3] = node.translation or [0.0, 0.0, 0.0]
    r = np.eye(4)
    r[:3, :3] = quat_to_matrix(node.rotation or (0.0, 0.0, 0.0, 1.0))  # glTF's (x, y, z, w)
    s = np.diag(np.array([*(node.scale or [1.0, 1.0, 1.0]), 1.0]))
    return t @ r @ s
