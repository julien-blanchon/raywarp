"""A textured glTF quad (tooling/test_gltf_to_*.py, adapters/usd/test_rw_normal_maps.py)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _png(rgb: tuple[int, int, int]) -> bytes:
  """A 2 x 2 PNG of one colour."""
  buf = io.BytesIO()
  Image.fromarray(np.full((2, 2, 3), rgb, dtype=np.uint8)).save(buf, format="PNG")
  return buf.getvalue()


def quad_glb(
  path: Path,
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
  texture_transform: dict | None = None,
  material_extensions: dict | None = None,
):
  """A unit quad in glTF's XY plane, textured with base color, normal, MR and occlusion maps, as
  a .glb at `path` (needs pygltflib: the gltf extra)."""
  import pygltflib

  pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
  normal = np.tile(np.array([0, 0, 1], dtype=np.float32), (4, 1))
  uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
  idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
  images = [_png((200, 10, 10)), _png((128, 128, 255)), _png((0, 100, 200)), _png((50, 50, 50))]

  blob = b""
  views = []
  for data in [pos.tobytes(), normal.tobytes(), uv.tobytes(), idx.tobytes(), *images]:
    views.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
    blob += data + b"\0" * (-len(data) % 4)

  def acc(view, ctype, count, kind, **kw):
    return pygltflib.Accessor(bufferView=view, componentType=ctype, count=count, type=kind, **kw)

  doc = pygltflib.GLTF2(
    scene=0,
    scenes=[pygltflib.Scene(nodes=[0])],
    nodes=[pygltflib.Node(mesh=0, translation=[0.0, 2.0, 0.0], scale=list(scale))],
    meshes=[
      pygltflib.Mesh(
        primitives=[
          pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2), indices=3, material=0)
        ]
      )
    ],
    accessors=[
      acc(0, 5126, 4, "VEC3", min=[0, 0, 0], max=[1, 1, 0]),
      acc(1, 5126, 4, "VEC3"),
      acc(2, 5126, 4, "VEC2"),
      acc(3, 5123, 6, "SCALAR"),
    ],
    materials=[
      pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
          baseColorFactor=[1.0, 1.0, 1.0, 1.0],
          metallicFactor=0.25,
          roughnessFactor=0.75,
          baseColorTexture=pygltflib.TextureInfo(
            index=0, extensions={"KHR_texture_transform": texture_transform} if texture_transform else {}
          ),
          metallicRoughnessTexture=pygltflib.TextureInfo(index=2),
        ),
        normalTexture=pygltflib.NormalMaterialTexture(index=1),
        occlusionTexture=pygltflib.OcclusionTextureInfo(index=3),
        extensions=material_extensions or {},
      )
    ],
    textures=[pygltflib.Texture(source=i) for i in range(4)],
    images=[pygltflib.Image(bufferView=4 + i, mimeType="image/png") for i in range(4)],
    bufferViews=views,
    buffers=[pygltflib.Buffer(byteLength=len(blob))],
  )
  doc.set_binary_blob(blob)
  doc.save_binary(str(path))
