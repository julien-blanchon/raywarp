"""One preparation reuses decoded pixels, while material bindings/reports remain independent."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import raywarp as rw
from raywarp.scene.lighting import emissive
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import LinearRGBA, TextureId


def _scene() -> rw.Scene:
  b = rw.SceneBuilder()
  pixels = np.full((16, 16, 4), 192, np.uint8)
  pixels[:, :, 1] = 128
  pixels[:, :, 2] = 64
  pixels[:, :, 3] = 255
  texture = b.add_texture(pixels, colorspace="g22_rec709")
  mesh = b.add_mesh(
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32),
    np.array([[0, 1, 2], [0, 2, 3]], np.int32),
    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
  )
  for channel in (-1, 0, 1):
    material = b.add_material(
      rw.Material(
        emission_luminance=100.0,
        textures=((rw.scene.Slot.EMISSION_COLOR, rw.scene.TextureRef(texture, channel=channel)),),
      )
    )
    b.add_instance("mesh", mesh=mesh, material=material, xform=np.c_[np.eye(3), [2 * channel, 0, 2]])
  return b.build()


def test_complete_emissive_lights_and_reports_equal_without_reduction_repetition(monkeypatch: pytest.MonkeyPatch) -> None:
  # The in-process reuse is under test: the disk cache of fits is off (it would also key the
  # edited images below by their content digests, which the edit leaves stale).
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")
  host = _scene()
  original = emissive._texels
  requests: list[TextureId] = []
  reductions: list[TextureId] = []

  def tracked(scene: rw.Scene, texture: TextureId, cache: dict[TextureId, LinearRGBA]) -> LinearRGBA:
    requests.append(texture)
    if texture not in cache:
      reductions.append(texture)
    return original(scene, texture, cache)

  def uncached(scene: rw.Scene, texture: TextureId, cache: dict[TextureId, LinearRGBA]) -> LinearRGBA:
    """Preserve the original once-per-material computation as a full-table reference."""
    return original(scene, texture, {})

  monkeypatch.setattr(emissive, "TEXEL_LIMIT", 4)
  with monkeypatch.context() as patch:
    patch.setattr(emissive, "_texels", tracked)
    candidate = emissive.add_panel_lights(dataclasses.replace(host, report=LoadReport()))
  with monkeypatch.context() as patch:
    patch.setattr(emissive, "_texels", uncached)
    reference = emissive.add_panel_lights(dataclasses.replace(host, report=LoadReport()))
  assert len(requests) == 3 and len(reductions) == 1
  assert len(candidate.lights) == 3
  for name, values in reference.lights.arrays().items():
    np.testing.assert_array_equal(candidate.lights.arrays()[name], values, err_msg=name)
  assert candidate.report.to_dict() == reference.report.to_dict()
  key = "emissive_light.texture.colorspace:g22_rec709_scene(converted to Rec.709)"
  assert candidate.report.approximated[key] == 3
  # A later preparation must not reuse pixels retained from the first preparation.
  data = host.images.data.copy()
  data[:] = 255
  host = dataclasses.replace(host, images=dataclasses.replace(host.images, data=data))
  with monkeypatch.context() as patch:
    patch.setattr(emissive, "_texels", tracked)
    changed = emissive.add_panel_lights(dataclasses.replace(host, report=LoadReport()))
  assert len(reductions) == 2
  assert not np.array_equal(candidate.lights.color, changed.lights.color)


def test_texture_identity_keeps_different_color_space_interpretations_separate() -> None:
  b = rw.SceneBuilder()
  image = b.add_image(np.full((4, 4, 4), 128, np.uint8))
  linear = b.add_texture(image, colorspace="linear")
  gamma = b.add_texture(image, colorspace="g22_rec709")
  host = b.build()
  cache: dict[TextureId, LinearRGBA] = {}
  a = emissive._texels(host, linear, cache)
  c = emissive._texels(host, gamma, cache)
  assert len(cache) == 2
  assert not np.array_equal(a[..., :3], c[..., :3])
  np.testing.assert_array_equal(a, emissive._texels(host, linear, {}))
  np.testing.assert_array_equal(c, emissive._texels(host, gamma, {}))


def test_prefetch_reduces_only_the_textures_meshes_read(monkeypatch: pytest.MonkeyPatch) -> None:
  """Emissive primitives (a thin box) use the factor alone: their emission texture is not
  prefetched; the meshes' textures are."""
  b = rw.SceneBuilder()
  textures = [b.add_texture(np.full((8, 8, 4), v, np.uint8), colorspace="g22_rec709") for v in (64, 128, 192)]
  materials = [
    b.add_material(rw.Material(emission_luminance=100.0, textures=((rw.scene.Slot.EMISSION_COLOR, rw.scene.TextureRef(t)),)))
    for t in textures
  ]
  mesh = b.add_mesh(
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32),
    np.array([[0, 1, 2], [0, 2, 3]], np.int32),
    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
  )
  b.add_instance("mesh", mesh=mesh, material=materials[0], xform=np.c_[np.eye(3), [0, 0, 2]])
  b.add_instance("mesh", mesh=mesh, material=materials[1], xform=np.c_[np.eye(3), [2, 0, 2]])
  b.add_instance("box", size=(0.5, 0.5, 0.004), material=materials[2], xform=np.c_[np.eye(3), [4, 0, 2]])
  original = emissive._reduced
  reduced: list[TextureId] = []

  def tracked(scene: rw.Scene, texture: TextureId) -> LinearRGBA:
    reduced.append(texture)
    return original(scene, texture)

  monkeypatch.setattr(emissive, "_reduced", tracked)
  emissive.add_panel_lights(b.build())
  assert sorted(reduced) == sorted(textures[:2])
