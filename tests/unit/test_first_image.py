"""The first-image check (`raywarp.core.outputs.first_image`): blown-out and black first images
warn once per camera, in every color layout; `Renderer.render` runs it once, never per frame."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import warp as wp
from conftest import MakeScene

from raywarp.core.outputs import first_image
from raywarp.core.outputs.render_outputs import RenderOutputs

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom type="box" size="0.3 0.3 0.3" pos="0 0 0.3" rgba="1 0 0 1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


def _vec4(images: np.ndarray) -> wp.array:
  """(nworld, ncam, H, W, 4) uint8 -> a vec4ub array on the CPU."""
  return wp.array(images, dtype=wp.vec4ub, device="cpu")


def test_verdict_flags_clipped_and_black_images_only() -> None:
  grey = np.full((2, 4, 4, 3), 0.5, np.float32)
  assert first_image.verdict(grey) is None
  assert "black" in str(first_image.verdict(np.zeros_like(grey)))
  white = grey.copy()
  white[:, :2] = 1.0  # half the pixels
  assert "50%" in str(first_image.verdict(white))
  white[:, :2] = 0.5
  white[:, 0, 0] = 1.0  # 1 pixel of 16: under the share
  assert first_image.verdict(white) is None


def test_camera_pixels_reads_every_layout_per_camera() -> None:
  """vec4ub (nworld, ncam, H, W) and planar (nworld * ncam, 3, H, W) uint8 / float16: per camera,
  (worlds, H, W, 3) in [0, 1], at most MAX_WORLDS worlds."""
  images = np.zeros((20, 2, 3, 3, 4), np.uint8)
  images[:, 1, ..., :3] = 255
  images[..., 3] = 255  # alpha: ignored
  cams = first_image.camera_pixels(_vec4(images), 20)
  assert len(cams) == 2 and cams[0].shape == (first_image.MAX_WORLDS, 3, 3, 3)
  assert cams[0].max() == 0.0 and cams[1].min() == 1.0
  planar = np.zeros((4 * 2, 3, 3, 3), np.float16)
  planar[1::2] = 1.0  # camera 1 of every world (world-major rows)
  cams = first_image.camera_pixels(wp.array(planar, dtype=wp.float16, device="cpu"), 4)
  assert [c.max() for c in cams] == [0.0, 1.0] and cams[1].shape == (4, 3, 3, 3)
  cams = first_image.camera_pixels(wp.array((planar * 255).astype(np.uint8), dtype=wp.uint8, device="cpu"), 4)
  assert [c.max() for c in cams] == [0.0, 1.0]


def test_check_warns_per_flagged_camera() -> None:
  images = np.full((2, 3, 4, 4, 4), 128, np.uint8)
  images[:, 0] = 0  # camera 0 black
  images[:, 2, ..., :3] = 255  # camera 2 white
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first_image.check(RenderOutputs(rgb=_vec4(images)), 2)
  text = [str(w.message) for w in caught]
  assert len(text) == 2 and "camera 0" in text[0] and "black" in text[0] and "camera 2" in text[1] and "100%" in text[1]


def test_renderer_checks_its_first_frame_only(scene: MakeScene, monkeypatch: pytest.MonkeyPatch) -> None:
  """One check, on the first render; the frames after it never read back."""
  calls: list[int] = []
  monkeypatch.setattr(first_image, "check", lambda outputs, nworld: calls.append(nworld))
  s = scene(XML, cam_res=(8, 8), nworld=2)
  s.render().render().render()
  assert calls == [2]
