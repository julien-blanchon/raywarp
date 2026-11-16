"""tools/video: the RGBA -> NV12 (BT.709 limited range) kernel against the spec, tiling and odd
sizes, and an end-to-end recording per back end decoded back (x264 on the CPU device when ffmpeg
is installed; NVENC only on a CUDA device that has one)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import warp as wp

from raywarp.video import Tiling, recorder
from raywarp.video.colour import KB, KR, convert
from raywarp.video.encoders import nvenc_status

# ITU-R BT.709 8-bit limited-range Y'CbCr of the primaries, white and black.
SPEC = {
  (255, 255, 255): (235, 128, 128),
  (0, 0, 0): (16, 128, 128),
  (255, 0, 0): (63, 102, 240),
  (0, 255, 0): (173, 42, 26),
  (0, 0, 255): (32, 240, 118),
}


def nv12_of(images: np.ndarray, tiling: Tiling) -> np.ndarray:
  """(frames, 3 H / 2, W) uint8: the kernel's output for (n, h, w, 4) uint8 images, on the CPU."""
  h, w = tiling.frame_size
  out = wp.zeros((tiling.frames, h * 3 // 2, w), dtype=wp.uint8, device="cpu")
  convert(wp.array(images, dtype=wp.vec4ub, device="cpu"), tiling, out)
  return out.numpy()


def reference(rgb: np.ndarray) -> np.ndarray:
  """(h, w, 3) R'G'B' (even h, w) -> NV12 by the BT.709 formulas in float64 (2x2 chroma mean)."""
  c = rgb.astype(np.float64) / 255
  y = KR * c[..., 0] + (1 - KR - KB) * c[..., 1] + KB * c[..., 2]
  h, w = y.shape
  m = c.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
  my = KR * m[..., 0] + (1 - KR - KB) * m[..., 1] + KB * m[..., 2]
  cb, cr = (m[..., 2] - my) / (2 * (1 - KB)), (m[..., 0] - my) / (2 * (1 - KR))
  uv = np.stack([128 + 224 * cb, 128 + 224 * cr], axis=-1).reshape(h // 2, w)
  return np.round(np.concatenate([16 + 219 * y, uv])).astype(np.uint8)


def test_spec_colours() -> None:
  colours = list(SPEC)
  images = np.zeros((len(colours), 2, 2, 4), np.uint8)
  images[..., 3] = 255
  for i, c in enumerate(colours):
    images[i, ..., :3] = c
  nv12 = nv12_of(images, Tiling(len(colours), 2, 2))
  for i, c in enumerate(colours):
    assert (int(nv12[i, 0, 0]), int(nv12[i, 2, 0]), int(nv12[i, 2, 1])) == SPEC[c], c
    assert (nv12[i, :2] == nv12[i, 0, 0]).all()


def test_matches_formulas() -> None:
  rng = np.random.default_rng(3)
  images = rng.integers(0, 256, (3, 6, 10, 4), dtype=np.uint8)
  nv12 = nv12_of(images, Tiling(3, 6, 10))
  for i in range(3):
    diff = np.abs(nv12[i].astype(int) - reference(images[i, ..., :3]))
    assert diff.max() <= 1  # float32 vs float64 at a rounding boundary


def test_mosaic_and_padding() -> None:
  rng = np.random.default_rng(4)
  images = rng.integers(0, 256, (3, 3, 5, 4), dtype=np.uint8)  # odd sizes, 3 tiles of a 2 x 2 grid
  tiling = Tiling.mosaic(3, 3, 5)
  assert (tiling.rows, tiling.cols, tiling.frames, tiling.frame_size) == (2, 2, 1, (6, 10))
  nv12 = nv12_of(images, tiling)[0]
  y = nv12[:6]
  # Tiles 0-2 hold the images' own luma (each 3 x 5 tile placed at its grid cell); tile 3 is black.
  for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0)]):
    expect = reference(np.pad(images[i, ..., :3], ((0, 1), (0, 1), (0, 0)), mode="edge"))[:3, :5]
    got = y[3 * r : 3 * r + 3, 5 * c : 5 * c + 5]
    assert np.abs(got.astype(int) - expect).max() <= 1
  assert (y[3:6, 5:10] == 16).all() and (nv12[8, 6:10] == 128).all()  # the chroma blocks inside tile 3
  # The padding row / column repeat the last image row / column.
  one = nv12_of(images[:1], Tiling(1, 3, 5))[0]
  assert one.shape == (6, 6)
  assert (one[3, :5] == one[2, :5]).all() and (one[:3, 5] == one[:3, 4]).all()


def frames(n: int, worlds: int, h: int, w: int) -> np.ndarray:
  """(n, worlds, 1, h, w, 4) smooth moving gradients (lossy-codec friendly), a tint per world."""
  t = np.arange(n)[:, None, None, None, None]
  wid = np.arange(worlds)[None, :, None, None, None]
  y, x = np.arange(h)[:, None], np.arange(w)
  out = np.full((n, worlds, 1, h, w, 4), 255, np.uint8)
  out[..., 0] = np.clip(2 * x + 3 * t, 0, 255)
  out[..., 1] = np.clip(3 * y + 40 * wid, 0, 255)
  out[..., 2] = 120 + 50 * wid
  return out


def probe(path: Path) -> dict[str, str]:
  cmd = [
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=color_range,color_space,color_primaries,color_transfer,nb_frames",
    "-of",
    "json",
  ]
  return json.loads(subprocess.run([*cmd, str(path)], capture_output=True, check=True, text=True).stdout)["streams"][0]


def decoded(path: Path, h: int, w: int) -> np.ndarray:
  """Every frame of `path`, (n, h, w, 3) RGB, as ffmpeg decodes it (exact rounding: the default
  scaler truncates, about -1 per channel)."""
  cmd = [
    "ffmpeg",
    "-nostdin",
    "-loglevel",
    "error",
    "-i",
    str(path),
    "-sws_flags",
    "accurate_rnd+full_chroma_int",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "rgb24",
    "-",
  ]
  raw = subprocess.run(cmd, capture_output=True, check=True).stdout
  return np.frombuffer(raw, np.uint8).reshape(-1, h + (h & 1), w + (w & 1), 3)[:, :h, :w]


def roundtrip(tmp_path: Path, device: str, backend: Literal["nvenc", "x264"], h: int, w: int) -> None:
  n, worlds = 8, 2
  src = frames(n, worlds, h, w)
  with recorder(
    tmp_path / "w{world}.mp4", (worlds, 1, h, w), fps=10, backend=backend, device=device, bits_per_pixel=2.0
  ) as rec:
    for i in range(n):
      rec.write(wp.array(src[i], dtype=wp.vec4ub, device=device))
  assert rec.stats.backend == backend and rec.stats.frames == n
  for k in range(worlds):
    tags = probe(tmp_path / f"w{k}.mp4")
    assert (tags["color_range"], tags["color_space"], tags["color_primaries"], tags["color_transfer"]) == (
      "tv",
      "bt709",
      "bt709",
      "bt709",
    )
    err = decoded(tmp_path / f"w{k}.mp4", h, w).astype(float) - src[:, k, 0, ..., :3]
    assert len(err) == n
    assert np.abs(err).mean() < 1.5, np.abs(err).mean()
    assert np.abs(err.reshape(-1, 3).mean(0)).max() < 0.75  # a range or matrix error biases by several codes


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_x264_roundtrip(tmp_path: Path) -> None:
  roundtrip(tmp_path, "cpu", "x264", 33, 160)  # odd height: padded


def _nvenc() -> str | None:
  if not wp.is_cuda_available():
    return "no CUDA device"
  return nvenc_status(wp.get_device("cuda:0"), "h264")


@pytest.mark.skipif(shutil.which("ffmpeg") is None or _nvenc() is not None, reason="needs NVENC (PyNvVideoCodec) and ffmpeg")
def test_nvenc_roundtrip(tmp_path: Path) -> None:
  roundtrip(tmp_path, "cuda:0", "nvenc", 129, 256)  # odd height; above NVENC's minimum frame size
