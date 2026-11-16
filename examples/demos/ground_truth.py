"""Ground truth beside the image: rgb, depth, instance segmentation, camera-frame normals,
normalized object coordinates (NOCS) and screen-space motion, all from one render. A 3 x 2 grid of
world 0; the camera swings so the motion output has something to show.

  python examples/demos/ground_truth.py --size 1280x720 --out ground_truth.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp

import raywarp as rw
from raywarp.settings import NormalOutput

OUTPUTS = rw.Outputs.of("rgb", "depth", "seg", "motion", normal=NormalOutput(frame="shading"))


def flow_colors(motion: np.ndarray, scale: float) -> _common.Image:
  """(H, W, 2) pixel motion as colour: hue the direction, brightness the length."""
  angle = np.arctan2(motion[..., 1], motion[..., 0])
  length = np.clip(np.linalg.norm(motion, axis=-1) / scale, 0.0, 1.0)
  rgb = 0.5 + 0.5 * np.stack([np.cos(angle), np.cos(angle - 2.094), np.cos(angle + 2.094)], axis=-1)
  return (rgb * length[..., None] * 255).astype(np.uint8)


def panels(r: rw.Renderer, stage: _common.Stage) -> list[_common.Image]:
  """World 0's six images (host copies: this is for looking, not for training)."""
  out = r.outputs
  assert out.rgb is not None and out.depth is not None and out.seg is not None and out.motion is not None
  h, w = out.rgb.shape[2:]
  depth = out.depth.numpy()[0, 0]
  hit = depth[depth > 0]
  far = float(np.percentile(hit, 98)) if hit.size else 1.0
  normals = r.camera_normals().numpy()[0].reshape(h, w, 3)  # OpenCV camera frame, zero on a miss
  nocs = r.nocs(stage.sim).numpy()[0].reshape(h, w, 3)  # each object's own unit box
  return [
    out.rgb.numpy()[0, 0, ..., :3],
    _common.colorize(np.where(depth > 0, far - depth, np.nan), 0.0, far),  # near: bright; a miss: black
    _common.label_colors(out.seg.numpy()[0, 0]),
    ((normals * [1, -1, -1] * 0.5 + 0.5) * (normals != 0).any(-1, keepdims=True) * 255).astype(np.uint8),
    (np.clip(nocs, 0.0, 1.0) * 255).astype(np.uint8),
    flow_colors(out.motion.numpy()[0, 0], 0.03 * w),
  ]


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "ground_truth.mp4",
  scene: str = "franka",
) -> None:
  """Records the 3 x 2 grid of world 0's outputs."""
  width, height = _common.parse_size(size)
  tile = (width // 3 // 2 * 2 - 4, height // 2 // 2 * 2 - 4)
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    r = _common.renderer(stage, worlds, tile, rw.RenderSettings(outputs=OUTPUTS))
    canvas = _common.Canvas(width, height)
    with _common.record(out, canvas.rgb) as rec:
      for f in range(frames):
        _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 70.0))
        stage.step()
        r.render(stage.sim)
        p = panels(r, stage)
        canvas.show(_common.grid([p[:3], p[3:]]))
        rec.write(canvas.rgb)
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
