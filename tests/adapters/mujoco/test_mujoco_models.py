"""Compatibility sweep: every MuJoCo `model/` example under $MJPBR_ASSET_DIR loads and renders.

Each model is compiled by MuJoCo, put on the CPU by MuJoCo Warp, turned into a scene and rendered
once (one world, 24 x 16, a free camera on the model's centre). A model MuJoCo or MuJoCo Warp
cannot load is skipped with the reason (a file included by others, a missing plugin, an
unsupported MuJoCo Warp feature); every model they load must render finite pixels, and see
something. Skipped without the MuJoCo model gallery (`$MJPBR_ASSET_DIR/mujoco/model`).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
import raywarp.integrations.mujoco

# ~4.5 min on 4 CPU workers (one megakernel specialisation per model): deselect with -m 'not slow'.
pytestmark = pytest.mark.slow

MODELS = Path(os.environ.get("MJPBR_ASSET_DIR", Path.home() / ".cache" / "raywarp")) / "mujoco" / "model"
FILES = sorted(MODELS.rglob("*.xml")) if MODELS.is_dir() else []


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
  """(7,) camera pose (position, quaternion x y z w) at `eye` looking at `target`, +z up."""
  front = (target - eye) / np.linalg.norm(target - eye, keepdims=True)
  right = np.cross(front, (0.0, 0.0, 1.0))
  right = right / max(math.sqrt(float(right @ right)), 1e-9)
  rot = np.stack([right, np.cross(right, front), -front], axis=1)
  w = 0.5 * np.sqrt(max(1.0 + np.trace(rot), 1e-12))
  quat = [(rot[2, 1] - rot[1, 2]) / (4 * w), (rot[0, 2] - rot[2, 0]) / (4 * w), (rot[1, 0] - rot[0, 1]) / (4 * w), w]
  return np.array([*eye, *quat], np.float32)


def _load(path: Path) -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of a model file, or skip with MuJoCo's / MuJoCo Warp's reason."""
  try:
    mjm = mujoco.MjModel.from_xml_path(str(path))
  except Exception as e:  # noqa: BLE001 - MuJoCo's own load error: not a renderer issue
    pytest.skip(f"MuJoCo: {str(e).splitlines()[0][:120]}")
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  try:
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
  except Exception as e:  # noqa: BLE001 - MuJoCo Warp's: not a renderer issue
    pytest.skip(f"MuJoCo Warp: {type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}")
  return mjm, m, d


@pytest.mark.skipif(not FILES, reason=f"no MuJoCo model gallery under {MODELS}")
@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(MODELS)) for p in FILES])
def test_model_renders(path: Path):
  """The example loads into a scene and renders finite pixels showing something."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _load(path)
    scene = rw.integrations.mujoco.scene(mjm, m, adapter=rw.integrations.mujoco.Options(geom_groups=(0, 1, 2, 3, 4, 5)))
    settings = rw.RenderSettings().replace(**{"outputs": rw.Outputs.of("rgb", "depth"), "performance.traversal": "software"})
    r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=45.0)], resolution=(24, 16), settings=settings)
    center, extent = np.asarray(mjm.stat.center, np.float64), max(float(mjm.stat.extent), 1e-3)
    eye = center + extent * np.array([1.2, -1.6, 0.9])
    r.camera_pose.assign(_look_at(eye, center).reshape(1, 1, 7))
    r.render(d)
    rgb = required(r.outputs.rgb).numpy()[0, 0, ..., :3]
    depth = required(r.outputs.depth).numpy()[0, 0]
  assert np.all(np.isfinite(depth[depth > 0])), "non-finite depth"
  assert np.any(depth > 0.0), "nothing hit"
  assert rgb.max() > 0, "black image"
