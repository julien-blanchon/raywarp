"""The static GASes' disk cache (`gas.build_static`): a warm start relocates the cached bytes and
renders the image of a fresh build; another mesh misses."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
from raywarp.core.trace.optix import api, gas
from raywarp.core.trace.optix.context import device_context

pytestmark = pytest.mark.skipif(
  not wp.is_cuda_available() or api.unavailable_reason() is not None, reason="needs CUDA and the driver's OptiX"
)


def _scene(lift: float) -> rw.Scene:
  """A lit floor and a mesh pyramid (its apex at `lift`)."""
  b = rw.SceneBuilder()
  pts = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, lift]], np.float32)
  tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]], np.int32)
  mat = b.add_material(rw.Material(base_color=(0.7, 0.6, 0.5), specular_roughness=0.6))
  b.add_instance("plane", material=mat, size=(3.0, 3.0, 0.0))
  b.add_instance("mesh", mesh=b.add_mesh(pts, tris), material=mat, xform=np.c_[np.eye(3), [0.0, 0.0, 0.01]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def _gases(scene: rw.GpuScene) -> list[gas.Accel]:
  """The scene's static mesh GASes (`gas.mesh_gases`, every mesh)."""
  ctx = device_context(scene.device)
  return gas.mesh_gases(ctx, scene.meshes, None, used=None, has_mesh=True, has_box=False, device=scene.device).gases


def _rgb(scene: rw.GpuScene) -> np.ndarray:
  """The scene's image (OptiX traversal)."""
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=50.0)], resolution=(64, 64))
  r.camera_pose.assign(np.array([[[0.0, -2.5, 1.2, 0.2, 0.0, 0.0, 0.98]]], np.float32))
  r.render()
  wp.synchronize()
  return required(r.outputs.rgb).numpy().copy()


def test_relocated_gases_render_like_built_ones(tmp_path, monkeypatch):
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  scene = rw.upload(_scene(0.8))
  built = _gases(scene)
  assert all(not g.keep for g in built)  # a miss: built, then stored
  assert list((tmp_path / "bakes" / "gas").glob("*.npz"))
  restored = _gases(scene)
  assert all(g.keep for g in restored)  # a hit: views of the relocated blob
  assert [g.output_bytes for g in restored] == [g.output_bytes for g in built]
  first = _rgb(scene)  # restored again inside the Renderer
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")
  np.testing.assert_array_equal(_rgb(rw.upload(_scene(0.8))), first)  # caching off: built


def test_another_mesh_misses(tmp_path, monkeypatch):
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  _gases(rw.upload(_scene(0.8)))
  assert all(not g.keep for g in _gases(rw.upload(_scene(0.9))))
  assert len(list((tmp_path / "bakes" / "gas").glob("*.npz"))) == 2
