"""Prepared geometry must change actual first hits on both GPU traversal paths."""

from typing import Literal

import numpy as np
import pytest
import warp as wp

import raywarp as rw


@pytest.mark.skipif(not wp.is_cuda_available(), reason="GPU traversal parity")
@pytest.mark.parametrize("traversal", ["software", "optix"])
def test_prepared_displacement_changes_depth(traversal: Literal["software", "optix"]) -> None:
  """A 20 cm displacement must move the planar first hit from 1 m to 80 cm."""
  builder = rw.SceneBuilder()
  mesh = builder.add_mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]])
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=0.2)))
  builder.add_instance("mesh", mesh=mesh, material=material)
  gpu = rw.upload(builder.build(), "cuda:0")
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth", "seg")).replace(**{"performance.traversal": traversal})
  renderer = rw.Renderer(gpu, nworld=2, cameras=[rw.CameraSpec(fovy=45)], resolution=(64, 64), settings=settings)
  try:
    renderer.camera_pose.assign(np.tile(np.asarray([0, 0, 1, 0, 0, 0, 1], np.float32), (2, 1, 1)))
    renderer.render()
    depth = renderer.outputs.depth
    assert depth is not None
    np.testing.assert_allclose(depth.numpy(), 0.8, rtol=0, atol=2e-6)
    assert renderer.features.traversal == traversal
  finally:
    renderer.close()
