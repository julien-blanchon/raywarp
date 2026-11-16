"""Pixel-center observations, independent random keys and graph-owned device seeds."""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, required
from test_api import XML

import raywarp as rw
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.trace.rays.seeds import pixel_seed

DEVICE: Final = "cuda:0" if wp.is_cuda_available() else "cpu"


@wp.kernel
def _keys(
  # In:
  coordinates: wp.array2d[wp.int32],
  # Out:
  output: wp.array1d[wp.int32],
) -> None:
  """Evaluate selected random coordinates without allocating their implied large images."""
  i = wp.tid()
  output[i] = pixel_seed(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2])


@wp.kernel
def _advance_seed(
  # In / Out:
  seed: wp.array1d[wp.int32],
) -> None:
  """Stand in for device-side simulation/policy work that selects the render seed."""
  seed[0] = seed[0] + 1


def test_random_coordinates_do_not_slide_or_wrap_world_indices() -> None:
  """Adjacent seeds do not translate pixels; large world coordinates are mixed independently."""
  coords = np.array(
    [(seed, world, pixel) for seed in (0, 1, 2147483647, -2147483648) for world in (0, 1024, 2048) for pixel in (0, 1, 17)],
    dtype=np.int32,
  )
  source = wp.array(coords, dtype=wp.int32, device=DEVICE)
  out = wp.empty(len(coords), dtype=wp.int32, device=DEVICE)
  wp.launch(_keys, dim=len(coords), inputs=[source], outputs=[out], device=DEVICE)
  keys = out.numpy()
  assert len(np.unique(keys)) == len(coords)
  # Old seed + world * nray + pixel aliases (seed 0, pixel 1) with (seed 1, pixel 0),
  # and worlds 0/2048 with nray=2**21. No multi-gigapixel allocation is needed to catch either.
  assert keys[1] != keys[9]
  assert keys[0] != keys[6]


@pytest.mark.parametrize("traversal", ["software", "optix"])
@pytest.mark.parametrize("spp,motion", [(2, False), (3, False), (2, True)])
def test_observations_stay_at_pixel_centers(
  scene: MakeScene, traversal: Literal["software", "optix"], spp: int, motion: bool
) -> None:
  """Depth, labels, material values and guides stay aligned when color samples move (the plain
  and the motion body)."""
  if traversal == "optix" and not device_caps(DEVICE).optix.available:
    pytest.skip("OptiX unavailable")
  if traversal == "optix" and spp > 1 and motion:
    pytest.skip("OptiX supersampling with motion is explicitly unsupported")
  outputs = rw.Outputs.of("rgb", "depth", "seg", "position", "albedo", "normal", "roughness")
  with wp.ScopedDevice(DEVICE):
    base = scene(
      XML,
      cameras=["front"],
      cam_res=(12, 10),
      outputs=outputs,
      aa=rw.settings.AntiAliasing(spp=1),
      performance=rw.settings.Performance(traversal=traversal),
    ).render()
    sampled = scene(
      XML,
      cameras=["front"],
      cam_res=(12, 10),
      outputs=rw.Outputs.of("motion", *outputs.kinds) if motion else outputs,
      aa=rw.settings.AntiAliasing(spp=spp),
      performance=rw.settings.Performance(traversal=traversal),
    )
    for _ in range(2):
      sampled.render()
      np.testing.assert_array_equal(required(sampled.out.seg).numpy(), required(base.out.seg).numpy())
      for name in ("depth", "position", "albedo", "normal", "roughness"):
        np.testing.assert_allclose(
          sampled.out.arrays()[name].numpy(),
          base.out.arrays()[name].numpy(),
          rtol=2e-5,
          atol=2e-6,
          err_msg=name,
        )


def test_device_seed_matches_integer_and_checks_shape(scene: MakeScene) -> None:
  """A valid scalar has integer-seed behavior; malformed buffers fail before kernel access."""
  with wp.ScopedDevice(DEVICE):
    s = scene(XML, cameras=["front"], cam_res=(12, 10), aa=rw.settings.AntiAliasing(), outputs=rw.Outputs.of("hdr"))
    s.r.render(s.d, seed=17)
    expected = required(s.out.hdr).numpy().copy()
    seed = wp.array([17], dtype=wp.int32, device=DEVICE)
    s.r.render(s.d, seed=seed)
    np.testing.assert_array_equal(required(s.out.hdr).numpy(), expected)
    for bad in (wp.zeros(2, dtype=wp.int32, device=DEVICE), wp.zeros(1, dtype=wp.float32, device=DEVICE)):
      with pytest.raises(ValueError, match="seed must be an int32 array"):
        s.r.render(s.d, seed=bad)  # pyrefly: ignore[bad-argument-type]  # intentionally invalid seed buffers


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("traversal", ["software", "optix"])
def test_captured_device_seed_matches_eager_and_integer_seed_is_owned(
  scene: MakeScene, traversal: Literal["software", "optix"]
) -> None:
  """Device writes change captured sampling; integer capture owns its scalar independently."""
  if traversal == "optix" and not device_caps(DEVICE).optix.available:
    pytest.skip("OptiX unavailable")
  with wp.ScopedDevice(DEVICE):
    s = scene(
      XML,
      cameras=["front"],
      cam_res=(32, 24),
      aa=rw.settings.AntiAliasing(),
      shadows=rw.settings.Shadows(samples=4),
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal=traversal),
    )
    seed = wp.zeros(1, dtype=wp.int32, device=DEVICE)

    def advance_seed() -> None:
      wp.launch(_advance_seed, dim=1, inputs=[seed], device=DEVICE)

    graph = s.r.capture(advance_seed, sim=s.d, seed=seed)
    rendered = []
    for value in (20, 21):
      seed.fill_(value - 1)
      graph.launch(s.d)
      actual = required(s.out.hdr).numpy().copy()
      s.r.render(s.d, seed=value)
      np.testing.assert_array_equal(actual, required(s.out.hdr).numpy())
      rendered.append(actual)
    assert not np.array_equal(rendered[0], rendered[1]), "the stochastic scene must actually respond to the seed"
    fixed = s.r.capture(sim=s.d, seed=3)
    s.r.render(s.d, seed=3)
    expected = required(s.out.hdr).numpy().copy()
    s.r.render(s.d, seed=77)
    fixed.launch(s.d)
    np.testing.assert_array_equal(required(s.out.hdr).numpy(), expected)
