"""The dominant reflection ray's share of a GGX lobe (`kernels/reflections.core_share`): the NDF's
projected mass within the core slope, integrated numerically (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from raywarp.core.kernels import reflections


@wp.kernel
def _shares(alpha: wp.array[float], out: wp.array[float]):
  i = wp.tid()
  out[i] = reflections.core_share(alpha[i])


def _ggx_projected_mass(alpha: float, slope: float) -> float:
  """Integral of D(h) cos(theta_h) over theta_h < atan(slope) (GGX, numerically)."""
  theta = np.linspace(0.0, np.arctan(slope), 200001)
  t2 = np.tan(theta) ** 2
  d = alpha**2 / (np.pi * np.cos(theta) ** 4 * (alpha**2 + t2) ** 2)
  f = d * np.cos(theta) * np.sin(theta) * 2.0 * np.pi
  return float(np.sum((f[1:] + f[:-1]) * 0.5 * np.diff(theta)))


@pytest.mark.parametrize("roughness", [0.05, 0.15, 0.25, 0.3, 0.45])
def test_core_share_is_the_ggx_mass_within_the_core(roughness: float):
  alpha = roughness * roughness
  out = wp.zeros(1, dtype=float, device="cpu")
  wp.launch(_shares, dim=1, inputs=[wp.array([alpha], dtype=float, device="cpu")], outputs=[out], device="cpu")
  expected = _ggx_projected_mass(alpha, reflections.CORE_SLOPE)
  assert out.numpy()[0] == pytest.approx(expected, abs=1e-4)


def test_mirror_finishes_keep_the_ray_and_rough_metals_blur():
  """Chrome keeps its mirror, a 0.25 steel is mostly the blurred environment term."""
  alphas = np.array([0.05, 0.25], np.float32) ** 2
  out = wp.zeros(2, dtype=float, device="cpu")
  wp.launch(_shares, dim=2, inputs=[wp.array(alphas, dtype=float, device="cpu")], outputs=[out], device="cpu")
  chrome, steel = out.numpy()
  assert chrome > 0.99 and steel < 0.5
