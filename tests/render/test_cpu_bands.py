"""The CPU megakernel runs in bands of rays over the cores (`pixel.megakernel`'s banded form,
`launches.BandedLaunch`): the same pixels as one launch over every ray, for the shaded, the
temporal and the geometry-only kernels."""

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.trace import software_backend

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
"""

_FORMS: dict[str, dict[str, object]] = {
  "shaded": {"outputs": rw.Outputs.of("rgb", "depth", "seg")},
  "temporal": {"outputs": rw.Outputs.of("rgb", "depth", "motion")},
  "geometry": {"outputs": rw.Outputs.of("depth", "seg")},
}


def _frames(scene, kw: dict[str, object]) -> list[np.ndarray]:
  """Two frames' outputs of a 2-world 64 x 48 render (3072 rays: 3 bands of
  `software_backend._CPU_BAND`)."""
  s = scene(XML, nworld=2, cam_res=(64, 48), **kw)
  out = []
  for _ in range(2):
    o = s.r.render(s.d, seed=3)
    out += [getattr(o, k).numpy().copy() for k in ("rgb", "depth", "seg") if getattr(o, k) is not None]
  return out


@pytest.mark.parametrize("form", list(_FORMS))
def test_bands_render_the_single_launch_pixels(scene, form: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """Banded and single-launch renders are bit-identical (CPU)."""
  if not wp.get_device().is_cpu:
    pytest.skip("the banded form is the CPU's")
  banded = _frames(scene, _FORMS[form])
  monkeypatch.setattr(software_backend, "_banded", lambda _f, _caps: False)
  single = _frames(scene, _FORMS[form])
  assert len(banded) == len(single) > 0
  for a, b in zip(banded, single, strict=True):
    np.testing.assert_array_equal(a, b)
