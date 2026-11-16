"""Anisotropic texture filtering (B48, `surfaces.anisotropy`) on an obliquely seen sign.

The bug (Office "RECEPTION" sign, balanced): the taps were spread by the surface footprint alone
and filtered at the isotropic texel density, so a texture denser along one UV axis than the other
(a square texture on a wide sign, a rotated UV layout) got taps further apart than their filter
width: ghosted, smeared copies along the sign, worse than isotropic filtering. The footprint is now
measured in each texture's texels (the hardware scheme).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
from raywarp.scene.model.types import Slot, TextureRef

_RES = (128, 64)


def _sign(aspect: int, uv_rotated: bool) -> rw.Scene:
  """A 2 x 0.5 m quad (x-z plane) turned 72 degrees about z, textured with an `aspect`:1 checker
  of 8-texel cells (UVs turned by 90 degrees with `uv_rotated`), lit from the camera."""
  b = rw.SceneBuilder()
  y, x = np.mgrid[0:64, 0 : 64 * aspect]
  v = np.where(((x // 8) + (y // 8)) % 2 == 0, 240, 20).astype(np.uint8)
  tex = b.add_texture(np.stack([v, v, v, np.full_like(v, 255)], -1))
  m = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=0.0, specular_roughness=1.0).with_texture(
      Slot.BASE_COLOR, TextureRef(tex)
    )
  )
  uvs = [[1, 0], [1, 1], [0, 1], [0, 0]] if uv_rotated else [[0, 0], [1, 0], [1, 1], [0, 1]]
  pts = [[-1.0, 0, -0.25], [1.0, 0, -0.25], [1.0, 0, 0.25], [-1.0, 0, 0.25]]
  mesh = b.add_mesh(np.array(pts), [[0, 1, 2], [0, 2, 3]], uvs=np.array(uvs, np.float64))
  c, s = np.cos(np.radians(72.0)), np.sin(np.radians(72.0))
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[[[c, -s, 0], [s, c, 0], [0, 0, 1]], [0.0, 0.0, 1.0]])
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.DISTANT,
      intensity=4.0e5,
      angle=0.0,
      xform=np.c_[[[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], np.zeros(3)],
    )
  )
  return b.build()


def _render(scene: rw.GpuScene, spp: int, anisotropy: int) -> tuple[np.ndarray, np.ndarray]:
  """(display RGB, (H, W) mask of the pixels inside the sign, away from its silhouette) of the
  sign seen from (0, -3, 1) along +y."""
  s = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("rgb", "seg"),
      "aa.spp": spp,
      "surfaces.anisotropy": anisotropy,
      "performance.traversal": "software",
    }
  )
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=22.0)], resolution=_RES, settings=s)
  q = np.sin(np.pi / 4)
  r.camera_pose.assign(np.array([[[0.0, -3.0, 1.0, q, 0.0, 0.0, q]]], np.float32))
  r.render()
  inside = required(r.outputs.seg).numpy()[0, 0] == 0
  inner = inside.copy()
  for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    inner &= np.roll(inside, (dy, dx), axis=(0, 1))
  return required(r.outputs.rgb).numpy()[0, 0, ..., :3].astype(np.float32) / 255.0, inner


# (texture aspect, UVs turned, largest RMSE ratio anisotropy 8 / 1). A square texture on the 4:1
# sign has 4x the texel density along v: seen at 72 degrees its footprint is about round in texels
# (one tap: as good as isotropic); before the fix anisotropy 8 was 2x worse there (the ghosting),
# 1.4x with the UVs turned. An 8:1 label (uniform density) is the control: right before and after.
# The square texture is the slowest case (~40 s on the CPU): slow; the rotated-UV case keeps the
# fix's effect in the quick subset.
_CASES = [
  pytest.param(1, False, 1.02, id="square_texture", marks=pytest.mark.slow),
  pytest.param(8, False, 0.85, id="label_8x1"),
  pytest.param(4, True, 0.85, id="rotated_uvs"),
]


@pytest.mark.parametrize(("aspect", "uv_rotated", "ratio"), _CASES)
def test_anisotropic_filtering_beats_isotropic_on_an_oblique_sign(aspect, uv_rotated, ratio):
  """Inside the sign, anisotropy 8 is closer to a 64-spp reference than anisotropy 1 (never
  worse), whatever the texel density along each UV axis."""
  with wp.ScopedDevice("cpu"):
    scene = rw.upload(_sign(aspect, uv_rotated))
    ref, inner = _render(scene, 64, 1)
    rmse = {a: float(np.sqrt(((_render(scene, 1, a)[0] - ref)[inner] ** 2).mean())) for a in (1, 8)}
  assert rmse[8] < ratio * rmse[1], rmse
