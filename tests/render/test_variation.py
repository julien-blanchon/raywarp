"""Texture variation without copies (`rw.randomize`): per-world UV offset /
rotation and hue on one shared texture, through the per-material `material.tex_variation` table
(contrast: `material.tex_affine`'s scale / bias rows)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp._base.types import TexRole
from raywarp.scene.model.types import MaterialId, Slot, TextureRef
from raywarp.settings.fields import OutputKind

_QUAD = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_RGB = int(TexRole.RGB)


def _quad(image: np.ndarray, slot: Slot = Slot.BASE_COLOR) -> rw.Scene:
  """A 2 x 2 m quad at the origin with `image` (linear) in texture `slot`, under a sun."""
  b = rw.SceneBuilder()
  tex = b.add_texture(image, colorspace="linear")
  ref = (
    TextureRef(tex, scale=(2.0, 2.0, 2.0, 1.0), bias=(-1.0, -1.0, -1.0, 0.0))
    if slot == Slot.GEOMETRY_NORMAL
    else TextureRef(tex)
  )
  mat = rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=0.0, specular_roughness=1.0).with_texture(slot, ref)
  b.add_instance("mesh", mesh=b.add_mesh(_QUAD, [[0, 1, 2], [0, 2, 3]], uvs=_UV), material=b.add_material(mat))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def _render(
  image: np.ndarray, rows: list[np.ndarray | None], slot: Slot = Slot.BASE_COLOR, kind: OutputKind = "hdr"
) -> np.ndarray:
  """(nworld, H, W, 3) output `kind` of the quad (`image` in `slot`) seen from above, world k with
  texture variation `rows[k]` (None: the uploaded one)."""
  with wp.ScopedDevice("cpu"):
    scene = rw.upload(_quad(image, slot))
    s = rw.RenderSettings().replace(
      **{
        "outputs": rw.Outputs.of(kind),
        "environment.enabled": False,
        "performance.traversal": "software",
        "gi.enabled": False,
      }
    )
    n = len(rows)
    r = rw.Renderer(
      scene,
      nworld=n,
      cameras=[rw.CameraSpec(fovy=30.0)],
      resolution=(24, 24),
      settings=s,
      per_world=rw.PerWorld(material=["tex_variation"]),
    )
    r.camera_pose.assign(np.tile(np.array([0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 1.0], np.float32), (n, 1, 1)))
    table = r.per_world["material.tex_variation"].numpy()
    assert table.shape == (n, 1, 4)
    for k, row in enumerate(rows):
      if row is not None:
        table[k, 0] = row
    r.per_world["material.tex_variation"].assign(table)
    r.render()
    return getattr(r.outputs, kind).numpy()[:, 0, ..., :3]


def _checker(n: int = 16) -> np.ndarray:
  """A (n, n, 4) float32 red / blue checker of 4-texel cells, with a green corner block."""
  y, x = np.mgrid[0:n, 0:n]
  img = np.zeros((n, n, 4), np.float32)
  red = ((x // 4 + y // 4) % 2) == 0
  img[..., 0], img[..., 2], img[..., 3] = red, ~red, 1.0
  img[: n // 4, : n // 4] = (0.0, 1.0, 0.0, 1.0)
  return img


def test_zero_variation_is_bit_identical_and_offsets_of_one_period_too():
  """A zero row renders the texture as uploaded; a whole-texture offset (1, 1) wraps onto itself."""
  img = _render(_checker(), [None, np.zeros(4, np.float32), np.array([1.0, 1.0, 0.0, 0.0], np.float32)])
  np.testing.assert_array_equal(img[1], img[0])
  np.testing.assert_allclose(img[2], img[0], atol=1e-3)


def test_offset_and_rotation_move_the_texture():
  """Half an offset renders as the texture rolled by half its size, a half turn as the texture
  turned by 180 degrees (both unambiguous whatever the axes' directions); both move the green
  corner."""
  checker = _checker()
  img = _render(checker, [None, np.array([0.5, 0.5, 0.0, 0.0], np.float32), np.array([0.0, 0.0, np.pi, 0.0], np.float32)])
  green = [np.unravel_index(np.argmax(im[..., 1] - im[..., 0] - im[..., 2]), im.shape[:2]) for im in img]
  assert green[1] != green[0] and green[2] != green[0]
  rolled = _render(np.roll(checker, checker.shape[0] // 2, axis=(0, 1)), [None])[0]
  np.testing.assert_allclose(img[1], rolled, atol=1e-3)
  turned = _render(np.ascontiguousarray(checker[::-1, ::-1]), [None])[0]
  np.testing.assert_allclose(img[2], turned, atol=1e-3)


def _normal_field(n: int = 16) -> np.ndarray:
  """A (n, n, 4) float32 normal map ([0, 1] encoded, OpenGL +Y up) of a smooth, asymmetric
  relief: the slope turns with the column and grows with the row."""
  y, x = np.mgrid[0:n, 0:n] + 0.5
  angle, tilt = 2.0 * np.pi * x / n, 0.2 + 0.5 * y / n
  return _encode(np.sin(tilt) * np.cos(angle), np.sin(tilt) * np.sin(angle))


def _encode(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
  """(n, n, 4) [0, 1] encoding of the unit tangent-space normals of slopes (`nx`, `ny`)."""
  img = np.ones((*nx.shape, 4), np.float32)
  img[..., 0], img[..., 1], img[..., 2] = (nx + 1.0) / 2.0, (ny + 1.0) / 2.0, (np.sqrt(1.0 - nx**2 - ny**2) + 1.0) / 2.0
  return img


def test_uv_rotation_turns_the_normal_map_relief():
  """A UV rotation of +pi / 2 renders the normals of the normal map turned on the host: its texels
  turned a quarter turn clockwise (`np.rot90(k=-1)`: a UV rotation by +theta shows the texture
  turned by -theta) and their tangent-space slopes turned alike ((x, y) -> (y, -x) in the image's
  +Y-up axes). The rotation used to move the texels only: the relief was read in the unturned
  tangent frame (it rendered as the texels turned with their slopes kept)."""
  img = _normal_field()
  nx, ny = img[..., 0] * 2.0 - 1.0, img[..., 1] * 2.0 - 1.0
  turned = _encode(np.rot90(ny, -1), np.rot90(-nx, -1))
  varied = _render(img, [None, np.array([0.0, 0.0, np.pi / 2.0, 0.0], np.float32)], Slot.GEOMETRY_NORMAL, "normal")[1]
  host = _render(turned, [None], Slot.GEOMETRY_NORMAL, "normal")[0]
  assert np.abs(host[..., :2]).max() > 0.3  # the relief shows
  # float16 texels: the turned slopes round differently (<= 4e-3 measured); the bug was off by 0.87.
  np.testing.assert_allclose(varied, host, atol=1e-2)


def test_hue_rotation_turns_red_to_green_and_keeps_grey():
  """A third of a turn maps red to green, blue to red; grey (the axis) stays."""
  img = np.zeros((4, 4, 4), np.float32)
  img[..., 0], img[..., 3] = 1.0, 1.0
  out = _render(img, [None, np.array([0.0, 0.0, 0.0, 2.0 * np.pi / 3.0], np.float32)])
  red, turned = out[0].reshape(-1, 3).mean(0), out[1].reshape(-1, 3).mean(0)
  np.testing.assert_allclose(turned, red[[2, 0, 1]], rtol=1e-3, atol=1e-4)
  grey = np.full((4, 4, 4), 0.5, np.float32)
  out = _render(grey, [None, np.array([0.0, 0.0, 0.0, 1.0], np.float32)])
  np.testing.assert_allclose(out[1], out[0], rtol=1e-4, atol=1e-5)


def test_jitter_draws_per_world_variations_and_contrast():
  """`rw.randomize.jitter`: offsets / rotation / hue per (world, material) in their ranges, zero
  ranges keep the table; `rw.randomize.contrast` folds a factor into the color roles' scale /
  bias only, on a layered (15-role) table too (BUG-059: it rejected one)."""
  table = np.zeros((3, 2, 4), np.float32)
  rng = np.random.default_rng(0)
  np.testing.assert_array_equal(rw.randomize.jitter(table, rng), table)
  out = rw.randomize.jitter(table, rng, uv_offset=1.0, uv_rotation=np.pi, hue=0.5)
  assert out.shape == table.shape and np.ptp(out[:, 0, 0]) > 0.0  # differs per world
  assert (out[..., :2] >= 0.0).all() and (out[..., :2] < 1.0).all()
  assert (np.abs(out[..., 2]) <= np.pi).all() and (np.abs(out[..., 3]) <= 0.5).all()
  only = rw.randomize.jitter(table, rng, hue=0.5, materials=np.array([1]))
  assert (only[:, 0] == 0.0).all() and (only[:, 1, 3] != 0.0).all() and (only[:, 1, :3] == 0.0).all()
  affine = np.tile(np.array([[1.0] * 4, [0.0] * 4], np.float32), (3, 2, len(TexRole), 1, 1))
  np.testing.assert_array_equal(rw.randomize.contrast(affine, rng, amount=0.0), affine)
  out = rw.randomize.contrast(affine, rng, amount=0.3)
  c = out[:, :, _RGB, 0, 0]
  assert np.ptp(c) > 0.0
  np.testing.assert_allclose(out[:, :, _RGB, 1, 0], (1.0 - c) * rw.randomize.CONTRAST_PIVOT, rtol=1e-6)
  np.testing.assert_array_equal(out[:, :, int(TexRole.ORM)], affine[:, :, int(TexRole.ORM)])


def test_variation_rejects_bad_ranges_and_materials():
  """Review C O1: a negative material id wrapped to the last material, an amount > 1 drew negative
  contrast factors (inverted colors) and NaN ranges were accepted; they now raise at the call."""
  rng = np.random.default_rng(0)
  table = np.zeros((2, 3, 4), np.float32)
  with pytest.raises(ValueError, match=r"materials: ids must be in \[0, 3\)"):
    rw.randomize.jitter(table, rng, hue=0.5, materials=[MaterialId(-1)])
  with pytest.raises(ValueError, match="materials"):
    rw.randomize.jitter(table, rng, hue=0.5, materials=[MaterialId(3)])
  with pytest.raises(ValueError, match="hue must be a finite value"):
    rw.randomize.jitter(table, rng, hue=float("nan"))
  with pytest.raises(ValueError, match="uv_offset"):
    rw.randomize.jitter(table, rng, uv_offset=-1.0)
  with pytest.raises(ValueError, match=r"jitter: a \(nworld, nmaterial, 4\) table"):
    rw.randomize.jitter(np.zeros((2, 3), np.float32), rng)  # pyrefly: ignore[bad-argument-type]  # a 2-D table on purpose
  affine = np.tile(np.array([[1.0] * 4, [0.0] * 4], np.float32), (2, 3, len(TexRole), 1, 1))
  with pytest.raises(ValueError, match=r"amount must be a finite value in \[0, 1.0\]"):
    rw.randomize.contrast(affine, rng, amount=1.5)
  with pytest.raises(ValueError, match="contrast: a"):
    rw.randomize.contrast(affine[:, :, :2], rng, amount=0.1)
  only = rw.randomize.contrast(affine, rng, amount=0.5, materials=[MaterialId(2)])
  np.testing.assert_array_equal(only[:, :2], affine[:, :2])
