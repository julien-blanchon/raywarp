"""Analytic UV gradients of the primitive mappings (P8) against finite differences."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pytest
import warp as wp
from conftest import render_host, required
from fixtures.textures import make_texture

import raywarp as rw
from raywarp._base.types import TEX_LAYOUT_IDENTITY, ShapeType, TexRole, uv_xform
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.textures import formats, uploader
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.materials import maps, texture
from raywarp.core.shade.materials.material import Materials
from raywarp.core.shade.materials.uv_frame import world_uv_frame
from raywarp.scene.images import mips
from raywarp.scene.model.types import UV_SET_WORLD, Slot, TextureRef

_TYPES = (ShapeType.PLANE, ShapeType.SPHERE, ShapeType.ELLIPSOID, ShapeType.CYLINDER, ShapeType.CAPSULE, ShapeType.BOX)


_primitive_uv = texture.make_primitive_uv(tuple(int(t) for t in _TYPES), 0)  # bound 0: every type compiled in


@wp.kernel
def _gradients(
  gtype: int,
  local: wp.array1d[wp.vec3],
  normal: wp.array1d[wp.vec3],
  size: wp.vec3,
  analytic: wp.array2d[wp.vec2],
  numeric: wp.array2d[wp.vec2],
):
  """Per point: (du/dt1, dv/dt1), (du/dt2, dv/dt2) analytic and by central differences."""
  i = wp.tid()
  t1, t2 = texture.orthonormal_basis(normal[i])
  _uv, _rep, _off, gu, gv = _primitive_uv(0, gtype, local[i], normal[i], size, wp.vec2(1.0), False)
  analytic[i, 0] = wp.vec2(wp.dot(gu, t1), wp.dot(gv, t1))
  analytic[i, 1] = wp.vec2(wp.dot(gu, t2), wp.dot(gv, t2))
  h = 1.0e-3
  for k in range(2):
    t = wp.where(k == 0, t1, t2)
    a, _r1, _o1, _gu1, _gv1 = _primitive_uv(0, gtype, local[i] + t * h, normal[i], size, wp.vec2(1.0), False)
    b, _r2, _o2, _gu2, _gv2 = _primitive_uv(0, gtype, local[i] - t * h, normal[i], size, wp.vec2(1.0), False)
    d = a - b
    # Fold the azimuth seam.
    d = d - wp.vec2(wp.round(d[0]), wp.round(d[1]))
    numeric[i, k] = d / (2.0 * h)


def _surface_points(gtype: ShapeType, size: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
  """Random points on a primitive (away from poles, edges and cap seams) and their normals."""
  n = 64
  if gtype == ShapeType.PLANE:
    p = np.c_[rng.uniform(-0.9, 0.9, (n, 2)) * size[:2], np.zeros(n)]
    return p, np.tile([0.0, 0.0, 1.0], (n, 1))
  if gtype in (ShapeType.SPHERE, ShapeType.ELLIPSOID):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[np.abs(d[:, 2]) < 0.9]
    radii = np.full(3, size[0]) if gtype == ShapeType.SPHERE else size
    p = d * radii
    nrm = d / radii
    return p, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
  if gtype == ShapeType.CYLINDER:
    a = rng.uniform(0, 2 * np.pi, n)
    p = np.c_[size[0] * np.cos(a), size[0] * np.sin(a), rng.uniform(-0.9, 0.9, n) * size[1]]
    return p, np.c_[np.cos(a), np.sin(a), np.zeros(n)]
  if gtype == ShapeType.CAPSULE:
    a = rng.uniform(0, 2 * np.pi, n)
    e = rng.uniform(0.1, 1.3, n)  # elevation on the top cap
    d = np.c_[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
    return d * size[0] + [0.0, 0.0, size[1]], d
  # Box: points on the six faces.
  face = rng.integers(0, 6, n)
  axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
  p = rng.uniform(-0.9, 0.9, (n, 3)) * size
  p[np.arange(n), axis] = sign * size[axis]
  nrm = np.zeros((n, 3))
  nrm[np.arange(n), axis] = sign
  return p, nrm


@pytest.mark.parametrize("gtype", _TYPES, ids=lambda t: t.name.lower())
def test_primitive_uv_gradients_match_finite_differences(gtype):
  size = np.array([0.7, 0.4, 0.3]) if gtype != ShapeType.PLANE else np.array([2.0, 1.5, 0.1])
  local, normal = _surface_points(gtype, size, np.random.default_rng(int(gtype)))
  with wp.ScopedDevice("cpu"):
    analytic = wp.zeros((len(local), 2), dtype=wp.vec2)
    numeric = wp.zeros((len(local), 2), dtype=wp.vec2)
    wp.launch(
      _gradients,
      dim=len(local),
      inputs=[int(gtype), wp.array(local, dtype=wp.vec3), wp.array(normal, dtype=wp.vec3), wp.vec3(*size)],
      outputs=[analytic, numeric],
    )
  a, b = analytic.numpy(), numeric.numpy()
  np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------------------------
# Texture filtering depends on the footprint ellipse, not its chosen spanning axes.


_fetch = maps.make_fetch(8)


@wp.kernel
def _sample(
  # Model:
  mats: Materials,
  # In:
  axes: wp.array2d[wp.vec2],
  # Out:
  values: wp.array1d[wp.vec4],
):
  """Sample one common UV with each equivalent ellipse parameterization."""
  i = wp.tid()
  fp = maps.Footprint()
  fp.cone = 0.0
  fp.major = axes[i, 0]
  fp.minor = axes[i, 1]
  values[i] = _fetch(mats, 0, wp.vec2(0.5, 0.30), fp, 1.0)


@wp.kernel
def _principal(
  # In:
  axes: wp.array2d[wp.vec2],
  # Out:
  values: wp.array1d[wp.vec4],
):
  """Expose actual filtering axes for an independent host singular-value comparison."""
  i = wp.tid()
  axis, longer, shorter = maps.ellipse_axes(axes[i, 0], axes[i, 1], wp.vec2(1.0))
  values[i] = wp.vec4(axis[0], axis[1], longer, shorter)


def _device(name: str) -> wp.Device:
  """Use CPU locally and the identical probe on CUDA in the coordinated checkpoint."""
  if name.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  return wp.get_device(name)


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
@pytest.mark.parametrize("width", [64, 256])
def test_filter_is_invariant_to_footprint_basis_rotation(device_name: str, width: int) -> None:
  """A16:1 ellipse keeps transverse detail under basis rotation, including non-square textures."""
  device = _device(device_name)
  texels: np.ndarray[tuple[Literal[64], int, Literal[4]], np.dtype[np.float32]] = np.ones((64, width, 4), np.float32)
  # Horizontal strips: a long horizontal footprint must not blur the narrow vertical detail.
  texels[..., :3] = ((np.arange(64) % 16) < 8)[..., None, None]
  tex = wp.Texture2D(
    texels,
    filter_mode=wp.TextureFilterMode.LINEAR,
    mip_filter_mode=wp.TextureFilterMode.LINEAR,
    num_mip_levels=0,
    device=device,
  )
  mats = Materials()
  mats.textures = wp.array([tex], dtype=wp.Texture2D, device=device)
  mats.tex_size = wp.array([[float(width), 64.0]], dtype=wp.vec2, device=device)
  mats.tex_layout = wp.array([TEX_LAYOUT_IDENTITY], dtype=wp.int32, device=device)
  rows = []
  for degrees in (0.0, 15.0, 45.0, 77.0, -45.0):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    basis = np.array([[16 * c, s], [-16 * s, c]], np.float64)
    # Rows span exactly the same ellipse; this checks the deciding geometry independently.
    np.testing.assert_allclose(basis.T @ basis, np.diag(np.array([256.0, 1.0])), atol=1.0e-12)
    rows.append(basis / np.array([width, 64]))
  axes = wp.array(np.asarray(rows, np.float32), dtype=wp.vec2, device=device)
  values = wp.empty(len(rows), dtype=wp.vec4, device=device)
  wp.launch(_sample, dim=len(rows), inputs=[mats, axes], outputs=[values], device=device)
  actual = values.numpy()
  np.testing.assert_allclose(actual, np.ones_like(actual), atol=1.0e-5)


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
def test_needle_footprint_keeps_the_texel(device_name: str) -> None:
  """A footprint over 2^31:1 in texels (a sliver UV triangle) still averages its taps: the tap
  count is capped before the float -> int conversion, which on the CPU wrapped to a negative
  count (no tap: a black texel)."""
  device = _device(device_name)
  tex = wp.Texture2D(np.ones((4, 4, 4), np.float32), filter_mode=wp.TextureFilterMode.LINEAR, device=device)
  mats = Materials()
  mats.textures = wp.array([tex], dtype=wp.Texture2D, device=device)
  mats.tex_size = wp.array([[4.0, 4.0]], dtype=wp.vec2, device=device)
  mats.tex_layout = wp.array([TEX_LAYOUT_IDENTITY], dtype=wp.int32, device=device)
  data = np.array([[[1.0, 0.0], [0.0, 1.0e-12]], [[1.0, 0.0], [1.0e-3, 1.0e-12]]], np.float32)
  axes = wp.array(data, dtype=wp.vec2, device=device)
  values = wp.empty(len(data), dtype=wp.vec4, device=device)
  wp.launch(_sample, dim=len(data), inputs=[mats, axes], outputs=[values], device=device)
  np.testing.assert_allclose(values.numpy(), 1.0, atol=1.0e-5)


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
@pytest.mark.parametrize("scale", [1.0, 1.0e10, 1.0e-10])
def test_ellipse_lengths_match_independent_singular_values(device_name: str, scale: float) -> None:
  """Skewed, mirrored, round, empty and nearly singular footprints stay finite and geometric."""
  device = _device(device_name)
  data = np.array(
    [
      [[16, 0], [16, 1]],
      [[2, 3], [-4, 5]],
      [[0, -8], [2, 0]],
      [[1, 0], [0, 1]],
      [[0, 0], [0, 0]],
      [[16, 0], [16, 0.001]],
    ],
    np.float32,
  )
  axes = wp.array(data * scale, dtype=wp.vec2, device=device)
  values = wp.empty(len(data), dtype=wp.vec4, device=device)
  wp.launch(_principal, dim=len(data), inputs=[axes], outputs=[values], device=device)
  actual = values.numpy() / scale
  assert np.isfinite(actual).all()
  for basis, value in zip(data, actual, strict=True):
    _left, expected, _right = np.linalg.svd(basis.astype(np.float64), full_matrices=False, compute_uv=True)
    np.testing.assert_allclose(value[2:], expected, rtol=2.0e-6, atol=1.0e-7)
    axis, length = value[:2].astype(np.float64), float(value[2])
    np.testing.assert_allclose(math.hypot(float(axis[0]), float(axis[1])), length, rtol=2.0e-6, atol=1.0e-7)
    np.testing.assert_allclose((basis.T @ basis) @ axis, length**2 * axis, rtol=3.0e-6, atol=1.0e-7)


# ---------------------------------------------------------------------------------------------
# Material maps on a mesh without UVs: each role reads its texture's average.
#
# A constant emissive color is a 1 x 1 white texture tinted by the color
# (`pipeline/upload/materials.py`): before the fix a UV-less mesh skipped every map, so that
# emission was lost (black).


QUAD = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], np.float32)


TRIS = np.array([[0, 1, 2], [0, 2, 3]])


UVS = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], np.float32)


def _radiance(material: rw.Material, uvs: bool, texture: np.ndarray | None = None, sun: bool = False) -> np.ndarray:
  """Mean HDR radiance of a camera 1 m above a 2 x 2 m quad (with / without UVs) of `material`;
  `texture` (H, W, 4): its base-color map."""
  b = rw.SceneBuilder()
  if texture is not None:
    tex = b.add_texture(texture, colorspace="linear")
    material = material.with_texture(Slot.BASE_COLOR, TextureRef(tex))
  mesh = b.add_mesh(QUAD, TRIS, **({"uvs": UVS} if uvs else {}))
  b.add_instance("mesh", mesh=mesh, material=b.add_material(material), xform=np.c_[np.eye(3), np.zeros(3)])
  if sun:
    b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  r = render_host(b.build(), (0.0, 0.0, 1.0), fovy=0.5, res=4)
  return required(r.outputs.hdr).numpy()[0, 0, ..., :3].reshape(-1, 3).mean(0)


@pytest.mark.parametrize("uvs", [True, False])
def test_constant_emissive_color_without_uvs(uvs):
  """A black emitter of color (1, 0.5, 0.25) x 4 nits shows that color, with or without UVs."""
  glow = rw.Material(
    base_color=(0.0, 0.0, 0.0), specular_roughness=1.0, emission_luminance=4.0, emission_color=(1.0, 0.5, 0.25)
  )
  np.testing.assert_allclose(_radiance(glow, uvs), [4.0, 2.0, 1.0], rtol=1e-3)


def test_base_color_map_without_uvs_is_its_average():
  """A base-color map on a UV-less mesh shades with the map's average (its coarsest mip): a
  constant map exactly as with UVs, a two-color checker as its mean color."""
  plain = rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=0.0, specular_roughness=1.0)
  red = np.zeros((2, 2, 4), np.uint8)
  red[..., 0] = red[..., 3] = 255
  with_uv = _radiance(plain, True, red, sun=True)
  assert with_uv[0] > 20.0 * with_uv[1]  # red (green, blue: the specular lobe only)
  np.testing.assert_allclose(_radiance(plain, False, red, sun=True), with_uv, rtol=1e-5)
  checker = red.copy()
  checker[0, 1, :3] = checker[1, 0, :3] = (0, 255, 0)
  mean = red.copy()
  mean[..., :2] = 128  # (0.502, 0.502, 0): the checker's mean to 8 bits
  np.testing.assert_allclose(_radiance(plain, False, checker, sun=True), _radiance(plain, True, mean, sun=True), rtol=1e-2)


# ---------------------------------------------------------------------------------------------
# World-projected texture coordinates (TextureRef.uv_set UV_SET_WORLD, `material.uv_world`).
#
# The Isaac Hospital floor (MI_Hospital_Floor, a UE4 world-aligned triplanar projection) was read as
# its textures' means: no speckles, a flat normal map. A world-projected material now samples its
# maps at the hit's world position, on the plane of the dominant axis of its normal.


_RES = 32


_FOVY = 60.0


_HEIGHT = 2.0


# u = x / 2 + 0.5, v = y / 2 + 0.5 (scene metres).
_WORLD = ((0.5, 0.0, 0.5), (0.0, 0.5, 0.5))


def _floor(uv_set: int, offset: float) -> rw.Scene:
  """A 4 x 4 m floor quad without UVs at x offset `offset` (rotated 90 degrees about z), its base
  color a texture white on its left half (u < 0.5), black on the right, read through `_WORLD`."""
  b = rw.SceneBuilder()
  v = np.zeros((8, 8), np.uint8)
  v[:, :4] = 255
  tex = b.add_texture(np.stack([v, v, v, np.full_like(v, 255)], -1))
  ref = TextureRef(tex, uv_xform=_WORLD, uv_set=uv_set)
  m = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0).with_texture(Slot.BASE_COLOR, ref))
  pts = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]])
  mesh = b.add_mesh(pts, [[0, 1, 2], [0, 2, 3]])
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[[[0.0, -1.0, 0], [1.0, 0.0, 0], [0, 0, 1.0]], [offset, 0.0, 0.0]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=4.0e5, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def _row(scene: rw.Scene) -> tuple[np.ndarray, np.ndarray]:
  """(world x of each pixel column, the middle row's luminance) seen from (0, 0, _HEIGHT) down."""
  s = rw.RenderSettings().replace(**{"performance.traversal": "software", "surfaces.anisotropy": 1})
  r = rw.Renderer(rw.upload(scene), nworld=1, cameras=[rw.CameraSpec(fovy=_FOVY)], resolution=(_RES, _RES), settings=s)
  r.camera_pose.assign(np.array([[[0.0, 0.0, _HEIGHT, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  rgb = required(r.outputs.rgb).numpy()[0, 0, _RES // 2, :, :3].astype(np.float32).mean(-1)
  x = (2.0 * (np.arange(_RES) + 0.5) / _RES - 1.0) * _HEIGHT * np.tan(np.radians(_FOVY / 2))
  r.close()
  return x, rgb


@pytest.mark.parametrize("offset", [0.0, 0.7])
def test_world_projection_follows_the_world_not_the_instance(offset):
  """White at world x in (-1, 0), black in (0, 1) (the texture repeats every 2 m) whatever the
  instance's pose (no mesh UVs)."""
  x, lum = _row(_floor(UV_SET_WORLD, offset))
  assert lum[(x > -0.75) & (x < -0.25)].min() > 200.0 and lum[(x > 0.25) & (x < 0.75)].max() < 40.0


def test_mesh_uvs_without_uvs_read_the_mean():
  """Control: the same material on mesh UV set 0 (the quad has none) reads the texture's mean."""
  _, lum = _row(_floor(0, 0.0))
  assert np.ptp(lum) < 3.0 and lum.min() > 40.0


@wp.kernel
def _probe(m: uv_xform, points: wp.array[wp.vec3], normals: wp.array[wp.vec3], out: wp.array[wp.vec4]):
  """(uv, sum of the tangent's components, uv per length) of `world_uv_frame` at each point."""
  i = wp.tid()
  uv, t, _b, k = world_uv_frame(m, points[i], normals[i], wp.identity(n=3, dtype=float))
  out[i] = wp.vec4(uv[0], uv[1], t[0] + t[1] + t[2], k)


def test_world_frame_on_axis_planes():
  """The plane of the dominant normal axis: (y, z) / (x, z) / (x, y); frame = M^-1 columns."""
  p = np.array([[1.0, 2.0, 3.0]] * 3, np.float32)
  n = np.array([[-0.9, 0.1, 0.2], [0.1, 0.8, -0.3], [0.0, 0.1, -1.0]], np.float32)
  out = wp.zeros(3, dtype=wp.vec4, device="cpu")
  m = uv_xform(2.0, 0.0, 1.0, 0.0, 4.0, 0.0)
  wp.launch(
    _probe, 3, [m, wp.array(p, dtype=wp.vec3, device="cpu"), wp.array(n, dtype=wp.vec3, device="cpu"), out], device="cpu"
  )
  got = out.numpy()
  np.testing.assert_allclose(got[:, :2], [[5.0, 12.0], [3.0, 12.0], [3.0, 8.0]])
  np.testing.assert_allclose(got[:, 2], 0.5)  # dp/du = e0 / 2
  np.testing.assert_allclose(got[:, 3], np.sqrt(8.0))


# ---------------------------------------------------------------------------------------------
# Odd-size mip levels retain the complete source area, including its final row/column.


def _integrate(
  image: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
  """Independent cell-overlap integration over the normalized source-image extent."""
  h, w, channels = image.shape
  nh, nw = max(1, h // 2), max(1, w // 2)
  out = np.zeros((nh, nw, channels), np.float64)
  for y in range(nh):
    for x in range(nw):
      lo_y, hi_y = y * h / nh, (y + 1) * h / nh
      lo_x, hi_x = x * w / nw, (x + 1) * w / nw
      for sy in range(int(lo_y), int(np.ceil(hi_y))):
        for sx in range(int(lo_x), int(np.ceil(hi_x))):
          area = (min(hi_y, sy + 1) - max(lo_y, sy)) * (min(hi_x, sx + 1) - max(lo_x, sx))
          out[y, x] += image[sy, sx] * area / ((hi_y - lo_y) * (hi_x - lo_x))
  return out


@pytest.mark.parametrize("shape", [(3, 3), (5, 7), (1, 7), (5, 1), (6, 5)])
def test_float_mip_integrates_all_source_texels(shape: tuple[int, int]) -> None:
  """A trailing stripe survives and arbitrary texels match independent area integrals."""
  image = np.random.default_rng(4).random((*shape, 3)).astype(np.float32)
  image[-1] += 2.0
  image[:, -1] += 1.0
  ref = _integrate(image.astype(np.float64))
  nh, nw = ref.shape[:2]
  host = mips.downsample(image, transfer="linear")
  core = formats.halve_pairs(image, nw, nh)
  np.testing.assert_allclose(host, ref, rtol=2e-7, atol=2e-7)
  np.testing.assert_allclose(core, ref, rtol=2e-7, atol=2e-7)
  np.testing.assert_allclose(host.mean((0, 1)), image.mean((0, 1)), rtol=2e-7, atol=2e-7)


def test_odd_color_mips_keep_linear_alpha_and_coverage() -> None:
  """An odd edge contributes to sRGB color/alpha and carries the sole covered texels."""
  image = np.zeros((3, 3, 4), np.uint8)
  image[-1] = 255
  result = mips.downsample(image, transfer="srgb")
  np.testing.assert_allclose(mips.SRGB_TO_LINEAR[result[0, 0, :3]], 1 / 3, atol=0.004)
  assert result[0, 0, 3] == 85
  levels = formats.srgb8_mip_chain(image, None, [(3, 3, 1), (1, 1, 1)], ncolor=3, nsrgb=3)
  np.testing.assert_allclose(mips.SRGB_TO_LINEAR[levels[-1][0, 0, :3]], 1 / 3, atol=0.004)
  assert levels[-1][0, 0, 3] == 85
  color = np.zeros((3, 3, 3), np.float32)
  color[..., 0] = 1
  color[-1] = [0, 0, 1]
  weight = np.zeros((3, 3), np.float32)
  weight[-1] = 1
  reduced, coverage = formats.weighted_halve(color, weight, 3, 1, 1)
  assert coverage is not None
  np.testing.assert_allclose(reduced[0, 0], [0, 0, 1])
  np.testing.assert_allclose(coverage, 1 / 3)


@pytest.mark.parametrize("kind", ["u8", "f16"])
def test_odd_host_mips_match_area_reference(kind: Literal["u8", "f16"]) -> None:
  """Odd-axis reductions keep typed storage and integrate the complete image extent."""
  for shape in ((3, 3), (5, 7), (1, 7), (5, 1), (17, 18)):
    raw = np.random.default_rng(8).integers(0, 256, (*shape, 4)).astype(np.uint8)
    half = (raw.astype(np.float32) / 64).astype(np.float16)
    h, w = shape
    nh, nw = max(1, h // 2), max(1, w // 2)
    if kind == "u8":
      expected = np.rint(_integrate(raw.astype(np.float64))).astype(np.uint8)
      np.testing.assert_array_equal(uploader._halve_u8(raw, nw, nh), expected)
    else:
      expected = _integrate(half.astype(np.float64)).astype(np.float16)
      np.testing.assert_array_equal(uploader._halve_f16(half, nw, nh), expected)


@pytest.mark.parametrize("size", [3, 4])
def test_only_npot_texture_identity_changes(size: int) -> None:
  """Reusing base texels cannot revive pre-fix odd mips; ordinary power-two keys stay exact."""
  image = np.random.default_rng(4).integers(0, 256, (size, size, 4)).astype(np.uint8)
  built = make_texture(image, srgb=False, device=wp.get_device("cpu"))
  kept, _layout = formats.pack_channels(image)
  stored = np.ascontiguousarray(image[..., list(kept)])
  ncolor = sum(c < 3 for c in kept)
  address = int(wp.TextureAddressMode.WRAP)
  # The key without the odd-mip tag (the removed shader-side sRGB flag `srgb8` is no longer part of
  # it: exact color is stored linear, so its texels follow from `stored` and `srgb`).
  old = bake_cache.texture_digest(stored, False, ncolor, address, address, None)
  assert (bake_cache._texture(built.texture) == old) == (size == 4)


@wp.kernel
def _coarse(
  # Model:
  image: wp.Texture2D,
  # In:
  layout: int,
  # Out:
  value: wp.array1d[wp.vec4],
):
  """Read the actual final uploaded mip through the production layout decoder."""
  value[0] = texture.sample_layout(image, layout, wp.vec2(0.5), 20.0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
def test_uploaded_odd_material_and_environment_keep_final_row(device_name: str) -> None:
  """Both material/device and environment/host upload expose the corrected coarse mip."""
  if device_name.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  device = wp.get_device(device_name)
  image = np.zeros((3, 3, 4), np.uint8)
  image[..., 3] = 255
  image[-1, :, :3] = 255
  material = make_texture(image, srgb=False, device=device)
  environment = envmap.EnvironmentMap(image[..., :3].astype(np.float32) / 255, lighting=False, device=device)
  value = wp.empty(1, dtype=wp.vec4, device=device)
  for sampler, layout, expected in (
    (material.texture, material.layout, np.full(3, 1 / 3, np.float32)),
    (environment.radiance, TEX_LAYOUT_IDENTITY, np.full(3, 1 / 3, np.float16).astype(np.float32)),
  ):
    wp.launch(_coarse, dim=1, inputs=[sampler, layout], outputs=[value], device=device)
    np.testing.assert_allclose(value.numpy()[0, :3], expected, rtol=0, atol=1e-7)
  rgba = np.concatenate([image[..., :3].astype(np.float32) / 255, np.ones((3, 3, 1), np.float32)], axis=-1)
  assert bake_cache._texture(environment.radiance) != bake_cache.texture_digest(rgba.astype(np.float16), "envmap")


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
@pytest.mark.parametrize("field", ["midpoint", "hdr"])
def test_power_two_environment_preserves_original_warp(device_name: str, field: Literal["midpoint", "hdr"]) -> None:
  """Actual original Warp sampling and texture identity survive the shared constructor."""
  if device_name.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  device = wp.get_device(device_name)
  if field == "midpoint":
    values = np.array([[1, 2**-11], [2**-24, 0]], np.float32)
    image = np.repeat(values[..., None], 3, axis=2)
  else:
    image = (np.random.default_rng(39).random((4, 8, 3)) * 10000).astype(np.float32)
  environment = envmap.EnvironmentMap(image, lighting=False, device=device)
  rgba = np.concatenate([image, np.ones((*image.shape[:2], 1), np.float32)], axis=-1).astype(np.float16)
  original = wp.Texture2D(
    rgba,
    filter_mode=wp.TextureFilterMode.LINEAR,
    address_mode_u=wp.TextureAddressMode.WRAP,
    address_mode_v=wp.TextureAddressMode.CLAMP,
    num_mip_levels=0,
    mip_filter_mode=wp.TextureFilterMode.LINEAR,
    device=device,
  )
  actual, expected = wp.empty(1, dtype=wp.vec4, device=device), wp.empty(1, dtype=wp.vec4, device=device)
  for sampler, output in ((environment.radiance, actual), (original, expected)):
    wp.launch(_coarse, dim=1, inputs=[sampler, TEX_LAYOUT_IDENTITY], outputs=[output], device=device)
  if field == "midpoint":
    # The coarse texel (1 + 2^-11 + 2^-24) / 4 sits 1.5e-8 above a float16 tie: our chain averages
    # in float32, as Warp 1.18's host chain does, loses the 2^-24 term and rounds to 0.25; Warp
    # 1.19 averages in float64 and gets 0.25024 (one float16 step). Only our value is pinned.
    np.testing.assert_array_equal(actual.numpy()[0, :3], np.full(3, 0.25, np.float32))
  else:
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
  assert bake_cache._texture(environment.radiance) == bake_cache.texture_digest(rgba, "envmap")


# ---------------------------------------------------------------------------------------------
# The sole-color-map fast path follows live material roles without changing mapped shading.


@pytest.mark.parametrize("channels", (3, 4))
@pytest.mark.parametrize("normal_mapped", (False, True))
def test_live_absent_and_mapped_roles(channels: int, normal_mapped: bool) -> None:
  """World one gains and loses a live texture; world zero and exact reset stay unchanged."""
  builder = rw.SceneBuilder()
  texels = np.zeros((4, 4, channels), np.uint8)
  texels[..., 0] = 255
  if channels == 4:
    texels[..., 3] = 255
  color = builder.add_texture(texels, colorspace="linear")
  plain = rw.Material(base_color=(1.0, 1.0, 1.0), specular_weight=0.0)
  mapped = plain.with_texture(Slot.BASE_COLOR, rw.scene.TextureRef(color))
  if normal_mapped:
    normals = np.full((4, 4, 3), (220, 128, 220), np.uint8)
    normal = builder.add_texture(normals, colorspace="linear")
    mapped = mapped.with_texture(Slot.GEOMETRY_NORMAL, rw.scene.TextureRef(normal))
  plain_id = builder.add_material(plain)
  mapped_id = builder.add_material(mapped)
  mesh = builder.add_mesh(
    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
    [[0, 1, 2], [0, 2, 3]],
    uvs=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
  )
  builder.add_instance("mesh", mesh=mesh, material=plain_id)
  # Keep the mapped material in the rendered instance set so its role is compiled, but outside
  # the camera view; the visible instance starts on the absent-role path.
  builder.add_instance("mesh", mesh=mesh, material=mapped_id, xform=np.c_[np.eye(3), [10.0, 0.0, 0.0]])
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0))
  renderer = rw.Renderer(
    rw.upload(builder.build(), "cpu"),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=40.0)],
    resolution=(16, 16),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      shadows=rw.settings.Shadows(enabled=False),
      lighting=rw.settings.Lights(camera_light=False),
      outputs=rw.Outputs.of("hdr", "normal", "uv"),
      performance=rw.settings.Performance(traversal="software"),
    ).replace(),
    per_world=rw.PerWorld(material=["texture"]),
  )
  try:
    assert renderer.features.base_color_map
    assert renderer.features.normal_map == normal_mapped
    renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]] * 2, np.float32))
    table = renderer.per_world["material.texture"]
    rows = table.numpy().copy()
    color_roles = (int(TexRole.RGB), int(TexRole.RGBA))
    assert all(rows[0, 0, role] < 0 for role in color_roles)
    assert any(rows[0, 1, role] >= 0 for role in color_roles)

    def frame() -> dict[str, np.ndarray]:
      renderer.render(seed=17)
      return {name: required(getattr(renderer.outputs, name)).numpy().copy() for name in ("hdr", "normal", "uv")}

    before = frame()
    mapped_rows = rows.copy()
    mapped_rows[1, 0] = mapped_rows[0, 1]
    table.assign(mapped_rows)
    renderer.reset_history(wp.array([False, True], dtype=bool, device="cpu"))
    after = frame()
    for name in before:
      np.testing.assert_array_equal(after[name][0], before[name][0])
    assert float(np.max(np.abs(after["hdr"][1] - before["hdr"][1]))) > 0.1
    if normal_mapped:
      assert float(np.max(np.abs(after["normal"][1] - before["normal"][1]))) > 0.05
    table.assign(rows)
    renderer.reset_history(wp.array([False, True], dtype=bool, device="cpu"))
    reset = frame()
    for name in before:
      np.testing.assert_array_equal(reset[name], before[name])
  finally:
    renderer.close()
