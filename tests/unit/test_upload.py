"""Host scene -> GpuScene upload (`core/pipeline/upload/`, `core/pipeline/textures/uploader.py`) on
small SceneBuilder scenes (CPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, render_host, required

import raywarp as rw
from raywarp._base.types import CLOSED_PANE, TWO_SIDED, ColorRule, TexRole
from raywarp.core.pipeline.gpu_scene.binding import Binding, Slots
from raywarp.core.pipeline.textures import formats
from raywarp.core.shade.materials import texture
from raywarp.scene.model.material import Color
from raywarp.scene.model.types import AlphaMode, MaterialId, Slot, TextureId, TextureRef

DEVICE = "cpu"
QUAD = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
QUAD_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _upload(host: rw.Scene) -> rw.GpuScene:
  """The host scene uploaded on the CPU."""
  with wp.ScopedDevice(DEVICE):
    return rw.upload(host)


def test_opacity_minimum_survives_a_shared_texture_recipe():
  """An occlusion map and a blended opacity map reading the same channel share one core texture:
  the opacity minimum is still taken (it decides the translucency and the shadow transmittance).

  Before the fix the recipe key ignored the minimum request: the occlusion slot registered the
  texture first, the opacity slot reused it without a minimum (read as 0, fully see-through).
  """
  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((4, 4, 4), 255, np.uint8), colorspace="linear")
  base = rw.Material(base_color=(0.5, 0.5, 0.5))
  occluded = b.add_material(base.with_texture(Slot.OCCLUSION, TextureRef(tex, channel=0)))
  blended = b.add_material(
    rw.Material(base_color=(0.5, 0.5, 0.5), ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.BLEND)).with_texture(
      Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=0)
    )
  )
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  b.add_instance("mesh", mesh=mesh, material=occluded)
  b.add_instance("mesh", mesh=mesh, material=blended)
  scene = _upload(b.build())
  texid = scene.tables["material.texture"].numpy()[0]
  assert texid[occluded, int(TexRole.ORM)] == texid[blended, int(TexRole.OPACITY)]  # one texture
  assert not scene.capabilities.alpha_map  # the opacity map is 1 everywhere
  np.testing.assert_array_equal(scene.instance_shadow_transmit[1], 0.0)  # opaque: no light through


def test_multi_source_wrap_conflict_is_reported():
  """A packed texture of sources with different wrap modes keeps the first source's and says so."""
  b = rw.SceneBuilder()
  img = np.full((4, 4, 4), 128, np.uint8)
  clamped = b.add_texture(img, colorspace="linear", wrap=("clamp", "clamp"))
  repeated = b.add_texture(img + 1, colorspace="linear", wrap=("repeat", "repeat"))
  mat = rw.Material().with_texture(Slot.SPECULAR_ROUGHNESS, TextureRef(clamped, channel=0))
  mat = mat.with_texture(Slot.BASE_METALNESS, TextureRef(repeated, channel=0))
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=b.add_material(mat))
  host = b.build()
  _upload(host)
  assert host.report.approximated["gpu.texture.wrap(sources disagree: the first one's)"] == 1


@wp.kernel
def _sample_lod(tex: wp.Texture2D, layout: int, lod: float, out: wp.array1d[wp.vec4]):
  """Samples `tex` (texel layout `layout`) at the image centre at mip level `lod`."""
  out[0] = texture.sample_layout(tex, layout, wp.vec2(0.5, 0.5), lod)


def _texel(scene: rw.GpuScene, tex_id: int, lod: float) -> np.ndarray:
  """The centre texel of core texture `tex_id` at mip level `lod`, read through its layout."""
  out = wp.zeros(1, dtype=wp.vec4, device=DEVICE)
  layout = int(scene.tex_layout.numpy()[tex_id])
  wp.launch(_sample_lod, dim=1, inputs=[scene.textures[tex_id], layout, lod, out], device=DEVICE)
  return out.numpy()[0]


def test_base_color_mips_are_coverage_weighted():
  """B13: a cut-out's base color mips average the visible texels only (no dark halo).

  The 2 x 2 image is opaque red on the left, transparent black on the right; its 1 x 1 mip is red
  (it was (0.5, 0, 0): the cut-away black bled in).
  """
  img = np.zeros((2, 2, 4), np.uint8)
  img[:, 0] = (255, 0, 0, 255)
  b = rw.SceneBuilder()
  tex = b.add_texture(img, colorspace="srgb")
  mat = rw.Material(base_color=(1.0, 1.0, 1.0), ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5))
  mat = mat.with_texture(Slot.BASE_COLOR, TextureRef(tex)).with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3))
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=b.add_material(mat))
  scene = _upload(b.build())
  rgb = int(scene.tables["material.texture"].numpy()[0, 0, int(TexRole.RGB)])
  np.testing.assert_allclose(_texel(scene, rgb, 0.0)[:3], [0.5, 0.0, 0.0], atol=1e-3)  # bilinear at the centre
  np.testing.assert_allclose(_texel(scene, rgb, 1.0), [1.0, 0.0, 0.0, 1.0], atol=1e-3)


def test_coverage_mip_chain_keeps_uncovered_blocks():
  """A fully uncovered 2 x 2 block keeps its plain average; covered texels are weighted."""
  from raywarp.core.pipeline.textures.formats import coverage_mip_chain

  base = np.zeros((2, 4, 4), np.float16)
  base[:, :2, 0] = 1.0  # left block red, coverage 0: plain average
  base[0, 2:, 1] = 1.0  # right block: green on top (covered), black below (not)
  coverage = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], np.float32)
  levels = coverage_mip_chain(base, coverage, [(4, 2, 1), (2, 1, 1), (1, 1, 1)])
  np.testing.assert_allclose(levels[1][0, 0, :3], [1.0, 0.0, 0.0])
  np.testing.assert_allclose(levels[1][0, 1, :3], [0.0, 1.0, 0.0])
  np.testing.assert_allclose(levels[2][0, 0, :3], [0.0, 1.0, 0.0])  # only the green texels are covered


def test_roll_yaw_is_a_yaw_rotation():
  """P15: rolling columns equals resampling the map through the same yaw about +z."""
  from raywarp.core.shade.environment.latlong import rotate_latlong
  from raywarp.scene.images import latlong

  rng = np.random.default_rng(0)
  image = rng.random((8, 32, 3)).astype(np.float32)
  for yaw in (0.3, -1.1, 2.0 * np.pi / 32 * 5):
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(latlong.roll_yaw(image, yaw), rotate_latlong(image, rz, device=wp.get_device()), atol=1e-4)


def test_maps_with_different_yaws_keep_their_own():
  """P15: a second dome's own yaw is baked into its texels (it was replaced by the first map's)."""
  b = rw.SceneBuilder()
  img = np.zeros((4, 8, 3), np.float32)
  img[:, 0] = 1.0
  tex = b.add_texture(img, colorspace="linear")
  quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  b.add_environment(rw.scene.EnvironmentLight(texture=tex, intensity=1.0))
  b.add_environment(rw.scene.EnvironmentLight(texture=tex, intensity=1.0, rotation=quarter))
  host = b.build()
  scene = _upload(host)
  assert scene.environment.yaw == 0.0
  assert not host.report.approximated  # no "first map's yaw" approximation any more


def _uv_sphere(n: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """(points, triangles, normals, uvs) of a unit UV sphere (outward, counter-clockwise faces)."""
  th, ph = np.meshgrid(np.linspace(0.05, np.pi - 0.05, n), np.linspace(0.0, 2.0 * np.pi, 2 * n), indexing="ij")
  pts = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
  uv = np.stack([ph / (2.0 * np.pi), 1.0 - th / np.pi], -1).reshape(-1, 2)
  w = 2 * n
  tris = []
  for i in range(n - 1):
    for j in range(w - 1):
      a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
      tris += [(c, b, a), (c, d, b)]
  return pts, np.array(tris), pts.copy(), uv


def _stretched_scene(stretch: np.ndarray, baked: bool) -> rw.Scene:
  """A textured sphere mesh under a sun, posed by R S: S as the instance's or baked into the
  mesh."""
  pts, tris, nrm, uv = _uv_sphere()
  c, s = np.cos(0.4), np.sin(0.4)
  rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  b = rw.SceneBuilder()
  img = np.zeros((8, 8, 4), np.uint8)
  img[..., 3] = 255
  img[::2, :, 0] = 255
  img[:, ::2, 1] = 255
  tex = b.add_texture(img)
  mat = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=0.4).with_texture(Slot.BASE_COLOR, TextureRef(tex))
  )
  if baked:
    pts, nrm = pts @ stretch.T, nrm @ np.linalg.inv(stretch)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if np.linalg.det(stretch) < 0.0:
      tris = tris[:, ::-1]  # keep counter-clockwise front faces
    xform = np.c_[rot, [0.0, 0.0, 0.0]]
  else:
    xform = np.c_[rot @ stretch, [0.0, 0.0, 0.0]]
  mesh = b.add_mesh(pts, tris, normals=nrm, uvs=uv)
  b.add_instance("mesh", mesh=mesh, material=mat, xform=xform)
  plain = b.add_mesh(*_uv_sphere()[:2], normals=_uv_sphere()[2], uvs=_uv_sphere()[3])
  b.add_instance("mesh", mesh=plain, material=mat, xform=np.c_[np.eye(3), [3.0, 0.0, 0.0]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def _render_hdr(host: rw.Scene, res: int = 48) -> tuple[np.ndarray, rw.Renderer]:
  """HDR and the Renderer of a camera at z = 6 looking down at the origin."""
  r = render_host(host, (1.0, 0.0, 6.0), fovy=60.0, res=res, kinds=frozenset({"hdr", "seg"}))
  return required(r.outputs.hdr).numpy()[0, 0].reshape(res, res, 3), r


def test_mesh_stretch_renders_like_a_baked_copy():
  """P13 / BUG-743: a scaled mesh instance renders like a copy of its mesh baked by hand. Its
  uniform scale s = |det S|^(1/3) stays in its pose (s R, no copy: the kernels map rays and scale
  what they read from the mesh), the rest S / s of a non-uniform, sheared or mirrored stretch is
  baked (points, normals S^-T n, the winding reversed for a mirror so faces keep their side)."""
  for stretch, copies in (
    (np.diag(np.array([1.7, 1.7, 1.7])), 0),
    (np.diag(np.array([0.35, 0.35, 0.35])), 0),
    (np.diag(np.array([2.0, 0.5, 1.2])), 1),
    (np.array([[1.5, 0.3, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]]), 1),
    (np.diag(np.array([-1.3, 1.0, 0.7])), 1),
  ):
    host = _stretched_scene(stretch, baked=False)
    traced, r = _render_hdr(host)
    baked, rb = _render_hdr(_stretched_scene(stretch, baked=True))
    # Uniform: the scaled instance and the plain sphere read one mesh.
    assert r.scene.meshes.nmesh == 1 + copies and rb.scene.meshes.nmesh == 2
    assert host.report.counts.get("gpu.meshes_baked_stretch", 0) == copies
    s = abs(np.linalg.det(stretch)) ** (1.0 / 3.0)
    np.testing.assert_allclose(np.linalg.norm(r.scene.init_pose.rot[0], axis=0), [s, s, s], rtol=1e-5)
    seg, seg_baked = (required(x.outputs.seg).numpy()[0, 0].ravel() for x in (r, rb))
    assert np.mean(seg != seg_baked) < 0.01  # silhouettes up to float rounding
    same = seg == seg_baked
    np.testing.assert_allclose(traced.reshape(-1, 3)[same], baked.reshape(-1, 3)[same], atol=2e-3)
    assert traced.max() > 0.1


def test_scaled_copies_share_one_mesh():
  """BUG-743 repro (small): instances of one mesh at random uniform scales share it (no copy; the
  scale in each pose, s R; radii scaled); non-uniform stretches bake one copy per distinct shape
  S / s (two instances of one shape at different sizes share theirs), the points S p / s."""
  pts, tris, nrm, uv = _uv_sphere(4)
  b = rw.SceneBuilder()
  mesh = b.add_mesh(pts, tris, normals=nrm, uvs=uv)
  rng = np.random.default_rng(1)
  sizes = rng.uniform(0.2, 3.0, 400)
  b.add_instances("mesh", mesh=mesh, xforms=np.stack([np.c_[s * np.eye(3), [i, 0.0, 0.0]] for i, s in enumerate(sizes)]))
  shapes = np.linspace(0.5, 2.0, 20)
  stretch = [np.diag(np.array([t, 1.0, 2.0 - t / 2])) for t in shapes]
  xforms = np.stack([np.c_[k * m, [0.0, j, 0.0]] for j, m in enumerate(stretch) for k in (1.0, 2.5)])
  b.add_instances("mesh", mesh=mesh, xforms=xforms)
  host = b.build()
  scene = _upload(host)
  assert scene.meshes.nmesh == 1 + len(shapes)
  assert host.report.counts["gpu.meshes_baked_stretch"] == len(shapes)
  mesh_of = np.asarray(scene.tables["instance.mesh"].numpy()).reshape(-1)
  assert np.all(mesh_of[: len(sizes)] == mesh_of[0])
  rot = scene.init_pose.rot
  np.testing.assert_allclose(rot[: len(sizes)], sizes[:, None, None] * np.eye(3), rtol=1e-6, atol=1e-7)
  r0 = np.linalg.norm(pts, axis=1).max()
  np.testing.assert_allclose(scene.bound_radius[: len(sizes)], r0 * sizes, rtol=1e-5)
  for j, m in enumerate(stretch):
    a, c = len(sizes) + 2 * j, len(sizes) + 2 * j + 1
    assert mesh_of[a] == mesh_of[c]
    u = np.linalg.det(m) ** (1.0 / 3.0)
    np.testing.assert_allclose(scene.meshes.points[int(mesh_of[a])], pts @ (m / u), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(rot[c], axis=0), [2.5 * u] * 3, rtol=1e-5)
    radius = np.linalg.norm(pts @ m, axis=1).max()
    np.testing.assert_allclose(scene.bound_radius[[a, c]], [radius, 2.5 * radius], rtol=1e-5)


def test_mesh_copies_past_int32_addresses_raise_before_allocating(monkeypatch: pytest.MonkeyPatch):
  """Mesh copies that would take a host pool past its int32 addresses raise at once, naming the
  pool and its rows, before the pools are grown (a scattered scene of non-uniform instances
  failed after minutes and tens of GB with a bare OverflowError); within the limit the pools are
  filled in place to their exact size."""
  from raywarp.core.pipeline.upload import meshes
  from raywarp.scene.model.report import LoadReport

  pts, tris, nrm, uv = _uv_sphere(4)
  b = rw.SceneBuilder()
  mesh = b.add_mesh(pts, tris, normals=nrm, uvs=uv)
  shapes = [np.diag(np.array([t, 1.0, -1.0])) for t in np.linspace(0.5, 2.0, 8)]  # mirrored: tris copied too
  b.add_instances("mesh", mesh=mesh, xforms=np.stack([np.c_[m, [j, 0.0, 0.0]] for j, m in enumerate(shapes)]))
  host = b.build()
  paths = [""] * host.ninstance
  baked = meshes.bake_stretches(host, scaled=True, report=LoadReport(), paths=paths)
  assert len(baked.meshes.points) == len(pts) * (1 + len(shapes))
  assert len(baked.meshes.tris) == len(tris) * (1 + len(shapes))
  assert np.all(np.isfinite(baked.meshes.normals)) and len(baked.meshes.vert_adr) == 1 + len(shapes)
  limit = len(host.meshes.normals) * 4
  monkeypatch.setattr(meshes, "_MAX_POOL_ROWS", limit)
  with pytest.raises(ValueError, match=r"'normals': " + str(len(host.meshes.normals) * (1 + len(shapes)))):
    meshes.bake_stretches(host, scaled=True, report=LoadReport(), paths=paths)


class _RigidPoses(Binding[None]):
  """A binding serving rigid rotations (as a simulator's, e.g. Newton's): every stretch is baked
  (upload only: no slots)."""

  scaled_poses = False

  def slots(self, sim: None) -> Slots:
    """No per-call arrays."""
    del sim
    return {}


def test_rigid_pose_bindings_bake_every_stretch():
  """A binding whose rotations are rigid (`Binding.scaled_poses` False, Newton's) gets uniform
  scales baked into mesh copies (one per size), its poses rotations."""
  pts, tris, nrm, uv = _uv_sphere(4)
  b = rw.SceneBuilder()
  mesh = b.add_mesh(pts, tris, normals=nrm, uvs=uv)
  sizes = np.array([0.5, 2.0, 0.5])
  b.add_instances("mesh", mesh=mesh, xforms=np.stack([np.c_[s * np.eye(3), [i, 0.0, 0.0]] for i, s in enumerate(sizes)]))
  host = b.build()
  scene = rw.upload(host, DEVICE, binding=_RigidPoses())
  assert scene.meshes.nmesh == 2
  mesh_of = np.asarray(scene.tables["instance.mesh"].numpy()).reshape(-1)
  assert mesh_of[0] == mesh_of[2] != mesh_of[1]
  np.testing.assert_allclose(scene.init_pose.rot, np.tile(np.eye(3), (3, 1, 1)), atol=1e-6)
  np.testing.assert_allclose(scene.meshes.points[int(mesh_of[1])], 2.0 * pts, rtol=1e-6)


def _tinted(b: rw.SceneBuilder, color: TextureId, data: TextureId, normal: TextureId, k: float) -> rw.Material:
  """A material reading `color` / `data` / `normal` through factors that depend on `k`."""
  mat = rw.Material(
    base_color=(1.0, 1.0, 1.0),
    specular_roughness=1.0,
    emission_color=(k, 1.0, 0.5),
    emission_luminance=2.0,
    ext=rw.scene.RaywarpExt(occlusion_strength=k),
  )
  mat = mat.with_texture(Slot.BASE_COLOR, TextureRef(color, scale=(k, 0.8, 0.2, 1.0), bias=(0.0, 0.1 * k, 0.0, 0.0)))
  mat = mat.with_texture(Slot.EMISSION_COLOR, TextureRef(color))
  mat = mat.with_texture(Slot.OCCLUSION, TextureRef(data, channel=0))
  mat = mat.with_texture(Slot.SPECULAR_ROUGHNESS, TextureRef(data, channel=1, scale=(k, k, k, k)))
  mat = mat.with_texture(Slot.BASE_METALNESS, TextureRef(data, channel=2, bias=(0.1 * k,) * 4))
  scale, bias = rw.scene.model.normal_map.unpack((k, k))
  return mat.with_texture(Slot.GEOMETRY_NORMAL, TextureRef(normal, scale=scale, bias=bias))


def test_tinted_materials_share_one_core_texture():
  """B64: materials reading one image through different tints / scales / biases / strengths share
  its core textures; the factors are their texel maps (`material.tex_affine`).

  Before the fix every factor was baked into its own full-resolution copy (Isaac Warehouse: 198
  host textures -> 1790 core textures, ~71 GB).
  """
  rng = np.random.default_rng(1)
  b = rw.SceneBuilder()
  color = b.add_texture(rng.integers(0, 256, (8, 8, 4), dtype=np.uint8))
  data = b.add_texture(rng.integers(0, 256, (8, 8, 4), dtype=np.uint8), colorspace="linear")
  normal = b.add_texture(rng.integers(0, 256, (8, 8, 4), dtype=np.uint8), colorspace="linear")
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  ks = (1.0, 0.5, 0.25)
  for k in ks:
    b.add_instance("mesh", mesh=mesh, material=b.add_material(_tinted(b, color, data, normal, k)))
  scene = _upload(b.build())
  texid = scene.tables["material.texture"].numpy()[0]
  assert len(scene.textures) == 1 + 3  # placeholder + RGB (= EMISSIVE: the same image), ORM, NORMAL
  for role in (TexRole.RGB, TexRole.ORM, TexRole.NORMAL, TexRole.EMISSIVE):
    assert len(set(texid[:, int(role)])) == 1
  assert texid[0, int(TexRole.EMISSIVE)] == texid[0, int(TexRole.RGB)]
  affine = scene.tables["material.tex_affine"].numpy()[0]
  for i, k in enumerate(ks):
    np.testing.assert_allclose(affine[i, TexRole.RGB, :2], [[k, 0.8, 0.2, 1.0], [0.0, 0.1 * k, 0.0, 0.0]], rtol=1e-6)
    np.testing.assert_allclose(affine[i, TexRole.ORM, :2, :3], [[k, k, 1.0], [1.0 - k, 0.0, 0.1 * k]], rtol=1e-6)
    np.testing.assert_allclose(affine[i, TexRole.NORMAL, :2, :2], [[k, k], [0.5 * (1.0 - k)] * 2], rtol=1e-6)
    np.testing.assert_allclose(affine[i, TexRole.EMISSIVE, 0, :3], [k, 1.0, 0.5], rtol=1e-6)


def _textured_quad(image: np.ndarray, colorspace: str, scale: Color, bias: Color, factor: Color) -> rw.Scene:
  """A 2 x 2 m quad at the origin under a sun, its base color `factor` x (scale x image + bias)."""
  b = rw.SceneBuilder()
  tex = b.add_texture(image, colorspace=colorspace)
  mat = rw.Material(base_color=(factor[0], factor[1], factor[2]), specular_roughness=1.0)
  mat = mat.with_texture(Slot.BASE_COLOR, TextureRef(tex, scale=(*scale, 1.0), bias=(*bias, 0.0)))
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=b.add_material(mat))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def test_texel_map_renders_like_the_baked_texture():
  """B64: a tint applied where the texture is sampled renders like the same tint as the material's
  base color factor and like a texture with the tint baked in (the former recipe), up to
  float16."""
  from raywarp.core.post.grading.color import srgb_to_linear

  rng = np.random.default_rng(2)
  image = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
  tint, bias = (0.6, 0.9, 0.3), (0.05, 0.0, 0.1)
  shared, _ = _render_hdr(_textured_quad(image, "srgb", tint, bias, (1.0, 1.0, 1.0)))
  baked_texels = np.concatenate(
    [srgb_to_linear(image[..., :3] / 255.0) * np.array(tint) + np.array(bias), np.ones((8, 8, 1))], axis=2
  )
  baked, _ = _render_hdr(_textured_quad(baked_texels.astype(np.float32), "linear", (1.0,) * 3, (0.0,) * 3, (1.0, 1.0, 1.0)))
  assert shared.max() > 0.1
  np.testing.assert_allclose(shared, baked, rtol=2e-3, atol=1e-4)
  factor, _ = _render_hdr(_textured_quad(image, "srgb", (1.0,) * 3, (0.0,) * 3, tint))
  shared_no_bias, _ = _render_hdr(_textured_quad(image, "srgb", tint, (0.0,) * 3, (1.0, 1.0, 1.0)))
  np.testing.assert_allclose(shared_no_bias, factor, rtol=1e-5, atol=1e-6)


def test_texel_maps_are_randomised_per_world():
  """B64: `PerWorld(material=["tex_affine"])` tints a shared texture per world (no re-upload)."""
  rng = np.random.default_rng(3)
  image = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
  tint = (0.5, 0.25, 1.0)
  with wp.ScopedDevice(DEVICE):
    scene = rw.upload(_textured_quad(image, "srgb", (1.0,) * 3, (0.0,) * 3, (1.0, 1.0, 1.0)))
    s = rw.RenderSettings(
      gi=NO_GI,
      outputs=rw.Outputs.of("hdr"),
      environment=rw.settings.Environment(enabled=False),
      performance=rw.settings.Performance(traversal="software"),
    )
    per_world = rw.PerWorld(material=["tex_affine"])
    r = rw.Renderer(scene, nworld=2, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(32, 32), settings=s, per_world=per_world)
    r.camera_pose.assign(np.tile(np.array([1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 1, 1)))
    affine = r.per_world["material.tex_affine"].numpy()
    affine[1, 0, int(TexRole.RGB), 0, :3] = tint  # world 1: tinted
    r.per_world["material.tex_affine"].assign(affine)
    r.render()
    hdr = required(r.outputs.hdr).numpy()[:, 0].reshape(2, -1, 3)
  # World 0 renders the scene as uploaded; world 1 like the tint as the base color factor.
  plain, _ = _render_hdr(_textured_quad(image, "srgb", (1.0,) * 3, (0.0,) * 3, (1.0, 1.0, 1.0)), res=32)
  factor, _ = _render_hdr(_textured_quad(image, "srgb", (1.0,) * 3, (0.0,) * 3, tint), res=32)
  assert not np.allclose(hdr[0], hdr[1])
  np.testing.assert_allclose(hdr[0], plain.reshape(-1, 3), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(hdr[1], factor.reshape(-1, 3), rtol=1e-5, atol=1e-6)


def _albedo_of_quad(color: tuple[float, ...], base: tuple[float, ...], written: tuple[float, ...] | None = None) -> np.ndarray:
  """(nworld 2, 3) albedo at the centre of a quad (instance color `color`, material color
  `base`) seen from above; `written`: world 1's per-world instance color."""
  b = rw.SceneBuilder()
  mat = b.add_material(rw.Material(base_color=(base[0], base[1], base[2]), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=mat, color=color)
  with wp.ScopedDevice(DEVICE):
    scene = rw.upload(b.build())
    s = rw.RenderSettings(gi=NO_GI, outputs=rw.Outputs.of("albedo"), performance=rw.settings.Performance(traversal="software"))
    per_world = rw.PerWorld(material=["color"])
    r = rw.Renderer(scene, nworld=2, cameras=[rw.CameraSpec(fovy=40.0)], resolution=(8, 8), settings=s, per_world=per_world)
    r.camera_pose.assign(np.tile(np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 1, 1)))
    if written is not None:
      table = r.per_world["instance.color"]
      values = table.numpy()
      values[1, 0] = written
      table.assign(values)
    r.render()
    return required(r.outputs.albedo).numpy()[:, 0, 4, 4]


def test_a_grey_instance_color_tints_its_material():
  """An instance color of exactly (0.5, 0.5, 0.5, 1) is a 50 % grey tint like any other color,
  in the scene and written per world (`ColorRule.TINT`): before the fix that value was MuJoCo's
  "use the material color" sentinel in every adapter (white instead of grey, red instead of
  dark red)."""
  grey = (0.5, 0.5, 0.5, 1.0)
  albedo = _albedo_of_quad(grey, (1.0, 1.0, 1.0, 1.0))
  np.testing.assert_allclose(albedo, 0.5, rtol=1e-6)
  albedo = _albedo_of_quad((1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0), written=grey)
  np.testing.assert_allclose(albedo, [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], rtol=1e-6)


def test_texture_pool_entries_rebuild_textures_without_texels(tmp_path, monkeypatch):
  """A second upload of the same images reads its textures from their pool entries and cached
  levels (no texel read, conversion or hash): the same digests, layouts, sizes, red ranges and
  reports; another image content (equal shape) misses."""
  from raywarp.core.pipeline.platform import bake_cache
  from raywarp.core.pipeline.textures import pool as texture_pool
  from raywarp.scene.model.report import LoadReport

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  storage = formats.TextureStorage()

  def scene(value: int) -> rw.Scene:
    b = rw.SceneBuilder()
    rgba = np.full((8, 8, 4), value, np.uint8)
    rgba[..., 3] = (np.arange(64).reshape(8, 8) * 4).astype(np.uint8)
    base = b.add_texture(rgba, colorspace="g22_rec709")  # converted at upload: reported
    data = b.add_texture(np.full((8, 8, 3), value // 2, np.uint8), colorspace="data")
    material = b.add_material(
      rw.Material(
        textures=(
          (rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(base)),
          (rw.scene.Slot.SPECULAR_ROUGHNESS, rw.scene.TextureRef(data, channel=1)),
          (rw.scene.Slot.GEOMETRY_OPACITY, rw.scene.TextureRef(base, channel=3)),
        ),
        ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5),
      )
    )
    mesh = b.add_mesh(
      np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], np.float32),
      np.array([[0, 1, 2]], np.int32),
      uvs=np.array([[0, 0], [1, 0], [1, 1]], np.float32),
    )
    b.add_instance("mesh", mesh=mesh, material=material)
    return b.build()

  def pool(host: rw.Scene) -> tuple[rw.GpuScene[None], list[str | None]]:
    host = dataclasses.replace(host, report=LoadReport())
    gpu = rw.upload(host, device=wp.get_device(), options=rw.SceneOptions(textures=storage))
    digests = [bake_cache._digests.get(t) for t in gpu.textures]
    return gpu, digests

  first, first_digests = pool(scene(200))
  assert list(tmp_path.glob("textures/pool-*.json"))
  monkeypatch.setattr(texture_pool.TexturePool, "_texels", lambda *a: pytest.fail("texels read on a warm upload"))
  again, again_digests = pool(scene(200))
  assert again_digests == first_digests
  assert again.tex_layout.numpy().tolist() == first.tex_layout.numpy().tolist()
  assert again.tex_size.numpy().tolist() == first.tex_size.numpy().tolist()
  assert again.report.to_dict()["approximated"] == first.report.to_dict()["approximated"]
  with pytest.raises(pytest.fail.Exception, match="texels read"):
    pool(scene(100))
  # The pool entries name every input: another color space of the same image is another entry.
  monkeypatch.undo()
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  before = set(tmp_path.glob("textures/pool-*.json"))
  b = rw.SceneBuilder()
  texture = b.add_texture(np.full((8, 8, 4), 200, np.uint8), colorspace="srgb")
  mesh = b.add_mesh(np.eye(3, dtype=np.float32), np.array([[0, 1, 2]], np.int32), uvs=np.zeros((3, 2), np.float32))
  b.add_instance(
    "mesh",
    mesh=mesh,
    material=b.add_material(rw.Material(textures=((rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(texture)),))),
  )
  pool(b.build())
  assert set(tmp_path.glob("textures/pool-*.json")) - before


class _SimColors(Binding[None]):
  """A binding declaring MuJoCo's color rule and geom-id segmentation (no slots)."""

  color_rule = ColorRule.MUJOCO
  instance_segments = True

  def slots(self, sim: None) -> Slots:
    """No per-call slots."""
    return {}


def test_the_binding_declares_the_color_rule_and_the_segment_source():
  """`upload` takes the color rule and the segment source from its binding: a host scene keeps
  TINT and its segment column, a binding of MuJoCo's convention gets MUJOCO and instance ids."""
  builder = rw.SceneBuilder()
  builder.add_instance("sphere", size=(0.1, 0.1, 0.1), segment=7)
  host = builder.build()
  plain = rw.upload(host, DEVICE)
  assert plain.color_rule == ColorRule.TINT
  np.testing.assert_array_equal(required(plain.segment_id).numpy(), [[7]])
  sim = rw.upload(host, DEVICE, binding=_SimColors())
  assert sim.color_rule == ColorRule.MUJOCO and sim.segment_id is None
  assert sim.flex_segment == host.ninstance  # deformables number after the instances


def test_mesh_attribute_rows_of_the_wrong_width_raise():
  """(N, 3) UVs with a corner index are not reinterpreted as (1.5 N, 2) pairs: they raise."""
  b = rw.SceneBuilder()
  pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
  with pytest.raises(ValueError, match="rows of 2 values"):
    b.add_mesh(pts, [[0, 1, 2]], uvs=np.zeros((4, 3)), uv_idx=[[0, 1, 2]])


def test_an_empty_body_name_is_no_body():
  """`body=""` means no rigid body, as None (no body registered under an empty name)."""
  b = rw.SceneBuilder()
  b.add_instance("sphere", size=(0.1, 0.0, 0.0), body="")
  scene = b.build()
  assert scene.instances.body.tolist() == [-1] and scene.instances.bodies.tolist() == []


def test_texture_ref_channel_and_member_checks_raise_value_error():
  """A fractional channel / UV set and a non-integral enum value raise ValueError (not accepted,
  not a TypeError)."""
  from raywarp.scene.model import checks

  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((2, 2, 4), 255, np.uint8))
  # pyrefly: ignore[bad-argument-type]  (the invalid values under test)
  for ref in (rw.scene.TextureRef(tex, channel=0.5), rw.scene.TextureRef(tex, uv_set=0.5)):
    with pytest.raises(ValueError, match="must be an integer"):
      checks.texture_ref(ref, ntexture=1, where="t")
  with pytest.raises(ValueError, match="is not a"):
    checks.member("kind", 1.0, rw.scene.LightKind, where="t")  # pyrefly: ignore[bad-argument-type]  (under test)


def test_closed_panes_are_flagged_at_upload():
  """`CLOSED_PANE` marks a two-sided closed see-through shape (a blended box, a closed blended
  mesh), not an open one (a foliage quad), an opaque, a single-sided or a glass one: only closed
  panes are made single-sided (their exit faces culled) with both faces' coverage at the entry."""
  b = rw.SceneBuilder()

  def blended(transmission_weight: float = 0.0) -> MaterialId:
    ext = rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND)
    mat = rw.Material(base_color=(0.2, 0.4, 0.9), geometry_opacity=0.5, transmission_weight=transmission_weight, ext=ext)
    return b.add_material(mat)

  pane, glass = blended(), blended(transmission_weight=1.0)
  opaque = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  cube = b.add_mesh(
    np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]),
    [
      [0, 2, 1],
      [1, 2, 3],
      [4, 5, 6],
      [5, 7, 6],
      [0, 1, 4],
      [1, 5, 4],
      [2, 6, 3],
      [3, 6, 7],
      [0, 4, 2],
      [2, 4, 6],
      [1, 3, 5],
      [3, 7, 5],
    ],
  )
  quad = b.add_mesh(np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]), [[0, 1, 2], [0, 2, 3]])
  two = rw.scene.InstanceFlags.DEFAULT | rw.scene.InstanceFlags.DOUBLE_SIDED
  at = np.c_[np.eye(3), np.zeros(3)]
  b.add_instance("box", size=(0.3, 0.3, 0.01), xform=at, material=pane, flags=two)
  b.add_instance("mesh", mesh=cube, xform=at, material=pane, flags=two)
  b.add_instance("mesh", mesh=quad, xform=at, material=pane, flags=two)
  b.add_instance("box", size=(0.3, 0.3, 0.01), xform=at, material=opaque, flags=two)
  b.add_instance("box", size=(0.3, 0.3, 0.01), xform=at, material=pane)
  b.add_instance("box", size=(0.3, 0.3, 0.01), xform=at, material=glass, flags=two)
  flags = np.asarray(rw.upload(b.build(), "cpu").instance_twosided)
  assert list(flags & CLOSED_PANE) == [CLOSED_PANE, CLOSED_PANE, 0, 0, 0, 0], flags
  # A closed pane is single-sided (its exit face culled); the open and opaque ones stay two-sided.
  assert list(flags & TWO_SIDED) == [0, 0, TWO_SIDED, TWO_SIDED, 0, TWO_SIDED], flags


def test_nonfinite_environment_texels_are_zeroed_and_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
  """An environment map file decoded with NaN / infinite texels (a damaged or overflowed float EXR:
  array images are refused at ingest, files are not) uploads: those components carry no radiance
  (zero), the map is reported once, the stored radiance is finite (it used to fail the upload)."""
  from raywarp.core.pipeline.upload import environment as env_upload

  decode = env_upload._map_radiance  # the file decode, given damaged texels

  def damaged(*args: object) -> np.ndarray:
    rgb = decode(*args).copy()  # pyrefly: ignore[bad-argument-type]
    rgb[2, 3] = np.nan
    rgb[4, 5, 1] = np.inf
    return rgb

  monkeypatch.setattr(env_upload, "_map_radiance", damaged)
  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((8, 16, 3), 0.5, np.float32), colorspace="linear")
  b.add_environment(rw.scene.EnvironmentLight(texture=tex, path="/Dome"))
  b.add_instance("sphere", size=(1.0, 0.0, 0.0))
  scene = rw.upload(b.build(), device="cpu")
  assert scene.report.approximated["gpu.environment.nonfinite_radiance_zeroed"] == 1
  env = scene.environment.maps.maps[0]
  assert np.isfinite(env.sh).all() and np.isfinite(env.scale)


def test_vertex_alpha_capability_reads_only_rendered_vertex_colored_meshes():
  """`Capabilities.has_alpha` from vertex colors counts only the rendered instances whose material
  reads them: a translucent vertex color on a hidden mesh, or on a mesh whose material ignores
  vertex colors, compiles no alpha path (it did: any mesh's alpha below 1 turned it on)."""
  quad = ([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]])
  b = rw.SceneBuilder()
  vc = b.add_material(rw.Material(ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.VERTEX_COLOR)))
  plain = b.add_material(rw.Material(name="plain"))
  opaque = b.add_mesh(*quad, colors=[[0.2, 0.4, 0.6, 1.0]] * 4)
  translucent = b.add_mesh(*quad, colors=[[0.2, 0.4, 0.6, 0.5]] * 4)
  b.add_instance("mesh", mesh=opaque, material=vc)
  hidden = int(rw.scene.InstanceFlags.DEFAULT & ~rw.scene.InstanceFlags.VISIBLE)
  b.add_instance("mesh", mesh=translucent, material=vc, flags=hidden)
  b.add_instance("mesh", mesh=translucent, material=plain)
  caps = rw.upload(b.build(), device=DEVICE).capabilities
  assert caps.vertex_colors and not caps.has_alpha
  # A material edit can make the rendered plain instance read its translucent colors.
  assert caps.prepared_materials is not None and caps.prepared_materials.has_alpha


@pytest.mark.parametrize("mirror", [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)])
def test_a_mirrored_camera_keeps_its_view_direction(mirror):
  """A camera frame with a reflection (a negative scale up its hierarchy) looks along its own -Z
  with its own +Y up; the nearest rotation of the frame (`polar`) flipped the view for a mirrored
  X or Y axis (the camera looked backwards)."""
  from raywarp.core.pipeline.upload import environment

  b = rw.SceneBuilder()
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=b.add_material(rw.Material()))
  frame = np.c_[np.diag(mirror), [0.0, 0.0, 3.0]].astype(np.float32)
  b.add_camera("cam", rw.scene.Camera(xform=frame))
  host = b.build()
  rot = environment.cameras(host, host.report).rot[0]
  np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-6)
  np.testing.assert_allclose(-rot[:, 2], -frame[:, 2], atol=1e-6)  # the view direction
  np.testing.assert_allclose(rot[:, 1], frame[:, 1], atol=1e-6)  # up
  assert host.report.approximated["gpu.camera.mirrored(image not mirrored)"] == 1
