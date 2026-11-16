"""B52: emissive fixtures that are not one flat rectangle (the Isaac Hospital's ceiling lights).

The Hospital's 104 troffers emit from two fluorescent tubes inside an open-bottom housing (one
mesh, one material) and its 49 downlights from a round diffuser: before B52 neither became a
light, so the corridors were lit by the dome through the windows only.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

import raywarp as rw
from raywarp.core.pipeline.upload import lights as host_lights
from raywarp.scene.lighting import emissive
from raywarp.scene.lighting import emitter_fixture as fixtures
from raywarp.scene.lighting import emitter_shapes as shapes
from raywarp.scene.model.types import UV_SET_WORLD, LightKind, MaterialId, Slot, TextureRef


def _panel_lights(host: rw.Scene) -> host_lights.DerivedLights:
  """The derived light table of `host` plus its emissive panel lights."""
  lit = emissive.add_panel_lights(host)
  return host_lights.derive_lights(lit.lights)


def _tube(centre, radius, length, sides=8):
  """(points, tris) of an open octagonal tube along +y."""
  ang = 2.0 * np.pi * (np.arange(sides) + 0.5) / sides
  ring = np.stack([radius * np.cos(ang), np.zeros(sides), radius * np.sin(ang)], 1)
  half = np.array([0.0, 0.5 * length, 0.0])
  pts = np.concatenate([ring - half, ring + half]) + centre
  tris = []
  for k in range(sides):
    a, b = k, (k + 1) % sides
    tris += [[a, b, b + sides], [a, b + sides, a + sides]]
  return pts, np.array(tris)


def _housing(hx, hy, h):
  """(points, tris) of a box open at its bottom (z = 0): a troffer's housing."""
  pts = np.array([[x, y, z] for z in (0.0, h) for y in (-hy, hy) for x in (-hx, hx)])
  quads = [[4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]  # top, 4 sides
  return pts, np.array([t for q in quads for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])])


def _troffer():
  """(points, tris) of one mesh: a 0.6 x 1 x 0.1 m housing with a tube (r 2 cm, 0.8 m) inside."""
  hp, ht = _housing(0.3, 0.5, 0.1)
  tp, tt = _tube((0.0, 0.0, 0.05), 0.02, 0.8)
  return np.concatenate([hp, tp]).astype(np.float32), np.concatenate([ht, tt + len(hp)])


def _disk(radius, sides=24):
  """(points, tris) of a flat fan facing -z."""
  ang = 2.0 * np.pi * np.arange(sides) / sides
  pts = np.concatenate([[[0.0, 0.0, 0.0]], np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(sides)], 1)])
  return pts.astype(np.float32), np.array([[0, 1 + (k + 1) % sides, 1 + k] for k in range(sides)])


def test_troffer_tube_lights_through_its_open_side():
  b = rw.SceneBuilder()
  glow = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=500.0, emission_color=(1, 1, 1))
  )
  fixture = b.add_instance("mesh", mesh=b.add_mesh(*_troffer()), xform=np.c_[np.eye(3), [1, 2, 2.9]], material=glow)
  lit = _panel_lights(b.build())
  # One light: the tube's silhouette (0.8 x 0.04 m) under the tube, emitting down (the housing
  # blocks every other side), at the tube's radiance (an untextured emission).
  assert lit.instance.tolist() == [fixture]
  assert lit.kind[0] == LightKind.RECT
  np.testing.assert_allclose(sorted([lit.width[0], lit.height[0]]), [0.04, 0.8], rtol=1e-3)
  np.testing.assert_allclose(-lit.xform[0][:, 2], [0, 0, -1], atol=1e-6)
  np.testing.assert_allclose(lit.xform[0][:, 3], [1, 2, 2.95 - 0.02 - 2e-3], atol=1e-4)
  np.testing.assert_allclose(lit.radiance[0], [500, 500, 500])


def test_round_diffuser_is_a_disk_light():
  b = rw.SceneBuilder()
  glow = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=100.0, emission_color=(1, 1, 1))
  )
  down = b.add_instance("mesh", mesh=b.add_mesh(*_disk(0.1)), xform=np.c_[np.eye(3), [0, 0, 2.9]], material=glow)
  lit = _panel_lights(b.build())
  assert lit.instance.tolist() == [down]
  assert lit.kind[0] == LightKind.DISK
  np.testing.assert_allclose(lit.radius[0], 0.1, rtol=1e-5)
  np.testing.assert_allclose(-lit.xform[0][:, 2], [0, 0, -1], atol=1e-6)
  np.testing.assert_allclose(lit.xform[0][:, 3], [0, 0, 2.9 - 2e-3], atol=1e-6)
  # Point-light equivalent: radiance x pi r^2.
  np.testing.assert_allclose(lit.intensity[0], 100.0 * math.pi * 0.01, rtol=1e-5)


def test_shapes_reject_bars_and_split_parts():
  # A square bar is no tube (its side faces' centroids sit at 0.71 x the corner radius).
  pts, tris = _tube((0, 0, 0), 0.02, 0.8, sides=4)
  assert shapes.mesh_tube(pts, tris, 0.9) is None
  tube = shapes.mesh_tube(*_tube((0, 0, 0), 0.02, 0.8), 0.9)
  assert tube is not None
  np.testing.assert_allclose([tube.radius, tube.length], [0.02, 0.8], rtol=1e-6)
  # Two meshes in one triangle set (split vertices at a seam are welded).
  points, tris = _troffer()
  parts = shapes.components(points.astype(np.float64), tris)
  assert sorted(len(p) for p in parts) == [10, 16]


def test_box_panel_thickness_ratio_matches_meshes():
  """A box is a panel when thinner than `max_thickness` (0.02) x its largest side, as meshes are;
  a thicker one gets a light per face carrying >= MIN_PART_POWER of its power (its two large
  faces here, not its 0.03 m edges)."""
  b = rw.SceneBuilder()
  glow = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=500.0, emission_color=(1, 1, 1))
  )
  thin = b.add_instance("box", size=(0.5, 0.5, 0.004), material=glow)  # thickness 0.008 / side 1: 0.008
  thick = b.add_instance("box", size=(0.5, 0.5, 0.015), xform=np.c_[np.eye(3), [3, 0, 0]], material=glow)  # 0.03
  lit = _panel_lights(b.build())
  assert lit.instance.tolist() == [thin, thin, thick, thick]  # back to back; the thick box's faces


def test_zero_thickness_box_and_degenerate_plane_load():
  """A box of zero thickness (size 0 across it) is a two-sided panel, a plane flattened to a line
  by its transform none: neither stops the load with a degenerate light frame."""
  b = rw.SceneBuilder()
  glow = _glow(b)
  flat = b.add_instance("box", size=(0.5, 0.25, 0.0), material=glow)
  line = np.array([[1.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # y scaled to 0
  b.add_instance("plane", size=(0.5, 0.5, 0.0), xform=line, material=glow)
  lit = _panel_lights(b.build())
  assert lit.instance.tolist() == [flat, flat]  # two-sided: back to back along +-Z
  np.testing.assert_allclose(np.abs(np.asarray(lit.xform)[:, :, 2]), [[0.0, 0.0, 1.0]] * 2, atol=1e-6)
  np.testing.assert_allclose(sorted(np.asarray(lit.width).tolist()), [1.0, 1.0])


def _glow(b: rw.SceneBuilder) -> MaterialId:
  """An untextured 500 nit emissive material."""
  return b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=500.0, emission_color=(1, 1, 1))
  )


def _rotation(angle: float) -> np.ndarray[[3, 3], np.dtype[np.float64]]:
  """A rotation about (1, 2, 3) by `angle` [rad] (Rodrigues)."""
  k = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
  kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


def test_fits_in_the_mesh_frame_equal_world_fits():
  """A fixture placed by a similarity (scale x rotation + translation) is fitted once per (mesh,
  material, scale) in its own frame and placed: the lights equal those of the same geometry
  fitted in world space; a non-uniform scale is fitted in world space."""
  points, tris = _troffer()
  disk_points, disk_tris = _disk(0.1)
  points, disk_points = 100.0 * points, 100.0 * disk_points  # authored in cm, placed in m
  lin = 0.01 * _rotation(0.7)
  moved = []
  local = rw.SceneBuilder()
  glow = _glow(local)
  troffer, disk = local.add_mesh(points, tris), local.add_mesh(disk_points, disk_tris)
  world = rw.SceneBuilder()
  wglow = _glow(world)
  for k, (mesh, (p, t)) in enumerate([(troffer, (points, tris)), (troffer, (points, tris)), (disk, (disk_points, disk_tris))]):
    offset = np.array([k, 2.0 * k, 3.0])
    local.add_instance("mesh", mesh=mesh, xform=np.c_[lin, offset], material=glow)
    moved.append(world.add_mesh((p.astype(np.float64) @ lin.T + offset).astype(np.float32), t))
    world.add_instance("mesh", mesh=moved[-1], xform=np.c_[np.eye(3), np.zeros(3)], material=wglow)
  a, b = _panel_lights(local.build()), _panel_lights(world.build())
  assert a.kind.tolist() == b.kind.tolist() == [LightKind.RECT, LightKind.RECT, LightKind.DISK]
  # The disk: same centre and emitting axis (its in-plane axes are free: a disk is round).
  np.testing.assert_allclose(a.xform[2][:, 2:], b.xform[2][:, 2:], atol=2e-6)
  np.testing.assert_allclose(a.radiance, b.radiance, rtol=1e-5)
  np.testing.assert_allclose([a.width, a.height, a.radius], [b.width, b.height, b.radius], rtol=1e-4, atol=1e-7)
  # The tubes' open side is sampled in the fixture's frame: exactly its open bottom (the world
  # fit's 16 sampled directions, phased by the world frame, were 4.3 degrees off).
  down = -lin[:, 2] / np.linalg.norm(lin[:, 2], keepdims=True)
  np.testing.assert_allclose(-a.xform[0][:, 2], down, atol=1e-5)
  np.testing.assert_allclose(-a.xform[1][:, 2], down, atol=1e-5)
  assert np.degrees(np.arccos(float(-b.xform[0][:, 2] @ down))) > 1.0
  assert shapes.similarity(np.c_[lin, np.zeros(3)]) == 0.01
  assert shapes.similarity(np.c_[np.diag(np.array([1.0, 2.0, 1.0])), np.zeros(3)]) is None
  assert shapes.similarity(np.c_[np.diag(np.array([1.0, 1.0, -1.0])), np.zeros(3)]) is None  # a reflection


def test_hull_prefilter_keeps_the_minimum_rectangle(monkeypatch):
  rng = np.random.default_rng(3)
  for n in (5, 40, 400):
    pts = rng.normal(size=(n, 2)) @ np.array([[3.0, 1.0], [0.0, 0.5]])
    filtered = shapes.min_area_rect(pts)
    with monkeypatch.context() as patch:
      patch.setattr(shapes, "_outer", lambda p: p)
      reference = shapes.min_area_rect(pts)
    for x, y in zip(filtered, reference, strict=True):
      np.testing.assert_array_equal(x, y)


def test_upload_skips_the_fit_when_emissive_lights_are_off():
  b = rw.SceneBuilder()
  b.add_instance("box", size=(0.5, 0.5, 0.004), material=_glow(b))
  host = b.build()
  assert rw.upload(host, "cpu").nlight == 2
  assert rw.upload(host, "cpu", options=rw.SceneOptions(emissive_lights=False)).nlight == 0


def _quad(cx: float, cy: float, half: float, z: float, *, double: bool) -> tuple[np.ndarray, np.ndarray]:
  """(points, tris) of a square in the plane z facing -z (both ways when `double`)."""
  pts = np.array([[cx - half, cy - half, z], [cx + half, cy - half, z], [cx + half, cy + half, z], [cx - half, cy + half, z]])
  tris = [[0, 2, 1], [0, 3, 2]]  # facing -z
  if double:
    tris += [[0, 1, 2], [0, 2, 3]]
  return pts, np.array(tris)


def _mesh(*parts: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """One triangle set of several (points, tris)."""
  points, tris, n = [], [], 0
  for p, t in parts:
    points.append(p)
    tris.append(t + n)
    n += len(p)
  return np.concatenate(points).astype(np.float32), np.concatenate(tris)


def test_back_light_into_the_own_housing_is_dropped():
  """A two-sided diffuser inside its fixture's open-bottom housing (one emissive mesh) lights only
  down: its back faces the housing's top. Without the housing it lights both ways."""
  diffuser = _quad(0.0, 0.0, 0.25, 0.01, double=True)
  for housing, emits in ((True, [[0, 0, -1]]), (False, [[0, 0, -1], [0, 0, 1]])):
    b = rw.SceneBuilder()
    mesh = b.add_mesh(*(_mesh(_housing(0.3, 0.3, 0.1), diffuser) if housing else _mesh(diffuser)))
    b.add_instance("mesh", mesh=mesh, xform=np.c_[np.eye(3), [0, 0, 2.9]], material=_glow(b))
    lit = _panel_lights(b.build())
    np.testing.assert_allclose(sorted((-x[:, 2]).tolist() for x in lit.xform), sorted(emits), atol=1e-6)
    np.testing.assert_allclose(lit.radiance, 500.0)


def test_coplanar_parts_that_fill_a_rectangle_are_one_light():
  """Four 0.4 m squares 0.1 m apart (a diffuser split by frame bars: the faces cover 79 % of the
  0.9 m square, below the one-rectangle fill) become one 0.9 m light emitting their power; two
  squares 1 m apart stay two lights."""
  grid = _mesh(*(_quad(x, y, 0.2, 0.0, double=False) for x in (-0.25, 0.25) for y in (-0.25, 0.25)))
  apart = _mesh(_quad(-0.7, 0.0, 0.2, 0.0, double=False), _quad(0.7, 0.0, 0.2, 0.0, double=False))
  for (points, tris), n in ((grid, 1), (apart, 2)):
    b = rw.SceneBuilder()
    b.add_instance("mesh", mesh=b.add_mesh(points, tris), xform=np.c_[np.eye(3), [0, 0, 2.9]], material=_glow(b))
    lit = _panel_lights(b.build())
    assert len(lit.kind) == n and set(lit.kind.tolist()) == {LightKind.RECT}
    # The emitted power is the faces' (500 nit x their area), whatever the light count.
    power = (lit.radiance[:, 0] * lit.width * lit.height).sum()
    np.testing.assert_allclose(power, 500.0 * (4 if n == 1 else 2) * 0.16, rtol=1e-6)
    np.testing.assert_allclose(-lit.xform[:, :, 2], np.tile([0.0, 0.0, -1.0], (n, 1)), atol=1e-6)


def _square(x: float, half: float) -> shapes.Panel:
  """A one-sided square panel at (x, 0, 0) facing +z."""
  z, ax = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
  return shapes.Panel(
    centre=np.array([x, 0.0, 0.0]), normal=z, axis=ax, half=np.array([half, half]), half_thickness=0.0, side=1
  )


def test_repeated_merges_count_the_source_area_not_the_merged_rectangles():
  """Eight 1 m squares 0.4 m apart in a row: neighbours fill 83 % of their rectangle, all eight
  only 74 % (8 / 10.8 m^2), below MERGE_FILL. Counting merged rectangles as cover drifted up and
  made them one light; every merged light now holds at least MERGE_FILL of source area, and the
  power is kept."""
  items: list[tuple[shapes.Emitter, shapes.Vec3]] = [(_square(1.4 * k, 0.5), np.ones(3)) for k in range(8)]
  out = fixtures.merge_coplanar(items)
  assert 1 < len(out) < 8
  for panel, scale in out:  # scale x area = the source area it emits (unit radiance)
    assert scale[0] >= fixtures.MERGE_FILL - 1e-12, (shapes.area(panel), scale)
  np.testing.assert_allclose(sum(float(k[0]) * shapes.area(p) for p, k in out), 8.0, rtol=1e-12)


def test_parallel_ray_culling_and_chunks_match_the_full_test(monkeypatch):
  """`_hits` (bounding-box culling, chunks of pairs) equals the unculled test on 200 small random
  triangles and parallel rays (some starting among them), also with chunks of a few pairs."""
  rng = np.random.default_rng(8)
  points = (rng.uniform(-1.0, 1.0, (200, 1, 3)) + rng.normal(0.0, 0.1, (200, 3, 3))).reshape(-1, 3)
  tris = np.arange(600).astype(np.int64).reshape(200, 3)
  for k in range(6):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d, keepdims=True)
    origins = rng.uniform(-0.5, 0.5, (40, 3)) - d * (0.5 * (k % 2))
    want = fixtures._hits_of(points, tris, origins, d)
    assert 0 < want.sum() < len(want)
    np.testing.assert_array_equal(fixtures._hits(points, tris, origins, direction=d), want)
    with monkeypatch.context() as patch:
      patch.setattr(fixtures, "_HIT_PAIRS", 7)
      np.testing.assert_array_equal(fixtures._hits(points, tris, origins, direction=d), want)


def test_open_sides_of_a_million_triangle_fixture_stays_small():
  """A two-sided diffuser under a 1M-triangle housing ceiling (a 708 x 708 quad grid, 4 x 4 m)
  lights only down; the side test culls the triangles off its footprint, so it allocates far less
  than the 16 rays x 1M triangles x 3 float64 (384 MB per array) of an unculled test."""
  import tracemalloc

  n = 708
  g = np.linspace(-2.0, 2.0, n + 1)
  x, y = np.meshgrid(g, g)
  points = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.1)], 1)
  i = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None]).ravel()
  tris = np.concatenate([np.stack([i, i + 1, i + n + 2], 1), np.stack([i, i + n + 2, i + n + 1], 1)])
  assert len(tris) >= 1_000_000
  diffuser = dataclasses.replace(_square(0.0, 0.25), side=0)
  tracemalloc.start()
  try:
    lit = fixtures.open_sides(diffuser, points, tris, offset=1e-3)
    peak = tracemalloc.get_traced_memory()[1]
  finally:
    tracemalloc.stop()
  assert lit.side == -1
  assert peak < 200e6, peak


def test_a_repeat_fit_is_remembered_and_follows_its_inputs(tmp_path, monkeypatch):
  """The fit of a scene is remembered on disk: a repeat fits nothing and returns the same lights
  and report entries; moving the fixture or changing its emission fits again."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))

  def scene(z: float, luminance: float = 500.0) -> rw.Scene:
    b = rw.SceneBuilder()
    glow = b.add_material(rw.Material(base_color=(1, 1, 1), emission_luminance=luminance, emission_color=(1, 1, 1)))
    mesh = b.add_mesh(*_mesh(_housing(0.3, 0.3, 0.1), _quad(0.0, 0.0, 0.25, 0.01, double=True)))
    b.add_instance("mesh", mesh=mesh, xform=np.c_[np.eye(3), [0, 0, z]], material=glow)
    return b.build()

  first = emissive.add_panel_lights(scene(2.9))
  assert len(list((tmp_path / "emissive").glob("panels-*.npz"))) == 1
  fitted = emissive._panel_lights
  monkeypatch.setattr(emissive, "_panel_lights", lambda *a: (_ for _ in ()).throw(AssertionError("fitted again")))
  again = emissive.add_panel_lights(scene(2.9))
  for name, column in first.lights.arrays().items():
    np.testing.assert_array_equal(again.lights.arrays()[name], column, err_msg=name)
  assert again.report.counts == first.report.counts and again.report.counts["lights.emissive_panels"] == 1
  monkeypatch.setattr(emissive, "_panel_lights", fitted)
  moved, brighter = emissive.add_panel_lights(scene(2.5)), emissive.add_panel_lights(scene(2.9, 800.0))
  assert len(list((tmp_path / "emissive").glob("panels-*.npz"))) == 3
  assert moved.lights.xform[0][2, 3] < first.lights.xform[0][2, 3]
  np.testing.assert_allclose(
    brighter.lights.color * brighter.lights.intensity[:, None], 1.6 * first.lights.color * first.lights.intensity[:, None]
  )


def test_textured_two_sided_emitter_matches_the_untextured_one():
  """A double-layer 0.5 m diffuser (faces both ways: two lights back to back) emits the same
  radiance per side whether its emission is a constant (untextured) or a constant white texture:
  the faces' power spreads over both emitting sides, not over one side's area (2x radiance)."""
  points, tris = _quad(0.0, 0.0, 0.25, 0.0, double=True)
  uvs = (points[:, :2] + 0.25) / 0.5
  radiance = []
  for textured in (False, True):
    b = rw.SceneBuilder()
    material = rw.Material(
      base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=500.0, emission_color=(1, 1, 1)
    )
    if textured:
      white = b.add_texture(np.full((1, 1, 4), 255, np.uint8))
      material = material.with_texture(Slot.EMISSION_COLOR, TextureRef(white))
    mesh = b.add_mesh(points.astype(np.float64), tris, uvs=uvs.astype(np.float64))
    b.add_instance("mesh", mesh=mesh, xform=np.c_[np.eye(3), [0, 0, 2.9]], material=b.add_material(material))
    lit = _panel_lights(b.build())
    assert len(lit.kind) == 2  # one light per side
    radiance.append(lit.radiance[:, 0])
  np.testing.assert_allclose(radiance[1], radiance[0], rtol=1e-5)
  np.testing.assert_allclose(radiance[0], [500.0, 500.0], rtol=1e-6)


def test_world_projected_emission_texture_is_factor_only():
  """A world-projected emission texture (UV_SET_WORLD) is not read through the mesh's own UVs:
  the light uses the emission factor alone, reported as an approximation."""
  points, tris = _quad(0.0, 0.0, 0.25, 0.0, double=False)
  uvs = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]])  # inside the texture's dark half
  b = rw.SceneBuilder()
  half = np.zeros((2, 2, 4), np.uint8)
  half[:, 1] = 255  # dark left column, white right column
  ref = TextureRef(b.add_texture(half), uv_set=UV_SET_WORLD)
  material = rw.Material(
    base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=500.0, emission_color=(1, 1, 1)
  ).with_texture(Slot.EMISSION_COLOR, ref)
  mesh = b.add_mesh(points.astype(np.float64), tris, uvs=uvs)
  b.add_instance("mesh", mesh=mesh, xform=np.c_[np.eye(3), [0, 0, 2.9]], material=b.add_material(material))
  host = b.build()
  lit = emissive.add_panel_lights(host)
  derived = host_lights.derive_lights(lit.lights)
  np.testing.assert_allclose(derived.radiance[:, 0], [500.0], rtol=1e-6)
  assert lit.report.approximated["emissive_light.texture(factor only)"] == 1
