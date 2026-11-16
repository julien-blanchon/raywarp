"""Real-time renderer tests: analytic radiometry and parity with MuJoCo Warp."""

import mujoco_warp as mjw
import numpy as np
import pytest
from conftest import required, srgb
from fixtures.display_helpers import apply_display_transform

import raywarp as rw
from raywarp._base.types import ToneMapper
from raywarp.core.pipeline.platform import tuning
from raywarp.core.post.grading import color
from raywarp.settings.fields import OutputKind, exposure_from_ev100

_HDR = {"outputs": rw.Outputs.of("rgb", "hdr")}

# A camera looking straight down at a plane through a narrow frustum: every pixel sees the plane
# at (nearly) normal incidence.
_TOP_CAMERA = '<camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>'
_NO_HEADLIGHT = '<visual><headlight active="0"/></visual>'


def _plane_xml(material: str, light: str) -> str:
  return f"""
    <mujoco>
      {_NO_HEADLIGHT}
      <asset><material name="mat" {material}/></asset>
      <worldbody>
        {light}
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """


def test_lambertian_plane_point_light(scene):
  """Radiance of a white Lambertian plane under a point light: (albedo / π) · I / d²."""
  intensity, height, albedo = 3.0, 1.5, 0.8
  xml = _plane_xml(
    f'rgba="{albedo} {albedo} {albedo} 1" metallic="0" roughness="1" specular="0"',
    f'<light type="point" pos="0 0 {height}" intensity="{intensity}" diffuse="1 1 1" range="1000" castshadow="false"/>',
  )
  s = scene(xml, **_HDR, post=rw.settings.Post(tone_mapper="clamp")).render()
  hdr = s.hdr()[0]
  center = hdr[16, 16]
  expected = albedo / np.pi * intensity / height**2
  np.testing.assert_allclose(center, expected, rtol=2e-3)


def test_one_candela_point_light_is_exposure_not_range(scene):
  """B57 (not a renderer bug): a 1 cd point light 1.5 m over a white plane gives its exact
  radiance (1 / pi / 1.5^2 nits) at any exposure. At the default EV100 (~15, daylight) that is
  ~4e-6 after exposure: 8-bit black under any tone mapper (PBR Neutral's toe, 6.25 x^2 for
  greys below 0.08, darkens it further); at EV100 0 it is a visible grey."""
  xml = _plane_xml(
    'rgba="1 1 1 1" metallic="0" roughness="1" specular="0"',
    '<light type="point" pos="0 0 1.5" intensity="1" diffuse="1 1 1" castshadow="false"/>',
  )
  for ev100, black in ((14.97, True), (0.0, False)):
    s = scene(xml, **_HDR, exposure=rw.settings.Exposure(mode="manual", ev100=ev100)).render()
    np.testing.assert_allclose(s.hdr()[0, 16, 16], 1.0 / np.pi / 1.5**2, rtol=3e-3)
    assert (s.rgb()[0, 16, 16].max() == 0.0) == black


def test_hdr_holds_physical_radiance(scene):
  """Sunlight-level radiance (beyond float16 range) is stored exactly in the HDR output."""
  illuminance = 1.0e6
  xml = _plane_xml(
    'rgba="1 1 1 1" metallic="0" roughness="1" specular="0"',
    f'<light type="directional" dir="0 0 -1" intensity="{illuminance}" diffuse="1 1 1" castshadow="false"/>',
  )
  hdr = scene(xml, **_HDR).render().hdr()[0]
  np.testing.assert_allclose(hdr[16, 16], illuminance / np.pi, rtol=2e-3)


def test_fallback_directional_matches_diffuse_color(scene):
  """Unset intensity: a white Lambertian plane facing the light shows the light's diffuse color."""
  xml = _plane_xml(
    'rgba="1 1 1 1" metallic="0" roughness="1" specular="0"',
    '<light type="directional" dir="0 0 -1" diffuse="0.5 0.25 0.125" castshadow="false"/>',
  )
  s = scene(xml, post=rw.settings.Post(tone_mapper="clamp")).render()
  np.testing.assert_allclose(s.rgb()[0, 16, 16], srgb(np.array([0.5, 0.25, 0.125])), atol=2.5 / 255)


def test_default_lights_show_their_diffuse_color(scene):
  """MuJoCo GL-model lights (intensity 0) are converted to show their diffuse color on white."""
  xml = _plane_xml(
    'rgba="0.7 0.7 0.7 1" metallic="0" roughness="1" specular="0"',
    '<light type="directional" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>',
  )
  s = scene(xml, post=rw.settings.Post(tone_mapper="clamp")).render()
  np.testing.assert_allclose(s.rgb()[0, 16, 16], srgb(np.full(3, 0.7 * 0.6)), atol=2.5 / 255)


def test_smooth_metal_reflects_ambient_energy(scene):
  """A smooth metal under constant ambient radiance reflects (almost) all of it."""
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.5 0.5 0.5" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="metal" rgba="1 1 1 1" metallic="1" roughness="0"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="metal"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, **_HDR, post=rw.settings.Post(tone_mapper="clamp")).render()
  exposure = s.exposure
  np.testing.assert_allclose(s.hdr()[0, 16, 16] * exposure, 0.5, rtol=0.01)


def test_rough_metal_conserves_energy_with_compensation(scene):
  """Multiscatter compensation keeps rough white metal near the furnace value."""
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.5 0.5 0.5" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="metal" rgba="1 1 1 1" metallic="1" roughness="1"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="metal"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, **_HDR, post=rw.settings.Post(tone_mapper="clamp")).render()
  exposure = s.exposure
  np.testing.assert_allclose(s.hdr()[0, 16, 16] * exposure, 0.5, rtol=0.02)


def test_shadow_blocks_direct_light(scene):
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" roughness="1" specular="0"/></asset>
      <worldbody>
        <light type="directional" dir="0 0 -1" castshadow="true"/>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <geom type="box" pos="0 0 1" size="0.3 0.3 0.01" material="mat"/>
        <camera name="side" pos="0 -1 0.5" xyaxes="1 0 0 0 0.447 0.894" fovy="30"/>
      </worldbody>
    </mujoco>
  """
  lit = scene(xml, shadows=rw.settings.Shadows(enabled=False)).render().rgb()[0]
  shadowed = scene(xml, shadows=rw.settings.Shadows(enabled=True)).render().rgb()[0]
  # The box shadows the floor point under it; shadows only ever darken the image.
  assert np.all(shadowed <= lit + 1e-6)
  assert (lit - shadowed).max() > 0.5


@pytest.mark.parametrize("render_rgb", [True, False])
@pytest.mark.parametrize("ordered_traversal", [True, False])
def test_depth_and_segmentation_match_mujoco_warp(scene, render_rgb, ordered_traversal):
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="sphere" pos="0 0 0.4" size="0.3"/>
        <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
        <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2"/>
        <geom type="mesh" mesh="ico" pos="0.1 -0.5 0.35" euler="20 30 10"/>
        <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
      <asset>
        <mesh name="ico" scale="0.2 0.2 0.2"
          vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                  1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
      </asset>
    </mujoco>
  """
  outputs: set[OutputKind] = {"rgb", "depth", "seg"} if render_rgb else {"depth", "seg"}
  with tuning.override(ordered_traversal=ordered_traversal):
    s = scene(xml, cam_res=(48, 40), outputs=rw.Outputs.of(*outputs))
    s.render()
  ref = mjw.create_render_context(s.mjm, nworld=1, cam_res=(48, 40), render_rgb=True, render_depth=True, render_seg=True)
  mjw.refit_bvh(s.m, s.d, ref)
  mjw.render(s.m, s.d, ref)

  # seg is the int32 geom id (-1 on a miss); MuJoCo Warp writes (geom id, mjOBJ_GEOM) pairs.
  np.testing.assert_array_equal(s.seg().reshape(1, -1), ref.seg_data.numpy()[..., 0])
  # Rays are generated in-kernel (fast math) rather than precomputed, hence the looser tolerance.
  np.testing.assert_allclose(s.depth().reshape(1, -1), ref.depth_data.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("nworld", [1, 2])
def test_per_world_metallic(scene, nworld):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.3 0.3 0.3" diffuse="0.5 0.5 0.5"/></visual>
      <asset><material name="mat" rgba="0.8 0.2 0.2 1" roughness="0.4"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="mat"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="25"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, nworld=nworld, per_world=rw.PerWorld(material=["base_metalness"]))
  per_world = np.zeros((nworld, s.mjm.nmat), dtype=np.float32)
  per_world[nworld - 1] = 1.0  # last world is metal
  s.r.per_world["material.base_metalness"].assign(per_world)
  rgb = s.render().rgb()

  # The dielectric worlds render as a renderer without the override; the metal one differs.
  single = scene(xml).render().rgb()[0]
  np.testing.assert_allclose(rgb[:-1], np.broadcast_to(single, rgb[:-1].shape))
  assert np.abs(rgb[-1] - single).max() > 0.05


@pytest.mark.parametrize("tone_mapper", ["pbr_neutral", "aces", "clamp"])
@pytest.mark.parametrize("background", [(40, 44, 55), (200, 30, 90), (255, 255, 255)])
def test_background_color_is_the_display_color(scene, tone_mapper, background):
  """Rays that hit nothing show background_color as given, whatever the tone mapper."""
  # A sphere behind the camera: every pixel misses.
  xml = f"""<mujoco>{_NO_HEADLIGHT}<worldbody>
    <geom type="sphere" size="0.1" pos="0 0 3"/><camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
  </worldbody></mujoco>"""
  rgb = (
    scene(
      xml,
      environment=rw.settings.Environment(background_color=tuple(c / 255.0 for c in background)),
      post=rw.settings.Post(tone_mapper=tone_mapper),
    )
    .render()
    .rgb()[0]
  )
  # PBR Neutral's shoulder only approaches pure white: within 2/255 of it.
  assert np.abs(rgb * 255.0 - np.asarray(background)).max() <= 2.0


def test_inverse_display_transform_inverts_the_kernel():
  """The tone-mapper inverse (Newton steps on the device function, no host copy of the tone
  mappers) maps back to the display values, for every tone mapper."""
  colors = np.array([[0.0, 0.0, 0.0], [0.02, 0.03, 0.05], [0.07, 0.5, 0.2], [0.9, 0.4, 0.1], [3.0, 1.0, 8.0]])
  for tone_mapper in ToneMapper:
    display = apply_display_transform(colors, int(tone_mapper))
    back = apply_display_transform(color.inverse_display_transform(display, int(tone_mapper), "cpu"), int(tone_mapper))
    np.testing.assert_allclose(back, display, atol=1e-5)


def test_hdr_is_consistent_with_display_output(scene):
  xml = _plane_xml(
    'rgba="0.5 0.6 0.7 1" roughness="0.5"',
    '<light type="directional" dir="0.3 0 -1" castshadow="false"/>',
  )
  s = scene(xml, **_HDR, post=rw.settings.Post(tone_mapper="clamp")).render()
  exposure = s.exposure
  expected = srgb(np.clip(s.hdr()[0] * exposure, 0.0, 1.0))
  np.testing.assert_allclose(s.rgb()[0], expected, atol=1.5 / 255)


def test_exposure_from_ev100():
  # The daylight default camera (f/16, 1/125 s, ISO 100) is EV100 ≈ 14.97.
  ev100 = np.log2(16.0**2 / (1 / 125) * 100 / 100)
  assert exposure_from_ev100(ev100) == pytest.approx(2.0**-ev100)


@pytest.mark.parametrize("order", ["tiled", "world", "mixed"])
def test_render_orders_produce_identical_images(scene, order):
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.2 0.2 1"/>
        <camera name="a" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
        <camera name="b" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9"/>
      </worldbody>
    </mujoco>
  """
  # Sizes that are not multiples of the 4-row tile band exercise the last, shorter band; 5 worlds
  # the last, smaller world group of the mixed order.
  kwargs = {"nworld": 5, "cam_res": (37, 21), "outputs": rw.Outputs.of("rgb", "depth", "seg")}
  with tuning.override(block_dim=64):
    row = scene(xml, **kwargs).render()
  other = scene(xml, performance=rw.settings.Performance(render_order=order), **kwargs).render()
  np.testing.assert_array_equal(other.out.rgb.numpy(), row.out.rgb.numpy())
  np.testing.assert_array_equal(other.out.seg.numpy(), row.out.seg.numpy())


def _l_shape_mesh() -> str:
  """An axis-aligned L-shaped prism, inertia frame is rotated from the authored one."""

  def box(lo, hi, base):
    verts = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    faces = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4)]
    faces += [(2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    return verts, [tuple(i + base for i in f) for f in faces]

  v1, f1 = box((0, 0, 0), (2, 0.3, 0.2), 0)
  v2, f2 = box((0, 0.3, 0), (0.3, 1.2, 0.2), 8)
  vertex = " ".join(f"{c:g}" for v in v1 + v2 for c in v)
  face = " ".join(str(i) for f in f1 + f2 for i in f)
  return f'<mesh name="l" inertia="shell" scale="0.5 0.5 0.5" vertex="{vertex}" face="{face}"/>'


@pytest.mark.parametrize("ordered_traversal", [True, False])
def test_reframed_mesh_bvh_matches_mujoco_warp(scene, ordered_traversal):
  """A mesh BLAS rebuilt in the authored frame gives MuJoCo Warp's hits (seg exact, depth close)."""
  xml = f"""
    <mujoco>
      <asset>{_l_shape_mesh()}</asset>
      <worldbody>
        <light pos="0.5 -0.5 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="mesh" mesh="l" pos="-0.3 -0.2 0.1" euler="10 0 25"/>
        <camera name="cam" pos="0 -2.2 1.6" xyaxes="1 0 0 0 0.6 0.8"/>
      </worldbody>
    </mujoco>
  """
  # The software BVH (OptiX mesh GASes: test_optix.py::test_reframed_mesh_gas_matches_software).
  with tuning.override(ordered_traversal=ordered_traversal):
    s = scene(
      xml,
      cam_res=(48, 40),
      outputs=rw.Outputs.of("rgb", "depth", "seg"),
      performance=rw.settings.Performance(traversal="software"),
    )
    assert s.scene.capabilities.mesh_frames and s.r.features.mesh_frames
    s.render()
  ref = mjw.create_render_context(s.mjm, nworld=1, cam_res=(48, 40), render_rgb=True, render_depth=True, render_seg=True)
  mjw.refit_bvh(s.m, s.d, ref)
  mjw.render(s.m, s.d, ref)
  seg = s.seg().reshape(1, -1)
  np.testing.assert_array_equal(seg, ref.seg_data.numpy()[..., 0])
  assert (seg == 1).sum() > 50  # the mesh fills part of the image
  np.testing.assert_allclose(s.depth().reshape(1, -1), ref.depth_data.numpy(), rtol=1e-4, atol=1e-5)


_BUDGET_XML = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <worldbody>
      <light name="key" type="spot" pos="0.6 -0.6 2.5" dir="-0.2 0.2 -1" cutoff="60" diffuse="0.7 0.7 0.7" castshadow="true"/>
      <light name="fill" type="spot" pos="-1.2 -0.4 2" dir="0.5 0.2 -1" cutoff="60" diffuse="0.4 0.4 0.4" castshadow="true"/>
      <light name="weak" type="spot" pos="0 1.5 2" dir="0 -0.5 -1" cutoff="60" diffuse="0.001 0.001 0.001"
        specular="0 0 0" castshadow="true"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="box" pos="0 0 0.4" size="0.25 0.25 0.4"/>
      <camera name="cam" pos="0 -3 2.2" xyaxes="1 0 0 0 0.6 0.8"/>
    </worldbody>
  </mujoco>
"""


def test_shadow_budget(scene):
  """Shadow rays for the strongest lights only: exact without shadows, close if the rest is weak."""
  opts = {"cam_res": (48, 40)}
  full = scene(_BUDGET_XML, **opts).render().rgb()[0].astype(float)
  unshadowed = scene(_BUDGET_XML, shadows=rw.settings.Shadows(enabled=False), **opts).render().rgb()[0].astype(float)
  budget_off = (
    scene(_BUDGET_XML, shadows=rw.settings.Shadows(enabled=False, max_casters=1), **opts).render().rgb()[0].astype(float)
  )
  np.testing.assert_array_equal(budget_off, unshadowed)
  # The weak light is the one left out: the image matches full shadowing to rounding.
  two = scene(_BUDGET_XML, shadows=rw.settings.Shadows(max_casters=2), **opts).render().rgb()[0].astype(float)
  assert np.abs(two - full).max() <= 2.5 / 255
  # One shadow ray: still shadowed (darker than unshadowed), never brighter.
  one = scene(_BUDGET_XML, shadows=rw.settings.Shadows(max_casters=1), **opts).render().rgb()[0].astype(float)
  assert np.all(one <= unshadowed + 1.5 / 255) and (unshadowed - one).max() > 0.1


_CULL_XML = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <worldbody>
      <light name="near" type="point" pos="-0.6 0 1.5" diffuse="0.6 0.6 0.6" castshadow="true"/>
      <light name="walled" type="point" pos="2.5 0 1.0" diffuse="0.8 0.8 0.8" castshadow="true"/>
      <geom type="plane" size="4 4 0.1"/>
      <geom name="wall" type="box" pos="1.5 0 1.5" size="0.05 4 1.5"/>
      <geom type="sphere" pos="-0.4 0 0.3" size="0.3"/>
      <camera name="cam" pos="-0.5 -3 2" xyaxes="1 0 0 0 0.55 0.83"/>
    </worldbody>
  </mujoco>
"""


def test_light_culling(scene):
  """Culling preserves full shadows; a shadow budget benefits inside the finite grid.

  Outside its bounds the grid conservatively evaluates every light, so the budget has the
  unculled approximation's error rather than incorrectly reusing a boundary cell's light list.
  """
  opts = {"cam_res": (48, 40), "outputs": rw.Outputs.of("rgb", "position", "seg")}
  full = (
    scene(_CULL_XML, lighting=rw.settings.Lights(light_grid=False), shadows=rw.settings.Shadows(max_casters=None), **opts)
    .render()
    .rgb()[0]
    .astype(float)
  )
  culled = scene(
    _CULL_XML,
    lighting=rw.settings.Lights(light_grid=True, light_grid_cell=0.5),
    shadows=rw.settings.Shadows(max_casters=None),
    **opts,
  ).render()
  assert np.abs(culled.rgb()[0] - full).max() <= 1.5 / 255
  start = culled.r.context().light_grid.start
  counts = np.diff(start.numpy()[:-1])  # the last entry: movable lights
  # Cells on the camera side see the near light only; blind cells (inside the wall, no lit
  # neighbour) list no light.
  assert counts.max() == 2 and (counts == 1).any()
  # Budget 1 without the grid: the two lights take the small form, which traces every light (the
  # budget selects nothing there): no leak through the wall. With the grid (the general form) the
  # budget applies inside the grid's cells.
  unculled_budget = (
    scene(_CULL_XML, shadows=rw.settings.Shadows(max_casters=1), lighting=rw.settings.Lights(light_grid=False), **opts)
    .render()
    .rgb()[0]
  )
  leak = np.abs(unculled_budget - full).max(-1)
  budget = (
    scene(
      _CULL_XML,
      shadows=rw.settings.Shadows(max_casters=1),
      lighting=rw.settings.Lights(light_grid=True, light_grid_cell=0.5),
      **opts,
    )
    .render()
    .rgb()[0]
  )
  err = np.abs(budget - full).max(-1)
  grid = required(culled.r.context().light_grid)
  position = required(culled.out.position).numpy()[0, 0]
  hit = culled.seg()[0] >= 0
  q = (position - np.asarray(grid.origin)) / grid.cell
  inside = hit & np.all((q >= 0) & (q < np.asarray(grid.dims)), axis=-1)
  outside = hit & ~inside
  assert inside.any() and outside.any()  # the infinite plane exercises both paths
  assert leak.max() <= 1.5 / 255
  assert np.mean(err[inside] <= 1.5 / 255) > 0.97
  # Outside the grid's box the grid lists every light: the general form's budget there (its
  # documented approximation), not the small form's whole table.
  assert np.isfinite(budget[outside]).all()


# A closed room under a sun, lit through a 4 cm roof slit between the lattice's x planes (the
# grid's box starts at x = -2.1: points every 0.5 m, none in 0.13 .. 0.17): no lattice point inside
# sees the sun (the Warehouse's roof, lit through alpha cut-out openings).
_SLIT_XML = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <worldbody>
      <light directional="true" dir="0 0 -1" diffuse="1 1 1" castshadow="true"/>
      <geom type="box" pos="0 0 -0.05" size="2.1 2.1 0.05"/>
      <geom type="box" pos="-2.05 0 1.5" size="0.05 2.1 1.5"/>
      <geom type="box" pos="2.05 0 1.5" size="0.05 2.1 1.5"/>
      <geom type="box" pos="0 -2.05 1.5" size="2.1 0.05 1.5"/>
      <geom type="box" pos="0 2.05 1.5" size="2.1 0.05 1.5"/>
      <geom type="box" pos="-0.985 0 3.05" size="1.115 2.1 0.05"/>
      <geom type="box" pos="1.135 0 3.05" size="0.965 2.1 0.05"/>
      <camera name="cam" pos="0.15 -1.5 1.5" xyaxes="1 0 0 0 0.71 0.71"/>
    </worldbody>
  </mujoco>
"""


def test_light_grid_sun_through_slit(scene):
  """A cell that sees no light, nor its neighbours, still lists the directional lights: the sun
  through a slit narrower than the lattice lights the floor as without the grid (the Warehouse
  roof was black since blind cells list their neighbours' lights only, 39494e5). A second, black
  sun along the same direction keeps the grid on (a single directional light resolves it off)."""
  black = '<light directional="true" dir="0 0 -1" diffuse="0 0 0" specular="0 0 0"/>'
  xml = _SLIT_XML.replace("<worldbody>", "<worldbody>" + black)
  opts = {"cam_res": (48, 40), "shadows": rw.settings.Shadows(max_casters=None)}
  full = scene(xml, lighting=rw.settings.Lights(light_grid=False), **opts).render().rgb()[0].astype(float)
  culled = scene(xml, lighting=rw.settings.Lights(light_grid=True), **opts).render()
  assert full.max() > 0.5 and np.mean(full.max(-1) > 0.1) < 0.3  # a lit strip on a dark floor
  assert np.abs(culled.rgb()[0] - full).max() <= 1.5 / 255
  # Every cell lists the sun (inside the room only through the blind-cell rule).
  assert culled.r.features.culling
  assert (np.diff(culled.r.context().light_grid.start.numpy()[:-1]) == 2).all()  # the last entry: movable lights


def test_image_lights_are_not_point_lights(scene):
  """An image light lights through the environment (B38), never as a point light at its position."""
  light = '<light type="image" texture="sky" pos="0 0 1.5" intensity="3" castshadow="false"/>'
  xml = _plane_xml('rgba="1 1 1 1" metallic="0" roughness="1"', light).replace(
    "<asset>", '<asset><texture name="sky" type="2d" builtin="flat" rgb1="0 0 0" width="8" height="8"/>'
  )
  s = scene(xml, **_HDR)
  assert s.scene.tables["light.strength"].numpy()[0, 0] == 0.0
  np.testing.assert_array_equal(s.render().hdr()[0], 0.0)
  # A grey image lights the plane evenly, as a uniform environment does (a point light 1.5 m above
  # would light its centre most).
  grey = scene(xml.replace('rgb1="0 0 0"', 'rgb1="0.5 0.5 0.5"'), **_HDR).render().hdr()[0]
  assert grey.min() > 0.0 and np.ptp(grey) < 0.02 * grey.mean()


def _lights_xml(n: int) -> str:
  """A white rough plane under `n` point lights of decreasing intensity, seen from above."""
  lights = "".join(
    f'<light type="point" pos="{0.3 * np.cos(k)} {0.3 * np.sin(k)} {1.0 + 0.2 * k}" intensity="{10.0 / (k + 1)}" range="1000" castshadow="false"/>'
    for k in range(n)
  )
  return _plane_xml('rgba="0.8 0.8 0.8 1" metallic="0" roughness="1" specular="0"', lights)


def test_light_cap_keeps_the_strongest_lights_exact(scene):
  """P24: lighting.max_exact shades the K strongest lights through the full BRDF and the rest
  diffusely: on a rough Lambert-like plane (no specular) the image matches the uncapped one; the
  cap compiles in only below the light count."""
  xml = _lights_xml(6)
  # Point-form lights: the cap is a punctual-light loop.
  full = scene(xml, **_HDR, lighting=rw.settings.Lights(area_lights="point", max_exact=None)).render()
  capped = scene(xml, **_HDR, lighting=rw.settings.Lights(area_lights="point", max_exact=2)).render()
  assert full.r.features.light_cap == 0 and capped.r.features.light_cap == 2
  np.testing.assert_allclose(capped.hdr()[0], full.hdr()[0], rtol=0.02)
  assert scene(xml, **_HDR, lighting=rw.settings.Lights(area_lights="point", max_exact=6)).r.features.light_cap == 0
