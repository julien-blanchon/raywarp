"""Tests for textures, normal maps, anti-aliasing, translucency, reflections and soft shadows."""

import numpy as np
import pytest
from conftest import pane_stack_centre, recipe, required, srgb

import raywarp as rw

_TOP_CAMERA = '<camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>'


def _normal_map_xml(tilt_deg: float, light_deg: float) -> str:
  t = np.radians(tilt_deg)
  # Tangent-space normal tilted towards +u, encoded as a linear RGB texel.
  rgb = 0.5 * (np.array([np.sin(t), 0.0, np.cos(t)]) + 1.0)
  ld = np.radians(light_deg)
  return f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset>
        <texture name="nrm" type="2d" builtin="flat" width="8" height="8" rgb1="{rgb[0]} {rgb[1]} {rgb[2]}"
                 colorspace="linear"/>
        <material name="mat" rgba="1 1 1 1" metallic="0" roughness="1" specular="0">
          <layer texture="nrm" role="normal"/>
        </material>
      </asset>
      <worldbody>
        <light type="directional" dir="{-np.sin(ld)} 0 {-np.cos(ld)}" castshadow="false"/>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """


def test_normal_map_tilts_towards_plus_u(scene):
  """MuJoCo maps a plane's +u to +x: a +u-tilted normal map faces a light tilted towards +x."""
  flat = scene(_normal_map_xml(0.0, 30.0), outputs=rw.Outputs.of("rgb", "hdr")).render().hdr()[0, 16, 16]
  toward = scene(_normal_map_xml(30.0, 30.0), outputs=rw.Outputs.of("rgb", "hdr")).render().hdr()[0, 16, 16]
  away = scene(_normal_map_xml(-30.0, 30.0), outputs=rw.Outputs.of("rgb", "hdr")).render().hdr()[0, 16, 16]
  # Lambert: radiance ∝ NoL. Facing the light gives cos(0) = 1, flat cos(30°), away cos(60°).
  np.testing.assert_allclose(toward / flat, 1.0 / np.cos(np.radians(30.0)), rtol=0.02)
  np.testing.assert_allclose(away / flat, np.cos(np.radians(60.0)) / np.cos(np.radians(30.0)), rtol=0.02)


def test_texture_lod_filters_minified_checker(scene):
  """A checker minified far below a pixel averages to the mean of its (linear) texels."""
  xml = f"""
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset>
        <texture name="chk" type="2d" builtin="checker" width="256" height="256" rgb1="1 1 1" rgb2="0 0 0"/>
        <material name="mat" texture="chk" texrepeat="4000 4000" roughness="1" specular="0"/>
      </asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """
  s = scene(xml, post=rw.settings.Post(tone_mapper="clamp")).render()
  rgb = s.rgb()[0]
  np.testing.assert_allclose(rgb.mean(), srgb(np.asarray(0.5)), atol=0.02)
  assert rgb.std() < 0.02


def test_supersampling_resolves_edge_coverage(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody>
        <geom type="box" pos="-0.987 0 0" size="1 1 0.1" rgba="1 1 1 1"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  kwargs = {"post": rw.settings.Post(tone_mapper="clamp"), "cam_res": (8, 8)}
  hard = scene(xml, aa=rw.settings.AntiAliasing(spp=1), **kwargs).render().rgb()[0]
  soft = scene(xml, aa=rw.settings.AntiAliasing(spp=4), **kwargs).render().rgb()[0]
  # The box edge (x = 0.013) cuts through a pixel column right of the image centre.
  assert set(np.unique(hard.round(2))) <= {0.0, 1.0}
  assert np.any((soft > 0.05) & (soft < 0.95))


def test_translucent_layer_blends_over_background(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody>
        <geom type="box" pos="0 0 0" size="1 1 0.01" rgba="1 0 0 1"/>
        <geom type="box" pos="0 0 0.5" size="1 1 0.01" rgba="0 0 1 0.25"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, outputs=rw.Outputs.of("rgb", "hdr"), post=rw.settings.Post(tone_mapper="clamp")).render()
  assert s.r.features.transparency
  exposure = s.exposure
  c = s.hdr()[0, 16, 16] * exposure
  # Constant ambient through a matte surface: about 0.25 · blue + 0.75 · red.
  assert c[0] > 0.6 and c[2] > 0.15 and c[1] < 0.05


def test_mirror_reflects_scene(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="mirror" rgba="0 0 0 1" reflectance="1" roughness="0" specular="0"/></asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mirror"/>
        <geom type="box" pos="0 0 3" size="2 2 0.01" rgba="0 1 0 1"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, post=rw.settings.Post(tone_mapper="clamp")).render()
  assert s.r.features.reflection
  # The smooth black mirror (reflectance 1: its specular F0) shows the green box through its glossy
  # reflection ray (plus the box's own dielectric specular sheen).
  c = s.rgb()[0, 16, 16]
  assert c[1] > 0.9 and c[1] - c[0] > 0.5


def test_soft_shadows_have_penumbra(scene):
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <worldbody>
        <light type="point" pos="0 0 2" intensity="50" range="100" bulbradius="0.4" castshadow="true"/>
        <geom type="plane" size="5 5 0.1" rgba="1 1 1 1"/>
        <geom type="box" pos="0 0 1" size="0.25 0.25 0.01" rgba="1 1 1 1"/>
        <camera name="top" pos="0 0.001 0.6" xyaxes="1 0 0 0 1 0" fovy="120"/>
      </worldbody>
    </mujoco>
  """

  def row(**kwargs):
    return scene(xml, outputs=rw.Outputs.of("rgb", "hdr"), cam_res=(64, 64), **kwargs).render().hdr()[0, 32, :, 0]

  unshadowed = row(shadows=rw.settings.Shadows(enabled=False))
  hard = row(shadows=rw.settings.Shadows(samples=1)) / unshadowed
  soft = row(shadows=rw.settings.Shadows(samples=32)) / unshadowed
  penumbra = lambda v: int(np.sum((v > 0.05) & (v < 0.95)))  # noqa: E731
  assert penumbra(hard) == 0
  assert penumbra(soft) > 4


def test_camera_layer_loop_is_not_unrolled():
  """Warp force-unrolls a constant-range loop whose AST holds a `wp.static` call. The camera ray's
  translucent-layer loop (`pixel._make_trace_body`) must not: four unrolled copies of the surface
  shading grew the OptiX continuation stack from 320 B to 1856 B and made a scene with one
  translucent geom 5.9x slower (PX, MJX Panda). Its body keeps the compile-time choices in the
  layer steps, and its `break` makes Warp emit a dynamic loop. The first surface is shaded before
  the loop, straight-line (W6-S: as the loop's first iteration the own builds ran 2-4.7x slower
  on RTX with the same continuation stack): two shading call sites, `shade` before the loop and
  `shade_behind` in it (the same shading inline, or the OptiX build's layer program). The one
  `wp.static` in its body picks the blend's hit form (Features.world_optics: its range is a run-time
  value and the loop breaks, so Warp emits a dynamic loop either way)."""
  import ast
  import inspect

  from raywarp.core.kernels import pixel

  tree = ast.parse(inspect.getsource(pixel._make_trace_body))
  loops = [
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(1, layer_count(p, mats.bits, geo.bits2))"
  ]
  assert len(loops) == 1
  statics = [ast.unparse(n) for n in ast.walk(loops[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "wp.static"]
  assert statics == ["wp.static(flags.world_optics)"], statics
  assert any(isinstance(n, ast.Break) for n in ast.walk(loops[0]))
  body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "trace_pixel")
  every = [ast.unparse(n.func) for n in ast.walk(body) if isinstance(n, ast.Call)]
  assert every.count("shade") == 1 and every.count("shade_behind") == 1 and every.count("closest") == 2, every
  inside = [ast.unparse(n.func) for n in ast.walk(loops[0]) if isinstance(n, ast.Call)]
  assert inside.count("shade_behind") == 1 and "shade" not in inside, inside


def test_translucent_layers_composite_front_to_back(scene):
  """The layer loop (a dynamic loop since PX) composites `max_layers` surfaces front to back, the
  last one closing the ray with the background behind it (untraced): two black alpha-0.5 panes
  over a white wall under ambient light, each pane's sheen s (its shading times its alpha 0.5).
  One layer: the first pane's sheen alone; two: both sheens plus the background (not the wall)
  through both panes; three: both sheens plus the wall through both panes at 0.5 x 0.5 = 1/4 of
  its radiance."""
  xml = """<mujoco>
    <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
    <worldbody>
      <geom type="box" pos="0 0 0" size="1 1 0.01" rgba="1 1 1 1"/>
      <geom type="box" pos="0 0 0.3" size="1 1 0.01" rgba="0 0 0 0.5"/>
      <geom type="box" pos="0 0 0.6" size="1 1 0.01" rgba="0 0 0 0.5"/>
      <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
    </worldbody>
  </mujoco>"""

  def centre(layers: int) -> np.ndarray:
    s = scene(xml, outputs=rw.Outputs.of("rgb", "hdr"), transparency=rw.settings.Transparency(max_layers=layers)).render()
    return s.hdr()[0, 16, 16] * s.exposure

  wall = scene(xml.replace('rgba="0 0 0 0.5"', 'rgba="0 0 0 0"'), outputs=rw.Outputs.of("rgb", "hdr")).render()
  white = wall.hdr()[0, 16, 16] * wall.exposure
  one, two, three = centre(1), centre(2), centre(3)
  # A pane shows only its dielectric sheen (F0 0.04 of the ambient).
  assert 0.0 < one.max() < 0.03 * white.max()
  np.testing.assert_allclose(three, 1.5 * one + 0.25 * white, rtol=0.02)  # s + 0.5 s + wall / 4
  assert np.all(two >= 1.5 * one * 0.99) and np.all(two < three)  # + the (darker) background / 4


# (recipe_name, panes): its layer budget covers every pane, then the floor (fast 2: one pane;
# balanced 4: three), when a closed pane's two faces are one layer.
_WITHIN_BUDGET = [("fast", 1), ("default", 3)]


@pytest.mark.parametrize(("recipe_name", "npane"), _WITHIN_BUDGET)
def test_a_closed_pane_costs_one_layer(recipe_name, npane):
  """Stacked two-sided closed panes (thin boxes, a USD stage's glass) within a recipe's layer
  budget composite as with an unlimited one: a closed pane is single-sided (its exit face culled)
  and its entry face composites both faces' coverage, one layer. Two-sided, fast's 2
  stopped at the first pane's back face and its pane closed on the environment instead of the
  floor (the glass scene's opaque-looking panes)."""
  seen = pane_stack_centre(recipe_name, npane, "software")
  np.testing.assert_allclose(seen, pane_stack_centre(recipe_name, npane, "software", max_layers=8), rtol=1e-3, atol=1e-5)


PANES = """<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <material name="blue" rgba="0.2 0.2 1 {alpha}"/>
    <material name="green" rgba="0.2 1 0.2 {alpha}"/>
  </asset>
  <worldbody>
    <light pos="0 -2 3" dir="0 1 -1" directional="true"/>
    <geom type="box" pos="0 1 0" size="3 0.05 3" rgba="1 0.1 0.1 1"/>
    <geom type="box" pos="0 0 0" size="0.5 0.02 0.5" material="blue"/>
    <geom type="box" pos="0 0.3 0" size="0.5 0.02 0.5" material="green"/>
    <geom type="box" pos="0 0.6 0" size="0.5 0.02 0.5" material="blue"/>
    <camera name="cam" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="30"/>
  </worldbody>
</mujoco>"""


# The looks compile counted builds (about 25 s each on the CPU): slow.
_LOOKS = [pytest.param(p, marks=pytest.mark.slow) for p in ("rtx_look", "reference")]


@pytest.mark.parametrize("recipe_name", ["fast", "default", "quality", *_LOOKS])
def test_stacked_panes_stay_see_through_in_every_tier(recipe_name):
  """Three stacked 0.2-opacity box panes in front of a red wall, at every recipe's layer budget:
  never opaque (the user's "glass opaque"). A box's two faces are one layer; with the budget past
  all three (default, quality) the red wall shows through, short of it (fast: 2) what
  is behind the last pane is not traced and the background stands in."""
  import mujoco
  import mujoco_warp as mjw

  settings = recipe(recipe_name).replace(
    **{"outputs": rw.Outputs.of("rgb"), "exposure.mode": "manual", "exposure.ev100": 12.0, "performance.traversal": "software"}
  )
  centre = {}
  for alpha in (1.0, 0.2):
    mjm = mujoco.MjModel.from_xml_string(PANES.format(alpha=alpha))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd)
    with rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d), nworld=1, cameras=["cam"], resolution=(32, 32), settings=settings
    ) as r:
      rgb = required(r.render(d).rgb).numpy()
      budget = r.settings.transparency.max_layers
    centre[alpha] = rgb.reshape(32, 32, -1)[14:18, 14:18, :3].astype(float).mean(axis=(0, 1)) / 255.0
  opaque, seen = centre[1.0], centre[0.2]
  assert opaque[2] > opaque[0]  # opaque: the blue pane
  assert np.abs(seen - opaque).max() > 0.1, centre  # see-through, not the opaque pane
  if budget >= 3:  # one layer per pane (its two faces), then the wall
    assert seen[0] > seen[1], centre  # the red wall through the blue / green panes
