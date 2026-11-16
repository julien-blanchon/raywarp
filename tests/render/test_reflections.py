"""Traced glossy reflections (`reflections.enabled`: one ray along the GGX dominant direction)."""

import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, required, whole_table

import raywarp as rw
from raywarp.core.kernels.abi.structs import Params
from raywarp.core.kernels.reflections import shown_weight
from raywarp.core.shade.materials.material import Lobes


def _floor_xml(roughness: float, panel: bool = True, ambient: float = 0.0) -> str:
  # A white metal floor seen from above, under a green emissive panel behind the camera.
  box = '<geom type="box" pos="0 0 3" size="2 2 0.01" material="glow"/>' if panel else ""
  return f"""
    <mujoco>
      <visual><headlight active="1" ambient="{ambient} {ambient} {ambient}" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset>
        <material name="floor" rgba="1 1 1 1" metallic="1" roughness="{roughness}"/>
        <material name="glow" rgba="0 1 0 1" emission="1"/>
      </asset>
      <worldbody>
        <geom type="plane" size="50 50 0.1" material="floor"/>
        {box}
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """


def test_glossy_floor_reflects_emissive_panel(scene):
  xml = _floor_xml(0.1)
  off = (
    scene(xml, post=rw.settings.Post(tone_mapper="clamp"), reflections=rw.settings.Reflections(enabled=False))
    .render()
    .rgb()[0, 16, 16]
  )
  on = (
    scene(xml, post=rw.settings.Post(tone_mapper="clamp"), reflections=rw.settings.Reflections(enabled=True))
    .render()
    .rgb()[0, 16, 16]
  )
  assert off[1] < 0.05
  assert on[1] > 0.8 and on[0] < 0.1


def test_glossy_fades_to_the_environment_term_above_max_roughness(scene):
  xml = _floor_xml(0.8, ambient=0.3)
  off = scene(xml, reflections=rw.settings.Reflections(enabled=False)).render().rgb()
  on = scene(xml, reflections=rw.settings.Reflections(enabled=True)).render().rgb()
  assert on.shape == off.shape
  np.testing.assert_array_equal(on, off)


_LEANING = """
<mujoco>
  <visual><headlight active="1" ambient="0.5 0.5 0.5" diffuse="0 0 0" specular="0 0 0"/></visual>
  <asset>
    <material name="metal" rgba="1 0.5 0.2 1" metallic="1" roughness="0.05" emission="0.3"/>
    <!-- Vertex 4 (no face) only gives MuJoCo's collision hull a volume. -->
    <mesh name="quad" inertia="shell" vertex="-1 -1 0  1 -1 0  1 1 0  -1 1 0  0 0 -0.5" face="0 1 2  0 2 3"
      normal="0 0.139 0.99  0 0.139 0.99  0 0.139 0.99  0 0.139 0.99  0 0 -1"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="quad" material="metal"/>
    <camera name="low" pos="0 -3 0.5" xyaxes="1 0 0 0 0.17 0.985" fovy="30"/>
  </worldbody>
</mujoco>
"""


def test_reflections_under_the_surface_are_not_traced(scene):
  """B68: a reflected direction above the shading normal's horizon but under the geometric
  surface would hit the surface itself (a displaced copy of its texture: the ghosted sign). It
  takes the environment term, as a reflection that misses: here every ray, so the image is
  "off"'s."""
  kw = {"cam_res": (32, 24), "outputs": rw.Outputs.of("rgb", "seg")}
  off = scene(_LEANING, reflections=rw.settings.Reflections(enabled=False), **kw).render()
  on = scene(_LEANING, reflections=rw.settings.Reflections(enabled=True), **kw).render()
  quad = off.seg()[0] >= 0
  assert quad.mean() > 0.1
  np.testing.assert_array_equal(on.rgb()[0][quad], off.rgb()[0][quad])


def test_specular_curve_f90():
  """Unreal specular: F90 = s (1 - r) with s = 2 reflectance^2 for dielectrics, the base color
  for metals; with F90 = 1 the albedo is Filament's mix(dfg.x, dfg.y, f0)."""
  import warp as wp

  from raywarp.core.shade.materials import brdf

  @wp.kernel
  def k(out: wp.array[wp.vec3]):
    out[0] = brdf.f90_curve(wp.vec3(0.2, 0.4, 0.6), 0.0, 0.5, 0.25)
    out[1] = brdf.f90_curve(wp.vec3(0.2, 0.4, 0.6), 1.0, 0.5, 0.25)
    f0 = wp.vec3(0.04)
    out[2] = brdf.specular_albedo_rgb(f0, wp.vec3(1.0), wp.vec2(0.3, 0.9))
    out[3] = brdf.f_schlick_rgb(f0, wp.vec3(0.375), 0.0)

  out = wp.zeros(4, dtype=wp.vec3, device="cpu")
  wp.launch(k, dim=1, inputs=[out], device="cpu")
  o = out.numpy()
  np.testing.assert_allclose(o[0], 0.5 * 0.75, rtol=1e-6)
  np.testing.assert_allclose(o[1], [0.2, 0.4, 0.6], rtol=1e-6)
  np.testing.assert_allclose(o[2], 0.96 * 0.3 + 0.04 * 0.9, rtol=1e-6)
  np.testing.assert_allclose(o[3], 0.375, rtol=1e-6)  # grazing: F90


def test_reflection_rays_skip_emissive_panels_lit_as_area_lights():
  """A mirror floor under an emissive panel turned into a rect light: the LTC term reflects the
  panel; the glossy ray that hits the panel adds no second copy of its emission."""
  import warp as wp

  import raywarp as rw

  def build():
    b = rw.SceneBuilder()
    floor = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=1.0, specular_roughness=0.05))
    glow = b.add_material(
      rw.Material(base_color=(0.0, 0.0, 0.0), specular_roughness=1.0, emission_luminance=50.0, emission_color=(1, 1, 1))
    )
    b.add_instance("plane", size=(5, 5, 0), material=floor)
    b.add_instance("box", size=(1.0, 1.0, 0.004), xform=np.c_[np.eye(3), [0, 0, 2]], material=glow)
    scene = rw.upload(b.build(), device="cpu")
    assert scene.nlight == 2  # a two-sided panel: one light per face
    return scene

  def render(scene, reflections, light_on=True, panels=True):
    s = rw.RenderSettings(
      gi=NO_GI,
      outputs=rw.Outputs.of("hdr"),
      lighting=rw.settings.Lights(area_lights="ltc", emissive_lights=panels),
      reflections=reflections,
      environment=rw.settings.Environment(enabled=False),
    )
    per_world = None if light_on else rw.PerWorld(light=["active"])
    with wp.ScopedDevice("cpu"):
      r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(8, 8), per_world=per_world, settings=s)
      if not light_on:
        whole_table(r.per_world["light.active"]).zero_()
      r.camera_pose.assign(np.array([[[0, 0, 1, 0, 0, 0, 1]]], np.float32))  # looking down at the floor
      r.render()
      return required(r.outputs.hdr).numpy()[0, 0, 4, 4]

  lit = build()
  no_refl, glossy_refl = rw.settings.Reflections(enabled=False), rw.settings.Reflections(enabled=True)
  off, glossy = render(lit, no_refl), render(lit, glossy_refl)
  assert off.mean() > 1.0  # the LTC highlight of the panel
  np.testing.assert_allclose(glossy, off, rtol=0.02)  # not twice
  # Without the light (the setting off, or the light switched off) the reflection ray sees the
  # panel's emission.
  np.testing.assert_allclose(render(lit, glossy_refl, panels=False), off, rtol=0.05)
  np.testing.assert_allclose(render(lit, glossy_refl, light_on=False), off, rtol=0.05)


def test_reflection_shadows_keep_the_sun_out_of_shadowed_reflections():
  """A mirror floor reflects a wall that an overhead plate shades from a tilted sun: unshadowed
  reflection hits show it sunlit; "sun" (shadow rays for directional lights only) shades it."""
  import warp as wp

  import raywarp as rw

  b = rw.SceneBuilder()
  mirror = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=1.0, specular_roughness=0.02))
  white = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("plane", size=(5, 5, 0), material=mirror)
  b.add_instance("box", size=(3.0, 0.05, 2.0), xform=np.c_[np.eye(3), [0, 2.05, 0]], material=white)
  b.add_instance("box", size=(3.0, 1.75, 0.05), xform=np.c_[np.eye(3), [0, -0.25, 3.0]], material=white)
  c = s = np.sqrt(0.5)
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.DISTANT,
      intensity=3.0,
      angle=0.0,
      xform=np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0]], float),
    )
  )
  host = b.build()
  eye, target = np.array([0.0, -1.0, 1.0]), np.array([0.0, 1.0, 0.0])
  front = (target - eye) / np.linalg.norm(target - eye, keepdims=True)
  right = np.cross(front, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right, keepdims=True)
  up = np.cross(right, front)
  q = wp.quat_from_matrix(wp.mat33(*np.stack([right, up, -front], axis=1).ravel().tolist()))

  def render(mode):
    with wp.ScopedDevice("cpu"):
      scene = rw.upload(host)
      settings = rw.RenderSettings(
        gi=NO_GI,
        outputs=rw.Outputs.of("hdr"),
        reflections=rw.settings.Reflections(shadows=mode),
        environment=rw.settings.Environment(enabled=False),
      )
      r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=10.0)], resolution=(8, 8), settings=settings)
      r.camera_pose.assign(np.array([[[*eye, *q]]], np.float32))
      r.render()
      return float(required(r.outputs.hdr).numpy()[0, 0, 3:5, 3:5].mean())

  off, sun = render("off"), render("sun")
  assert off > 0.1
  assert sun < 0.1 * off
  with pytest.raises(ValueError, match=r"reflections\.shadows"):
    rw.settings.Reflections(shadows=True)  # pyrefly: ignore[bad-argument-type]


def _sun_room(*, sealed: bool) -> rw.Scene:
  """A mirror floor before a white wall at y = 1 whose face a tilted sun lights (from -y, 45
  degrees up); `sealed`: the wall is one side of a closed 2 x 2 x 2 m room that no sun reaches."""
  b = rw.SceneBuilder()
  mirror = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=1.0, specular_roughness=0.02))
  white = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("plane", size=(5, 5, 0), material=mirror)
  walls = [((-1.1, 1.0, 0.0), (1.1, 1.1, 2.0))]
  if sealed:
    walls += [((-1.1, -1.1, 2.0), (1.1, 1.1, 2.1)), ((-1.1, -1.0, 0.0), (-1.0, 1.0, 2.0))]
    walls += [((1.0, -1.0, 0.0), (1.1, 1.0, 2.0)), ((-1.1, -1.1, 0.0), (1.1, -1.0, 2.0))]
  for lo, hi in walls:
    lo, hi = np.asarray(lo), np.asarray(hi)
    b.add_instance("box", size=tuple((hi - lo) / 2), xform=np.c_[np.eye(3), (lo + hi) / 2], material=white)
  c = s = np.sqrt(0.5)
  # Travelling along (0, s, -s): from -y, down onto the wall's -y face.
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.DISTANT,
      intensity=3.0,
      angle=0.0,
      xform=np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0]], float),
    )
  )
  return b.build()


def _mirrored_wall(host: rw.Scene, *, gi: bool) -> float:
  """Mean HDR of the wall's reflection in the floor (reflections.shadows "off": no shadow ray at
  the hit), with probe GI or without (no sky visibility)."""
  eye, target = np.array([0.0, -0.6, 1.0]), np.array([0.0, 0.6, 0.0])
  front = (target - eye) / np.linalg.norm(target - eye, keepdims=True)
  right = np.cross(front, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right, keepdims=True)
  up = np.cross(right, front)
  q = wp.quat_from_matrix(wp.mat33(*np.stack([right, up, -front], axis=1).ravel().tolist()))
  gi_settings = rw.settings.GI(enabled=True, spacing=0.5, max_probes=64, rays=64, bake_updates=4) if gi else NO_GI
  with wp.ScopedDevice("cpu"):
    settings = rw.RenderSettings(
      gi=gi_settings,
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal="software"),
      reflections=rw.settings.Reflections(shadows="off"),
      environment=rw.settings.Environment(enabled=False),
    )
    r = rw.Renderer(rw.upload(host), nworld=1, cameras=[rw.CameraSpec(fovy=10.0)], resolution=(8, 8), settings=settings)
    r.camera_pose.assign(np.array([[[*eye, *q]]], np.float32))
    r.render()
    return float(required(r.outputs.hdr).numpy()[0, 0, 3:5, 3:5].mean())


def test_unshadowed_reflection_hits_take_the_sky_visibility_as_the_sun_s():
  """Without sun shadow rays, a reflection hit's directional lights take its sky visibility
  (`reflections.sun_visibility`): inside a sealed room the mirrored wall gets no sun (it glowed
  sunlit before); in the open the wall stays sunlit as without any visibility estimate."""
  sealed_open, sealed_gi = _mirrored_wall(_sun_room(sealed=True), gi=False), _mirrored_wall(_sun_room(sealed=True), gi=True)
  assert sealed_open > 0.1  # no estimate (GI off): the unshadowed sun
  assert sealed_gi < 0.05 * sealed_open
  outdoor_open, outdoor_gi = _mirrored_wall(_sun_room(sealed=False), gi=False), _mirrored_wall(_sun_room(sealed=False), gi=True)
  np.testing.assert_allclose(outdoor_gi, outdoor_open, rtol=0.15)


_MIRROR = """
  <mujoco>
    <visual><headlight active="0" ambient="0 0 0"/></visual>
    <asset><material name="floor" rgba="0 0 0 1" specular="0" roughness="0" reflectance="{r}"/></asset>
    <worldbody>
      <geom type="plane" size="20 20 0.1" material="floor"/>
      <camera name="grazing" pos="0 -2.819 1.026" xyaxes="1 0 0 0 0.342 0.9397" fovy="2"/>
    </worldbody>
  </mujoco>
"""


def test_mujoco_reflectance_is_a_fresnel_weighted_mirror(scene):
  """W22: MuJoCo `reflectance` r is the specular lobe's F0 (MuJoCo's Filament mapping): a smooth
  floor reflects what its glossy ray sees by the lobe's Fresnel-weighted albedo, energy-consistent
  with the rest of its shading. Seen 70 degrees off the normal F = 0.342 for r = 0.25; the flat
  blend weighted it by r."""
  white = np.ones((16, 32, 3), np.float32)

  def pixel(r: float) -> float:
    s = scene(_MIRROR.format(r=r), envmap=white, envmap_intensity=1.0, outputs=rw.Outputs.of("hdr"))
    return float(s.render().hdr()[0, 16, 16].mean())

  surface, mirrored = pixel(0.0), pixel(0.25)
  nov = np.cos(np.radians(70.0))
  # The smooth lobe's albedo is Schlick's Fresnel at NoV: F0 + (1 - F0)(1 - NoV)^5 (F0 = 0: the
  # black dielectric's own grazing reflection).
  assert mirrored == pytest.approx(0.25 + 0.75 * (1.0 - nov) ** 5, abs=0.01)
  assert surface == pytest.approx((1.0 - nov) ** 5, abs=0.01)


# ---------------------------------------------------------------------------------------------
# A textured receiver must not jump in brightness at the reflection ray cutoff.


@wp.kernel
def _weights(
  # In:
  roughness: wp.array[float],
  albedo: wp.array[float],
  cutoff: float,
  # Out:
  weights: wp.array[float],
):
  """Normal-incidence dielectric lobe with a textured roughness straddling the cutoff."""
  i = wp.tid()
  p = Params()
  p.rt.glossy = 1  # reflections on at run time
  p.reflection_max_roughness = 0.5
  p.reflection_fade_roughness = 0.25
  p.reflection_min_weight = cutoff
  lobes = Lobes()
  lobes.perceptual_roughness = roughness[i]
  lobes.spec_albedo = wp.vec3(albedo[i])
  lobes.energy_comp = wp.vec3(1.0)
  weights[i] = shown_weight(p, lobes, 1.0)  # the fade and cutoff alone (a lobe sample's share)


def test_textured_dielectric_reflection_cutoff_is_continuous():
  """Infinitesimal roughness variation cannot toggle half the incident radiance."""
  roughness = wp.array([0.375 - 1e-5, 0.375 + 1e-5, 0.0, 0.5], dtype=float, device="cpu")
  albedo = wp.full(4, 0.04, dtype=float, device="cpu")
  weights = wp.zeros(4, dtype=float, device="cpu")
  wp.launch(_weights, dim=4, inputs=[roughness, albedo, 0.02, weights], device="cpu")
  values = weights.numpy()
  assert abs(values[0] - values[1]) < 1e-3
  np.testing.assert_array_equal(values[2:], [1.0, 0.0])


@pytest.mark.parametrize("cutoff", [0.0, 0.005, 0.02, 0.1])
def test_cutoff_sweeps_roughness_and_specular_albedo(cutoff):
  """Albedo changes (Fresnel/normal/DR) and roughness changes stay bounded and continuous."""
  samples = 2049
  rough = np.r_[np.linspace(0, 0.6, samples), np.full(samples, 0.3)].astype(np.float32)
  albedo = np.r_[np.full(samples, 0.04), np.linspace(0, 1, samples)].astype(np.float32)
  output = wp.zeros(2 * samples, dtype=float, device="cpu")
  wp.launch(
    _weights,
    dim=2 * samples,
    inputs=[wp.array(rough, device="cpu"), wp.array(albedo, device="cpu"), cutoff, output],
    device="cpu",
  )
  actual = output.numpy().reshape(2, samples)
  full = np.clip((0.5 - rough) / 0.25, 0, 1).reshape(2, samples)
  assert np.isfinite(actual).all() and (actual >= 0).all()
  assert np.all(actual <= full + 1e-6)
  # Check the visible contribution: at zero albedo a skipped ray has zero energy,
  # even though its unmodulated blend weight need not approach zero.
  visible = actual * albedo.reshape(2, samples)
  assert np.abs(np.diff(visible, axis=1)).max() < 0.002
  if cutoff == 0:
    np.testing.assert_allclose(actual[:, 1:], full[:, 1:], atol=1e-6)
  else:
    old = full * (full * albedo.reshape(2, samples) > cutoff)
    # For fixed dominant rays this implies no increase in absolute per-channel HDR
    # error relative to the full reference, whatever the sign of traced minus sky.
    assert np.all(actual >= old - 1e-6)
    previous = old > 0
    np.testing.assert_allclose(actual[previous], old[previous], atol=1e-6)
