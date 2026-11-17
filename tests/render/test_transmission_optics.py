"""Transmissive materials' optics at camera rays (`kernels/layer.transmit_step`,
`pipeline/upload/optics.py`): Snell refraction of solids, total internal reflection, thin walls
straight, frosted cones, and Beer-Lambert absorption over the distance travelled inside."""

import math

import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
from raywarp._base.types import NTEXROLE, OPTICS_ROWS, TEX_IDENTITY, TexRole
from raywarp.core.kernels import layer
from raywarp.core.pipeline.derive.materials import base_shares, derive
from raywarp.core.pipeline.upload.materials import _fold_base_maps
from raywarp.core.pipeline.upload.optics import compose, material_optics, optics_inputs
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import Slot, TextureRef


@wp.kernel
def _step(
  optics: wp.array2d[wp.vec4],
  normals: wp.array[wp.vec3],
  directions: wp.array[wp.vec3],
  roughness: float,
  out_dir: wp.array[wp.vec3],
  out_tint: wp.array[wp.vec3],
  out_spread: wp.array[float],
):
  i = wp.tid()
  d, spread = layer.transmit_step(optics, i % 2, normals[i], roughness, directions[i], 0.0)
  out_dir[i] = d
  tint, _shaded = layer.absorb_exit(optics, i % 2, normals[i], directions[i], 0.5, wp.vec3(1.0), wp.vec3(0.0))
  out_tint[i] = tint
  out_spread[i] = spread


def _run(normals, directions, roughness=0.0):
  """transmit_step and absorb_exit (after 0.5) of instance i % 2 (0: a solid of IOR 1.5 and
  extinction (1, 2, 3); 1: thin)."""
  rows = np.zeros((2, OPTICS_ROWS, 4), np.float32)
  rows[0, 0] = (1.0, 2.0, 3.0, 1.5)
  optics = wp.array(rows, dtype=wp.vec4, device="cpu")
  n = len(normals)
  out = [wp.zeros(n, dtype=t, device="cpu") for t in (wp.vec3, wp.vec3, float)]
  vec = lambda a: wp.array(np.asarray(a, np.float32), dtype=wp.vec3, device="cpu")  # noqa: E731
  wp.launch(_step, dim=n, inputs=[optics, vec(normals), vec(directions), roughness], outputs=out, device="cpu")
  return [o.numpy() for o in out]


def test_snell_refraction_tir_and_thin_walls():
  """Entering a solid at 45 degrees bends to asin(sin 45 / 1.5), unabsorbed; leaving it (the back
  face) bends back, absorbed by its medium over the distance inside; past the critical angle the
  ray reflects; a thin wall keeps the ray straight, unabsorbed. Roughness widens the cone by
  what a microfacet tilted by alpha turns the refracted ray, |1 - n_in / n_out| alpha: 1/3 alpha
  entering IOR 1.5, 1/2 alpha leaving, 5/6 alpha through a thin wall (both faces), 2 alpha at
  total internal reflection; not a reflection's 2 alpha per crossing, which read glass's
  surroundings through mips 4-6x too blurry (a milky glass sphere under a bright HDRI)."""
  s45 = math.sqrt(0.5)
  inside = math.sin(math.radians(60.0))
  normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
  # 0: entering solid at 45 deg; 1: thin at 45 deg; 2: leaving the solid (ray along -n of the back
  # face) at a small angle; 3 is thin again (instance 3 % 2).
  bent = math.asin(s45 / 1.5)
  leave = (math.sin(bent), 0.0, math.cos(bent))  # the refracted ray, now leaving through a face (0, 0, 1)
  directions = [(s45, 0, -s45), (s45, 0, -s45), leave, (0, 0, -1)]
  d, tint, spread = _run(normals, directions, roughness=0.5)
  np.testing.assert_allclose(d[0], (math.sin(bent), 0.0, -math.cos(bent)), atol=1e-5)
  np.testing.assert_allclose(tint[0], 1.0)  # entering: nothing crossed yet
  np.testing.assert_allclose(d[1], directions[1], atol=1e-6)
  np.testing.assert_allclose(tint[1], 1.0)
  np.testing.assert_allclose(d[2], (s45, 0, s45), atol=1e-5)  # back out at 45 degrees
  np.testing.assert_allclose(tint[2], np.exp(-0.5 * np.array([1.0, 2.0, 3.0])), rtol=1e-5)
  np.testing.assert_allclose(spread, np.array([1 / 3, 5 / 6, 1 / 2, 5 / 6]) * 0.25, rtol=1e-5)
  # Total internal reflection: leaving at 60 degrees inside (critical angle 41.8).
  d, _, spread = _run([(0, 0, 1)], [(inside, 0, 0.5)], roughness=0.5)
  np.testing.assert_allclose(d[0], (inside, 0, -0.5), atol=1e-5)
  np.testing.assert_allclose(spread, 2 * 0.25, rtol=1e-5)


def test_material_optics_rows():
  """Extinction -ln(color) / depth of solid transmissive materials, their IOR: the authored depth,
  else the median mean chord of the material's closed instances (boxes 0.2 and 0.05 m thick:
  2abc / (ab + bc + ca)); a solid on an open plane is no volume (surface tint, IOR only); a thin
  wall or an opaque material has no row. Shadow rays pass exp(-sigma chord / 2) per face."""
  b = rw.SceneBuilder()
  color = (1.0, 0.5, 0.25)
  ex = rw.scene.OpenPbrExtras(transmission_depth=0.5)
  authored = b.add_material(rw.Material(transmission_weight=1.0, transmission_color=color, specular_ior=1.4, extras=ex))
  thin = b.add_material(rw.Material(transmission_weight=1.0, transmission_color=color, geometry_thin_walled=True, extras=ex))
  opaque = b.add_material(rw.Material())
  derived = b.add_material(rw.Material(transmission_weight=1.0, transmission_color=color))
  pane = b.add_material(rw.Material(transmission_weight=1.0, transmission_color=color, specular_ior=1.45))
  for material in (authored, thin, opaque):
    b.add_instance("box", size=(0.5, 0.5, 0.1), material=material)
  for half in (0.1, 0.025):
    b.add_instance("box", size=(0.5, 0.5, half), material=derived)
  b.add_instance("plane", size=(0.5, 0.5, 0.0), material=pane)
  scene = b.build()
  optics = material_optics(scene)
  rows = optics.rows
  tau = -np.log(color)
  np.testing.assert_allclose(rows[0, 0], (*(tau / 0.5), 1.4), rtol=1e-5)
  np.testing.assert_array_equal(rows[1:3], 0)
  chords = [2 * t / (1 + 2 * t) for t in (0.2, 0.05)]  # a 1 x 1 x t box
  np.testing.assert_allclose(optics.chord[3:5], chords, rtol=1e-6)
  np.testing.assert_allclose(rows[3, 0, :3], tau / np.median(chords), rtol=1e-5)
  np.testing.assert_allclose(rows[4, 0], (0, 0, 0, 1.45), rtol=1e-6)
  np.testing.assert_array_equal(optics.volume, [True, False, False, True, False])
  # The faces' share of the shadow transmittance: a white see-through share (`compose`).
  m = len(rows)
  ones = np.ones((m, 4), np.float32)
  n = len(scene.instances.material)
  inputs = optics_inputs(
    scene, optics, see_alpha=np.zeros(m, np.float32), transmissive=np.zeros(m, bool), base=ones, doubled=np.zeros(n, bool)
  )
  ior = np.asarray(scene.materials.specular_ior, np.float32)
  face = compose(inputs, np.asarray(scene.instances.material, np.int32), ior).shadow
  np.testing.assert_allclose(face[3], np.exp(-rows[3, 0, :3] * chords[0] / 2), rtol=1e-5)
  np.testing.assert_array_equal(face[[1, 2, 5]], 1.0)


RES = 24


def _slab_scene(*, depth: float, thin: bool = False, thickness: tuple[float, ...] = (0.2,)) -> rw.Scene:
  """A white emissive wall behind 1 x 1 m glass slabs of `thickness` (transmission color
  (1, 0.5, 0.25)), side by side along x, 0.6 m apart."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=100.0, emission_color=(1.0, 1.0, 1.0)))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  ex = rw.scene.OpenPbrExtras(transmission_depth=depth) if depth > 0 else None
  glass = rw.Material(
    transmission_weight=1.0,
    transmission_color=(1.0, 0.5, 0.25),
    specular_roughness=0.0,
    geometry_thin_walled=thin,
    extras=ex,
  )
  material = b.add_material(glass)
  for k, t in enumerate(thickness):
    x = 0.6 * (k - (len(thickness) - 1) / 2)
    b.add_instance(
      "box", size=(0.5 if len(thickness) == 1 else 0.25, 0.5, t / 2), material=material, xform=np.c_[np.eye(3), [x, 0.0, 0.0]]
    )
  return b.build()


def _render(scene: rw.Scene, pose: tuple[float, ...] = (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0), layers: int = 4) -> np.ndarray:
  rs = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    environment=rw.settings.Environment(enabled=False, background_color=(0.5, 0.5, 0.5)),
    gi=rw.settings.GI(enabled=False),
    transparency=rw.settings.Transparency(max_layers=layers),
    exposure=rw.settings.Exposure(mode="manual", ev100=0.0),
  )
  with rw.Renderer(
    rw.upload(scene, "cpu"), nworld=1, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(RES, RES), settings=rs
  ) as r:
    r.camera_pose.assign(np.array([[pose]], np.float32))
    r.render()
    return np.asarray(required(r.outputs.hdr).numpy()).reshape(RES, RES, -1)[..., :3].astype(np.float64)


def test_beer_lambert_absorbs_over_the_slab_thickness():
  """Through a 0.2 m slab whose color is reached at depth 0.1 m (two depths): green 0.5^2 and blue
  0.25^2 of red (Fresnel cancels in the ratios), at depth 0.2 m: one depth. Without a depth the
  color is reached at the slab's mean chord, 2 abc / (ab + bc + ca) = 0.286 m: 0.7 of it."""
  c = RES // 2
  thick = _render(_slab_scene(depth=0.1))[c - 2 : c + 2, c - 2 : c + 2].mean(axis=(0, 1))
  np.testing.assert_allclose(thick[1:] / thick[0], (0.25, 0.0625), rtol=0.05)
  one = _render(_slab_scene(depth=0.2))[c - 2 : c + 2, c - 2 : c + 2].mean(axis=(0, 1))
  np.testing.assert_allclose(one[1:] / one[0], (0.5, 0.25), rtol=0.05)
  chord = 2 * 0.2 / (1 + 2 * 0.2)
  derived = _render(_slab_scene(depth=0.0))[c - 2 : c + 2, c - 2 : c + 2].mean(axis=(0, 1))
  np.testing.assert_allclose(derived[1:] / derived[0], np.array((0.5, 0.25)) ** (0.2 / chord), rtol=0.05)


def test_unauthored_glass_tints_by_thickness():
  """Two slabs of one glass without a depth, 0.2 and 0.05 m thick: the color is reached at the
  median of their mean chords, so the thick one is tinted more than the thin one (a ratio of
  c^(t / median), per slab), not once per face whatever the thickness."""
  scene = _slab_scene(depth=0.0, thickness=(0.2, 0.05))
  image = _render(scene, pose=(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0))
  chords = [2 * t * 0.5 / (0.5 + t + 0.5 * t) for t in (0.2, 0.05)]  # a 0.5 x 1 x t box
  median = sorted(chords)[0] / 2 + sorted(chords)[1] / 2  # the median of two
  for t, column in ((0.2, RES // 2 - 3), (0.05, RES // 2 + 3)):
    seen = image[RES // 2 - 1 : RES // 2 + 1, column - 1 : column + 1].mean(axis=(0, 1))
    np.testing.assert_allclose(seen[1:] / seen[0], np.array((0.5, 0.25)) ** (t / median), rtol=0.05)


def test_solid_glass_refracts_the_background():
  """Seen at an angle, a solid slab shifts the wall behind it (its edge moves) while a thin-walled
  one does not: the two images differ inside the slab and agree outside."""
  angle = math.radians(25.0)
  q = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))  # turned about y
  pose = (3.0 * math.sin(angle), 0.0, 3.0 * math.cos(angle), *q)
  solid = _render(_slab_scene(depth=0.0), pose)
  thin = _render(_slab_scene(depth=0.0, thin=True), pose)
  assert np.abs(solid - thin).max() > 0.0


def _glass_material(*, thin: bool) -> rw.Material:
  """Clear glass, solid or thin-walled."""
  return rw.Material(transmission_weight=1.0, specular_roughness=0.0, geometry_thin_walled=thin)


def test_a_solid_glass_ball_inverts_what_is_behind_it():
  """A solid sphere of IOR 1.5 is a ball lens (focal length n r / (2 (n - 1)) = 0.53 m from its
  centre for r 0.35 m): a wall 2 m behind it, emissive on its left half and dark on its right,
  shows inverted through it (its left side dark, its right side bright), and the right way round
  through a thin-walled sphere (no bend)."""
  c = RES // 2

  def ball(*, thin: bool) -> rw.Scene:
    b = rw.SceneBuilder()
    lit = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=100.0))
    dark = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0)))
    b.add_instance("box", size=(2.5, 5.0, 0.01), material=lit, xform=np.c_[np.eye(3), [-2.5, 0.0, -2.0]])
    b.add_instance("box", size=(2.5, 5.0, 0.01), material=dark, xform=np.c_[np.eye(3), [2.5, 0.0, -2.0]])
    material = b.add_material(_glass_material(thin=thin))
    b.add_instance("sphere", size=(0.35, 0.0, 0.0), material=material, xform=np.c_[np.eye(3), [0.0, 0.0, 0.0]])
    return b.build()

  for thin in (False, True):
    image = _render(ball(thin=thin)).mean(axis=2)
    left, right = image[c - 1 : c + 1, c - 5 : c - 3].mean(), image[c - 1 : c + 1, c + 3 : c + 5].mean()
    if thin:
      assert left > 10.0 * right, (left, right)
    else:
      assert right > 10.0 * left, (left, right)


@pytest.mark.parametrize(
  ("kind", "size"), [("box", (0.5, 0.5, 0.05)), ("box", (0.5, 0.5, 0.025)), ("sphere", (0.35, 0.0, 0.0))]
)
def test_a_thin_walled_pane_shows_what_is_behind_it_in_a_glass_build(kind: str, size: tuple[float, float, float]):
  """An opaque dark bar behind a thin-walled glass slab (0.1 and 0.05 m thick: BUG-731's two
  slabs) or sphere shows through it as dark as it is (a solid glass instance elsewhere in the
  scene compiles the solid-glass class: the thin-walled shape passes straight through its layer
  loop, not as a solid)."""
  c = RES // 2
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=100.0))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  bar = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0)))
  b.add_instance("box", size=(0.1, 3.0, 0.01), material=bar, xform=np.c_[np.eye(3), [0.0, 0.0, -1.0]])
  thin = b.add_material(_glass_material(thin=True))
  b.add_instance(kind, size=size, material=thin, xform=np.c_[np.eye(3), [0.0, 0.0, 0.0]])
  solid = b.add_material(_glass_material(thin=False))
  b.add_instance("box", size=(0.2, 0.2, 0.2), material=solid, xform=np.c_[np.eye(3), [4.0, 0.0, 0.0]])
  image = _render(b.build()).mean(axis=2)
  behind, beside = image[c - 2 : c + 2, c - 1 : c + 1].mean(), image[c - 2 : c + 2, c + 4 : c + 6].mean()
  assert behind < 0.1 * beside, (behind, beside)


def test_a_solid_s_exit_face_is_no_layer():
  """A solid's back face adds no light (`layer.absorb_exit`) and is not counted against
  `transparency.max_layers` (as a closed pane's): with two layers the slab's front face is one and
  the wall behind it the second, the same image as with four (fast's two layers saw the
  background behind the floor through solid glass: near-black props). With one layer, the budget
  ends inside the slab: the ambient light ahead shows through it (here none: no environment, GI
  or fill), darker than the wall."""
  c = RES // 2
  window = (slice(c - 2, c + 2), slice(c - 2, c + 2))
  two, four = _render(_slab_scene(depth=0.0), layers=2), _render(_slab_scene(depth=0.0), layers=4)
  np.testing.assert_allclose(two[window], four[window], rtol=1e-5, atol=1e-6)
  one = _render(_slab_scene(depth=0.0), layers=1)[window].mean()
  assert one < 0.1 * four[window].mean(), (one, four[window].mean())


def test_past_the_layer_budget_solid_glass_shows_the_ambient_light_ahead():
  """Two solid glass balls one behind the other in front of a white wall, lit by a uniform
  environment that rays leaving the scene do not show (`environment.background` off: black). With
  fast's two layers the second ball's entry face is the last counted surface: behind it the ray
  sees the ambient light ahead (the environment's SH9 irradiance over pi), a bright ball as the
  four-layer trace's (the wall through both), not the black background (black glass)."""
  c = RES // 2
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_weight=0.0))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  glass = b.add_material(_glass_material(thin=False))
  for z in (0.0, -0.9):
    b.add_instance("sphere", size=(0.35, 0.0, 0.0), material=glass, xform=np.c_[np.eye(3), [0.0, 0.0, z]])
  b.add_environment(rw.scene.EnvironmentLight(intensity=1.0))
  scene = b.build()

  def seen(layers: int) -> float:
    rs = rw.RenderSettings(
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal="software"),
      environment=rw.settings.Environment(enabled=True, background=False),
      gi=rw.settings.GI(enabled=False),
      transparency=rw.settings.Transparency(max_layers=layers),
      exposure=rw.settings.Exposure(mode="manual", ev100=0.0),
    )
    with rw.Renderer(
      rw.upload(scene, "cpu"), nworld=1, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(RES, RES), settings=rs
    ) as r:
      r.camera_pose.assign(np.array([[(0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0)]], np.float32))
      r.render()
      assert r.features.glass_shapes
      hdr = np.asarray(required(r.outputs.hdr).numpy()).reshape(RES, RES, -1)[..., :3]
      return float(hdr[c - 1 : c + 1, c - 1 : c + 1].mean())

  budget, traced = seen(2), seen(4)
  assert 0.5 * traced < budget < 2.0 * traced, (budget, traced)


def _textured_slab(*, texture: bool, flags: int = 0) -> rw.Scene:
  """`_slab_scene`'s thin-surface slab (no depth), its transmission color white and its base
  color white, optionally textured by a uniform (1, 0.5, 0.25) base-color map; `flags`: its
  MaterialFlags."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=100.0, emission_color=(1.0, 1.0, 1.0)))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  glass = rw.Material(
    base_color=(1.0, 1.0, 1.0),
    transmission_weight=1.0,
    transmission_color=(1.0, 1.0, 1.0),
    specular_roughness=0.0,
    ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags(flags)),
  )
  if texture:
    img = np.empty((2, 2, 4), np.uint8)
    img[...] = (255, 128, 64, 255)  # ~(1, 0.5, 0.25) linear
    glass = glass.with_texture(Slot.BASE_COLOR, TextureRef(b.add_texture(img, colorspace="linear")))
  b.add_instance("box", size=(1.0, 1.0, 0.2), material=b.add_material(glass), xform=np.c_[np.eye(3), [0.0, 0.0, 0.0]])
  return b.build()


def test_base_texture_tints_transmission_only_under_the_gltf_flag():
  """OpenPBR: a fully transmissive material's tint is `transmission_color` alone, so its
  base-color texture changes nothing (textured clear glass is as clear as untextured glass). glTF
  (`MaterialFlags.TRANSMISSION_TINT`): the textured base color tints transmitted light at each
  of the slab's two faces: green 0.5^2 and blue 0.25^2 of red."""
  c = RES // 2
  window = (slice(c - 2, c + 2), slice(c - 2, c + 2))
  clear = _render(_textured_slab(texture=False))[window]
  textured = _render(_textured_slab(texture=True))[window]
  np.testing.assert_allclose(textured, clear, rtol=1e-4, atol=1e-6)
  tinted = _render(_textured_slab(texture=True, flags=int(rw.scene.MaterialFlags.TRANSMISSION_TINT)))[window].mean(axis=(0, 1))
  np.testing.assert_allclose(tinted[1:] / tinted[0], (0.25, 0.0625), rtol=0.05)
  np.testing.assert_allclose(tinted[0], clear.mean(axis=(0, 1))[0], rtol=0.05)


def test_base_shares_fold_the_texture_to_the_opaque_share():
  """`derive.base_shares` sums to the derived base color; the upload's base-color texel map then
  gives (opaque texel + transmitted) / S: at transmission 0.5 half the texture reaches the base,
  at 1 none (white texel), with the glTF flag all of it (identity map)."""
  b = rw.SceneBuilder()
  tint = int(rw.scene.MaterialFlags.TRANSMISSION_TINT)
  for t, flags in ((0.0, 0), (0.5, 0), (1.0, 0), (1.0, tint)):
    b.add_material(
      rw.Material(
        base_color=(0.8, 0.4, 0.2),
        transmission_weight=t,
        transmission_color=(1.0, 1.0, 0.5),
        ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags(flags)),
      )
    )
  scene = b.build()
  surface = np.zeros(len(scene.materials), bool)
  opaque, transmitted = base_shares(scene.materials, scene.material_extras, surface)
  base = derive(scene.materials, scene.material_extras, LoadReport(), volume=surface).base[:, :3]
  np.testing.assert_allclose(opaque + transmitted, base, rtol=1e-6)
  affine = np.tile(TEX_IDENTITY, (4, NTEXROLE, 1, 1))
  _fold_base_maps(affine, opaque, transmitted)
  texel = np.array([0.5, 0.25, 1.0])
  mapped = affine[:, TexRole.RGB, 0, :3] * texel + affine[:, TexRole.RGB, 1, :3]
  shaded = base * mapped
  np.testing.assert_allclose(shaded[0], np.array((0.8, 0.4, 0.2)) * texel, rtol=1e-6)  # opaque: the texture
  np.testing.assert_allclose(shaded[1], 0.5 * np.array((0.8, 0.4, 0.2)) * texel + 0.5 * np.array((1.0, 1.0, 0.5)), rtol=1e-6)
  np.testing.assert_allclose(shaded[2], (1.0, 1.0, 0.5), rtol=1e-6)  # glass: transmission_color only
  np.testing.assert_allclose(shaded[3], np.array((0.8, 0.4, 0.2)) * texel, rtol=1e-6)  # glTF: the base tints
  np.testing.assert_array_equal(affine[[0, 3]], np.tile(TEX_IDENTITY, (2, NTEXROLE, 1, 1)))
