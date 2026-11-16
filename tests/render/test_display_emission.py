"""Display-relative emission on lit materials (`MaterialFlags.DISPLAY_EMISSION`, BUG-681: glTF's
emissive factor): the emission shows as authored at any exposure (1: the brightest LDR value),
where the same emission as a radiance (1 nit) follows the exposure; the surface stays lit. Unlit
materials (`MaterialFlags.UNLIT`: glTF KHR_materials_unlit, MaterialX surface_unlit) show their
base color the same way, unlit."""

import numpy as np
import warp as wp

import raywarp as rw

_QUAD = ([[-0.4, -0.4, 0], [0.4, -0.4, 0], [0.4, 0.4, 0], [-0.4, 0.4, 0]], [[0, 1, 2], [0, 2, 3]])
_RES = 48


def _scene(sun: float, *, unlit: bool = False) -> rw.Scene:
  """Two emissive quads (1, 0.5, 0) side by side: display-relative (left), 1 nit (right); black
  bases, a sun of illuminance `sun`, a grey floor. `unlit`: the left quad is an unlit material of
  base color (1, 0.5, 0) instead (no emission)."""
  b = rw.SceneBuilder()
  flags = rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.DISPLAY_EMISSION)
  kw = {"base_color": (0.0, 0.0, 0.0), "emission_luminance": 1.0, "emission_color": (1.0, 0.5, 0.0)}
  display = b.add_material(rw.Material(name="display", ext=flags, **kw))
  if unlit:
    ext = rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.UNLIT)
    display = b.add_material(rw.Material(name="unlit", ext=ext, base_color=(1.0, 0.5, 0.0), specular_roughness=0.1))
  nits = b.add_material(rw.Material(name="nits", **kw))
  plain = b.add_material(rw.Material(name="plain", base_color=(0.5, 0.5, 0.5)))
  quad = b.add_mesh(*_QUAD)
  floor = b.add_mesh([[-3, -3, -0.01], [3, -3, -0.01], [3, 3, -0.01], [-3, 3, -0.01]], [[0, 1, 2], [0, 2, 3]])
  b.add_instance("mesh", mesh=floor, material=plain)
  for x, mat in ((-0.45, display), (0.45, nits)):
    b.add_instance("mesh", mesh=quad, material=mat, xform=np.c_[np.eye(3), [x, 0.0, 0.3]].astype(np.float32))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=sun))
  return b.build()


def _patches(sun: float, ev: float, *, unlit: bool = False) -> tuple[np.ndarray, np.ndarray]:
  """(display-relative or unlit quad, 1-nit quad) mean 8-bit rgb of their centres, seen from
  above (`_scene`)."""
  s = rw.RenderSettings().replace(
    **{"exposure.mode": "manual", "exposure.ev100": ev, "post.tone_mapper": "clamp"}, outputs=rw.Outputs.of("rgb")
  )
  r = rw.Renderer(
    rw.upload(_scene(sun, unlit=unlit)), nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(_RES, _RES), settings=s
  )
  # Looking straight down at the quads (pose: position, quaternion x y z w).
  r.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render(None)
  wp.synchronize()
  rgb = r.outputs.rgb
  assert rgb is not None
  img = np.asarray(rgb.numpy()).copy()
  r.close()
  img = (img.view(np.uint8) if img.dtype == np.uint32 else img).reshape(_RES, _RES, -1)[..., :3].astype(np.float64)
  y, h = _RES // 2, 2
  left, right = round(_RES * (0.5 - 0.45 / 2.3)), round(_RES * (0.5 + 0.45 / 2.3))
  return img[y - h : y + h, left - h : left + h].mean((0, 1)), img[y - h : y + h, right - h : right + h].mean((0, 1))


def test_display_relative_emission_shows_as_authored():
  """The flagged quad shows sRGB(1, 0.5, 0) at two exposures and two sun strengths (plus the
  sun's specular off its black base: lit); the 1-nit quad is near black at EV 10 and follows the
  exposure."""
  rows = {(sun, ev): _patches(sun, ev) for sun, ev in ((3.0, 10.0), (30.0, 10.0), (3.0, 6.0))}
  for key, (display, _nits) in rows.items():
    np.testing.assert_allclose(display[:2], [255.0, 188.0], atol=3.0, err_msg=str(key))
    assert display[2] < 12.0, key
  assert rows[3.0, 10.0][1].max() < 8.0
  assert rows[3.0, 6.0][1].max() > rows[3.0, 10.0][1].max() + 4.0


def test_unlit_material_shows_its_base_color():
  """An unlit material (`MaterialFlags.UNLIT`) shows sRGB of its base color (1, 0.5, 0) at two
  exposures and two sun strengths: no light, no specular reaches it (a lit glossy surface would
  show the sun's highlight and follow the exposure)."""
  for sun, ev in ((3.0, 10.0), (30.0, 10.0), (3.0, 6.0)):
    unlit, _nits = _patches(sun, ev, unlit=True)
    np.testing.assert_allclose(unlit, [255.0, 188.0, 0.0], atol=3.0, err_msg=str((sun, ev)))
