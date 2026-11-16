"""The layered lobes through the whole pipeline (import -> upload -> Features -> megakernel), on
the CPU: the anisotropic lobe of a plane seen orthographically against Heitz's closed form,
rotation and sign conventions, compile-time absence, the fuzz layer, the transmissive pane's
specular kept at zero opacity."""

import math

import numpy as np
import pytest

import raywarp as rw

IDENT = (0.0, 0.0, 0.0, 1.0)
LIGHT = np.array([0.35, 0.2, 1.0])
HEIGHT = 2.0  # orthographic view height [m]
RES = 24


def _plane(material: rw.Material, *, light: np.ndarray = LIGHT) -> rw.Scene:
  b = rw.SceneBuilder()
  b.add_instance("box", size=(3.0, 3.0, 0.01), material=b.add_material(material))
  xform = np.c_[np.eye(3), light]
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=1000.0, treat_as_point=True, xform=xform))
  return b.build()


def _render(scene: rw.Scene, **settings: object) -> tuple[np.ndarray, rw.Renderer[None]]:
  rs = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    environment=rw.settings.Environment(enabled=False),
    gi=rw.settings.GI(enabled=False),
    shadows=rw.settings.Shadows(enabled=False),
    lighting=rw.settings.Lights(camera_light=False),
  ).replace(**settings)
  cam = rw.CameraSpec(fovy=HEIGHT, projection="orthographic")
  r = rw.Renderer(rw.upload(scene, "cpu"), nworld=1, cameras=[cam], resolution=(RES, RES), settings=rs)
  r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, *IDENT]]], np.float32))
  r.render()
  hdr = r.outputs.hdr
  assert hdr is not None
  return np.asarray(hdr.numpy()).reshape(RES, RES, -1)[..., :3].astype(np.float64), r


def _metal(anisotropy: float = 0.0, rotation: float = 0.0) -> rw.Material:
  extras = rw.scene.OpenPbrExtras(specular_roughness_anisotropy=anisotropy) if anisotropy else None
  return rw.Material(
    base_color=(1.0, 1.0, 1.0),
    base_metalness=1.0,
    specular_roughness=0.5,
    extras=extras,
    ext=rw.scene.RaywarpExt(anisotropy_rotation=rotation),
  )


def _points() -> np.ndarray:
  """(RES, RES, 3) the plane's points under the pixels (rows top down: +y up)."""
  c = (np.arange(RES) + 0.5) / RES * HEIGHT - HEIGHT / 2
  x, y = np.meshgrid(c, -c)
  return np.stack((x, y, np.full_like(x, 0.01)), -1)


def _ratio(tangent: np.ndarray, anisotropy: float) -> np.ndarray:
  """D V of the anisotropic lobe over the isotropic one (F and the compensation cancel), Heitz."""
  p = _points().reshape(-1, 3)
  l = LIGHT - p
  l /= np.linalg.norm(l, axis=1, keepdims=True)
  v = np.array([0.0, 0.0, 1.0])
  h = l + v
  h /= np.linalg.norm(h, axis=1, keepdims=True)
  t = tangent
  b = np.cross([0.0, 0.0, 1.0], t)
  alpha = 0.25
  bb = 1.0 - anisotropy
  at = alpha * math.sqrt(2.0 / (1.0 + bb * bb))
  ab = at * bb

  def dv(ax: float, ay: float) -> np.ndarray:
    hx, hy, hz = h @ t, h @ b, h[:, 2]
    d = 1.0 / (np.pi * ax * ay * ((hx / ax) ** 2 + (hy / ay) ** 2 + hz**2) ** 2)

    def lam(w: np.ndarray) -> np.ndarray:
      return 0.5 * (-1.0 + np.sqrt(1.0 + ((w @ t * ax) ** 2 + (w @ b * ay) ** 2) / w[:, 2] ** 2))

    vv = np.broadcast_to(v, l.shape)
    return d / (1.0 + lam(vv) + lam(l)) / (4.0 * l[:, 2])

  return (dv(at, ab) / dv(alpha, alpha)).reshape(RES, RES)


def test_anisotropic_plane_matches_closed_form_and_rotation():
  iso, r_iso = _render(_plane(_metal()))
  aniso, r = _render(_plane(_metal(0.8)))
  turned, _ = _render(_plane(_metal(0.8, math.pi / 2)))
  signed, _ = _render(_plane(_metal(0.8, math.pi)))
  assert not r_iso.features.layered and r.features.specular_anisotropy
  got = aniso[..., 0] / iso[..., 0]
  ex, ey = _ratio(np.array([1.0, 0.0, 0.0]), 0.8), _ratio(np.array([0.0, 1.0, 0.0]), 0.8)
  # The box's top face maps u along x or y: the stretch follows dp/du, a quarter turn the other.
  first, second = (ex, ey) if np.abs(got - ex).max() < np.abs(got - ey).max() else (ey, ex)
  np.testing.assert_allclose(got, first, rtol=2e-3)
  np.testing.assert_allclose(turned[..., 0] / iso[..., 0], second, rtol=2e-3)
  # A half turn is the same stretch (the lobe is symmetric under t -> -t).
  np.testing.assert_allclose(signed, aniso, rtol=1e-4)


def test_negative_anisotropy_is_a_quarter_turn():
  turned, _ = _render(_plane(_metal(0.8, math.pi / 2)))
  # Per world: a per-world anisotropy of -0.8 (Filament's signed convention) equals rotation pi/2.
  rs_scene = _plane(_metal(0.8))
  rs = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    environment=rw.settings.Environment(enabled=False),
    gi=rw.settings.GI(enabled=False),
    shadows=rw.settings.Shadows(enabled=False),
    lighting=rw.settings.Lights(camera_light=False),
  )
  cam = rw.CameraSpec(fovy=HEIGHT, projection="orthographic")
  r = rw.Renderer(
    rw.upload(rs_scene, "cpu"),
    nworld=2,
    cameras=[cam],
    resolution=(RES, RES),
    settings=rs,
    per_world=rw.PerWorld(material=["specular_anisotropy"]),
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 3.0, *IDENT]]] * 2, np.float32))
  r.per_world["material.specular_anisotropy"].assign(np.array([[[0.8, 0.0]], [[-0.8, 0.0]]], np.float32))
  r.render()
  hdr = r.outputs.hdr
  assert hdr is not None
  out = np.asarray(hdr.numpy()).reshape(2, RES, RES, -1)[..., :3]
  np.testing.assert_allclose(out[1], turned, rtol=1e-4, atol=1e-7)
  assert not np.allclose(out[0], out[1], rtol=1e-2)


def test_fuzz_adds_a_grazing_sheen_and_scales_the_base():
  base = {"base_color": (0.2, 0.2, 0.2), "specular_weight": 0.0}
  plain, _ = _render(_plane(rw.Material(**base)))
  fuzz, r = _render(_plane(rw.Material(**base, extras=rw.scene.OpenPbrExtras(fuzz_weight=1.0, fuzz_color=(1.0, 0.0, 0.0)))))
  assert r.features.fuzz
  # Green / blue: the base under the fuzz only, scaled by 1 - E_fuzz (darker); red: + the sheen.
  assert np.all(fuzz[..., 1] < plain[..., 1]) and np.all(fuzz[..., 1] > 0.3 * plain[..., 1])
  assert np.all(fuzz[..., 0] > fuzz[..., 1])


def test_transmissive_pane_keeps_its_specular():
  glass = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=0.3, transmission_weight=1.0)
  pane, r = _render(_plane(glass), **{"transparency.enabled": True})
  assert r.features.transmissive
  # Nothing behind the pane: what shows is its reflection, unscaled by its opacity (2^-10): as
  # bright as the same dielectric's opaque specular-only highlight (no diffuse: base color black).
  opaque, _ = _render(_plane(rw.Material(base_color=(0.0, 0.0, 0.0), specular_roughness=0.3)))
  np.testing.assert_allclose(pane.max(), opaque.max(), rtol=0.05)


@pytest.mark.parametrize("anisotropy", [0.0])
def test_absent_lobes_compile_out(anisotropy: float):
  _out, r = _render(_plane(_metal(anisotropy)))
  f = r.features
  assert not (f.layered or f.fuzz or f.coat_layer or f.coat_normal or f.detail_normal or f.horizon_fade)


def test_transmissive_pane_tints_what_is_behind_it():
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_weight=0.0))
  b.add_instance("box", size=(3.0, 3.0, 0.01), material=floor)
  blue = rw.Material(transmission_weight=1.0, transmission_color=(0.2, 0.4, 1.0), specular_roughness=0.3)
  pane = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.5))
  b.add_instance("box", size=(0.5, 3.0, 0.01), material=b.add_material(blue), xform=pane)
  xform = np.c_[np.eye(3), [0.0, 0.0, 2.0]]
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=1000.0, treat_as_point=True, xform=xform))
  # A solid pane (a box of glass) is two layers, its entry and its exit face: one more reaches the
  # floor behind it.
  out, r = _render(b.build(), **{"transparency.enabled": True, "transparency.max_layers": 3})
  assert r.features.transmissive
  seen = out[1, RES // 2]  # the floor through the pane, away from its highlight
  bare = out[1, 1]  # the floor beside it
  assert seen[2] > seen[0] * 2.0 and bare[2] == pytest.approx(bare[0], rel=1e-3)


def test_emission_output_is_the_emission_under_the_coat():
  """BUG-060: the emission output of a coated emitter is the emission seen through the coat (the
  same term the image adds), not the bare emission: a red coat over a white emitter keeps the red
  and removes most of the green and blue from both."""
  emitter = rw.Material(
    base_color=(0.0, 0.0, 0.0),
    specular_weight=0.0,
    emission_luminance=100.0,
    coat_weight=1.0,
    coat_roughness=0.5,
    extras=rw.scene.OpenPbrExtras(coat_color=(1.0, 0.05, 0.05)),
  )
  _hdr, r = _render(_plane(emitter), outputs=rw.Outputs.of("hdr", "emission"))
  assert r.features.coat_layer
  out = r.outputs.emission
  assert out is not None
  emission = np.asarray(out.numpy()).reshape(RES, RES, -1)[RES // 2, RES // 2, :3]
  assert emission[1] < 0.5 * emission[0] and emission[2] < 0.5 * emission[0]
