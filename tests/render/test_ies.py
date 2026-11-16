"""IES LM-63 profiles: the parser, the gridded profile and its flux, the upload's scale per mode,
and the rendered irradiance of IES lights against the analytic one (CPU, software traversal)."""

from __future__ import annotations

import dataclasses
import math
from typing import TypedDict, Unpack

import numpy as np
import pytest
from conftest import DATA, required

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.pipeline.upload.lights import derive_lights, ies_lights, light_arrays
from raywarp.core.shade.lights import ies as ies_lib
from raywarp.scene.lighting import ies_profile
from raywarp.scene.lighting.rig import add_lights
from raywarp.scene.model import ies
from raywarp.scene.model.lights import IesMode, Light, Lights, Shaping
from raywarp.scene.model.types import LightFlags, LightKind
from raywarp.settings.fields import OutputKind

IES = DATA / "ies"
FILES = sorted(p.name for p in IES.glob("*.ies"))
H = 1.5  # light height above the floor [m]
RES, FOV, CAM_H = 24, 90.0, 2.0


def _text(name: str) -> str:
  """A fixture's checked text."""
  return ies.read(str(IES / name))


def test_parser_formats_types_and_factors():
  """Formats, photometric types, keywords, units, ballast factors and the tilt block."""
  formats = {name: ies.parse(_text(name)) for name in FILES}
  assert formats["downlight_1995.ies"].format == "LM-63-1995"
  assert formats["tilt_1986.ies"].format == "LM-63-1986"
  assert formats["bilateral_2019.ies"].format == "LM-63-2019"
  assert formats["flood_b_1995.ies"].photometric_type == ies.PhotometricType.B
  assert formats["auto_a_2002.ies"].photometric_type == ies.PhotometricType.A
  assert formats["downlight_1995.ies"].keywords["TEST"] == "downlight"
  # 1986 / 1995: the ballast-lamp factor multiplies; 2002 / 2019: "future use" does not.
  tilt = formats["tilt_1986.ies"]
  assert tilt.ballast_lamp_factor == 0.95 and tilt.opening[0] == pytest.approx(ies.FEET)
  assert tilt.tilt is not None and tilt.tilt.factor(45.0) == pytest.approx(0.9)
  assert formats["bilateral_2019.ies"].ballast_lamp_factor == 1.0
  asym = formats["asym_2002.ies"]
  assert asym.lumens_per_lamp < 0.0  # absolute photometry
  np.testing.assert_allclose(asym.candela, asym.values * 2.0 * 0.5)


@pytest.mark.parametrize(
  "text,match",
  [
    ("IESNA:LM-63-2002\n[TEST] x\n1 1 1 2 1 1 2 0 0 0\n", "TILT"),
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5\n", "numbers"),
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 1 2 1 7 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "photometric type"),
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 1 2 1 1 2 0 0 0 1 1 1\n90 0\n0\n5 5\n", "increase"),
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 -1\n", ">= 0"),
    ("IESNA:LM-63-2002\nTILT=lamp.tlt\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "inlined"),
    # Non-finite or impossible header / tilt values: named, not a crash or a NaN profile.
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 1 inf 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "finite"),
    ("IESNA:LM-63-2002\nTILT=NONE\n1 1 -1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "multiplier"),
    ("IESNA:LM-63-1995\nTILT=NONE\n1 1 1 2 1 1 2 0 0 0 1 -1 1\n0 90\n0\n5 5\n", "ballast"),
    ("IESNA:LM-63-2002\nTILT=INCLUDE\n1\n2\n0 90\nnan 1\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "finite"),
    ("IESNA:LM-63-2002\nTILT=INCLUDE\n1\n1.5\n0 90\n1 1\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n", "integer"),
  ],
)
def test_parser_rejects_malformed_files(text, match):
  """Malformed files fail with a message naming the problem."""
  with pytest.raises(ies.IesError, match=match):
    ies.parse(text)


def test_tilt_file_is_inlined(tmp_path):
  """TILT=<file>: the tilt block is copied into the stored text (self-contained profile)."""
  (tmp_path / "lamp.tlt").write_text("1\n2\n0 90\n1.0 0.5\n")
  body = "IESNA:LM-63-2002\nTILT=lamp.tlt\n1 1 1 2 1 1 2 0 0 0 1 1 1\n0 90\n0\n5 5\n"
  (tmp_path / "f.ies").write_text(body)
  data = ies.parse(ies.read(str(tmp_path / "f.ies")))
  assert data.tilt is not None and data.tilt.factor(90.0) == 0.5


@pytest.mark.parametrize("name", FILES)
def test_grid_flux_is_the_interpolant_integral(name):
  """The analytic flux of the gridded profile equals a fine numerical sphere quadrature."""
  g = ies_profile.profile(_text(name))
  dirs, dw = ies_profile.sphere_directions(1024, 1024)
  assert g.flux == pytest.approx(g.peak * float((ies_profile.sample(g, dirs) * dw).sum()), rel=2e-4)
  assert g.exact


def test_grid_symmetries_reproduce_the_file():
  """Mirrored ranges read the file's values: quadrant (C 0-90), bilateral (C 90-270), type B
  lateral / vertical mirrors; 1D for an axially symmetric file."""
  down = ies_profile.profile(_text("downlight_1995.ies"))
  assert down.nb == 1
  data = ies.parse(_text("wallwash_2002.ies"))
  g = ies_profile.profile(_text("wallwash_2002.ies"))
  for h, phi in ((22.5, 22.5), (22.5, 157.5), (22.5, 202.5), (67.5, 292.5)):
    row = data.candela[list(data.horizontal).index(h)]
    t = math.radians(30.0)
    p = math.radians(phi)
    d = np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), -math.cos(t)]])
    assert ies_profile.sample(g, d)[0] * g.peak == pytest.approx(row[list(data.vertical).index(30.0)], rel=1e-5)
  bil = ies.parse(_text("bilateral_2019.ies"))
  gb = ies_profile.profile(_text("bilateral_2019.ies"))
  t = math.radians(30.0)
  for phi, h in ((45.0, 135.0), (-45.0, 225.0)):  # value(phi) = value(180 - phi)
    p = math.radians(phi)
    d = np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), -math.cos(t)]])
    expected = bil.candela[list(bil.horizontal).index(h)][list(bil.vertical).index(30.0)]
    assert ies_profile.sample(gb, d)[0] * gb.peak == pytest.approx(expected, rel=1e-5)
  flood = ies_profile.profile(_text("flood_b_1995.ies"))
  # Type B: V = asin(x), H = atan2(y, -z); lateral mirror H -> -H, behind: dark.
  v, h = math.radians(20.0), math.radians(30.0)
  d = lambda sv, sh: np.array([[math.sin(sv * v), math.cos(v) * math.sin(sh * h), -math.cos(v) * math.cos(h)]])  # noqa: E731
  assert ies_profile.sample(flood, d(1, 1))[0] == pytest.approx(ies_profile.sample(flood, d(1, -1))[0], rel=1e-6)
  assert ies_profile.sample(flood, np.array([[0.0, 0.0, 1.0]]))[0] == 0.0


def test_inexact_grid_resamples_at_the_smallest_step():
  """Angles without a common step are resampled (flagged inexact), knots still within 1 %."""
  v = [0.0, 1.33, 7.17, 30.0, 61.77, 90.0]
  cd = [100.0, 98.0, 90.0, 60.0, 10.0, 0.0]
  text = (
    "IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 6 1 1 2 0 0 0 1 1 1\n"
    + " ".join(map(str, v))
    + "\n0\n"
    + " ".join(map(str, cd))
    + "\n"
  )
  g = ies_profile.grid(ies.parse(text))
  assert not g.exact
  for angle, value in zip(v, cd, strict=True):
    t = math.radians(angle)
    assert ies_profile.sample(g, np.array([[math.sin(t), 0.0, -math.cos(t)]]))[0] * g.peak == pytest.approx(value, abs=1.0)


class _IesShaping(TypedDict, total=False):
  ies_normalize: bool
  ies_mode: IesMode


def _record(name: str, **shaping: Unpack[_IesShaping]) -> Light:
  """A point light with fixture `name`."""
  return Light(
    kind=LightKind.SPHERE,
    radius=0.0,
    treat_as_point=True,
    intensity=2.0,
    shaping=Shaping(cone_angle=180.0, ies=_text(name), **shaping),
  )


@pytest.mark.parametrize(
  "shaping,expected",
  [
    ({}, lambda g: 2.0),  # mask: the light's intensity is the peak
    ({"ies_normalize": True}, lambda g: 2.0 * g.peak / g.mean),  # the unshaped flux kept
    ({"ies_mode": "measured"}, lambda g: 2.0 * g.peak),  # the file's candela x intensity
  ],
)
def test_upload_scales_per_mode(shaping, expected):
  """Point intensity (the profile's peak direction) per mode; the profile marks the light shaped
  and punctual, a POINT light (no cone), with its profile index."""
  derived = derive_lights(Lights.from_records([_record("downlight_1995.ies", **shaping)], bodies=[-1]))
  g = ies_profile.profile(_text("downlight_1995.ies"))
  assert float(derived.intensity[0, 0]) == pytest.approx(expected(g), rel=1e-6)
  assert derived.flags[0] & LightFlags.SHAPED
  slots, info = light_arrays(derived)
  assert info.has_ies and int(slots["light.type"][0, 0]) == int(rw._base.types.LightType.POINT)
  assert float(slots["light.strength"][0, 0]) == pytest.approx(expected(g), rel=1e-6)
  table = ies_lights(derived)
  assert table.light_profile.tolist() == [0] and len(table.grids) == 1


@pytest.mark.parametrize("mode,normalize", [("mask", False), ("measured", False), ("measured", True)])
def test_bulb_radius_keeps_the_point_intensity(mode, normalize):
  """`Light.point(intensity=1, radius=r, ies=...)` has the point intensity of the same light with
  radius 0 in every IES mode: "measured" gives the file's candela, not candela / (pi r^2)."""
  peak = []
  for radius in (0.0, 0.1):
    ies_ref = rw.scene.Ies(IES / "downlight_1995.ies", mode=mode, normalize=normalize)
    light = Light.point((0.0, 0.0, H), intensity=1.0, radius=radius, ies=ies_ref)
    derived = derive_lights(Lights.from_records([light], bodies=[-1]))
    peak.append(float(derived.intensity[0, 0]))
  assert peak[1] == pytest.approx(peak[0], rel=1e-6)
  if mode == "measured" and not normalize:
    assert peak[0] == pytest.approx(ies_profile.profile(_text("downlight_1995.ies")).peak, rel=1e-6)


def test_profiles_are_shared_and_tilt_follows_the_light():
  """Lights with one profile share one grid and instance; the TILT factor is evaluated per frame
  at each light's orientation (`light_pack.pack`): straight down 1, the nadir horizontal 0.7, and a
  light turned at runtime (per-world light.dir) follows."""
  down = _record("tilt_1986.ies")
  side = np.c_[np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]), np.zeros(3)].astype(np.float32)
  tilted = dataclasses.replace(down, xform=side)
  derived = derive_lights(Lights.from_records([down, down, tilted], bodies=[-1, -1, -1]))
  table = ies_lights(derived)
  assert table.light_profile.tolist() == [0, 0, 0] and len(table.grids) == 1 and len(table.instances) == 1
  b = rw.SceneBuilder()
  b.add_instance("plane", material=b.add_material(rw.Material()), size=(5.0, 5.0, 0.0))
  for rec in (down, tilted):
    b.add_light(rec)
  settings = rw.RenderSettings().replace(**{"performance.traversal": "software", "outputs": rw.Outputs.of("rgb")})
  scene = rw.upload(b.build(), device="cpu")
  cams = [rw.CameraSpec()]
  with rw.Renderer(
    scene, nworld=2, cameras=cams, resolution=(4, 4), settings=settings, per_world=rw.PerWorld(light=["dir"])
  ) as r:
    pack = r._tables.light_pack
    r.render()
    strength = record_field(pack.lights.numpy(), "intensity")
    assert strength[0, 1] == pytest.approx(0.7 * strength[0, 0], rel=1e-5)
    d = r.per_world["light.dir"]
    dirs = d.numpy()
    dirs[1, 0] = (0.0, -1.0, 0.0)  # world 1 turns the straight light horizontal
    d.assign(dirs)
    r.render()
    strength = record_field(pack.lights.numpy(), "intensity")
    assert strength[1, 0] == pytest.approx(0.7 * strength[0, 0], rel=1e-5)


def _render(specs, *, extra=None, nworld=1, per_world=None, outputs: tuple[OutputKind, ...] = ("rgb", "hdr")):
  """HDR of a 50 % Lambertian floor (z = 0) seen from (0, 0, CAM_H) looking down."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_weight=0.0, specular_roughness=1.0))
  b.add_instance("plane", material=m, size=(20.0, 20.0, 0.0))
  for spec in specs:
    if isinstance(spec, Light):
      b.add_light(spec)
  add_lights(b, [x for x in specs if not isinstance(x, Light)], center=np.zeros(3), extent=1.0, exposure=1.0)
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of(*outputs),
      "performance.traversal": "software",
      "post.tone_mapper": "clamp",
      "environment.enabled": False,
      "gi.enabled": False,
      "lighting.light_grid": False,
      "lighting.emissive_lights": False,
      "shadows.enabled": False,
      "reflections.enabled": False,
      **(extra or {}),
    }
  )
  scene = rw.upload(b.build(), device="cpu")
  r = rw.Renderer(
    scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=FOV)], resolution=(RES, RES), settings=settings, per_world=per_world
  )
  r.camera_pose.assign(np.tile(np.array([[[0.0, 0.0, CAM_H, 0.0, 0.0, 0.0, 1.0]]], np.float32), (nworld, 1, 1)))
  return r


def _floor_dirs() -> tuple[np.ndarray, np.ndarray]:
  """(light-to-floor unit directions (RES * RES, 3), distances) of the floor pixels from a light
  at (0, 0, H) (image row 0 is +y)."""
  t = math.tan(math.radians(FOV) / 2)
  c = (np.arange(RES) + 0.5) / RES * 2 - 1
  x, y = np.meshgrid(c * t * CAM_H, -c * t * CAM_H)
  d = np.stack([x, y, -H * np.ones_like(x)], -1).reshape(-1, 3)
  r = np.linalg.norm(d, axis=-1)
  return d / r[:, None], r


def _octa_lookup(
  octa: np.ndarray[[int], np.dtype[np.float32]], dirs: np.ndarray[[int, int], np.dtype[np.float64]]
) -> np.ndarray[[int], np.dtype[np.float64]]:
  """`ies.factor`'s read of map `octa` (OCTA^2) toward `dirs` (m, 3) in its frame (x, y, cos
  from the nadir), on the host: the oracle of the device lookup."""
  v = np.asarray(dirs, np.float64)
  p = v[:, :2] / np.maximum(np.abs(v).sum(axis=1, keepdims=True), 1.0e-20)
  sx, sy = np.where(p[:, 0] >= 0.0, 1.0, -1.0), np.where(p[:, 1] >= 0.0, 1.0, -1.0)
  folded = np.stack([(1.0 - np.abs(p[:, 1])) * sx, (1.0 - np.abs(p[:, 0])) * sy], axis=1)
  uv = np.where(v[:, 2:3] < 0.0, folded, p) * 0.5 + 0.5
  xy = np.clip(uv * ies_lib.OCTA - 0.5, 0.0, ies_lib.OCTA - 1)
  ij = np.minimum(xy.astype(np.int64), ies_lib.OCTA - 2)
  t = xy - ij
  grid = np.asarray(octa, np.float64).reshape(ies_lib.OCTA, ies_lib.OCTA)
  i, j = ij[:, 0], ij[:, 1]
  v0 = grid[j, i] * (1.0 - t[:, 0]) + grid[j, i + 1] * t[:, 0]
  v1 = grid[j + 1, i] * (1.0 - t[:, 0]) + grid[j + 1, i + 1] * t[:, 0]
  return v0 * (1.0 - t[:, 1]) + v1 * t[:, 1]


def _profile(g: ies_profile.IesGrid, dirs: np.ndarray, scale: float) -> np.ndarray:
  """The device lookup's oracle toward light-local `dirs` (nadir -Z): `_octa_lookup` of the
  profile's octahedral map (its frame: z = cos from the nadir)."""
  return _octa_lookup(ies_lib.octa_profile(g, scale), np.c_[dirs[:, 0], dirs[:, 1], -dirs[:, 2]])


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("scale", [0.0, 0.7])
def test_octahedral_map_follows_the_profile(name, scale):
  """The baked map (OCTA^2) resamples the photometric grid: over the sphere within 0.2 % of the peak
  on average and 2 % at the 99th percentile (the fixtures: mean <= 0.04 %, p99 <= 0.5 %, max 11 % at
  a sharp cut-off, blurred over about one texel)."""
  g = ies_profile.profile(_text(name))
  rng = np.random.default_rng(7)
  d = rng.normal(size=(20000, 3))
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  exact = ies_profile.sample(g, d, scale)
  err = np.abs(_profile(g, d, scale) - exact)
  assert err.mean() < 0.002, err.mean()
  assert np.quantile(err, 0.99) < 0.02, np.quantile(err, 0.99)


@pytest.mark.parametrize("name", FILES)
def test_rendered_irradiance_matches_the_profile(name):
  """Measured mode, every fixture (2D, A / B, angle scales): floor radiance = 0.5 / pi x I(dir)
  x cos / d^2, I from the oracle (`ies_profile.sample`): inverse square, angular cuts and the
  profile's orientation (local +X along world +x)."""
  g = ies_profile.profile(_text(name))
  dirs, dist = _floor_dirs()
  for scale in (0.0, 0.7) if name == "asym_2002.ies" else (0.0,):
    spec = rw.Light.point(
      (0.0, 0.0, H), intensity=1.0, ies=rw.scene.Ies(IES / name, mode="measured", axis=(1.0, 0.0, 0.0), angle_scale=scale)
    )
    with _render([spec]) as r:
      hdr = required(r.render().hdr).numpy()[0, 0].mean(-1).reshape(-1)
    expected = 0.5 / math.pi * g.peak * _profile(g, dirs, scale) * (H / dist) / dist**2
    tilt = ies.parse(_text(name)).tilt
    expected *= 1.0 if tilt is None else tilt.factor(0.0)
    np.testing.assert_allclose(hdr, expected, rtol=5e-3, atol=1e-4 * float(expected.max()))


def test_cone_profile_capped_estimate_and_per_world_strength():
  """A spot cone multiplies the profile; with lighting.max_exact the capped estimate ranks by the
  profiled light (a dark-direction profile does not crowd out a weaker light); per-world
  light.strength scales the profiled light."""
  spot = rw.Light.spot(
    (0.0, 0.0, H), (0.0, 0.0, -1.0), cone_angle=30.0, intensity=10.0, ies=rw.scene.Ies(IES / "downlight_1995.ies")
  )
  plain = rw.Light.spot((0.0, 0.0, H), (0.0, 0.0, -1.0), cone_angle=30.0, intensity=10.0)
  with _render([spot]) as r:
    a = required(r.render().hdr).numpy()[0, 0].mean(-1).reshape(-1)
  with _render([plain]) as r:
    b = required(r.render().hdr).numpy()[0, 0].mean(-1).reshape(-1)
  g = ies_profile.profile(_text("downlight_1995.ies"))
  dirs, _ = _floor_dirs()
  np.testing.assert_allclose(a, b * _profile(g, dirs, 0.0), rtol=5e-3, atol=1e-6)
  with _render([spot], nworld=2, per_world=rw.PerWorld(light=["strength"])) as r:
    strength = r.per_world["light.strength"]
    s = strength.numpy()
    s[1] *= 3.0
    strength.assign(s)
    hdr = required(r.render().hdr).numpy()[:, 0].mean(-1)
  np.testing.assert_allclose(hdr[1], 3.0 * hdr[0], rtol=1e-5)
  # A downlight aimed up above the floor: no light on it; a capped loop picks the plain light.
  up = rw.Light.point((0.0, 0.0, H), intensity=100.0, ies=rw.scene.Ies(IES / "downlight_1995.ies", direction=(0.0, 0.0, 1.0)))
  weak = rw.Light.point((0.5, 0.0, H), intensity=1.0)
  with _render([up, weak], extra={"lighting.max_exact": 1}) as r:
    capped = required(r.render().hdr).numpy()[0, 0].mean(-1)
  with _render([weak]) as r:
    alone = required(r.render().hdr).numpy()[0, 0].mean(-1)
  np.testing.assert_allclose(capped, alone, rtol=1e-4, atol=1e-7)


def test_no_ies_compiles_the_feature_out():
  """A scene without a profile resolves Features.ies False (the kernels read no profile)."""
  with _render([rw.Light.point((0.0, 0.0, H), intensity=1.0)]) as r:
    assert not r.features.ies
  with _render([rw.Light.point((0.0, 0.0, H), intensity=1.0, ies=rw.scene.Ies(IES / "downlight_1995.ies"))]) as r:
    assert r.features.ies


def _profile_text(kind: int, v: list[float], h: list[float], cd: list[list[float]]) -> str:
  """An LM-63 file of photometric type `kind` with angles `v`, `h` and candela rows `cd`."""
  head = f"IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 {len(v)} {len(h)} {kind} 2 0 0 0 1 1 1\n"
  return (
    head + " ".join(map(str, v)) + "\n" + " ".join(map(str, h)) + "\n" + "\n".join(" ".join(map(str, r)) for r in cd) + "\n"
  )


@pytest.mark.parametrize(
  "kind,v,h",
  [(1, [0.0], [0.0, 90.0]), (2, [0.0, 45.0, 90.0], [0.0])],  # one vertical angle; a type B fan of one horizontal angle
)
def test_profile_covering_no_solid_angle_raises(kind, v, h):
  """A profile whose grid spans no solid angle has flux 0 (normalising divides by it): it raises."""
  cd = [[100.0] * len(v) for _ in h]
  with pytest.raises(ValueError, match="no solid angle"):
    ies_profile.grid(ies.parse(_profile_text(kind, v, h, cd)))


def test_inexact_periodic_grid_keeps_its_angles():
  """An irregular type C horizontal grid whose smallest step (7.3 deg) does not divide 360 is
  resampled on a grid ending at 360, so sample k sits at angle k x db: a profile linear in the
  horizontal angle reads its value at 270 deg (a grid at the 7.3 deg step, looked up at
  db = 360 / nb, read 1.7 deg off)."""
  h = [0.0, 7.3, 20.0, 90.0, 180.0, 270.0, 350.0]
  g = ies_profile.grid(ies.parse(_profile_text(1, [0.0, 90.0], h, [[x + 100.0] * 2 for x in h])))
  assert not g.exact
  for angle in (90.0, 180.0, 270.0):
    b = math.radians(angle)
    # Type C: horizontal angle b about the nadir axis, at vertical angle 90 (the horizon).
    d = np.array([[math.cos(b), math.sin(b), 0.0]])
    assert ies_profile.sample(g, d)[0] * g.peak == pytest.approx(angle + 100.0, rel=1e-4)
