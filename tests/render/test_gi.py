"""Irradiance probe GI (gi.enabled, core/shade/gi/)."""

import dataclasses
from typing import Literal

import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
from raywarp._base.named import construct_named
from raywarp.core.kernels import ambient, probe
from raywarp.core.kernels.abi import rays as trace_lib
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import ProbeFeatures
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.generic import concrete
from raywarp.core.pipeline.platform import tuning
from raywarp.core.shade import gi
from raywarp.core.shade.gi import cube
from raywarp.core.shade.materials import octahedral
from raywarp.core.trace.accel import query as query_lib
from raywarp.core.trace.software_backend import SoftwareBackend


def test_distance_moments_of_a_large_volume_stay_finite():
  """Distance moments are stored in float16 in units of the volume's max distance: in metres a
  mean distance of 300 m squared (9e4) overflowed float16 (65504) and the Chebyshev visibility of
  probe spacings above ~100 m was NaN."""
  from raywarp.core.shade.gi import kernels as gi_kernels
  from raywarp.core.shade.gi import probes

  @wp.kernel
  def read(vol: probes.Volume, d: wp.array[wp.vec3], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = probes._sample_distance(vol, 0, d[i])

  vol = gi.ProbeVolume(np.zeros(3), np.ones(3), spacing=1.0, cell=0.25, settings=_ONE_PROBE, scale=1.0, device=wp.get_device())
  struct = vol.struct
  struct.max_distance = 400.0
  ntexel = struct.distance.shape[1]
  sums = wp.array(np.tile(np.array([300.0, 300.0**2, 1.0], np.float32), (1, ntexel, 1)), dtype=wp.vec3)
  wp.launch(gi_kernels.resolve_static, dim=(1, ntexel), inputs=[sums], outputs=[struct])
  vol.distance_atlas.refresh(struct.distance)
  out = wp.zeros(1, dtype=wp.vec2)
  wp.launch(read, dim=1, inputs=[struct, wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3)], outputs=[out])
  np.testing.assert_allclose(out.numpy()[0], [300.0, 9.0e4], rtol=2e-3)


def _room(window: bool, glow_strength: float = 100.0, glass: bool = False, coat: float = 0.0, block: bool = False) -> rw.Scene:
  """A closed 2 x 2 x 2 m box room (walls 0.1 m) with an emissive ceiling panel; `window`: a hole
  in the +y wall, under a constant sky; `glass`: a translucent pane (a thin box) in the hole;
  `coat`: the walls' clearcoat weight; `block`: a 0.6 m white block on the floor (instance 0)."""
  b = rw.SceneBuilder()
  if block:
    white = b.add_material(
      rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=0.0, specular_weight=0.0, specular_roughness=1.0)
    )
    b.add_instance("box", size=(0.3, 0.3, 0.3), xform=np.c_[np.eye(3), [0.0, 0.0, 0.3]], material=white)
  if glass:
    pane = b.add_material(
      rw.Material(
        base_color=(1.0, 1.0, 1.0),
        specular_roughness=0.1,
        geometry_opacity=0.15,
        ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
      )
    )
    b.add_instance("box", size=(0.4, 0.02, 0.4), xform=np.c_[np.eye(3), [0.0, 1.05, 1.2]], material=pane)
  wall = b.add_material(
    rw.Material(
      base_color=(0.7, 0.7, 0.7),
      base_metalness=0.0,
      specular_weight=0.0,
      specular_roughness=1.0,
      coat_weight=coat,
      coat_roughness=0.1,
      coat_ior=1.5,
      ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.COAT_GLTF),
    )
  )
  glow = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      base_metalness=0.0,
      specular_roughness=1.0,
      emission_luminance=glow_strength,
      emission_color=(1.0, 1.0, 1.0),
    )
  )
  boxes = [
    ((-1.1, -1.1, -0.1), (1.1, 1.1, 0.0)),
    ((-1.1, -1.1, 2.0), (1.1, 1.1, 2.1)),
    ((-1.1, -1.0, 0.0), (-1.0, 1.0, 2.0)),
    ((1.0, -1.0, 0.0), (1.1, 1.0, 2.0)),
    ((-1.1, -1.1, 0.0), (1.1, -1.0, 2.0)),
  ]
  if window:
    boxes += [((-1.1, 1.0, 0.0), (-0.4, 1.1, 2.0)), ((0.4, 1.0, 0.0), (1.1, 1.1, 2.0)), ((-0.4, 1.0, 0.0), (0.4, 1.1, 0.8))]
    boxes += [((-0.4, 1.0, 1.6), (0.4, 1.1, 2.0))]
    b.add_environment(rw.scene.EnvironmentLight(intensity=50.0))
  else:
    boxes += [((-1.1, 1.0, 0.0), (1.1, 1.1, 2.0))]
    b.add_instance("box", size=(0.3, 0.3, 0.01), xform=np.c_[np.eye(3), [0.0, 0.0, 1.98]], material=glow)
  for lo, hi in boxes:
    lo, hi = np.asarray(lo), np.asarray(hi)
    b.add_instance("box", size=tuple((hi - lo) / 2), xform=np.c_[np.eye(3), (lo + hi) / 2], material=wall)
  return b.build()


def _render(
  host: rw.Scene,
  transparency: rw.settings.Transparency | None = None,
  environment: rw.settings.Environment | None = None,
  **gi_kw: object,
) -> tuple[np.ndarray, rw.Renderer[None]]:
  """HDR of the two worlds, `gi_kw`: `rw.settings.GI` fields over a small probe budget (GI off
  unless `enabled`)."""
  scene = rw.upload(host)
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    gi=rw.settings.GI(enabled=False, spacing=0.5, max_probes=64, rays=32, bake_updates=4),
    transparency=transparency or rw.settings.Transparency(),
    environment=environment or rw.settings.Environment(),
    # The panel stays plain emissive geometry (its upload-time rect light off): GI carries it.
    lighting=rw.settings.Lights(emissive_lights=False),
  ).replace(**{f"gi.{k}": v for k, v in gi_kw.items()})
  r = rw.Renderer(scene, nworld=2, cameras=[rw.CameraSpec(fovy=90.0)], resolution=(12, 12), settings=settings)
  # Camera at the room's centre looking down at the floor (camera -z = world -z).
  r.camera_pose.assign(np.tile(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 1, 1)))
  r.render()
  hdr = r.outputs.hdr
  assert hdr is not None
  return hdr.numpy(), r


def test_settings_validation():
  with pytest.raises(TypeError, match="gi must be a GI"):
    rw.RenderSettings(gi="probes")  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(ValueError, match=r"gi\.rays"):
    rw.settings.GI(rays=0)
  with pytest.raises(ValueError, match=r"gi\.bounds"):
    rw.settings.GI(bounds=((0, 0, 0), (1, 0, 1)))


def test_emissive_panel_lights_the_floor_through_probes():
  host = _room(window=False)
  off, r_off = _render(host)
  on, r = _render(host, enabled=True)
  assert r_off.debug.gi_volume is None and not r_off.features.gi
  vol = r.debug.gi_volume
  assert vol is not None and vol.nprobe <= 64 and int(vol.struct.active.numpy().sum()) > 0
  # No lights: without GI the floor is black; the probes bring the panel's light to it.
  assert off.max() == 0.0
  assert on.min() > 0.0 and np.isfinite(on).all()
  # One volume shared by the worlds.
  np.testing.assert_array_equal(on[0], on[1])


def test_sky_reaches_the_room_through_the_window_only():
  host = _room(window=True)
  off, _ = _render(host, environment=rw.settings.Environment())
  on, r = _render(host, enabled=True, rays=128, bake_updates=8)
  # Split-sum IBL lights the floor with the unoccluded sky (radiance 50 x albedo); the probes
  # only see the sky through the window.
  assert on.mean() < 0.5 * off.mean()
  assert on.mean() > 0.0
  vol = r.debug.gi_volume
  assert vol is not None
  # The probes inside the room (the volume's padding puts others outside, under the open sky).
  c = np.indices(vol.counts[::-1]).reshape(3, -1)[::-1].T
  pos = np.asarray(vol.struct.origin) + c * vol.spacing + vol.struct.offset.numpy()
  inside = np.all(np.abs(pos[:, :2]) < 1.0, axis=1) & (pos[:, 2] > 0.0) & (pos[:, 2] < 2.0)
  sky = vol.struct.irradiance.numpy()[inside, :, 3].astype(np.float32)
  assert inside.any() and 0.0 < sky.mean() < 0.5


def test_probe_hits_use_the_clearcoat_lobe():
  """B34: probe hits shade with the camera rays' lobes. A clearcoat on the walls reflects part
  of the bounce light specularly and the base receives the rest (1 - coat albedo): the probes'
  diffuse bounce drops. Probe shading used to ignore the coat (and the specular curve)."""
  irradiance = []
  for coat in (0.0, 1.0):
    _hdr, r = _render(_room(window=False, coat=coat), enabled=True, bake_updates=8)
    assert r.features.clearcoat == (coat > 0.0)
    assert r.debug.gi_volume is not None
    irradiance.append(float(r.debug.gi_volume.struct.irradiance.numpy()[..., :3].astype(np.float32).mean()))
  assert irradiance[1] < 0.97 * irradiance[0], irradiance


def test_probe_rays_cross_a_glass_pane_once():
  """Probe rays meet both faces of a pane (no back-face culling): the far one is passed, so the
  sky still reaches the room through a window pane with a single layer per camera ray."""
  layers = rw.settings.Transparency(enabled=True, max_layers=1)
  open_, _ = _render(_room(window=True), layers, enabled=True)
  glazed, _ = _render(_room(window=True, glass=True), layers, enabled=True)
  # 0.85 of the sky through the pane, plus a little of the pane's own shading.
  assert 0.6 * open_.mean() < glazed.mean() <= 1.05 * open_.mean()


def _glass_roof(translucent: Literal["off", "transmittance", "clear"], *, fast: bool = True) -> rw.Renderer[None]:
  """A GI renderer of a floor under a glass roof lit by a sun, with `shadows.translucent`, on the
  fast field set (`fast`) or the default."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_weight=0.0, specular_roughness=1.0))
  glass = b.add_material(
    rw.Material(
      base_color=(0.2, 0.4, 1.0),
      specular_roughness=0.1,
      geometry_opacity=0.1,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  b.add_instance("box", size=(1.0, 1.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=wall)
  b.add_instance("box", size=(1.0, 1.0, 0.02), xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]], material=glass)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0, xform=np.c_[np.eye(3), [0.0, 0.0, 3.0]]))
  settings = (rw.RenderSettings().replace(rw.FAST) if fast else rw.RenderSettings()).replace(
    outputs=rw.Outputs.of("hdr"),
    gi=rw.settings.GI(enabled=True, spacing=0.5, max_probes=64, rays=32, bake_updates=4),
    **{"performance.traversal": "software", "shadows.translucent": translucent},
  )
  r = rw.Renderer(rw.upload(b.build()), nworld=1, cameras=[rw.CameraSpec(fovy=90.0)], resolution=(8, 8), settings=settings)
  r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  return r


def _irradiance(r: rw.Renderer) -> np.ndarray:
  """The probe volume's irradiance maps after a render."""
  r.render()
  assert r.debug.gi_volume is not None
  return r.debug.gi_volume.struct.irradiance.numpy()


def test_compile_time_change_re_bakes_the_probes():
  """A compile-time setting the probe kernel reads (shadows.translucent: the sun through the glass
  roof) changed in place re-bakes the volume: the same maps as a renderer built with it."""
  r = _glass_roof("off")
  _irradiance(r)
  assert required(r.outputs.hdr).numpy().max() == 0.0  # the roof blocks the sun from the floor
  r.update_settings(**{"shadows.translucent": "clear"})
  after = _irradiance(r)
  assert after[..., :3].max() > 0.0
  np.testing.assert_array_equal(after, _irradiance(_glass_roof("clear")))


def test_camera_only_change_keeps_the_probes():
  """Camera-only settings (the launch's block size, aa.spp, the motion output) keep the
  baked volume and its probe kernel: no re-bake (it would drop the periodic updates)."""
  r = _glass_roof("clear")
  _irradiance(r)
  vol, kernel = r.debug.gi_volume, probe.probe_kernel(r.features)
  with tuning.override(block_dim=128):
    r.update_settings(**{"aa.spp": 2, "outputs": rw.Outputs.of("hdr", "motion")})
    r.render()
  assert r.features.block_dim == 128 and r.debug.gi_volume is vol and probe.probe_kernel(r.features) is kernel


def test_camera_only_change_leaving_the_tiers_keeps_the_probes():
  """A camera-only change that moves the default onto the settings' own build on the software
  path (spp 3 with the motion output: the motion body compiles its sample count, no tier holds
  it) keeps the baked volume: the bake depends
  on the probe-relevant settings, not on the build serving them."""
  r = _glass_roof("clear", fast=False)
  _irradiance(r)
  vol, tier = r.debug.gi_volume, r.features
  r.update_settings(**{"aa.spp": 3, "outputs": rw.Outputs.of("hdr", "motion")})
  r.render()
  assert r.features != tier and r.debug.gi_volume is vol


_FIELDS = frozenset(f.name for f in dataclasses.fields(Features))


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Recording(Features):
  """Features recording which fields are read (into `reads`)."""

  def __getattribute__(self, name: str) -> object:
    """The field, recorded."""
    if name in _FIELDS:
      _reads.add(name)
    return object.__getattribute__(self, name)

  # Identity equality and hash: each recorder is a fresh `@specialize` key, so the factories run
  # (and record their reads) instead of returning a build cached for an equal Features; hashing by
  # identity also reads no field.
  __eq__ = object.__eq__
  __hash__ = object.__hash__


_reads: set[str] = set()


def test_probe_kernel_reads_no_camera_only_field():
  """`probe.CAMERA_ONLY` (reset by `probe_features`) holds no field the probe kernel factory reads,
  over GI renderers with transparency, translucent shadows, area lights, textures off / on, glossy
  and mirror reflections and the headlight (every `CAMERA_ONLY` field set in some configuration)."""
  configs = [
    {"shadows.translucent": "clear", "transparency.enabled": True},
    {"lighting.area_lights": "ltc", "lighting.emissive_lights": True},
    {"shadows.enabled": False},
    {"reflections.enabled": True, "reflections.shadows": "sun", "reflections.max_exact": 1},
    {"lighting.camera_light": True},
  ]
  # The glass roof has no mirror material: mirror reflections and two-sided shading are set on the
  # resolved Features of the last configuration.
  overrides = [{}, {}, {}, {}, {"reflection": True, "two_sided": True, "backface_culling": True}]
  for kw, extra in zip(configs, overrides, strict=True):
    r = _glass_roof("off")
    r.update_settings(**kw)
    rec = construct_named(_Recording, {n: extra.get(n, getattr(r.features, n)) for n in _FIELDS})
    tracer = SoftwareBackend.tracer(ProbeFeatures(rec.shape_types, rec.trees), cull=False)
    _reads.clear()
    gi.make_trace_kernel(rec, probe.make_probe_shading(rec, tracer))
    assert _reads and not (_reads & probe.CAMERA_ONLY), kw


def test_camera_ray_fields_do_not_rebake_probes():
  """Regression: the camera rays' back-face culling, headlight and glossy reflection fields were
  part of the probe kernel's Features (`bake_keys.probe_key`): changing one (e.g.
  `reflections.enabled`) re-baked the GI volume for an identical probe kernel. (A mirror
  reflectance is a material's specular F0, read by the probe hits too: no setting toggles it.)"""
  f = _glass_roof("off").features
  flipped = {
    "backface_culling": dataclasses.replace(f, backface_culling=not f.backface_culling),
    "headlight": dataclasses.replace(f, headlight=not f.headlight),
    "two_sided": dataclasses.replace(f, two_sided=not f.two_sided),
    "glossy": dataclasses.replace(f, glossy=1 if f.glossy != 1 else 2),
    "reflection_shadows": dataclasses.replace(f, reflection_shadows=(f.reflection_shadows + 1) % 3),
    "hit_light_cap": dataclasses.replace(f, hit_light_cap=f.hit_light_cap + 1),
    "motion": dataclasses.replace(f, motion=not f.motion),
  }
  for name, g in flipped.items():
    assert probe.probe_features(g) == probe.probe_features(f), name


def test_bright_light_stays_finite():
  """Regression: the float16 maps overflowed (inf, then NaN at the next blend: black image) for a
  bounce of physical-unit light ~2x display white (the Office's lamp near the ceiling)."""
  host = _room(window=False)
  ref, _ = _render(host, enabled=True)
  bright, r = _render(_room(window=False, glow_strength=1.0e8), enabled=True)
  irr = required(r.debug.gi_volume).struct.irradiance.numpy().astype(np.float32)
  assert np.isfinite(irr).all() and np.isfinite(bright).all()
  # Linear in the light (no clamp at this level: the maps hold display-relative values).
  np.testing.assert_allclose(bright / 1.0e6, ref, rtol=0.02)


# One probe of 8 rays (the volume tests below).
_ONE_PROBE = rw.settings.GI(spacing=1.0, max_probes=1, rays=8, normal_bias=0.0)


def test_blend_drops_non_finite_rays():
  vol = gi.ProbeVolume(np.zeros(3), np.ones(3), spacing=1.0, cell=0.25, settings=_ONE_PROBE, scale=1.0, device=wp.get_device())
  rays = np.ones((1, 8, 4), np.float32)
  rays[0, 0] = (np.nan, 1.0, 1.0, 1.0)
  rays[0, 1] = (np.inf, 1.0, 1.0, np.inf)
  rays[0, 2, :3] = 1.0e9  # beyond float16: clamped
  ray_radiance, ray_sky = vol.ray_buffers()
  ray_radiance.assign(rays)
  ray_sky.fill_(1.0)
  ndist = gi.probes.DIST_TEXELS**2
  distance_sum = wp.zeros((1, ndist), dtype=wp.vec3)
  for h in (0.0, 0.5):
    vol.directions()
    vol._blend_chunk(0, 1, h, gate=vol.always)  # the blends alone
    wp.launch(
      gi.kernels.accumulate_static,
      dim=(1, ndist),
      inputs=[vol.struct, ray_radiance, vol.distance_lobes, 0],
      outputs=[distance_sum],
    )
    vol.count(vol.always)
  wp.launch(gi.kernels.resolve_static, dim=(1, ndist), inputs=[distance_sum], outputs=[vol.struct])
  assert np.isfinite(vol.struct.irradiance.numpy().astype(np.float32)).all()
  assert np.isfinite(vol.struct.distance.numpy().astype(np.float32)).all()
  assert vol.updates.numpy()[0] == 2


def test_octahedral_round_trip():
  @wp.kernel
  def round_trip(d: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = octahedral.oct_decode(octahedral.oct_encode(d[i]))

  rng = np.random.default_rng(0)
  d = rng.normal(size=(256, 3)).astype(np.float32)
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  out = wp.zeros(256, dtype=wp.vec3)
  wp.launch(round_trip, dim=256, inputs=[wp.array(d, dtype=wp.vec3)], outputs=[out])
  np.testing.assert_allclose(out.numpy(), d, atol=1e-5)


def test_distance_moments_vary_continuously_with_direction():
  """B53: the Chebyshev visibility read the distance moments of the nearest texel, a step function
  of the direction: at coarse probe spacing the lookup's weights jumped at the octahedral texel
  borders and leaked light in sharp triangular patches (Office ceiling, gi.spacing 2). Bilinear
  taps make them continuous: along a fine sweep of directions over a random map, no step is
  larger than the map's texel-to-texel variation allows."""
  from raywarp.core.shade.gi import probes

  @wp.kernel
  def sweep(vol: probes.Volume, d: wp.array[wp.vec3], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = probes._sample_distance(vol, 0, d[i])

  vol = gi.ProbeVolume(np.zeros(3), np.ones(3), spacing=1.0, cell=0.25, settings=_ONE_PROBE, scale=1.0, device=wp.get_device())
  rng = np.random.default_rng(1)
  moments = rng.uniform(0.0, 1.0, size=(*vol.struct.distance.shape, 2)).astype(np.float16)
  vol.struct.distance.assign(moments.reshape(*vol.struct.distance.shape, 2))
  n = 4096
  phi = np.linspace(0.1, 0.1 + 2.0 * np.pi, n)
  d = np.stack([np.cos(phi) * 0.8, np.sin(phi) * 0.8, np.full(n, 0.6)], axis=1).astype(np.float32)
  out = wp.zeros(n, dtype=wp.vec2)
  wp.launch(sweep, dim=n, inputs=[vol.struct, wp.array(d, dtype=wp.vec3)], outputs=[out])
  steps = np.abs(np.diff(out.numpy()[:, 0]))
  # Nearest texel: steps up to ~1 (two random texels); bilinear: a fraction of a texel per step.
  assert steps.max() < 0.05, steps.max()


def _moved_block_probes(render_first: bool) -> np.ndarray:
  """Probe irradiance after baking the room with its block, moving the block to a corner, then
  `update_gi` (after a render of the moved block when `render_first`)."""
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    gi=rw.settings.GI(enabled=True, spacing=0.5, max_probes=64, rays=32, bake_updates=2),
    lighting=rw.settings.Lights(emissive_lights=False),
  )
  scene = rw.upload(_room(window=False, block=True))
  r = rw.Renderer(
    scene, nworld=1, cameras=[rw.CameraSpec(fovy=90.0)], resolution=(8, 8), settings=settings, per_world=rw.PerWorld(pose=[0])
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()  # the bake, with the block at the centre
  r.per_world.columns("instance.pos", 0).assign(np.array([[[0.6, 0.6, 0.3]]], np.float32))
  if render_first:
    r.render()  # refits the scene BVH (gi.update_every=0: no probe update)
  r.update_gi(updates=2)
  assert r.debug.gi_volume is not None
  return r.debug.gi_volume.struct.irradiance.numpy().astype(np.float32)


def test_update_gi_traces_the_current_poses():
  """Review C O1: `update_gi(sim)` wrote the new poses but traced the scene BVH of the last render
  (no refit): a moved block was missed where it went and its old place shadowed nothing, so the
  probes differed from the same updates after a render."""
  np.testing.assert_array_equal(_moved_block_probes(False), _moved_block_probes(True))


def _periodic_renderer(device: str, every: int) -> rw.Renderer:
  """A renderer of the glowing room with periodic GI updates after the bake."""
  with wp.ScopedDevice(device):
    settings = rw.RenderSettings(
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal="software"),
      gi=rw.settings.GI(enabled=True, max_probes=64, rays=32, bake_updates=4, update_every=every),
    )
    r = rw.Renderer(
      rw.upload(_room(window=False)), nworld=1, cameras=[rw.CameraSpec(fovy=90.0)], resolution=(8, 8), settings=settings
    )
    r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()  # the bake
  return r


def test_rebuild_gi_rebakes_on_the_next_render():
  """`rebuild_gi` drops the probe volume: the next render bakes a new one (untested before O1)."""
  r = _periodic_renderer("cpu", 0)
  vol = r.debug.gi_volume
  assert vol is not None
  version = r.debug.graph_version()
  r.rebuild_gi()
  assert r.debug.gi_volume is None and r.debug.graph_version() != version  # captured graphs are stale
  r.render()
  assert r.debug.gi_volume is not None and r.debug.gi_volume is not vol


def test_periodic_update_reads_nothing_back(monkeypatch):
  """B5: a periodic GI update (gi.update_every) read the light table back to the host and
  allocated new arrays (the CUDA-graph capture failed, error 906). Counts every host transfer and
  allocation while the renderer's GI step runs on frames after the bake."""
  r = _periodic_renderer("cpu", 1)
  array_type = type(r.outputs.hdr)
  calls: list[str] = []
  active = [False]

  def counted(name, fn):
    def wrapper(*args, **kwargs):
      # A view of existing memory (`ptr=`: the CPU texture copy's) is neither a transfer nor an
      # allocation.
      if active[0] and not (name == "__init__" and kwargs.get("ptr") is not None):
        calls.append(name)
      return fn(*args, **kwargs)

    return wrapper

  for name in ("numpy", "assign", "__init__"):
    monkeypatch.setattr(array_type, name, counted(name, getattr(array_type, name)))
  # __init__ above counts array construction without replacing the array class used by isinstance.
  for name in ("copy", "zeros", "empty", "zeros_like", "empty_like", "clone", "full"):
    monkeypatch.setattr(wp, name, counted(name, getattr(wp, name)))
  from raywarp.core.pipeline.baked import Baked

  step = Baked.probes

  def probes(self, ctx, frame, backend, force=0):
    active[0] = True
    try:
      step(self, ctx, frame, backend, force)
    finally:
      active[0] = False

  monkeypatch.setattr(Baked, "probes", probes)
  vol = r.debug.gi_volume
  assert vol is not None
  before = int(vol.updates.numpy()[0])
  r.render()
  r.render()
  assert calls == []
  assert int(vol.updates.numpy()[0]) == before + 2


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph capture needs a GPU")
def test_periodic_update_replays_in_a_cuda_graph():
  """B5: a frame with a periodic GI update captures in a CUDA graph and its replays keep the
  cadence (update_every=2: one update every second replay)."""
  r = _periodic_renderer("cuda:0", 2)
  vol = r.debug.gi_volume
  assert vol is not None
  before = int(vol.updates.numpy()[0])
  with wp.ScopedDevice("cuda:0"):
    with wp.ScopedCapture() as capture:
      r.render()
    for _ in range(4):
      wp.capture_launch(required(capture.graph))
    wp.synchronize()
  assert int(vol.updates.numpy()[0]) == before + 2
  assert np.isfinite(required(r.outputs.hdr).numpy()).all()


@wp.kernel
def _cube_lookups(
  vol: gi.Volume,
  worlds: wp.array[int],
  points: wp.array[wp.vec3],
  normals: wp.array[wp.vec3],
  out: wp.array[wp.vec4],
):
  i = wp.tid()
  out[i] = gi.irradiance(vol, worlds[i], points[i], normals[i], normals[i])


def _cube_volume(nworld: int, monkeypatch: pytest.MonkeyPatch | None = None) -> gi.ProbeVolume:
  """A 3 x 2 x 2 probe volume over [0, 3] x [0, 2] x [0, 2] with one group per world."""
  if monkeypatch is not None:
    # Few slabs per texture row: the slabs of every group wrap over several rows.
    monkeypatch.setattr(gi.cube, "MAX_SIDE", 16)
  digests = np.stack([np.arange(nworld), np.zeros(nworld)], axis=1).astype(np.uint32) + 1
  groups = gi.Groups(nworld, digests=digests, device=wp.get_device())
  settings = rw.settings.GI(spacing=1.0, max_probes=64, rays=8)
  return gi.ProbeVolume(
    np.zeros(3),
    np.array([3.0, 2.0, 2.0]),
    spacing=1.0,
    cell=0.25,
    settings=settings,
    scale=2.0,
    device=wp.get_device(),
    groups=groups,
  )


@pytest.mark.parametrize("wrapped", [False, True], ids=["one_row", "wrapped"])
def test_ambient_cube_reads_its_groups_slab_for_each_face(monkeypatch, wrapped):
  """Every (group, face) slab filled with its own constant: a lookup anywhere (inside, at the
  box's edges, outside it) with an axis normal returns exactly that slab's value (no filtering
  across slabs), unscaled; a diagonal normal blends the three facing slabs by n²."""
  nworld = 3
  vol = _cube_volume(nworld, monkeypatch if wrapped else None)
  cube = vol.cube
  texels = np.zeros(required(cube.texels).shape, np.float16)
  _nx, ny, nz = cube.struct.counts
  tiles = cube.struct.tiles
  assert (texels.shape[1] > ny) == wrapped
  value = {}
  for g in range(vol.ngroup):
    for face in range(6):
      slab = 6 * g + face
      y, z = (slab // tiles) * ny, (slab % tiles) * nz
      value[g, face] = 1.0 + g * 10.0 + face
      texels[z : z + nz, y : y + ny, :, :3] = value[g, face] * 2.0  # map units: times the scale
      texels[z : z + nz, y : y + ny, :, 3] = face / 8.0
  cube.staging().assign(texels)
  cube.struct.texture.copy_from(cube.staging())
  axes = np.concatenate([np.eye(3), -np.eye(3)])[[0, 3, 1, 4, 2, 5]]  # face order +x, -x, +y, ...
  corners = np.array([[0.01, 0.01, 0.01], [2.99, 1.99, 1.99], [1.5, 1.0, 1.0], [-1.0, 3.0, 5.0], [3.5, -2.0, 1.0]])
  worlds, points, normals, want = [], [], [], []
  for w in range(nworld):
    g = int(vol.groups.world_group.numpy()[w])
    for p in corners:
      for face in range(6):
        worlds.append(w)
        points.append(p)
        normals.append(axes[face])
        want.append([value[g, face]] * 3 + [face / 8.0])
      n = np.array([1.0, -2.0, 2.0]) / 3.0
      worlds.append(w)
      points.append(p)
      normals.append(n)
      blend = n[0] ** 2 * value[g, 0] + n[1] ** 2 * value[g, 3] + n[2] ** 2 * value[g, 4]
      want.append([blend] * 3 + [(n[0] ** 2 * 0 + n[1] ** 2 * 3 + n[2] ** 2 * 4) / 8.0])
  out = wp.zeros(len(worlds), dtype=wp.vec4)
  wp.launch(
    _cube_lookups,
    dim=len(worlds),
    inputs=[
      vol.struct,
      wp.array(worlds, dtype=int),
      wp.array(np.asarray(points, np.float32), dtype=wp.vec3),
      wp.array(np.asarray(normals, np.float32), dtype=wp.vec3),
    ],
    outputs=[out],
  )
  np.testing.assert_allclose(out.numpy(), np.asarray(want, np.float32), rtol=2e-3)


@wp.kernel
def _sight_bits(c: cube.Cube, idx: wp.array[wp.vec3i], out: wp.array[int]):
  t = wp.tid()
  out[t] = cube.wall_channel(c, idx[t][0], idx[t][1], idx[t][2], 3)


def test_ambient_cube_resamples_the_probe_lookup():
  """Bake the emissive room: at the cube's valid texel centres (axis normals) the camera-hit lookup
  equals the probe lookup biased along the normal over the probes in sight of the texel, times
  the face's contact open share (the resample)."""
  _, r = _render(_room(window=False), enabled=True)
  with r:
    vol = required(r.debug.gi_volume)
    c = vol.cube.struct
    cell = 1.0 / np.asarray(c.inv_cell)
    origin = np.asarray(c.origin)
    rng = np.random.default_rng(1)
    counts = np.asarray(c.counts)
    idx = rng.integers(1, counts - 1, size=(32, 3))
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    normals = axes[rng.integers(0, 6, 32)]
    # The lookup is pushed one texel along the normal: query one texel behind the centre.
    centres = origin + (idx + 0.5) * cell
    points = centres - normals * c.bias
    sight_out = wp.zeros(32, dtype=int)
    wp.launch(_sight_bits, dim=32, inputs=[c, wp.array(idx.astype(np.int32), dtype=wp.vec3i)], outputs=[sight_out])
    sight = sight_out.numpy()
    out = wp.zeros(32, dtype=wp.vec4)
    args = [wp.array(np.zeros(32, int), dtype=int), wp.array(points.astype(np.float32), dtype=wp.vec3)]
    wp.launch(
      _cube_lookups, dim=32, inputs=[vol.struct, *args, wp.array(normals.astype(np.float32), dtype=wp.vec3)], outputs=[out]
    )
    ref = wp.zeros(32, dtype=wp.vec4)
    seen = wp.zeros(32, dtype=float)
    ao = vol.cube.ao.numpy()
    wp.launch(
      _probe_lookups,
      dim=32,
      inputs=[
        vol.struct,
        wp.array(centres.astype(np.float32), dtype=wp.vec3),
        wp.array(normals.astype(np.float32), dtype=wp.vec3),
        wp.array(sight, dtype=int),
      ],
      outputs=[ref, seen],
    )
  valid = seen.numpy() >= cube.VISIBLE_MIN
  # The resample scales each face by its baked contact open share (`kernels.contact_open`).
  axis = np.abs(normals).argmax(1)
  face = 2 * axis + (normals[np.arange(32), axis] < 0)
  open_share = ao[idx[:, 2], idx[:, 1], idx[:, 0], face] / 255.0
  want = ref.numpy() * open_share[:, None]
  got, want = out.numpy()[valid], want[valid]
  assert valid.sum() >= 16 and want[:, :3].max() > 0.0
  np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3 * float(want[:, :3].max()))


def test_bake_bounces_light_the_traced_hits_again(monkeypatch):
  """The bake traces its rays once and lights the recorded hits again for the later bounces
  (`ProbeVolume.bake`): on the baked room, lighting the hits of a traced update from the cube
  (`kernels.bounce_hits`) gives the traced radiance (opaque layers: the record is exact up to its
  float16 weights and packed normals), and the bounce is part of it."""
  captured = {}
  bake = gi.Schedule.bake

  def capture(self, kernel, inputs):
    captured.update(kernel=kernel, inputs=inputs)
    bake(self, kernel, inputs)

  monkeypatch.setattr(gi.Schedule, "bake", capture)
  _, r = _render(_room(window=False, block=True), enabled=True)
  with r:
    vol = required(r.debug.gi_volume)
    n = min(vol.chunk, vol.ngroup)
    hits = vol._hits(vol.chunk * vol.nprobe)
    inputs = captured["inputs"]
    vol._trace_chunk(captured["kernel"], inputs, inputs.lights(), vol.always, 0, n, hits)
    ray_radiance, _ = vol.ray_buffers()
    traced = ray_radiance.numpy().copy()
    vol._bounce_chunk(hits, vol.always, 0, n)
    bounced = ray_radiance.numpy()
    fixed = hits.fixed.numpy()
  lit = traced[..., 3] > 0.0
  np.testing.assert_allclose(bounced[lit], traced[lit], rtol=1e-2, atol=1e-3 * float(traced[lit][:, :3].max()))
  bounce = traced[lit][:, :3] - fixed[lit]
  assert bounce.min() > -1e-3 * float(traced[lit][:, :3].max()) and bounce.mean() > 0.05 * traced[lit][:, :3].mean()


def test_back_faces_a_query_misses_are_recovered(monkeypatch):
  """`shade.gi.kernels.with_back_faces` (the hardware GI's queries over spheres and capsules, whose
  intersector reports no hit from inside): over a query that drops back faces (the culled software
  closest-hit query), rays from inside a sphere get the sphere's back face at the unculled query's
  distance, and rays from outside keep their hit."""
  captured = {}
  bake = gi.Schedule.bake

  def capture(self, kernel, inputs):
    captured.update(inputs=inputs)
    bake(self, kernel, inputs)

  monkeypatch.setattr(gi.Schedule, "bake", capture)
  b = rw.SceneBuilder()
  white = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7)))
  b.add_instance("sphere", size=(0.4, 0.4, 0.4), xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]], material=white)
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=white)
  _, r = _render(b.build(), enabled=True)
  with r:
    geo = captured["inputs"].geo
    f = r.features
    features = ProbeFeatures(f.shape_types, f.trees)
    culled = trace_lib.widen(SoftwareBackend.tracer(features, cull=True).closest)
    unculled = trace_lib.widen(SoftwareBackend.tracer(features, cull=False).closest)
    recovered = gi.kernels.with_back_faces(culled)

    @wp.kernel(enable_backward=False, module="unique", module_options=query_lib.module_options(f.trees))
    @concrete(geo=trace_lib.software_geometry_type(live_uv=f.live_material_uv))
    def queries(geo: trace_lib.SoftwareGeometry, origins: wp.array[wp.vec3], dirs: wp.array[wp.vec3], out: wp.array2d[float]):
      i = wp.tid()
      a = recovered(geo, 0, origins[i], dirs[i], 1.0e6)
      b = unculled(geo, 0, origins[i], dirs[i], 1.0e6)
      out[i, 0] = float(a.instance_id)
      out[i, 1] = a.dist
      out[i, 2] = float(b.instance_id)
      out[i, 3] = b.dist

    rng = np.random.default_rng(3)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    inside = np.array([0.0, 0.0, 1.0]) + 0.2 * rng.uniform(-1, 1, size=(64, 3))
    outside = np.tile([1.5, 0.0, 1.0], (64, 1))
    origins = np.concatenate([inside, outside]).astype(np.float32)
    dirs = np.concatenate([d, d]).astype(np.float32)
    out = wp.zeros((128, 4), dtype=float)
    wp.launch(
      queries,
      dim=128,
      inputs=[geo, wp.array(origins, dtype=wp.vec3), wp.array(dirs, dtype=wp.vec3)],
      outputs=[out],
    )
    o = out.numpy()
  np.testing.assert_array_equal(o[:, 0], o[:, 2])
  np.testing.assert_allclose(o[:, 1], o[:, 3], rtol=1e-3, atol=1e-3)
  assert (o[:64, 0] == 0).all()  # every inside ray meets the sphere's back face


@wp.kernel
def _probe_lookups(
  vol: gi.Volume,
  points: wp.array[wp.vec3],
  normals: wp.array[wp.vec3],
  sight: wp.array[int],
  out: wp.array[wp.vec4],
  seen: wp.array[float],
):
  i = wp.tid()
  p = points[i]
  v, s = gi.probes.lookup(vol, 0, p, p + normals[i] * vol.normal_bias, p, normals[i], sight[i])
  out[i] = gi.probes._unscale(vol, v)
  seen[i] = s


def _two_rooms() -> rw.Scene:
  """Rooms A (x < 0) and B (x > 0), 3 x 3 x 2.5 m, a 10 cm partition at x = 0, a light in A only."""
  b = rw.SceneBuilder()
  white = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  t = 0.05
  for half, at in (
    ((3.1, 1.6, t), (0.0, 0.0, -t)),
    ((3.1, 1.6, t), (0.0, 0.0, 2.5 + t)),
    ((3.1, t, 1.25), (0.0, -1.5 - t, 1.25)),
    ((3.1, t, 1.25), (0.0, 1.5 + t, 1.25)),
    ((t, 1.6, 1.25), (-3.0 - t, 0.0, 1.25)),
    ((t, 1.6, 1.25), (3.0 + t, 0.0, 1.25)),
    ((0.05, 1.5, 1.25), (0.0, 0.0, 1.25)),
  ):
    b.add_instance("box", material=white, size=half, xform=np.c_[np.eye(3), at])
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=60.0, radius=0.1, xform=np.c_[np.eye(3), [-1.2, -0.5, 1.8]]))
  return b.build()


# ---------------------------------------------------------------------------------------------
# Characterize cube contact filtering independently of probe transport.
#
# These are reconstruction/ownership tests, not a claim that the leaking field is
# physically correct. An opaque partition separates an ideal unit field from zero;
# the texel centres already contain the exact answer. Any positive dark-side value
# therefore comes solely from reconstruction, with no exposure or probe error.


@wp.kernel
def _sample(
  # Model:
  field: cube.Cube,
  # In:
  points: wp.array[wp.vec3],
  normal: wp.vec3,
  # Out:
  values: wp.array2d[wp.vec4],
):
  group, i = wp.tid()
  values[group, i] = cube.ambient(field, group, points[i], normal, normal)


@pytest.mark.parametrize("bias_cells", [0.0, 0.5, 1.0])
def test_tangent_contact_filter_is_not_repaired_by_normal_bias(bias_cells: float):
  """At a floor/wall contact, normal offsets cannot remove tangential mixing.

  Two groups have opposite fields; testing both also detects cross-group leakage.
  The 1 cm point in the unlit room receives 46% of the other room's unit field.
  """
  with wp.ScopedDevice("cpu"):
    texture = cube.CubeTexture((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), (1, 1, 1), 2 * cube.FACES, wp.get_device())
    c = texture.struct
    nx, ny, nz = c.counts
    cell = 1.0 / float(c.inv_cell[0])
    staging = required(texture.texels)
    texels = np.zeros(staging.shape, np.float16)
    for group in range(2):
      for face in range(6):
        slab = group * 6 + face
        y, z = (slab // c.tiles) * ny, (slab % c.tiles) * nz
        side = slice(0, nx // 2) if group == 0 else slice(nx // 2, nx)
        texels[z : z + nz, y : y + ny, side, :3] = 1.0
    staging.assign(texels)
    texture.struct.texture.copy_from(texture.staging())
    c.bias = bias_cells * cell
    xs = np.asarray([0.0, 0.01, 0.05, cell / 2, cell], np.float32)
    points = np.c_[xs, np.zeros_like(xs), np.zeros_like(xs)].astype(np.float32)
    values = wp.zeros((2, len(xs)), dtype=wp.vec4)
    wp.launch(_sample, dim=values.shape, inputs=[c, wp.array(points, dtype=wp.vec3), wp.vec3(0.0, 0.0, 1.0)], outputs=[values])
    actual = values.numpy()[:, :, 0]
    expected = np.clip(0.5 - xs / cell, 0.0, 1.0)
    # Lookups are snapped to the texture filter's 1/256 texel grid (`cube.fetch`).
    step = 1.0 / cube.FILTER_STEPS
    np.testing.assert_allclose(actual[0], expected, atol=step)
    np.testing.assert_allclose(actual[1], 1.0 - expected, atol=step)
    print({"bias_cells": bias_cells, "cell_m": cell, "distance_m": xs.tolist(), "dark_side": actual[0].tolist()})


def _selected_kernel(features: Features) -> wp.Kernel:
  """Exercise the production ambient factory (camera and reflection hits share it), not a
  test-side GI selector."""
  diffuse_ambient = ambient.make_diffuse_ambient(dataclasses.replace(features, env_lit=False))

  @wp.kernel(module="unique")
  def sample(
    # Model:
    env: EnvironmentArrays,
    # In:
    points: wp.array[wp.vec3],
    # Out:
    values: wp.array[wp.vec3],
  ):
    i = wp.tid()
    up = wp.vec3(0.0, 0.0, 1.0)
    _fill, diffuse, _probes, _sky = diffuse_ambient(env, 0, points[i], up, up, 1.0)
    values[i] = diffuse

  return sample


def test_sealed_room_stays_dark_next_to_a_lit_one():
  """A sealed unlit room has exactly zero physical bounce: the ambient cube's wall-aware lookup
  returns 0 at its partition contact; the lit side remains lit (no global darkening)."""
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal="software"),
    gi=rw.settings.GI(enabled=True, spacing=1.0, rays=64, bake_updates=4),
  )
  with rw.Renderer(rw.upload(_two_rooms()), nworld=1, cameras=[rw.CameraSpec()], resolution=(4, 4), settings=settings) as r:
    r.render()
    vol = required(r.debug.gi_volume)
    y = np.linspace(-1.3, 1.3, 32)
    dark = np.c_[np.full(32, 0.051), y, np.zeros(32)]
    points = wp.array(np.concatenate([dark, dark * [-1, 1, 1]]).astype(np.float32), dtype=wp.vec3)
    env = EnvironmentArrays()
    env.gi = vol.struct
    env.params = Params()
    env.exposure = wp.ones(1, dtype=float)
    values = wp.zeros(64, dtype=wp.vec3)
    wp.launch(_selected_kernel(r.features), dim=64, inputs=[env, points], outputs=[values])
    result = values.numpy()
    np.testing.assert_array_equal(result[:32], 0.0)
    assert result[32:].mean() > 0.01


@wp.kernel
def _probe_groups(
  # Model:
  vol: gi.Volume,
  # In:
  points: wp.array[wp.vec3],
  # Out:
  values: wp.array2d[wp.vec4],
):
  world, i = wp.tid()
  up = wp.vec3(0.0, 0.0, 1.0)
  p = points[i]
  values[world, i] = gi.probes.irradiance_at(vol, gi.probes.group_row(vol, world), p, p + up * vol.normal_bias, up)


def test_probe_lookup_preserves_open_field_scale_groups_and_remapping():
  """Constant open fields have exact known values, including sky and exposure scale.

  Remap one world to another group (the lookup-side reset contract); unrelated
  worlds keep their values. Existing group tests own classification/invalidation.
  """
  vol = _cube_volume(3)
  maps = np.zeros((*vol.struct.irradiance.shape, 4), np.float16)
  for group in range(3):
    maps[group * vol.nprobe : (group + 1) * vol.nprobe] = [2 * (group + 1), 4 * (group + 1), 6 * (group + 1), 0.5]
  vol.struct.irradiance.assign(maps)
  vol.struct.distance.fill_(wp.vec2h(100.0, 10000.0))
  vol.refresh()
  points = wp.array([[0.6, 0.6, 0.6], [1.5, 1.0, 1.0], [2.4, 1.4, 1.4]], dtype=wp.vec3)
  values = wp.zeros((3, 3), dtype=wp.vec4)
  for groups in ([0, 1, 2], [2, 1, 2]):
    vol.struct.world_group.assign(np.asarray(groups, np.int32))
    wp.launch(_probe_groups, dim=values.shape, inputs=[vol.struct, points], outputs=[values])
    want = np.asarray([[g + 1, 2 * (g + 1), 3 * (g + 1), 0.5] for g in groups], np.float32)
    np.testing.assert_allclose(values.numpy(), np.repeat(want[:, None], 3, axis=1), rtol=2e-3)


def test_ground_quads_are_left_out_of_the_probe_box():
  """A flat instance reaching past every other instance (a 222 m floor quad under a classroom, a
  ground quad under a robot) bounds the probe box no more than a plane primitive; the walls and
  floor of a room (within the others' bounds) still do."""
  from raywarp.core.shade.gi import volume

  lo = np.array([[-110, -110, 0], [3, -21, 0], [-4.5, -6, 0], [-4.5, -6, 3.0]], np.float32)
  hi = np.array([[110, 110, 0], [8, 23, 2.7], [4.5, 6, 3.06], [4.5, 6, 3.0]], np.float32)
  assert volume._ground_meshes(lo, hi, np.ones(4, bool)).tolist() == [True, False, False, False]
  room_lo = np.array([[-1, -1, 0], [-1, -1, 2], [-1, 1, 0], [-1, -1, 0], [1, -1, 0], [0, 0, 0]], np.float32)
  room_hi = np.array([[1, 1, 0], [1, 1, 2], [1, 1, 2], [-1, 1, 2], [1, 1, 2], [0.6, 0.6, 0.6]], np.float32)
  assert not volume._ground_meshes(room_lo, room_hi, np.ones(6, bool)).any()


def _outward_quad_room(double_sided: bool) -> rw.Scene:
  """A 2 x 2 x 2 m room of single quads whose winding faces OUT (the Bitterli stages' walls),
  flagged double-sided or not, an emissive panel under the ceiling facing down."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_weight=0.0, specular_roughness=1.0))
  glow = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), specular_weight=0.0, emission_luminance=100.0))
  flags = int(rw.scene.InstanceFlags.DEFAULT | (rw.scene.InstanceFlags.DOUBLE_SIDED if double_sided else 0))
  quad = b.add_mesh(
    np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.int32)
  )
  # Each wall: the quad's +z (its winding) rotated to face away from the room centre.
  for rot, at in (
    (np.diag(np.array([1.0, -1.0, -1.0])), (0, 0, 0)),  # floor, +z -> -z (out)
    (np.eye(3), (0, 0, 2)),  # ceiling, +z (out)
    (np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float), (1, 0, 1)),  # +x wall
    (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float), (-1, 0, 1)),  # -x wall
    (np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float), (0, 1, 1)),  # +y wall
    (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), (0, -1, 1)),  # -y wall
  ):
    b.add_instance("mesh", mesh=quad, material=wall, xform=np.c_[rot, at], flags=flags)
  b.add_instance("box", size=(0.3, 0.3, 0.01), xform=np.c_[np.eye(3), [0.0, 0.0, 1.95]], material=glow)
  return b.build()


def test_two_sided_walls_wound_outwards_have_no_inside():
  """Back faces of double-sided geometry are front faces for the GI: a room of outward-wound,
  double-sided quads (the Bitterli Cornell box) keeps its probes; with single-sided quads its
  inside is "inside geometry" and the probes in it are disabled (the old rule for both)."""
  actives = {}
  for two in (True, False):
    _, r = _render(_outward_quad_room(two), enabled=True)
    with r:
      vol = required(r.debug.gi_volume)
      idx = np.stack(np.unravel_index(np.arange(vol.nprobe), vol.counts[::-1])[::-1], axis=1)
      pos = np.asarray(vol.struct.origin) + idx * np.asarray(vol.spacing)
      inside = np.all((pos > [-0.9, -0.9, 0.1]) & (pos < [0.9, 0.9, 1.9]), axis=1)
      actives[two] = float(vol.struct.active.numpy()[inside].mean())
  assert actives[True] == 1.0 and actives[False] == 0.0, actives
