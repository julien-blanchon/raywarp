"""OptiX traversal backend: parity with the software traversal (skipped without OptiX)."""

import re
import warnings
from typing import Final

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import MjScene, caster_transmittance, make_scene, optix_build, pane_stack_centre, recipe, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.kernels.abi.features import software_view
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.lights.sources import make_cell
from raywarp.core.trace.frame_structs import lights_struct
from raywarp.core.trace.optix.device import params as params_lib
from raywarp.core.trace.optix.device import programs
from raywarp.core.trace.optix.device import trace as rt_trace
from raywarp.core.trace.optix.graph import Segments
from raywarp.core.trace.optix_backend import OptixBackend
from raywarp.integrations.mujoco.binding import MjwBinding
from raywarp.settings.fields import OutputKind

_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <light type="directional" dir="0.3 0.2 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2" euler="0 0 25"/>
      <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2" euler="40 0 0"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2" euler="30 0 0"/>
      <geom type="ellipsoid" pos="0.3 0.6 0.3" size="0.1 0.2 0.15"/>
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="mesh" mesh="ico" euler="20 30 10"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="ico" scale="0.2 0.2 0.2"
        vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
    </asset>
  </mujoco>
"""


def _available() -> bool:
  if not wp.is_cuda_available():
    return False
  with wp.ScopedDevice("cuda:0"):
    mjm = mujoco.MjModel.from_xml_string(_XML)
    caps = device_caps("cuda:0")
    return (
      OptixBackend.device_reason(caps) is None
      and OptixBackend.scene_reason(rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))) is None
    )


pytestmark = pytest.mark.skipif(not _available(), reason="OptiX backend unavailable (CUDA / driver OptiX / SDK headers)")


_ALL: Final = frozenset[OutputKind]({"rgb", "depth", "seg"})


def _move(m, d, mjd, seed: int, far: bool = False):
  """Moves the free body differently in every world (the refits must follow)."""
  rng = np.random.default_rng(seed)
  qpos = np.tile(mjd.qpos, (d.nworld, 1))
  qpos[:, :3] += rng.normal(0.0, 0.3 if far else 0.2, (d.nworld, 3))
  if far:
    qpos[-1, :3] = [2.5, 1.0, 0.5]  # a large displacement
  quat = rng.normal(size=(d.nworld, 4))
  qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  d.qpos.assign(qpos.astype(np.float32))
  mjw.forward(m, d)


def _setup(  # noqa: PLR0913  # make_scene's keywords
  xml: str,
  nworld: int,
  traversal: str = "optix",
  *,
  cam_res: tuple[int, int] = (32, 32),
  rays: list[np.ndarray] | None = None,
  per_world: rw.PerWorld | None = None,
  **settings: object,
) -> MjScene:
  """`make_scene` with `traversal` (checked to be the one resolved) and `settings`."""
  s = make_scene(
    xml,
    nworld=nworld,
    cam_res=cam_res,
    cameras=None,
    rays=rays,
    per_world=per_world,
    **{"performance.traversal": traversal, **settings},
  )
  assert s.r.features.traversal == traversal
  return s


type _Frame = tuple[np.ndarray | None, np.ndarray, np.ndarray]


def _outputs(s: MjScene) -> _Frame:
  """Host copies of (rgba, depth, seg); rgba is None when the rgb output is off."""
  out = s.out
  rgba = None if out.rgb is None else out.rgb.numpy().copy()
  return rgba, required(out.depth).numpy().copy(), required(out.seg).numpy().copy()


def _render_both(
  xml: str,
  nworld: int,
  *,
  move: bool = False,
  cam_res: tuple[int, int] = (32, 32),
  **settings: object,
) -> dict[str, _Frame]:
  """Renders with both traversals; returns {traversal: (rgba, depth, seg)}."""
  out: dict[str, _Frame] = {}
  for traversal in ("software", "optix"):
    s = _setup(xml, nworld, traversal, cam_res=cam_res, rays=None, per_world=None, **settings)
    if move:
      _move(s.m, s.d, s.mjd, 0, far=True)
    s.render()
    out[traversal] = _outputs(s)
  return out


def _rgb8(rgba: np.ndarray | None) -> np.ndarray:
  """The 8-bit RGB channels of a rendered rgba (which must be on) as ints."""
  return required(rgba)[..., :3].astype(int)


@pytest.mark.parametrize("shaded", [True, False])
@pytest.mark.parametrize("move", [False, True])
def test_depth_and_segmentation_match_software(shaded, move):
  outputs: frozenset[OutputKind] = _ALL if shaded else frozenset({"depth", "seg"})
  out = _render_both(_XML, nworld=3, move=move, cam_res=(48, 40), outputs=rw.Outputs.of(*outputs))
  (_, ds, ss), (_, do, so) = out["software"], out["optix"]
  # Grazing hits may flip between geoms at silhouettes: allow a few pixels.
  assert np.mean(ss == so) > 0.998
  same = ss == so
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("spp", [1, 2])
def test_rgb_close_to_software(spp):
  out = _render_both(
    _XML, nworld=2, move=True, cam_res=(64, 48), outputs=rw.Outputs.of(*_ALL), aa=rw.settings.AntiAliasing(spp=spp)
  )
  a, b = _rgb8(out["software"][0]), _rgb8(out["optix"][0])
  diff = np.abs(a - b).max(axis=-1)
  assert a.mean() > 20.0  # something was rendered
  assert np.mean(diff > 2) < 0.005
  assert np.abs(a - b).mean() < 0.1
  # Depth / segmentation of the first sub-pixel sample.
  assert np.mean(out["software"][2] == out["optix"][2]) > 0.998


@pytest.mark.parametrize("spp", [1, 2])
def test_camera_rays_and_guides_match_software(spp):
  # A fisheye-like bundle: equidistant projection over 140 degrees (no pinhole equivalent).
  w, h = 48, 40
  u = (np.arange(w) + 0.5) / w * 2.0 - 1.0
  v = 1.0 - (np.arange(h) + 0.5) / h * 2.0
  x, y = np.meshgrid(u, v * h / w)
  theta = np.hypot(x, y) * np.radians(70.0)
  phi = np.arctan2(y, x)
  direction = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], axis=-1)
  rays = np.stack([np.zeros_like(direction), direction], axis=2).astype(np.float32)
  out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
  for traversal in ("software", "optix"):
    s = _setup(
      _XML,
      2,
      traversal,
      cam_res=(w, h),
      outputs=rw.Outputs.of(*_ALL, "normal"),
      aa=rw.settings.AntiAliasing(spp=spp),
      rays=[rays],
    ).render()
    rgba, depth, seg = _outputs(s)
    out[traversal] = (required(rgba), depth, seg, required(s.out.normal).numpy().copy())
  (rgb_s, depth_s, seg_s, normal_s), (rgb_o, depth_o, seg_o, normal_o) = out["software"], out["optix"]
  same = seg_s == seg_o
  assert np.mean(same) > 0.998
  np.testing.assert_allclose(depth_o[same], depth_s[same], rtol=2e-4, atol=1e-5)
  np.testing.assert_allclose(normal_o[same], normal_s[same], atol=2e-3)
  diff = np.abs(_rgb8(rgb_s) - _rgb8(rgb_o)).max(axis=-1)
  assert _rgb8(rgb_s).mean() > 20.0
  assert np.mean(diff > 2) < 0.005


def test_transparent_geoms_are_skipped():
  xml = _XML.replace(
    '<geom type="sphere" pos="0 0 0.4" size="0.3"/>', '<geom type="sphere" pos="0 0 0.4" size="0.3" rgba="1 0 0 0"/>'
  )
  out = _render_both(xml, nworld=2, cam_res=(32, 24), outputs=rw.Outputs.of("depth", "seg"))
  ss, so = out["software"][2], out["optix"][2]
  assert not np.any(so == 1)  # geom 1 is the invisible sphere
  assert np.mean(ss == so) > 0.998


def test_per_world_visibility_matches_software():
  """`PerWorld.visible` keeps the static geoms shared; OptiX hides them per world with the
  any-hit programs (shared box and curves), the shared intersection programs (cylinder,
  ellipsoid), the analytic plane test and mask 0 (the per-world mesh) as the software traversal."""
  per_world = rw.PerWorld(visible=list(range(7)))  # plane, sphere, box, capsule, cylinder, ellipsoid, mesh
  out = {}
  for traversal in ("software", "optix"):
    s = _setup(_XML, 3, traversal, per_world=per_world, cam_res=(48, 40), outputs=rw.Outputs.of(*_ALL))
    vis = s.r.per_world["instance.visible"]
    shown = np.ones(vis.shape, np.int32)
    shown[1, 1::2] = 0
    shown[2, 0::2] = 0
    vis.assign(shown)
    s.render()
    out[traversal] = _outputs(s)
  scene = optix_build(s.r).scene
  assert scene.shared_visible and scene.visible_curves and scene.shared.tri_ids and scene.shared_primitive_types
  (rs, ds, ss), (ro, do, so) = out["software"], out["optix"]
  assert np.mean(ss == so) > 0.998
  same = ss == so
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)
  assert np.mean(np.abs(_rgb8(rs) - _rgb8(ro)).max(axis=-1) > 2) < 0.005
  for w in (1, 2):
    for g in range(7):
      assert np.any(so[w] == g) == bool(shown[w, g]) or not np.any(so[0] == g), (w, g)


def test_diverged_world_leaves_the_other_worlds_unchanged():
  """World 1's geoms at 1e20 (a diverged simulation) leave worlds 0 and 2 exactly as before:
  the shared world cell ignores them (every primitive kind per world, in free bodies)."""
  free = re.sub(r'(<geom type="(?:sphere|box|capsule|cylinder|ellipsoid)"[^>]*/>)', r"<body><freejoint/>\1</body>", _XML)
  s = _setup(free, 3, cam_res=(48, 40), outputs=rw.Outputs.of(*_ALL))
  rt = optix_build(s.r).scene
  assert rt.worlds.tri_ids and rt.has_world_curves and rt.primitive_types
  s.render()
  before = _outputs(s)
  xpos = s.d.geom_xpos.numpy()
  xpos[1] = 1e20
  s.d.geom_xpos.assign(xpos)
  s.render()
  after = _outputs(s)
  for b, a in zip(before, after, strict=True):
    np.testing.assert_array_equal(required(a)[[0, 2]], required(b)[[0, 2]])
  assert not np.array_equal(after[2][1], before[2][1])  # world 1's geoms are gone


# A bumpy height field beside a sphere, or a cloth and a soft box (deformable surfaces, dim 2 / 3)
# over a floor (MuJoCo Warp has no flex-height field collision): the OptiX triangle GASes of both
# kinds (`trace/optix/dynamic.py`).
_TERRAIN = " ".join(str(v) for v in np.random.default_rng(3).uniform(0.0, 1.0, 81).round(3))
_DYNAMIC_XML = """
  <mujoco>
    <asset><hfield name="h" nrow="9" ncol="9" size="1.2 1.2 0.3 0.1" elevation="{terrain}"/></asset>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <light type="directional" dir="0.3 0.2 -1"/>
      {ground}
      <geom type="sphere" pos="0.5 0.4 0.5" size="0.15"/>
      {flex}
      <camera name="cam" pos="0 -2.4 1.6" xyaxes="1 0 0 0 0.6 0.8"/>
    </worldbody>
  </mujoco>
"""
_FLEX = (
  '<flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.15 0.15 0.15" pos="-0.3 0 0.7" radius="0.02" dim="2"'
  ' rgba="0.2 0.7 0.3 1"><edge equality="true"/><contact contype="0" conaffinity="0"/></flexcomp>'
  '<flexcomp name="box" type="box" count="3 3 3" spacing="0.2 0.2 0.2" pos="0.4 -0.3 0.6" dim="3" rgba="0.8 0.3 0.2 1">'
  '<contact contype="0" conaffinity="0"/></flexcomp>'
)


def _dynamic_frames(traversal: str, flex: str) -> list[_Frame]:
  """(rgba, depth, seg) of 2 worlds over `_DYNAMIC_XML` before and after a terrain change (no
  `flex`) or a per-world move of the deformables' vertices (the GAS refits must follow both)."""
  ground = '<geom type="box" size="1.2 1.2 0.05" pos="0 0 -0.05"/>' if flex else '<geom type="hfield" hfield="h"/>'
  xml = _DYNAMIC_XML.format(terrain=_TERRAIN, flex=flex, ground=ground)
  s = _setup(xml, 2, traversal, cam_res=(64, 48), outputs=rw.Outputs.of(*_ALL))
  frames = [_outputs(s.render())]
  if not flex:
    s.m.hfield_data.assign((1.0 - s.mjm.hfield_data).astype(np.float32))
    binding = s.scene.binding
    assert isinstance(binding, MjwBinding)
    binding.refit_hfields()
  else:
    vert = s.d.flexvert_xpos.numpy()
    vert[1] += np.array([0.1, 0.05, -0.15], np.float32)  # world 1's deformables only
    s.d.flexvert_xpos.assign(vert)
  frames.append(_outputs(s.render()))
  return frames


@pytest.mark.parametrize("flex", ["", _FLEX])
def test_height_fields_and_deformables_match_software(flex):
  """Height fields (triangle GASes refitted after `refit_hfields`) and deformable surfaces (a
  triangle GAS per part in each world's cell, refitted every frame) trace as the software
  BVHs do, before and after the terrain changes and one world's deformables move: no stale
  geometry after a refit."""
  out = {t: _dynamic_frames(t, flex) for t in ("software", "optix")}
  for k in range(2):
    (rs, ds, ss), (ro, do, so) = out["software"][k], out["optix"][k]
    assert np.mean(ss == so) > 0.995
    same = ss == so
    np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)
    a, b = _rgb8(rs), _rgb8(ro)
    assert a.mean() > 20.0
    assert np.mean(np.abs(a - b).max(axis=-1) > 2) < 0.01
  for t in out:  # the frames differ: the refits reached the traces
    assert np.abs(out[t][1][1] - out[t][0][1]).max() > 0.05


# A wide cloth between a ceiling light and the floor: probes under it see it unless the bake leaves
# the deformables out.
_GI_CLOTH_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 2.5" dir="0 0 -1"/>
      <geom type="box" size="1.5 1.5 0.05" pos="0 0 -0.05" rgba="0.8 0.8 0.8 1"/>
      <geom type="box" size="0.2 0.2 0.2" pos="0.8 0.8 0.2" rgba="0.8 0.3 0.2 1"/>
      <flexcomp name="cloth" type="grid" count="9 9 1" spacing="0.25 0.25 0.25" pos="0 0 0.8" radius="0.02" dim="2"
        rgba="0.1 0.1 0.1 1"><edge equality="true"/><contact contype="0" conaffinity="0"/></flexcomp>
      <camera name="cam" pos="0 -3 1.5" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
"""


def test_gi_bake_leaves_deformables_out_on_both_backends():
  """The GI bake of a scene with a wide cloth is the same on both backends: neither one's probe
  or placement rays see the deformable (OptiX by the visibility mask `trace.BAKE_RAY_MASK`, the
  software traversal by hidden parts). Before, the OptiX probes under the cloth saw it: shorter
  distances, darker irradiance."""
  gi = rw.settings.GI(enabled=True, max_probes=128, rays=64, update_every=1000)
  states = {}
  for t in ("software", "optix"):
    s = _setup(_GI_CLOTH_XML, 1, t, cam_res=(32, 24), outputs=rw.Outputs.of(*_ALL), gi=gi)
    s.render()
    states[t] = required(s.r.debug.gi_volume).state()
  sw, ox = states["software"], states["optix"]
  np.testing.assert_allclose(ox["box"], sw["box"], rtol=1e-5, atol=1e-5)
  np.testing.assert_array_equal(ox["active"], sw["active"])
  # Mean shifts with the cloth seen (software, CPU): irradiance -13 %, distance -2.9 %.
  for k, tol in (("irradiance", 0.03), ("distance", 0.01)):
    a, b = np.asarray(sw[k], np.float64), np.asarray(ox[k], np.float64)
    assert np.isfinite(b).all()
    assert abs(b.mean() - a.mean()) <= tol * abs(a.mean()), k


@pytest.mark.parametrize("spp", [1, 2])
def test_graph_capture_matches_eager(spp):
  s = _setup(_XML, 3, cam_res=(32, 24), outputs=rw.Outputs.of(*_ALL), aa=rw.settings.AntiAliasing(spp=spp))
  graph = s.r.capture(sim=s.d)  # one eager render (compile, build), then the capture
  assert len(graph.segments.graphs) == 2  # cut at the OptiX calls
  for seed in range(3):
    _move(s.m, s.d, s.mjd, seed)
    s.render()
    eager = _outputs(s)
    for a in (s.out.rgb, s.out.depth, s.out.seg):
      required(a).zero_()
    graph.launch(s.d)
    replay = _outputs(s)
    assert (eager[2] == 6).sum() > 20  # the moving mesh is in view
    for a, b in zip(eager, replay, strict=False):
      np.testing.assert_array_equal(a, b)


def test_graph_replays_back_to_back_without_synchronisation():
  """Hundreds of queued launches (the former host-node bridge deadlocked after ~30)."""
  s = _setup(_XML, 2, cam_res=(16, 16), outputs=rw.Outputs.of(*_ALL))
  _move(s.m, s.d, s.mjd, 0)
  s.render()
  eager = _outputs(s)
  graph = s.r.capture(sim=s.d)
  for _ in range(300):
    graph.launch(s.d)
  wp.synchronize()
  for a, b in zip(eager, _outputs(s), strict=False):
    np.testing.assert_array_equal(a, b)


def test_graph_capture_of_a_physics_step():
  s = _setup(_XML, 2, cam_res=(16, 16), outputs=rw.Outputs.of(*_ALL))
  graph = s.r.capture(lambda: mjw.step(s.m, s.d), sim=s.d)
  for _ in range(50):
    graph.launch(s.d)
  wp.synchronize()
  ref = _setup(_XML, 2, cam_res=(16, 16), outputs=rw.Outputs.of(*_ALL))
  for _ in range(51):  # the eager run of capture, then the launches
    mjw.step(ref.m, ref.d)
    ref.render()
  np.testing.assert_array_equal(ref.d.qpos.numpy(), s.d.qpos.numpy())
  assert s.d.time.numpy()[0] > 0.0
  for a, b in zip(_outputs(ref), _outputs(s), strict=False):
    np.testing.assert_array_equal(a, b)


def test_a_captured_gi_update_traces_the_gi_programs():
  """A periodic GI update inside `Renderer.capture` traces the build's GI programs (cut out of the
  graph like the render; every launch tests the update's gate), never the software probe-ray
  kernel: its module is neither in the warm-up nor loaded (a bundle need not hold it), and the
  replays keep the cadence (update_every=2: one update every second replay)."""
  from raywarp.core.kernels import probe
  from raywarp.core.pipeline import warmup

  gi = rw.settings.GI(enabled=True, max_probes=64, rays=32, bake_updates=2, update_every=2)
  s = _setup(_XML, 2, cam_res=(16, 16), outputs=rw.Outputs.of(*_ALL), gi=gi)
  graph = s.r.capture(sim=s.d)  # the eager render bakes, then the capture
  vol = required(s.r.debug.gi_volume)
  before = int(vol.updates.numpy()[0])
  for _ in range(4):
    graph.launch(s.d)
  wp.synchronize()
  assert int(vol.updates.numpy()[0]) == before + 2
  ctx = s.r.context()
  software = probe.probe_kernel(software_view(ctx.features, ctx.effective)).module
  assert not software.execs  # never loaded
  default, render = warmup.first_frame(ctx, s.r._backend).modules()
  assert software.name not in {m.name for m in (*default, *render)}
  assert np.isfinite(_outputs(s)[0]).all()


def test_raw_capture_raises():
  s = _setup(_XML, 2, cam_res=(16, 16))
  s.render()
  with pytest.raises(RuntimeError, match=r"Renderer\.capture"):
    with wp.ScopedCapture():
      s.render()


def test_stale_binding_after_new_data():
  s = _setup(_XML, 2, cam_res=(16, 16))
  graph = s.r.capture(sim=s.d)
  other = mjw.put_data(s.mjm, s.mjd, nworld=2)
  with pytest.raises(rw.StaleBinding, match=r"instance\.pos"):
    graph.launch(other)


def test_primitive_layout():
  """Spheres and capsules are linear curves, boxes unit-cube instances, the rest custom."""
  rng = np.random.default_rng(1)
  geoms = []
  for i in range(24):
    kind = ("sphere", "capsule", "box", "cylinder", "ellipsoid", "capsule")[i % 6]
    size = {"sphere": "0.12", "capsule": "0.06 0.15", "box": "0.1 0.05 0.12", "cylinder": "0.07 0.1"}
    pos = " ".join(f"{x:.3f}" for x in (rng.uniform(-1.2, 1.2), rng.uniform(-0.8, 0.8), rng.uniform(0.15, 0.8)))
    euler = " ".join(f"{x:.1f}" for x in rng.uniform(-90, 90, 3))
    geoms.append(f'<geom type="{kind}" size="{size.get(kind, "0.1 0.06 0.12")}" pos="{pos}" euler="{euler}"/>')
  xml = f"""
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        {"".join(geoms)}
        <camera name="cam" pos="0 -2.8 1.4" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  out = _render_both(xml, nworld=2, cam_res=(64, 48), outputs=rw.Outputs.of(*_ALL))
  (rs, ds, ss), (ro, do, so) = out["software"], out["optix"]
  assert np.mean(ss == so) > 0.998
  same = (ss == so) & (ss > 0)
  assert same.sum() > 500
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)
  diff = np.abs(_rgb8(rs) - _rgb8(ro)).max(axis=-1)
  assert np.mean(diff > 2) < 0.005

  s = _setup(xml, 1, cam_res=(8, 8)).render()
  scene = optix_build(s.r).scene
  types = s.mjm.geom_type
  # Static geoms: shared by every world, in the shared IAS's static curve / custom GASes.
  assert not (scene.worlds.curve_ids or scene.worlds.tri_ids or scene.worlds.prim_ids)
  assert {int(types[g]) for g in scene.shared.curve_ids} == {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
  }
  assert {int(types[g]) for g in scene.shared.tri_ids} == {int(mujoco.mjtGeom.mjGEOM_BOX)}
  assert set(scene.shared_primitive_types) == {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}


def _l_shape_mesh() -> str:
  """An axis-aligned L-shaped prism; MuJoCo's inertia frame is rotated from the authored one."""

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


def test_reframed_mesh_gas_matches_software():
  """A triangle GAS built in the authored frame gives the software traversal's hits."""
  xml = f"""
    <mujoco>
      <asset>{_l_shape_mesh()}</asset>
      <worldbody>
        <light pos="0.5 -0.5 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <body pos="-0.3 -0.2 0.1" euler="10 0 25">
          <freejoint/>
          <geom type="mesh" mesh="l"/>
        </body>
        <camera name="cam" pos="0 -2.2 1.6" xyaxes="1 0 0 0 0.6 0.8"/>
      </worldbody>
    </mujoco>
  """
  out = _render_both(xml, nworld=2, move=True, cam_res=(48, 40), outputs=rw.Outputs.of(*_ALL))
  (_, ds, ss), (_, do, so) = out["software"], out["optix"]
  assert (so == 1).sum() > 100  # the mesh fills part of the image
  assert np.mean(ss == so) > 0.998
  same = ss == so
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("translucent", [False, True])
def test_reflections_match_software(translucent):
  """Mirror reflections, traced by the camera-layer loop in the OptiX program, match software."""
  xml = _XML.replace('<geom type="plane" size="3 3 0.1"/>', '<geom type="plane" size="3 3 0.1" material="mirror"/>').replace(
    "<asset>", '<asset><material name="mirror" rgba="0.6 0.6 0.7 1" reflectance="0.4"/>'
  )
  if translucent:
    xml = xml.replace('size="0.3"/>', 'size="0.3" rgba="0.2 0.8 0.2 0.5"/>')
  out = _render_both(xml, nworld=2, move=True, cam_res=(64, 48), outputs=rw.Outputs.of(*_ALL))
  a, b = _rgb8(out["software"][0]), _rgb8(out["optix"][0])
  diff = np.abs(a - b).max(axis=-1)
  assert np.mean(diff > 2) < 0.005
  assert np.abs(a - b).mean() < 0.15


def test_light_grid_with_optix():
  """The light visibility grid (built with the software BVH) works with OptiX traversal."""
  out = {}
  for traversal in ("software", "optix"):
    out[traversal] = _setup(_XML, 1, traversal, cam_res=(32, 24), lighting=rw.settings.Lights(light_grid=True)).render().rgb()
  assert np.abs(out["software"] - out["optix"]).max(axis=-1).mean() < 0.01


def test_auto_traversal():
  """performance.traversal="auto" picks by the device: OptiX on a GPU with RT cores, the software
  traversal without (A100 / H100), for mesh and primitive-only scenes alike."""
  want = "optix" if device_caps("cuda:0").rt_cores else "software"
  assert make_scene(_XML, cam_res=(8, 8), performance=rw.settings.Performance(traversal="auto")).r.features.traversal == want
  no_mesh = _XML.replace('<geom type="mesh" mesh="ico" euler="20 30 10"/>', '<geom type="sphere" size="0.1"/>')
  assert make_scene(no_mesh, cam_res=(8, 8), performance=rw.settings.Performance(traversal="auto")).r.features.traversal == want


def _glass_roof() -> rw.Scene:
  """A floor under a sun, half shaded by a glass pane, half by the pane and a board below it."""
  b = rw.SceneBuilder()
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_roughness=0.1,
      geometry_opacity=0.4,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  matte = b.add_material(
    rw.Material(base_color=(0.7, 0.7, 0.7), base_metalness=0.0, specular_weight=0.0, specular_roughness=1.0)
  )
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=matte)
  b.add_instance("box", size=(1.0, 1.0, 0.01), xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]], material=glass)
  b.add_instance("box", size=(0.5, 1.0, 0.01), xform=np.c_[np.eye(3), [0.5, 0.0, 1.0]], material=matte)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0))  # straight down (-z)
  return b.build()


@pytest.mark.parametrize(("recipe_name", "npane"), [("fast", 1), ("default", 3)])
def test_a_closed_pane_costs_one_layer_as_on_software(recipe_name, npane):
  """OptiX composites stacked two-sided closed panes within the layer budget as with an unlimited
  one and as the software traversal does (`tests/render/test_effects.py::
  test_a_closed_pane_costs_one_layer`; the glass scene's panes looked opaque at fast)."""
  seen = pane_stack_centre(recipe_name, npane, "optix")
  np.testing.assert_allclose(seen, pane_stack_centre(recipe_name, npane, "optix", max_layers=8), rtol=1e-3, atol=1e-5)
  np.testing.assert_allclose(seen, pane_stack_centre(recipe_name, npane, "software"), rtol=1e-2, atol=1e-4)


def test_translucent_shadows_match_software():
  """Sun shadows through glass: OptiX's shadow trace (the glass's any-hit program multiplying the
  transmittance in) gives the software traversal's image."""
  out = {}
  for traversal in ("software", "optix"):
    settings = rw.RenderSettings(
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal=traversal),
      shadows=rw.settings.Shadows(translucent="transmittance"),
      transparency=rw.settings.Transparency(enabled=True),
      gi=rw.settings.GI(enabled=False),  # direct sunlight only: probes would light the blocked region
    )
    r = rw.Renderer(
      rw.upload(_glass_roof()),
      nworld=2,
      cameras=[rw.CameraSpec(fovy=90.0)],
      resolution=(16, 16),
      settings=settings,
    )
    r.camera_pose.assign(np.tile(np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 1, 1)))
    r.render()
    if traversal == "optix":
      assert optix_build(r).scene.translucent_triangles
    out[traversal] = required(r.outputs.hdr).numpy()[..., :3]
  np.testing.assert_allclose(out["optix"], out["software"], rtol=1e-3, atol=1e-5)
  img = out["optix"][0, 0].reshape(16, 16, 3).mean(-1)
  glass_only, blocked = img[4:12, 2:6].mean(), img[4:12, 10:14].mean()
  assert glass_only > 0.1 and blocked == 0.0  # sunlight through the pane; none past the board


@pytest.mark.parametrize("caster", ["pane", "sheet", "cylinder"])
def test_a_translucent_caster_attenuates_once_as_on_software(caster):
  """OptiX's shadow trace (the casters' any-hit / intersection programs) attenuates a closed pane,
  a two-sided sheet and a solid cylinder once each, as the software walk does
  (`tests/render/test_shadows.py::test_a_translucent_caster_attenuates_once`)."""
  optix = caster_transmittance(caster, "optix")
  assert optix == pytest.approx(caster_transmittance(caster, "software"), rel=1e-3)
  assert optix == pytest.approx(0.6, rel=0.02)


def test_translucent_shadows_through_a_cylinder_match_software():
  """Sun shadows through three glass cylinders (custom primitives): each attenuates the shadow ray
  once (its shadow intersection program, the software walk's one candidate per instance): t^3 on
  both paths."""
  b = rw.SceneBuilder()
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_roughness=0.1,
      geometry_opacity=0.4,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  matte = b.add_material(
    rw.Material(base_color=(0.7, 0.7, 0.7), base_metalness=0.0, specular_weight=0.0, specular_roughness=1.0)
  )
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=matte)
  for z in (1.0, 1.5, 2.0):
    b.add_instance("cylinder", size=(0.5, 0.1, 0.0), xform=np.c_[np.eye(3), [0.0, 0.0, z]], material=glass)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0))  # straight down (-z)
  scene = b.build()
  out = {}
  for traversal in ("software", "optix"):
    settings = rw.RenderSettings(
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal=traversal),
      shadows=rw.settings.Shadows(translucent="transmittance"),
      transparency=rw.settings.Transparency(enabled=True),
      gi=rw.settings.GI(enabled=False),  # direct sunlight only
    )
    cams = [rw.CameraSpec(fovy=90.0)]
    r = rw.Renderer(rw.upload(scene), nworld=1, cameras=cams, resolution=(16, 16), settings=settings)
    assert r.features.backface_culling
    r.camera_pose.assign(np.array([[[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
    out[traversal] = required(r.outputs.hdr).numpy()[..., :3]
  np.testing.assert_allclose(out["optix"], out["software"], rtol=1e-3, atol=1e-5)
  img = out["optix"][0, 0].reshape(16, 16, 3).mean(-1)
  assert img[7, 7] == pytest.approx(0.216 * img[0, 0], rel=0.02)  # under the stack: 0.6^3 of the sun


def test_two_sided_instances_seen_from_behind_match_software():
  """Two-sided (USD doubleSided) planes and custom primitives are not culled from behind, as on
  the software path (OptiX culled them): a plane above camera 0, a cylinder around camera 1, both
  cameras looking up."""
  b = rw.SceneBuilder()
  two = rw.scene.InstanceFlags.DEFAULT | rw.scene.InstanceFlags.DOUBLE_SIDED
  plane = b.add_instance("plane", size=(1.0, 1.0, 0.1), xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]], flags=two)
  cylinder = b.add_instance("cylinder", size=(0.3, 0.3, 0.0), xform=np.c_[np.eye(3), [5.0, 0.0, 0.0]], flags=two)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=1.0, angle=0.0))
  scene = b.build()
  seg = {}
  for traversal in ("software", "optix"):
    settings = rw.RenderSettings(outputs=rw.Outputs.of("seg"), performance=rw.settings.Performance(traversal=traversal))
    cams = [rw.CameraSpec(fovy=60.0), rw.CameraSpec(fovy=60.0)]
    r = rw.Renderer(rw.upload(scene), nworld=1, cameras=cams, resolution=(8, 8), settings=settings)
    assert r.features.two_sided
    up = [1.0, 0.0, 0.0, 0.0]  # half a turn about x: the cameras look along +z
    r.camera_pose.assign(np.array([[[0.0, 0.0, 0.0, *up], [5.0, 0.0, 0.0, *up]]], np.float32))
    r.render()
    seg[traversal] = required(r.outputs.seg).numpy()
  assert (seg["software"][0, 0] == plane).all() and (seg["software"][0, 1] == cylinder).all()
  np.testing.assert_array_equal(seg["optix"], seg["software"])


def _solid_glass() -> rw.Scene:
  """A floor, a glass box resting on it (its bottom face flush with the floor's top) and a glass
  sphere: `tests/render/test_dr_scenarios.py`'s glass scene."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_roughness=0.8))
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      transmission_weight=1.0,
      transmission_color=(0.6, 0.9, 0.6),
      specular_ior=1.5,
      specular_roughness=0.0,
      extras=rw.scene.OpenPbrExtras(transmission_depth=0.35),
    )
  )
  eye = np.eye(3)
  b.add_instance("box", material=floor, size=(2.0, 2.0, 0.05), xform=np.c_[eye, [0.0, 0.0, -0.05]])
  b.add_instance("box", material=glass, size=(0.3, 0.3, 0.3), xform=np.c_[eye, [-0.4, 0.0, 0.3]])
  b.add_instance("sphere", material=glass, size=(0.25, 0.25, 0.25), xform=np.c_[eye, [0.5, 0.0, 0.25]])
  b.add_light(rw.Light.sun((0.4, 0.2, -0.9), illuminance=5.0))
  b.add_environment(rw.scene.EnvironmentLight(intensity=0.5))
  return b.build()


@pytest.mark.parametrize("recipe_name", ["fast", "default"])
def test_solid_glass_matches_software(recipe_name):
  """Camera rays through solid glass on OptiX leave it as on the software path (BUG-735): a ray
  inside the sphere meets its exit face (OptiX's curve intersector reports none from inside: the
  sphere showed the floor untinted), and the box's exit face flush with the floor wins the tie as
  the software traversal's higher instance (one OptiX trace left it unordered: speckle)."""
  out = {}
  for traversal in ("software", "optix"):
    settings = recipe(recipe_name).replace(outputs=rw.Outputs.of("hdr"), gi={"enabled": False})
    settings = settings.replace(**{"performance.traversal": traversal})
    r = rw.Renderer(
      rw.upload(_solid_glass()), nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(128, 96), settings=settings
    )
    assert r.features.traversal == traversal
    r.camera_pose.assign(np.array([[[0.0, -2.0, 1.6, 0.383, 0.0, 0.0, 0.924]]], np.float32))
    r.render()
    out[traversal] = required(r.outputs.hdr).numpy()[0, 0, ..., :3].reshape(-1, 3)
    r.close()
  apart = np.abs(out["optix"] - out["software"]).max(-1) > 0.02
  assert apart.mean() < 0.005, f"{apart.mean():.3f} of the pixels differ"


_COPLANAR_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom name="flush" type="box" pos="-0.3 0 -0.1" size="0.3 0.3 0.1"/>
      <geom name="shelf" type="box" pos="0.3 0 0.2" size="0.3 0.3 0.2"/>
      <body pos="0.45 0.15 0.3">
        <freejoint/>
        <geom name="block" type="box" size="0.15 0.15 0.1"/>
      </body>
      <body pos="0.3 -0.6 -0.2">
        <freejoint/>
        <geom name="buried" type="box" size="0.1 0.1 0.1"/>
      </body>
      <camera name="top" pos="0.1 -0.3 2.5" xyaxes="1 0 0 0 1 0"/>
      <camera name="low" pos="0 -2.5 0.6" xyaxes="1 0 0 0 0.2 1"/>
    </worldbody>
  </mujoco>
"""


@pytest.mark.parametrize("move", [False, True])
def test_coplanar_and_hidden_hits_match_software(move):
  """The closest query's order (planes, then shared scenery, then the world's cell, each bounding
  the next) keeps the software traversal's ties: a static box flush with the plane, a moving block
  whose top is coplanar with a static shelf, and a block buried under the plane (its cell trace
  skipped). Seen from above and at a grazing angle."""
  out = _render_both(_COPLANAR_XML, nworld=3, move=move, cam_res=(48, 40), outputs=rw.Outputs.of(*_ALL))
  (_, ds, ss), (_, do, so) = out["software"], out["optix"]
  assert np.mean(ss == so) > 0.998
  same = ss == so
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# OptiX renderer lifetime, error reporting, rebuilds and supersampling edge cases.


_LIFETIME_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2"/>
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="mesh" mesh="ico"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="ico" scale="0.2 0.2 0.2"
        vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
    </asset>
  </mujoco>
"""


def _lifetime_setup(
  nworld: int = 2, traversal: str = "optix", *, cam_res: tuple[int, int] = (32, 32), **settings: object
) -> MjScene:
  """`make_scene` of `_LIFETIME_XML` with `traversal` (checked to be the one resolved) and
  `settings`."""
  s = make_scene(
    _LIFETIME_XML,
    nworld=nworld,
    cam_res=cam_res,
    cameras=None,
    rays=None,
    per_world=None,
    **{"performance.traversal": traversal, **settings},
  )
  assert s.r.features.traversal == traversal
  return s


def _capture(s: MjScene) -> Segments:
  """The CUDA-graph segments of a capture of `s`."""
  return s.r.capture(sim=s.d).segments  # launched without the binding check


def test_rebuild_on_setting_change_and_close():
  s = _lifetime_setup(cam_res=(16, 16)).render()
  first = optix_build(s.r)
  ref = required(s.out.rgb).numpy().copy()
  # A compiled-in build change (the `tuning` counters, same pixels): the next frame rebuilds.
  with tuning.override(counters=True):
    s.r.update_settings(**{"performance.rebuild_every": 128})  # not the default: a settings change
    with pytest.warns(UserWarning, match="rebuilding"):
      s.render()
    assert optix_build(s.r) is not first and first.closed and required(first.pipeline)._destroyed
    assert required(optix_build(s.r).pipeline).module is not None
    np.testing.assert_array_equal(required(s.out.rgb).numpy(), ref)
    second = optix_build(s.r)
    s.render()
    assert optix_build(s.r) is second  # no further rebuild
  wp.synchronize()


def test_rebuild_on_compiled_in_setting_change():
  s = _lifetime_setup(cam_res=(16, 16)).render()
  first = optix_build(s.r)
  ref = required(s.out.rgb).numpy().copy()
  # A runtime setting reuses the programs.
  s.r.update_settings(**{"performance.rebuild_every": 128})
  s.render()
  assert optix_build(s.r) is first
  # The render order is a runtime field (d0446a8e): no rebuild, same pixels.
  with warnings.catch_warnings():
    warnings.filterwarnings("error", message=".*rebuilding", category=UserWarning)
    s.r.update_settings(**{"performance.render_order": "world"})
    s.render()
  assert optix_build(s.r) is first and not first.closed
  np.testing.assert_array_equal(required(s.out.rgb).numpy(), ref)
  # A compiled-in build change that keeps the pixels (the light-cost counters,
  # `core.pipeline.platform.tuning`).
  with tuning.override(counters=True):
    s.r.update_settings(**{"performance.rebuild_every": 256})
    with pytest.warns(UserWarning, match="rebuilding"):
      s.render()
  assert optix_build(s.r) is not first and first.closed
  np.testing.assert_array_equal(required(s.out.rgb).numpy(), ref)
  # A compile-time setting read inside the traversal also rebuilds (the tone mapper is a runtime
  # setting since 1e1be6e: back-face culling is compiled into the ray flags and hit tests).
  second = optix_build(s.r)
  s.r.update_settings(**{"surfaces.backface_culling": False})
  with pytest.warns(UserWarning, match="rebuilding"):
    s.render()
  assert optix_build(s.r) is not second
  wp.synchronize()


def test_graph_after_rebuild_raises():
  s = _lifetime_setup(cam_res=(16, 16))
  graph = s.r.capture(sim=s.d)
  graph.launch(s.d)
  segments = graph.segments
  old = optix_build(s.r)
  with tuning.override(counters=True):  # a compiled-in build change (the light-cost counters): a rebuild
    s.r.update_settings(**{"performance.rebuild_every": 128})  # not the default: a settings change
    with pytest.warns(UserWarning, match="rebuilding"):
      s.render()  # closes the renderer the graph was captured with
    assert old.closed
    with pytest.raises(rw.StaleBinding):
      graph.launch(s.d)
    with pytest.raises(RuntimeError, match="capture again"):
      segments.launch()
    wp.synchronize()
    s.render()


def test_optix_errors_raise_from_launch():
  s = _lifetime_setup(cam_res=(16, 16))
  segments = _capture(s)
  pipeline = required(optix_build(s.r).pipeline)

  def fail(stream: wp.Stream, nmajor: int, minor: int, program: str | None = None) -> None:  # Pipeline.trace's
    """An OptiX launch that fails."""
    raise ValueError("injected")

  pipeline.trace = fail
  with pytest.raises(ValueError, match="injected"):
    segments.launch()
  del pipeline.trace
  wp.synchronize()


def test_one_pixel_camera_supersampled():
  out = {}
  for traversal in ("software", "optix"):
    s = _lifetime_setup(
      traversal=traversal, cam_res=(1, 1), outputs=rw.Outputs.of("rgb", "depth", "seg"), aa=rw.settings.AntiAliasing(spp=2)
    ).render()
    out[traversal] = (s.depth(), s.seg())
    if traversal == "optix":
      assert required(optix_build(s.r).supersampling).hi.flat["depth"].shape[1] == 4
  np.testing.assert_allclose(out["optix"][0], out["software"][0], rtol=2e-4, atol=1e-5)
  np.testing.assert_array_equal(out["optix"][1], out["software"][1])


def test_cell_follows_the_frame_after_a_far_move():
  """BUG-012: a world's cell is its current geometry's on refit frames too: after a move far away
  and back the cell shrinks at once, and the frame matches a fresh renderer's exactly (the
  software traversal's reset determinism)."""
  s = _lifetime_setup(cam_res=(8, 8), performance=rw.settings.Performance(rebuild_every=0))
  m, d = s.m, s.d
  qpos = np.tile(s.mjd.qpos, (d.nworld, 1))
  far = qpos.copy()
  far[:, :3] = [20.0, 0.0, 0.5]
  d.qpos.assign(far.astype(np.float32))
  mjw.forward(m, d)
  s.render()  # frame 0: built
  scene = optix_build(s.r).scene

  def upper_x() -> float:
    return float(scene.tables.cell.numpy()[rt_trace.CELL_UPPER][0])

  assert upper_x() > 19.0
  d.qpos.assign(qpos.astype(np.float32))
  mjw.forward(m, d)
  s.render()  # frame 1: refit, the cell is this frame's
  assert upper_x() < 5.0
  fresh = _lifetime_setup(cam_res=(8, 8), performance=rw.settings.Performance(rebuild_every=0))
  fresh.r.render(d)
  a = required(s.out.rgb).numpy().astype(int)
  b = required(fresh.out.rgb).numpy().astype(int)
  assert np.abs(np.subtract(a, b)).max() == 0


@pytest.mark.parametrize("moving", [False, True])
def test_light_grid_rebuild_keeps_the_rebuild_schedule(monkeypatch, moving):
  """Rebuilding the light grid on the OptiX programs leaves the rebuild schedule
  (`performance.rebuild_every`) in step with the frames. A static scene's grid shares the frame's
  refit; with a moving body the grid's structures are built apart (its body hidden), and the
  frame's are then rebuilt (not refitted from the hidden body)."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")  # no stored grid: the rebuild traces
  xml = _LIFETIME_XML if moving else _LIFETIME_XML.replace("<freejoint/>", "")
  perf = rw.settings.Performance(traversal="optix", rebuild_every=2)
  s = make_scene(xml, nworld=2, cam_res=(8, 8), lighting=rw.settings.Lights(light_grid=True), performance=perf)
  s.render()
  optix = optix_build(s.r)
  assert optix.visibility
  builds, apart = [], []
  real, real_apart = type(optix.scene).build, type(optix.scene).build_apart

  def build(scene, stream):
    builds.append(scene.stale or scene.rebuild_due())
    real(scene, stream)

  def build_apart(scene, *args, **kwargs):
    apart.append(True)
    real_apart(scene, *args, **kwargs)

  monkeypatch.setattr(type(optix.scene), "build", build)
  monkeypatch.setattr(type(optix.scene), "build_apart", build_apart)
  frames = optix.scene.frames
  s.r.rebuild_light_grid()
  s.render()  # frame 1: one refit (static), a rebuild after the grid's apart build (moving)
  s.render()  # frame 2: a rebuild frame, as without the grid rebuild
  assert len(apart) == int(moving)
  assert len(builds) == 2
  if moving:  # a static scene has no per-world group: no schedule
    assert builds == [True, True]
    assert optix.scene.frames == frames + 2


# A spot light over a floor, under a slab of moving bodies (box, capsule, cylinder, mesh): the grid
# must see through them, as the software kernel does (back faces culled: the grid's rays are not,
# the cylinder's intersection program included).
_SLAB = """
  <mujoco>
    <worldbody>
      <light pos="0 0 2.5" dir="0 0 -1" cutoff="60"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="box" pos="1.2 0 0.3" size="0.2 0.2 0.3"/>
      <body pos="0 0 1.5"><freejoint/><geom type="box" size="1 1 0.05"/></body>
      <body pos="0.5 0.5 1"><freejoint/><geom type="capsule" size="0.2 0.3" euler="90 0 0"/></body>
      <body pos="-0.5 0.5 1"><freejoint/><geom type="cylinder" size="0.2 0.3"/></body>
      <body pos="0 -0.5 1"><freejoint/><geom type="mesh" mesh="ico"/></body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="ico" scale="0.2 0.2 0.2"
        vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
    </asset>
  </mujoco>
"""


def test_moving_bodies_grid_traced_by_the_programs_as_the_software_kernel(monkeypatch):
  """With moving bodies the OptiX build traced its light grid with the software
  kernel (B67 again: Office 6.2 s on L4 instead of 0.1 s). The programs trace it, the moving
  bodies see-through (hidden in the grid's apart build), with the software kernel's culling.

  The grid box is the static instances' clipped to the lights' (`grid._bounds`: the light
  +- LIGHT_MARGIN_CELLS cells): 1 m cells put the point under the slab inside both grids (at
  0.25 m the box ends 0.5 m below the light, the point outside it). A cell that sees no light
  lists its neighbours' (`_fill_blind`), so a 1 m cell around the 0.1 m slab keeps the light."""
  from raywarp.core.shade.lights import grid

  cell_range, _ = make_cell(1, culling=True)

  @wp.kernel
  def probe(
    # Model:
    lit: Lights,
    # In:
    point: wp.vec3,
    # Out:
    count: wp.array[int],
  ) -> None:
    _cell, first, last, _far = cell_range(lit, 0, point, wp.vec3(0.0, 0.0, 1.0))
    count[0] = last - first

  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")  # no stored grid: the builds run
  traced = []
  build = grid.build

  def spy(**kwargs):
    traced.append(kwargs["tracer"] is not None)
    return build(**kwargs)

  monkeypatch.setattr(grid, "build", spy)
  grids, images = {}, {}
  counts: dict[tuple[str, str], int] = {}
  point = wp.vec3(0.85, 0.0, 0.65)  # free space below the slab, inside both grids
  for xml, name in ((_SLAB, "moving"), (_SLAB.replace("<freejoint/>", ""), "static")):
    for traversal in ("software", "optix"):
      lighting = rw.settings.Lights(light_grid=True, light_grid_cell=1.0)
      perf = rw.settings.Performance(traversal=traversal)
      s = make_scene(xml, nworld=2, cam_res=(16, 16), lighting=lighting, performance=perf)
      s.render()
      ctx = s.r.context()
      g = required(ctx.light_grid)
      grids[name, traversal] = (g.start.numpy(), g.lights.numpy())
      images[name, traversal] = s.rgb()
      assert s.scene.nlight == 1
      q = (np.asarray(point) - np.asarray(g.origin)) / g.cell
      assert np.all((q >= 0) & (q < np.asarray(g.dims)))  # exercise culling, not outside fallback
      count = wp.empty(1, dtype=int, device=s.r.device)
      wp.launch(probe, dim=1, inputs=[lights_struct(ctx, g), point, count], device=s.r.device)
      counts[name, traversal] = int(count.numpy()[0])
  assert traced == [False, True, False, True]
  for name in ("moving", "static"):
    for a, b in zip(grids[name, "optix"], grids[name, "software"], strict=True):
      np.testing.assert_array_equal(a, b)
  # Dynamic geometry is excluded from grid bounds, so total entry counts are incomparable. The
  # moving slab is hidden in the grid: the point under it keeps the light; both kernels agree.
  for name in ("moving", "static"):
    assert counts[name, "optix"] == counts[name, "software"]
  assert counts["moving", "software"] == 1
  # Moving: the slab still shadows the floor (shadow rays), as on the software path.
  assert np.abs(images["moving", "optix"] - images["moving", "software"]).max(axis=-1).mean() < 0.01


def test_cold_build_traces_the_light_grid_with_its_programs(monkeypatch):
  """B67: a build whose PTX was not cached traced its first light grid with the software kernel
  (whose grid differs from the programs' in a few (cell, light) pairs: the image depended on the
  kernel cache; Office: 6.2 s on L4 instead of 0.1 s). It waits for the programs instead."""
  from raywarp.core.build.kernel_cache import prebuilt
  from raywarp.core.shade.lights import grid
  from raywarp.core.trace.optix import pipeline

  # A cold build: no PTX cache and no bundle (both served the PTX at once, `PtxCompile.ready`; the
  # empty cache directory also leaves no stored grid, so the grid build runs).
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")
  monkeypatch.setattr(prebuilt, "ptx", lambda *_, **__: None)
  traced = []
  build = grid.build

  def spy(**kwargs):
    traced.append(kwargs["tracer"] is not None)
    return build(**kwargs)

  monkeypatch.setattr(grid, "build", spy)
  compiled = []
  compile_ptx = pipeline.PtxCompile._compile

  def counted(*args: object) -> bytes:
    compiled.append(True)
    return compile_ptx(*args)  # pyrefly: ignore[bad-argument-type]

  monkeypatch.setattr(pipeline.PtxCompile, "_compile", staticmethod(counted))
  s = make_scene(
    _LIFETIME_XML,
    cam_res=(8, 8),
    lighting=rw.settings.Lights(light_grid=True),
    performance=rw.settings.Performance(traversal="optix"),
  )
  s.render()
  assert optix_build(s.r).visibility
  assert traced == [True]
  assert compiled, "the PTX was not compiled: the build was not cold"


def test_shared_alpha_follows_every_frame():
  """B21: a shared (static) box set to alpha 0 after the first frame disappears from OptiX's
  image."""
  out = {}
  for traversal in ("software", "optix"):
    s = _lifetime_setup(traversal=traversal, cam_res=(32, 24), outputs=rw.Outputs.of("rgb", "seg")).render()
    box = int(np.flatnonzero(s.mjm.geom_type == mujoco.mjtGeom.mjGEOM_BOX)[0])
    assert (s.seg() == box).any()
    rgba = s.m.geom_rgba.numpy()
    rgba[..., box, 3] = 0.0
    s.m.geom_rgba.assign(rgba)
    s.render()
    out[traversal] = s.seg()
    assert not (out[traversal] == box).any()
  np.testing.assert_array_equal(out["optix"], out["software"])


def test_one_pipeline_binds_the_element_strides() -> None:
  """A build has one pipeline, binding the innermost stride of every launch-parameter array (its
  element size): words that do not depend on the scene or the batch, so another world count or
  table size never compiles a module."""
  s = _lifetime_setup(cam_res=(32, 24)).render()
  build = optix_build(s.r)
  pipeline = required(build.pipeline)
  ref = required(s.out.rgb).numpy().copy()
  p = build.params
  bound = tuple((offset, bytes.fromhex(value)) for offset, value in pipeline.record.bound)
  assert bound == params_lib.bound_words(p) and bound
  struct = params_lib._struct_of(p)  # the launch parameters' Warp struct type
  words = {*params_lib.word_offsets(struct).values(), *params_lib.output_offsets(struct).values()}
  strides = [v for offset, v in bound if offset not in words]  # the rest: the arrays' element sizes
  assert strides and all(int.from_bytes(v, "little") > 0 for v in strides)
  kept = p.nworld
  p.nworld = kept + 5
  try:
    assert params_lib.bound_words(p) == bound
  finally:
    p.nworld = kept
  s.render()
  assert build.pipeline is pipeline
  np.testing.assert_array_equal(required(s.out.rgb).numpy(), ref)


def test_graph_replays_the_build_s_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
  """A graph replays the build's one pipeline after eager frames; once the build is closed the
  replay raises before any segment runs."""
  s = _lifetime_setup(cam_res=(16, 16)).render()
  graph = s.r.capture(sim=s.d)
  graph.launch(s.d)
  build = optix_build(s.r)
  captured = required(build.pipeline)
  ref = required(s.out.rgb).numpy().copy()
  s.render()
  assert build.pipeline is captured
  graph.launch(s.d)
  np.testing.assert_array_equal(required(s.out.rgb).numpy(), ref)
  build.close()
  replayed: list[object] = []
  monkeypatch.setattr(wp, "capture_launch", lambda *a, **k: replayed.append(a))
  with pytest.raises(RuntimeError, match="rebuilt or closed"):
    graph.segments.launch()
  assert replayed == []  # refused before the first segment: nothing written


@pytest.mark.parametrize("move", [False, True])
def test_light_grid_and_render_bind_the_same_words(monkeypatch: pytest.MonkeyPatch, move: bool) -> None:
  """The light grid's visibility launches and the render trace the build's one pipeline, never a
  compile per grid round."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")  # no bake cache: the programs trace the grid
  s = _lifetime_setup(2, cam_res=(32, 24), lighting=rw.settings.Lights(light_grid=True)).render()
  build = optix_build(s.r)
  assert build.visibility
  first = required(build.pipeline)
  if move:
    qpos = s.d.qpos.numpy()
    qpos[1, 0] += 0.4
    s.d.qpos.assign(qpos)
    mjw.forward(s.m, s.d)
  s.render()
  assert build.pipeline is first
  assert any(program == programs.VISIBILITY for program, _chunks in first.params)  # the programs traced the grid
  wp.synchronize()
