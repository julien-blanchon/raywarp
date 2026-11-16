"""Adapter conformance kit (`fixtures/conformance.py`): MuJoCo, USD, Newton and Isaac Lab
(through fakes of Isaac Lab's renderer objects, `isaaclab_fake`)."""

import dataclasses
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import REPO, UNLIT_FAST_CHANGES
from fixtures import conformance

import raywarp as rw
import raywarp.formats.usd
import raywarp.integrations.mujoco
from raywarp.core.pipeline import warmup
from raywarp.core.post.grading.color import srgb_to_linear

pxr = pytest.importorskip("pxr")
newton = pytest.importorskip("newton")

from fixtures import isaaclab_fake as fk
from pxr import (
  Gf,
  Usd,
  UsdGeom,
)

import raywarp.integrations.newton  # after the skip
from raywarp.integrations.isaaclab import Cameras
from raywarp.integrations.isaaclab.session import CameraSession

# One feature scene: a floor slab, a box, a sphere and a capsule, colored (MJCF and USD below).
SHAPES = [
  # (name, kind, position, size (MuJoCo), rgba)
  ("floor", "box", (0.0, 0.0, -0.05), (1.5, 1.5, 0.05), (0.7, 0.7, 0.7, 1.0)),
  ("crate", "box", (0.5, 0.2, 0.25), (0.2, 0.15, 0.25), (0.9, 0.2, 0.1, 1.0)),
  ("ball", "sphere", (-0.4, 0.1, 0.3), (0.3, 0.0, 0.0), (0.1, 0.4, 0.9, 1.0)),
  ("pill", "capsule", (0.0, -0.5, 0.4), (0.1, 0.2, 0.0), (0.2, 0.8, 0.3, 1.0)),
]
CAMERA = rw.CameraSpec(fovy=50.0)
RES = (48, 36)


def _mjcf() -> str:
  """The feature scene as MJCF (geoms on the world body)."""
  geoms = "".join(
    f'<geom name="{n}" type="{k}" pos="{" ".join(map(str, p))}" size="{" ".join(str(v) for v in s if v)}" rgba="{" ".join(map(str, c))}"/>'
    for n, k, p, s, c in SHAPES
  )
  return f'<mujoco><worldbody><light pos="0 0 3" dir="0 0 -1"/>{geoms}</worldbody></mujoco>'


def _usd(path: Path) -> Path:
  """The feature scene as a USD stage (Z up, metres)."""
  stage = Usd.Stage.CreateNew(str(path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  for n, k, p, s, c in SHAPES:
    if k == "box":
      prim = UsdGeom.Cube.Define(stage, f"/World/{n}")
      prim.CreateSizeAttr(2.0)
      prim.AddScaleOp().Set(Gf.Vec3f(*s))
    elif k == "sphere":
      prim = UsdGeom.Sphere.Define(stage, f"/World/{n}")
      prim.CreateRadiusAttr(s[0])
    else:
      prim = UsdGeom.Capsule.Define(stage, f"/World/{n}")
      prim.CreateRadiusAttr(s[0])
      prim.CreateHeightAttr(2.0 * s[1])
      prim.CreateAxisAttr("Z")
    prim.AddTranslateOp(opSuffix="pos", precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*p))
    xf = UsdGeom.Xformable(prim)
    xf.SetXformOpOrder([xf.GetOrderedXformOps()[-1], *xf.GetOrderedXformOps()[:-1]])
    prim.CreateDisplayColorAttr([Gf.Vec3f(*c[:3])])
  stage.GetRootLayer().Save()
  return path


def _pose() -> np.ndarray:
  """(7,) a free camera at (0, -3, 1.6) looking at (0, 0, 0.2)."""
  eye, target = np.array([0.0, -3.0, 1.6]), np.array([0.0, 0.0, 0.2])
  back = (eye - target) / np.linalg.norm(eye - target, keepdims=True)
  right = np.cross([0.0, 0.0, 1.0], back)
  right /= np.linalg.norm(right, keepdims=True)
  up = np.cross(back, right)
  q = wp.quat_from_matrix(wp.mat33(*np.stack([right, up, back], axis=1).flatten().tolist()))
  return np.array([*eye, q[0], q[1], q[2], q[3]], np.float32)


def _mujoco_case() -> conformance.ConformanceCase[mjw.Data]:
  """The MuJoCo adapter (a DirectAdapter) on the MJCF feature scene."""
  mjm = mujoco.MjModel.from_xml_string(_mjcf())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  case = conformance.direct_case("mujoco", rw.integrations.mujoco, mjm, m, d)
  return dataclasses.replace(case, step=lambda: mjw.step(m, d))


def _usd_case(tmp: Path) -> conformance.ConformanceCase:
  """The USD adapter (a HostAdapter) on the USD feature scene (a static stage)."""
  return conformance.host_case("usd", rw.formats.usd, str(_usd(tmp / "scene.usda")), None, device="cpu")


def _newton_case() -> conformance.ConformanceCase:
  """The Newton adapter (a HostAdapter) on Newton's import of the MJCF feature scene."""
  builder = newton.ModelBuilder()
  builder.add_mjcf(_mjcf())
  model = builder.finalize(device="cpu")
  return conformance.host_case("newton", rw.integrations.newton, model, model.state(), device="cpu")


@pytest.fixture(scope="module")
def mujoco_scene() -> tuple[rw.GpuScene[mjw.Data], mjw.Data]:
  """(GpuScene, mjw.Data) of the MJCF feature scene: the reference of `compare`."""
  case = _mujoco_case()
  return case.scene, case.sim


@pytest.fixture(scope="module", params=["mujoco", "usd", "newton"])
def case(request, tmp_path_factory):
  """One ConformanceCase per adapter."""
  if request.param == "mujoco":
    return _mujoco_case()
  if request.param == "usd":
    return _usd_case(tmp_path_factory.mktemp("usd"))
  return _newton_case()


# Base-color tolerance per adapter against the MuJoCo reference: Newton's import of the MJCF
# stores MuJoCo's linear rgba into shape_color, which Newton reads as sRGB (decided: follow
# Newton; `conformance` module doc): geometry only, the gap is that decode (0.24 here).
_ALBEDO = {"usd": conformance.Tolerance().albedo, "newton": 1.0}


def test_case_renders_the_reference_scene(case, mujoco_scene: tuple[rw.GpuScene[mjw.Data], mjw.Data]) -> None:
  """Each adapter's feature scene agrees with the MuJoCo adapter's (geometry, base colors)."""
  if case.name == "mujoco":
    pytest.skip("the reference")
  tol = conformance.Tolerance(albedo=_ALBEDO[case.name])
  result = conformance.compare(mujoco_scene, (case.scene, case.sim), camera=CAMERA, pose=_pose(), resolution=RES, tol=tol)
  assert not result.failures, result
  if case.name == "newton":
    assert result.albedo > 0.1, result  # the known decode gap, not a silent match


# The USD stage has no light and Newton's MJCF import drops the MJCF light: their lit first frame
# is black by construction (this case checks compiles, not images): the first-image warning is
# expected there.
@pytest.mark.filterwarnings("ignore:raywarp. camera 0's first image is black:RuntimeWarning")
def test_case_per_world_randomisation_never_recompiles(case):
  """Each adapter's per-world tables (instance colors, exposure) are written without
  reconfiguring or compiling."""
  decl = rw.PerWorld(material=["color"], env=["exposure"])
  r = rw.Renderer(case.scene, nworld=1, cameras=[CAMERA], resolution=(16, 12), per_world=decl)
  r.camera_pose.assign(_pose().reshape(1, 1, 7))  # the free camera sees the scene (unposed: inside the floor, black)
  colors = np.random.default_rng(0).uniform(0.0, 1.0, (1, case.scene.ninstance, 4)).astype(np.float32)
  colors[..., 3] = 1.0
  conformance.check_no_recompile(r, case.sim, {"instance.color": colors, "exposure": np.array([1e-4], np.float32)})


def test_case_load_report_drops_nothing(case):
  """A host adapter's report on the feature scene: nothing dropped or approximated (a
  DirectAdapter makes no report)."""
  if case.report is None:
    pytest.skip("a DirectAdapter (MuJoCo) has no load report")
  assert conformance.report_mismatches(case.report, {}) == {}


def test_case_step_and_render_replay_in_one_graph(case):
  """Step + render in one CUDA graph replays the eager frame (skipped without CUDA)."""
  r = rw.Renderer(case.scene, nworld=1, cameras=[CAMERA], resolution=(16, 12))
  r.camera_pose.assign(_pose().reshape(1, 1, 7))  # a replay of a black frame would prove nothing
  if not conformance.check_graph_replay(r, case.step, case.sim):
    pytest.skip("graph capture needs CUDA")


def test_load_report_counters_on_a_real_asset():
  """The load report of a shipped feature scene keeps its counters (nothing newly dropped)."""
  root = REPO / "tests" / "data" / "scenes"
  scene = rw.formats.usd.load(str(root / "materials.usda"))
  assert conformance.report_mismatches(scene.report, EXPECTED_MATERIALS_REPORT) == {}


def _isaac_frame(cams: Cameras, names: list[str]) -> tuple[CameraSession, fk.CameraData]:
  """(session, CameraData) of initialized Isaac cameras (fake) seen from `_pose()`."""
  w, h = RES
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(w, h, names), 1))
  data = fk.CameraData.allocate(names, 1, w, h)
  cams.set_outputs(session, data.output)
  pose = _pose()
  # Isaac's world-convention orientation (+X forward, +Y left, +Z up) of the same look-at, built
  # from the eye and target directly (not from the adapter's frame conversion).
  forward = np.array([0.0, 0.0, 0.2]) - pose[:3]
  forward /= np.linalg.norm(forward, keepdims=True)
  left = np.cross([0.0, 0.0, 1.0], forward)
  left /= np.linalg.norm(left, keepdims=True)
  frame = np.stack([forward, left, np.cross(forward, left)], axis=1)
  q = wp.quat_from_matrix(wp.mat33(*frame.flatten().tolist()))
  fy = 0.5 * h / np.tan(np.radians(CAMERA.fovy) / 2.0)
  k = wp.array([wp.mat33(fy, 0.0, 0.5 * w, 0.0, fy, 0.5 * h, 0.0, 0.0, 1.0)], dtype=wp.mat33, device="cpu")
  rot = wp.array([q], dtype=wp.quatf, device="cpu")
  cams.update_camera(
    session, fk.ProxyArray(wp.array([pose[:3]], dtype=wp.vec3, device="cpu")), fk.ProxyArray(rot), fk.ProxyArray(k)
  )
  cams.update_transforms()
  cams.render(session)
  return session, data


def _isaac_sim() -> fk.Simulation:
  """The fake Isaac simulation of the MJCF feature scene through Newton."""
  builder = newton.ModelBuilder()
  builder.add_mjcf(_mjcf())
  model = builder.finalize(device="cpu")
  return fk.Simulation(model, model.state())


def test_isaaclab_renders_the_same_scene(mujoco_scene: tuple[rw.GpuScene[mjw.Data], mjw.Data]) -> None:
  """Isaac Lab's renderer calls (pose in its world convention, its K, its NHWC outputs) render the
  MJCF scene (through Newton) like the MuJoCo adapter (geometry: the W41 color gap as above)."""
  cams = Cameras(fk.Config(settings=UNLIT_FAST_CHANGES))
  cams.initialize(_isaac_sim())
  _, data = _isaac_frame(cams, ["depth", "albedo"])
  depth = data.output["depth"].warp.numpy()[0, ..., 0]
  albedo = srgb_to_linear(data.output["albedo"].warp.numpy()[0, ..., :3] / 255.0)
  ref = conformance.render_view(*mujoco_scene, CAMERA, _pose(), RES)
  result = conformance.measure(ref, (depth, albedo.astype(np.float32)), tol=conformance.Tolerance(albedo=1.0))
  assert not result.failures and result.coverage > 0.99, result


@wp.kernel(module="unique")  # its own module: no other test loads it
def _fresh(a: wp.array1d[wp.float32]):
  a[wp.tid()] = 2.0


def test_module_loads_sees_a_module_new_to_the_process() -> None:
  """The kit's compile counter records a module the process had not loaded (the no-recompile
  checks are not vacuous)."""
  with conformance.module_loads() as loads:
    wp.launch(_fresh, 2, inputs=[wp.zeros(2, dtype=wp.float32, device="cpu")], device="cpu")
  assert loads == [warmup.kernel(_fresh).module.name]


def test_isaaclab_material_writes_never_recompile() -> None:
  """Prepared canonical material writes and intrinsics updates retain the compiled renderer."""
  sim = _isaac_sim()
  sim.materials = [f"/World/Looks/{label}" for label in sim.model.shape_label]
  cams = Cameras(fk.Config(settings=dict(rw.FAST)))
  cams.initialize(sim)
  colors = wp.array([wp.vec3(1.0)] * len(sim.materials), dtype=wp.vec3, device="cpu")
  writer = cams.material_writer(sim.materials, colors)
  writer()  # Isaac prepares and synchronizes material storage before creating cameras.
  session, data = _isaac_frame(cams, ["rgb", "albedo"])
  r = session.renderer
  before = data.output["albedo"].warp.numpy().copy()
  features = r.features
  with conformance.module_loads() as loads:
    colors.assign(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(sim.materials), 1)))
    writer()
    cams.render(session)
  after = data.output["albedo"].warp.numpy()
  assert np.any(after != before)
  assert after[..., 2].sum() > after[..., 0].sum()
  assert r.features is features and not loads, loads
  conformance.check_no_recompile(r, sim.state, {"camera.intrinsics": np.array([[[30.0, 30.0, 1.0, 0.0]]], np.float32)})
  cams.close()


# Counters of tests/data/scenes/materials.usda (a change must be explained). Its primitives'
# materials are untextured: `primitive.textures_without_uv` is reported only for textured ones.
EXPECTED_MATERIALS_REPORT: dict[str, int] = {
  "approximated:UsdPreviewSurface.useSpecularWorkflow": 1,
}
