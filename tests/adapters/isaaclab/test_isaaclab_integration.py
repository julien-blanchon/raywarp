"""Isaac Lab end to end (runs only where Isaac Lab 3.0 is importable, on CUDA): a two-environment
Newton-physics scene seen by two cameras, Isaac's own Newton Warp renderer and raywarp's; both
render the same geometry (`conformance.measure`), in Isaac's output contract."""

import numpy as np
import pytest
import warp as wp

isaaclab = pytest.importorskip("isaaclab")
if not wp.get_cuda_device_count():
  pytest.skip("Isaac Lab cameras need CUDA", allow_module_level=True)

import isaaclab.sim as sim_utils
import torch
from fixtures import conformance
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import (
  InteractiveScene,
  InteractiveSceneCfg,
)
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import (
  MJWarpSolverCfg,
  NewtonCfg,
)
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from raywarp.integrations.isaaclab.cfg import RaywarpRendererCfg

W, H = 96, 64
NAMES = ["rgb", "depth", "distance_to_camera", "instance_segmentation"]


def _camera(renderer_cfg, names):
  """A pinhole camera per environment rendered by `renderer_cfg`."""
  spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0))
  return CameraCfg(
    prim_path="{ENV_REGEX_NS}/" + ("PbrCamera" if isinstance(renderer_cfg, RaywarpRendererCfg) else "NewtonCamera"),
    height=H,
    width=W,
    update_period=0.0,
    update_latest_camera_pose=True,
    data_types=names,
    renderer_cfg=renderer_cfg,
    spawn=spawn,
  )


@configclass
class _SceneCfg(InteractiveSceneCfg):
  """Ground, a falling red cube, two lights and the two cameras."""

  # A kinematic box as ground (as Isaac's RenderBenchmark task and
  # internal/benchmarks/lab/isaac_runtime.py): the terrain importer's plane is a Nucleus asset,
  # which needs Kit.
  ground = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ground",
    spawn=sim_utils.CuboidCfg(
      size=(3.0, 3.0, 0.1),
      rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
      mass_props=sim_utils.MassCfg(mass=1.0),
      collision_props=sim_utils.UsdPhysicsCollisionCfg(),
      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
  )
  cube = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
      size=(0.4, 0.4, 0.4),
      rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
      mass_props=sim_utils.MassCfg(mass=1.0),
      collision_props=sim_utils.UsdPhysicsCollisionCfg(),
      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
  )
  # Isaac's RenderBenchmark lights (the adapter renders the authored lighting).
  dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
  distant_light = AssetBaseCfg(
    prim_path="/World/LightDistant",
    spawn=sim_utils.DistantLightCfg(intensity=200.0, normalize=True),
    init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3251, 0.3251, 0.0, 0.8881)),
  )
  newton_camera = _camera(NewtonWarpRendererCfg(), ["depth"])
  pbr_camera = _camera(RaywarpRendererCfg(), NAMES)


def test_raywarp_renders_an_isaac_lab_scene():
  """Both renderers see the cube where physics puts it (depth agrees), in Isaac's layouts."""
  sim_cfg = SimulationCfg(dt=1.0 / 120.0, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1))
  with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
    sim._app_control_on_stop_handle = None
    scene = InteractiveScene(_SceneCfg(num_envs=2, env_spacing=4.0))
    sim.reset()
    eyes = scene.env_origins.clone() + torch.tensor([-2.0, 0.0, 0.8], device=scene.device)
    targets = scene.env_origins.clone() + torch.tensor([0.0, 0.0, 0.2], device=scene.device)
    for name in ("newton_camera", "pbr_camera"):
      scene[name].set_world_poses_from_view(eyes, targets)
    for _ in range(30):  # the cube falls onto the ground
      sim.step()
      scene.update(sim.get_physics_dt())
    ours, ref = scene["pbr_camera"].data, scene["newton_camera"].data
    rgb = ours.output["rgb"].torch
    assert rgb.shape == (2, H, W, 3) and rgb.dtype == torch.uint8 and rgb.float().mean() > 5.0
    assert ours.output["instance_segmentation"].torch.dtype == torch.int32
    assert "idToLabels" in ours.info["instance_segmentation"]
    for env in range(2):
      depth = ours.output["depth"].torch[env, ..., 0].cpu().numpy()
      ref_depth = ref.output["depth"].torch[env, ..., 0].cpu().numpy()
      zeros = np.zeros((H, W, 3), np.float32)
      result = conformance.measure((ref_depth, zeros), (depth, zeros), tol=conformance.Tolerance(coverage=0.97, depth=0.02))
      assert not result.failures, (env, result)
