"""Helpers of the GPU-free OptiX tests: a parity scene and the Features of OptiX builds."""

from __future__ import annotations

import dataclasses
import types

import mujoco
import mujoco_warp as mjw
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.kernels.abi.features import Features

# Meshes and every primitive type (the parity scene of `test_optix.py`, static).
PARITY_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
      <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2"/>
      <geom type="ellipsoid" pos="0.3 0.6 0.3" size="0.1 0.2 0.15"/>
      <geom type="mesh" mesh="tet" pos="0.1 -0.5 0.35"/>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="tet" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/>
    </asset>
  </mujoco>
"""


def preset_features(settings: rw.RenderSettings, xml: str = PARITY_XML) -> Features:
  """The Features of an OptiX build of `xml` under `settings`, resolved on the CPU (software)."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  r = rw.Renderer(scene, nworld=1, resolution=(8, 8), settings=settings.replace(**{"performance.traversal": "software"}))
  try:
    return dataclasses.replace(r.features, traversal="optix", ordered_traversal=False, mesh_frames=False, out_word=0)
  finally:
    r.close()


def scene_flags(**flags: object) -> types.SimpleNamespace:
  """The feature flags of an `accel.OptixScene` that `programs.build` reads (meshes, curves,
  customs)."""
  base = {
    "has_mesh": True,
    "has_box": True,
    "has_instances": True,
    "has_shared": False,
    "translucent_triangles": False,
    "translucent_curves": (False, False),
    "shared_visible": False,
    "primitive_types": (4, 5),  # ellipsoid, cylinder
    "shared_primitive_types": (),
    "has_world_curves": True,
    "has_shared_curves": False,
    "has_omm": False,
  }
  return types.SimpleNamespace(**(base | flags))


def cuda_source(module: wp.Module) -> str:
  """The module's CUDA source as Warp's build would generate it (private Warp API: test only)."""
  from warp._src.context import ModuleBuilder

  options = module.resolve_options(wp.config, block_dim=256) | {"output_arch": None, "block_dim": 256}
  return ModuleBuilder(module, options).codegen("cuda")
