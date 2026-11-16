"""Fakes of Isaac Lab's renderer-facing objects (no Isaac Lab needed), typed by the adapter's
Protocols (`raywarp.integrations.isaaclab.types`).

Shapes follow Isaac Lab commit 647e39c5a108e6b217014e4f57ab86efeef7cdaa
(`isaaclab/renderers/base_renderer.py`, `camera_render_spec.py`,
`sensors/camera/camera_data.py` `CameraData.allocate`, `sensors/camera/camera.py`): outputs are
(N, H, W, C) Warp arrays wrapped in a `ProxyArray`, "rgb" a strided view of "rgba"; camera
positions (N,) vec3, orientations (N,) quatf x y z w in the world convention (+X forward, +Z up),
intrinsics (N,) mat33 with cx = W / 2 (pixel centres at i + 0.5). `Simulation.segmentation`
follows Isaac's Newton Warp rule (`isaaclab_newton/renderers/segmentation.py` `_build_mapping`):
the nearest labelled ancestor of a shape's prim; instance ids group shapes by that ancestor,
semantic ids by its label payload ({type: "a,b"}), both from 2 in shape order; 0 background, 1
unlabelled; int keys.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, MutableMapping, Sequence

import newton
import numpy as np
import warp as wp
from conftest import required

from raywarp.integrations.isaaclab import types
from raywarp.integrations.isaaclab.contract import OUTPUTS
from raywarp.integrations.isaaclab.types import Buffer, SegmentInfo, ShapeIds
from raywarp.integrations.isaaclab.usd_source import UsdSource
from raywarp.scene.lighting.rig import Lighting

# Prim labels as Isaac's `get_labels` returns them: {type: [labels]}.
type Labels = dict[str, list[str]]


@dataclasses.dataclass
class ProxyArray[A]:
  """Isaac's ProxyArray (the Warp side)."""

  warp: A


@dataclasses.dataclass
class Spawn:
  """PinholeCameraCfg's fields read."""

  clipping_range: tuple[float, float] = (0.01, 1.0e6)
  distortion: object | None = None


@dataclasses.dataclass
class CameraCfg:
  """CameraCfg's fields read."""

  width: int
  height: int
  data_types: Sequence[str]
  spawn: types.Spawn | None = dataclasses.field(default_factory=Spawn)
  background_color: tuple[float, float, float] | None = None
  update_period: float = 0.0


@dataclasses.dataclass
class RenderSpec:
  """CameraRenderSpec."""

  cfg: types.CameraCfg
  num_instances: int
  device: str = "cpu"


@dataclasses.dataclass
class CameraData:
  """CameraData: the output buffers and info, allocated like `CameraData.allocate`."""

  output: dict[str, ProxyArray[Buffer]]
  info: MutableMapping[str, object]

  @staticmethod
  def allocate(names: list[str], n: int, width: int, height: int, device: str = "cpu") -> CameraData:
    """(N, H, W, C) buffers per output; "rgb" aliases "rgba" (strided, like Isaac Lab)."""
    wanted = set(names) | ({"rgba"} if "rgb" in names else set())
    out: dict[str, ProxyArray[Buffer]] = {}
    for name in sorted(wanted - {"rgb"}):
      c = OUTPUTS[name]
      out[name] = ProxyArray(wp.zeros((n, height, width, c.channels), dtype=c.dtype, device=device))
    if "rgb" in names:
      rgba = out["rgba"].warp
      strides = (height * width * 4, width * 4, 4, 1)
      out["rgb"] = ProxyArray(
        wp.array(ptr=rgba.ptr, shape=(n, height, width, 3), strides=strides, dtype=wp.uint8, device=device)
      )
    return CameraData(out, dict.fromkeys(out))


@dataclasses.dataclass
class Config:
  """RaywarpRendererCfg's fields (same defaults)."""

  settings: Mapping[str, object] = dataclasses.field(default_factory=dict)
  lighting: Lighting | None = None
  load_textures: bool = True
  depth_clipping_behavior: str = "none"
  use_cuda_graph: bool = True


@dataclasses.dataclass
class Simulation:
  """The simulation side: a Newton model, a state whose body poses the physics publishes, prim
  labels and material bindings by prim path."""

  model: newton.Model
  state: newton.State
  labels: dict[str, Labels] = dataclasses.field(default_factory=dict)
  materials: list[str] = dataclasses.field(default_factory=list)
  source: UsdSource | None = None

  def usd_source(self) -> UsdSource | None:
    """Optional actual USD source used by focused stage integration cases."""
    return self.source

  def body_poses(self) -> wp.array[wp.transformf, [int]] | None:
    """The physics' body poses (Newton's layout: borrowed; None without bodies)."""
    return self.state.body_q

  def _match(self, path: str) -> tuple[Labels, str] | None:
    """(labels, path) of the nearest labelled ancestor of `path` (the prim included)."""
    while path:
      if path in self.labels:
        return self.labels[path], path
      path = path.rpartition("/")[0]
    return None

  def segmentation(self, kind: str) -> tuple[ShapeIds, SegmentInfo]:
    """Isaac's Newton Warp ids per shape and info of `kind` (see the module doc)."""
    ids = np.ones(len(self.model.shape_label), np.int32)
    groups: dict[object, int] = {}
    labels: dict[int, object] = {}
    semantics: dict[int, object] = {}
    for s, path in enumerate(self.model.shape_label):
      match = self._match(path)
      if match is None:
        continue
      payload = {t: ",".join(v) for t, v in match[0].items()}
      key: object = match[1] if kind == "instance_segmentation" else tuple(sorted(payload.items()))
      if key not in groups:
        groups[key] = 2 + len(groups)
        labels[groups[key]] = match[1] if kind == "instance_segmentation" else payload
        semantics[groups[key]] = payload
      ids[s] = groups[key]
    if kind == "semantic_segmentation":
      return ids, {"idToLabels": {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}, **labels}}
    reserved = {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}}
    return ids, {"idToLabels": {0: "BACKGROUND", 1: "UNLABELLED", **labels}, "idToSemantics": {**reserved, **semantics}}

  def shape_materials(self) -> list[str]:
    """Material path per shape."""
    return self.materials


def boxes(nworld: int = 2, device: str = "cpu") -> tuple[newton.Model, newton.State]:
  """A ground plane and a free 0.6 x 0.4 x 0.5 box at (0, 0, 0.5) per world, labelled like Isaac
  Lab's replicated prims (`/World/envs/env_<w>/Box/geom`)."""
  env = newton.ModelBuilder()
  body = env.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), label="Box")
  env.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.8, 0.2, 0.1), label="Box/geom")
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(env, nworld)
  model = scene.finalize(device=device)
  worlds = required(model.shape_world).numpy()
  model.shape_label = [
    "/World/ground" if w < 0 else f"/World/envs/env_{w}/{label}" for w, label in zip(worlds, model.shape_label, strict=True)
  ]
  return model, model.state()


def intrinsics(n: int, width: int, height: int, hfov_deg: float = 60.0, device: str = "cpu") -> wp.array[wp.mat33, [int]]:
  """(n,) Isaac pixel intrinsics of a centred pinhole camera with horizontal fov `hfov_deg`."""
  fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
  k = np.array([[fx, 0.0, 0.5 * width], [0.0, fx, 0.5 * height], [0.0, 0.0, 1.0]], np.float32)
  return wp.array(np.repeat(k[None], n, axis=0), dtype=wp.mat33, device=device)


def poses(
  eyes: list[tuple[float, float, float]],
  quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
  device: str = "cpu",
) -> tuple[wp.array[wp.vec3, [int]], wp.array[wp.quatf, [int]]]:
  """(N,) camera positions and world-convention orientations (default: looking along +X)."""
  pos = wp.array(np.asarray(eyes, np.float32), dtype=wp.vec3, device=device)
  rot = wp.array(np.tile(np.asarray(quat_xyzw, np.float32), (len(eyes), 1)), dtype=wp.quatf, device=device)
  return pos, rot
