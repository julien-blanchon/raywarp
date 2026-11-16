"""Bindings: how per-step simulator state reaches the kernels, and the slot vocabulary.

Vocabulary: a slot is a named array the kernels read, `<group>.<quantity>` with the groups
`instance`, `shared`, `material`, `light`, `camera` and `flex` (`Slots`). Its kernel field (a
`frame.Frame` field) and the struct field / kernel parameter holding it are the slot name with "."
replaced by "_" ("instance.pos" -> `instance_pos`, "light.radius" -> `light_radius`); derived
arrays extend that name (`prev_instance_pos`, `instance_rot_trace`). The core and its kernels use
only this vocabulary: "instance" is a rendered shape (a MuJoCo geom, a Newton shape, a USD
gprim), "material" its material, "flex" a deformable. Simulator names (MuJoCo Warp's `geom_xpos`,
`mat_rgba`, ...) appear only in the adapters, each mapping them to slots in one table
(`raywarp/integrations/mujoco/binding.py`).

Contract (`Binding[S]`, S the simulator-state type a Renderer's `render(sim)` takes):

* The arrays of a frame are, by priority: a Renderer's per-world overrides, the binding's arrays
  for the call (`Binding.slots(sim)`), the scene's tables (`GpuScene.tables`). Bindings keep no
  per-step arrays: they re-resolve them from the `sim` passed to every `Renderer.render(sim)`, so
  swapped or reallocated simulator state is picked up eagerly and detected (`StaleBinding`) by
  captured graphs, which also check the binding's other per-call inputs (`Binding.sources`).
* Layouts: per-instance / per-material / per-light tables, the light and scene-camera poses and
  the deformable vertices are (1 or nworld, n) (kernels read row `min(worldid, shape[0] - 1)`:
  one row is shared by every world; an adapter tiles a K-row table to nworld rows).
  Instance poses are (nworld, ncolumn) vec3 positions and mat33 rotations (the MuJoCo Warp
  layout, so MuJoCo needs no pose kernel).
* Sharing: which instances every world shares is decided by the Renderer
  (`core/pipeline/worlds/sharing.py`) from `Binding.dynamic()` (the instances the simulator moves
  per world) and its `PerWorld` declarations. An aliasing binding (MuJoCo) holds every instance in
  its poses (a shared one is read from world 0's row); a pose-writing binding (`writes_poses`:
  host scenes, Newton) owns no arrays either: the Renderer allocates the pose buffers of its
  per-world columns once (`allocate`), fills its shared pose table ("shared.pos" / "shared.rot",
  (nshared,)) once (`write_shared`), and every `render(sim)` lets the binding write the poses it
  drives (`write`, one kernel or nothing for static scenes) before the frame's slots are resolved.
* Static scenery: `shared_motion` names the arrays the shared instances' poses come from
  (`RowsSource` / `TransformSource`), checked on the device every render to stay where the first
  frame had them (`core/pipeline/worlds/validator.py`; a move raises), as are the binding's
  `feature_checks` (live data that needs a compiled-out feature).
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Final, Protocol, TypedDict, cast, final, overload, override

import numpy as np
import warp as wp

from raywarp._base.types import ColorRule
from raywarp.core.pipeline.gpu_scene.deformables import Deformables

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import AnyDeviceArray, Ids, Mask, PerWorld, Table
  from raywarp._base.types import tex_affine, uv_xform
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.pipeline.frame import Frame

# Slot name -> array. B: batched table (1 or nworld, n) (`dims.Table`); D: per-step pose (nworld,
# ncolumn); S: shared pose table (nshared,), Renderer-owned. Every key is optional: a scene's
# tables, a binding's per-call arrays and a Renderer's overrides each hold some of them. The value
# types are shaped annotations (forward references: the dimension aliases are type-only).
Slots = TypedDict(
  "Slots",
  {
    "instance.mesh": "Table[int, int, int]",  # B: mesh / height field id
    "instance.material": "Table[int, int, int]",  # B: material id, -1 none
    "instance.size": "Table[wp.vec3, int, int]",  # B: primitive half sizes / radii
    "instance.color": "Table[wp.vec4, int, int]",  # B: color and alpha (alpha 0: invisible to rays)
    "instance.shading": "Table[int, int, int]",  # B: ShadingFlag bits (optional: absent = 0)
    "instance.pos": "PerWorld[wp.vec3, int, int]",  # D: per-world pose columns (nworld, ncolumn)
    "instance.rot": "PerWorld[wp.mat33, int, int]",  # D: rotations R; a mesh instance's s R (uniform scale s > 0)
    "instance.visible": "Table[int, int, int]",  # (nworld, nvisible): 0 hides (Renderer-owned, optional)
    "shared.pos": "wp.array[wp.vec3, [int]]",  # S: the shared instances' poses by shared slot
    "shared.rot": "wp.array[wp.mat33, [int]]",  # S
    "material.base_color": "Table[wp.vec4, int, int]",  # B
    "material.emission": "Table[float, int, int]",  # B: emission strength x base color
    "material.mirror": "Table[float, int, int]",  # B: mirror reflectance
    "material.texture": "wp.array[int, [int, int, int]]",  # B (nmat, prepared roles): texture id per TexRole, -1 none
    "material.tex_affine": "wp.array[tex_affine, [int, int, int]]",  # B (nmat, prepared roles): texel map per TexRole
    "material.tex_variation": "Table[wp.vec4, int, int]",  # B: texture variation (u, v offset, UV rotation, hue)
    "material.tex_repeat": "Table[wp.vec2, int, int]",  # B
    "material.tex_uniform": "Table[bool, int, int]",  # B
    "material.uv_world": "Table[uv_xform, int, int]",  # B: world-projected UV map, zeros: mesh UVs (optional)
    "material.alpha_cutoff": "Table[float, int, int]",  # B: alpha test cutoff on the opacity map, 0 none (optional)
    "material.base_diffuse_roughness": "Table[float, int, int]",  # optional EON roughness
    "material.base_metalness": "Table[float, int, int]",  # B
    "material.specular_roughness": "Table[float, int, int]",  # B: perceptual roughness
    "material.specular_weight": "Table[float, int, int]",  # optional canonical weight carrier
    "material.reflectance": "Table[float, int, int]",  # B: dielectric reflectance, F0 = 0.16 r^2
    "material.specular_anisotropy": "Table[wp.vec2, int, int]",  # optional: anisotropy a, rotation [rad]
    "material.fuzz": "Table[wp.vec2, int, int]",  # optional: fuzz weight, Charlie roughness
    "material.fuzz_color": "Table[wp.vec3, int, int]",  # optional: fuzz color
    "material.coat_color": "Table[wp.vec3, int, int]",  # optional: OpenPBR coat color (T^2)
    "material.coat_eta": "Table[float, int, int]",  # optional: the coat's IOR (derived from coat_ior)
    "light.type": "Table[int, int, int]",  # B: LightType
    "light.cast_shadow": "Table[bool, int, int]",  # B
    "light.active": "Table[bool, int, int]",  # B
    "light.color": "Table[wp.vec3, int, int]",  # B: linear color (normalised)
    "light.cutoff": "Table[float, int, int]",  # B: spot outer cone [degrees]
    "light.cone_inner": "Table[float, int, int]",  # B: inner / outer cone angle (optional: absent = 0.75)
    "light.strength": "Table[float, int, int]",  # B: lux (directional) / candela
    "light.range": "Table[float, int, int]",  # B: falloff window radius [m]; 0: no falloff (point / spot)
    "light.attenuation": "Table[wp.vec3, int, int]",  # B: optional classic (constant, linear, quadratic); absent: (1, 0, 0)
    "light.radius": "Table[float, int, int]",  # B: emitter radius (soft shadows); directional: disc angular radius [rad]
    "light.pos": "Table[wp.vec3, int, int]",  # B
    "light.dir": "Table[wp.vec3, int, int]",  # B: unit direction the light travels (rect / disk: normal)
    "light.shape": "Table[int, int, int]",  # B: LightShape (lighting.area_lights="ltc"; absent: all NONE)
    "light.axis": "Table[wp.vec3, int, int]",  # B: rect width / tube axis direction (unit)
    "light.size": "Table[wp.vec2, int, int]",  # B: rect half extents / tube (half length, -) [m]
    "camera.pos": "Table[wp.vec3, int, int]",  # B: scene cameras (mounts)
    "camera.rot": "Table[wp.mat33, int, int]",  # B
    "flex.vert": "Table[wp.vec3, int, int]",  # B: deformable vertices (nvert columns)
  },
  total=False,
)

# Slot name -> kernel field: the slot name with "." -> "_".
SLOTS: Final = {slot: slot.replace(".", "_") for slot in Slots.__annotations__}

# Slots that are never per-world overridable (`Renderer.per_world`). Instance poses are not listed:
# they come from the binding and are overridable per world for pose bindings; the light poses are
# ordinary tables, overridable like the other light fields.
FIXED_SLOTS: Final = frozenset({"shared.pos", "shared.rot", "camera.pos", "camera.rot", "flex.vert"})


def slot(slots: Slots, name: str) -> AnyDeviceArray | None:
  """The array of slot `name` (a name known only at runtime) in `slots`, None when absent."""
  return cast("dict[str, AnyDeviceArray]", slots).get(name)


def merged(slots: Slots, arrays: Mapping[str, AnyDeviceArray]) -> Slots:
  """A copy of `slots` updated with the entries of `arrays` named by a slot (others ignored)."""
  out = cast("dict[str, AnyDeviceArray]", dict(slots))
  out.update({name: a for name, a in arrays.items() if name in SLOTS})
  return cast("Slots", out)


class StaleBinding(RuntimeError):
  """A captured RenderGraph was launched with arrays it was not captured with."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class RowsSource:
  """Shared instances' poses aliased per world (MuJoCo: `d.geom_xpos` / `d.geom_xmat`).

  Attributes:
    pos: (nworld, n) positions.
    rot: (nworld, n) rotations.
    index: (nshared,) pose column of each shared instance (-1: none).
  """

  pos: PerWorld[wp.vec3, int, int]
  rot: PerWorld[wp.mat33, int, int]
  index: Ids[int]


@final
@dataclasses.dataclass(frozen=True, slots=True)
class TransformSource:
  """Shared instances' poses from one source transform each (Newton: `model.shape_transform`).

  Attributes:
    transforms: (m,) source transforms.
    index: (nshared,) transform of each shared instance (-1: none).
  """

  transforms: wp.array[wp.transformf, [int]]
  index: Ids[int]


@final
@dataclasses.dataclass(frozen=True, slots=True)
class PoseBuffers[NWorld: IntVar, NColumn: IntVar]:
  """Renderer-owned instance poses of a pose-writing binding: the per-world columns.

  Attributes:
    pos: (nworld, ncolumn) positions ("instance.pos").
    rot: (nworld, ncolumn) rotations ("instance.rot").
  """

  pos: wp.array[wp.vec3, [NWorld, NColumn]]
  rot: wp.array[wp.mat33, [NWorld, NColumn]]


class FeatureCheck(Protocol):
  """A check that live data needs a feature the kernels were compiled without.

  Bindings return them from `Binding.feature_checks()`; `enqueue` launches a kernel that sets the
  `Word.COMPILED_OUT` bit of `flags[0]` (`validator_kernels.raise_flag`) when the frame's data
  needs a compiled-out feature.
  """

  def enqueue(self, features: Features, frame: Frame, flags: wp.array[wp.int32, [int]]) -> None:
    """Enqueues the check of `frame` against `features` (device work only)."""
    ...

  def explain(self, features: Features) -> str:
    """What the data needs and how to compile it in (the error's text)."""
    ...


class Binding[S](abc.ABC):
  """Per-call resolution of a scene's slots from simulator state `S` (see the module docstring).

  Only `slots` is required; the defaults describe a binding that aliases its simulator's poses
  and moves every instance per world.

  Attributes:
    writes_poses: the poses live in Renderer-owned buffers the binding writes (`allocate`,
      `write`, `write_shared`); else the binding's slots alias them (every instance a column).
    color_rule: how the instance colors its slots serve combine with their material's base color
      (`GpuScene.color_rule`): TINT multiplies them (host scenes); a simulator whose color table
      has its own convention declares it (MuJoCo: MUJOCO).
    instance_segments: the `seg` output writes the instance index (`GpuScene.segment_id` None)
      instead of the host scene's segment column (MuJoCo: the geom id).
    scaled_poses: the binding serves a mesh instance's rotation with the uniform scale of its
      host transform, s R (`StaticPoses`: the scene's poses as they are): `upload` then keeps
      that scale out of the mesh copies it bakes (`pipeline/upload/meshes.bake_stretches`). A
      simulator writing rigid rotations (Newton) leaves it False: every stretch is baked into its
      mesh.
    hides_by_color: the alpha 0 of the "instance.color" its slots serve hides an instance in a
      world (Newton: the pooled objects a world does not show); a per-world "instance.color"
      table keeps that hiding (`PerWorldTables.write_colors`).
  """

  writes_poses: ClassVar[bool] = False
  scaled_poses: ClassVar[bool] = False
  color_rule: ClassVar[ColorRule] = ColorRule.TINT
  instance_segments: ClassVar[bool] = False
  hides_by_color: ClassVar[bool] = False

  @abc.abstractmethod
  def slots(self, sim: S | None) -> Slots:
    """Slot name -> array for this call (host only; no allocation, no device work).

    `sim` None: the arrays that do not come from a simulator state (model tables).
    """

  def deformables(self, device: wp.Device) -> Deformables:
    """The simulator's deformable geometry (`core/pipeline/gpu_scene/deformables.py`) on `device`,
    its vertices read from the "flex.vert" slot (default: none). `upload` stores it in the
    GpuScene."""
    return Deformables.empty(device)

  def sources(self, sim: S | None) -> dict[str, AnyDeviceArray]:  # noqa: ARG002  # default hook of the Binding contract
    """Other per-call arrays the binding reads (e.g. Newton's body_q), checked by captured
    graphs."""
    return {}

  def cameras(self, sim: S | None) -> dict[str, AnyDeviceArray]:  # noqa: ARG002  # default hook of the Binding contract
    """The simulator's live scene-camera tables by per-world field ("camera.fovy": (rows,
    nscenecam) float, "camera.intrinsics": (rows, nscenecam) vec4; rows 1 or nworld), read every
    frame for the Renderer's mounted cameras (MuJoCo: `m.cam_fovy` / `m.cam_intrinsic`); none by
    default: the CameraSpecs' values hold."""
    return {}

  def live_inputs(self) -> dict[str, AnyDeviceArray]:
    """The simulator's live material tables a derived slot reads every frame, by live input
    (`derived_fields.LIVE_INPUTS`: "material.shininess", (rows, nmat) float, rows 1, K or nworld
    read as world % rows; MuJoCo: `m.mat_shininess`), re-read every call; none by default."""
    return {}

  def nworld(self, sim: S | None) -> int | None:  # noqa: ARG002  # default hook of the Binding contract
    """Number of worlds of `sim` (None: any), checked against the Renderer's."""
    return None

  def dynamic(self, ninstance: int) -> Mask[int]:
    """(ninstance,) bool: the instances the simulator moves (or poses) per world."""
    return np.ones(ninstance, bool)

  def moving_lights(self) -> tuple[int, ...]:
    """The lights the simulator moves (a light on a moving body): the Renderer warns when one is
    not declared dynamic (`PerWorld.dynamic_lights`), which re-lights the GI on every move.
    Default: none known."""
    return ()

  def shared_motion(self, sim: S | None, ids: Ids[int]) -> RowsSource | TransformSource | None:  # noqa: ARG002  # default hook of the Binding contract
    """The source of the shared instances `ids`' poses checked every render (None: nothing moves
    them)."""
    return None

  def feature_checks(self) -> tuple[FeatureCheck, ...]:
    """Per-frame checks that the simulator's live data needs a feature the kernels were compiled
    without (`FeatureCheck`); none by default."""
    return ()

  def allocate[NWorld: IntVar, NColumn: IntVar](
    self,
    nworld: Int[NWorld],  # noqa: ARG002  # default hook of the Binding contract
    columns: Ids[NColumn],  # noqa: ARG002
    *,
    device: wp.Device,  # noqa: ARG002
  ) -> PoseBuffers[NWorld, NColumn] | None:
    """Pose buffers of instances `columns` with their initial poses, on `device` (the scene's).

    None for an aliasing binding. `write` then writes only the columns the binding drives.
    """
    return None

  def stage(self, sim: S | None) -> None:  # noqa: B027  # default hook: no host work
    """The host work of a frame before its device work, which a CUDA graph cannot replay: called
    by an eager render and before each graph replay, outside the graph (e.g. uploading host
    arrays the frame reads: the device copies a graph then reads)."""

  def write_tables(self, sim: S | None) -> None:  # noqa: B027  # default hook: nothing to write
    """Writes the tables this call's slots (`slots`, `cameras`) serve that the binding derives
    from `sim` (device work only, no host sync): called once per frame before any slot is read,
    so that `slots`, which a frame calls several times, stays free of device work."""

  def write[NWorld: IntVar, NColumn: IntVar](self, poses: PoseBuffers[NWorld, NColumn], sim: S | None) -> None:  # noqa: B027  # default hook: nothing to write
    """Writes this call's poses into the buffers of `allocate` (device work only, no host sync)."""

  def write_shared[NShared: IntVar](  # noqa: B027  # default hook: nothing to write
    self, ids: Ids[NShared], *, pos: wp.array[wp.vec3, [NShared]], rot: wp.array[wp.mat33, [NShared]], sim: S | None
  ) -> None:
    """Writes the current poses of the shared instances `ids` into `pos` / `rot` (in place)."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class InstancePoses[N: IntVar]:
  """The instances' poses a GpuScene starts from (host, float32): its one pose store.

  The world BVH topology is built from them (`trace.accel.bvh.SceneBvh`), and a static scene's
  binding (`StaticPoses`) serves them as they are.

  Attributes:
    pos: (n, 3) positions.
    rot: (n, 3, 3) rotations; a mesh instance's may carry a uniform scale, s R (s > 0).
  """

  pos: np.ndarray[[N, 3], np.dtype[np.float32]]
  rot: np.ndarray[[N, 3, 3], np.dtype[np.float32]]


@overload
def _tiled[NWorld: IntVar, NColumn: IntVar](
  values: np.ndarray[[NColumn, 3], np.dtype[np.float32]],
  *,
  nworld: Int[NWorld],
  dtype: type[wp.vec3],
  device: wp.Device,
) -> PerWorld[wp.vec3, NWorld, NColumn]: ...


@overload
def _tiled[NWorld: IntVar, NColumn: IntVar](
  values: np.ndarray[[NColumn, 3, 3], np.dtype[np.float32]],
  *,
  nworld: Int[NWorld],
  dtype: type[wp.mat33],
  device: wp.Device,
) -> PerWorld[wp.mat33, NWorld, NColumn]: ...


def _tiled[NWorld: IntVar, NColumn: IntVar](
  values: np.ndarray[[NColumn, 3], np.dtype[np.float32]] | np.ndarray[[NColumn, 3, 3], np.dtype[np.float32]],
  *,
  nworld: Int[NWorld],
  dtype: type[wp.vec3] | type[wp.mat33],
  device: wp.Device,
) -> PerWorld[wp.vec3, NWorld, NColumn] | PerWorld[wp.mat33, NWorld, NColumn]:
  """`values` (n, ...) tiled to (nworld, n) Warp `dtype` rows on `device`."""
  return wp.array(np.ascontiguousarray(np.broadcast_to(values, (nworld, *values.shape))), dtype=dtype, device=device)


@final
class StaticPoses(Binding[None]):
  """Instance poses of a static host scene (`rw.upload`): nothing moves them.

  The per-world columns are tiled to every world once and never written by the binding
  (`Renderer.per_world.columns("instance.pos", pattern)` poses them per world).

  Attributes:
    poses: the scene's initial poses (`GpuScene.init_pose`, the same arrays).
  """

  writes_poses = True
  scaled_poses = True

  def __init__(self, poses: InstancePoses) -> None:
    """The binding of a static scene's poses (`GpuScene.init_pose`)."""
    self.poses = poses

  @override
  def slots(self, sim: None) -> Slots:
    """No per-call slots."""
    return {}

  @override
  def dynamic(self, ninstance: int) -> Mask[int]:
    """Nothing is moved per world by a simulator."""
    return np.zeros(ninstance, bool)

  @override
  def allocate[NWorld: IntVar, NColumn: IntVar](
    self, nworld: Int[NWorld], columns: Ids[NColumn], *, device: wp.Device
  ) -> PoseBuffers[NWorld, NColumn]:
    """The initial poses of instances `columns` tiled to (nworld, len(columns)) buffers on
    `device`."""
    cols = np.asarray(columns, np.int64)
    pos, rot = self.poses.pos[cols], self.poses.rot[cols]
    return PoseBuffers(
      _tiled(pos, nworld=nworld, dtype=wp.vec3, device=device),
      _tiled(rot, nworld=nworld, dtype=wp.mat33, device=device),
    )

  @override
  def write_shared[NShared: IntVar](
    self, ids: Ids[NShared], *, pos: wp.array[wp.vec3, [NShared]], rot: wp.array[wp.mat33, [NShared]], sim: None
  ) -> None:
    """The shared instances' poses."""
    pos.assign(self.poses.pos[ids])  # fancy indexing: a contiguous copy
    rot.assign(self.poses.rot[ids])
