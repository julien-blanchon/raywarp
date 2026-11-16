"""OptiX acceleration structures of a Renderer: static GASes and two instance groups.

- One triangle GAS per mesh, shared by all worlds. It is built in the mesh frame, or in the
  authored frame `R` of `Meshes.frames` (vertices `R v`) where MuJoCo's inertia frame degrades the
  BVH; the instance transform then holds `rot @ R^T`.
- SDFs: instances of their mesh's GAS (traced as their mesh, as the software traversal does).
- Height fields: one refittable triangle GAS per height field, refitted after a terrain change
  (`dynamic.HeightFieldGases`).
- Deformable surfaces (dim 2 / 3): one triangle GAS per part over every world's triangles, each
  world's in its cell, refitted every frame (`dynamic.DeformableGases`); identity instances in
  the per-world group. Dim-1 deformables (segments) are not traced (`optix_backend._unsupported`).
- Boxes: instances of one unit-cube triangle GAS, scaled to the box.
- Spheres and capsules: a GAS of OptiX built-in linear curves (round linear, i.e. linear swept
  spheres; hardware-intersected on Blackwell RT cores), a capsule being one segment along its
  axis and a sphere a zero-length segment. World-space, refitted every frame.
- Ellipsoids and cylinders: a custom-primitive GAS (world-space boxes, refitted every frame) whose
  intersection programs run MuJoCo Warp's `ray_*` functions (`hits.make_intersection`).
- Planes are not instanced (intersected analytically, `trace.py`).

Two `InstanceGroup`s hold the instances, each an IAS over its mesh and box instances plus one
identity instance of its curve and custom-primitive GASes:
- `worlds`: per-world instances, each world's geometry uniformly normalized into its unit cell
  on a fixed grid in one binade per axis (`trace.to_cell`; the instances `trace.accel.reach` hides
  are left out); instance id = the instance (`instances.write_instances`).
- `shared`: the shared instances (`sharing.Layout.shared_ids`), stored once, untranslated, posed by
  the shared pose table with world 0's sizes, alpha and meshes, traced by every world after its
  own cell; instance id = instance.

Every frame, on the stream and without host synchronisation: Warp kernels compute each world's
bounds from the current geometry (its cell transform depends on this frame alone, not on earlier
frames) and write both groups' build inputs from the frame (`OptixScene.write`; so
a shared instance's alpha 0 hides it from the next frame on), then OptiX refits (or rebuilds) the
structures (`OptixScene.build`). The per-world group is rebuilt every `rebuild_every` frames (0:
refitted only). Inside a CUDA graph the writes are captured and `build` runs at every replay
(`graph.py`), so the schedule holds there too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import TWO_SIDED, ShapeType
from raywarp.core.kernels.abi.rays import FlexPart, world_entry
from raywarp.core.pipeline.platform.device import host_copy, stream_handle
from raywarp.core.trace import omm_sources
from raywarp.core.trace.optix import api, dynamic, gas
from raywarp.core.trace.optix import instances as ops
from raywarp.core.trace.optix.context import address, device_context
from raywarp.core.trace.optix.device import trace as rt_trace
from raywarp.core.trace.optix.group import HitGroups, InstanceGroup, InstanceTables, Members, custom_types, device_ids
from raywarp.core.trace.optix.omm import micromaps as omm
from raywarp.scene.model.types import InstanceId

if TYPE_CHECKING:
  from raywarp._base.dims import Ids, Mask
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.binding import Slots
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.sharing import Layout

# Instance types with a triangle GAS of their own: meshes, SDFs (traced as their mesh, as the
# software traversal does), height fields (`dynamic.HeightFieldGases`).
_TRIANGLE_GASES: Final = (int(ShapeType.MESH), int(ShapeType.SDF), int(ShapeType.HFIELD))


@final
class OptixScene:
  """The OptiX structures of a Renderer: static GASes, the per-world and shared groups (module doc).

  `write` launches the Warp kernels that write the cell and both groups' build inputs from a
  frame; `build` then refits (or rebuilds) the structures with OptiX calls on a given stream.
  Inside a CUDA graph capture `write` is captured and `build` runs at each replay (`graph.py`).

  Attributes:
    plane_ids: rendered planes (intersected analytically, not instanced).
    worlds, shared: the per-world instances (every world on its cell) and the shared scenery.
    has_mesh, has_box: any triangle instance with a face normal from the hit program (a mesh, SDF,
      height field or deformable) / box instance (either group).
    deformables: the deformables' GASes and world-space meshes (None: no surface part).
    flex_part: (max(ndeformable, 1),) each deformable part as its hits are shaded
      (`GeometryFields.flex_part`; its world-space mesh: `deformables`').
    hfields: the height fields' GASes.
    has_instances, has_shared: the per-world / shared group has members.
    has_world_curves, has_shared_curves: curve members (per world, shared): the pipeline's
      built-in curve intersector.
    primitive_types, shared_primitive_types: each group's custom-primitive types, in SBT order.
    hit_groups: the shader binding table's hit groups over both groups (`group.HitGroups`).
    translucent_curves: per world / shared, a sphere or capsule is a translucent shadow caster
      (with translucent shadows): its curve GAS enforces the any-hit program.
    cut_curves: per world / shared, a sphere or capsule is cut out (with `alpha_cutout`): its curve
      GAS enforces the any-hit program (the alpha test; the triangles' runs on the cut-out
      instances only, `tables.flags`; the custom primitives' and planes' is in their intersection
      programs and queries).
    cut_gases: the GASes with opacity micromaps of the static cut-out meshes
      (`omm/micromaps.py`; None: none), `has_omm` whether there are any (the pipeline allows
      micromaps).
    shared_visible: shared instances have a per-world visibility (`PerWorld.visible`): their
      triangle records enforce the any-hit program, the shared curves' too when one of them is
      declared (`visible_curves`), and the shared intersection programs test it.
    clear: per instance, a translucent shadow caster (`_translucent`; none without translucent
      shadows; with per-world optics any instance may be one).
    optics: the per-world optics' shadow transmittance and two-sided tables (world-major,
      `world_optics.WorldOptics`: Features.world_optics), None without: the instance flags are
      then per world, written each frame (`world_flags`) from `base_flags`.
    tables: what the instance writers read (meshes, flags, masks, the cell, `grid`).
    bounds: `instances.BOUNDS` values per world: lower xyz, upper xyz of its per-world members in
      reach in the current frame (its independent cell transform), its reference position;
      `bounds_members`: any per-world instance (else the boxes start empty, widened by the
      deformables only).
    rebuild_every: the per-world group is rebuilt every this many frames (0: refitted only;
      `performance.rebuild_every`, set every frame by the renderer).
    frames: frames built so far (the rebuild schedule).
    stale: the structures were built for a frame apart (`build_apart`: the light grid's, the GI
      bake's): the next `build` rebuilds them.
  """

  def __init__(
    self,
    scene: GpuScene,
    nworld: int,
    layout: Layout,
    *,
    device: wp.Device,
    mesh_used: np.ndarray[[int], np.dtype[np.bool_]] | None,
    mesh_frame: np.ndarray[[int, 3, 3], np.dtype[np.float32]] | None = None,
    alpha_cutout: bool = False,
    translucent_shadows: bool = False,
    cut_all: bool = False,
    opacity_micromaps: bool = False,
    optics: tuple[wp.array[wp.vec3, [int]], wp.array[int, [int]]] | None = None,
  ) -> None:
    """Builds the static GASes and allocates the groups on `device`'s OptiX context (`mesh_used`:
    the meshes with their own GAS, `RendererTables.mesh_used`; `cut_all`: see `cut_instances`;
    `opacity_micromaps`: the alpha test's tables are static, the cut-out meshes may get
    micromaps, `omm/micromaps.py`).

    Raises:
      ValueError: when the instance ids (the instances themselves) would reach
        `trace.FLEX_INSTANCE`.
    """
    self.ctx, self.device = device_context(device), device
    self.nworld, self.layout = nworld, layout
    self.rebuild_every, self.frames = 0, 0
    self.stale = False
    instance_type = np.asarray(host_copy(scene.instance_type), np.int32)
    per_world = np.asarray(layout.per_world, bool)
    if len(instance_type) > rt_trace.FLEX_INSTANCE:
      raise ValueError(f"{len(instance_type)} instances exceed OptiX's {rt_trace.FLEX_INSTANCE} instance ids")
    world = _members([InstanceId(int(g)) for g in scene.rendered if per_world[g]], instance_type)
    shared = _members([InstanceId(int(g)) for g in layout.shared_ids], instance_type)
    members = {"worlds": world, "shared": shared}
    # Planes (per-world or shared) are intersected analytically, not instanced.
    self.plane_ids = rendered_planes(scene.rendered, instance_type)
    # Their ids as the programs read them (run-time data: `GeometryFields.plane_ids`).
    self.plane_table = wp.array(np.asarray(self.plane_ids, np.int32), dtype=int, device=device)
    declared = np.asarray(layout.visible_slot, np.int32) >= 0
    self.shared_visible = any(declared[g] for kind in members["shared"] for g in kind)
    self.visible_curves = any(declared[g] for g in members["shared"][1])
    # Translucent shadow casters (`trace.make_shadow`): their any-hit programs (triangles, curves)
    # or intersection programs (custom primitives) multiply a shadow ray's transmittance in.
    self.optics = optics
    clear = _translucent(scene) if translucent_shadows else np.zeros(len(instance_type), bool)
    if optics is not None and translucent_shadows:
      clear = np.ones(len(instance_type), bool)
    self.clear = clear
    world_clear, shared_clear = (any(clear[g] for g in members[k][1]) for k in ("worlds", "shared"))
    self.translucent_curves: tuple[bool, bool] = (world_clear, shared_clear)
    self.mesh_used = mesh_used
    self.deformables = dynamic.DeformableGases(self.ctx, scene.deformables, nworld, device=device) if deforms(scene) else None
    self.flex_part = self.deformables.bvhs.parts if self.deformables is not None else wp.zeros(1, dtype=FlexPart, device=device)
    self.tables = self._tables(
      scene, instance_type, members, mesh_frame, alpha_cutout=alpha_cutout, cut_all=cut_all, micromaps=opacity_micromaps
    )
    cols = _columns(layout)
    self._groups(members, cols, _slots(layout), instance_type)
    self.bounds = wp.empty(ops.BOUNDS * nworld, dtype=float, device=device)
    per_world_ids = [g for kind in members["worlds"] for g in kind]
    self.bounds_members = bool(per_world_ids)
    self.bounds_ids = device_ids(per_world_ids, device)
    self.bounds_cols = device_ids([int(cols[g]) for g in per_world_ids], device)

  def _tables(
    self,
    scene: GpuScene,
    instance_type: Ids[int],
    members: dict[str, Members],
    mesh_frame: np.ndarray[[int, 3, 3], np.dtype[np.float32]] | None,
    *,
    alpha_cutout: bool,
    cut_all: bool,
    micromaps: bool,
  ) -> InstanceTables:
    """The static GASes and per-instance tables; sets the mask / any-hit / mesh / box flags
    (`micromaps`: see `__init__`)."""
    ninstance = len(instance_type)
    tri_ids = members["worlds"][0] + members["shared"][0]
    mask = np.where(np.asarray(self.layout.per_world, bool), rt_trace.WORLD_BITS, rt_trace.SHARED_BITS).astype(np.uint32)
    mask = mask if ninstance else np.full(1, rt_trace.WORLD_BITS, np.uint32)
    # Cut-out instances (with alpha_cutout): the triangle hit group's any-hit program runs on their
    # instances only (ENFORCE_ANYHIT); a group's curve GAS enforces its any-hit program when one of
    # its curves is cut out.
    cut = cut_instances(scene, ninstance, cut_all=cut_all) & alpha_cutout
    self.cut_curves: tuple[bool, bool] = (any(cut[g] for g in members["worlds"][1]), any(cut[g] for g in members["shared"][1]))
    flags = np.full(max(ninstance, 1), api.INSTANCE_FLAG_DISABLE_ANYHIT, np.uint32)
    flags[:ninstance][cut] = api.INSTANCE_FLAG_ENFORCE_ANYHIT
    # Shared triangle instances with a per-world visibility: the any-hit program tests it.
    shared_tri = np.asarray(members["shared"][0], np.int64)
    shared_seen = shared_tri[np.asarray(self.layout.visible_slot)[shared_tri] >= 0]
    flags[shared_seen] = api.INSTANCE_FLAG_ENFORCE_ANYHIT
    # Cut-out meshes with micromaps: OptiX resolves their micro-triangles, the any-hit program
    # runs on the unknown ones (geometry flag; ENFORCE_ANYHIT would call it on opaque ones too).
    handle = np.zeros(max(ninstance, 1), np.uint64)
    self.cut_gases: omm.CutGases | None = None
    if micromaps and any(cut[g] for g in tri_ids):
      seen = set(shared_seen.tolist())
      ids = [g for g in tri_ids if cut[g] and instance_type[g] == int(ShapeType.MESH) and g not in seen]
      self.cut_gases = self._micromaps(scene, ids, mesh_frame, handle)
      flags[handle != 0] = api.INSTANCE_FLAG_NONE
    self.has_omm = self.cut_gases is not None
    # Translucent shadow casters: the any-hit program multiplies a shadow ray's transmittance in
    # (camera rays disable it by ray flag); the opaque instances keep it disabled.
    self.translucent_triangles = bool(np.any(self.clear[tri_ids])) if tri_ids else False
    if self.optics is None:
      flags[:ninstance][self.clear] = api.INSTANCE_FLAG_ENFORCE_ANYHIT
      # Double-sided instances (USD doubleSided) are never back-face culled.
      culling = np.where((scene.instance_twosided & TWO_SIDED) != 0, api.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING, 0)
      flags[:ninstance] |= culling.astype(np.uint32)
    else:
      # Each world's: these flags with its translucent casters' and two-sided instances' (`write`).
      self.base_flags = wp.array(flags, dtype=wp.uint32, device=self.device)
      flags = np.tile(flags, self.nworld)
    # Mesh GASes: meshes and SDFs. Triangles with a face normal from the hit program: those, height
    # fields and deformables (`programs.py`, the `has_mesh` the programs read).
    mesh_gases = any(int(instance_type[g]) in (int(ShapeType.MESH), int(ShapeType.SDF)) for g in tri_ids)
    self.has_mesh = any(int(instance_type[g]) in _TRIANGLE_GASES for g in tri_ids) or self.deformables is not None
    self.has_box = any(instance_type[g] == int(ShapeType.BOX) for g in tri_ids)
    self.hfields = dynamic.HeightFieldGases(self.ctx, scene.meshes, device=self.device)
    hfield_half = scene.meshes.hfield_bounds_size
    self.grid = _grid(self.nworld)
    dev = self.device
    return InstanceTables(
      instance_type=scene.instance_type,
      color_rule=int(scene.color_rule),
      meshes=gas.mesh_gases(
        self.ctx, scene.meshes, mesh_frame, used=self.mesh_used, has_mesh=mesh_gases, has_box=self.has_box, device=dev
      ),
      hfield_handle=self.hfields.handle,
      hfield_half=hfield_half if hfield_half is not None and hfield_half.shape[0] else wp.zeros(1, dtype=wp.vec3, device=dev),
      flags=wp.array(flags, dtype=wp.uint32, device=dev),
      handle=wp.array(handle, dtype=wp.uint64, device=dev),
      mask=wp.array(mask, dtype=wp.uint32, device=dev),
      visible_slot=wp.array(self.layout.visible_slot if ninstance else np.full(1, -1, np.int32), dtype=int, device=dev),
      # The worlds' cells, then the shared IAS's box (`trace.shared_span`).
      cell=wp.zeros(self.nworld * rt_trace.CELL_SIZE + rt_trace.SHARED_BOX, dtype=wp.vec3, device=dev),
      grid=wp.vec3i(*self.grid),
      world_flags=self.optics is not None,
    )

  def _micromaps(
    self,
    scene: GpuScene,
    ids: list[InstanceId],
    mesh_frame: np.ndarray[[int, 3, 3], np.dtype[np.float32]] | None,
    handle: np.ndarray[[int], np.dtype[np.uint64]],
  ) -> omm.CutGases | None:
    """The GASes with micromaps of cut-out mesh instances `ids` whose alpha test is world-
    invariant (`omm_sources.source_of`); writes their handles into `handle` (None: no such
    instance)."""
    host = omm_sources.host_tables(scene)
    found = {g: omm_sources.source_of(host, int(g)) for g in ids}
    sources = {g: s for g, s in found.items() if s is not None}
    if not sources:
      return None
    unique = list(dict.fromkeys(sources.values()))
    frames = gas.mesh_frames(mesh_frame, len(scene.meshes.faces))
    built = omm.cut_gases(self.ctx, scene, unique, frames, device=self.device)
    for g, s in sources.items():
      handle[g] = built.handle[s]
    return built

  def _groups(self, members: dict[str, Members], cols: Ids[int], slots: Ids[int], instance_type: Ids[int]) -> None:
    """The per-world and shared groups (`cols`: per-world pose columns, `slots`: shared pose slots),
    their SBT records and the flags the programs read."""
    off = api.INSTANCE_FLAG_DISABLE_ANYHIT
    world_curves = api.INSTANCE_FLAG_ENFORCE_ANYHIT if self.translucent_curves[0] or self.cut_curves[0] else off
    shared_any = self.visible_curves or self.translucent_curves[1] or self.cut_curves[1]
    curves = api.INSTANCE_FLAG_ENFORCE_ANYHIT if shared_any else off
    w, s = members["worlds"], members["shared"]
    self.has_world_curves, self.has_shared_curves = bool(w[1]), bool(s[1])
    world_types, shared_types = custom_types(w[2], instance_type), custom_types(s[2], instance_type)
    self.primitive_types, self.shared_primitive_types = world_types, shared_types
    self.hit_groups = HitGroups(self.has_world_curves, world_types, self.has_shared_curves, shared_types)
    self.worlds = InstanceGroup(
      ctx=self.ctx,
      device=self.device,
      shared=False,
      rows=self.nworld,
      members=w,
      cols=cols,
      instance_type=instance_type,
      hit_groups=self.hit_groups,
      gas_flags=(world_curves, off),
      triangle_gases=self.deformables.gases if self.deformables is not None else (),
    )
    self.shared = InstanceGroup(
      ctx=self.ctx,
      device=self.device,
      shared=True,
      rows=1,
      members=s,
      cols=slots,
      instance_type=instance_type,
      hit_groups=self.hit_groups,
      gas_flags=(curves, off),
      # Its box after the worlds' cells (`trace.shared_span`): 8-byte aligned (12-byte entries,
      # an even count before it).
      bounds=address(self.tables.cell) + 12 * self.nworld * rt_trace.CELL_SIZE,
    )
    self.has_instances, self.has_shared = not self.worlds.empty, not self.shared.empty

  def rebuild_due(self) -> bool:
    """Whether the next `build` rebuilds the per-world structures instead of refitting them."""
    return not self.worlds.built or (self.rebuild_every > 0 and self.frames % self.rebuild_every == 0)

  def _write_shared(self, f: Frame) -> None:
    """Writes the shared group's build inputs: the shared poses (`Frame.shared_poses`), world 0's
    sizes and alpha."""
    self.shared.write(f, (f.instance_pos, f.instance_rot, *f.shared_poses(self.layout)), self.tables, visibility=False)

  def write(self, f: Frame, *, visibility: bool) -> None:
    """Writes the cell and both groups' build inputs from frame `f` (Warp kernels).

    Each world's cell bounds are its per-world geometry's in this frame. `visibility`: the frame's
    `instance_visible` hides instances (`Features.visibility`).
    """
    if self.optics is not None:
      wp.launch(
        write_world_flags,
        dim=(self.nworld, self.base_flags.shape[0]),
        inputs=[self.base_flags, *self.optics],
        outputs=[self.tables.flags],
        device=self.device,
      )
    self._write_shared(f)
    if self.worlds.empty:
      return
    m, flex = self.tables.meshes, self.deformables
    if not self.bounds_members:  # deformables only
      wp.launch(ops.no_bounds, dim=self.nworld, outputs=[self.bounds], device=self.device)
    else:
      wp.launch(
        ops.world_bounds,
        dim=self.nworld,
        inputs=[
          self.tables.instance_type,
          f.instance_mesh,
          f.instance_size,
          m.lower,
          m.upper,
          m.frame_t,
          self.tables.hfield_half,
          f.instance_pos,
          f.instance_rot,
          self.bounds_ids,
          self.bounds_cols,
        ],
        outputs=[self.bounds],
        device=self.device,
      )
    if flex is not None:
      flex.write_world(f, self.bounds)
    wp.launch(
      ops.update_cell, dim=self.nworld, inputs=[self.bounds, self.tables.grid], outputs=[self.tables.cell], device=self.device
    )
    if flex is not None:
      flex.write_cells(f, self.tables.color_rule, self.tables.cell)
    self.worlds.write(f, (f.instance_pos, f.instance_rot, *f.shared_poses(self.layout)), self.tables, visibility=visibility)

  def build(self, stream: wp.Stream) -> None:
    """Refits (or rebuilds) both groups on `stream` (OptiX calls): one frame of the rebuild
    schedule.

    The shared group is built on the first frame and refitted after (its writes follow the frame's
    alpha). After `build_apart` both groups are rebuilt.
    """
    stale, self.stale = self.stale, False
    self.hfields.refit(stream_handle(stream))
    self.shared.build(stream, rebuild=stale)
    if self.worlds.empty:
      return
    self.worlds.build(stream, rebuild=stale or self.rebuild_due())
    self.frames += 1

  def build_apart(self, f: Frame, stream: wp.Stream, *, visibility: bool) -> None:
    """Writes and rebuilds both groups for frame `f` alone, outside the frames' schedule: a frame
    whose see-through instances are hidden (the light grid's, `baked._see_through_dynamic`; the GI
    bake's, `baked._probe_geometry`'s colors).
    """
    self.write(f, visibility=visibility)
    self.rebuild_apart(stream)

  def rebuild_apart(self, stream: wp.Stream) -> None:
    """Rebuilds both groups from the build inputs `write` wrote for a frame apart (`build_apart`;
    inside a capture its launch's cut, `launch.OptixRenderer.graph_work`). The next `build`
    rebuilds them again (`stale`): trees built around hidden instances (curves of radius 0, point
    boxes) would bound them poorly once refitted to the frame's."""
    self.hfields.refit(stream_handle(stream))
    self.shared.build(stream, rebuild=True)
    self.worlds.build(stream, rebuild=True)
    self.stale = True


# The OptiX instance flag bits a translucent caster keeps (any-hit disabled cleared).
_KEEP_ANYHIT: Final = ~api.INSTANCE_FLAG_DISABLE_ANYHIT


@wp.kernel(enable_backward=False, module="unique")
def write_world_flags(
  # Model:
  base: wp.array1d[wp.uint32],
  # Data in:
  shadow_transmit: wp.array1d[wp.vec3],
  twosided: wp.array1d[int],
  # Out:
  flags_out: wp.array1d[wp.uint32],
):
  """The OptiX instance flags of every (world, instance) (`trace.optix.group.InstanceTables.flags`,
  world-major): `base` (no face-culling bit; any-hit as cut-out and shared visibility ask), plus
  face culling disabled where world w's instance is two-sided and the any-hit program enforced
  where it is a translucent shadow caster."""
  w, i = wp.tid()
  # The flags are small bit sets: int arithmetic.
  f = int(base[i])
  e = world_entry(base.shape[0], w, i)
  t = shadow_transmit[e]
  if wp.max(t[0], wp.max(t[1], t[2])) > 0.0:
    f = (f & wp.static(_KEEP_ANYHIT)) | wp.static(api.INSTANCE_FLAG_ENFORCE_ANYHIT)
  if (twosided[e] & wp.static(TWO_SIDED)) != 0:
    f = f | wp.static(api.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING)
  flags_out[e] = wp.uint32(f)


def rendered_planes(rendered: Ids[int], instance_type: Ids[int]) -> tuple[InstanceId, ...]:
  """The rendered planes (`rendered`: instance ids; `instance_type`: (ninstance,) ShapeType), the
  instances the OptiX queries intersect analytically."""
  return tuple(InstanceId(int(g)) for g in rendered if instance_type[g] == int(ShapeType.PLANE))


def deforms(scene: GpuScene) -> bool:
  """Whether `scene` has deformables with surfaces (`Deformables.dim` 2 or 3): their world-space
  triangle GASes are members of the per-world group (`has_instances`) and its mesh hits
  (`has_mesh`)."""
  return any(int(d) != 1 for d in scene.deformables.dim)


def _members(ids: list[InstanceId], instance_type: Ids[int]) -> Members:
  """`ids` split into triangle (mesh, SDF, height field, then box), sphere / capsule and ellipsoid /
  cylinder instances."""

  def of(*types: int) -> list[InstanceId]:
    """The members of `ids` of these types."""
    return [g for g in ids if int(instance_type[g]) in types]

  return of(*_TRIANGLE_GASES) + of(int(ShapeType.BOX)), of(*rt_trace.CURVE_TYPES), of(*rt_trace.CUSTOM_TYPES)


def _columns(layout: Layout) -> Ids[int]:
  """Per instance: its per-world pose column (`Layout.pose_slot`; the instance itself when not
  compact)."""
  slot = np.asarray(layout.pose_slot, np.int32)
  return slot if layout.compact else np.arange(len(slot)).astype(np.int32)


def _slots(layout: Layout) -> Ids[int]:
  """Per instance: its shared pose slot (`-1 - pose_slot`; the instance itself when not compact)."""
  slot = np.asarray(layout.pose_slot, np.int32)
  return (-1 - slot).astype(np.int32) if layout.compact else np.arange(len(slot)).astype(np.int32)


def _translucent(scene: GpuScene) -> Mask[int]:
  """Per instance: a translucent shadow caster (an RGB shadow transmittance > 0)."""
  t = scene.instance_shadow_transmit
  return (t.reshape(len(t), 3) > 0.0).any(axis=1)


def cut_instances(scene: GpuScene, ninstance: int, *, cut_all: bool) -> Mask[int]:
  """Per instance: whether its material has an alpha cutoff > 0 in some world.

  `cut_all`: every instance may be cut out (the cutoffs or materials are per-world declarations,
  or live binding slots: `launch._cut_all`). Otherwise the static tables, the scene's or its
  binding's. Material ids outside the cutoff table cut nothing here (the frame's validation
  reports them).
  """
  if cut_all:
    return np.ones(ninstance, bool)
  model: Slots = {**scene.tables, **scene.binding.slots(None)}
  cutoff, matid = model.get("material.alpha_cutoff"), model.get("instance.material")
  # No material: nothing is cut out (an empty table cannot be reshaped by rows).
  if cutoff is None or matid is None or not ninstance or cutoff.shape[-1] == 0:
    return np.zeros(ninstance, bool)
  # Tables are (1 or nworld, n): broadcast the single-row one to the other's rows.
  m = np.asarray(host_copy(matid)).reshape(-1, matid.shape[-1])[:, :ninstance]
  c = np.asarray(host_copy(cutoff)).reshape(-1, cutoff.shape[-1])
  rows = max(len(m), len(c))
  m = np.broadcast_to(m, (rows, m.shape[1]))
  c = np.broadcast_to(c, (rows, c.shape[1]))
  valid = (m >= 0) & (m < c.shape[1])
  cutoff_of = np.take_along_axis(c, np.where(valid, m, 0), axis=1)
  return np.any(valid & (cutoff_of > 0.0), axis=0)


def _grid(nworld: int) -> tuple[int, int, int]:
  """Cells per axis of a near-cubic grid holding nworld cells."""
  gx = max(1, int(np.ceil(nworld ** (1.0 / 3.0) - 1.0e-9)))
  gy = max(1, int(np.ceil(np.sqrt(nworld / gx) - 1.0e-9)))
  gz = (nworld + gx * gy - 1) // (gx * gy)
  return (gx, gy, gz)
