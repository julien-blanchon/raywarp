"""Usd.Stage -> SceneBuilder: traversal, instancing, transforms, units.

The stage is walked once without instance proxies. Each native prototype (and each
PointInstancer prototype) is converted once into a list of items (`items.Item`: shapes with their
transform relative to the prototype root, nested instances, lights, cameras); instances then emit
those items with one batched matrix product per (prototype, item), so the per-instance Python
work is a few array operations. Lights and cameras are read once per prim and placed K times.
Purpose (guide / proxy by default), invisible and render-invisible subtrees are pruned. The
stage root transform converts to metres and Z-up (metersPerUnit, upAxis Y -> +90 deg about X).
"""

from __future__ import annotations

import dataclasses
import itertools
import os
from typing import Final, final

import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from raywarp.formats.usd import render_settings, util
from raywarp.formats.usd.materials import context
from raywarp.formats.usd.options import Options
from raywarp.formats.usd.stage import camera as usd_camera
from raywarp.formats.usd.stage import lux, semantics
from raywarp.formats.usd.stage.gprims import PRIMITIVES, Gprims
from raywarp.formats.usd.stage.items import INHERIT, CameraItem, Instance, Item, PointInstancer, Shape, StageLight
from raywarp.formats.usd.util import Matrices4, Matrix4
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.build.displacement import DisplacementOptions
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import MaterialId

_CONTAINERS: Final = {"", "Xform", "Scope", "SkelRoot"}
# Subtrees never walked (shading networks, subsets, skeleton data).
_SKIP_SUBTREE: Final = {"Material", "Shader", "NodeGraph", "GeomSubset", "Skeleton", "SkelAnimation", "BlendShape"}
# Prim types skipped without a report entry: physics scene data, and the UsdRender prims
# (`rw.formats.usd.settings` reads them on request; a load never applies render settings). Joints
# (every UsdPhysics.Joint type) are skipped likewise (`_item`).
_SILENT: Final = {
  "PhysicsScene",
  "PhysicsMaterial",
  "PhysicsCollisionGroup",
  "RenderSettings",
  "RenderProduct",
  "RenderVar",
  "RenderPass",
}

# A binding passed down to prototype items: (material, the mesh attributes it reads).
type Binding = tuple[MaterialId, context.MeshNeeds]


def stage_root(stage: Usd.Stage) -> Matrix4:
  """(4, 4) map from stage coordinates to the scene's (metres, Z-up)."""
  root = np.eye(4)
  root[:3, :3] *= float(UsdGeom.GetStageMetersPerUnit(stage))
  if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
    root = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]]) @ root
  return root


def prim_transforms(stage: str | os.PathLike[str] | Usd.Stage, paths: list[str], time: float | None = None) -> Matrices4:
  """(n, 4, 4) world transforms of prims in the scene's frame (identity for a missing prim)."""
  if not isinstance(stage, Usd.Stage):
    stage = util.open_stage(stage)
  t = Usd.TimeCode.EarliestTime() if time is None else Usd.TimeCode(time)
  cache = UsdGeom.XformCache(t)
  root = stage_root(stage)
  out = np.tile(np.eye(4), (len(paths), 1, 1))
  for i, path in enumerate(paths):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
      out[i] = root @ util.matrix(cache.GetLocalToWorldTransform(prim))
  return out


@final
@dataclasses.dataclass
class _Walk:
  """State of one subtree walk (`StageReader.collect`)."""

  items: list[Item]
  base_inv: Matrix4 | None  # inverse of the subtree root's world transform (None: identity)
  prefix_len: int  # prim-path prefix removed from the item suffixes
  # Walking a prototype: what the binding its unbound gprims inherit from the instance reads
  # (None: not a prototype).
  inherited: context.MeshNeeds | None


def displacement_options(options: Options) -> DisplacementOptions:
  """The scene builder's displacement preparation policy of USD `options`."""
  return DisplacementOptions(
    mode="true" if options.displacement else "off",
    max_level=options.displacement_max_level,
    max_triangles=options.displacement_max_triangles,
  )


def builder_options(builder: SceneBuilder) -> Options:
  """The default USD options with `builder`'s displacement preparation policy."""
  d = builder.displacement_options
  return dataclasses.replace(
    Options(), displacement=d.mode == "true", displacement_max_level=d.max_level, displacement_max_triangles=d.max_triangles
  )


def report_mtlx_layers(stage: Usd.Stage, report: LoadReport) -> None:
  """Reports every `.mtlx` document a layer of `stage` sublayers, references or loads as a payload
  (`usd.mtlx_layer`, dropped: usd-core cannot open them, their materials are not read; MaterialX
  networks authored in the stage are)."""
  for layer in stage.GetUsedLayers():
    for asset in layer.GetCompositionAssetDependencies():
      if asset.lower().endswith(".mtlx"):
        report.drop("usd.mtlx_layer", f"{layer.identifier}: {asset}")


def _double_sided(stage: Usd.Stage, options: Options) -> bool:
  """Whether every gprim is double-sided: `options.double_sided`, else (None) unless the stage's
  RTX settings enable face culling (Omniverse RTX culls by `doubleSided` only then)."""
  if options.double_sided is not None:
    return options.double_sided
  return not bool(render_settings.authored(stage).get("hydra:faceCulling:enabled", False))


@final
class StageReader:
  """Reads one stage into a SceneBuilder (`read`)."""

  def __init__(
    self,
    stage: Usd.Stage,
    options: Options,
    baker: context.MaterialBaker,
    builder: SceneBuilder | None = None,
    *,
    max_texture_size: int | None,
  ) -> None:
    """A reader of `stage` with `options` into `builder` (a new one by default); MDL by `baker`;
    images decoded at most `max_texture_size` (None: no cap)."""
    self.stage = stage
    self.options = options
    preparation = displacement_options(options)
    if builder is not None and builder.displacement_options != preparation:
      raise ValueError("USD displacement options conflict with the supplied SceneBuilder preparation policy")
    self.builder = SceneBuilder(displacement_options=preparation) if builder is None else builder
    self.report: LoadReport = self.builder.report
    self.time = Usd.TimeCode.EarliestTime() if options.time is None else Usd.TimeCode(options.time)
    self.xcache = UsdGeom.XformCache(self.time)
    self.mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    self.up_z = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    self.root = stage_root(stage)
    frame = context.StageFrame(self.mpu, np.linalg.inv(self.root[:3, :3]))
    report_mtlx_layers(stage, self.report)
    self.shade = context.ShadeContext(
      builder=self.builder,
      report=self.report,
      max_texture_size=max_texture_size,
      time=self.time,
      baker=baker,
      frame=frame,
      contexts=options.material_contexts,
    )
    self.gprims = Gprims(
      builder=self.builder,
      report=self.report,
      shade=self.shade,
      time=self.time,
      xcache=self.xcache,
      double_sided=_double_sided(stage, options),
    )
    # Prototype items by (prototype path, the attributes its inherited binding reads): instances
    # binding materials that read other mesh attributes get their own read of the meshes.
    self.prototypes: dict[tuple[str, context.MeshNeeds], list[Item]] = {}
    self.pi_prototypes: set[str] = set()
    # Semantic labels (UsdSemantics / Isaac SemanticsAPI), looked up only on stages that apply one.
    self.labels: semantics.Labels | None = None

  def read(self) -> SceneBuilder:
    """Reads the stage into the builder."""
    report = self.report
    with report.timer("usd.traverse"):
      self._scan()
      items = self.collect(self.stage.GetPseudoRoot(), None, inherited=None)
    with report.timer("usd.emit"):
      self.emit(items, self.root[None], [""], (INHERIT, context.MeshNeeds()), None)
    report.count("usd.prototypes", len(self.prototypes))
    report.count("instances", self.builder.ninstance)
    report.count("meshes", self.builder.nmesh)
    return self.builder

  def _scan(self) -> None:
    """Finds the PointInstancer prototypes (walked only through their instancer), whether any
    prim is labelled and the materials prims bind, on the stage and inside its native prototypes
    (which `Traverse` skips)."""
    ranges = [self.stage.Traverse(), *(Usd.PrimRange(p) for p in self.stage.GetPrototypes())]
    for prim in itertools.chain.from_iterable(ranges):
      for name in context.BINDINGS:  # the materials the read binds (the MDL baker prefetches them)
        rel = prim.GetRelationship(name)
        if rel:
          self.shade.bound.update(dict.fromkeys(rel.GetTargets()[:1]))
      if prim.GetTypeName() == "PointInstancer":
        for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
          self.pi_prototypes.add(str(target))
      if self.labels is None and semantics.applies(prim):
        self.labels = semantics.Labels(self.stage)

  def _world(self, prim: Usd.Prim) -> Matrix4:
    """World transform of `prim` (stage units)."""
    return util.matrix(self.xcache.GetLocalToWorldTransform(prim))

  # Traversal ------------------------------------------------------------------------------------

  def collect(self, root: Usd.Prim, base: Matrix4 | None, *, inherited: context.MeshNeeds | None) -> list[Item]:
    """Items of the subtree under `root`, with transforms relative to `base` (None = identity);
    `inherited` as `_Walk.inherited`."""
    walk = _Walk(
      [], None if base is None else np.linalg.inv(base), 0 if root.IsPseudoRoot() else len(str(root.GetPath())), inherited
    )
    if root.IsPseudoRoot():
      for child in root.GetChildren():
        self._visit(walk, child, "default", None, is_root=False)
    else:
      self._visit(walk, root, "default", None, is_root=True)
    return walk.items

  def _rel(self, walk: _Walk, prim: Usd.Prim) -> Matrix4:
    """Transform of `prim` relative to the walk's root."""
    w = self._world(prim)
    return w if walk.base_inv is None else walk.base_inv @ w

  def _invisible(self, walk: _Walk, prim: Usd.Prim, body: str | None) -> bool:
    """Whether `prim` is invisible (its subtree pruned; its lights kept on request)."""
    attr = prim.GetAttribute("visibility")
    if not (attr and attr.Get(self.time) == UsdGeom.Tokens.invisible):
      return False
    self.report.count("pruned.invisible")
    for light in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):  # lights inside instances too
      if light.HasAPI(UsdLux.LightAPI) and not light.HasAPI(UsdLux.MeshLightAPI):
        if self.options.invisible_lights:
          self._light(walk, light, body)
        else:
          self.report.drop("UsdLux.invisible_light(off)", light)
    return True

  def _visit(self, walk: _Walk, prim: Usd.Prim, purpose: str, body: str | None, *, is_root: bool) -> None:
    """Adds the items of `prim`'s subtree to the walk."""
    path = str(prim.GetPath())
    type_name = prim.GetTypeName()
    if (not is_root and path in self.pi_prototypes) or type_name in _SKIP_SUBTREE:
      return
    if self._invisible(walk, prim, body):
      return
    attr = prim.GetAttribute("purpose")
    if attr and attr.HasAuthoredValue():
      purpose = str(attr.Get())
    if purpose not in self.options.purposes:
      self.report.count(f"pruned.purpose.{purpose}")
      return
    # UsdGeomVisibilityAPI: render-purpose geometry hidden for final renders.
    attr = prim.GetAttribute("renderVisibility")
    if purpose == "render" and attr and attr.Get(self.time) == UsdGeom.Tokens.invisible:
      self.report.count("pruned.render_visibility")
      return
    self._report_animation(prim)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
      body = path
    if prim.IsInstance() and not is_root:
      material, needs = self.gprims.material(prim, inside=True)
      local = self._rel(walk, prim)
      suffix = path[walk.prefix_len :]
      item = Instance(prototype=prim.GetPrototype(), local=local, suffix=suffix, material=material, needs=needs, body=body)
      walk.items.append(item)
      self.report.count("usd.instances")
      return
    if not self._item(walk, prim, type_name, body):
      return
    for child in prim.GetChildren():
      self._visit(walk, child, purpose, body, is_root=False)

  def _report_animation(self, prim: Usd.Prim) -> None:
    """Reports a prim whose transform or visibility is time-sampled: the read is one time's
    (`Options.time`), animation is not played back."""
    xformable = UsdGeom.Xformable(prim)
    if xformable and xformable.TransformMightBeTimeVarying():
      self.report.approx("usd.time_samples.xform(one time read)", prim)
    attr = prim.GetAttribute("visibility")
    if attr and attr.ValueMightBeTimeVarying():
      self.report.approx("usd.time_samples.visibility(one time read)", prim)

  def _item(self, walk: _Walk, prim: Usd.Prim, type_name: str, body: str | None) -> bool:
    """Adds the item of `prim` itself; False when its children are not walked."""
    suffix = str(prim.GetPath())[walk.prefix_len :]
    if type_name == "Mesh":
      walk.items.extend(self.gprims.mesh(prim, self._rel(walk, prim), suffix, body, inherited=walk.inherited))
    elif type_name in PRIMITIVES:
      inside = walk.inherited is not None
      shape = self.gprims.primitive(prim, self._rel(walk, prim), suffix, body, inside=inside)
      if shape is not None:
        walk.items.append(shape)
    elif type_name == "PointInstancer":
      pi = self.gprims.point_instancer(self.stage, prim, self._rel(walk, prim), suffix, body)
      if pi is not None:
        walk.items.append(pi)
      return False
    elif type_name in lux.LIGHT_TYPES or type_name in lux.DOME_TYPES or prim.HasAPI(UsdLux.LightAPI):
      self._light(walk, prim, body)
    elif type_name == "Camera":
      spec = usd_camera.read_camera(self.report, prim, self.time)
      walk.items.append(CameraItem(spec=spec, local=self._rel(walk, prim), suffix=suffix))
    elif type_name in _SILENT or prim.IsA(UsdPhysics.Joint):
      return False
    elif type_name not in _CONTAINERS:
      self.report.drop(f"prim.{type_name}", prim)  # an unknown type: its children are still read
    return True

  def _light(self, walk: _Walk, prim: Usd.Prim, body: str | None) -> None:
    """Adds the light of a UsdLux prim (read once, here)."""
    self.report.count(f"lights.{prim.GetTypeName()}")
    spec = lux.read_light(
      self.builder,
      self.report,
      prim,
      time=self.time,
      up_z=self.up_z,
      mpu=self.mpu,
      lux_dialect=self.options.lux_dialect,
    )
    if spec is not None:
      suffix = str(prim.GetPath())[walk.prefix_len :]
      walk.items.append(StageLight(spec=spec, local=self._rel(walk, prim), suffix=suffix, body=body))

  def _prototype(self, prim: Usd.Prim, needs: context.MeshNeeds, *, pi_root: bool) -> list[Item]:
    """Items of a native prototype (relative to its root) or PointInstancer prototype (cached),
    its meshes read for an inherited binding that reads `needs`."""
    key = (str(prim.GetPath()), needs)
    if key not in self.prototypes:
      if prim.IsInstance():
        # An instanceable PointInstancer prototype: its own transform is in the instance xforms.
        material, own = self.gprims.material(prim, inside=True)
        instance = Instance(prototype=prim.GetPrototype(), local=np.eye(4), suffix="", material=material, needs=own, body=None)
        self.prototypes[key] = [instance]
      else:
        self.prototypes[key] = self.collect(prim, self._world(prim) if pi_root else None, inherited=needs)
    return self.prototypes[key]

  # Emission -------------------------------------------------------------------------------------

  def emit(self, items: list[Item], parents: Matrices4, paths: list[str], inherited: Binding, body: str | None) -> None:
    """Emits `items` under each of the (K, 4, 4) `parents` (whose prim paths are `paths`);
    `inherited`: the binding unbound items take."""
    for item in items:
      match item:
        case Shape():
          self._emit_shape(item, parents, paths, inherited[0], body)
        case Instance():
          binding = inherited if item.material == INHERIT else (item.material, item.needs)
          sub_paths = [p + item.suffix for p in paths]
          prototype = self._prototype(item.prototype, binding[1], pi_root=False)
          self.emit(prototype, parents @ item.local, sub_paths, binding, item.body or body)
        case PointInstancer():
          self._emit_point_instancer(item, parents, paths, inherited, body)
        case StageLight():
          for parent, path in zip(parents, paths, strict=True):
            lux.emit_light(
              self.builder, self.report, item.spec, world=parent @ item.local, path=path + item.suffix, body=item.body or body
            )
        case CameraItem():
          for parent, path in zip(parents, paths, strict=True):
            usd_camera.emit_camera(self.builder, self.report, item.spec, parent @ item.local, path + item.suffix)

  def _emit_shape(self, item: Shape, parents: Matrices4, paths: list[str], inherited: MaterialId, body: str | None) -> None:
    """Adds the K instances of a shape."""
    material = inherited if item.material == INHERIT else item.material
    if material < 0:
      material = self.shade.fallback(item.display)
    self.builder.add_instances(
      item.kind,
      xforms=parents @ item.local,
      mesh=item.mesh,
      material=material,
      size=item.size,
      flags=item.flags,
      paths=[p + item.suffix for p in paths],
      body=item.body or body,
      labels=None if self.labels is None else [self.labels(p + item.suffix) for p in paths],
    )

  def _emit_point_instancer(
    self, item: PointInstancer, parents: Matrices4, paths: list[str], inherited: Binding, body: str | None
  ) -> None:
    """Emits each prototype of a PointInstancer under the K x n instance transforms."""
    binding = inherited if item.material == INHERIT else (item.material, item.needs)
    base = parents @ item.local  # (K, 4, 4)
    for j, proto in enumerate(item.prototypes):
      sel = np.nonzero(item.indices == j)[0]
      if not len(sel):
        continue
      worlds = (base[:, None] @ item.xforms[sel][None]).reshape(-1, 4, 4)
      sub_paths = [f"{p}{item.suffix}[{int(item.ids[k])}]" for p in paths for k in sel]
      self.emit(self._prototype(proto, binding[1], pi_root=True), worlds, sub_paths, binding, item.body or body)
