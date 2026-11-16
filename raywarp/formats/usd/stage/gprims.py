"""Gprims and PointInstancers -> stage items (`items.Shape` / `items.PointInstancer`).

Meshes become one shape per materialBind part (`mesh.read_mesh`), UsdGeom primitives the core's
analytic shapes (a Cone is tessellated), with their bound material (`INHERIT` inside a
prototype without a binding of its own; -1 unbound: the displayColor fallback).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, final

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

from raywarp.formats.usd import util
from raywarp.formats.usd.materials import context as shade
from raywarp.formats.usd.materials import select
from raywarp.formats.usd.stage import lux
from raywarp.formats.usd.stage import mesh as usd_mesh
from raywarp.formats.usd.stage.items import INHERIT, Display, PointInstancer, Shape
from raywarp.formats.usd.util import Matrices3, Matrices4, Matrix4, Time
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.build.mesh_pool import MeshAttributes, finite_attributes
from raywarp.scene.lighting import emitter_shapes
from raywarp.scene.model import geometry
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_MATERIAL, NO_MESH, InstanceFlags, InstanceKind, MaterialId, MeshId

PRIMITIVES: Final = {"Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane", "Cylinder_1", "Capsule_1"}

# Segments of a tessellated UsdGeom.Cone (its round section).
CONE_SEGMENTS: Final = 32

# Authored primitive attributes read here (the rest is reported as dropped).
_PRIMITIVE_ATTRS: Final = {
  "size",
  "radius",
  "height",
  "axis",
  "width",
  "length",
  "radiusTop",
  "radiusBottom",
  "doubleSided",
  "extent",
  "purpose",
  "visibility",
  "xformOpOrder",
  "orientation",
  "kind",
}


def _mesh_needs(material: MaterialId, needs: list[shade.MeshNeeds]) -> usd_mesh.MeshNeeds:
  """The attributes a mesh reads for its parts' materials (`needs`; `material`: the mesh's own):
  UV set 0 / 1 names, the vertex-color primvar (an unbound mesh shows its displayColor:
  per-vertex values become vertex colors)."""
  uv0 = next((n.uv[0] for n in needs if n.uv), None)
  uv1 = next((n.uv[1] for n in needs if len(n.uv) > 1), None)
  color = next((n.color for n in needs if n.color), None)
  unbound = material == NO_MATERIAL
  if unbound and color is None:
    color = usd_mesh.DISPLAY_COLOR
  return usd_mesh.MeshNeeds(uv0=uv0, uv1=uv1, color=color, unbound=unbound, inherit=material == INHERIT)


def _binding_target(rel: Usd.Relationship) -> Sdf.Path:
  """The material path of the binding relationship `rel` (direct or collection-based)."""
  binding = UsdShade.MaterialBindingAPI
  if binding.CollectionBinding.IsCollectionBindingRel(rel):
    return binding.CollectionBinding(rel).GetMaterialPath()
  return binding.DirectBinding(rel).GetMaterialPath()


# The size attributes of each primitive type: (name, fallback, zero allowed). A Plane of zero width
# or length would be an infinite plane to the core (a half size 0); a Cone of zero size, a line.
_DIMENSIONS: Final = {
  "Cube": (("size", 2.0, True),),
  "Sphere": (("radius", 1.0, True),),
  "Cylinder": (("height", 2.0, True), ("radius", 1.0, True)),
  "Capsule": (("height", 1.0, True), ("radius", 0.5, True)),
  "Cylinder_1": (("height", 2.0, True), ("radiusTop", 1.0, True), ("radiusBottom", 1.0, True)),
  "Capsule_1": (("height", 1.0, True), ("radiusTop", 0.5, True), ("radiusBottom", 0.5, True)),
  "Cone": (("height", 2.0, False), ("radius", 1.0, False)),
  "Plane": (("width", 2.0, False), ("length", 2.0, False)),
}


def _valid_dimensions(prim: Usd.Prim, t: str, time: Time) -> bool:
  """Whether primitive `prim` of type `t` has finite dimensions in UsdGeom's range (>= 0; > 0 for
  a Plane's and a Cone's)."""
  for name, fallback, zero in _DIMENSIONS.get(t, ()):
    value = util.float_value(prim, name, fallback, time)
    if not (math.isfinite(value) and (value >= 0.0 if zero else value > 0.0)):
      return False
  return True


# Quaternions shorter than this are none (the identity rotation, as USD composes a zero quaternion).
_MIN_QUAT_NORM: Final = 1e-12


def _rotations(quats: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> Matrices3:
  """(n, 3, 3) column-vector rotations of (n, 4) quaternions (i, j, k, real: the Vt array order),
  normalised (a zero quaternion: the identity)."""
  norm = np.linalg.norm(quats, axis=1, keepdims=True)
  q = np.where(norm > _MIN_QUAT_NORM, quats / np.maximum(norm, _MIN_QUAT_NORM), (0.0, 0.0, 0.0, 1.0))
  x, y, z, w = q.T
  return np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
    ],
    axis=1,
  )


def instance_transforms(
  pi: UsdGeom.PointInstancer, protos: list[Usd.Prim], indices: np.ndarray[tuple[int], np.dtype[np.int64]], time: Time
) -> Matrices4 | None:
  """(n, 4, 4) column-vector transforms of the n instances of `pi` (prototypes `protos`, each
  instance's prototype `indices`) at `time`, composed as USD's
  `UsdGeom.PointInstancer.ComputeInstanceTransformsAtTime` does (the prototype root's local
  transform, then the instance's scale, orientation and position), each orientation normalised:
  the schema's orientations are unit quaternions, usually authored as `quath` (11-bit
  significands), and USD composes a rounded one unnormalised, a stretch of up to ~1e-3 that made
  every instance of a mesh a mesh copy at upload (`core/pipeline/upload/meshes.bake_stretches`).
  `orientationsf` takes precedence over `orientations` (the schema's rule). None: the positions,
  scales or orientations are not one per instance. Velocities and accelerations are not applied
  (`point_instancer` reports them)."""
  n = len(indices)
  positions = pi.GetPositionsAttr().Get(time)
  if positions is None or len(positions) != n:
    return None
  scales = pi.GetScalesAttr().Get(time)
  quat_attr = pi.GetOrientationsfAttr() if pi.GetOrientationsfAttr().HasAuthoredValue() else pi.GetOrientationsAttr()
  quats = quat_attr.Get(time)
  if (scales is not None and len(scales) not in (0, n)) or (quats is not None and len(quats) not in (0, n)):
    return None
  linear = np.tile(np.eye(3), (n, 1, 1))
  if quats is not None and len(quats):
    linear = _rotations(np.asarray(quats, np.float64).reshape(n, 4))
  if scales is not None and len(scales):
    linear = linear * np.asarray(scales, np.float64).reshape(n, 1, 3)
  local = np.stack([_proto_local(p, time) for p in protos]) if protos else np.zeros((0, 4, 4))
  proto = local[np.clip(indices, 0, max(len(protos) - 1, 0))]
  out = np.zeros((n, 4, 4))
  out[:, :3, :3] = linear @ proto[:, :3, :3]
  out[:, :3, 3] = np.einsum("nij,nj->ni", linear, proto[:, :3, 3]) + np.asarray(positions, np.float64).reshape(n, 3)
  out[:, 3, 3] = 1.0
  return out


def _proto_local(proto: Usd.Prim, time: Time) -> Matrix4:
  """A prototype root's local transform at `time` (the identity for a missing or non-xformable
  prim)."""
  xformable = UsdGeom.Xformable(proto) if proto.IsValid() else None
  return util.matrix(xformable.GetLocalTransformation(time)) if xformable else np.eye(4)


@final
@dataclasses.dataclass(kw_only=True)
class Gprims:
  """Gprim conversion state of one stage read (builder, materials, tessellated cones)."""

  builder: SceneBuilder
  report: LoadReport
  shade: shade.ShadeContext
  time: Time
  xcache: UsdGeom.XformCache  # the reader's, at `time`
  double_sided: bool  # every gprim double-sided (`Options.double_sided`, resolved)
  cones: dict[tuple[float, float], MeshId] = dataclasses.field(default_factory=dict)

  def material(self, prim: Usd.Prim, *, inside: bool) -> tuple[MaterialId, shade.MeshNeeds]:
    """(material id, mesh attributes it reads) bound to `prim` (full purpose, else all purpose);
    unbound: INHERIT inside a prototype (or for an instance), else NO_MATERIAL."""
    material, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(UsdShade.Tokens.full)
    if material:
      return select.material(self.shade, material)
    # The winning binding (USD's resolution: purposes, collections, strengths) may target a
    # material that is not a stage prim (e.g. one of a `.mtlx` document the stage references).
    target = _binding_target(rel) if rel else Sdf.Path.emptyPath
    if not target.isEmpty:
      self.report.drop("formats.usd.binding.unresolved", f"{prim.GetPath()} -> {target}")
    return (INHERIT if inside else NO_MATERIAL), shade.MeshNeeds()

  def flags(self, prim: Usd.Prim, *, double_sided: bool) -> int:
    """InstanceFlags of a gprim (double-sided, `primvars:doNotCastShadows`)."""
    flags = int(InstanceFlags.DEFAULT)
    if double_sided or self.double_sided:
      flags |= InstanceFlags.DOUBLE_SIDED
    attr = prim.GetAttribute("primvars:doNotCastShadows")
    if attr and attr.Get(self.time):
      flags &= ~int(InstanceFlags.CAST_SHADOW)
    return flags

  # Meshes -----------------------------------------------------------------------------------------

  def mesh(
    self, prim: Usd.Prim, local: Matrix4, suffix: str, body: str | None, *, inherited: shade.MeshNeeds | None
  ) -> list[Shape]:
    """The shapes of a UsdGeom.Mesh: one per materialBind part. `inherited`: inside a prototype,
    the attributes the binding of the instance placing it reads (None: not in a prototype)."""
    report = self.report
    report.count("usd.Mesh")
    inside = inherited is not None
    material, needs = self.material(prim, inside=inside)
    subsets = usd_mesh.material_subsets(prim, self.time, report)
    sub_mats = [self.material(subset, inside=inside) for subset, _ in subsets]
    # A part inheriting the instance's binding reads what that material reads (vertex colors, st1).
    parts_needs = [needs, *(n for _, n in sub_mats)]
    if inherited is not None and INHERIT in (material, *(m for m, _ in sub_mats)):
      parts_needs.append(inherited)
    wanted = _mesh_needs(material, parts_needs)
    data = usd_mesh.read_mesh(prim, self.time, wanted, report)
    if data is None:
      return []
    for sub, n in [(prim, needs), *((sp, n) for (sp, _), (_, n) in zip(subsets, sub_mats, strict=True))]:
      if len(n.uv) > 1 and n.uv[1] != wanted.uv1:
        report.drop(f"Mesh.uv_set1:{n.uv[1]}(the mesh's second set is {wanted.uv1})", sub)
    usd_mesh.report_attributes(prim, usd_mesh.CONSUMED_ATTRS, report)
    # An unbound mesh with per-vertex displayColor: its fallback material takes the vertex colors.
    display = Display(data.display_color, data.display_opacity, wanted.unbound and data.colors is not None)
    parts = [material] + [material if m in (NO_MATERIAL, INHERIT) else m for m, _ in sub_mats]
    ids = self._add_parts(prim, data, [faces for _, faces in subsets])
    if prim.HasAPI(UsdLux.MeshLightAPI):
      parts = self._mesh_light(prim, data, parts, display, inside=inside)
    flags = self.flags(prim, double_sided=data.double_sided)
    return [
      Shape(
        kind=InstanceKind.MESH,
        mesh=mesh_id,
        material=mat,
        display=display,
        size=(0.0, 0.0, 0.0),
        flags=flags,
        local=local,
        suffix=suffix,
        body=body,
      )
      for mesh_id, mat in zip(ids, parts, strict=True)
      if mesh_id >= 0
    ]

  def _add_parts(self, prim: Usd.Prim, data: usd_mesh.MeshData, subsets: list[usd_mesh.Faces]) -> list[MeshId]:
    """Adds the mesh split into its parts: the faces outside every subset, then each subset's."""
    tri_part = None
    if subsets:
      face_part = np.zeros(data.nface, dtype=np.int64)
      for i, faces in enumerate(subsets):
        face_part[faces[(faces >= 0) & (faces < data.nface)]] = i + 1
      tri_part = face_part[data.tri_face]
      self.report.count("usd.GeomSubset", len(subsets))
    attributes = MeshAttributes(
      normals=data.normals,
      normal_idx=data.normal_idx,
      uvs=data.uvs,
      uv_idx=data.uv_idx,
      uvs1=data.uvs1,
      uv1_idx=data.uv1_idx,
      colors=data.colors,
      color_idx=data.color_idx,
    )
    finite = finite_attributes(attributes, self.report, prim)
    ids = self.builder.add_mesh_parts(data.points, data.tris, tri_part=tri_part, nparts=len(subsets) + 1, **finite)
    self.report.count("triangles", len(data.tris))
    return ids

  def _mesh_light(
    self, prim: Usd.Prim, data: usd_mesh.MeshData, parts: list[MaterialId], display: Display, *, inside: bool
  ) -> list[MaterialId]:
    """Materials of a MeshLightAPI mesh's parts: they emit the light's radiance from the surface.

    `normalize` divides by the mesh's world surface area in stage units, the rule of the area
    lights (`lux`; RTX does not light from MeshLightAPI meshes). Inside a prototype (`inside`)
    the material is shared by every instance: the area is the prototype's, without the instances'
    scales (reported). The upload turns emitting parts shaped like a rectangle, disk or tube into
    lights (`raywarp.scene.lighting.emissive`, setting `lighting.emissive_lights`); other shapes
    light the scene through GI and reflections only.
    """
    rad = lux.radiance(self.report, prim, self.time)
    if not np.isfinite(rad).all() or (rad < 0.0).any():
      # One bad authored light must not abort the stage (as `lux.emit_light`'s invalid lights).
      self.report.drop("MeshLightAPI.invalid_radiance", f"{prim.GetPath()}: {tuple(float(c) for c in rad)}")
      return parts
    if util.bool_value(prim, lux.input_names("normalize"), default=False, time=self.time):
      world = util.matrix(self.xcache.GetLocalToWorldTransform(prim))
      area = float(emitter_shapes.face_areas(data.points.astype(np.float64) @ world[:3, :3].T, data.tris).sum())
      rad = rad / max(area, 1e-12)
      if inside:
        self.report.approx("MeshLightAPI.normalize(instanced: the prototype's area)", prim)
    self.report.approx("MeshLightAPI(emissive surface)", prim)
    return [self.shade.emitting(m if m >= 0 else self.shade.fallback(display), rad) for m in parts]

  # Primitives -------------------------------------------------------------------------------------

  def primitive(self, prim: Usd.Prim, local: Matrix4, suffix: str, body: str | None, *, inside: bool) -> Shape | None:
    """The shape of a UsdGeom primitive (Cube, Sphere, Cylinder, Capsule, Cone, Plane); None for
    authored dimensions out of UsdGeom's range (dropped, reported: the stage still loads)."""
    report = self.report
    t = prim.GetTypeName()
    report.count(f"usd.{t}")
    if not _valid_dimensions(prim, t, self.time):
      report.drop(f"{t}.invalid_size", prim)
      return None
    material, _ = self.material(prim, inside=inside)
    flags = self.flags(prim, double_sided=util.bool_value(prim, "doubleSided", default=False, time=self.time))
    axis = util.token_value(prim, "axis", "Z", self.time)
    rot = np.eye(4)
    rot[:3, :3] = geometry.axis_rotation(axis)
    kind, mesh, size = self._shape_of(prim, t, axis)
    # Analytic shapes map UVs procedurally (approximated); a tessellated Cone has none at all.
    if material >= 0 and self.builder.material(material).textures:
      report.approx("primitive.textures_without_uv", prim)
    usd_mesh.report_attributes(prim, _PRIMITIVE_ATTRS, report)
    display = Display(*usd_mesh.display(UsdGeom.PrimvarsAPI(prim), self.time, report))  # averaged when not constant
    return Shape(
      kind=kind,
      mesh=mesh,
      material=material,
      display=display,
      size=size,
      flags=flags,
      local=local @ rot,
      suffix=suffix,
      body=body,
    )

  def _shape_of(self, prim: Usd.Prim, t: str, axis: str) -> tuple[InstanceKind, MeshId, tuple[float, float, float]]:
    """(kind, mesh id, half sizes) of primitive type `t` along local +Z (`axis_rotation`)."""

    def get(name: str, default: float) -> float:
      """A float attribute of the primitive."""
      return util.float_value(prim, name, default, self.time)

    if t == "Cube":
      half = 0.5 * get("size", 2.0)
      return InstanceKind.BOX, NO_MESH, (half, half, half)
    if t == "Sphere":
      return InstanceKind.SPHERE, NO_MESH, (get("radius", 1.0), 0.0, 0.0)
    if t in ("Cylinder", "Capsule", "Cylinder_1", "Capsule_1"):
      cylinder = t.startswith("Cylinder")
      height = get("height", 2.0 if cylinder else 1.0)
      if t.endswith("_1"):
        default = 1.0 if cylinder else 0.5  # the schemas' fallbacks
        top, bottom = get("radiusTop", default), get("radiusBottom", default)
        radius = 0.5 * (top + bottom)
        if abs(top - bottom) > 1e-6 * max(top, bottom, 1e-12):
          self.report.approx(f"{t}.tapered(mean radius)", prim)
      else:
        radius = get("radius", 1.0 if t == "Cylinder" else 0.5)
      return InstanceKind.CYLINDER if cylinder else InstanceKind.CAPSULE, NO_MESH, (radius, 0.5 * height, 0.0)
    if t == "Cone":
      key = (get("radius", 1.0), get("height", 2.0))
      if key not in self.cones:
        points, tris, normals, normal_idx = geometry.cone_mesh(*key, CONE_SEGMENTS)
        self.cones[key] = self.builder.add_mesh(points, tris, normals=normals, normal_idx=normal_idx)
      self.report.approx("Cone.tessellated", prim)
      return InstanceKind.MESH, self.cones[key], (0.0, 0.0, 0.0)
    # Plane. UsdGeomPlane: width along X (axis Z / Y) or Z (axis X), length along Y (axis Z / X)
    # or Z (axis Y). The cyclic `axis_rotation` takes local X to world Y (axis X) / Z (axis Y):
    # the local half sizes swap for those two axes.
    width, length = get("width", 2.0), get("length", 2.0)
    sides = (0.5 * width, 0.5 * length) if axis == "Z" else (0.5 * length, 0.5 * width)
    return InstanceKind.PLANE, NO_MESH, (*sides, 0.0)

  # Point instancers -------------------------------------------------------------------------------

  def point_instancer(
    self, stage: Usd.Stage, prim: Usd.Prim, local: Matrix4, suffix: str, body: str | None
  ) -> PointInstancer | None:
    """The instances of a UsdGeom.PointInstancer (None: empty or inconsistent, reported), with the
    binding its prototypes inherit."""
    report = self.report
    pi = UsdGeom.PointInstancer(prim)
    protos = [stage.GetPrimAtPath(t) for t in pi.GetPrototypesRel().GetTargets()]
    indices = pi.GetProtoIndicesAttr().Get(self.time)
    if not protos or indices is None or not len(indices):
      report.drop("PointInstancer.empty", prim)
      return None
    idx = np.asarray(indices, dtype=np.int64)
    xforms = instance_transforms(pi, protos, idx, self.time)
    if xforms is None:
      report.drop("PointInstancer.size_mismatch", prim)
      return None
    mask = np.asarray(pi.ComputeMaskAtTime(self.time), dtype=bool)
    ids = np.arange(len(idx)).astype(np.int64)
    keep = np.ones(len(idx), dtype=bool)
    if mask.size == len(idx):
      keep = mask
      report.count("PointInstancer.invisible", int((~mask).sum()))
    keep &= (idx >= 0) & (idx < len(protos))
    valid = np.array([p.IsValid() for p in protos], bool)
    missing = keep & ~valid[np.clip(idx, 0, len(protos) - 1)]
    if missing.any():
      report.drop("PointInstancer.missing_prototype", prim, n=int(missing.sum()))
      keep &= ~missing
    if any(a.ValueMightBeTimeVarying() for a in (pi.GetPositionsAttr(), pi.GetOrientationsAttr(), pi.GetProtoIndicesAttr())):
      report.approx("PointInstancer.time_samples(one time read)", prim)
    for name in ("velocities", "angularVelocities", "accelerations"):
      if util.authored(prim, name):
        report.drop(f"PointInstancer.{name}", prim)
    report.count("usd.PointInstancer.instances", int(keep.sum()))
    material, needs = self.material(prim, inside=True)
    return PointInstancer(
      prototypes=protos,
      indices=idx[keep],
      xforms=xforms[keep],
      ids=ids[keep],
      local=local,
      suffix=suffix,
      material=material,
      needs=needs,
      body=body,
    )
