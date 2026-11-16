"""Emissive geometry -> rect / disk lights (a pass over a built `Scene`, its public tables only).

Emission is otherwise only seen at the hit: an emissive ceiling panel lights nothing. This pass
fits lights to the emitting parts of emissive instances and gives them the parts' emitted
radiance (`Materials.emission`: `emission_color` x `emission_luminance`, nits), so the fixture
lights the room with `lighting.area_lights="ltc"` (with "point": a point light of intensity
L A cos(theta)).

Detection, per instance with an emissive material (radiance >= `min_radiance`; shapes in
`raywarp.scene.lighting.emitter_shapes`):

* boxes whose thinnest side is < `max_thickness` x their largest, and finite planes; other boxes
  one light per face, emitting outwards (faces carrying less than MIN_PART_POWER of the box's
  power, a strip's ends, stay geometry only);
* meshes: the emitting faces (below) as one flat rectangle: the world-space points' oriented
  bounding box (normal = least-variance axis, then the minimum-area rectangle of the projected
  convex hull) must be flat (thickness < `max_thickness` x the largest side) and rectangular (the
  faces' area covering the rectangle, counted on the side that covers most of it, >= `min_fill`
  x its area);
* else one light per connected part of those faces (at most MAX_PARTS parts; parts carrying less
  than MIN_PART_POWER of the power stay geometry): a flat rectangle, a flat disk (a round
  downlight's diffuser) or a tube (a fluorescent tube in its troffer). A tube becomes its
  silhouette (a 2 r x l rect at its surface) facing the side its emission leaves the instance's
  own geometry by: the core's tube light samples its axis, inside the tube's own triangles.

A one-sided emitter (planes; meshes whose faces agree on a side) gets one light, emitting along the
faces' normals. A two-sided one (boxes, closed or double-sided meshes) gets two, back to back,
one per side, unless one side emits into the instance's own geometry (a diffuser's back facing
its housing: rays from the face hit the mesh), which then gets none. Coplanar rectangles of one
fixture that together fill a rectangle (a diffuser split by frame bars) are one light. Each
light is placed `offset` in front of the emitting face, so shadow rays towards it end before
reaching the geometry: the panel does not shadow its own light.
`Lights.instance` records the instance.

Textured emission (meshes with UVs): the texture (the scene's decoded texels, `images/decode.py`,
read at most TEXEL_LIMIT texels wide) is read at each face's centroid UV, and only the faces
emitting at least half the peak (a fixture's diffuser, a screen, a tube) are fitted; a light's
radiance is the factor times the texture's mean over its faces (FACE_SAMPLES stratified points per
face, area-weighted), divided by the light's area over theirs: the light emits the faces' power (a
diffuser's grid lines and dimmer edges included); a tube's silhouette shows the tube's mean surface
radiance. A fixture whose emitting parts fit none of these shapes (a clock face, a sign's letters)
stays emissive geometry only. For primitives the factor alone is used (approximate).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from raywarp.cache.code_version import code_digest
from raywarp.cache.paths import cache_dir, write_atomic
from raywarp.scene.images import colorspace, mips
from raywarp.scene.images import decode as imaging
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.lighting import emitter_fixture as fixtures
from raywarp.scene.lighting import emitter_shapes as shapes
from raywarp.scene.lighting.checks import check_light
from raywarp.scene.lighting.emitter_shapes import Emitter, Panel
from raywarp.scene.model import lights as scene_lights
from raywarp.scene.model import tables as scene_tables
from raywarp.scene.model import types as scene_types
from raywarp.scene.model.lights import Light, Lights
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.tables import MeshView
from raywarp.scene.model.types import (
  CHANNEL_RGB,
  UV_SET_WORLD,
  Array,
  Column,
  InstanceId,
  InstanceKind,
  LightKind,
  LinearRGBA,
  MaterialId,
  PixelType,
  Rows,
  Slot,
  TextureId,
)

# Emission textures are read at most this many texels wide (box-filtered in linear space), on at
# most PREFETCH_THREADS threads.
TEXEL_LIMIT: Final = 256
PREFETCH_THREADS: Final = min(16, os.cpu_count() or 1)

type Vec3 = np.ndarray[[3], np.dtype[np.float64]]
# A fixture's emitting faces become at most this many lights (one per connected part); a part
# carrying less than MIN_PART_POWER of the fixture's emitted power stays glowing geometry only.
MAX_PARTS: Final = 8
MIN_PART_POWER: Final = 0.02


def _face_samples(g: int = 8) -> Rows[np.float64, 3]:
  """Stratified barycentric points per face: a g x g grid on the unit square, folded onto the
  triangle."""
  u, v = (a.ravel() for a in np.meshgrid((np.arange(g) + 0.5) / g, (np.arange(g) + 0.5) / g))
  fold = u + v > 1.0
  u, v = np.where(fold, 1.0 - u, u), np.where(fold, 1.0 - v, v)
  # Pulled 1 % towards the centroid: points on an edge would read the neighbouring texel.
  return (0.99 * np.stack([1.0 - u - v, u, v], 1) + 0.01 / 3.0).astype(np.float64)


FACE_SAMPLES: Final = _face_samples()


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Emission:
  """The emissive texture binding of a material (`Materials` columns of `Slot.EMISSION_COLOR`)."""

  texels: np.ndarray[[int, int, 4], np.dtype[np.float32]]
  channel: int
  scale: np.ndarray[[3], np.dtype[np.float32]]
  bias: np.ndarray[[3], np.dtype[np.float32]]
  uv_xform: np.ndarray[[2, 3], np.dtype[np.float64]]
  uv_set: int

  def at(self, uvs: Rows[np.float64, 2]) -> Rows[np.float32, 3]:
    """Emitted factor (texel x scale + bias, nearest texel) at mesh UVs `uvs`."""
    m = self.uv_xform
    st = uvs @ m[:, :2].T + m[:, 2]
    h, w = self.texels.shape[:2]
    col = np.mod(np.floor(st[:, 0] * w), w).astype(np.int64)
    row = np.mod(np.floor((1.0 - st[:, 1]) * h), h).astype(np.int64)
    texel = self.texels[row, col]
    c = self.channel
    vals = texel[:, :3] if c == CHANNEL_RGB else np.repeat(texel[:, c : c + 1], 3, 1)
    return vals * self.scale + self.bias


def _texels(scene: Scene, texture: TextureId, cache: dict[TextureId, LinearRGBA]) -> LinearRGBA:
  """Linear RGBA texels of a scene texture, at most TEXEL_LIMIT wide (`_reduced`; read once per
  preparation through `cache`), its color space reported per material binding."""
  space = colorspace.from_code(int(scene.textures.colorspace[texture]))
  pixel = int(scene.images.pixel[int(scene.textures.image[texture])])
  # Reported per material binding, even when another binding reuses these texels.
  if not colorspace.stored_as_is(space, float_texels=pixel != PixelType.U8):
    scene.report.approx(f"emissive_light.texture.colorspace:{space.value}(converted to Rec.709)", f"texture {texture}")
  if texture not in cache:
    cache[texture] = _reduced(scene, texture)
  return cache[texture]


def _reduced(scene: Scene, texture: TextureId) -> LinearRGBA:
  """Linear RGBA texels of a scene texture, at most TEXEL_LIMIT wide (the nearest mip,
  box-filtered)."""
  images = scene.images
  image = int(scene.textures.image[texture])
  space = colorspace.from_code(int(scene.textures.colorspace[texture]))
  size = max(int(images.width[image]), int(images.height[image]))
  level = 0
  while level + 1 < int(images.levels[image]) and (size >> level) > TEXEL_LIMIT:
    level += 1
  pixels = images.level(image, level)
  if colorspace.stored_as_is(space, float_texels=pixels.dtype != np.uint8):
    texels = imaging.linear_rgba(pixels, transfer="srgb" if space == ColorSpace.SRGB_REC709 else "linear")
  else:
    texels = colorspace.decode(pixels, space)
  factor = 1
  while max(texels.shape[:2]) > TEXEL_LIMIT * factor:
    factor *= 2
  return mips.box_reduce(texels, factor)


def _prefetch(scene: Scene, materials: list[MaterialId], cache: dict[TextureId, LinearRGBA]) -> None:
  """Reduces the emission textures of `materials` on a thread pool into `cache` (numpy releases
  the GIL in the decode and the box filter); `_texels` then finds them."""
  slot = Slot.EMISSION_COLOR
  textures = sorted({TextureId(int(scene.materials.tex[m, slot])) for m in materials} - {TextureId(-1)})
  if len(textures) < 2:
    return
  with concurrent.futures.ThreadPoolExecutor(min(PREFETCH_THREADS, len(textures))) as pool:
    for texture, texels in zip(textures, pool.map(lambda t: _reduced(scene, t), textures), strict=True):
      cache[texture] = texels


def _emission(
  scene: Scene, material: MaterialId, cache: dict[MaterialId, _Emission | None], texels: dict[TextureId, LinearRGBA]
) -> _Emission | None:
  """The emissive texture binding of `material` (None: untextured), its texels read once."""
  if material in cache:
    return cache[material]
  mats = scene.materials
  texture = TextureId(int(mats.tex[material, Slot.EMISSION_COLOR]))
  out = None
  if texture >= 0:
    out = _Emission(
      texels=_texels(scene, texture, texels),
      channel=int(mats.tex_channel[material, Slot.EMISSION_COLOR]),
      scale=mats.tex_scale[material, Slot.EMISSION_COLOR, :3],
      bias=mats.tex_bias[material, Slot.EMISSION_COLOR, :3],
      uv_xform=mats.uv_xform[material, Slot.EMISSION_COLOR].astype(np.float64),
      uv_set=int(mats.uv_set[material, Slot.EMISSION_COLOR]),
    )
  cache[material] = out
  return out


def _box_panel(xform: np.ndarray[[3, 4], np.dtype[np.float64]], size: Vec3, max_thickness: float) -> Panel | None:
  """The panel of a box thinner than `max_thickness` x its largest side (two-sided; a zero
  thickness included), or None."""
  axes = xform[:, :3] * size[None, :]  # world half-extent vectors
  lengths = np.linalg.norm(axes, axis=0)
  k = int(np.argmin(lengths))
  others = [j for j in range(3) if j != k]
  # Half extents on both sides: thickness / largest side = lengths[k] / max(lengths).
  if lengths.min() >= max_thickness * lengths.max() or lengths[others].min() <= 0:
    return None
  if lengths[k] > 0.0:
    normal = axes[:, k] / lengths[k]
  else:
    # A zero-thickness box (size or scale 0 across it): the normal of its two sides' plane.
    normal = np.cross(axes[:, others[0]], axes[:, others[1]])
    if not shapes.norm3(normal) > 0.0:
      return None
    normal = normal / shapes.norm3(normal)
  return Panel(
    centre=xform[:, 3],
    normal=normal,
    axis=axes[:, others[0]] / lengths[others[0]],
    half=lengths[others],
    half_thickness=float(lengths[k]),
    side=0,
  )


def _box_faces(xform: np.ndarray[[3, 4], np.dtype[np.float64]], size: Vec3) -> list[Panel]:
  """The faces of a solid box (not a panel) as one-sided panels emitting outwards, but those
  carrying less than MIN_PART_POWER of the box's emitted power (a strip's ends): an emissive box
  then lights its surroundings as the area lights it is (the GI probes and the camera rays sample
  it directly, not by chance hits)."""
  axes = xform[:, :3] * size[None, :]  # world half-extent vectors
  lengths = np.linalg.norm(axes, axis=0)
  if lengths.min() <= 0.0:
    return []
  areas = [4.0 * float(np.prod(np.delete(lengths, k))) for k in range(3)]
  total = 2.0 * sum(areas)
  faces = []
  for k in range(3):
    if areas[k] < MIN_PART_POWER * total:
      continue
    others = [j for j in range(3) if j != k]
    for sign in (1.0, -1.0):
      normal = sign * axes[:, k] / lengths[k]
      faces.append(
        Panel(
          centre=xform[:, 3] + sign * axes[:, k],
          normal=normal,
          axis=axes[:, others[0]] / lengths[others[0]],
          half=lengths[others],
          half_thickness=0.0,
          side=1,
        )
      )
  return faces


def _buried_test(scene: Scene):
  """buried(light, i): whether `light` (of instance i) sits inside another box instance (an
  emissive box's face against a wall or the floor: its light would light nothing)."""
  inst = scene.instances
  boxes = [j for j in range(len(inst)) if int(inst.kind[j]) == InstanceKind.BOX]
  frames = [np.linalg.pinv(inst.xform[j][:, :3].astype(np.float64) * inst.size[j].astype(np.float64)[None, :]) for j in boxes]

  def buried(light: Light, i: int) -> bool:
    p = np.asarray(light.xform, np.float64)[:, 3]
    for j, inv in zip(boxes, frames, strict=True):
      if j != i and np.all(np.abs(inv @ (p - inst.xform[j][:, 3])) < 1.0):
        return True
    return False

  return buried


def _plane_panel(xform: np.ndarray[[3, 4], np.dtype[np.float64]], size: Vec3) -> Panel | None:
  """The panel of a finite plane (one-sided, along its +Z), or None (no area: a zero size, or a
  transform that flattens it to a line or a point)."""
  if size[0] <= 0 or size[1] <= 0:
    return None
  lin = xform[:, :3]
  sx, sy = shapes.norm3(lin[:, 0]), shapes.norm3(lin[:, 1])
  normal = np.cross(lin[:, 0], lin[:, 1])
  if not shapes.norm3(normal) > 0.0:
    return None
  normal = normal / shapes.norm3(normal)
  return Panel(
    centre=xform[:, 3],
    normal=normal,
    axis=lin[:, 0] / sx,
    half=np.array([size[0] * sx, size[1] * sy]),
    half_thickness=0.0,
    side=1,
  )


def _subset(points: Rows[np.float64, 3], tris: Rows[np.int64, 3]) -> tuple[Rows[np.float64, 3], Rows[np.int64, 3]]:
  """(the points `tris` use, `tris` re-indexed into them)."""
  used = np.unique(tris)
  remap = np.zeros(len(points), np.int64)
  remap[used] = np.arange(len(used))
  return points[used], remap[tris]


def _emitting_area(e: Emitter) -> float:
  """The area a flat emitter's faces' power spreads over: one side's area per side the faces
  cover (a two-sided fit, `side` 0, emits from both: a closed thin box or a double-layer sheet,
  whose faces' power counts both layers), fixed before `open_sides` drops a side into the
  fixture."""
  return shapes.area(e) * (2.0 if e.side == 0 else 1.0)


def _part(
  points: Rows[np.float64, 3],
  tris: Rows[np.int64, 3],
  fixture: tuple[Rows[np.float64, 3], Rows[np.int64, 3]],
  *,
  limits: tuple[float, float],
  offset: float,
) -> tuple[Emitter, float] | None:
  """(emitter, area its power spreads over) of one connected emitting part, or None.

  A flat rectangle or disk emits the part's power over its own area, per emitting side
  (`_emitting_area`); a tube's silhouette (facing the side its emission leaves the `fixture`
  (points, triangles) by) at the tube's surface radiance, the part's power over its faces' area.
  """
  max_thickness, min_fill = limits
  flat = shapes.flat(points, tris, max_thickness)  # the rectangle and the disk test the same fit
  panel = shapes.mesh_panel(flat, min_fill=min_fill)
  if panel is not None:
    return panel, _emitting_area(panel)
  disk = shapes.mesh_disk(flat, min_fill=min_fill)
  if disk is not None:
    return disk, _emitting_area(disk)
  tube = shapes.mesh_tube(points, tris, limits[1])
  if tube is None:
    return None
  side = fixtures.open_side(tube, fixture[0], fixture[1], offset=offset)
  if side is None:
    return None
  faces = float(shapes.face_areas(points, tris).sum())
  silhouette = Panel(
    centre=tube.centre + side * tube.radius,
    normal=side,
    axis=tube.axis,
    half=np.array([0.5 * tube.length, tube.radius]),
    half_thickness=0.0,
    side=1,
  )
  return silhouette, faces


def _fit(
  points: Rows[np.float64, 3],
  tris: Rows[np.int64, 3],
  power: Rows[np.float64, 3],
  *,
  uniform: bool,
  fixture: tuple[Rows[np.float64, 3], Rows[np.int64, 3]],
  limits: tuple[float, float],
  offset: float,
) -> list[tuple[Emitter, Vec3]]:
  """(emitter, radiance scale) of emitting faces `tris` with power `power` (per face: factor x
  area): one flat rectangle over all of them, else one emitter per connected part (`_part`) for
  at most MAX_PARTS parts; a part carrying less than MIN_PART_POWER of the power stays geometry.
  A two-sided emitter loses the side that emits into its `fixture` (`emitter_fixture.open_sides`),
  and coplanar part rectangles filling a common one become it (`emitter_fixture.merge_coplanar`).
  `uniform`: an untextured emission (scale 1: each emitter shows the faces' radiance).
  """
  sub_points, sub_tris = _subset(points, tris)
  whole = shapes.mesh_panel(shapes.flat(sub_points, sub_tris, limits[0]), min_fill=limits[1])
  if whole is not None:
    scale = np.ones(3) if uniform else power.sum(0) / _emitting_area(whole)
    return [(fixtures.open_sides(whole, *fixture, offset=offset), scale)]
  parts = shapes.components(sub_points, sub_tris)
  if len(parts) > MAX_PARTS:
    return []
  total = float(power.sum(0).max())
  out: list[tuple[Emitter, Vec3]] = []
  for faces in parts:
    p = power[faces].sum(0)
    if float(p.max()) < MIN_PART_POWER * total:
      continue
    part_points, part_tris = _subset(sub_points, sub_tris[faces])
    found = _part(part_points, part_tris, fixture, limits=limits, offset=offset)
    if found is not None:
      emitter = fixtures.open_sides(found[0], *fixture, offset=offset)
      out.append((emitter, np.ones(3) if uniform else p / max(found[1], 1e-30)))
  return fixtures.merge_coplanar(out)


def _mesh_emitters(
  mesh: MeshView,
  xform: np.ndarray[[3, 4], np.dtype[np.float64]],
  *,
  emission: _Emission | None,
  limits: tuple[float, float],
  offset: float,
) -> tuple[list[tuple[Emitter, Vec3]], bool]:
  """((emitter, radiance scale) list, textured) of a mesh placed by `xform` (the frame the
  emitters are fitted in; `offset` in that frame's units).

  Textured (meshes with UVs): the faces (with UVs) emitting at least half the peak at their
  centroid, their power the texture's mean over each face (FACE_SAMPLES) times its area;
  without UVs, or for a world-projected texture (UV_SET_WORLD), the texture is ignored (factor
  only, reported by `_panel_lights`): every face, uniform.
  """
  world = mesh.points.astype(np.float64) @ xform[:, :3].T + xform[:, 3]
  tris = mesh.tris.astype(np.int64)
  uv1 = emission is not None and emission.uv_set == 1
  uvs, uv_idx = (mesh.uvs1, mesh.uv1_idx) if uv1 else (mesh.uvs, mesh.uv_idx)
  if len(tris) == 0:
    return [], False
  if emission is None or len(uvs) == 0 or emission.uv_set == UV_SET_WORLD:
    area = shapes.face_areas(world, tris)
    return _fit(
      world, tris, np.repeat(area[:, None], 3, 1), uniform=True, fixture=(world, tris), limits=limits, offset=offset
    ), False
  tri_uv = uvs[np.maximum(uv_idx, 0)].astype(np.float64)  # corner UVs (faces without: masked)
  e = emission.at(tri_uv.mean(1))
  lum = np.where(np.all(uv_idx >= 0, axis=1), e.max(1), 0.0)
  if lum.max() <= 0:
    return [], True
  lit = lum >= 0.5 * lum.max()
  sub = tris[lit]
  area = shapes.face_areas(world, sub)
  points = np.einsum("sk,fkd->fsd", FACE_SAMPLES, tri_uv[lit]).reshape(-1, 2)
  face_mean = emission.at(points).reshape(int(lit.sum()), len(FACE_SAMPLES), 3).mean(1)
  power: Rows[np.float64, 3] = (face_mean * area[:, None]).astype(np.float64)
  return _fit(world, sub, power, uniform=False, fixture=(world, tris), limits=limits, offset=offset), True


def _lights(e: Emitter, radiance: Vec3, offset: float, *, instance: InstanceId, name: str) -> list[Light]:
  """The rect (panel) or disk light(s) of an emitter: one per emitting side, `offset` in front of
  its face; the color is the panel's radiance [nits] (intensity 1, exposure 0: the upload's
  radiance is exactly that color)."""
  out: list[Light] = []
  if isinstance(e, Panel):
    kind, axis, width, height, radius = LightKind.RECT, e.axis, float(2.0 * e.half[0]), float(2.0 * e.half[1]), 0.0
  else:
    kind, axis, width, height, radius = LightKind.DISK, shapes.frame(e.normal)[0], 0.0, 0.0, float(e.radius)
  r, g, b = (float(c) for c in radiance)
  for side in (1, -1) if e.side == 0 else (e.side,):
    emit = e.normal * side
    z = -emit  # a rect / disk light emits along its -Z
    x = axis - z * float(axis @ z)
    x = x / shapes.norm3(x)
    xform = np.zeros((3, 4))
    xform[:, 0], xform[:, 1], xform[:, 2] = x, np.cross(z, x), z
    xform[:, 3] = e.centre + emit * (e.half_thickness + offset)
    suffix = "#emissive_light" if side == 1 or e.side != 0 else "#emissive_light_back"
    out.append(
      Light(
        kind=kind,
        color=(r, g, b),
        radius=radius,
        width=width,
        height=height,
        length=0.0,
        angle=0.0,
        xform=xform,
        instance=instance,
        path=name + suffix,
      )
    )
  return out


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Fits:
  """What one preparation reuses: emission bindings, reduced texels and the mesh fits per (mesh,
  material, scale) in the mesh's own frame (`_instance_emitters`)."""

  emissions: dict[MaterialId, _Emission | None] = dataclasses.field(default_factory=dict)
  texels: dict[TextureId, LinearRGBA] = dataclasses.field(default_factory=dict)
  local: dict[tuple[int, int, float], tuple[list[tuple[Emitter, Vec3]], bool]] = dataclasses.field(default_factory=dict)


def _instance_emitters(
  scene: Scene, i: int, fits: _Fits, *, limits: tuple[float, float], offset: float
) -> tuple[list[tuple[Emitter, Vec3]], bool]:
  """((emitter, radiance scale) list, textured) of instance `i` (meshes, boxes, planes).

  A mesh under a similarity (uniform scale x rotation + translation) is fitted once per (mesh,
  material, scale) in its own frame and the emitters placed (`emitter_shapes.placed`: the fits
  are equivariant, the radiance scales invariant); under another transform, in world space.
  """
  inst = scene.instances
  kind, xform, size = int(inst.kind[i]), inst.xform[i].astype(np.float64), inst.size[i].astype(np.float64)
  if kind == InstanceKind.MESH and int(inst.mesh[i]) >= 0:
    mesh_id, material = int(inst.mesh[i]), MaterialId(int(inst.material[i]))
    emission = _emission(scene, material, fits.emissions, fits.texels)
    scale = shapes.similarity(xform)
    if scale is None:
      return _mesh_emitters(scene.meshes.get(mesh_id), xform, emission=emission, limits=limits, offset=offset)
    key = (mesh_id, int(material), scale)
    if key not in fits.local:
      own = np.c_[np.eye(3), np.zeros(3)]
      fits.local[key] = _mesh_emitters(scene.meshes.get(mesh_id), own, emission=emission, limits=limits, offset=offset / scale)
    emitters, textured = fits.local[key]
    return [(shapes.placed(e, xform, scale), k) for e, k in emitters], textured
  panel = None
  if kind == InstanceKind.BOX:
    panel = _box_panel(xform, size, limits[0])
    if panel is None:
      return [(face, np.ones(3)) for face in _box_faces(xform, size)], False
  elif kind == InstanceKind.PLANE:
    panel = _plane_panel(xform, size)
  return ([] if panel is None else [(panel, np.ones(3))]), False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Limits:
  """The detection thresholds of `add_panel_lights`."""

  min_radiance: float
  max_thickness: float
  min_fill: float
  offset: float


def add_panel_lights(
  scene: Scene,
  *,
  min_radiance: float = 1.0,
  max_thickness: float = 0.02,
  min_fill: float = 0.9,
  offset: float = 2.0e-3,
) -> Scene:
  """`scene` plus rect / disk lights for the flat rectangles, disks and tubes of every emissive
  instance.

  The fit is remembered on disk (`<cache dir>/emissive/panels-<key>.npz`: the added lights and
  the report entries) under the digest of everything it reads (`_fit_key`): the emissive
  instances, their materials' emission and emission textures (image content digests), their
  meshes, the thresholds and the code; a repeat upload of the same scene fits nothing.
  """
  limits = _Limits(min_radiance=min_radiance, max_thickness=max_thickness, min_fill=min_fill, offset=offset)
  lit = _lit_instances(scene, min_radiance)
  root = cache_dir()
  path = None if root is None or not len(lit) else root / "emissive" / f"panels-{_fit_key(scene, lit, limits)}.npz"
  cached = None if path is None else _read_fit(path)
  if cached is None:
    report = LoadReport(source=scene.report.source)
    added = _panel_lights(dataclasses.replace(scene, report=report), lit, limits)
    if path is not None:
      _write_fit(path, added, report)
  else:
    added, report = cached
  scene.report.merge(report)
  return dataclasses.replace(scene, lights=scene.lights.concat(added))


def _lit_instances(scene: Scene, min_radiance: float) -> Column[np.int64]:
  """The instances with an emissive material (radiance >= `min_radiance`)."""
  mats = scene.materials
  emitted = mats.emission()
  bright = np.flatnonzero(emitted.max(axis=1) >= min_radiance) if len(mats) else np.zeros(0, np.int64)
  return np.flatnonzero(np.isin(np.asarray(scene.instances.material), bright)).astype(np.int64, copy=False)


# The fit entries' format (`_write_fit`) and the modules deciding a fit (`_fit_key`).
_FIT_FORMAT: Final = 1
_FIT_MODULES: Final = (
  sys.modules[__name__],
  shapes,
  fixtures,
  scene_lights,
  colorspace,
  imaging,
  mips,
  scene_tables,
  scene_types,
)


@functools.cache
def _fit_code() -> str:
  """`code_digest` of the modules deciding a fit (`_FIT_MODULES`, named by their dotted names)."""
  return code_digest((f"{m.__name__}.py", Path(m.__file__).read_bytes()) for m in _FIT_MODULES if m.__file__ is not None)


def _fit_key(scene: Scene, lit: Column[np.int64], limits: _Limits) -> str:
  """The digest of every input of `_panel_lights` (`add_panel_lights`). Images are identified by
  their content digests (`Images.digest`, set by `SceneBuilder`; an edit of an image's texels must
  set its digest anew).

  Raises:
    ValueError: for an emission image without a content digest.
  """
  inst, mats, textures = scene.instances, scene.materials, scene.textures
  h = hashlib.blake2b(digest_size=20)

  def add(*arrays: npt.ArrayLike) -> None:
    """Hashes each array's dtype, shape and bytes."""
    for value in arrays:
      a: Array = np.ascontiguousarray(value)
      h.update(f"{a.dtype.str}{a.shape}".encode())
      h.update(a.data)

  h.update(repr((_FIT_FORMAT, _fit_code(), dataclasses.astuple(limits), len(inst), len(textures))).encode())
  add(lit, inst.kind[lit], inst.xform[lit], inst.size[lit], inst.mesh[lit], inst.material[lit], inst.prim[lit], inst.body[lit])
  prims = np.asarray(inst.prim)[lit]
  h.update(repr([inst.paths[int(p)] if p >= 0 else "" for p in prims]).encode())
  material = np.unique(np.asarray(inst.material)[lit])
  slot = Slot.EMISSION_COLOR
  add(mats.emission()[material], mats.tex[material, slot], mats.tex_channel[material, slot], mats.tex_scale[material, slot])
  add(mats.tex_bias[material, slot], mats.uv_xform[material, slot], mats.uv_set[material, slot])
  texture = np.asarray(mats.tex[material, slot])
  texture = texture[texture >= 0]
  digests = np.asarray(scene.images.digest)[np.asarray(textures.image)[texture]]
  if len(digests) and not digests.reshape(len(digests), -1).any(axis=1).all():
    raise ValueError("an emission image has no content digest (Images.digest): a scene not made by SceneBuilder")
  add(textures.colorspace[texture], digests)
  for mesh in np.unique(np.asarray(inst.mesh)[lit]).tolist():
    if mesh >= 0:
      add(*dataclasses.astuple(scene.meshes.get(mesh)))
  return h.hexdigest()


def _read_fit(path: Path) -> tuple[Lights, LoadReport] | None:
  """A remembered fit (`_write_fit`), or None (a miss; a damaged file warns and is fitted again)."""
  try:
    with np.load(path, allow_pickle=False) as data:
      arrays = {k[2:]: data[k] for k in data.files if k.startswith("l.")}
      report = LoadReport.from_dict(json.loads(bytes(data["report"]).decode()))
    return Lights.from_arrays(arrays), report
  except FileNotFoundError:
    return None
  except (OSError, ValueError, KeyError, TypeError) as e:
    warnings.warn(f"raywarp cache: unreadable {path} ({e}); fitting the emissive lights again", RuntimeWarning, stacklevel=3)
    return None


def _write_fit(path: Path, added: Lights, report: LoadReport) -> None:
  """Remembers a fit atomically (`cache.paths.write_atomic`: a failed write warns and leaves no
  entry)."""
  text = np.frombuffer(json.dumps(report.to_dict()).encode(), np.uint8)
  lights = {f"l.{k}": v for k, v in added.arrays().items()}
  write_atomic(path, lambda f: np.savez(f, allow_pickle=False, report=text, **lights), "emissive fit cache")


def _panel_lights(scene: Scene, lit: Column[np.int64], limits: _Limits) -> Lights:
  """The lights `add_panel_lights` adds for the emissive instances `lit` (reported to
  `scene.model.report`)."""
  inst, mats, report = scene.instances, scene.materials, scene.report
  thresholds = (limits.max_thickness, limits.min_fill)
  fits = _Fits()
  emitted = mats.emission()
  material = np.asarray(inst.material)
  # Only meshes read emission textures (`_instance_emitters`; primitives use the factor alone).
  meshes = lit[(np.asarray(inst.kind)[lit] == InstanceKind.MESH) & (np.asarray(inst.mesh)[lit] >= 0)]
  _prefetch(scene, [MaterialId(int(m)) for m in np.unique(material[meshes])], fits.texels)
  added: list[Light] = []
  bodies: list[int] = []
  buried = _buried_test(scene)
  for i in lit.tolist():
    mid = int(material[i])
    radiance = emitted[mid]
    emitters, textured = _instance_emitters(scene, i, fits, limits=thresholds, offset=limits.offset)
    prim = int(inst.prim[i])
    name = inst.paths[prim] if prim >= 0 else f"instance{i}"
    if emitters and int(mats.tex[mid, Slot.EMISSION_COLOR]) >= 0 and not textured:
      report.approx("emissive_light.texture(factor only)", name)
    for k, (emitter, scale) in enumerate(emitters):
      if textured and (radiance * scale).max() < limits.min_radiance:
        continue
      # One path per light: a fixture's parts are numbered.
      part = name if len(emitters) == 1 else f"{name}#part{k}"
      new = [
        li for li in _lights(emitter, radiance * scale, limits.offset, instance=InstanceId(i), name=part) if not buried(li, i)
      ]
      for light in new:
        check_light(light, ntexture=len(scene.textures), ninstance=len(inst), where=f"emissive light {light.path!r}")
      added += new
      bodies += [int(inst.body[i])] * len(new)
  report.count("lights.emissive_panels", len(added))
  return Lights.from_records(added, bodies=bodies)
