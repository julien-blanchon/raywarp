"""Contact ambient occlusion of the dynamic objects: an analytic occlusion of each pixel by box and
sphere proxies of the per-world instances, applied by the deferred display pass
(`post/display/deferred.py`) from the render's depth output.

GI probes and the environment's visibility are baked over the static scenery: they are blind to
the per-world instances (a robot, a grasped or a randomised object), which then cast no indirect
shadow under environment or GI light. Those instances are exactly what this pass adds: each one
is a proxy, its local box (a mesh's box, a primitive's extents) or, for a sphere, the sphere,
posed per world and camera every frame (`ProxySet.gather`, one small launch; a hidden instance
has none). Groups of 8 consecutive proxies (a body's parts, mostly) get the bounding sphere of
their reach, and 16 x 16 pixel screen tiles list the groups whose sphere they may see (two more
small launches): a pixel tests its tile's groups' spheres, then only the proxies of the groups it
lies in. Static contacts stay
the probes' and the environment's (no double darkening).

Per pixel, the view-space point and its normal come from the depth output (the normal from the
depth's central differences, taking the smaller step on each axis so that it does not cross a
silhouette). Each proxy occludes the projected solid angle it covers (the form factor of its
silhouette; Quilez's analytic box and sphere occlusion), faded to 0 at REACH times its size;
the proxies combine as independent occluders, visible = prod(1 - F_i); a proxy below the point's
tangent plane covers nothing. The receivers are the static scenery: a point inside a proxy is on
a dynamic object and is left as it is (overlapping proxies of a robot's links would otherwise
draw seams where one's box ends on the other).
No noise, no taps, no temporal state: the same pixel shows the same occlusion every frame.

Only the ambient part of the pixel is darkened. It is estimated after shading: the share of the
exposed color below AMBIENT_LEVEL of display white (a dim pixel is mostly indirect light, a
bright one mostly direct light, which the occlusion does not shadow). This is an approximation,
cheap and noise-free.

With the AO on, the render writes a planar "depth" output if the outputs hold none
(`RenderSettings.written_outputs`); the strength `lighting.contact_ao` scales the occlusion (0:
off: the display kernel variant without this code, no proxies gathered).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.output_layout import OUT_DEPTH
from raywarp.core.kernels.abi.rays import frame_scale
from raywarp.core.kernels.abi.structs import OutputArrays, f16s, f32s
from raywarp.core.kernels.abi.tables import instance_rgba, instance_shown
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.post.grading import color

if TYPE_CHECKING:
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.sharing import Layout

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# A proxy occludes out to REACH times its bounding radius (its form factor is below ~1 / REACH^2
# there), faded to 0 over that distance.
REACH: Final = 3.0
# A point within INSIDE times a proxy's extents is on a dynamic object (the depth's float
# rounding, a mesh's surface inside its box): it is not occluded.
INSIDE: Final = 1.02
# A round cross-section's box: the square of the disc's area (half side r sqrt(pi) / 2), so a
# cylinder's or capsule's proxy covers as much as the shape, not its bounding square.
ROUND: Final = math.sqrt(math.pi) / 2.0
# The exposed luminance up to which a pixel counts as ambient light (display-relative: middle
# grey 0.18 and a little above); a brighter pixel's darkening is scaled down by its share below it.
AMBIENT_LEVEL: Final = 0.25
# Proxy kinds: the occluder's form factor is a box's but for spheres, and a point is on the object
# within its true shape (`_inside`): a box, a sphere, a round prism along z (capsules, cylinders:
# radius half / ROUND) or an ellipsoid (semi-axes half / ROUND).
KIND_BOX: Final = 0
KIND_SPHERE: Final = 1
KIND_ROUND: Final = 2
KIND_ELLIPSOID: Final = 3
# The shapes that get a proxy (planes, height fields, deformables do not).
_PROXY_SHAPES: Final = (
  ShapeType.SPHERE,
  ShapeType.CAPSULE,
  ShapeType.ELLIPSOID,
  ShapeType.CYLINDER,
  ShapeType.BOX,
  ShapeType.MESH,
  ShapeType.SDF,
)
_INV_TWO_PI: Final = 1.0 / (2.0 * math.pi)
# Screen tiles [px] and the slots of a tile's bin (a count and BIN - 1 proxy indices): a pixel tests
# only the proxies of its tile (`bin_kernel`), not every one.
TILE: Final = 16
BIN: Final = 16
# Proxies per group (`Proxies.groups`): a pixel tests a group's sphere, then its proxies.
GROUP: Final = 8
# A proxy whose reach comes this close to the camera plane [view-space depth, m] covers every tile.
_NEAR_PLANE: Final = 1.0e-4


@wp.struct
class Proxies:
  """The proxies of a frame in each camera's view space, (nworld, ncam x count) each: the centre,
  the box axes (rows: unit axes, view space; the local point is axes @ (p - centre)), the half
  extents (a sphere: its radius in x), the kind; `count` proxies per camera. The proxies form
  `ngroup` groups of GROUP consecutive ones (one body's parts, mostly: instances are listed body by
  body); `groups` (nworld, ncam x ngroup) holds each group's bounding sphere of its proxies' reach
  (centre, radius). `bins` (nworld, ncam x max_tiles x BIN) lists per TILE x TILE pixel tile the
  groups whose sphere its pixels may fall in: a count, then their indices (a count of -1: more than
  BIN - 1, every group is tested); `tiles_x` (ncam,) the tiles per image row of each camera."""

  centre: wp.array2d[wp.vec3]
  axes: wp.array2d[wp.mat33]
  half: wp.array2d[wp.vec3]
  kind: wp.array[int]
  count: int
  groups: wp.array2d[wp.vec4]
  ngroup: int
  bins: wp.array2d[int]
  tiles_x: wp.array[int]
  max_tiles: int


@wp.func
def _local_box(
  # In:
  gtype: int,
  size: wp.vec3,
  mesh: int,
  mesh_centre: wp.array[wp.vec3],
  mesh_half: wp.array[wp.vec3],
) -> tuple[wp.vec3, wp.vec3]:
  """(centre, half extents) of an instance's proxy box in its own frame (`bounds.world_aabb`'s
  shapes): a mesh's box, a primitive's extents (capsules and cylinders along local z; round
  cross-sections by their area-equal square, `ROUND`)."""
  if gtype == ShapeType.MESH or gtype == ShapeType.SDF:
    if mesh < 0:
      return wp.vec3(0.0), wp.vec3(0.0)
    return mesh_centre[mesh], mesh_half[mesh]
  r = size[0] * wp.static(ROUND)
  if gtype == ShapeType.SPHERE:
    return wp.vec3(0.0), wp.vec3(size[0], size[0], size[0])
  if gtype == ShapeType.CAPSULE:
    return wp.vec3(0.0), wp.vec3(r, r, size[1] + size[0])
  if gtype == ShapeType.CYLINDER:
    return wp.vec3(0.0), wp.vec3(r, r, size[1])
  if gtype == ShapeType.ELLIPSOID:
    return wp.vec3(0.0), size * wp.static(ROUND)
  return wp.vec3(0.0), size  # box


@wp.kernel
def gather_kernel(
  # Model:
  color_rule: int,
  instance_type: wp.array[int],
  instance_mesh: wp.array2d[int],
  instance_size: wp.array2d[wp.vec3],
  instance_material: wp.array2d[int],
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  visible_slot: wp.array[int],
  mesh_centre: wp.array[wp.vec3],
  mesh_half: wp.array[wp.vec3],
  # Data in:
  instance_visible: wp.array2d[int],
  instance_pos: wp.array2d[wp.vec3],
  instance_rot: wp.array2d[wp.mat33],
  camera_pos: wp.array2d[wp.vec3],
  camera_rot: wp.array2d[wp.mat33],
  # In:
  use_visible: int,
  ids: wp.array[int],
  columns: wp.array[int],
  count: int,
  # Out:
  centre_out: wp.array2d[wp.vec3],
  axes_out: wp.array2d[wp.mat33],
  half_out: wp.array2d[wp.vec3],
) -> None:
  """Proxy k of world w in camera c's view space (camera frame: x right, y up, -z forward); a
  hidden instance (alpha 0, or hidden in its world: `instance_shown`) gets no extent."""
  w, c, k = wp.tid()
  gi = ids[k]
  col = columns[k]
  pos = instance_pos[w, col]
  frame = instance_rot[w, col]  # a mesh instance's: s R (its box scaled by s)
  scale = frame_scale(frame)
  rot = frame * (1.0 / scale)
  mesh = instance_mesh[w % instance_mesh.shape[0], gi]
  if mesh >= mesh_half.shape[0]:
    mesh = -1
  local, half = _local_box(instance_type[gi], instance_size[w % instance_size.shape[0], gi], mesh, mesh_centre, mesh_half)
  rgba, _ = instance_rgba(color_rule, instance_material, instance_color, material_base_color, w, gi)
  if not instance_shown(visible_slot, instance_visible, rgba, use_visible != 0, w, gi):
    half = wp.vec3(0.0)
  view = wp.transpose(camera_rot[w, c])
  i = c * count + k
  centre_out[w, i] = view @ (pos + frame @ local - camera_pos[w, c])
  # Rows: the instance's axes in view space.
  axes_out[w, i] = wp.transpose(view @ rot)
  half_out[w, i] = half * scale


@wp.func
def _screen_rect(
  # In:
  centre: wp.vec3,
  reach: float,
  pixel: float,
  width: int,
  height: int,
) -> wp.vec4:
  """(x0, y0, x1, y1) [px] bounding the projection of the cube of half side `reach` about view-space
  `centre` (its 8 corners; one at or behind the camera plane: the whole image)."""
  lo = wp.vec2(1.0e30, 1.0e30)
  hi = wp.vec2(-1.0e30, -1.0e30)
  for k in range(8):
    corner = centre + wp.vec3(
      wp.where((k & 1) != 0, reach, -reach), wp.where((k & 2) != 0, reach, -reach), wp.where((k & 4) != 0, reach, -reach)
    )
    depth = -corner[2]
    if depth <= wp.static(_NEAR_PLANE):
      return wp.vec4(0.0, 0.0, float(width), float(height))
    sx = corner[0] / (depth * pixel) + 0.5 * float(width)
    sy = 0.5 * float(height) - corner[1] / (depth * pixel)
    lo = wp.vec2(wp.min(lo[0], sx), wp.min(lo[1], sy))
    hi = wp.vec2(wp.max(hi[0], sx), wp.max(hi[1], sy))
  return wp.vec4(lo[0], lo[1], hi[0], hi[1])


@wp.kernel
def group_kernel(
  # In:
  proxies: Proxies,
  # Out:
  groups: wp.array2d[wp.vec4],
) -> None:
  """Group g of camera c of world w: the bounding sphere of its proxies' reach (REACH x their
  bounding radius; radius 0: no visible proxy)."""
  w, c, g = wp.tid()
  first = c * proxies.count
  end = wp.min((g + 1) * wp.static(GROUP), proxies.count)
  centre = wp.vec3(0.0)
  n = float(0.0)
  for k in range(g * wp.static(GROUP), end):
    if wp.length_sq(proxies.half[w, first + k]) > 0.0:
      centre += proxies.centre[w, first + k]
      n += 1.0
  radius = float(0.0)
  if n > 0.0:
    centre = centre / n
    for k in range(g * wp.static(GROUP), end):
      half = proxies.half[w, first + k]
      if wp.length_sq(half) > 0.0:
        reach = wp.static(REACH) * wp.length(half)
        radius = wp.max(radius, wp.length(proxies.centre[w, first + k] - centre) + reach)
  groups[w, c * proxies.ngroup + g] = wp.vec4(centre[0], centre[1], centre[2], radius)


@wp.kernel
def bin_kernel(
  # Data in:
  fovy: wp.array2d[float],
  # In:
  sizes: wp.array[wp.vec2i],
  proxies: Proxies,
  # Out:
  bins: wp.array2d[int],
) -> None:
  """Tile t of camera c of world w: the groups whose sphere projects onto it (`Proxies.bins`)."""
  w, c, t = wp.tid()
  width, height = sizes[c][0], sizes[c][1]
  tx = proxies.tiles_x[c]
  ty = (height + wp.static(TILE) - 1) // wp.static(TILE)
  if t >= tx * ty:
    return
  x0 = float((t % tx) * wp.static(TILE))
  y0 = float((t // tx) * wp.static(TILE))
  pixel = 2.0 * wp.tan(0.5 * fovy[w % fovy.shape[0], c] * wp.static(math.pi / 180.0)) / float(height)
  base = (c * proxies.max_tiles + t) * wp.static(BIN)
  n = int(0)
  for g in range(proxies.ngroup):
    sphere = proxies.groups[w, c * proxies.ngroup + g]
    if sphere[3] > 0.0:
      r = _screen_rect(wp.vec3(sphere[0], sphere[1], sphere[2]), sphere[3], pixel, width, height)
      if r[0] < x0 + wp.static(float(TILE)) and r[2] >= x0 and r[1] < y0 + wp.static(float(TILE)) and r[3] >= y0:
        if n < wp.static(BIN - 1):
          bins[w, base + 1 + n] = g
        n += 1
  bins[w, base] = wp.where(n < wp.static(BIN), n, -1)


@final
class ProxySet:
  """The proxies of a Renderer's per-world instances (module doc) and their per-frame buffers.

  Attributes:
    count: the number of proxies (0: the AO darkens nothing).
  """

  def __init__(self, scene: GpuScene, layout: Layout, nworld: int, sizes: list[tuple[int, int]]) -> None:
    """The proxies of `scene`'s rendered instances of a proxied shape whose geometry changes per
    world: moved by the simulator or declared per world (pose, size, mesh). Instances per world
    for another reason (a per-world colour or optics table) are static scenery: the probes see
    them."""
    types = host_copy(scene.instance_type).astype(np.int64)
    rendered = np.zeros(len(types), bool)
    rendered[np.asarray(scene.rendered, np.int64)] = True
    moving = np.asarray(layout.dynamic, bool).copy()
    for group in ("pose", "size", "mesh"):
      moving |= np.asarray(layout.groups[group], bool)
    chosen = rendered & moving & np.isin(types, [int(s) for s in _PROXY_SHAPES])
    ids = np.flatnonzero(chosen).astype(np.int32)
    self.count = len(ids)
    device = scene.device
    n = max(self.count, 1)
    pad = np.zeros(1, np.int32)
    self._ids = wp.array(ids if self.count else pad, dtype=int, device=device)
    self._columns = wp.array(np.asarray(layout.pose_slot, np.int32)[ids] if self.count else pad, dtype=int, device=device)
    self._visible_slot = wp.array(np.asarray(layout.visible_slot, np.int32), dtype=int, device=device)
    self._use_visible = int(layout.nvisible > 0)
    self._scene = scene
    ncam = self._ncam = len(sizes)
    tiles = [(-(-w // TILE), -(-h // TILE)) for w, h in sizes]
    max_tiles = max(tx * ty for tx, ty in tiles)
    self._sizes = wp.array([wp.vec2i(w, h) for w, h in sizes], dtype=wp.vec2i, device=device)
    # Fetched here, not per frame: their first use uploads them (never inside a graph capture).
    self._mesh_boxes = scene.meshes.local_boxes()
    self.proxies = Proxies()
    self.proxies.centre = wp.zeros((nworld, ncam * n), dtype=wp.vec3, device=device)
    self.proxies.axes = wp.zeros((nworld, ncam * n), dtype=wp.mat33, device=device)
    self.proxies.half = wp.zeros((nworld, ncam * n), dtype=wp.vec3, device=device)
    kind = np.tile(_kinds(types[ids]) if self.count else pad, ncam)
    self.proxies.kind = wp.array(kind, dtype=int, device=device)
    self.proxies.count = self.count
    self.proxies.ngroup = ngroup = -(-n // GROUP)
    self.proxies.groups = wp.zeros((nworld, ncam * ngroup), dtype=wp.vec4, device=device)
    self.proxies.bins = wp.zeros((nworld, ncam * max_tiles * BIN), dtype=int, device=device)
    self.proxies.tiles_x = wp.array([tx for tx, _ in tiles], dtype=int, device=device)
    self.proxies.max_tiles = max_tiles

  def gather(self, frame: Frame, camera_pos: wp.array2d[wp.vec3], camera_rot: wp.array2d[wp.mat33]) -> Proxies:
    """The frame's proxies in each camera's view space, their groups' spheres and the screen
    tiles' bins (three launches; none without proxies)."""
    if self.count:
      p, scene = self.proxies, self._scene
      wp.launch(
        gather_kernel,
        dim=(p.centre.shape[0], self._ncam, self.count),
        inputs=[
          int(scene.color_rule),
          scene.instance_type,
          frame.instance_mesh,
          frame.instance_size,
          frame.instance_material,
          frame.instance_color,
          frame.material_base_color,
          self._visible_slot,
          *self._mesh_boxes,
          frame.instance_visible,
          frame.instance_pos,
          frame.instance_rot,
          camera_pos,
          camera_rot,
          self._use_visible,
          self._ids,
          self._columns,
          self.count,
        ],
        outputs=[p.centre, p.axes, p.half],
        device=scene.device,
      )
      wp.launch(
        group_kernel, dim=(p.centre.shape[0], self._ncam, p.ngroup), inputs=[p], outputs=[p.groups], device=scene.device
      )
      wp.launch(
        bin_kernel,
        dim=(p.centre.shape[0], self._ncam, p.max_tiles),
        inputs=[frame.camera_fovy, self._sizes, p],
        outputs=[p.bins],
        device=scene.device,
      )
    return self.proxies


def _kinds(types: np.ndarray) -> np.ndarray:
  """(n,) int32 proxy kinds of instances of ShapeTypes `types`."""
  kind = np.full(len(types), KIND_BOX, np.int32)
  kind[types == int(ShapeType.SPHERE)] = KIND_SPHERE
  kind[(types == int(ShapeType.CAPSULE)) | (types == int(ShapeType.CYLINDER))] = KIND_ROUND
  kind[types == int(ShapeType.ELLIPSOID)] = KIND_ELLIPSOID
  return kind


@wp.func
def depth_at(
  # In:
  out: OutputArrays,
  world: int,
  adr: int,
) -> float:
  """The depth output at pixel `adr` [m] (float32 or float16; radial depth read as planar, close
  enough at a pixel's reach); 0 for a miss (zero, inf or beyond float16)."""
  z = float(0.0)
  if out_half(out.half, wp.static(OUT_DEPTH)):
    z = float(f16s(out, out.table.depth16, 1)[world, adr])
  else:
    z = f32s(out, out.table.depth)[world, adr]
  if not (z > 0.0 and z < 1.0e30):
    return 0.0
  return z


@wp.func
def view_point(
  # In:
  x: float,
  y: float,
  z: float,
  width: int,
  height: int,
  pixel: float,
) -> wp.vec3:
  """The view-space point of pixel (x, y) (centres at + 0.5) at planar depth z, `pixel` the
  image plane's pixel size at unit depth (2 tan(fovy / 2) / height)."""
  return wp.vec3((x - 0.5 * float(width)) * pixel * z, (0.5 * float(height) - y) * pixel * z, -z)


@wp.func
def _point(
  # In:
  out: OutputArrays,
  world: int,
  base: int,
  x: int,
  y: int,
  width: int,
  height: int,
  pixel: float,
) -> wp.vec3:
  """The view-space point of camera pixel (x, y) (z 0: a miss, or outside the image: the normal
  then takes the other side's step)."""
  if x < 0 or x >= width or y < 0 or y >= height:
    return wp.vec3(0.0)
  z = depth_at(out, world, base + y * width + x)
  return view_point(float(x) + 0.5, float(y) + 0.5, z, width, height, pixel)


@wp.func
def _step(
  # In:
  p: wp.vec3,
  a: wp.vec3,
  b: wp.vec3,
) -> wp.vec3:
  """The smaller of the steps p -> a and b -> p by depth (the side that crosses no silhouette;
  a miss's side, z 0, never)."""
  da = wp.where(a[2] == 0.0, 1.0e30, wp.abs(a[2] - p[2]))
  db = wp.where(b[2] == 0.0, 1.0e30, wp.abs(p[2] - b[2]))
  return wp.where(da <= db, a - p, p - b)


@wp.func
def _edge(
  # In:
  n: wp.vec3,
  a: wp.vec3,
  b: wp.vec3,
) -> float:
  """One silhouette edge's term of the projected solid angle: n . normalize(a x b) x angle(a, b)
  (a, b unit directions)."""
  cr = wp.cross(a, b)
  len_cr = wp.length(cr)
  if len_cr <= 1.0e-12:
    return 0.0
  return wp.dot(n, cr) / len_cr * wp.acos(wp.clamp(wp.dot(a, b), -1.0, 1.0))


@wp.func
def box_occlusion(
  # In:
  p: wp.vec3,
  n: wp.vec3,
  half: wp.vec3,
) -> float:
  """The form factor of the box of `half` extents about the origin seen from point p (outside it)
  with normal n, both in the box frame: its silhouette hexagon's projected solid angle / pi
  (Quilez, "box occlusion"; no clipping at the horizon)."""
  f = wp.vec3(half[0] * wp.sign(p[0]), half[1] * wp.sign(p[1]), half[2] * wp.sign(p[2]))
  s = wp.vec3(wp.sign(half[0] - wp.abs(p[0])), wp.sign(half[1] - wp.abs(p[1])), wp.sign(half[2] - wp.abs(p[2])))
  v0 = wp.normalize(wp.cw_mul(wp.vec3(1.0, 1.0, -1.0), f) - p)
  v1 = wp.normalize(wp.cw_mul(wp.vec3(1.0, s[0], s[0]), f) - p)
  v2 = wp.normalize(wp.cw_mul(wp.vec3(1.0, -1.0, 1.0), f) - p)
  v3 = wp.normalize(wp.cw_mul(wp.vec3(s[2], s[2], 1.0), f) - p)
  v4 = wp.normalize(wp.cw_mul(wp.vec3(-1.0, 1.0, 1.0), f) - p)
  v5 = wp.normalize(wp.cw_mul(wp.vec3(s[1], 1.0, s[1]), f) - p)
  total = _edge(n, v0, v1) + _edge(n, v1, v2) + _edge(n, v2, v3) + _edge(n, v3, v4) + _edge(n, v4, v5) + _edge(n, v5, v0)
  return wp.abs(total) * wp.static(_INV_TWO_PI)


@wp.func
def sphere_occlusion(
  # In:
  d: wp.vec3,
  n: wp.vec3,
  radius: float,
) -> float:
  """The form factor of a sphere of `radius` at offset d from a point with normal n, the sphere
  above the tangent plane: cos(theta) (r / |d|)^2 (Quilez, "sphere ambient occlusion")."""
  l2 = wp.dot(d, d)
  return wp.max(wp.dot(n, d), 0.0) * radius * radius / (l2 * wp.sqrt(l2))


@wp.func
def _inside(
  # In:
  kind: int,
  local: wp.vec3,
  half: wp.vec3,
) -> bool:
  """Whether `local` (in the proxy's frame) lies within INSIDE times the proxy's true shape (the
  kinds above): on the dynamic object itself."""
  a = wp.vec3(wp.abs(local[0]), wp.abs(local[1]), wp.abs(local[2]))
  if kind == wp.static(KIND_SPHERE):
    return wp.length_sq(local) <= wp.static(INSIDE * INSIDE) * half[0] * half[0]
  if kind == wp.static(KIND_ELLIPSOID):
    e = wp.cw_div(local, half * wp.static(1.0 / ROUND))
    return wp.length_sq(e) <= wp.static(INSIDE * INSIDE)
  if kind == wp.static(KIND_ROUND):
    r = half[0] * wp.static(INSIDE / ROUND)
    return local[0] * local[0] + local[1] * local[1] <= r * r and a[2] <= half[2] * wp.static(INSIDE)
  return a[0] <= half[0] * wp.static(INSIDE) and a[1] <= half[1] * wp.static(INSIDE) and a[2] <= half[2] * wp.static(INSIDE)


@wp.func
def proxy_visibility(
  # In:
  proxies: Proxies,
  world: int,
  cam: int,
  x: int,
  y: int,
  p: wp.vec3,
  n: wp.vec3,
) -> float:
  """The visible fraction (1: open) of view-space point p with normal n of camera pixel (x, y)
  under the proxies of its tile's bin (module doc); 1 for a point inside a proxy (on a dynamic
  object)."""
  visible = float(1.0)
  first = cam * proxies.count
  tile = (y // wp.static(TILE)) * proxies.tiles_x[cam] + x // wp.static(TILE)
  base = (cam * proxies.max_tiles + tile) * wp.static(BIN)
  listed = proxies.bins[world, base]
  total = wp.where(listed < 0, proxies.ngroup, listed)
  for j in range(total):
    g = j
    if listed >= 0:
      g = proxies.bins[world, base + 1 + j]
    sphere = proxies.groups[world, cam * proxies.ngroup + g]
    if wp.length_sq(wp.vec3(sphere[0], sphere[1], sphere[2]) - p) >= sphere[3] * sphere[3]:
      continue
    for k in range(g * wp.static(GROUP), wp.min((g + 1) * wp.static(GROUP), proxies.count)):
      i = first + k
      half = proxies.half[world, i]
      d = proxies.centre[world, i] - p
      r2 = wp.dot(half, half)
      l2 = wp.dot(d, d)
      reach2 = wp.static(REACH * REACH) * r2
      if l2 >= reach2 or r2 <= 0.0:
        continue
      axes = proxies.axes[world, i]
      local = -(axes @ d)  # p in the proxy's frame
      kind = proxies.kind[i]
      if _inside(kind, local, half):
        return 1.0
      # The proxy's support along n: entirely below the tangent plane, it covers nothing.
      an = axes @ n
      if wp.dot(n, d) + wp.abs(an[0]) * half[0] + wp.abs(an[1]) * half[1] + wp.abs(an[2]) * half[2] <= 0.0:
        continue
      occ = float(0.0)
      if kind == wp.static(KIND_SPHERE):
        occ = sphere_occlusion(d, n, half[0])
      else:
        occ = box_occlusion(local, an, half)
      visible *= 1.0 - wp.clamp(occ, 0.0, 1.0) * (1.0 - l2 / reach2)
  return visible


@wp.func
def surface(
  # In:
  out: OutputArrays,
  world: int,
  base: int,
  x: int,
  y: int,
  width: int,
  height: int,
  pixel: float,
) -> tuple[wp.vec3, wp.vec3, bool]:
  """(view-space point, normal facing the camera, found) of camera pixel (x, y) from the depth
  output (`base`: the camera's first pixel index; not found: a miss or no usable normal)."""
  p = _point(out, world, base, x, y, width, height, pixel)
  if p[2] >= 0.0:
    return p, wp.vec3(0.0), False
  dx = _step(
    p, _point(out, world, base, x + 1, y, width, height, pixel), _point(out, world, base, x - 1, y, width, height, pixel)
  )
  dy = _step(
    p, _point(out, world, base, x, y - 1, width, height, pixel), _point(out, world, base, x, y + 1, width, height, pixel)
  )
  n = wp.cross(dx, dy)
  if wp.length_sq(n) <= 0.0:
    return p, wp.vec3(0.0), False
  n = wp.normalize(n)
  if wp.dot(n, p) > 0.0:  # facing the camera
    n = -n
  return p, n, True


@wp.func
def apply(
  # In:
  c: wp.vec3,
  out: OutputArrays,
  proxies: Proxies,
  world: int,
  cam: int,
  base: int,
  x: int,
  y: int,
  width: int,
  height: int,
  fovy: float,
  strength: float,
) -> wp.vec3:
  """Exposed color `c` of camera pixel (x, y) with its ambient share occluded (module doc); as it
  is without the depth output (`out.bits`), a strength or proxies."""
  if strength <= 0.0 or proxies.count == 0 or not out_on(out.bits, wp.static(OUT_DEPTH)):
    return c
  pixel = 2.0 * wp.tan(0.5 * fovy * wp.static(math.pi / 180.0)) / float(height)
  p, n, found = surface(out, world, base, x, y, width, height, pixel)
  if not found:
    return c
  visible = proxy_visibility(proxies, world, cam, x, y, p, n)
  lum = color.luminance(c)
  share = wp.where(lum > wp.static(AMBIENT_LEVEL), wp.static(AMBIENT_LEVEL) / wp.max(lum, 1.0e-30), 1.0)
  return c * (1.0 - strength * share * (1.0 - visible))
