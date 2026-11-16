"""The megakernel's per-pixel body with the motion output (`Features.history`,
`make_render_pixel`): it keeps what the motion output needs from one frame to the next.

The first-hit outputs and depth / segmentation from the pixel center, the first hit's motion,
motion status and depth for the next frame (`state.make_write_history`), and the color
(`store.make_store`). The pass around the shading call: `begin` (the camera pixel of a thread's
ray, `locate`; the pass's sample of it, `make_prepare`: seed, sub-pixel cell, cone scale; its ray,
cone, headlight and exposure, `make_aim`), the sample's shading (`make_trace_sample`:
`pixel.make_trace_pixel`) and `end` (the pass's writes, `make_end`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import warp as wp

from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.output_layout import OUT_FIRST
from raywarp.core.kernels.abi.rays import BodyClosest, GeometryFields, Hit
from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.kernels.first_hit import FirstHitWriters, SurfaceRecord, zero_record
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.post.temporal.state import Temporal, WriteHistory, make_write_history
from raywarp.core.post.temporal.store import SampleRayOf, TracePixel, make_more_samples, make_store
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import MaterialTables
from raywarp.core.trace.rays import camera, primary, seeds

if TYPE_CHECKING:
  from raywarp.core.kernels.center import CenterTrace

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class SampleRay:
  """One camera-ray sample of a pixel pass: what `trace_sample` traces and the pass's `end` writes.

  Attributes:
    valid: the sample is traced (False: nothing to shade; `prepare` wrote what the pixel needs).
    pix: the pixel.
    adr: its camera-pixel index over every camera (`Temporal` buffers).
    subdiv, cell: the sample's sub-pixel cell (sx, sy) of a subdiv x subdiv grid (`prepare`).
    cone: the sample's ray cone over the pixel's (`prepare`).
    origin, direction, max_dist, depth_axis: the camera ray (`primary.make_camera_rays`,
    `make_aim`). width, spread: its ray cone.
    headlight_dir: the camera's headlight direction.
    pixel_seed, stream: the random streams of its shading (`pixel.make_trace_pixel`).
    exposure: the pixel's exposure (world x camera).
  """

  valid: bool
  pix: camera.Pixel
  adr: int
  subdiv: int
  cell: wp.vec2i
  cone: float
  origin: wp.vec3
  direction: wp.vec3
  max_dist: float
  depth_axis: wp.vec3
  width: float
  spread: float
  headlight_dir: wp.vec3
  pixel_seed: int
  stream: int
  exposure: float


@wp.struct
class CentreShade:
  """The pixel centre's own shading in a pass whose sample is off it (`make_centre`: the outputs
  only the shading gives, `center.shading_outputs`): whether it was traced, its first hit, base
  color and record."""

  traced: bool
  hit: Hit
  albedo: wp.vec3
  rec: SurfaceRecord


@wp.func
def centre_values(
  # In:
  c: CentreShade,
) -> tuple[Hit, wp.vec3, SurfaceRecord]:
  """(first hit, base color, record) of the centre's shading `c` (copies)."""
  return c.hit, c.albedo, c.rec


# trace_sample(geo, mats, lit, env, worldid, s) -> (radiance, first hit, base color, record).
type TraceSample = Callable[
  [GeometryFields, MaterialTables, Lights, EnvironmentArrays, int, SampleRay], tuple[wp.vec3, Hit, wp.vec3, SurfaceRecord]
]
# aim(cams, env, worldid, s) -> `s` with its camera ray, cone, headlight and exposure.
type Aim = Callable[[Cameras, EnvironmentArrays, int, SampleRay], SampleRay]
# begin(geo, cams, env, seed, out, tmp, worldid, rayid) -> the thread's sample.
type Begin = Callable[[GeometryFields, Cameras, EnvironmentArrays, int, OutputArrays, Temporal, int, int], SampleRay]
# end(geo, mats, lit, cams, env, out, tmp, worldid, s, radiance, hit, albedo, record, centre).
type End = Callable[
  [
    GeometryFields,
    MaterialTables,
    Lights,
    Cameras,
    EnvironmentArrays,
    OutputArrays,
    Temporal,
    int,
    SampleRay,
    wp.vec3,
    Hit,
    wp.vec3,
    SurfaceRecord,
    CentreShade,
  ],
  None,
]
# centre(cams, s) -> `s` aimed at its pixel's centre (`make_centre`).
type Centre = Callable[[Cameras, SampleRay], SampleRay]


# locate(cams, rayid) -> (camid, px, py).
type RenderLocate = Callable[[Cameras, int], wp.vec3i]
# prepare(geo, cams, seed, out, tmp, worldid, rayid, pix) -> the first sample.
type RenderPrepare = Callable[[GeometryFields, Cameras, int, OutputArrays, Temporal, int, int, camera.Pixel], SampleRay]

# render_pixel(geo, mats, lit, cams, env, seed, out, tmp, worldid, rayid): the temporal body.
type RenderPixel = Callable[
  [GeometryFields, MaterialTables, Lights, Cameras, EnvironmentArrays, int, OutputArrays, Temporal, int, int], None
]
# geometry_pixel(geo, cams, out, tmp, pix, worldid, adr): a camera without RGB.
type GeometryPixel = Callable[[GeometryFields, Cameras, OutputArrays, Temporal, camera.Pixel, int, int], None]
# geometry_ray(cams, pix) -> the ray of a camera without RGB (the pixel centre).
type GeometryRay = Callable[[Cameras, camera.Pixel], tuple[wp.vec3, wp.vec3, float, wp.vec3]]


@specialize
def make_render_pixel(  # noqa: PLR0913  # the pass's inputs, built by `pixel.make_render_pixel`
  f: Features,
  model: primary.CameraModel,
  *,
  trace_closest: BodyClosest,
  trace_pixel: TracePixel,
  primary_ray: primary.PrimaryRay,
  pixel_cone: primary.PixelCone,
  writes: FirstHitWriters,
  center_trace: CenterTrace,
  shading: int,
) -> RenderPixel:
  """The temporal per-pixel body of camera model `model` (see `pixel.make_render_pixel`): `begin`
  (`_make_begin`), the sample's shading (`make_trace_sample`), `end` (`make_end`; the sample rays,
  first-hit writers and history writer shared with the geometry-only cameras, `make_prepare`).
  With `shading` (`center.shading_outputs` of a pass whose sample is off the pixel centre), the
  outputs the word asks for among them shade the centre too (`make_centre`): a loop of at most
  two rays around the one shading call site.

  Returns render_pixel(geo, mats, lit, cams, env, seed, out, tmp, worldid, rayid).
  """
  geometry_ray = make_sample_rays(f, primary_ray)[1]
  write_history = make_write_history(f, model)
  geometry_pixel = make_geometry_pixel(trace_closest, geometry_ray=geometry_ray, writes=writes, write_history=write_history)
  aim = make_aim(primary_ray, pixel_cone)
  begin = _make_begin(cast(RenderLocate, locate), make_prepare(f, geometry_pixel), aim, any_rgb=f.any_rgb)
  end = make_end(f, model, trace_pixel=trace_pixel, primary_ray=primary_ray, writes=writes, center_trace=center_trace)
  trace_sample = make_trace_sample(trace_pixel)
  centre = make_centre(primary_ray, pixel_cone)

  @wp.func
  @device_generic
  def render_pixel[C: Cameras, G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    cams: C,
    env: kernel_structs.EnvironmentArrays,
    # In:
    seed: int,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    worldid: int,
    rayid: int,
  ) -> None:
    """Renders ray `rayid` of world `worldid` with the frame's motion and history."""
    s = begin(geo, cams, env, seed, out, tmp, worldid, rayid)
    if s.valid:
      c = CentreShade()
      c.traced = False
      if wp.static(shading == 0):
        radiance, hit, albedo, rec = trace_sample(geo, mats, lit, env, worldid, s)
      else:
        radiance = wp.vec3(0.0)
        hit = trace.Hit()
        albedo = wp.vec3(0.0)
        rec = zero_record()
        rays_n = int(1)
        if out_on(out.bits, wp.static(shading)):
          rays_n = 2
        for i in range(rays_n):
          r = s
          if i == 1:
            r = centre(cams, s)
          ri, hi, ai, reci = trace_sample(geo, mats, lit, env, worldid, r)
          if i == 0:
            radiance = ri
            hit = hi
            albedo = ai
            rec = reci
          else:
            c.traced = True
            c.hit = hi
            c.albedo = ai
            c.rec = reci
      end(geo, mats, lit, cams, env, out, tmp, worldid, s, radiance, hit, albedo, rec, c)

  return render_pixel


@specialize
def make_trace_sample(trace_pixel: TracePixel) -> TraceSample:
  """trace_sample(geo, mats, lit, env, worldid, s) -> (radiance, first hit, base color, record) of
  sample `s` (`pixel.make_trace_pixel`)."""

  @wp.func(inline=True)
  @device_generic
  def trace_sample[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: kernel_structs.EnvironmentArrays,
    # In:
    worldid: int,
    s: SampleRay,
  ) -> tuple[wp.vec3, trace.Hit, wp.vec3, SurfaceRecord]:
    """The shading of the sample's camera ray."""
    return trace_pixel(
      geo,
      mats,
      lit,
      env,
      env.params,
      worldid,
      s.origin,
      s.direction,
      s.max_dist,
      s.width,
      s.spread,
      s.headlight_dir,
      s.exposure,
      s.pixel_seed,
      s.stream,
    )

  return trace_sample


@specialize
def make_centre(primary_ray: primary.PrimaryRay, pixel_cone: primary.PixelCone) -> Centre:
  """centre(cams, s) -> sample `s` aimed at its pixel's centre with the pixel's cone
  (stream 0): the ray of the outputs only the shading gives (`CentreShade`)."""

  @wp.func(inline=True)
  @device_generic
  def centre[C: Cameras](
    # Model:
    cams: C,
    # In:
    s: SampleRay,
  ) -> SampleRay:
    """`s` through the pixel centre."""
    r = s
    origin, direction, max_dist, depth_axis = primary_ray(cams, s.pix, 1, 0, 0)
    spread, width = pixel_cone(cams, s.pix)
    r.origin = origin
    r.direction = direction
    r.max_dist = max_dist
    r.depth_axis = depth_axis
    r.width = width
    r.spread = spread
    r.stream = 0
    return r

  return centre


@specialize
def make_aim(primary_ray: primary.PrimaryRay, pixel_cone: primary.PixelCone) -> Aim:
  """aim(cams, env, worldid, s) -> `s` with its camera ray, cone, headlight and exposure."""

  @wp.func(inline=True)
  @device_generic
  def aim[C: Cameras](
    # Model:
    cams: C,
    env: kernel_structs.EnvironmentArrays,
    # In:
    worldid: int,
    s: SampleRay,
  ) -> SampleRay:
    """The ray through sub-pixel cell `s.cell` of `s.subdiv`, its cone the pixel's x `s.cone`."""
    pix = s.pix
    pixel_spread, pixel_width = pixel_cone(cams, pix)
    origin, direction, max_dist, depth_axis = primary_ray(cams, pix, s.subdiv, s.cell[0], s.cell[1])
    r = s
    r.width = pixel_width * s.cone
    r.spread = pixel_spread * s.cone
    r.headlight_dir = camera.headlight_dir(pix)
    r.exposure = exposure_at(env.exposure, worldid) * pix.lens.exposure
    r.origin = origin
    r.direction = direction
    r.max_dist = max_dist
    r.depth_axis = depth_axis
    return r

  return aim


@specialize
def _make_begin(locate: RenderLocate, prepare: RenderPrepare, aim: Aim, *, any_rgb: bool) -> Begin:
  """begin(geo, cams, env, seed, out, tmp, worldid, rayid) -> `SampleRay` of ray `rayid`: its
  pixel (`locate`, `camera.pixel_at`), the pass's sample of it (`prepare`) aimed (`aim`)."""

  @wp.func(inline=True)
  @device_generic
  def begin[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    env: kernel_structs.EnvironmentArrays,
    # In:
    seed: int,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    worldid: int,
    rayid: int,
  ) -> SampleRay:
    """The thread's sample, set up for `trace_sample` (`valid` False: nothing to trace)."""
    loc = locate(cams, rayid)
    if loc[0] == -1:
      none = SampleRay()
      none.valid = False
      return none
    pix = camera.pixel_at(cams, out, wp.static(any_rgb), worldid, loc[0], loc[1], loc[2])
    s = prepare(geo, cams, seed, out, tmp, worldid, rayid, pix)
    if not s.valid:
      return s
    return aim(cams, env, worldid, s)

  return begin


@wp.func
@device_generic
def locate[C: Cameras](
  # Model:
  cams: C,
  # In:
  rayid: int,
) -> wp.vec3i:
  """(camid, px, py) of ray `rayid` in the frame's order (camid -1: none): the render pass's
  `RenderLocate` (the render launch maps its threads to rendered worlds only,
  `camera.launch_ids`)."""
  return camera.locate_ray(cams, camera.order_rows(cams.order[0]), rayid)


@specialize
def make_prepare(f: Features, geometry_pixel: GeometryPixel) -> RenderPrepare:
  """prepare(geo, cams, seed, out, tmp, worldid, rayid, pix) -> the render pass's
  `SampleRay` of pixel `pix` before `make_aim`: its first sample.

  Not traced (`valid` False): a camera without RGB (its depth / segmentation and history are
  written here, from the pixel centre).
  """
  n_aa = f.spp
  cone = (1.0 / float(n_aa * n_aa)) ** 0.5  # one sample's ray cone over the pixel's

  @wp.func(inline=True)
  @device_generic
  def prepare[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    # In:
    seed: int,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    worldid: int,
    _rayid: int,
    pix: camera.Pixel,
  ) -> SampleRay:
    """The first sample of ray `rayid` of world `worldid`."""
    s = SampleRay()
    s.valid = False
    adr = pix.camid * pix.width * pix.height + pix.pixelid
    if not pix.want_rgb:
      # A camera without RGB: depth / segmentation and history of the pixel centre only.
      geometry_pixel(geo, cams, out, tmp, pix, worldid, adr)
      return s
    s.valid = True
    s.pix = pix
    s.adr = adr
    s.pixel_seed = seeds.pixel_seed(seed, worldid, camera.buffer_index(pix))
    s.stream = 0
    s.cone = wp.static(cone)
    # Sample (0, 0) of the n x n grid (`make_sample_rays`).
    s.subdiv = wp.static(n_aa)
    s.cell = wp.vec2i(0, 0)
    return s

  return prepare


@specialize
def make_geometry_pixel(
  trace_closest: BodyClosest, *, geometry_ray: GeometryRay, writes: FirstHitWriters, write_history: WriteHistory
) -> GeometryPixel:
  """geometry_pixel(geo, cams, out, tmp, pix, worldid, adr): depth / segmentation, the hit
  outputs and history of a camera without RGB (its pixel centre, `geometry_ray`)."""
  write_geometry, write_hit = writes.geometry, writes.hit

  @wp.func
  @device_generic
  def geometry_pixel[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    pix: camera.Pixel,
    worldid: int,
    adr: int,
  ) -> None:
    """Traces the pixel centre and writes its geometry outputs and history."""
    o0, d0, m0, a0 = geometry_ray(cams, pix)
    hit0 = trace_closest(geo, worldid, o0, d0, m0)
    write_geometry(out, worldid, pix, hit0, a0)
    write_hit(geo, out, worldid, camera.buffer_index(pix), hit0, o0, d0)
    write_history(geo, out, tmp, pix, worldid, adr, hit0, o0, d0, a0)

  return geometry_pixel


@specialize
def make_end(
  f: Features,
  model: primary.CameraModel,
  *,
  trace_pixel: TracePixel,
  primary_ray: primary.PrimaryRay,
  writes: FirstHitWriters,
  center_trace: CenterTrace,
) -> End:
  """The render pass's writes after its first sample `s` (`End`; traced: radiance, first
  hit, its base color and `first_hit.SurfaceRecord`).

  The pixel's first-hit outputs, depth / segmentation and history (`write_history`): from this
  sample, or with multisampling from the pixel center (`center.make_trace`);
  then the grid's other samples (`make_more_samples`, sample rays and history writer cached:
  `make_render_pixel`'s own) and the outputs (`make_store`).
  """
  pixel_cone = primary.make_camera_rays(model)[1]
  write_history = make_write_history(f, model)
  more_samples = make_more_samples(f, trace_pixel, make_sample_rays(f, primary_ray)[0])
  store = make_store(f)
  write_geometry, write_hit, write_surface = writes
  # The pixel centre is traced apart from the first sample only when the sample moved off it
  # and something reads the centre's hit: a first-hit output of the word, motion (compiled in),
  # or depth / segmentation (per camera).
  off_center = f.spp > 1

  @wp.func(inline=True)
  @device_generic
  def end[C: Cameras, G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    cams: C,
    env: kernel_structs.EnvironmentArrays,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    worldid: int,
    s: SampleRay,
    radiance: wp.vec3,
    hit: trace.Hit,
    albedo: wp.vec3,
    rec: SurfaceRecord,
    centre: CentreShade,
  ) -> None:
    """The first-hit outputs, geometry and history of the first sample (or of the pixel centre:
    its own shading `centre`, else its hit); the other samples; the outputs."""
    pix = s.pix
    ghit, galbedo, grec = hit, albedo, rec
    go, gd, ga = wp.vec3(s.origin), wp.vec3(s.direction), wp.vec3(s.depth_axis)
    if wp.static(off_center):
      if centre.traced:
        go, gd, _gm, ga = primary_ray(cams, pix, 1, 0, 0)
        ghit, galbedo, grec = centre_values(centre)
      elif wp.static(f.motion) or out_on(out.bits, wp.static(OUT_FIRST)) or pix.want_depth or pix.want_seg:
        go, gd, gm, ga = primary_ray(cams, pix, 1, 0, 0)
        spread, width = pixel_cone(cams, pix)
        ghit, galbedo, grec = center_trace(geo, mats, env, env.params, out, worldid, go, gd, gm, width, spread)
    write_surface(geo, lit, out, worldid, camera.buffer_index(pix), galbedo, grec, s.pixel_seed)
    write_geometry(out, worldid, pix, ghit, ga)
    write_hit(geo, out, worldid, camera.buffer_index(pix), ghit, go, gd)
    write_history(geo, out, tmp, pix, worldid, s.adr, ghit, go, gd, ga)
    hdr, covered = more_samples(
      geo,
      mats,
      lit,
      cams,
      env,
      worldid,
      pix,
      s.width,
      s.spread,
      s.headlight_dir,
      s.exposure,
      s.pixel_seed,
      radiance,
      hit,
    )
    store(out, env, worldid, pix, hdr, covered, s.exposure)

  return end


@specialize
def make_sample_rays(f: Features, primary_ray: primary.PrimaryRay) -> tuple[SampleRayOf, GeometryRay]:
  """(sample_ray, geometry_ray) of the temporal body.

  sample_ray(cams, pix, sx, sy): the primary ray of sample (sx, sy) of the n x n grid;
  geometry_ray(cams, pix): the ray of a camera without RGB, the pixel centre.
  """
  n_aa = f.spp

  @wp.func
  @device_generic
  def sample_ray[C: Cameras](
    # Model:
    cams: C,
    # In:
    pix: camera.Pixel,
    sx: int,
    sy: int,
  ) -> tuple[wp.vec3, wp.vec3, float, wp.vec3]:
    """(origin, direction, max distance, depth axis) of sample (sx, sy)."""
    return primary_ray(cams, pix, wp.static(n_aa), sx, sy)

  @wp.func
  @device_generic
  def geometry_ray[C: Cameras](
    # Model:
    cams: C,
    # In:
    pix: camera.Pixel,
  ) -> tuple[wp.vec3, wp.vec3, float, wp.vec3]:
    """Ray of a camera without RGB: the pixel centre."""
    return primary_ray(cams, pix, 1, 0, 0)

  return sample_ray, geometry_ray
