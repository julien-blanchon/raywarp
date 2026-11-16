"""The geometry-only body: renders that request no shaded output trace one closest hit per pixel.

When only geometry outputs are requested (depth, seg, semantic, position, uv, material, a
geometric normal, motion: `Features.any_rgb` False), the per-pixel body is the pixel's centre ray,
its closest hit (alpha cut-outs included) and the first-hit writes (`first_hit.make_writers`):
no shading, light, environment or texture code is built, and the kernel (or OptiX program) takes
no material, light or environment struct. Motion adds the temporal state (`Temporal`: the
previous poses, the motion buffer). The bodies are the ones the shaded passes use for their
cameras without RGB (`primary.make_geometry_only`, `pixel.make_geometry_pixel`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import warp as wp

from raywarp.core.kernels import first_hit
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.queries import make_queries
from raywarp.core.kernels.abi.rays import GeometryFields, SoftwareGeometry, Tracer, software_geometry_type
from raywarp.core.kernels.abi.structs import Cameras, OutputArrays
from raywarp.core.kernels.generic import concrete, device_generic
from raywarp.core.post.temporal import pixel
from raywarp.core.post.temporal.state import Temporal, make_write_history
from raywarp.core.trace.rays import camera, primary

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# The geometry-only body takes temporal state only for history/motion output.
type Body = Callable[[GeometryFields, Cameras, OutputArrays, int, int], None]
type TemporalBody = Callable[[GeometryFields, Cameras, OutputArrays, Temporal, int, int], None]


def make_geometry_body(f: Features, tracer: Tracer, model: primary.CameraModel):
  """render_geometry(geo, cams, out, worldid, rayid), with motion render_geometry(geo, cams, out,
  tmp, worldid, rayid): the geometry outputs of ray `rayid` (a camera pixel) of world `worldid`
  (the module doc) through camera model `model`'s rays; the ray queries are `tracer`'s (software
  or OptiX)."""
  q = make_queries(f, tracer)
  primary_ray = primary.make_camera_rays(model)[0]
  writes = first_hit.make_writers(f, primary.write_geometry, None)
  locate = cast(pixel.RenderLocate, pixel.locate)
  if f.history:
    geometry_ray = pixel.make_sample_rays(f, primary_ray)[1]
    geometry_pixel = pixel.make_geometry_pixel(
      q.closest, geometry_ray=geometry_ray, writes=writes, write_history=make_write_history(f, model)
    )
    return _make_temporal_body(locate, geometry_pixel)
  geometry_only = primary.make_geometry_only(trace_closest=q.closest, primary_ray=primary_ray, writes=writes)

  @wp.func
  @device_generic
  def render_geometry[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    rayid: int,
  ) -> None:
    """Traces the pixel's centre ray and writes its geometry outputs."""
    loc = locate(cams, rayid)
    if loc[0] == -1:
      return
    pix = camera.pixel_at(cams, out, False, worldid, loc[0], loc[1], loc[2])
    geometry_only(geo, cams, out, pix, worldid, 1, 0)

  return render_geometry


def _make_temporal_body(locate: pixel.RenderLocate, geometry_pixel: pixel.GeometryPixel) -> TemporalBody:
  """The geometry body with motion (the temporal state: previous poses, motion buffer)."""

  @wp.func
  @device_generic
  def render_geometry[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    worldid: int,
    rayid: int,
  ) -> None:
    """Traces the pixel's centre ray and writes its geometry outputs and motion."""
    loc = locate(cams, rayid)
    if loc[0] == -1:
      return
    pix = camera.pixel_at(cams, out, False, worldid, loc[0], loc[1], loc[2])
    adr = pix.camid * pix.width * pix.height + pix.pixelid
    geometry_pixel(geo, cams, out, tmp, pix, worldid, adr)

  return render_geometry


def make_megakernel(f: Features, tracer: Tracer, model: primary.CameraModel, *, banded: bool):
  """The software-traversal geometry kernel of `f` and camera model `model`: one thread per
  (world, ray) of the (nworld, nray) launch (`camera.launch_ids`), (geo, cams, out[, tmp])
  arguments; `banded`: the CPU form (`pixel.megakernel`), (geo, cams, first, out[, tmp])."""
  options = kernel_structs.kernel_options(f)
  body = make_geometry_body(f, tracer, model)
  if banded:
    return _banded_megakernel(f, body)
  if f.history:
    temporal_body = cast(TemporalBody, body)  # Same specialization branch as make_geometry_body.

    @wp.kernel(**options)
    @concrete(geo=software_geometry_type(live_uv=f.live_material_uv), cams=Cameras)
    def render_geometry_temporal_megakernel(
      # Model:
      geo: SoftwareGeometry,
      cams: Cameras,
      # Out:
      out: OutputArrays,
      tmp: Temporal,
    ) -> None:
      """One thread per (world, ray); the temporal state (motion) is read and updated."""
      i, j = wp.tid()
      worldid, rayid = camera.launch_ids(cams, cams.nworld, i, j)
      if worldid < 0:
        return  # a hole of the active list
      temporal_body(geo, cams, out, tmp, worldid, rayid)

    return render_geometry_temporal_megakernel

  geometry_body = cast(Body, body)

  @wp.kernel(**options)
  @concrete(geo=software_geometry_type(live_uv=f.live_material_uv), cams=Cameras)
  def render_geometry_megakernel(
    # Model:
    geo: SoftwareGeometry,
    cams: Cameras,
    # Out:
    out: OutputArrays,
  ) -> None:
    """One thread per (world, ray)."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, j)
    if worldid < 0:
      return  # a hole of the active list
    geometry_body(geo, cams, out, worldid, rayid)

  return render_geometry_megakernel


def _banded_megakernel(f: Features, body: Body | TemporalBody):
  """`make_megakernel`'s banded form (CPU): thread (i, j) renders ray `first` + j."""
  options = kernel_structs.kernel_options(f)
  if f.history:
    temporal_body = cast(TemporalBody, body)

    @wp.kernel(**options)
    @concrete(geo=software_geometry_type(live_uv=f.live_material_uv), cams=Cameras)
    def render_banded_geometry_temporal_megakernel(
      # Model:
      geo: SoftwareGeometry,
      cams: Cameras,
      # In:
      first: wp.int32,
      # Out:
      out: OutputArrays,
      tmp: Temporal,
    ) -> None:
      """One thread per (world, ray) of the band's rays; the temporal state is read and updated."""
      i, j = wp.tid()
      worldid, rayid = camera.launch_ids(cams, cams.nworld, i, first + j)
      if worldid < 0:
        return  # a hole of the active list
      temporal_body(geo, cams, out, tmp, worldid, rayid)

    return render_banded_geometry_temporal_megakernel

  geometry_body = cast(Body, body)

  @wp.kernel(**options)
  @concrete(geo=software_geometry_type(live_uv=f.live_material_uv), cams=Cameras)
  def render_banded_geometry_megakernel(
    # Model:
    geo: SoftwareGeometry,
    cams: Cameras,
    # In:
    first: wp.int32,
    # Out:
    out: OutputArrays,
  ) -> None:
    """One thread per (world, ray) of the band's rays."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, first + j)
    if worldid < 0:
      return  # a hole of the active list
    geometry_body(geo, cams, out, worldid, rayid)

  return render_banded_geometry_megakernel
