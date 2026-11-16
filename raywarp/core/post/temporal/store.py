"""The end of the motion per-pixel body: a pixel's samples after the first, then its outputs
(`temporal/pixel.make_end`).

`make_more_samples`: the n x n grid's other samples, summed in HDR; `make_store`: the frame's
outputs (the mean's display).
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields, Hit
from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays, Params
from raywarp.core.kernels.display import coverage, store_hdr, write_display
from raywarp.core.kernels.first_hit import SurfaceRecord
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.post.grading.transform import make_display
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import MaterialTables
from raywarp.core.trace.rays import camera

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# trace_pixel(geo, mats, lit, env, p, worldid, origin, direction, max_dist, width, spread,
# headlight_dir, exposure, pixel_seed, stream) -> (radiance, first hit, its base color, its
# record).
type TracePixel = Callable[
  [
    GeometryFields,
    MaterialTables,
    Lights,
    EnvironmentArrays,
    Params,
    int,
    wp.vec3,
    wp.vec3,
    float,
    float,
    float,
    wp.vec3,
    float,
    int,
    int,
  ],
  # (radiance, first hit, base color, the first surface's `first_hit.SurfaceRecord`)
  tuple[wp.vec3, Hit, wp.vec3, SurfaceRecord],
]
# sample_ray(cams, pix, sx, sy) -> (origin, direction, max distance, depth axis) of sample
# (sx, sy) of the pixel's n x n grid.
type SampleRayOf = Callable[[Cameras, camera.Pixel, int, int], tuple[wp.vec3, wp.vec3, float, wp.vec3]]
# store(out, env, worldid, pix, hdr, covered, exposure): the outputs.
type Store = Callable[[OutputArrays, EnvironmentArrays, int, camera.Pixel, wp.vec3, float, float], None]


@specialize
def make_more_samples(f: Features, trace_pixel: TracePixel, sample_ray: SampleRayOf):
  """more_samples(geo, mats, lit, cams, env, worldid, pix, width, spread, headlight_dir,
  exposure, pixel_seed, radiance0, hit0) -> (hdr sum, covered samples).

  Starts from the first sample (radiance0 / hit0, traced by the caller) and adds the n x n
  grid's other samples: linear HDR (averaged before the display transform) and the samples that
  hit something (the coverage alpha, `ColorOutput.alpha`: `display.coverage`).
  """
  n_aa = f.spp

  @wp.func(inline=True)
  @device_generic
  def more_samples[C: Cameras, G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    cams: C,
    env: kernel_structs.EnvironmentArrays,
    # In:
    worldid: int,
    pix: camera.Pixel,
    width: float,
    spread: float,
    headlight_dir: wp.vec3,
    exposure: float,
    pixel_seed: int,
    radiance0: wp.vec3,
    hit0: trace.Hit,
  ) -> tuple[wp.vec3, float]:
    """(hdr sum, covered samples) of the pixel's n x n samples."""
    p = env.params
    hdr = radiance0
    covered = wp.where(hit0.instance_id != -1, 1.0, 0.0)
    for smp in range(1, wp.static(n_aa * n_aa)):
      origin, direction, max_dist, _depth_axis = sample_ray(cams, pix, smp % wp.static(n_aa), smp // wp.static(n_aa))
      radiance, hit, _albedo, _normal = trace_pixel(
        geo,
        mats,
        lit,
        env,
        p,
        worldid,
        origin,
        direction,
        max_dist,
        width,
        spread,
        headlight_dir,
        exposure,
        pixel_seed,
        2 * smp,
      )
      hdr += radiance
      if hit.instance_id != -1:
        covered += 1.0
    return hdr, covered

  return more_samples


@specialize
def make_store(f: Features) -> Store:
  """store(out, env, worldid, pix, hdr, covered, exposure): the frame's outputs.

  `hdr` / `covered`: sums over the n x n samples: rgb (the display of the HDR mean) and hdr.
  """
  n_aa = f.spp
  inv_n2 = 1.0 / float(n_aa * n_aa)
  display = make_display(graded=False)

  @wp.func
  def store(
    # Out:
    out: OutputArrays,
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    pix: camera.Pixel,
    hdr_sum: wp.vec3,
    covered: float,
    exposure: float,
  ) -> None:
    """Writes the pixel's outputs."""
    radr = camera.buffer_index(pix)
    npix = pix.width * pix.height
    hdr = hdr_sum * wp.static(inv_n2)
    alpha = coverage(out, covered, wp.static(inv_n2))
    store_hdr(out, worldid, radr, hdr)
    write_display(out, worldid, radr, pix.lens.adr, npix, display(env, worldid, hdr * exposure), alpha)

  return store
