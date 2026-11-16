"""Pixel-center observations, independent of color antialiasing."""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from raywarp.core.kernels import first_hit
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.output_layout import OUT_EMISSION, OUT_GUIDES, OUT_RECORDED, OUT_SPLIT
from raywarp.core.kernels.abi.rays import BodyClosest, GeometryFields, Hit
from raywarp.core.kernels.abi.structs import EnvironmentArrays, OutputArrays, Params
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.kernels.first_hit import SurfaceRecord
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.evaluate import make_eval_surface
from raywarp.core.shade.materials.material import MaterialTables, make_relative_emission
from raywarp.core.trace.rays import camera

# center_trace(geo, mats, env, params, out, worldid, origin, direction, max_dist, width, spread)
# -> (first hit, its base color, its record) (`make_trace`).
type CenterTrace = Callable[..., tuple[Hit, wp.vec3, SurfaceRecord]]


def shading_outputs(f: Features) -> int:
  """The outputs (`OUT_*` bits) whose pixel-centre value only the full shading gives: direct /
  indirect and a coated emission (`first_hit.split_lighting`); 0 in a build without the split.
  A pass whose sample is off the centre shades the centre ray itself for them (its one shading
  call site: `pixel.make_pixel_body`, `temporal_pixel.make_render_pixel`)."""
  if not first_hit.split_lighting(f):
    return 0
  return OUT_SPLIT | (OUT_EMISSION if f.coat_layer else 0)


@specialize
def make_trace(f: Features, *, closest: BodyClosest):
  """A center hit and its surface values without shading: one closest ray and, for the guides and
  recorded outputs, its material (no lights, reflections or transparency layers). The outputs only
  the shading gives (`shading_outputs`) come from the pass's own shading of the centre ray."""
  evaluate = make_eval_surface(f)
  relative_emission = make_relative_emission(f)

  @wp.func
  @device_generic
  def trace_center[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    env: EnvironmentArrays,
    params: Params,
    out: OutputArrays,
    # In:
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    width: float,
    spread: float,
  ) -> tuple[Hit, wp.vec3, SurfaceRecord]:
    """The first surface on the center ray, with a zero record on a miss."""
    hit = closest(geo, worldid, origin, direction, max_dist)
    albedo = wp.vec3(0.0)
    rec = first_hit.zero_record()
    if out_on(out.bits, wp.static(OUT_GUIDES | OUT_RECORDED)):
      if hit.instance_id != -1:
        point = origin + direction * hit.dist
        surface, normal = evaluate(geo, mats, worldid, hit, point, direction, width + spread * hit.dist)
        albedo = surface.base_color
        relative = relative_emission(params.relative_emission, surface)
        emitted = first_hit.emission_term(surface, exposure_at(env.exposure, worldid), relative)
        rec = first_hit.capture(surface, normal, point, emitted, wp.vec3(0.0), wp.vec3(0.0))
    return hit, albedo, rec

  return trace_center


@specialize
def make_write(writes: first_hit.FirstHitWriters):
  """Write every first-hit observation the output word has from one consistent camera ray."""
  geometry, hit_values, surface = writes

  @wp.func(inline=True)
  @device_generic
  def write[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    pix: camera.Pixel,
    hit: Hit,
    depth_axis: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    albedo: wp.vec3,
    rec: SurfaceRecord,
    seed: int,
  ) -> None:
    """Depth, labels, guides and surface channels at the same hit."""
    adr = camera.buffer_index(pix)
    geometry(out, worldid, pix, hit, depth_axis)
    hit_values(geo, out, worldid, adr, hit, origin, direction)
    surface(geo, lit, out, worldid, adr, albedo, rec, seed)

  return write
