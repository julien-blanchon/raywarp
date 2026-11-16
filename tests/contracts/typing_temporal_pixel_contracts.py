"""Checker-only render pass contracts, including rejected callback mismatches.

These probes are checked with the ordinary Pyrefly tests gate, never executed. Exact suppressions
are negative probes: the unused-ignore diagnostic catches callbacks becoming permissive again.
"""

from __future__ import annotations

from typing import assert_type

import warp as wp

from raywarp.core.kernels import center, first_hit, geometry, pixel
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import BodyClosest, GeometryFields, Hit, Tracer
from raywarp.core.kernels.abi.structs import EnvironmentArrays
from raywarp.core.post.temporal import pixel as temporal_pixel
from raywarp.core.post.temporal import store
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import Materials
from raywarp.core.trace.rays import primary


def _factory_contracts(  # noqa: PLR0913  # one value of each factory input type
  f: Features,
  tracer: Tracer,
  ray: primary.PrimaryRay,
  closest: BodyClosest,
  trace_pixel: store.TracePixel,
  cone: primary.PixelCone,
  writes: first_hit.FirstHitWriters,
  center_trace: center.CenterTrace,
) -> None:
  geometry_body: geometry.Body | geometry.TemporalBody = geometry.make_geometry_body(f, tracer, primary.PINHOLE)
  render_pixel: pixel.PixelBody = pixel.make_render_pixel(f, tracer, primary.PINHOLE)
  del geometry_body, render_pixel
  body = temporal_pixel.make_render_pixel(
    f,
    primary.PINHOLE,
    trace_closest=closest,
    trace_pixel=trace_pixel,
    primary_ray=ray,
    pixel_cone=cone,
    writes=writes,
    center_trace=center_trace,
    shading=0,
  )
  assert_type(body, temporal_pixel.RenderPixel)
  assert_type(temporal_pixel.make_sample_rays(f, ray), tuple[store.SampleRayOf, temporal_pixel.GeometryRay])
  assert_type(store.make_store(f), store.Store)


def _sample_record(
  sample: temporal_pixel.TraceSample,
  geo: GeometryFields,
  mats: Materials,
  lit: Lights,
  env: EnvironmentArrays,
  ray: temporal_pixel.SampleRay,
) -> None:
  assert_type(sample(geo, mats, lit, env, 0, ray), tuple[wp.vec3, Hit, wp.vec3, first_hit.SurfaceRecord])


def _wrong_sample(value: float) -> float:
  return value


def _reject_sample(
  f: Features,
  trace_pixel: store.TracePixel,
  sample_ray: store.SampleRayOf,
) -> None:
  temporal_pixel.make_trace_sample(_wrong_sample)  # pyrefly: ignore[bad-argument-type]  # wrong TracePixel signature
  store.make_more_samples(f, _wrong_sample, sample_ray)  # pyrefly: ignore[bad-argument-type]  # wrong TracePixel inputs/outputs
  store.make_more_samples(f, trace_pixel, _wrong_sample)  # pyrefly: ignore[bad-argument-type]  # wrong SampleRayOf inputs/outputs
