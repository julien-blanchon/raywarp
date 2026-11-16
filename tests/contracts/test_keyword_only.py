"""Flags and same-typed argument runs that cross package boundaries are keyword-only (a
positional `True` silently switched another flag when an argument list changed)."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from raywarp.core.kernels.abi import rays
from raywarp.core.pipeline.platform import device
from raywarp.core.pipeline.textures import uploader
from raywarp.core.shade.gi import kernels as gi_kernels
from raywarp.core.shade.lights import evaluators, grid
from raywarp.core.trace.accel import cutout, instances, query

_CASES: list[tuple[Callable[..., object], tuple[str, ...]]] = [
  (cutout.make_word_cut_away, ("compact", "live_uv", "world_uv", "bound")),
  (cutout.make_cut_triangle, ("has_mesh", "has_box")),
  (query.unculled_queries, ("trees", "alpha_cutout")),
  (gi_kernels.static_kernel, ("trees", "alpha_cutout")),
  (gi_kernels.walls_kernel, ("trees", "alpha_cutout")),
  (grid.visibility_kernel, ("area", "trees", "alpha_cutout")),
  (gi_kernels.survey_kernel, ("trees", "alpha_cutout")),
  (evaluators.make_punctual, ("has_spot", "bsdf_eval", "light_visibility")),
  (query.module_options, ("ordered",)),
  (rays.instance_pose, ("compact",)),
  (instances.pose_column, ("compact",)),
  (uploader.host_texture, ("srgb",)),
  (device.launch_bands, ("device", "n", "launch", "min_band", "per_core")),
  (query.software_tracer, ("cull",)),
]


@pytest.mark.parametrize(("function", "names"), _CASES, ids=[f.__qualname__ for f, _ in _CASES])
def test_flags_are_keyword_only(function: Callable[..., object], names: tuple[str, ...]):
  """Each listed parameter is keyword-only (the `@specialize` factories keep their signature)."""
  params = inspect.signature(function).parameters
  assert {n: params[n].kind for n in names} == dict.fromkeys(names, inspect.Parameter.KEYWORD_ONLY)
