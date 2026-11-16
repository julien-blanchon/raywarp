"""The hot light record (`punctual.LightHot`) evaluates every light form bit for bit like
`Light`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, assert_type

import numpy as np
import pytest
import warp as wp
from conftest import DOWN, required

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.kernels.abi import counters, rays
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.queries import Queries
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import concrete
from raywarp.core.pipeline.platform import tuning
from raywarp.core.shade.lights import area as area_lights
from raywarp.core.shade.lights import grid, loop, ltc, pack, punctual, sources
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.core.shade.shadows import budget as shadow_budget
from raywarp.core.shade.shadows.visibility import ShadowRay, make_shadow_ray
from raywarp.core.trace.optix.device import intrinsics as optix_device

_TYPES = (punctual.LIGHT_SPOT, punctual.LIGHT_DIRECTIONAL, punctual.LIGHT_POINT, punctual.LIGHT_COSINE, punctual.LIGHT_TUBE)


def _records(n: int, rng: np.random.Generator) -> list[Light]:
  """`n` packed lights of every type: classic polynomials or none, cones or none, some off."""
  out = []
  for i in range(n):
    light = Light()
    light.type = _TYPES[i % len(_TYPES)]
    light.active = rng.random() > 0.2
    light.cast_shadow = rng.random() > 0.5
    light.pos = wp.vec3(*rng.uniform(-3.0, 3.0, 3))
    light.dir = wp.normalize(wp.vec3(*rng.normal(size=3)))
    light.color = wp.vec3(*rng.uniform(0.1, 1.0, 3))
    light.intensity = float(rng.uniform(1.0, 500.0))
    light.range = float(rng.choice([0.0, rng.uniform(1.0, 8.0)]))
    light.attenuation = wp.vec3(1.0, 0.0, 0.0) if rng.random() < 0.5 else wp.vec3(*rng.uniform(0.0, 0.5, 3))
    coned = rng.random() < 0.5
    light.cos_outer = float(np.cos(rng.uniform(0.3, 1.4))) if coned else punctual.NO_CONE
    light.cone_start = float(rng.uniform(0.0, 0.3))
    light.cone_scale = float(rng.uniform(1.0, 10.0))
    out.append(light)
  return out


@wp.kernel
def _both(
  lit: Lights,
  points: wp.array[wp.vec3],
  full: wp.array2d[wp.vec3],
  hot: wp.array2d[wp.vec3],
):
  """(scale x NoL-free direction sum) of every light at every point, both records, spot on."""
  p, li = wp.tid()
  point = points[p]
  l, dist, scale = punctual.eval_light(punctual.fetch(lit, 0, li), True, point, lit.falloff_floor)
  full[p, li] = scale + l * dist
  h = punctual.fetch_hot(lit, 0, li)
  l2, dist2, scale2 = punctual.eval_hot(lit, 0, li, h, True, point)
  hot[p, li] = scale2 + l2 * dist2


def _lights_struct(records: list[Light]) -> Lights:
  """A one-row light struct over `records` with its hot records."""
  lit = Lights()
  lit.lights = wp.array(records, dtype=Light, device="cpu").reshape((1, len(records)))
  lit.hot = pack.hot_records(lit.lights)
  lit.nlight = len(records)
  lit.falloff_floor = punctual.falloff_floor(1.0)
  return lit


def test_eval_hot_matches_eval_light_bitwise() -> None:
  """Every type, classic polynomial, cone, range and off light: identical bits."""
  rng = np.random.default_rng(7)
  lit = _lights_struct(_records(60, rng))
  points = wp.array(rng.uniform(-4.0, 4.0, (200, 3)).astype(np.float32), dtype=wp.vec3, device="cpu")
  full = wp.zeros((200, 60), dtype=wp.vec3, device="cpu")
  hot = wp.zeros((200, 60), dtype=wp.vec3, device="cpu")
  wp.launch(_both, dim=(200, 60), inputs=[lit, points], outputs=[full, hot], device="cpu")
  np.testing.assert_array_equal(full.numpy().view(np.uint32), hot.numpy().view(np.uint32))


def test_hot_flags() -> None:
  """The flags carry type, on, caster, classic polynomial and cone."""
  rng = np.random.default_rng(3)
  records = _records(20, rng)
  flags = record_field(_lights_struct(records).hot.numpy()[0], "flags")
  for light, f in zip(records, flags, strict=True):
    assert f & punctual.HOT_TYPE_MASK == light.type
    assert bool(f & punctual.HOT_ACTIVE) == light.active
    assert bool(f & punctual.HOT_CASTER) == light.cast_shadow
    assert bool(f & punctual.HOT_CLASSIC) == (tuple(light.attenuation) != (1.0, 0.0, 0.0))
    assert bool(f & punctual.HOT_CONED) == (light.cos_outer > punctual.NO_CONE)


def test_grid_cells_list_strongest_first() -> None:
  """Each cell's lights: suns first, then by luminance over squared distance from its centre,
  ties in light order; the cell grouping and membership are unchanged."""
  records = []
  for pos, intensity, ltype in (
    ((0.5, 0.5, 0.5), 1.0, punctual.LIGHT_POINT),  # weak, in cell 0
    ((3.5, 0.5, 0.5), 50.0, punctual.LIGHT_POINT),  # strong, in cell 3
    ((0.0, 0.0, 9.0), 2.0, punctual.LIGHT_DIRECTIONAL),
    ((3.5, 0.5, 0.5), 50.0, punctual.LIGHT_POINT),  # a tie with light 1
  ):
    light = Light()
    light.type, light.pos, light.intensity, light.color = ltype, wp.vec3(*pos), intensity, wp.vec3(1.0)
    records.append(light)
  lit = _lights_struct(records)
  mask = np.array(
    [[True, True, True, True], [False, False, False, False], [False, True, False, False], [True, True, False, True]]
  )
  pairs = grid._pairs(mask, lit, lo=np.zeros(3), cell=1.0, dims=[4, 1, 1])
  ids = grid._strongest_first(pairs, np.ones(len(pairs.ids), bool))
  # Cell 0: the sun, then 50 / 3^2 (lights 1, 3: a tie, light order) before 1 / 1 (light 0).
  np.testing.assert_array_equal(ids, [2, 1, 3, 0, 1, 1, 3, 0])


# ---------------------------------------------------------------------------------------------
# Light-loop inference probes and the real scalar/RGB Warp overload boundary.


@wp.kernel
def _attenuation_cases(
  # Out:
  out: wp.array1d[wp.vec3],
) -> None:
  c = wp.vec3(2.0, 4.0, 8.0)
  out[0] = punctual.attenuate(c, 0.25)
  out[1] = punctual.attenuate(c, wp.vec3(0.25, 0.5, 1.0))


def test_scalar_and_rgb_visibility_keep_real_warp_overloads() -> None:
  out = wp.empty(2, dtype=wp.vec3, device="cpu")
  wp.launch(_attenuation_cases, dim=1, outputs=[out], device="cpu")
  np.testing.assert_array_equal(out.numpy(), [[0.5, 1.0, 2.0], [0.5, 2.0, 8.0]])


def _converter_types(flag: bool) -> None:
  assert_type(punctual.visibility_ops(False), tuple[type[float], Callable[[float], float]])
  assert_type(punctual.visibility_ops(True), tuple[type[wp.vec3], Callable[[float], wp.vec3]])
  assert_type(punctual.visibility_ops(flag), tuple[type[float] | type[wp.vec3], punctual.VisibilityOf])


def _matched_context(
  context: loop.Context[int],
  plain: loop.Unshadowed[int],
  area: loop.Unshadowed[ltc.Frame],
  visibility: loop.LightVisibilityAt,
) -> None:
  assert_type(loop.Evaluators(context, plain, visibility, int), loop.Evaluators[int])
  loop.Evaluators(context, area, visibility, int)  # type: ignore[bad-argument-type]  # LTC evaluator cannot consume an int context


def _slot_shapes(visibility: loop.LightVisibilityAt) -> None:
  budget: Literal[3] = 3
  slots: shadow_budget.TraceSlots[Literal[3]] = shadow_budget._make_trace_slots(budget, visibility)
  del slots


def _slot_shape_mismatch(slots: shadow_budget.TraceSlots[Literal[3]], geo: GeometryFields, lit: punctual.Lights) -> None:
  slots(
    geo,
    lit,
    0,
    wp.vec3i(-1),
    wp.vec4(0.0),  # type: ignore[bad-argument-type]  # three slots require three luminances
    wp.mat33(),
    wp.vec3(),
    wp.vec3(),
    wp.vec3(),
    0,
    0,
  )


def _shadow_result(f: Features, q: Queries) -> None:
  shadow: ShadowRay[punctual.Visibility] = make_shadow_ray(f, q)
  del shadow
  invalid: ShadowRay[str] = make_shadow_ray(f, q)  # type: ignore[bad-assignment]  # return is no longer a free type parameter
  del invalid


def _source_shapes() -> None:
  cell, light = sources.make_cell(8, culling=False)
  cap: Literal[3] = 3
  assert_type(sources._uncapped(cell, light, -1, culling=False), sources.Source[int])
  assert_type(
    sources._capped(cell, light, cap=cap, has_spot=False, columns=(-1, -1)),
    sources.Source[wp.Vector[int, Literal[3]]],
  )
  area: area_lights.EvalAreaLight = area_lights.make_eval_area_light(clearcoat=True)
  del area


def _source_shape_mismatch(
  begin: sources.Begin[wp.vec3i],
  plain: sources.LightOf[int],
  three_slots: sources.LightOf[wp.vec3i],
  four_slots: sources.LightOf[wp.Vector[int, Literal[4]]],
) -> None:
  assert_type(sources.Source(begin, three_slots, sources.no_remainder, True, True), sources.Source[wp.vec3i])
  sources.Source(begin, plain, sources.no_remainder, True, True)  # type: ignore[bad-argument-type]  # scalar lookup cannot consume slot IDs
  sources.Source(begin, four_slots, sources.no_remainder, True, True)  # type: ignore[bad-argument-type]  # producer and lookup slot widths must agree


# ---------------------------------------------------------------------------------------------
# Candidate light entries differ from nonzero terms and shadow-caster eligibility.


def _renderer(
  *,
  counted: bool,
  shadows: bool,
  budget: int,
  device: str = "cpu",
  glossy: bool = False,
  backend: Literal["software", "optix"] = "software",
) -> rw.Renderer[None]:
  builder = rw.SceneBuilder()
  material = builder.add_material(
    rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=1.0 if glossy else 0.0, specular_roughness=0.2 if glossy else 1.0)
  )
  builder.add_instance("plane", material=material, size=(5.0, 5.0, 0.1))
  if glossy:
    ceiling = builder.add_material(rw.Material(base_color=(0.6, 0.3, 0.1), specular_roughness=1.0))
    builder.add_instance("box", material=ceiling, size=(5.0, 5.0, 0.1), xform=np.c_[np.eye(3), [0.0, 0.0, 3.0]])
  for cast, intensity, upward in (
    (True, 2.0, False),
    (True, 3.0, False),
    (False, 1.0, False),
    (True, 2.0, True),
    (True, 0.0, False),
  ):
    builder.add_light(
      rw.Light(
        kind=rw.scene.LightKind.DISTANT,
        intensity=intensity,
        angle=0.0,
        shadow_enable=cast,
        xform=np.array([[1, 0, 0, 0], [0, -1 if upward else 1, 0, 0], [0, 0, -1 if upward else 1, 1]], dtype=np.float64),
      )
    )
  with tuning.override(counters=counted):
    renderer = rw.Renderer(
      rw.upload(builder.build(), device),
      nworld=2,
      cameras=[rw.CameraSpec(fovy=5)],
      resolution=(4, 4),
      settings=rw.RenderSettings(
        outputs=rw.Outputs.of("rgb", "hdr"),
        gi=rw.settings.GI(enabled=False),
        environment=rw.settings.Environment(enabled=False),
        lighting=rw.settings.Lights(camera_light=False, emissive_lights=False, light_grid=False, max_exact=None),
        shadows=rw.settings.Shadows(enabled=shadows, max_casters=budget or None),
        reflections=rw.settings.Reflections(enabled=glossy, shadows="sun", max_exact=None),
        aa=rw.settings.AntiAliasing(),
        performance=rw.settings.Performance(traversal=backend),
      ),
    )
  renderer.camera_pose.assign(np.array([[[0, 0, 2, *DOWN]], [[0, 0, 2, *DOWN]]], dtype=np.float32))
  return renderer


@pytest.mark.parametrize("shadows,budget", [(False, 0), (True, 0), (True, 1)])
def test_nonzero_terms_and_casters_before_shadow_selection(shadows: bool, budget: int) -> None:
  """Candidate entries include rejected terms; caster eligibility remains with shadows off."""
  plain = _renderer(counted=False, shadows=shadows, budget=budget)
  counted = _renderer(counted=True, shadows=shadows, budget=budget)
  try:
    for renderer in (plain, counted):
      assert renderer._tables.counters.dtype == wp.int64
      assert renderer._tables.counters.capacity == 2 * counters.NCOUNTER * 8
    plain.render(seed=19)
    counted.render(seed=19)
    np.testing.assert_array_equal(required(counted.outputs.hdr).numpy(), required(plain.outputs.hdr).numpy())
    np.testing.assert_array_equal(required(counted.outputs.rgb).numpy(), required(plain.outputs.rgb).numpy())
    actual = counted.counters()
    assert actual["lights"] == 5.0
    assert actual["light_terms"] == 3.0
    assert actual["light_casters"] == 2.0
    assert actual["hit_light_terms"] == actual["hit_light_casters"] == 0.0
    if not shadows:
      assert actual["shadow_rays"] == 0.0
    # Every world has an independent row; these totals are not just a correct global average.
    raw = counted._tables.counters.numpy()
    np.testing.assert_array_equal(
      raw[:, [counters.LIGHTS, counters.LIGHT_TERMS, counters.LIGHT_CASTERS]], [[80, 48, 32], [80, 48, 32]]
    )
  finally:
    counted.close()
    plain.close()


def test_glossy_hit_terms_have_separate_columns() -> None:
  renderer = _renderer(counted=True, shadows=True, budget=0, glossy=True)
  try:
    renderer.render(seed=7)
    actual = renderer.counters()
    assert 0.0 < actual["hit_light_terms"] <= actual["hit_lights"]
    assert actual["hit_light_casters"] == actual["hit_light_terms"]
    assert actual["light_terms"] <= actual["lights"]
  finally:
    renderer.close()


@pytest.mark.parametrize("counted", [False, True])
def test_optix_counter_increments_compile_out(counted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Generate real OptiX CUDA without a GPU; disabled builds have no counter atomics."""
  from fixtures.optix_codegen import cuda_source, scene_flags

  from raywarp.core.trace.optix.device import programs

  plane_only = scene_flags(has_mesh=False, has_box=False, primitive_types=(), has_world_curves=False)
  (tmp_path / "optix_device.h").write_text("")
  monkeypatch.setattr(optix_device, "INCLUDE_DIR", tmp_path)
  renderer = _renderer(counted=counted, shadows=True, budget=0)
  try:
    source = cuda_source(programs.build(renderer.features, plane_only))
    assert ("wp::atomic_add" in source) is counted
    # `counters.term` (the light evaluators' own inline `term` is always there).
    assert ("// def term[G: GeometryFields](" in source) is counted
  finally:
    renderer.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA/OptiX counter capture gate")
def test_optix_counted_capture_preserves_full_output() -> None:
  """Same-input on/off full images and per-frame reset through actual native graph replay."""
  from raywarp.core.pipeline.platform.device import device_caps

  if not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  plain = _renderer(counted=False, shadows=True, budget=0, device="cuda:0", backend="optix")
  counted = _renderer(counted=True, shadows=True, budget=0, device="cuda:0", backend="optix")
  try:
    plain.render(seed=19)
    graph = counted.capture(seed=19)
    graph.launch()
    first = counted.counters()
    graph.launch()
    assert counted.counters() == first
    assert first["lights"] == 5.0 and first["light_terms"] == 3.0 and first["light_casters"] == 2.0
    np.testing.assert_array_equal(required(counted.outputs.hdr).numpy(), required(plain.outputs.hdr).numpy())
    np.testing.assert_array_equal(required(counted.outputs.rgb).numpy(), required(plain.outputs.rgb).numpy())
  finally:
    counted.close()
    plain.close()


# ---------------------------------------------------------------------------------------------
# Large batched light totals must not wrap signed or unsigned 32-bit counters.


def test_optix_shadow_path_columns_keep_existing_indices() -> None:
  assert counters.SHADOW == 8
  assert counters.CURVE_ANYHIT_IGNORED == 16
  assert counters.NCOUNTER == 31
  assert counters.NAMES[17:] == (
    "plane_any_terminated",
    "shared_any_calls",
    "shared_any_hits",
    "world_cell_rejected",
    "world_any_calls",
    "world_any_hits",
    "translucent_passes",
    "scene_closest_walks",
    "scene_any_walks",
    "scene_nodes",
    "instance_tests",
    "blas_nodes",
    "blas_triangles",
    "stack_drops",
  )
  # The mesh queries add their three counts at consecutive addresses.
  assert counters.BLAS_TRIANGLES == counters.BLAS_NODES + 1
  assert counters.STACK_DROPS == counters.BLAS_NODES + 2


@wp.kernel
@concrete(geo=rays.software_geometry_type(live_uv=False))
def _large_increments(
  # Out:
  geo: rays.SoftwareFields,
):
  i = wp.tid()
  world = i % 2
  # Each individual increment fits int32; the two independent world totals exceed 2^32.
  counters.add(geo, world, counters.LIGHTS, 1073741824 + world)
  counters.add(geo, world, counters.HIT_LIGHTS, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_full_width_atomic_totals_and_graph_clear(device: str) -> None:
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  table = wp.zeros((2, counters.NCOUNTER), dtype=wp.int64, device=device)
  geo = rays.software_geometry_type(live_uv=False)()
  geo.counters = table
  wp.launch(_large_increments, dim=9, inputs=[geo], device=device)
  expected = np.zeros((2, counters.NCOUNTER), dtype=np.int64)
  expected[:, counters.LIGHTS] = [5 * 1073741824, 4 * 1073741825]
  expected[:, counters.HIT_LIGHTS] = [5, 4]
  assert np.all(expected[:, counters.LIGHTS] > 2**32)
  assert table.numpy().dtype == np.int64
  np.testing.assert_array_equal(table.numpy(), expected)
  means = counters.per_pixel(table, 1)
  assert means["lights"] == float(expected[:, counters.LIGHTS].sum()) / 2
  assert means["hit_lights"] == 4.5
  if device != "cpu":
    with wp.ScopedCapture(device=device) as capture:
      table.zero_()
      wp.launch(_large_increments, dim=9, inputs=[geo], device=device)
    assert capture.graph is not None
    for _ in range(2):
      wp.capture_launch(capture.graph)
      np.testing.assert_array_equal(table.numpy(), expected)
