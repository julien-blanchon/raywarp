"""Declared light-type variation preserves the angular-light shader capability."""

from __future__ import annotations

from typing import Final, Literal, cast

import numpy as np
import pytest
import warp as wp
from conftest import MakeScene

import raywarp as rw
from raywarp._base.types import LightType
from raywarp.core.kernels.abi.rays import GeometryFields, SoftwareGeometry
from raywarp.core.kernels.generic import concrete, device_generic
from raywarp.core.shade.lights import evaluators as light_evaluators
from raywarp.core.shade.lights import loop, pack, sources
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.core.shade.materials.material import Lobes
from raywarp.core.shade.shadows.visibility import LightVisibility

XML: Final = """<mujoco><visual><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
<worldbody><light type="{kind}" pos="0 0 2" dir="0 0 -1" cutoff="20" softness="0.25" bulbradius="0" intensity="10" range="10" castshadow="false"/>
<geom type="plane" size="3 3 .1" rgba=".7 .7 .7 1"/>
<camera name="cam" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="65"/></worldbody></mujoco>"""


def test_declared_type_change_matches_prepared_spot(scene: MakeScene) -> None:
  """World1 becomes a spot after preparation; world0 retains the point-light result."""
  dynamic = scene(XML.format(kind="point"), nworld=2, per_world=rw.PerWorld(light=["type"]), outputs=rw.Outputs.of("hdr"))
  assert not dynamic.scene.capabilities.has_spot
  assert dynamic.r.features.has_spot
  dynamic.render()
  before = dynamic.hdr().copy()
  types = dynamic.r.per_world["light.type"]
  values = types.numpy()
  values[1, 0] = int(LightType.SPOT)
  types.assign(values)
  dynamic.render()
  after = dynamic.hdr()
  reference = scene(XML.format(kind="spot"), outputs=rw.Outputs.of("hdr")).render().hdr()[0]
  np.testing.assert_array_equal(before[0], after[0])
  np.testing.assert_allclose(after[1], reference, atol=1e-5, rtol=1e-5)
  assert np.max(np.abs(after[1] - before[1])) > 1e-3


@pytest.mark.parametrize("shaded", [False, True])
def test_undeclared_or_unshaded_type_needs_no_spot_code(scene: MakeScene, shaded: bool) -> None:
  settings = rw.Outputs.of("hdr") if shaded else rw.Outputs.of("depth")
  declaration = None if shaded else rw.PerWorld(light=["type"])
  s = scene(XML.format(kind="point"), per_world=declaration, outputs=settings)
  assert not s.r.features.has_spot


# ---------------------------------------------------------------------------------------------
# Prepared punctual geometry preserves ordered radiance and complete shadow-call inputs.


@wp.func
def _brdf(
  # In:
  _lobes: Lobes,
  _normal: wp.vec3,
  _view: wp.vec3,
  _nov: float,
  _direction: wp.vec3,
  nol: float,
  scale: wp.vec3,
  _widen: float,
) -> wp.vec3:
  """A simple positive term exposing the exact eval_light geometry and radiance scale."""
  return scale * nol


@wp.func
@device_generic
def _begin[G: GeometryFields](
  # Model:
  _geo: G,
  lit: Lights,
  # In:
  _world: int,
  _point: wp.vec3,
  _normal: wp.vec3,
) -> tuple[int, int, int, wp.vec3]:
  """Enumerate authored table entries without any scene-dependent routing."""
  return 0, 0, lit.nlight, wp.vec3(0.0)


@wp.func
def _light_of(
  # Model:
  _lit: Lights,
  # In:
  _slots: int,
  index: int,
) -> int:
  """Read each authored light exactly once in its original table order."""
  return index


def _visibility(tinted: bool) -> LightVisibility[lights.Visibility]:
  """Record the full shadow invocation through typed diagnostic geometry arrays."""
  Result, convert = lights.visibility_ops(tinted)

  @wp.func
  @concrete(**{"return": Result})
  @device_generic
  def visible[G: GeometryFields](
    # Out:
    geo: G,
    # In:
    world: int,
    _light: Light,
    origin: wp.vec3,
    normal: wp.vec3,
    direction: wp.vec3,
    distance: float,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ):
    """Preserve origin/direction/distance/world/light/random keys plus per-light call counts."""
    geo.instance_pos[world, li] = origin
    geo.instance_size[world, li] = direction
    geo.instance_color[world, li] = wp.vec4(distance, float(seed), float(stream), float(nlight))
    geo.material_base_color[world, li] = wp.vec4(float(li), 0.0, 0.0, float(world))
    wp.atomic_add(geo.counters, world, li, wp.int64(1))
    v = wp.clamp(wp.dot(normal, direction) * 0.25 + distance * 0.01, 0.0, 1.0)
    return convert(v)

  return visible


def _kernel(path: str, tinted: bool, prepared: bool, small: int = 0, *, head: bool = False) -> wp.Kernel:
  """Build shared loops with/without the capability over identical callbacks (`small`: the small
  form of that many entries over the loop's own source; `head`: the camera headlight on)."""
  punctual = light_evaluators.make_punctual(has_spot=True, bsdf_eval=_brdf, light_visibility=_visibility(tinted))
  if not prepared:
    punctual = loop.Evaluators(punctual.context, punctual.unshadowed, punctual.visibility, int)
  source = sources.Source(_begin, _light_of, sources.no_remainder, path == "capped", path == "capped")
  form = loop.Small(source, small) if small else None
  shade = loop.make_light_loop(
    source=source,
    shadows=True,
    budget=2 if path == "budget" else 0,
    sun_only=False,
    evaluators=punctual,
    headlight=head,
    small=form,
  )
  head_scale = wp.vec3(0.3, 0.2, 0.1) if head else wp.vec3(0.0)

  @wp.kernel
  def evaluate(
    # Model:
    lit: Lights,
    # Out:
    geo: SoftwareGeometry,
    output: wp.array[wp.vec3],
  ):
    """Shade each world independently while exposing every consumed shadow input."""
    world = wp.tid()
    output[world] = shade(
      geo,
      lit,
      world,
      Lobes(),
      wp.vec3(0.0),
      wp.vec3(0.0, 0.0, 1.0),
      wp.vec3(0.0, 0.0, 0.01),
      wp.vec3(0.0, 0.0, 1.0),
      1.0,
      world + 117,
      3,
      1.0,
      wp.vec3(0.0, 0.0, -1.0),
      head_scale,
    )

  return evaluate


def _inputs(nlight: int = 8) -> Lights:
  """Authored mixed light types include per-world active, intensity, range, cone and color DR."""
  records = []
  kinds = (lights.LIGHT_POINT, lights.LIGHT_DIRECTIONAL, lights.LIGHT_COSINE, lights.LIGHT_TUBE)
  for world in range(2):
    for i in range(nlight):
      light = Light()
      light.type = kinds[i % len(kinds)]
      light.active = i != world
      light.cast_shadow = i != 7
      light.pos = wp.vec3(float(i % 3), 0.25, 2.0)
      light.dir = wp.vec3(0.0, 0.0, -1.0)
      light.color = wp.vec3(0.2 + world, 0.5, 1.0)
      light.intensity = float(i + 1)
      light.range = 0.5 if i == 4 else 8.0
      light.attenuation = wp.vec3(1.0, 0.1, 0.01) if world else wp.vec3(1.0, 0.0, 0.0)
      light.cos_outer = 0.75 if i == 6 else lights.NO_CONE
      light.cone_start, light.cone_scale = 0.1, 1.0
      records.append(light)
  lit = Lights()
  lit.lights = wp.array(records, dtype=Light, device="cpu").reshape((2, nlight))
  lit.hot = pack.hot_records(lit.lights)
  lit.nlight = nlight
  return lit


def _outputs() -> tuple[SoftwareGeometry, wp.array[wp.vec3, [int]]]:
  """Allocate diagnostic arrays whose unused geometry fields never participate in traversal."""
  geo = SoftwareGeometry()
  geo.instance_pos = wp.zeros((2, 8), dtype=wp.vec3, device="cpu")
  geo.instance_size = wp.zeros((2, 8), dtype=wp.vec3, device="cpu")
  geo.instance_color = wp.zeros((2, 8), dtype=wp.vec4, device="cpu")
  geo.material_base_color = wp.zeros((2, 8), dtype=wp.vec4, device="cpu")
  geo.counters = wp.zeros((2, 8), dtype=wp.int64, device="cpu")
  return geo, wp.zeros(2, dtype=wp.vec3, device="cpu")


@pytest.mark.parametrize("path", ["uncapped", "capped", "budget"])
@pytest.mark.parametrize("tinted", [False, True])
def test_prepared_inputs_and_outputs_exact(path: Literal["uncapped", "capped", "budget"], tinted: bool) -> None:
  """Scalar/tinted values and every recorded shadow input remain exact across all loop families."""
  lit = _inputs()
  first, second = _outputs(), _outputs()
  for enabled, (geo, output) in ((False, first), (True, second)):
    wp.launch(_kernel(path, tinted, enabled), dim=2, inputs=[lit], outputs=[geo, output], device="cpu")
  np.testing.assert_array_equal(first[1].numpy(), second[1].numpy())
  assert np.any(first[1].numpy() > 0.0)
  for name in ("instance_pos", "instance_size", "instance_color", "material_base_color", "counters"):
    np.testing.assert_array_equal(getattr(first[0], name).numpy(), getattr(second[0], name).numpy())
  assert np.any(first[0].counters.numpy() > 0)


def _same_shading(lit: Lights, a: wp.Kernel, b: wp.Kernel, rtol: float = 0.0) -> None:
  """`a` and `b` shade `lit` identically (within `rtol`), with identical shadow-ray inputs."""
  first, second = _outputs(), _outputs()
  for kernel, (geo, output) in ((a, first), (b, second)):
    wp.launch(kernel, dim=2, inputs=[lit], outputs=[geo, output], device="cpu")
  np.testing.assert_allclose(first[1].numpy(), second[1].numpy(), rtol=rtol, atol=0.0)
  for name in ("instance_pos", "instance_size", "instance_color", "material_base_color", "counters"):
    np.testing.assert_array_equal(getattr(first[0], name).numpy(), getattr(second[0], name).numpy())


@pytest.mark.parametrize("nlight", [3, 4])
def test_static_small_light_loop_preserves_order_and_shadow_inputs(nlight: int) -> None:
  """Small constant loops preserve live per-world active bits, order and shadow inputs."""
  _same_shading(_inputs(nlight), _kernel("uncapped", False, True), _kernel("uncapped", False, True, nlight))


@pytest.mark.parametrize("path", ["uncapped", "capped"])
def test_headlight_term_out_of_line(path: Literal["uncapped", "capped"]) -> None:
  """The loops' headlight entry adds what the small form's straight-line entry does."""
  lit = _inputs(3)
  with_head = _kernel("uncapped", False, True, 4, head=True)
  _same_shading(lit, _kernel(path, False, True, head=True), with_head, rtol=1e-6)
  output = wp.zeros(2, dtype=wp.vec3, device="cpu")
  wp.launch(with_head, dim=2, inputs=[lit], outputs=[_outputs()[0], output], device="cpu")
  base = _outputs()[1]
  wp.launch(_kernel("uncapped", False, True, 4), dim=2, inputs=[lit], outputs=[_outputs()[0], base], device="cpu")
  assert np.all(output.numpy()[:, 0] > base.numpy()[:, 0])


@pytest.mark.parametrize("nlight", [1, 2, 3, 4])
def test_small_loop_source_is_full_authored_table(nlight: int) -> None:
  """Every eligible uncapped, non-grid source starts at zero and retains authored indices."""
  source = sources._uncapped(*sources.make_cell(nlight, culling=False), -1, culling=False)
  lit = _inputs(nlight)
  geo, _ = _outputs()
  result = wp.full(nlight + 2, -9, dtype=int, device="cpu")

  @wp.kernel
  def inspect(
    # Model:
    lit: Lights,
    geo: SoftwareGeometry,
    # Out:
    result: wp.array[int],
  ):
    """Capture begin's interval and its exact per-slot light identity."""
    slots, first, last, _rest = source.begin(geo, lit, 0, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0))
    result[0] = first
    result[1] = last
    for j in range(first, last):
      result[j + 2] = source.light_of(lit, slots, j)

  wp.launch(inspect, dim=1, inputs=[lit, geo], outputs=[result], device="cpu")
  np.testing.assert_array_equal(result.numpy(), [0, nlight, *range(nlight)])


def test_small_loop_rejects_grid_source() -> None:
  """A future culling caller cannot select a fixed authored-table loop by count alone."""
  cell_range, cell_light = sources.make_cell(3, culling=True)
  source = sources._uncapped(cell_range, cell_light, -1, culling=True)
  punctual = light_evaluators.make_punctual(has_spot=True, bsdf_eval=_brdf, light_visibility=_visibility(False))
  with pytest.raises(ValueError, match="small form"):
    loop.make_light_loop(
      source=source,
      shadows=True,
      budget=0,
      sun_only=False,
      evaluators=punctual,
      small=loop.Small(source, 3),
    )


@pytest.mark.parametrize("cap", [2, 3])
def test_capped_source_within_its_cap_takes_every_light(cap: int) -> None:
  """A cell of at most `cap` lights takes all of them in order, as the uncapped source: none is
  dropped for a zero cheap estimate (world 0's light 0 is off; an area light's punctual form can
  rank 0 where its LTC form lights the point), so a tier build's cap renders as a scene's own
  build without one."""
  source = cast(  # cap 2-3 > 0: a slotted source
    "sources.Source[wp.Vector[int, int]]",
    sources.make_source(nlight=-1, culling=False, cap=cap, has_spot=True, columns=(-1, -1)),
  )
  exact = sources.make_exact_test(nlight=-1, culling=False, cap=cap, has_spot=True)
  lit = _inputs(2)
  geo, _ = _outputs()
  result = wp.full((2, 4), -9, dtype=int, device="cpu")

  @wp.kernel
  def inspect(
    # Model:
    lit: Lights,
    geo: SoftwareGeometry,
    # Out:
    result: wp.array2d[int],
  ):
    """Capture each world's slots, their count and the exact-light test of both lights."""
    w = wp.tid()
    point, normal = wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0)
    slots, _first, last, _rest = source.begin(geo, lit, w, point, normal)
    result[w, 0] = last
    for j in range(last):
      result[w, 1 + j] = source.light_of(lit, slots, j)
    result[w, 3] = int(exact(geo, lit, w, 0, point, normal)) + 2 * int(exact(geo, lit, w, 1, point, normal))

  wp.launch(inspect, dim=2, inputs=[lit, geo], outputs=[result], device="cpu")
  np.testing.assert_array_equal(result.numpy(), [[2, 0, 1, 3], [2, 0, 1, 3]])
