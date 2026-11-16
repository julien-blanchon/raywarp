"""The light a shaded point receives straight from the scene's lights: each light's reflected
radiance, shadowed, summed over the lights (shadow rays, the BRDF of one light, the light loops).

Every piece is a `wp.func` built by its own factory and specialised on `Features` (`wp.static`):

  make_visibility     -> light_visibility: shadow rays of a punctual light
                         (`shade/shadows/visibility.py`)
  make_bsdf_eval      -> bsdf_eval: one light's unshadowed radiance through the lobes (the one
                         BSDF)
  make_direct_lights  -> direct_lights: the camera rays' light loop added to a radiance
  make_lighting       -> the camera rays' loop and the cheaper one at glossy-reflection hits
  make_scene_lights   -> the GI probe hits' loop (a fixed cap and shadow budget, not the
                         camera's)

The light loops are `shade/lights/loop.py`'s (one loop for punctual and area lights, the light
grid and the light caps of `shade/lights/sources.py`, shadows and the shadow budget of
`shade/shadows/budget.py`). A light loop has the signature (geo, lit, worldid, lobes, point,
normal, origin, view, nov, seed, stream, sun, head_dir, head) -> radiance: `point` / `normal`
the shading point and its shading normal, `origin` its secondary-ray origin (`Surface.origin`),
`head_dir` / `head` the camera headlight's direction and radiance scale: the camera rays' loops
evaluate it as one more light entry (no shadow; zero `head`: off).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Final, cast, final

import warp as wp

from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.features import Features, small_form
from raywarp.core.kernels.abi.queries import Queries
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.lights import evaluators as light_evaluators
from raywarp.core.shade.lights import loop as light_loop
from raywarp.core.shade.lights import ltc, sources
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import brdf, material, openpbr_lobes
from raywarp.core.shade.materials.material import COAT_F0, Lobes
from raywarp.core.shade.shadows.visibility import LightVisibility, make_shadow_pass, make_visibility
from raywarp.settings.fields import REFLECTION_SHADOWS

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# bsdf_eval(lobes, normal, view, nov, l, nol, scale, widen) -> unshadowed radiance
# (`make_bsdf_eval`).
type BsdfEval = Callable[[Lobes, wp.vec3, wp.vec3, float, wp.vec3, float, wp.vec3, float], wp.vec3]

# direct_lights(geo, lit, worldid, lobes, point, normal, origin, view, nov, seed, stream,
# head_dir, head, radiance_in) -> radiance (`make_direct_lights`).
type DirectLights = Callable[
  [
    GeometryFields,
    Lights,
    int,
    Lobes,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    float,
    int,
    int,
    wp.vec3,
    wp.vec3,
    wp.vec3,
  ],
  wp.vec3,
]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DirectLighting:
  """The direct-lighting funcs of the camera rays of one `Features` (see the module docstring)."""

  direct_lights: DirectLights
  # The light loop at glossy-reflection hits (None without glossy reflections; a
  # `light_loop.make_light_loop` func).
  hit_lights: light_loop.LightLoop | None
  # The shadow rays of one punctual light (`make_visibility`; the shadow output).
  light_visibility: LightVisibility[lights.Visibility]


@wp.func
def specular_term(
  # In:
  lobes: Lobes,
  view: wp.vec3,
  nov: float,
  l: wp.vec3,
  nol: float,
  h: wp.vec3,
  noh: float,
  loh: float,
) -> wp.vec3:
  """The compensated base specular D V F (no cosine): anisotropic GGX in the lobes' tangent frame
  where `Lobes.anisotropy` > 0 (Filament `anisotropicLobe`), else isotropic."""
  if lobes.anisotropy > 0.0:
    t = lobes.tangent
    b = lobes.bitangent
    d = brdf.d_ggx_anisotropic(lobes.alpha_t, lobes.alpha_b, wp.dot(t, h), wp.dot(b, h), noh)
    v = brdf.v_smith_ggx_correlated_anisotropic(
      lobes.alpha_t, lobes.alpha_b, wp.dot(t, view), wp.dot(b, view), wp.dot(t, l), wp.dot(b, l), nov, nol
    )
    return wp.cw_mul(brdf.f_schlick_rgb(lobes.f0, lobes.f90, loh) * (d * v), lobes.energy_comp)
  return wp.cw_mul(brdf.specular_lobe_rgb(lobes.f0, lobes.f90, lobes.alpha, noh, nov, nol, loh), lobes.energy_comp)


@wp.func
def cloth_diffuse(
  # In:
  lobes: Lobes,
  nol: float,
) -> wp.vec3:
  """Filament's cloth diffuse with NoL: the wrapped cosine (`brdf.wrap_diffuse`), tinted
  saturate(tint + NoL), in place of Lambert's NoL."""
  tint = lobes.wrap_tint + wp.vec3(nol)
  tint = wp.vec3(wp.clamp(tint[0], 0.0, 1.0), wp.clamp(tint[1], 0.0, 1.0), wp.clamp(tint[2], 0.0, 1.0))
  return wp.cw_mul(lobes.diffuse_color, tint) * (brdf.diffuse_lambert() * brdf.wrap_diffuse(nol, lobes.wrap))


@specialize
def _make_specular(
  f: Features,
):
  """specular(lobes, view, nov, l, nol, h, noh, loh, widen) -> `make_bsdf_eval`'s base
  specular lobe."""
  anisotropy = f.specular_anisotropy
  gate = toggles.bound_on(f.bound)  # the layers class bit (`Lobes.bits`)
  field = toggles.bound_on(f.bound, high=True)  # the layered lobes' field bits (`toggles.LAYER_*`)

  @wp.func
  def specular(
    # In:
    lobes: Lobes,
    view: wp.vec3,
    nov: float,
    l: wp.vec3,
    nol: float,
    h: wp.vec3,
    noh: float,
    loh: float,
    widen: float,
  ) -> wp.vec3:
    """The base specular lobe D V F x compensation (no cosine), its roughness widened by `widen`
    (`light_evaluators.source_widening`; the anisotropic lobe keeps its own)."""
    fr = wp.vec3(0.0)
    anisotropic = bool(False)
    if wp.static(anisotropy):
      anisotropic = (
        gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
        and field(lobes.bits2, wp.static(toggles.LAYER_ANISOTROPY))
        and lobes.anisotropy > 0.0
      )
    if anisotropic:
      fr = specular_term(lobes, view, nov, l, nol, h, noh, loh)
    else:
      # Isotropic (whether or not the anisotropic lobe is compiled in: one expression).
      alpha = wp.min(wp.sqrt(lobes.alpha * lobes.alpha + widen * widen), 1.0)
      fr = wp.cw_mul(brdf.specular_lobe_rgb(lobes.f0, lobes.f90, alpha, noh, nov, nol, loh), lobes.energy_comp)
    return fr

  return specular


@specialize
def _make_coat_term(
  f: Features,
):
  """coat_term(lobes, normal, nov, l, nol, h, noh, loh, scale, widen) -> `make_bsdf_eval`'s
  coat lobe."""
  layered_coat = f.coat_layer or f.coat_normal
  gate = toggles.bound_on(f.bound)  # the layers class bit (`Lobes.bits`)

  @wp.func
  def coat_term(
    # In:
    lobes: Lobes,
    normal: wp.vec3,
    nov: float,
    l: wp.vec3,
    nol: float,
    h: wp.vec3,
    noh: float,
    loh: float,
    scale: wp.vec3,
    widen: float,
  ) -> tuple[wp.vec3, float]:
    """(the coat lobe's radiance, its NoL floored for `coat_base_scale`): its GGX lobe about its
    own normal, NoV and F0 (the base's for a glTF coat)."""
    cn = normal
    cnov = nov
    f0 = wp.static(COAT_F0)
    cnol = nol
    cnoh = noh
    layered = False
    if wp.static(layered_coat):
      layered = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
    if layered:
      cn = lobes.coat_normal
      cnov = lobes.coat_nov
      f0 = lobes.coat_f0
      cnol = wp.dot(cn, l)
      cnoh = wp.max(wp.dot(cn, h), 0.0)
    coat = wp.vec3(0.0)
    if cnol > 0.0:
      frc = brdf.coat_lobe(f0, wp.min(wp.sqrt(lobes.coat_alpha * lobes.coat_alpha + widen * widen), 1.0), cnoh, cnov, cnol, loh)
      coat = scale * (lobes.coat * frc * cnol)
    return coat, wp.max(cnol, 1.0e-4)

  return coat_term


@specialize
def make_bsdf_eval(f: Features):
  """bsdf_eval(lobes, normal, view, nov, l, nol, scale, widen) -> unshadowed radiance of
  one light from `l` (NoL > 0; a wrapped cloth, Features.fuzz: NoL > -w) of radiance scale `scale`:
  the one BSDF of every punctual light (camera, reflection and probe hits, the headlight).

  The base lobes (GGX specular, anisotropic with Features.specular_anisotropy; Lambert, EON with
  Features.diffuse_roughness, the cloth's wrapped diffuse with Features.fuzz) under the coat and
  fuzz layers (`material.make_layers`: the law the area and image-based terms share). The diffuse
  lobe already carries the (1 - E_spec) coupling (`brdf.coupled_diffuse`). `widen`: the
  roughness the light's size adds to the specular and coat lobes
  (`light_evaluators.source_widening`; 0: none).
  """
  wrapped = f.fuzz
  clearcoat = f.clearcoat
  rough = f.diffuse_roughness
  layers = material.layers_of(f)
  specular = _make_specular(f)
  coat_term = _make_coat_term(f)
  gate = toggles.bound_on(f.bound)  # the layers, coat and materials class bits (`Lobes.bits`)
  field = toggles.bound_on(f.bound, high=True)  # the fuzz and EON fields (`Lobes.bits2`)

  @wp.func
  def bsdf_eval(
    # In:
    lobes: Lobes,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    l: wp.vec3,
    nol: float,
    scale: wp.vec3,
    widen: float,
  ) -> wp.vec3:
    """Unshadowed radiance of a light arriving from `l` (`make_bsdf_eval`)."""
    h = wp.normalize(l + view)
    noh = wp.max(wp.dot(normal, h), 0.0)
    loh = wp.max(wp.dot(l, h), 0.0)
    lit = nol
    fr = wp.vec3(0.0)
    cloth = False
    if wp.static(wrapped):
      cloth = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS)) and field(lobes.bits2, wp.static(toggles.LAYER_FUZZ))
    if cloth:
      lit = wp.max(nol, 0.0)  # a cloth wrapped past the terminator: its diffuse only
      if nol > 0.0:
        fr = specular(lobes, view, nov, l, nol, h, noh, loh, widen)
    else:
      fr = specular(lobes, view, nov, l, nol, h, noh, loh, widen)
    fd = lobes.diffuse_color * brdf.diffuse_lambert()
    if wp.static(rough):
      eon = gate(lobes.bits, wp.static(toggles.CLASS_MATERIALS)) and field(
        lobes.bits2, wp.static(toggles.MAT_DIFFUSE_ROUGHNESS)
      )
      if eon and lobes.diffuse_roughness > 0.0:
        fd = openpbr_lobes.eon_eval_split(
          lobes.diffuse_single, lobes.diffuse_multi, lobes.diffuse_roughness, nol, nov, wp.dot(l, view)
        )
    base = (fd + fr) * lit
    if cloth:
      if lobes.wrap > 0.0:
        base = cloth_diffuse(lobes, nol) + fr * lit
    coat = wp.vec3(0.0)
    coat_nol = nol
    if wp.static(clearcoat):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT)) and lobes.coat > 0.0:
        coat, coat_nol = coat_term(lobes, normal, nov, l, nol, h, noh, loh, scale, widen)
    sheen = wp.vec3(0.0)
    if cloth:
      if lobes.fuzz_weight > 0.0 and lit > 0.0:
        d = brdf.d_charlie(lobes.fuzz_alpha, noh)
        sheen = wp.cw_mul(lobes.fuzz_color, scale) * (lobes.fuzz_weight * lobes.fuzz_norm * d * brdf.v_neubelt(nov, nol) * nol)
    return layers(lobes, wp.cw_mul(base, scale), coat, coat_nol, sheen)

  return bsdf_eval


@specialize
def make_direct_lights(loop: light_loop.LightLoop):
  """direct_lights(..., radiance_in) -> radiance_in plus every light's radiance (`loop`: a light
  loop, module doc)."""

  @wp.func
  @device_generic
  def direct_lights[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    head_dir: wp.vec3,
    head: wp.vec3,
    radiance_in: wp.vec3,
  ) -> wp.vec3:
    """`radiance_in` plus the radiance from the scene's lights and the headlight (`head`;
    accumulated in the loop's order; every directional caster is traced or open: `sun` 1)."""
    return radiance_in + loop(geo, lit, worldid, lobes, point, normal, origin, view, nov, seed, stream, 1.0, head_dir, head)

  return direct_lights


@specialize
def make_evaluators(f: Features, q: Queries):
  """The light loop's per-light evaluators: punctual lights, plus the LTC area lights when
  compiled in (`light_evaluators.make_area_evaluators`); with `q.light_visibility`, their
  visibility is that call (a program of its own: the loops then hold one call site per shadow-ray
  site, not the inlined rays; no prepared punctual terms)."""
  evaluators = _inline_evaluators(f, q)
  if q.light_visibility is None:
    return evaluators
  return evaluators._replace(visibility=q.light_visibility, prepared=None)


def light_visibility(f: Features, q: Queries) -> light_loop.LightVisibilityAt:
  """The light loops' visibility of one light, inline (`make_evaluators` without
  `q.light_visibility`): the body of the program that call runs (camera and GI probe rays)."""
  return _inline_evaluators(f, q).visibility


@specialize
def _inline_evaluators(f: Features, q: Queries) -> light_loop.Evaluators[int] | light_loop.Evaluators[ltc.Frame]:
  """`make_evaluators` with the evaluators' own visibility."""
  punctual = light_evaluators.make_punctual(
    has_spot=f.has_spot,
    bsdf_eval=make_bsdf_eval(f),
    light_visibility=make_visibility(f, q),
    wrapped=f.fuzz,
    ies=f.ies,
    bound=f.bound,
  )
  if not f.area_lights:
    return punctual
  return light_evaluators.make_area_evaluators(
    punctual=punctual,
    ies=f.ies,
    bound=f.bound,
    shadow_samples=f.shadow_samples,
    shadow_pass=make_shadow_pass(f, q),
    specular_curve=f.specular_curve,
    specular_weight=f.specular_weight,
    tinted=f.tinted_shadows,
    clearcoat=f.clearcoat,
    layers=(f.specular_anisotropy, f.fuzz, f.coat_normal, f.coat_layer),
    switches=toggles.switches_of(f),
  )


@specialize
def _camera_loop(
  f: Features, q: Queries, columns: sources.Columns, *, term_column: int, headlight: bool
) -> light_loop.LightLoop:
  """The camera rays' light loop of `f` (a `light_loop.make_light_loop` func), counted in
  `columns`: the cell's lights (the K strongest with lighting.max_exact), the shadow budget's K
  (0: every caster traced; `canonical.canonical` zeroes a budget >= nlight), the headlight entry
  (`headlight`); its form (`Features.light_forms`): the small table's straight-line loop or the
  source's loop."""
  if size := small_form(f.light_forms):

    def plain(counted: sources.Columns) -> sources.Source[int]:
      """Every light of the table, uncapped, counted in `counted`."""
      return cast(
        "sources.Source[int]",
        sources.make_source(
          nlight=f.nlight, culling=False, cap=0, has_spot=f.has_spot, ies=f.ies, columns=counted, bound=f.bound
        ),
      )

    source = plain(columns)
    # The small loop reads its source once per entry and pass: it counts the lights once itself.
    quiet = source if columns[0] < 0 else plain((-1, columns[1]))
    small = light_loop.Small(source=quiet, size=size, lights_column=columns[0])
  else:
    source, small = _source(f, f.light_cap, columns, sources.CAMERA_CAP), None
  return light_loop.make_light_loop(
    source=source,  # pyrefly: ignore[bad-argument-type]  # a union
    shadows=f.shadows,
    budget=f.budget,
    sun_only=False,
    evaluators=make_evaluators(f, q),  # pyrefly: ignore[bad-argument-type]  # a union
    term_column=term_column,
    run_counts=f.run_counts,
    run_budget=toggles.reads(f, toggles.RUN_COUNTS),
    switches=toggles.switches_of(f),
    headlight=headlight,
    small=small,
    bound=f.bound,
  )


@specialize
def _camera_lighting(f: Features, q: Queries, *, headlight: bool) -> DirectLighting:
  """The camera rays' lighting of `f` without the glossy-hit loop: it reads no reflection field of
  `f`, so the GI probe rays build it too (`probe.CAMERA_ONLY`). The caller selects `headlight`;
  only the camera entry point may read the headlight."""
  light_visibility = make_visibility(f, q)
  return DirectLighting(
    direct_lights=make_direct_lights(
      _camera_loop(
        f,
        q,
        _columns(f, counters.LIGHTS, counters.CHEAP),
        term_column=counters.LIGHT_TERMS if f.counters else -1,
        headlight=headlight,
      )
    ),
    hit_lights=None,
    light_visibility=light_visibility,
  )


# The GI probe hits' light loop (`make_scene_lights`): its own fixed policy, not the camera
# settings': the PROBE_LIGHT_CAP strongest lights of the cell shaded exactly (the rest aggregated),
# shadow rays for the PROBE_SHADOW_BUDGET strongest casters (the others take their visibility
# ratio, `shadow_budget`; deterministic: no sampled remainder). The probe irradiance is a
# low-frequency sum over many hits: the light loop was 90 % of a probe update (289 lights)
# and the bake depended on the camera settings (one probe kernel and one bake per light cap).
PROBE_LIGHT_CAP: Final = 4
PROBE_SHADOW_BUDGET: Final = 2


@specialize
def make_scene_lights(f: Features, q: Queries):
  """The probe rays' direct lighting (PROBE_LIGHT_CAP, PROBE_SHADOW_BUDGET), independent of all
  camera-only fields and sequences and of the scene's light count (the same cap and budget for
  any count, no small form: a constant count unrolled up to four light terms, each
  with its shadow rays, into the probe kernel: 128 s of NVRTC on a feature-dense room, 49 s
  without)."""
  return make_direct_lights(
    light_loop.make_light_loop(
      source=sources.make_source(  # pyrefly: ignore[bad-argument-type]  # a union
        nlight=f.nlight,
        culling=f.culling,
        cap=PROBE_LIGHT_CAP,
        has_spot=f.has_spot,
        ies=f.ies,
        bound=f.bound,
        columns=(-1, -1),
      ),
      shadows=f.shadows,
      budget=PROBE_SHADOW_BUDGET,
      sun_only=False,
      evaluators=make_evaluators(f, q),  # pyrefly: ignore[bad-argument-type]  # a union
    )
  )


@specialize
def make_lighting(f: Features, q: Queries):
  """Wires the direct-lighting funcs of the camera rays of `f` (see the module docstring)."""
  lighting = _camera_lighting(f, q, headlight=f.headlight)
  if not f.glossy:
    return lighting
  return dataclasses.replace(lighting, hit_lights=_hit_lights(f, q))


def _hit_lights(f: Features, q: Queries) -> light_loop.LightLoop:
  """The light loop at glossy-reflection hits: unshadowed lights (directional ones shadowed with
  reflections.shadows="sun"), no budget, capped by reflections.max_exact; the headlight as one
  more entry."""
  columns = _columns(f, counters.HIT_LIGHTS, counters.HIT_CHEAP)
  term_column = counters.HIT_LIGHT_TERMS if f.counters else -1
  if f.run_counts:
    # The reflection hits' shadows (off / sun) and cap at run time (`Toggles`).
    return light_loop.make_light_loop(
      source=_source(f, f.hit_light_cap, columns, sources.HIT_CAP),  # pyrefly: ignore[bad-argument-type]  # a union
      shadows=f.shadows,
      budget=0,
      sun_only=True,
      evaluators=make_evaluators(f, q),  # pyrefly: ignore[bad-argument-type]  # a union
      term_column=term_column,
      run_counts=True,
      hit=True,
      headlight=f.headlight,
      bound=f.bound,
    )
  return light_loop.make_light_loop(
    source=_source(f, f.hit_light_cap, columns, sources.HIT_CAP),  # pyrefly: ignore[bad-argument-type]  # a union
    shadows=f.reflection_shadows == REFLECTION_SHADOWS["sun"],
    budget=0,
    sun_only=True,
    evaluators=make_evaluators(f, q),  # pyrefly: ignore[bad-argument-type]  # a union
    term_column=term_column,
    headlight=f.headlight,
    bound=f.bound,
    switches=toggles.switches_of(f),
  )


def _columns(f: Features, lights_column: int, cheap_column: int) -> sources.Columns:
  """The counter columns of a light loop of `f` ((-1, -1): not counted)."""
  return (lights_column, cheap_column) if f.counters else (-1, -1)


def _source(f: Features, cap: int, columns: sources.Columns, toggle: int) -> sources.Sources:
  """The light source of a loop of `f` capped at `cap` lights, counted in `columns`; the cap read
  at run time is `toggle`'s (`counts`)."""
  return sources.make_source(
    nlight=f.nlight,
    culling=f.culling,
    cap=cap,
    has_spot=f.has_spot,
    ies=f.ies,
    bound=f.bound,
    columns=columns,
    counts=sources.counts_of(f, toggle),
  )
