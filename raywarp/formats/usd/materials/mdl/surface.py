"""A material distilled to `transmissive_pbr` -> the canonical OpenPBR `Material`.

The target is a stack of at most five lobes (MDL SDK `dist_rules_transmissive_pbr.mdltl`):

    [coat: custom_curve_layer(F0, F90, weight, ggx)]          coat_weight / coat_roughness
      weighted_layer(normal)                                  geometry_normal
        normalized_mix(dielectric: w_d, metal: w_m ggx(tint)) base_metalness
          custom_curve_layer(F0, F90, weight, ggx)            specular_weight / roughness / IOR
            normalized_mix(transmission: ggx transmit(tint),  transmission_weight / color
                           diffuse: diffuse_reflection(tint)) base_color

beside `thin_walled`, `ior`, the emission (radiant exitance: radiance = I / pi), the volume's
absorption and `geometry.cutout_opacity`. Lobes may be missing. A curve layer over another
glossy lobe is the coat; the base keeps its F0 against air under it (`COAT_GLTF`). A grazing
reflectivity that is not 1 (Unreal's `1 - roughness`) is the core's `SPECULAR_CURVE`. A fully
rough transmission is the distiller's diffuse transmission: OpenPBR's subsurface. A transmission
whose weight varies is Unreal's translucency: a blended opacity.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, final

from raywarp.formats.usd.materials.context import set_ext, with_flags
from raywarp.formats.usd.materials.mdl import detail, keys
from raywarp.formats.usd.materials.mdl.bake import Request
from raywarp.formats.usd.materials.mdl.bake_eval import node_key
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.values import RGB, UnsupportedError, comps
from raywarp.formats.usd.materials.mdl.writer import Field, Writer
from raywarp.scene.model.material import Material
from raywarp.scene.model.openpbr import anisotropic_roughness
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import AlphaMode, MaterialFlags, Slot

# The normal-incidence reflectance of IOR 1.5 (OpenPBR's default specular and coat).
F0_15: Final = 0.04
IOR_15: Final = 1.5
# The thickness a volume's absorption tints a thin transmission through [m] (a glass pane).
GLASS_DEPTH: Final = 0.01
# The share of the distiller's sheen GGX a textured base keeps when the sheen becomes fuzz (its
# maps are the distilled ones, the sheen's tint folded in; a constant base is read under the
# sheen instead: `_sheen_base`): half, the measured match to Omniverse RTX of vMaterials' cotton
# jersey before the core rendered fuzz (FLIP 0.123 against 0.185 with none kept and 0.227 with
# all kept). One asset: re-measure with the fuzz lobe.
FUZZ_SHARE: Final = 0.5
# A transmission at least this rough (GGX alpha) is diffuse (`diffuse_transmission_bsdf`: 1; its
# textured mixes average lower): no background shows through it.
DIFFUSE_ALPHA: Final = 0.8
# The depth a subsurface's volume scatters within [m]: light that does not stays the tint's.
SCATTER_DEPTH: Final = 0.1
_TOL: Final = 1e-6
# A constant tangent shorter than this in the tangent plane has no direction there.
_UNIT_XY: Final = 1e-3
SHEENS: Final = frozenset({"sheen_bsdf", "microflake_sheen_bsdf"})
# A sheen layer weight below this is no fuzz (a thousandth of the sheen: invisible; glTF PBR's
# graph carries an off sheen as a weightless layer).
MIN_FUZZ: Final = 1e-3
_MIXES: Final = ("df::normalized_mix", "df::clamped_mix", "df::unbounded_mix")
_GGX: Final = "df::microfacet_ggx_vcavities_bsdf"
_DIFFUSE: Final = "df::diffuse_reflection_bsdf"
_CURVE: Final = "df::custom_curve_layer"
_ADD_DETAIL: Final = "nvidia::distilling_support::add_detail_normal"
_COMBINE: Final = "nvidia::distilling_support::combine_normals"
# Calls holding distribution functions in their arguments (`lossy` reads only those).
_DISTRIBUTIONS: Final = ("df::", "material", "T[]", "operator?:")
# Distribution functions `transmissive_pbr` drops or approximates (reported per material).
LOSSY: Final = frozenset(
  f"df::{n}"
  for n in (
    "measured_bsdf sheen_bsdf microflake_sheen_bsdf thin_film backscattering_glossy_reflection_bsdf measured_curve_factor "
    "measured_factor measured_curve_layer color_measured_curve_layer ward_geisler_moroder_bsdf coat_absorption_factor "
    "chiang_hair_bsdf measured_edf spot_edf anisotropic_vdf fog_vdf"
  ).split()
)


@final
@dataclasses.dataclass
class Lobes:
  """The paths of a distilled surface's lobes (None: absent)."""

  coat: str | None = None
  specular: str | None = None
  metal: str | None = None
  metal_weight: str | None = None
  diffuse: str | None = None
  transmission: str | None = None
  transmission_weight: str | None = None
  normal: str | None = None
  unknown: list[str] = dataclasses.field(default_factory=list)


def lobes(dag: Dag, path: str, out: Lobes) -> Lobes:
  """`out` with the lobes below `path` (`surface.scattering`) recorded."""
  name, args = dag.node(path)
  if name == _CURVE:
    if out.coat is None and out.specular is None and _glossy_below(dag, args["base"]):
      out.coat = path
    elif out.specular is None:
      out.specular = path
    lobes(dag, args["base"], out)
  elif name == "df::weighted_layer":
    out.normal = out.normal or args["normal"]
    lobes(dag, args["layer"], out)
  elif name in _MIXES:
    for comp in dag.node(args["components"])[1].values():
      _component(dag, f"{comp}.component", f"{comp}.weight", out)
  elif name in (_GGX, _DIFFUSE):
    _component(dag, path, None, out)
  elif name is not None:
    out.unknown.append(name)
  return out


def _transmits(dag: Dag, ggx: str) -> bool:
  """Whether a GGX lobe transmits."""
  try:
    return dag.value(f"{ggx}.mode") == "scatter_transmit"
  except UnsupportedError:
    return False


def _component(dag: Dag, path: str, weight: str | None, out: Lobes) -> None:
  """A mix component (or a lone lobe): metal, transmission, diffuse, or a stack."""
  name = dag.node(path)[0]
  if name == _GGX and _transmits(dag, path):
    out.transmission, out.transmission_weight = path, weight
  elif name == _GGX:
    out.metal, out.metal_weight = path, weight
  elif name == _DIFFUSE:
    out.diffuse = path
  else:
    lobes(dag, path, out)


def _glossy_below(dag: Dag, path: str) -> bool:
  """Whether a reflecting glossy lobe or a curve layer sits below `path`."""
  todo = [path]
  while todo:
    p = todo.pop()
    name, args = dag.node(p)
    if name == _CURVE or (name == _GGX and not _transmits(dag, p)):
      return True
    if name is not None and name.startswith(("df::", "T[]")):
      todo.extend(args.values())
  return False


def lossy(dag: Dag) -> dict[str, str]:
  """The distribution functions of a compiled material that `transmissive_pbr` drops or
  approximates (measured BSDFs, sheen, thin film, ...), each with its first path."""
  found: dict[str, str] = {}
  try:
    for n, _, path in dag.walk(dag.material.get_body(), "", lambda n: n.startswith(_DISTRIBUTIONS)):
      if n in LOSSY:
        found.setdefault(n.removeprefix("df::"), path.removeprefix("."))
  except UnsupportedError:
    return {"material_too_large": ""}
  return found


def plan(w: Writer) -> list[Request]:
  """The bakes the mapping of the writer's material asks for (a pass that binds nothing and
  reports nothing)."""
  planner = Writer(dataclasses.replace(w.ctx, report=LoadReport()), w.dag, w.prim, w.bakes)
  planner.requests = []
  material(planner, sheen=None)
  return planner.requests


def material(w: Writer, *, sheen: Writer | None) -> Material:
  """The canonical material of the writer's distilled material (`sheen`: a writer over the
  compiled material when it has a sheen: OpenPBR's fuzz). The distiller turns a sheen into a
  tinted GGX beside the base: dropped (not a metal); the base keeps the distiller's share of it at
  `FUZZ_SHARE` (the fuzz lobe the core does not render reflects the rest)."""
  found = lobes(w.dag, "surface.scattering", Lobes())
  for name in sorted(set(found.unknown)):
    w.report(f"distilled_lobe.{name}", approx=True)
  share = 1.0
  recovered = False
  if sheen is not None and _fuzz(w, sheen):
    distilled = found.diffuse is not None
    if (distilled or _base_dropped(w)) and _sheen_base(w, sheen, allow_varying=not distilled):
      recovered = True
      found.metal = found.metal_weight = None
    elif distilled:
      # The distiller's sheen GGX beside a diffuse is the fuzz, not a metal.
      if found.metal_weight is not None:
        share = 1.0 - FUZZ_SHARE * w.mean(found.metal_weight, 1, 0.0)[0]
      found.metal = found.metal_weight = None
  found.normal = _base_normal(w, found)
  if not recovered:
    _base(w, found)
  if share != 1.0:
    w.set(base_color=tuple(share * c for c in w.mat.base_color))
    w.report("fuzz(the base under it at the distiller's share)", approx=True)
  if not recovered:
    _specular(w, found)
  _coat(w, found)
  _transmission(w, found)
  _normal(w, found.normal)
  _emission(w)
  _opacity(w)
  thin = w.const("thin_walled", 1)
  w.set(geometry_thin_walled=bool(thin and thin[0]))
  for note in sorted(w.dag.notes):
    w.report(note, approx=True)
  return w.mat


def _normal(w: Writer, normal: str | None) -> None:
  """The shading normal: a texture, a base and a detail map (`detail.split`: geometry_normal and
  geometry_detail_normal, each with its own UV transform), else a bake; a constant tilt away
  from +Z is reported (the core has no per-material constant normal)."""
  if normal is None or w.dag.node(normal)[0] in (None, "state::normal"):
    return
  tilt = w.const(normal, RGB)
  if tilt is not None:
    if abs(tilt[2] - 1.0) > _TOL:
      w.report("normal(a constant tilt: ignored)", approx=True)
    return
  maps = detail.split(w.dag, normal)
  if maps is not None and w.bind(Slot.GEOMETRY_NORMAL, maps[0]):
    if w.bind(Slot.GEOMETRY_NORMAL, maps[1], "geometry_detail_normal"):
      return
    w.report("detail_normal(unreadable: the maps' sum baked)", approx=True)
  w.field("geometry_normal", Slot.GEOMETRY_NORMAL, normal, kind="normal")


def _roughness(dag: Dag, ggx: str) -> str:
  """The GGX alpha of a lobe: the distiller's `roughness_info` wrappers skipped."""
  inner = f"{ggx}.roughness_u.s.r.roughness.s.roughness_u"
  return inner if dag.expr(inner) is not None else f"{ggx}.roughness_u"


def _base(w: Writer, found: Lobes) -> None:
  """Base color (the diffuse tint, else the metal's), metalness, roughness."""
  tinted = found.diffuse or found.metal
  if tinted is None:
    w.set(base_color=(0.0, 0.0, 0.0))
  else:
    w.field("base_color", Slot.BASE_COLOR, f"{tinted}.tint", kind="color")
  if found.diffuse is not None:
    _diffuse_roughness(w, found.diffuse)
  if found.metal is None or found.metal_weight is None:
    w.set(base_metalness=0.0 if found.metal is None else 1.0)
  else:
    w.field("base_metalness", Slot.BASE_METALNESS, found.metal_weight, kind="scalar")
  glossy = f"{found.specular}.layer" if found.specular is not None else found.metal
  if glossy is not None:
    w.field("specular_roughness", Slot.SPECULAR_ROUGHNESS, _roughness(w.dag, glossy), kind="roughness")
    _anisotropy(w, glossy)


def _diffuse_roughness(w: Writer, diffuse: str) -> None:
  """OpenPBR `base_diffuse_roughness`: the diffuse lobe's Oren-Nayar roughness
  (`df::diffuse_reflection_bsdf`'s; both in [0, 1], 0 the Lambert lobe; MaterialX's OpenPBR writes
  its input there), textured into `OpenPbrExtras.textures`."""
  path = f"{diffuse}.roughness"
  if w.dag.expr(path) is None:
    return
  value = w.const(path, 1)
  if value is not None and value[0] <= _TOL:
    return
  w.field("base_diffuse_roughness", Slot.SPECULAR_ROUGHNESS, path, kind="scalar", extra="base_diffuse_roughness")


def _anisotropic(w: Writer, ggx: str) -> tuple[str, str] | None:
  """(roughness_u, roughness_v) paths of a GGX lobe whose two alphas differ (None: isotropic)."""
  u = _roughness(w.dag, ggx)
  v = u.replace("roughness_u", "roughness_v")
  if w.dag.expr(v) is None:
    return None
  try:
    same = [node_key(c) for c in comps(w.dag.value(u))] == [node_key(c) for c in comps(w.dag.value(v))]
  except UnsupportedError:  # two expressions (OmniPBR's r^2 + aniso.x^2, r^2 + aniso.y^2): their values
    same = keys.key(w.dag, u) == keys.key(w.dag, v) or abs(w.mean(u, 1, 0.0)[0] - w.mean(v, 1, 0.0)[0]) <= _TOL
  return None if same else (u, v)


def _anisotropy(w: Writer, ggx: str) -> None:
  """A GGX lobe whose alpha_v (across `tangent_u`) differs from its alpha_u (along it): OpenPBR's
  roughness and anisotropy of the same two alphas (`openpbr.anisotropic_roughness`, exact for
  constants), stretched along tangent_u or, when alpha_v is the larger, a quarter turn from it;
  `ext.anisotropy_rotation` is tangent_u's angle from the UV tangent when it is a constant turn
  of it. A textured alpha is read as isotropic (reported)."""
  paths = _anisotropic(w, ggx)
  if paths is None:
    return
  au, av = w.const(paths[0], 1), w.const(paths[1], 1)
  if au is None or av is None:
    w.report("anisotropy(textured: as isotropic)", approx=True)
    return
  alpha_u, alpha_v = max(au[0], 0.0), max(av[0], 0.0)
  r, a = anisotropic_roughness(max(alpha_u, alpha_v), min(alpha_u, alpha_v))
  w.set(specular_roughness=r)
  w.set_extras(specular_roughness_anisotropy=a)
  # The distiller's wrappers hold the lobe's tangent_u beside its alphas.
  wrapped = paths[0].endswith(".s.roughness_u")
  angle = _tangent_angle(w, paths[0].rpartition(".")[0] + ".tangent" if wrapped else f"{ggx}.tangent_u")
  if angle is None:
    w.report("anisotropy.tangent_u(varying: the UV tangent)", approx=True)
    angle = 0.0
  rotation = math.fmod(angle + (0.5 * math.pi if alpha_v > alpha_u else 0.0) + math.tau, math.tau)
  if min(rotation, math.tau - rotation) > _TOL:
    w.mat = set_ext(w.mat, anisotropy_rotation=rotation)


def _tangent_angle(w: Writer, tangent: str) -> float | None:
  """The angle [rad] of a constant tangent in the baker's tangent frame (state::texture_tangent_u
  along +x, texture_tangent_v along +y), counter-clockwise from the UV tangent; None when it
  varies or leaves the tangent plane."""
  t = w.const(tangent, RGB)
  if t is None or math.hypot(t[0], t[1]) < _UNIT_XY or abs(t[2]) > _UNIT_XY:
    return None
  return math.atan2(t[1], t[0])


def _specular(w: Writer, found: Lobes) -> None:
  """The dielectric specular: weight x F0 against IOR 1.5 (the material's when it transmits; a
  textured weight's mean), the specular tint's mean in the weight (the core renders no tint), the
  Unreal curve when F90 is not 1."""
  if found.specular is None:
    w.set(specular_weight=0.0)
    if found.metal is not None:
      # The metal's tinted GGX stays without a dielectric layer: the core's approximation of it,
      # not scaled by the absent layer's weight.
      w.mat = with_flags(w.mat, MaterialFlags.DIELECTRIC_SPECULAR_LEVEL)
      w.report("metal_fresnel(Schlick instead of constant tint)", approx=True)
    return
  f0 = w.mean(f"{found.specular}.normal_reflectivity", 1, F0_15)[0]
  ior = w.const("ior", RGB) or (1.0,)
  eta = sum(ior) / len(ior) if found.transmission is not None and min(ior) > 1.0 else IOR_15
  w.set(specular_ior=eta)
  tint = w.mean(f"{found.specular}.layer.tint", RGB, 1.0)
  mean = sum(tint) / RGB or 1.0
  w.set(specular_color=tuple(c / mean for c in tint))
  w.field(  # a texture's mean: the core renders no specular_weight texture
    "specular_weight",
    None,
    f"{found.specular}.weight",
    kind="scalar",
    factor=mean * f0 / ((eta - 1.0) / (eta + 1.0)) ** 2,
  )
  gr = w.const(f"{found.specular}.grazing_reflectivity", 1)
  if (gr is None or abs(gr[0] - 1.0) > _TOL) and eta == IOR_15:  # the curve assumes IOR 1.5
    w.mat = with_flags(w.mat, MaterialFlags.SPECULAR_CURVE)
  else:
    # The weight is the dielectric layer's, not the metal's beside it. The core keeps the
    # dielectric grazing reflectance at 1: reported where the authored weight or curve moves it.
    w.mat = with_flags(w.mat, MaterialFlags.DIELECTRIC_SPECULAR_LEVEL)
    weight = w.const(f"{found.specular}.weight", 1)
    if gr is None or weight is None or abs(gr[0] * weight[0] - 1.0) > _TOL:
      w.report("specular_grazing(fixed at 1)", approx=True)


def _coat(w: Writer, found: Lobes) -> None:
  """The coat: weight x F0 / 0.04 (at most 1), roughness; glTF's (the base unchanged under it)."""
  if found.coat is None:
    return
  q = w.mean(f"{found.coat}.normal_reflectivity", 1, F0_15)[0] / F0_15
  w.set(coat_weight=min(q * w.mean(f"{found.coat}.weight", 1, 0.0)[0], 1.0), coat_ior=IOR_15)
  tint = w.mean(f"{found.coat}.layer.tint", RGB, 1.0)
  if any(abs(c - 1.0) > _TOL for c in tint):
    w.set_extras(coat_color=tint)
  w.field("coat_roughness", None, _roughness(w.dag, f"{found.coat}.layer"), kind="roughness")
  w.mat = with_flags(w.mat, MaterialFlags.COAT_GLTF)
  if _anisotropic(w, f"{found.coat}.layer") is not None:
    w.report("coat_anisotropy(as isotropic: roughness_u)", approx=True)
  _coat_normal(w, f"{found.coat}.normal", found.normal)
  if w.mat.extras is not None:  # the base unchanged under a glTF coat, in OpenPBR terms
    w.set_extras(coat_darkening=0.0)


def _flat(w: Writer, normal: str | None) -> bool:
  """Whether the normal at `normal` is the geometry's (absent, `state::normal`, or a constant
  +Z in the baker's tangent frame: the distiller's `add_detail_normal(state::normal(), n)` of an
  unperturbed n)."""
  if normal is None or w.dag.node(normal)[0] in (None, "state::normal"):
    return True
  tilt = w.const(normal, RGB)
  return tilt is not None and all(abs(t - z) < _TOL for t, z in zip(tilt, (0.0, 0.0, 1.0), strict=True))


def _base_normal(w: Writer, found: Lobes) -> str | None:
  """The base's own normal under the distiller's wrappers: `add_detail_normal(nd, n)` with one
  operand the geometry's is the other, and below a coat `combine_normals(1 - c, n_base, c, n_coat)`
  (the distiller's `weighted_layer` rule pulling the base's normal toward the layer's by its
  weight, its only normal for both) is n_base: the coat keeps its own normal (`_coat_normal`)."""
  path = found.normal
  while path is not None:
    name, args = w.dag.node(path)
    if name == _ADD_DETAIL and _flat(w, args["n"]):
      path = args["nd"]
    elif name == _ADD_DETAIL and _flat(w, args["nd"]):
      path = args["n"]
    elif name == _COMBINE and found.coat is not None and _layer_blend(w, args, f"{found.coat}.normal"):
      path = args["n1"]
    else:
      return path
  return path


def _layer_blend(w: Writer, args: dict[str, str], coat_normal: str) -> bool:
  """Whether combine_normals(w1, n1, w2, n2) is the `weighted_layer` rule's blend toward the
  coat's normal: w1 = 1 - w2 (as an expression) and n2 the coat's normal."""
  name, minus = w.dag.node(args["w1"])
  one = w.const(minus["x"], 1) if name == "operator-" and "y" in minus else None
  if one is None or abs(one[0] - 1.0) > _TOL or w.dag.hash(minus["y"]) != w.dag.hash(args["w2"]):
    return False
  return (_flat(w, args["n2"]) and _flat(w, coat_normal)) or _same(w.dag, args["n2"], coat_normal)


def _same(dag: Dag, a: str, b: str) -> bool:
  """Whether the expressions at `a` and `b` are equal (the SDK's hash, else their values)."""
  if dag.hash(a) == dag.hash(b):
    return True
  try:
    return [node_key(c) for c in comps(dag.value(a, direction=True))] == [
      node_key(c) for c in comps(dag.value(b, direction=True))
    ]
  except UnsupportedError:
    return False


def _coat_normal(w: Writer, normal: str, base: str | None) -> None:
  """The coat's normal (the curve layer's `normal`): the geometry's (no flag: the unmapped
  normal), the base's expression (COAT_BASE_NORMAL: OmniPBR_ClearCoat's unflattened coat), else
  its own map as `geometry_coat_normal` (a texture, or a bake; a constant tilt is reported)."""
  if _flat(w, normal):
    return
  if base is not None and not _flat(w, base) and _same(w.dag, normal, base):
    w.mat = with_flags(w.mat, MaterialFlags.COAT_BASE_NORMAL)
    return
  if w.const(normal, RGB) is not None:
    w.report("coat_normal(a constant tilt: ignored)", approx=True)
    return
  w.field("geometry_coat_normal", Slot.GEOMETRY_NORMAL, normal, kind="normal", extra="geometry_coat_normal")


def _base_dropped(w: Writer) -> bool:
  """Whether the distilled material is a layer over no base (`bsdf()`): what the distillation
  makes of a full-weight sheen (MaterialX's OpenPBR fuzz at weight 1 and a white color)."""
  if w.dag.node("surface.scattering")[0] != "df::weighted_layer":
    return False
  e = w.dag.expr("surface.scattering.base")
  if e is None or e.get_kind().name != "EK_CONSTANT":
    return False
  return e.get_interface(w.dag.api.bindings.IExpression_constant).get_value().get_kind().name == "VK_INVALID_DF"


def _sheen_base(w: Writer, sheen: Writer, *, allow_varying: bool) -> bool:
  """The base under the compiled material's first sheen: the diffuse tint and the dielectric
  layer's roughness and level (means), the sheen's GGX dropped (the core renders the fuzz). Read
  where the distillation dropped the base (`_base_dropped`, `allow_varying`: a varying tint as
  its mean) or folded the sheen's tint into it (a distilled diffuse beside the sheen's GGX: only
  for a constant tint, a varying one keeps the distilled maps). False, nothing set, when that
  base has no diffuse lobe or (not `allow_varying`) a varying tint."""
  path = next(p for n, p in lossy(sheen.dag).items() if n in SHEENS)
  parent, _, member = path.rpartition(".")
  if member == "layer" and "base" in sheen.dag.node(parent)[1]:
    under = f"{parent}.base"  # a weighted layer over its base
  elif "multiscatter" in sheen.dag.node(path)[1]:
    under = f"{path}.multiscatter"  # the compiler's fold of a full-weight layer: the sheen's own base
  else:
    return False
  while sheen.dag.node(under)[0] == "df::thin_film":  # an inactive thin film around the base
    under = f"{under}.base"
  base = lobes(sheen.dag, under, Lobes())
  if base.diffuse is None or (not allow_varying and sheen.const(f"{base.diffuse}.tint", RGB) is None):
    return False
  w.set(base_color=sheen.mean(f"{base.diffuse}.tint", RGB, 0.8), base_metalness=0.0)
  glossy = f"{base.specular}.layer" if base.specular is not None else None
  if glossy is None:
    w.set(specular_weight=0.0)
  else:
    w.set(specular_roughness=math.sqrt(max(sheen.mean(_roughness(sheen.dag, glossy), 1, 0.09)[0], 0.0)))
    f0 = sheen.mean(f"{base.specular}.normal_reflectivity", 1, F0_15)[0] * sheen.mean(f"{base.specular}.weight", 1, 1.0)[0]
    w.set(specular_weight=f0 / F0_15, specular_ior=IOR_15)
  w.report("fuzz(the base read under the sheen)", approx=True)
  return True


def _fuzz(w: Writer, sheen: Writer) -> bool:
  """OpenPBR fuzz from the compiled material's first sheen: its enclosing layer's weight, its
  tint and roughness (means when they vary); False, nothing set, for a sheen below
  `MIN_FUZZ`."""
  path = next(p for n, p in lossy(sheen.dag).items() if n in SHEENS)
  parent, _, member = path.rpartition(".")
  args = sheen.dag.node(parent)[1]
  weight = sheen.mean(args["weight"], 1, 0.0)[0] if member == "layer" and "weight" in args else 1.0
  if weight < MIN_FUZZ:
    return False
  fuzz: dict[str, Field] = {"fuzz_weight": weight, "fuzz_color": sheen.mean(f"{path}.tint", RGB, 1.0)}
  if "roughness" in sheen.dag.node(path)[1]:
    fuzz["fuzz_roughness"] = sheen.mean(f"{path}.roughness", 1, 0.5)[0]
  w.set_extras(**fuzz)
  return True


def _transmission(w: Writer, found: Lobes) -> None:
  """Transmission: subsurface (fully rough), a blended opacity (a varying weight), else
  specular transmission tinted by the volume's absorption through `GLASS_DEPTH`."""
  ggx, weight = found.transmission, found.transmission_weight
  if ggx is None:
    return
  alpha = w.mean(_roughness(w.dag, ggx), 1, 0.0)
  if alpha[0] >= DIFFUSE_ALPHA:
    tint, s = w.mean(f"{ggx}.tint", RGB, 1.0), (1.0,) if weight is None else w.mean(weight, 1, 0.0)
    w.set_extras(subsurface_weight=s[0], subsurface_color=tuple(t * a for t, a in zip(tint, _volume_albedo(w), strict=True)))
    return
  if weight is not None and w.const(weight, 1) is None:
    _see_through(w, weight)
    return
  w.set(transmission_weight=1.0 if weight is None else w.mean(weight, 1, 0.0)[0])
  sigma = w.const("volume.absorption_coefficient", RGB) or (0.0,) * RGB
  tint = w.mean(f"{ggx}.tint", RGB, 1.0)
  w.set(transmission_color=tuple(c * math.exp(-a * GLASS_DEPTH) for c, a in zip(tint, sigma, strict=True)))


def _volume_albedo(w: Writer) -> tuple[float, ...]:
  """The colour a scattering volume gives a subsurface (OmniSurface skin: the colour is in the
  coefficients, the transmission tint white): van de Hulst's multiple-scattering albedo
  Rd(s) = (1 - s)(1 - 0.139 s) / (1 + 1.17 s), s = sqrt(1 - sigma_s / sigma_t), weighted by the
  share of light that scatters within `SCATTER_DEPTH` (1 - exp(-sigma_t d): an optically thin
  volume, vMaterials' translucent plastics, keeps its tint); 1 without a constant volume."""
  sigma_s = w.const("volume.scattering_coefficient", RGB)
  sigma_a = w.const("volume.absorption_coefficient", RGB) or (0.0,) * RGB
  if sigma_s is None or min(sigma_s) <= 0.0:
    return (1.0,) * RGB
  out: list[float] = []
  for x, a in zip(sigma_s, sigma_a, strict=True):
    t = x + max(a, 0.0)
    r = math.sqrt(max(1.0 - x / t, 0.0))
    thick = 1.0 - math.exp(-t * SCATTER_DEPTH)
    out.append(thick * (1.0 - r) * (1.0 - 0.139 * r) / (1.0 + 1.17 * r) + 1.0 - thick)
  return tuple(out)


def _see_through(w: Writer, weight: str) -> None:
  """A varying transmission weight as the blended opacity 1 - weight (Unreal's translucency:
  the core renders no transmission texture)."""
  if w.field("geometry_opacity", Slot.GEOMETRY_OPACITY, weight, kind="scalar") is None:
    return  # no bake: opaque
  ref = w.mat.texture(Slot.GEOMETRY_OPACITY)
  if ref is None:
    w.set(geometry_opacity=1.0 - w.mat.geometry_opacity)
  else:
    s, b = ref.scale, ref.bias
    w.mat = w.mat.with_texture(
      Slot.GEOMETRY_OPACITY,
      dataclasses.replace(ref, scale=(-s[0], -s[1], -s[2], -s[3]), bias=(1.0 - b[0], 1.0 - b[1], 1.0 - b[2], 1.0 - b[3])),
    )
  w.mat = set_ext(w.mat, alpha_mode=AlphaMode.BLEND)
  w.report("transmission_weight(varying: as the opacity 1 - weight)", approx=True)


def _emission(w: Writer) -> None:
  """Emission: the intensity as radiant exitance, radiance = I / pi [nits]."""
  path = "surface.emission.intensity"
  value = w.const(path, RGB)
  if value is not None and max(value) <= 0.0:
    return
  mode = w.dag.value("surface.emission.mode")
  if mode != "intensity_radiant_exitance":
    w.report(f"emission.{mode}(as radiant exitance)", approx=True)
  if value is not None:
    w.set(emission_color=tuple(x / max(value) for x in value), emission_luminance=max(value) / math.pi)
    return
  gain = w.field("emission_color", Slot.EMISSION_COLOR, path, kind="emission")
  if gain is not None:
    w.set(emission_luminance=gain / math.pi)


def _opacity(w: Writer) -> None:
  """geometry.cutout_opacity: a constant below 1 or a texture blends; a binary texture (a
  thresholded opacity) cuts out at 0.5. Over a see-through opacity (`_see_through`) a constant
  multiplies it; a varying one has no second slot (reported)."""
  value = w.const("geometry.cutout_opacity", 1)
  if value is not None and value[0] >= 1.0 - _TOL:
    return
  if w.mat.ext.alpha_mode != AlphaMode.OPAQUE:
    _scale_opacity(w, value)
    return
  w.field("geometry_opacity", Slot.GEOMETRY_OPACITY, "geometry.cutout_opacity", kind="scalar")
  textured = w.mat.texture(Slot.GEOMETRY_OPACITY) is not None
  if textured or w.mat.geometry_opacity < 1.0 - _TOL:
    w.mat = set_ext(w.mat, alpha_mode=AlphaMode.MASK if w.binary_opacity and textured else AlphaMode.BLEND, alpha_cutoff=0.5)


def _scale_opacity(w: Writer, value: tuple[float, ...] | None) -> None:
  """Multiplies the blended opacity already set by a constant cutout opacity `value` (None: a
  varying one, which the single opacity slot cannot hold: reported)."""
  if value is None:
    w.report("cutout_opacity(varying, over a varying transmission: dropped)", approx=True)
    return
  c = max(value[0], 0.0)
  ref = w.mat.texture(Slot.GEOMETRY_OPACITY)
  if ref is None:
    w.set(geometry_opacity=w.mat.geometry_opacity * c)
    return
  s, b = ref.scale, ref.bias
  w.mat = w.mat.with_texture(
    Slot.GEOMETRY_OPACITY,
    dataclasses.replace(ref, scale=(s[0] * c, s[1] * c, s[2] * c, s[3] * c), bias=(b[0] * c, b[1] * c, b[2] * c, b[3] * c)),
  )
