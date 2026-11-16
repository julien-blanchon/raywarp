"""Zero compiles by construction (CPU): every recipe with every single compile-relevant value of
the input space (`tools/prebuilt/zero_space.py`: each compile-time settings field at every choice
and threshold, each scene value the resolution reads: capabilities, lights, environment,
deformables, material inputs, per-world declarations, layouts, cameras) routes to a tier build
whose OptiX PTX key the fill compiles (`tools/prebuilt/zero.py`: the fill's own enumeration).
The configurations that still compile are `zero_holes.json`, each with its reason: a new hole
fails, and so does a closed one still listed (update the file with `python -m
tools.prebuilt.zero holes --write`). The lens cameras included: their programs are prebuilt
beside the pinhole build."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Final

import pytest

import raywarp as rw

zero = pytest.importorskip("tools.prebuilt.zero")
zero_space = pytest.importorskip("tools.prebuilt.zero_space")

HOLES = Path(zero.__file__).with_name("zero_holes.json")


@pytest.fixture(scope="module")
def keys() -> frozenset[str]:
  """The OptiX fill's PTX keys (`zero.fill_keys`)."""
  return zero.fill_keys()


def test_settings_space_is_every_compile_time_field() -> None:
  """`zero_space.SETTINGS` holds every compile-time settings field (`RenderSettings.fields_table`,
  the outputs as one axis) but the excluded ones: a new compile-time setting fails here until it
  has a domain."""
  from raywarp.settings import RenderSettings

  compiled = {f.name for f in RenderSettings.fields_table() if not f.runtime}
  compiled = {"outputs" if n.startswith("outputs.") else n for n in compiled}
  assert compiled - zero_space.EXCLUDED <= set(zero_space.SETTINGS), sorted(compiled - set(zero_space.SETTINGS))


def test_scene_space_is_every_declared_field_the_resolution_reads() -> None:
  """`zero.DECLARED` holds every per-world field `resolve.py` tests for (`"..." in declared`): a
  new declaration the kernels depend on fails here until it is an axis."""
  import ast

  from raywarp.core.build.variants import resolve

  tree = ast.parse(Path(resolve.__file__).read_text())
  read = {
    n.left.value
    for n in ast.walk(tree)
    if isinstance(n, ast.Compare)
    and isinstance(n.left, ast.Constant)
    and any(ast.unparse(c) == "declared" for c in n.comparators)
  }
  assert read <= set(zero.DECLARED), sorted(read - set(zero.DECLARED), key=str)


def test_optix_fill_keys_ignore_geometry_and_layout(keys: frozenset[str]) -> None:
  """On OptiX the geometry, the sharing layout, the planes and the deformables are bound
  (`tiers.BOUND_FREE`): the base scenes of one layout give every fill key (`zero.base_points`)."""
  from raywarp.core.build.variants import tiers

  points = (
    zero.Scene(),
    zero.Scene(world=True),
    zero.Scene(world=True, shared=False),
    zero.Scene(compact=False),
    zero.Scene(visible=True),
    zero.Scene(planes=1),
    zero.Scene(flex=2, flex_material=True),
  )
  for tier in ("unlit", "fast", "default"):
    routes = {zero.route(zero.Probe(repr(p), p), tiers.TIERS[tier]).key for p in points}
    assert len(routes) == 1 and routes <= keys, (tier, routes)


def _hole_record(h: object) -> dict[str, object]:
  """A hole as `zero_holes.json` lists it."""
  d = dataclasses.asdict(h)  # pyrefly: ignore[bad-argument-type]
  return {"base": d["base"], "settings": {k: repr(v) for k, v in d["settings"]}, "scene": dict(d["scene"])}


@pytest.mark.slow  # 1 443 routes of ~80 ms and the fill's 14 400 points (~95 s): ~4 min on 1 core
@pytest.mark.parametrize("base", zero_space.BASES)
def test_every_single_value_routes_to_a_fill_point(keys: frozenset[str], base: str) -> None:
  """Every single settings / scene value from recipe `base` (`tools/recipes.py`) routes to a tier
  build whose PTX key is a fill key, but the listed holes (`zero_holes.json`), and none lands on a
  build costlier than it asks for (`zero_space.cost_anomaly`: a counted / sampled variant for
  counts within its tier family's)."""
  del keys  # the fixture warms `zero.fill_keys` once
  tally = zero_space.walk(1, (base,))
  holes = [_hole_record(h) for h in tally.holes]
  listed = [
    {k: h[k] for k in ("base", "settings", "scene")} for h in json.loads(HOLES.read_text())["holes"] if h["base"] == base
  ]
  new = [h for h in holes if h not in listed]
  closed = [h for h in listed if h not in holes]
  assert not new, f"configurations that compile (a hole: close it, or list it with its reason): {new}"
  assert not closed, f"closed holes still listed in {HOLES.name}: {closed}"
  assert not tally.anomalies, [(dict(a.settings), a.tier, a.outside) for a in tally.anomalies]


# Cut-outs on every geometry kind (spheres / capsules, ellipsoids / cylinders, all of them, the
# analytic planes): the alpha test runs in each kind's hit programs and queries, so one PTX serves
# every one (their wrapped alpha test was another PTX per build, which no fill point held).
CUT_KINDS: Final = (("shapes", "round"), ("shapes", "custom"), ("shapes", "all"), ("layout", "plane"))


@pytest.mark.parametrize("kind", CUT_KINDS, ids=["_".join(k) for k in CUT_KINDS])
@pytest.mark.parametrize("base", zero_space.BASES)
def test_cut_outs_on_any_geometry_route_to_a_fill_point(keys: frozenset[str], base: str, kind: tuple[str, str]) -> None:
  """Recipe `base` on a scene with alpha cut-outs and geometry `kind` routes to a tier build whose
  PTX key is a fill key."""
  from tools import recipes

  axes = zero_space.scene_axes()
  axis, value = kind
  point = zero_space.scene_of({"cap.alpha_cutout": axes["cap.alpha_cutout"]["on"], axis: axes[axis][value]})
  r = zero.route(zero.Probe(repr(point), point), recipes.settings(base))
  assert r.tier is not None and r.key in keys, (base, kind, r.tier, r.outside)


# The documented exception: a lens camera (the scene axes of `zero_space.scene_axes`) renders with
# its pinhole camera's tier build plus its model's render program (`programs.build_lens`), which
# the fill leaves out by default (`fill.fill(lens=)`): compiled once per build on first use or by
# `Renderer.warm`.
LENS: Final = (("camera", "distortion"), ("camera", "bundle"), ("declared.camera.distortion", "on"))


# The other documented exceptions (`fill.NICHE_TIERS`, `NICHE_FORMS`: out of the bundle by user
# decision): the motion-vector output sets of `zero_space.SETTINGS` and the routes to the niche
# default variants.
MOTION: Final = frozenset({"'rgb+motion'", "'depth+motion'", "'all'"})


def _documented(h: dict[str, object]) -> bool:
  """Whether listed hole `h` is a documented exception: a lens camera alone, a motion output, or a
  route to a niche tier whose key the fill leaves out."""
  from tools.prebuilt import fill

  settings, scene, why = h["settings"], h["scene"], h["why"]
  assert isinstance(settings, dict) and isinstance(scene, dict), h
  lens = not settings and len(scene) == 1 and next(iter(scene.items())) in LENS
  motion = settings.get("outputs") in MOTION
  niche = isinstance(why, str) and any(why == f"tier {t}: key not filled" for t in fill.NICHE_TIERS)
  return lens or motion or niche


def test_the_only_listed_holes_are_documented_exceptions() -> None:
  """Every listed hole is a lens camera, a motion output or a niche default variant, nothing
  else."""
  for h in json.loads(HOLES.read_text())["holes"]:
    assert _documented(h), h


@pytest.mark.parametrize("lens", LENS)
def test_a_lens_camera_compiles_only_its_lens_program(keys: frozenset[str], lens: tuple[str, str]) -> None:
  """A lens camera on the fast field set takes the pinhole camera's tier build, a fill key; only
  its lens model's program is not bundled."""
  axis, value = lens
  point = zero_space.scene_of({axis: zero_space.scene_axes()[axis][value]})
  r = zero.route(zero.Probe(repr(point), point), rw.RenderSettings().replace(rw.FAST))
  render, _, lens_key = (r.key or "").partition("+")
  assert r.tier is not None and render in keys and lens_key and r.key not in keys


# Configurations a tier build serves through a run-time bit (each was a hole): the translucent
# layers turned off on translucent content (`toggles.TRANSPARENCY`), declared material selection
# with live UV tables (`toggles.MAP_LIVE_UV`, OptiX `tiers.BOUND_FREE`) and with prepared vertex
# colours (one source: the shading bits).
BITS: Final = (
  ("default", {"transparency.enabled": False}, (("cap.has_alpha", "on"), ("cap.alpha_cutout", "on"), ("tex.opacity", "on"))),
  ("fast", {"transparency.enabled": False, "reflections.enabled": True}, (("cap.has_alpha", "on"),)),
  ("quality", {"transparency.enabled": False}, (("cap.transmissive", "on"),)),
  ("reference", {"transparency.enabled": False}, (("declared.alpha", "on"),)),
  ("rtx_look", {"transparency.enabled": False}, (("declared.instance.shading", "on"),)),
  ("fast", {}, (("declared.instance.material", "on"), ("material_uv", "on"))),
  ("quality", {}, (("declared.instance.material", "on"), ("material_uv", "on"))),
  ("unlit", {}, (("declared.instance.material", "on"), ("prepared", "rich"))),
  ("reference", {}, (("declared.instance.material", "on"), ("prepared", "rich"))),
  # The alpha test off as data (`frame_structs.alpha_cutoffs`): textures off on a cut-out scene
  # whose tier build compiles the test (the translucent surface class, a complete variant).
  ("default", {"surfaces.textures": False}, (("cap.alpha_cutout", "on"), ("cap.has_alpha", "on"))),
  ("quality", {"surfaces.textures": False}, (("cap.alpha_cutout", "on"),)),
  ("rtx_look", {"surfaces.textures": False}, (("declared.material.alpha_cutoff", "on"),)),
)


@pytest.mark.parametrize(("base", "changes", "scene"), BITS)
def test_run_time_bits_route_to_a_fill_point(
  keys: frozenset[str], base: str, changes: dict[str, object], scene: tuple[tuple[str, str], ...]
) -> None:
  """Each configuration of `BITS` routes to a tier build whose PTX key the fill compiles."""
  from tools import recipes

  axes = zero_space.scene_axes()
  point = zero_space.scene_of({a: axes[a][v] for a, v in scene})
  r = zero.route(zero.Probe(repr(point), point), zero_space.settings_of(recipes.settings(base), changes))
  assert r.tier is not None and r.key in keys, (r.tier, r.outside)


@pytest.mark.parametrize("base", ["default", "fast", "unlit", "quality", "reference", "rtx_look"])
def test_unlit_materials_route_to_a_fill_point(keys: frozenset[str], base: str) -> None:
  """A scene with unlit materials (`Capabilities.unlit_emission`, `ShadingFlag.UNLIT`) routes to a
  filled tier build: the materials class compiles them in, read from its field bit
  (`toggles.MAT_UNLIT`)."""
  from tools import recipes

  point = zero.Scene(caps=zero.capabilities(unlit_emission=True))
  r = zero.route(zero.Probe(repr(point), point), recipes.settings(base))
  assert r.tier is not None and r.key in keys, (r.tier, r.outside)
