"""The fixed kernel tiers (`core/build/variants/tiers.py`): each recipe renders with its tier's
kernels, a change that keeps a tier's compile-time settings or turns its switches off keeps them (no
compile) and renders as the configuration's own specialised kernels, and a configuration outside
every tier says why in `explain()`."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import DEFAULT_EV100, recipe
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp.core.build.variants import tiers
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import RUNTIME_LIGHTS
from raywarp.core.pipeline.platform import tuning

# Five shadow-casting lights (spot, point, directional with a sun disc) over a glossy floor: the
# light caps (4) and shadow budgets (2 / 4) of the default and `FAST` select among them.
XML = """
<mujoco>
  <asset><material name="glossy" roughness="0.15" reflectance="0.3"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <light pos="1.5 -1 2" dir="-0.5 0.3 -1" castshadow="true" cutoff="40" diffuse="0.8 0.5 0.3"/>
    <light pos="-1.5 1 2" dir="0.4 -0.2 -1" castshadow="true" diffuse="0.3 0.5 0.8"/>
    <light pos="2 2 1.5" dir="-1 -1 -1" castshadow="true" diffuse="0.4 0.4 0.4"/>
    <light directional="true" dir="0.3 0.2 -1" castshadow="true" diffuse="0.5 0.5 0.45"/>
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1" material="glossy"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
"""

OPTS = {"nworld": 2, "cam_res": (40, 30), "outputs": rw.Outputs.of("rgb", "hdr")}
# Recipe (`conftest.recipe`) -> the tier serving it.
TIER_OF = {
  "unlit_fast": "unlit",
  "unlit": "unlit",  # isotropic filtering: the unlit tier itself (anisotropy > 1: unlit_sampled)
  "fast": "fast",
  "default": "default",
  "quality": "default_ss2_counts",
  "rtx_look": "default_counts",
  "reference": "default_ss2_counts",
}


@pytest.fixture(autouse=True)
def exact_classes(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every class set prebuilt (`tiers.cover` the scene's own classes): the tests check what each
  class compiles, not which prebuilt superset serves it (`test_cover_*` in test_tiers.py)."""
  monkeypatch.setattr(tiers, "cover", lambda used, **_: used)


# A small probe bake (run-time GI fields: no kernel changes): the recipes' full one (up to 32768
# probes x 256 rays) is minutes of CPU time per renderer, and test processes running side by
# side were killed in it (out of memory or time); the tiers' kernels are the subject here.
_SMALL_GI = {"gi.max_probes": 64, "gi.rays": 32}


def _recipe(name: str, **changes: object) -> dict[str, object]:
  """Recipe `name`'s settings as "group.field" keys (outputs: `OPTS`'; the kernel build the
  default's), the reference exposure (the automatic one adapts across frames), a small probe
  bake (`_SMALL_GI`), plus `changes`."""
  flat = recipe(name).flat()
  out = {k: v for k, v in flat.items() if not k.startswith("outputs.") and k != "performance.kernels"}
  return {**out, "exposure.mode": "manual", "exposure.ev100": DEFAULT_EV100, **_SMALL_GI, **changes}


def _choice(s) -> tiers.Choice:
  """The renderer's tier choice (`Renderer.explain` reads it)."""
  choice = s.r._choice  # the routing under test
  assert choice is not None
  return choice


@pytest.mark.parametrize("name", list(TIER_OF))
def test_every_recipe_renders_with_its_tier(scene, name: str) -> None:
  """A recipe's configuration is its tier's (GI off in "unlit_fast" is moot: unlit)."""
  s = scene(XML, **OPTS, **_recipe(name))
  choice = _choice(s)
  assert choice.tier == TIER_OF[name]
  assert f"tier {TIER_OF[name]!r}" in str(s.r.explain())


def test_kernels_setting_routes_prebuilt_or_scene(scene) -> None:
  """`performance.kernels`: "prebuilt" (the default) serves the tier's build resolved without the
  scene (run-time light count), "scene" the settings' own build specialised on the scene (its
  light count compiled in), named by `Renderer.explain`."""
  prebuilt = scene(XML, **OPTS, **_recipe("fast"))
  assert prebuilt.r.settings.performance.kernels == "prebuilt"
  assert _choice(prebuilt).tier == "fast"
  assert prebuilt.r.context().features.nlight == RUNTIME_LIGHTS
  own = scene(XML, **OPTS, **_recipe("fast", **{"performance.kernels": "scene"}))
  assert own.r._choice is None  # the routing under test
  assert own.r.context().features == own.r.features
  assert own.r.features.nlight not in (0, RUNTIME_LIGHTS)
  assert "performance.kernels='scene'" in str(own.r.explain())


def test_cover_serves_the_cheapest_prebuilt_superset(monkeypatch: pytest.MonkeyPatch) -> None:
  """`tiers.cover`: a prebuilt set equal to the scene's classes serves it; else the superset adding
  the least `CLASS_COST`; a scene no set covers gets every class (always prebuilt). A visibility or
  deformable scene takes the sets prebuilt with them only (`class_sets.VISIBLE`, `FLEX`)."""
  monkeypatch.undo()  # the real router (not this module's exact one)
  sets = (frozenset({"maps", "base_map"}), frozenset({"maps", "base_map", "env"}), frozenset({"maps", "ies"}))
  monkeypatch.setattr(tiers.class_sets, "PREBUILT", sets)
  assert tiers.cover(frozenset({"maps", "base_map"})) == sets[0]
  assert tiers.cover(frozenset({"maps"})) == sets[0]  # + base map (3) beats + IES (12)
  assert tiers.cover(frozenset({"env"})) == sets[1]
  assert tiers.cover(frozenset({"glass", "surface"})) == frozenset(tiers.CLASSES)
  monkeypatch.setattr(tiers.class_sets, "VISIBLE", sets[1:2])
  monkeypatch.setattr(tiers.class_sets, "FLEX", sets[2:])
  assert tiers.cover(frozenset({"maps"}), visibility=True) == sets[1]
  assert tiers.cover(frozenset({"maps"}), flex=True) == sets[2]
  assert tiers.cover(frozenset({"maps"}), visibility=True, flex=True) == frozenset(tiers.CLASSES)


# Changes from a recipe a tier serves, with the tier: switches off (`tiers.SWITCHES`, reflections,
# GI), the compile-time settings read at run time (`tiers.SETTING_FIELDS`: shadows, their sampling,
# the fog, anisotropic taps), the configuration's scene axes (`tiers.Axes`: the light grid, the
# translucent class) and the variants (`tiers.VARIANTS`).
SWITCHED = [
  ("fast", {"gi.enabled": False}, "fast"),
  ("rtx_look", {"reflections.enabled": False, "gi.enabled": False}, "default_counts"),
  ("default", {"reflections.enabled": False}, "default"),
  ("default", {"gi.enabled": False, "reflections.enabled": False}, "default"),
  ("quality", {"gi.enabled": False}, "default_ss2_counts"),
  ("fast", {"surfaces.backface_culling": False}, "fast"),
  ("fast", {"lighting.light_grid": True}, "fast"),
  ("default", {"lighting.light_grid": False}, "default"),
  ("fast", {"transparency.enabled": True}, "fast"),
  ("fast", {"lighting.area_lights": "ltc"}, "fast_ltc"),
  ("fast", {"shadows.enabled": False}, "fast"),
  ("fast", {"shadows.samples": 2}, "fast_sampled"),
  ("default", {"shadows.samples": 4}, "default_sampled"),
  ("fast", {"atmosphere.start": 1.0, "atmosphere.end": 6.0}, "fast"),
  ("default", {"atmosphere.haze": 0.3}, "default"),
  ("fast", {"surfaces.anisotropy": 8}, "fast_sampled"),
  # Counts below the tier's, read at run time within its compiled ones.
  ("fast", {"shadows.max_casters": 1}, "fast"),
  ("default", {"lighting.max_exact": 2}, "default"),
  ("default", {"reflections.max_exact": 1}, "default"),
  ("default", {"shadows.max_casters": 3}, "default_casters"),
  ("default", {"shadows.max_casters": 1}, "default_casters"),
]


@pytest.mark.parametrize(("name", "changes", "tier"), SWITCHED)
def test_switched_configuration_renders_as_its_own_kernels(scene, name: str, changes: dict[str, object], tier: str) -> None:
  """A tier serving a changed configuration renders the image of the configuration's own
  specialised kernels (8-bit display within one level, linear HDR within float reordering)."""
  tiered = scene(XML, **OPTS, **_recipe(name, **changes))
  assert _choice(tiered).tier == tier
  got = tiered.render()
  own = scene(XML, **OPTS, **_recipe(name, **changes), **{"performance.kernels": "scene"}).render()
  assert own.r.context().features == own.r.features
  np.testing.assert_allclose(got.rgb(), own.rgb(), atol=1.0 / 255)
  np.testing.assert_allclose(got.hdr(), own.hdr(), rtol=1e-4, atol=1e-6)


def test_changes_within_the_tiers_compile_nothing(scene) -> None:
  """After each tier rendered once, switching recipes, turning switches off and on and changing
  run-time settings loads no Warp module the tiers' first frames did not (none compiles, even on
  an empty kernel cache), and the last frame renders as a renderer created with its settings."""
  s = scene(XML, **OPTS, **_recipe("fast", **{"gi.enabled": False}))
  for name in TIER_OF:
    s.r.update_settings(**_recipe(name, **{"gi.enabled": False}))
    s.render()
  wp.synchronize()
  changes = [
    _recipe("default", **{"gi.enabled": False, "reflections.enabled": False}),
    _recipe("fast", **{"gi.enabled": False, "lighting.fill": 0.1}),
    _recipe("fast", **{"gi.enabled": False, "post.tone_mapper": "aces", "exposure.mode": "manual", "exposure.ev100": 9.0}),
    _recipe("quality", **{"gi.enabled": False, "reflections.max_roughness": 0.3, "lighting.fill": 0.05}),
    _recipe("unlit_fast"),
    _recipe("fast", **{"gi.enabled": False, "sensor.noise": True, "sensor.vignette": (-0.3, 0.0, 0.0)}),
    _recipe("reference", **{"gi.enabled": False, "shadows.samples": 4}),
    _recipe(
      "default", **{"gi.enabled": False, "reflections.enabled": False, "exposure.mode": "manual", "exposure.ev100": 12.0}
    ),
  ]
  with module_loads() as compiled:
    for change in changes:
      s.r.update_settings(**change)
      s.render()
      wp.synchronize()
      assert _choice(s).tier is not None, change
  assert compiled == [], compiled
  expected = scene(XML, **OPTS, **changes[-1]).render()
  np.testing.assert_allclose(s.rgb(), expected.rgb(), atol=1.0 / 255)


def test_outside_the_tiers_names_what_differs(scene) -> None:
  """A build no tier has (the light-cost counters: a profiling build,
  `core.pipeline.platform.tuning`) renders with its own kernels, and `explain()` names the tier it
  is nearest to and the field that differs; a light cap above the default's (8), and a shadow budget
  below it with 2 x 2 supersampling (3), land on the counted variants (run-time counts)."""
  with tuning.override(counters=True):
    s = scene(XML, **OPTS, **_recipe("default"))
  choice = _choice(s)
  assert choice.tier is None
  assert choice.nearest == "default"
  assert choice.outside == ("counters",)
  assert s.r.context().features == s.r.features
  assert "own (outside the tiers; nearest 'default' differs in counters)" in str(s.r.explain())
  s.r.update_settings(**_recipe("default", **{"lighting.max_exact": 8}))
  assert _choice(s).tier == "default_counts"
  s.r.update_settings(**_recipe("default", **{"shadows.max_casters": 3, "aa.spp": 2}))
  assert _choice(s).tier is not None


def test_switches_are_read_at_run_time_only_in_tier_builds(scene) -> None:
  """A software tier build reads exactly its compiled-in switches at run time (those of its classes,
  shadows, the fog, culling, vertex normals and specular AA, the counts below its own;
  GI is a settings axis of the build, not a switch); the configuration's own Features (host
  decisions) have none."""
  s = scene(XML, **OPTS, **_recipe("default", **{"gi.enabled": False}))
  build, effective = s.r.context().features, s.r.features
  assert not build.gi and not effective.gi  # GI off: the tier's GI-off build
  # An opaque scene without an environment (no `tiers.CLASSES` surface or env class) compiles
  # neither the translucent-shadow switch nor the environment's.
  classes = dict(zip(tiers.CLASSES, tiers.feature_class(build)[: len(tiers.CLASSES)], strict=True))
  assert not classes["surface"] and not classes["glass"] and not classes["env"]
  settings = toggles.SHADOWS | toggles.FOG | toggles.LIGHT_PANELS | toggles.RUN_COUNTS
  assert build.switches == settings | toggles.BACKFACE_CULLING
  assert effective.switches == 0


def test_tiers_the_renderer_cannot_take_are_skipped(scene) -> None:
  """With per-camera resolutions, a configuration outside the tiers still renders with its own
  kernels."""
  cams = [rw.CameraSpec(fovy=60.0, resolution=(40, 30)), rw.CameraSpec(fovy=60.0, resolution=(20, 16))]
  # Outside every tier: the light-cost counters (a shadow budget is a counted variant's now).
  with tuning.override(counters=True):
    s = scene(XML, nworld=1, cameras=cams, outputs=OPTS["outputs"], **_recipe("fast"))
    s.render()
  choice = _choice(s)
  assert choice.tier is None
  assert choice.nearest == "fast"


# A cloth over a plane (MuJoCo flexcomp: a deformable surface, `ShapeType.FLEX`), lit and seen.
_FLEX_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -2.5 1.5" xyaxes="1 0 0 0 0.5 0.9"/>
    <geom type="plane" size="2 2 0.1"/>
    <geom type="box" pos="0.6 0 0.2" size="0.2 0.2 0.2"/>
    <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.2 0.2 0.2" pos="0 0 0.6" radius="0.02" dim="2">
      <edge equality="true"/><contact contype="0" conaffinity="0"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(("xml", "tiered"), [(_FLEX_XML, False), (XML, True)])
def test_optix_tiers_do_not_serve_deformables(scene, monkeypatch: pytest.MonkeyPatch, xml: str, tiered: bool) -> None:
  """On OptiX a scene with deformables resolves to its own scene build (the tier programs miss
  their hits: `tiers.UNSERVED_SHAPES`), the same Features as `performance.kernels="scene"`, and
  `explain` says why; a scene without them keeps its tier. Resolved on the CPU with the OptiX
  backend admitted (routing only: nothing is built or compiled)."""
  from raywarp._base.types import ShapeType
  from raywarp.core.build.variants import explain
  from raywarp.core.build.variants.resolve import resolve_features
  from raywarp.core.pipeline.platform.device import device_caps
  from raywarp.core.pipeline.renderer import shadow_casters
  from raywarp.core.trace.optix_backend import OptixBackend

  s = scene(xml, **OPTS, **_recipe("default", **{"performance.traversal": "software"}))
  r = s.r
  monkeypatch.setattr(OptixBackend, "device_reason", staticmethod(lambda _caps: None))
  monkeypatch.setattr(OptixBackend, "scene_reason", staticmethod(lambda _scene: None))
  casters = int(shadow_casters(r._scene_slots(None), r.per_world_fields).sum())  # the renderer's own count

  def resolved(kernels: str):
    """The resolution of the renderer's settings on OptiX with `kernels`."""
    settings = r.settings.replace(**{"performance.traversal": "optix", "performance.kernels": kernels})
    return resolve_features(settings, r._caps, r._per_world_decl, device_caps("cpu"), casters=casters)

  routed, own = resolved("prebuilt"), resolved("scene")
  assert (int(ShapeType.FLEX) in own.features.shape_types) is not tiered
  if tiered:
    assert routed.choice is not None and routed.choice.tier is not None and routed.features.scene_free
    return
  assert routed.choice is not None and routed.choice.tier is None
  assert routed.features == own.features and not routed.features.scene_free
  assert "shape_types are not served" in explain.kernels_line(routed.choice, shaded=True)
