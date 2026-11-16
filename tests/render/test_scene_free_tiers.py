"""Scene-free tier builds (`core/build/variants/tiers.py` SCENE_FREE): every scene renders a recipe
with the same tier kernels (the same Features: the Warp module hashes and the OptiX PTX key hold no
scene input), and those kernels render each scene as the scene's own specialised kernels do."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import DEFAULT_EV100, default_settings, recipe, required

import raywarp as rw
from raywarp.core.build.variants import tiers
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import RUNTIME_LIGHTS, SMALL_LIGHTS_MAX, Features, small_form
from raywarp.core.kernels.abi.output_layout import OUT_HDR_SQRT
from raywarp.core.trace.rays import primary
from raywarp.scene.model import lights

_GEOMS = """
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1" material="glossy"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
"""


def _xml(lights: str) -> str:
  """One room (a glossy floor, a sphere, a box) lit by `lights`."""
  return f"""
<mujoco>
  <asset><material name="glossy" roughness="0.15" reflectance="0.3"/></asset>
  <worldbody>{lights}{_GEOMS}</worldbody>
</mujoco>
"""


def _ring(n: int) -> str:
  """`n` shadow-casting point lights on a ring (more than the light grid's minimum at 12)."""
  out = []
  for i in range(n):
    a = 2.0 * np.pi * i / n
    x, y = 2.0 * np.cos(a), 2.0 * np.sin(a)
    out.append(f'<light pos="{x:.3f} {y:.3f} 2" dir="{-x:.3f} {-y:.3f} -2" castshadow="true" diffuse="0.2 0.2 0.2"/>')
  return "".join(out)


# The light sets: none (the headlight alone), one sun, one spot, five mixed casters (above the
# shadow budgets, at the light caps), twelve points (the light grid's "auto" threshold).
SCENES = {
  "none": _xml(""),
  "sun": _xml('<light directional="true" dir="0.3 0.2 -1" castshadow="true" diffuse="0.8 0.8 0.75"/>'),
  "spot": _xml('<light pos="1.5 -1 2" dir="-0.5 0.3 -1" castshadow="true" cutoff="40" diffuse="0.8 0.5 0.3"/>'),
  "mixed": _xml(
    '<light pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>'
    '<light pos="1.5 -1 2" dir="-0.5 0.3 -1" castshadow="true" cutoff="40" diffuse="0.8 0.5 0.3"/>'
    '<light pos="-1.5 1 2" dir="0.4 -0.2 -1" castshadow="false" diffuse="0.3 0.5 0.8"/>'
    '<light pos="2 2 1.5" dir="-1 -1 -1" castshadow="true" diffuse="0.4 0.4 0.4"/>'
    '<light directional="true" dir="0.3 0.2 -1" castshadow="true" diffuse="0.5 0.5 0.45"/>'
  ),
  "ring12": _xml(_ring(12)),
}
_SUN = '<light directional="true" dir="0.3 0.2 -1" castshadow="true" diffuse="0.8 0.8 0.75"/>'
_POINT = '<light pos="1.5 -1 2" dir="-0.5 0.3 -1" castshadow="true" diffuse="0.8 0.5 0.3"/>'
# Materials and meshes the light sets do not have: a checker base-color texture, a matte (no
# mirror) material, vertex-normal meshes.
MATERIAL_SCENES = {
  "textured": f"""
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.2 0.3 0.6" width="64" height="64"/>
    <material name="floor" texture="checker" texrepeat="4 4" roughness="0.6"/>
  </asset>
  <worldbody>{_SUN}{_POINT}
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" material="floor"/>
    <geom type="capsule" pos="0 0 0.5" size="0.3 0.3" rgba="0.8 0.3 0.2 1"/>
  </worldbody>
</mujoco>
""",
  "mesh": f"""
<mujoco>
  <asset>
    <mesh name="wedge" vertex="0 0 0  1 0 0  0 1 0  0 0 1  1 1 0.5"/>
    <material name="matte" roughness="0.8"/>
  </asset>
  <worldbody>{_POINT}
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.6 0.6 0.6 1" material="matte"/>
    <geom type="mesh" mesh="wedge" pos="-0.4 -0.3 0" rgba="0.3 0.7 0.3 1"/>
    <geom type="cylinder" pos="0.9 0.3 0.3" size="0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
""",
  # Triangle kinds only (a plane and meshes): the build without the "primitives" class.
  "triangles": f"""
<mujoco>
  <asset>
    <mesh name="wedge" vertex="0 0 0  1 0 0  0 1 0  0 0 1  1 1 0.5"/>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.2 0.3 0.6" width="64" height="64"/>
    <material name="floor" texture="checker" texrepeat="4 4" roughness="0.6"/>
  </asset>
  <worldbody>{_SUN}{_POINT}
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" material="floor"/>
    <geom type="mesh" mesh="wedge" pos="-0.4 -0.3 0" rgba="0.3 0.7 0.3 1"/>
    <geom type="mesh" mesh="wedge" pos="0.6 0.2 0" euler="0 0 90" rgba="0.8 0.3 0.2 1"/>
  </worldbody>
</mujoco>
""",
}
# A sky (zenith blue, horizon white) lighting the sun scene: the environment's light, background
# and sky occlusion, run-time switches of the scene-free builds.
_SKY = np.concatenate([np.linspace([0.3, 0.5, 1.0], [1.0, 1.0, 1.0], 8), np.full((8, 3), 0.2)])[:, None, :].repeat(32, 1)
SCENE_KW: dict[str, dict[str, object]] = {"env": {"envmap": _SKY.astype(np.float32), "envmap_intensity": 1.0}}
# A translucent box (rgba alpha) in the mixed light set: the layer loop.
_GLASS = SCENES["mixed"].replace(
  '<geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>',
  '<geom type="box" pos="0.2 -0.8 0.3" size="0.3 0.05 0.3" rgba="0.2 0.4 0.9 0.4"/>',
)
# The sun scene without MuJoCo's headlight (on in every other MJCF scene).
_NO_HEADLIGHT = SCENES["sun"].replace("<mujoco>", '<mujoco><visual><headlight active="0"/></visual>', 1)
ALL = {**SCENES, **MATERIAL_SCENES, "env": SCENES["sun"], "glass": _GLASS, "no_headlight": _NO_HEADLIGHT}
OPTS = {"nworld": 2, "cam_res": (40, 30), "outputs": rw.Outputs.of("rgb", "hdr")}
# The recipes served by a class-split tier build (the quality recipe takes a complete variant,
# `tiers.COMPLETE`: every class compiled).
RECIPES = ("fast", "default")
# The settings' own build, specialised on the scene: what a tier build is compared with.
SCENE_BUILD: dict[str, object] = {"performance.kernels": "scene"}


@pytest.fixture(autouse=True)
def exact_classes(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every class set prebuilt (`tiers.cover` the scene's own classes): the tests check what each
  class compiles, not which prebuilt superset serves it (`test_cover_*` in test_tiers.py)."""
  monkeypatch.setattr(tiers, "cover", lambda used, **_: used)


def _recipe(name: str) -> dict[str, object]:
  """The recipe's settings as "group.field" keys (outputs: `OPTS`'; the kernel build the
  default's), GI off (its bake is no kernel input of these fields), the reference exposure."""
  flat = recipe(name).flat()
  out = {k: v for k, v in flat.items() if not k.startswith("outputs.") and k != "performance.kernels"}
  return {**out, "gi.enabled": False, "exposure.mode": "manual", "exposure.ev100": DEFAULT_EV100}


@pytest.mark.parametrize("name", RECIPES)
def test_every_scene_takes_its_class_s_tier_build(scene, name: str) -> None:
  """Each scene renders the recipe with its tier, and the tier's build (the kernels' Features,
  i.e. their cache keys) is one per feature class and light-loop form (`tiers.feature_class`:
  the opaque scenes of one loop form share one, the translucent / layered ones another) for every
  light set, material, shape, environment and translucent surface, but in the fields still taken
  from the scene (`tiers.SCENE_AXES`); the light count is a run-time bound."""
  builds: dict[tuple[bool | int, ...], dict[str, Features]] = {}
  for key, xml in ALL.items():
    s = scene(xml, **OPTS, **SCENE_KW.get(key, {}), **_recipe(name))
    assert s.r._choice is not None and s.r._choice.tier is not None, key  # the routing under test
    build = s.r.context().features
    builds.setdefault(tiers.feature_class(build), {})[key] = build
  # Loop forms: the small ones (1 to SMALL_LIGHTS_MAX entries), the general loop with or without
  # the grid.
  assert len(builds) <= (SMALL_LIGHTS_MAX + 2) * 2 ** len(tiers.CLASSES)
  by_class: dict[tuple[bool | int, ...], list[str]] = {}
  for k, same in builds.items():
    by_class.setdefault(k[: len(tiers.CLASSES)], []).extend(same)
  # The MJCF fixtures have MuJoCo's headlight (a class of its own: "headlight") and, but one, the
  # analytic shapes ("primitives"); the material classes follow each scene's own kernels
  # (`test_maps_and_materials_follow_their_classes`).
  of = {key: _served(build) for same in builds.values() for key, build in same.items()}
  assert of["none"]["headlight"] and of["none"]["primitives"]
  assert not any(of["none"][c] for c in ("surface", "coat", "layers", "glass", "maps"))
  # The light count at run time, but a small form's: its entries.
  assert all(b.nlight in (RUNTIME_LIGHTS, small_form(b.light_forms)) for same in builds.values() for b in same.values())
  assert of["glass"]["surface"]  # the translucent box: the surface class
  assert of["triangles"]["headlight"] and not of["triangles"]["primitives"] and not of["triangles"]["surface"]
  for same in builds.values():
    first = next(iter(same.values()))
    for key, build in same.items():
      # The word: each scene's served classes (a value of the build, `tiers.word`).
      differ = [n for n in vars(first) if getattr(first, n) != getattr(build, n) and n not in (*tiers.SCENE_AXES, "word")]
      assert not differ, (key, differ)


@pytest.mark.parametrize("key", list(ALL))
@pytest.mark.parametrize("name", RECIPES)
def test_scene_free_build_renders_as_the_scene_s_own(scene, name: str, key: str) -> None:
  """The scene-free tier build renders each scene as the kernels specialised on it (8-bit display
  within one level, linear HDR within float reordering)."""
  _same_as_own(scene, key, _recipe(name))


@pytest.mark.parametrize("key", ["sun", "env"])
def test_gi_on_and_off_render_as_own(scene, key: str) -> None:
  """With GI on and off (two builds of the tier: GI is a settings axis, not a switch), with and
  without a lighting environment (sky occlusion in place of GI), fast renders each as the
  scene's own kernels."""
  _same_as_own(scene, key, {**_recipe("fast"), "gi.enabled": True})
  _same_as_own(scene, key, _recipe("fast"))


def _same_as_own(scene, key: str, settings: dict[str, object]) -> None:
  """Scene `key` with `settings`: the tier's scene-free build renders as its own kernels."""
  xml, kw = ALL[key], SCENE_KW.get(key, {})
  tiered = scene(xml, **OPTS, **kw, **settings).render()
  assert tiered.r._choice is not None and tiered.r._choice.tier is not None  # the routing under test
  own = scene(xml, **OPTS, **kw, **settings, **SCENE_BUILD).render()
  assert own.r.context().features == own.r.features
  assert own.r.features.nlight != RUNTIME_LIGHTS
  np.testing.assert_allclose(tiered.rgb(), own.rgb(), atol=1.0 / 255)
  np.testing.assert_allclose(tiered.hdr(), own.hdr(), rtol=1e-4, atol=1e-6)


def _coated() -> rw.Scene:
  """An opaque coated (clear coat) sphere in front of a grey wall, lit by a sun: the lobes class
  alone (no emissive panel: capped panel reflections are another test's,
  `test_panel_reflections`)."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  paint = rw.Material(base_color=(0.6, 0.1, 0.1), coat_weight=1.0, coat_roughness=0.1)
  b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=b.add_material(paint), xform=np.c_[np.eye(3), [0.3, 0.0, 0.0]])
  tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]], np.float32)
  b.add_light(lights.Light(kind=lights.LightKind.DISTANT, intensity=3.0, xform=np.c_[tilt, np.zeros(3)]))
  return b.build()


def _slab(*, thin: bool) -> rw.Scene:
  """An emissive wall behind a glass slab with a transmission depth: solid (a medium row:
  refraction and Beer-Lambert absorption), or thin-walled (none: a transmissive pane)."""
  b = rw.SceneBuilder()
  wall = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=100.0, emission_color=(1.0, 1.0, 1.0)))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=wall, xform=np.c_[np.eye(3), [0.0, 0.0, -2.0]])
  glass = rw.Material(
    transmission_weight=1.0,
    transmission_color=(1.0, 0.5, 0.25),
    geometry_thin_walled=thin,
    extras=rw.scene.OpenPbrExtras(transmission_depth=0.2),
  )
  b.add_instance("box", size=(1.0, 1.0, 0.2), material=b.add_material(glass), xform=np.c_[np.eye(3), [0.3, 0.0, 0.0]])
  return b.build()


def _render_slab(*, thin: bool, settings: dict[str, object], scene: rw.Scene | None = None) -> tuple[Features, np.ndarray]:
  """The slab (`_slab`; or `scene`) seen at an angle: (its kernels' Features, linear HDR)."""
  pose = np.array([[[1.0, 0.0, 2.8, 0.0, 0.17, 0.0, 0.985]]], np.float32)
  with rw.Renderer(
    rw.upload(_slab(thin=thin) if scene is None else scene, "cpu"),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=30.0)],
    resolution=(32, 24),
    settings=default_settings(**settings, outputs=rw.Outputs.of("hdr")),
  ) as r:
    r.camera_pose.assign(pose)
    r.render()
    return r.context().features, np.asarray(required(r.outputs.hdr).numpy())


@pytest.mark.parametrize("name", RECIPES)
def test_solid_glass_takes_the_glass_class(name: str) -> None:
  """A solid glass slab renders with its tier's translucent + glass build (`tiers.CLASSES`
  "glass": refraction, Beer-Lambert), the same slab thin-walled with the translucent build
  (refraction compiled out); each as the scene's own kernels."""
  settings = _recipe(name)
  solid, solid_hdr = _render_slab(thin=False, settings=settings)
  thin, thin_hdr = _render_slab(thin=True, settings=settings)
  solid_classes, thin_classes = _served(solid), _served(thin)
  lobes = ("coat", "layers")
  assert solid_classes["surface"] and solid_classes["glass"] and solid.refraction
  assert thin_classes["surface"] and not thin_classes["glass"]
  assert not any(solid_classes[k] or thin_classes[k] for k in lobes)
  assert np.abs(solid_hdr - thin_hdr).max() > 0.05  # the solid slab bends and absorbs
  for build, hdr, thin_walled in ((solid, solid_hdr, False), (thin, thin_hdr, True)):
    own, own_hdr = _render_slab(thin=thin_walled, settings={**settings, **SCENE_BUILD})
    assert own.refraction == _served(build)["glass"]
    np.testing.assert_allclose(hdr, own_hdr, rtol=1e-4, atol=1e-6)


_MAP_SLOTS = ("normal_map", "orm_map", "emissive_map", "opacity_map", "diffuse_roughness_map")
_MATERIAL_FIELDS = ("reflection", "diffuse_roughness", "specular_weight", "vertex_colors")


@pytest.mark.parametrize("key", ["sun", "textured", "mesh"])
def test_maps_and_materials_follow_their_classes(scene, key: str) -> None:
  """The texture slots but base color ("maps") and the material capabilities ("materials": EON,
  specular weight, vertex colors, mirror reflectance; the specular curve is in every build)
  are served by a tier build only when the scene's own kernels have one of them (`tiers.CLASSES`),
  all of the class then (`_served`: a bound class by its word bit, the build compiling it in; a
  build without a word compiles only the served ones). W6-S: compiled in and on unused they cost
  the coffee maker 28 % (maps) and anymal 12 % (materials) on RTX."""
  build = scene(ALL[key], **OPTS, **_recipe("fast")).r.context().features
  own = scene(ALL[key], **OPTS, **_recipe("fast"), **SCENE_BUILD).r.features
  classes = _served(build)
  assert classes["maps"] == any(getattr(own, n) for n in _MAP_SLOTS)
  assert classes["materials"] == any(getattr(own, n) for n in _MATERIAL_FIELDS)
  for name in (*_MAP_SLOTS, *_MATERIAL_FIELDS):
    assert getattr(build, name) >= getattr(own, name), name
  if not build.bound & toggles.DEFINED:
    for served, fields in ((classes["maps"], _MAP_SLOTS), (classes["materials"], _MATERIAL_FIELDS)):
      assert served or not any(getattr(build, n) for n in fields)


@pytest.mark.parametrize("name", RECIPES)
def test_coated_opaque_scene_takes_the_coat_class(name: str) -> None:
  """An opaque scene with a clear coat compiles the coat class (`tiers.CLASSES` "coat"; the layered
  lobes only when its own kernels have them, a class of their own: "layers"), no translucent
  layers, cut-outs or glass, and renders as its own kernels."""
  settings = _recipe(name)
  build, hdr = _render_slab(thin=False, settings=settings, scene=_coated())
  classes = _served(build)
  assert classes["coat"] and not classes["surface"] and not classes["glass"]
  assert build.clearcoat and not build.transparency
  own, own_hdr = _render_slab(thin=False, settings={**settings, **SCENE_BUILD}, scene=_coated())
  assert own.clearcoat
  assert classes["layers"] == own.layered
  np.testing.assert_allclose(hdr, own_hdr, rtol=1e-4, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="zero recompile not finished: tiers.SCENE_AXES not empty yet (lane W6-A)")
def test_no_scene_input_is_left_in_any_kernel_key() -> None:
  """User requirement (2026-11-04): no scene count, id or table in any compile key; a tier's build
  is chosen from a small fixed set of feature classes (`tiers.CLASSES`), all prebuilt. Every tier
  is resolved so, and no Features field is left to the scene."""
  assert tiers.SCENE_AXES == ()
  assert len(tiers.CLASSES) <= 11


def _served(build: Features) -> dict[str, bool]:
  """The feature classes build `build` serves its scene with (`tiers.CLASSES` name -> on): a bound
  class (`tiers.bound_classes`) by its bit of the build's word where its modules define it (a
  software tier build compiles every bound class in, `toggles.DEFINED`, its word the served set),
  a code-split class (or a build without a word) by what it compiles (`tiers.feature_class`)."""
  compiled = dict(zip(tiers.CLASSES, tiers.feature_class(build)[: len(tiers.CLASSES)], strict=True))
  if not build.bound & toggles.DEFINED:
    return {c: bool(on) for c, on in compiled.items()}
  bound = tiers.bound_classes(build.traversal)
  return {c: bool(build.word & tiers.CLASS_MASKS[c]) if c in bound else bool(on) for c, on in compiled.items()}


# The counts a bits-only build reads at run time (`tiers.SCENE_FREE`'s light-count fields).
_COUNTS = frozenset({"nlight", "light_cap", "hit_light_cap", "budget"})


@pytest.mark.parametrize("key", ["sun", "mixed", "ring12", "textured", "glass", "triangles"])
@pytest.mark.parametrize("name", ["fast", "default"])
def test_bits_only_build_keeps_the_scene_s_bits(scene, monkeypatch, name: str, key: str) -> None:
  """A build with only the counts at run time (the light count, caps and budget) keeps every
  feature-presence bit and the light-loop form of the scene's own build, and renders as it."""
  monkeypatch.setattr(tiers, "SCENE_FREE", _COUNTS)
  bits = scene(ALL[key], **OPTS, **SCENE_KW.get(key, {}), **_recipe(name)).render()
  build = bits.r.context().features
  own = scene(ALL[key], **OPTS, **SCENE_KW.get(key, {}), **_recipe(name), **SCENE_BUILD).render()
  differ = {n for n in vars(build) if getattr(build, n) != getattr(own.r.features, n)}
  # The tier's run-time switches (GI, environment, ...) and the fields they make moot (the probe
  # rays' layer count) are a tier's, not the scene's bits; the shape types are the scene's
  # geometry-kinds class (`tiers.geometry_kinds`: meshes in every build, so the mesh launch shape,
  # vertex normals and ordered traversal of a primitives-only scene).
  # The bound bits and the word: properties of the build (`tiers.differences` compares neither).
  allowed = {
    "switches",
    "bound",
    "word",
    "scene_free",
    "max_layers",
    "shape_types",
    "block_dim",
    "vertex_normals",
    "ordered_traversal",
  }
  assert differ <= _COUNTS | set(tiers.SWITCHES) | set(tiers.BOUND_FIELDS) | allowed, differ
  # The loop form is the scene's (a small form's entries its light count).
  assert _served(build) == _served(own.r.features)
  assert build.culling == own.r.features.culling
  assert build.light_forms == own.r.features.light_forms
  np.testing.assert_allclose(bits.rgb(), own.rgb(), atol=1.0 / 255)
  np.testing.assert_allclose(bits.hdr(), own.hdr(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["fast", "default"])
def test_manual_and_automatic_exposure_share_the_tier_build(scene, name: str) -> None:
  """One tier build serves automatic and manual exposure (the deferred display's "hdr" store is
  the output word's: `OUT_HDR_SQRT` under automatic exposure, nothing under a manual one): the same
  build (on the software path another word of it, `Features.out_word`), and a manual exposure
  renders as the scene's own kernels with its internal "hdr" buffer a one-pixel placeholder."""
  opts = {**OPTS, "outputs": rw.Outputs.of("rgb")}
  auto = scene(SCENES["mixed"], **opts, **{**_recipe(name), "exposure.mode": "auto"}).render()
  manual = scene(SCENES["mixed"], **opts, **_recipe(name)).render()
  build = manual.r.context().features
  assert dataclasses.replace(build, out_word=0) == dataclasses.replace(auto.r.context().features, out_word=0)
  assert auto.r._buffers.layout.bits & OUT_HDR_SQRT and not manual.r._buffers.layout.has("hdr")
  assert manual.r._buffers.flat["hdr"].shape == (OPTS["nworld"], 1)
  assert not manual.r.settings.deferred_display()
  own = scene(SCENES["mixed"], **opts, **_recipe(name), **SCENE_BUILD).render()
  np.testing.assert_allclose(manual.rgb(), own.rgb(), atol=1.0 / 255)


# Three lights, two of them casting shadows: fast's shadow budget (2 casters) covers every
# caster, so the scene's own build takes the 3-light small loop without a budget.
_THREE = _xml(
  '<light pos="0 0 3" dir="0 0 -1" castshadow="true" diffuse="0.5 0.5 0.5"/>'
  '<light pos="1.5 -1 2" dir="-0.5 0.3 -1" castshadow="true" diffuse="0.5 0.4 0.3"/>'
  '<light pos="-1.5 1 2" dir="0.4 -0.2 -1" castshadow="false" diffuse="0.3 0.4 0.5"/>'
)


def test_small_light_scene_keeps_its_shadow_budget(scene) -> None:
  """A scene whose lights take the small loop keeps its own shadow budget in the tier build: the
  small table of its light count, no budget (resolved without the scene, fast's
  budget of 2 sized the table below the 3 lights and sent them to the general loop with budgeted
  shadows: 2.3x the scene build's time on a 3-light room), rendering as the scene's own kernels."""
  settings = _recipe("fast")
  tiered = scene(_THREE, **OPTS, **settings).render()
  own = scene(_THREE, **OPTS, **settings, **SCENE_BUILD).render()
  build, mine = tiered.r.context().features, own.r.features
  assert tiers.small_loop(mine.light_forms) and mine.budget == 0
  assert build.light_forms == mine.light_forms and build.nlight == mine.nlight and build.budget == 0
  np.testing.assert_allclose(tiered.rgb(), own.rgb(), atol=1.0 / 255)


def _bent_coat() -> rw.Scene:
  """A coated, emission-textured quad (the "maps" class: a tier build compiles the normal maps and
  their horizon fade in) without a normal map (its own kernels have no horizon fade), turned 20
  degrees away from the camera, whose vertex normals lean 35 degrees further away (within
  `uv.vertex_normal`'s limit): the reflection about them dips below the face (where a fade would
  darken the coat's environment light), under a sun and a constant sky."""
  b = rw.SceneBuilder()
  texture = b.add_texture(np.full((8, 8, 4), 255, np.uint8), colorspace="g22_rec709")
  lean, turn = np.radians(35.0), np.radians(20.0)
  mesh = b.add_mesh(
    np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32),
    np.array([[0, 1, 2], [0, 2, 3]], np.int32),
    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
    normals=np.array([[-np.sin(lean), 0.0, np.cos(lean)]], np.float32),
    normal_idx=np.zeros((2, 3), np.int32),
  )
  paint = rw.Material(
    base_color=(0.6, 0.1, 0.1),
    coat_weight=1.0,
    coat_roughness=0.1,
    emission_luminance=1.0,
    textures=((rw.scene.Slot.EMISSION_COLOR, rw.scene.TextureRef(texture)),),
  )
  away = np.array([[np.cos(turn), 0.0, -np.sin(turn)], [0.0, 1.0, 0.0], [np.sin(turn), 0.0, np.cos(turn)]], np.float32)
  b.add_instance("mesh", mesh=mesh, material=b.add_material(paint), xform=np.c_[away, np.zeros(3)])
  tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]], np.float32)
  b.add_light(lights.Light(kind=lights.LightKind.DISTANT, intensity=3.0, xform=np.c_[tilt, np.zeros(3)]))
  b.add_environment(lights.EnvironmentLight(intensity=1.0))
  return b.build()


@pytest.mark.parametrize("name", RECIPES)
def test_horizon_fade_follows_the_build_word(name: str) -> None:
  """A tier build compiling the horizon fade in (`toggles.SCENE_HORIZON_FADE`) applies it only where
  its word has it: a scene without normal maps renders as its own kernels, the coat's and the
  reflection hits' ambient included (they faded with the bit off)."""
  settings = _recipe(name)
  build, hdr = _render_slab(thin=False, settings=settings, scene=_bent_coat())
  own, own_hdr = _render_slab(thin=False, settings={**settings, **SCENE_BUILD}, scene=_bent_coat())
  assert build.horizon_fade and build.clearcoat and not own.horizon_fade
  np.testing.assert_allclose(hdr, own_hdr, rtol=1e-4, atol=1e-6)


# A mirror-reflective floor with a texture (the "materials" class on through one field, mirror
# reflectance: the class's other fields compiled in and off by their second-half bits).
_REFLECTIVE = f"""
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.2 0.3 0.6" width="64" height="64"/>
    <material name="floor" texture="checker" texrepeat="4 4" roughness="0.3" reflectance="0.4"/>
  </asset>
  <worldbody>{_SUN}{_POINT}
    <camera name="cam" pos="0 -3 1.4" xyaxes="1 0 0 0 0.4 1"/>
    <geom type="plane" size="3 3 0.1" material="floor"/>
    <geom type="sphere" pos="0 0 0.5" size="0.4" rgba="0.8 0.3 0.2 1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def second_half_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  """Software tier builds bind the word's second half too (`tiers.FIELD_BITS`;
  on OptiX only in the library): their module literal folds the per-field gates as an OptiX module
  binding the word does, so the CPU renders the OptiX tier build's gate logic."""
  bound = tiers._bound  # the rule under test, widened

  def with_second(f: Features) -> int:
    b = bound(f)
    if f.traversal == "optix" or not f.scene_free or not b & toggles.DEFINED:
      return b
    return b | tiers._bits_of(f, tiers.FIELD_BITS)

  monkeypatch.setattr(tiers, "_bound", with_second)


@pytest.mark.usefixtures("second_half_bound")
@pytest.mark.parametrize("key", ["reflective", "textured", "ring12", "mixed"])
def test_the_tier_word_renders_as_the_scene_build(scene, key: str) -> None:
  """A tier build whose word holds both halves (the per-field bits and the light count) renders
  each scene as its scene build (8-bit within one level, linear HDR within float reordering): every
  second-half gate that is off reproduces the scene build's code without the field."""
  xml, settings = {**ALL, "reflective": _REFLECTIVE}[key], _recipe("fast")
  kw = SCENE_KW.get(key, {})
  tiered = scene(xml, **OPTS, **kw, **settings).render()
  assert tiered.r.context().features.bound >> toggles.HIGH, "the second half is bound"
  own = scene(xml, **OPTS, **kw, **settings, **SCENE_BUILD).render()
  np.testing.assert_allclose(tiered.rgb(), own.rgb(), atol=1.0 / 255)
  np.testing.assert_allclose(tiered.hdr(), own.hdr(), rtol=1e-4, atol=1e-6)


def _bundle_rays(width: int, height: int) -> np.ndarray:
  """(height, width, 2, 3) camera-frame rays (origin, direction) of a 60 degree pinhole-like
  bundle."""
  y, x = np.mgrid[0:height, 0:width].astype(np.float32)
  t = np.tan(np.radians(30.0))
  d = np.stack([(2 * (x + 0.5) / width - 1) * t * width / height, (1 - 2 * (y + 0.5) / height) * t, -np.ones_like(x)], -1)
  d /= np.linalg.norm(d, axis=-1, keepdims=True)
  return np.stack([np.zeros_like(d), d], axis=2)


# The cameras of each camera model (mounted on the scene's camera; `primary.CameraModel`): OpenCV
# distortion and a fisheye (the lens models), and the pinhole.
_LENSES: dict[str, rw.CameraSpec] = {
  "pinhole": rw.CameraSpec(mount="cam"),
  "distortion": rw.CameraSpec(mount="cam", distortion=(0.2, 0.05, 0.01, 0.0)),
  "fisheye": rw.CameraSpec(mount="cam", projection="fisheye", fovy=160.0),
}
_MODELS = {"pinhole": primary.PINHOLE, "distortion": primary.DISTORTION, "fisheye": primary.DISTORTION}


@pytest.mark.parametrize("lens", [*_LENSES, "rays"])
@pytest.mark.parametrize("name", RECIPES)
def test_lens_cameras_take_the_pinhole_build(scene, name: str, lens: str) -> None:
  """A lens camera's renderer takes the pinhole camera's tier build (its model a program of that
  build, `primary.CameraModel`) and renders as its scene build."""
  xml, settings = SCENES["sun"], _recipe(name)
  cam_res = OPTS["cam_res"]
  kw: dict[str, object] = {"rays": [_bundle_rays(*cam_res)]} if lens == "rays" else {"cameras": [_LENSES[lens]]}
  tiered = scene(xml, **OPTS, **kw, **settings).render()
  pinhole = scene(xml, **OPTS, cameras=[rw.CameraSpec(mount="cam")], **settings).r
  assert tiered.r.context().camera_model == (primary.BUNDLE if lens == "rays" else _MODELS[lens])
  assert tiered.r.context().features == pinhole.context().features
  own = scene(xml, **OPTS, **kw, **settings, **SCENE_BUILD).render()
  np.testing.assert_allclose(tiered.rgb(), own.rgb(), atol=1.0 / 255)
  np.testing.assert_allclose(tiered.hdr(), own.hdr(), rtol=1e-4, atol=1e-6)
