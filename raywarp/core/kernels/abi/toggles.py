"""The settings a render kernel reads at run time (`Toggles`, in `Params` and `GeometryFields`).

A tier build (`core/build/variants/tiers.py`) compiles in the effects of its tier and reads at run
time the ones a settings change may turn off without a new kernel (`Features.switches`, the bits of
`Toggles.flags`), the shadows' sampling (`Features.run_sampling`) and the light caps and shadow
budget (`Features.run_counts`). `resolve.resolve_features` fills `Toggles` from the settings'
specialised `Features`. Kernel code follows one pattern:

    on = toggles.on(f)                                  # in the factory
    if wp.static(f.shadows):                            # compiled in
      if on(geo.rt.flags, wp.static(toggles.SHADOWS)):  # a tier build's run-time switch
        ...

`on` is a constant True for the bits a build does not switch (the compiler folds it) and the
run-time bit of `Toggles.flags` for the others.

Bound bits (`Features.bound`): the feature classes, geometry layout and settings a build compiles
in and reads from its build word (`CLASS_BITS`, `GEO_*`, `SETTING_*`; the value `tiers.word`),
the same two-level form: `if wp.static(f.ies): if gate(lit.bits, CLASS_IES):` with `gate =
bound_on(f.bound)`. On OptiX the word is a launch-parameter value each module binds
(`trace/optix/device/params.bound_words`): OptiX folds it and deletes the branches of the bits off,
so one PTX serves every word, each in a module of its own. On the software path it is a literal of
the megakernel module's preamble (`DEFINED`, `preamble`): one module per word from one source, which
the bundle fill links from one pre-optimised LTO-IR (`tools/prebuilt/software.py`).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Final, Protocol

import warp as wp

from raywarp._base.types import ShapeType

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class Toggles:
  """The run-time settings of a build (each the specialised `Features` field it reads at run
  time, as an int; the switches as the bits of `flags`, to keep the launch parameters, which
  hold two copies, small).

  Attributes:
    flags: the switches (`FLAGS`: bit name -> the `Features` field): shadow rays, panel emission
      skipped at reflection hits (LTC area lights), GI, environment lighting, the environment
      seen by rays leaving the scene, the camera headlight, unlit surfaces, shadow rays through
      translucent casters, directional shadows at reflection hits, the fog, back-face culling,
      translucent layers.
    shadow_samples: area / sun-disc shadow rays per light (`Features.shadow_samples`).
    sun_disc: soft sun shadows (`Features.sun_disc`).
    budget: traced casters K (0: every caster; `Features.budget`).
    light_cap: lights shaded exactly (0: all; `Features.light_cap`).
    hit_light_cap: at glossy-reflection hits (`Features.hit_light_cap`).
    reflection_shadows: shadow rays at reflection hits (`Features.reflection_shadows`).
    glossy: the glossy reflection mode, 0 off (`Features.glossy`).
    anisotropy: texture taps along the footprint's major axis (`Features.anisotropy`).
  """

  flags: int
  shadow_samples: int
  sun_disc: int
  budget: int
  light_cap: int
  hit_light_cap: int
  reflection_shadows: int
  glossy: int
  anisotropy: int


# The switch bits of `Toggles.flags`, each the `Features` field it replaces.
LIGHT_PANELS: Final = 1 << 3
GI: Final = 1 << 5
ENV_LIT: Final = 1 << 7
ENV_BACKGROUND: Final = 1 << 8
HEADLIGHT: Final = 1 << 9
UNLIT: Final = 1 << 10
TRANSLUCENT_SHADOWS: Final = 1 << 11
# The compile-time settings a tier build reads at run time (`tiers.SETTING_FIELDS`): one bit
# value in both namespaces, `Toggles.flags` (the software path's run-time switch) and the word's
# second half (an OptiX module's bound bit, `high`: bits 23-27 and 15): `setting` reads both.
SHADOWS: Final = 1 << 23
SOFT_SHADOWS: Final = 1 << 24
# Shadow rays of the directional lights at glossy-reflection hits (`Features.reflection_shadows`,
# reflections.shadows="sun"; second half bit 14).
REFLECTION_SHADOWS: Final = 1 << 14
FOG: Final = 1 << 25
ANISOTROPY: Final = 1 << 26
BACKFACE_CULLING: Final = 1 << 27
# Translucent layers (`Features.transparency`, and the transmissive compositing and solid glass it
# carries: `Features.transmissive`, `refraction`); off: every surface opaque, one per camera ray.
TRANSPARENCY: Final = 1 << 15
FLAGS: Final = {
  "shadows": SHADOWS,
  "reflection_shadows": REFLECTION_SHADOWS,
  "light_panels": LIGHT_PANELS,
  "gi": GI,
  "env_lit": ENV_LIT,
  "env_background": ENV_BACKGROUND,
  "headlight": HEADLIGHT,
  "unlit": UNLIT,
  "translucent_shadows": TRANSLUCENT_SHADOWS,
  "fog": FOG,
  "backface_culling": BACKFACE_CULLING,
  "transparency": TRANSPARENCY,
}
ALL: Final = sum(FLAGS.values())

# The bits of the build word (`Features.bound`: what a build compiles in and reads from the word;
# `tiers.word`: the value). One namespace, held in several launch-parameter copies, each read where
# its struct is at hand: `GeometryFields.bits` (the trace, hit and pose code), `Lights.bits` (the
# light code), the material carrier's `bits` (`shade/materials/material.py`, carried on `Surface` /
# `Lobes`) and `Environment.bits` (the ambient code); `trace/optix/device/params.bound_words` binds
# them all. The bound feature classes (`tiers.CLASSES` names; "culling": the light grid's lookup of
# the general loop, `tiers.LOOP_FORM`): light classes from bit 0, deformables 4-5, material classes
# from bit 9.
CLASS_IES: Final = 1 << 0
CLASS_AREA: Final = 1 << 1
CLASS_HEADLIGHT: Final = 1 << 2
CLASS_CULLING: Final = 1 << 3
# Deformables: the flex hits' branches (`ShapeType.FLEX` in `Features.shape_types`) and their
# materials (`Features.flex_materials`), an OptiX scene-free build's (`tiers.BOUND_FREE`).
FLEX: Final = 1 << 4
FLEX_MATERIALS: Final = 1 << 5
CLASS_COAT: Final = 1 << 9
CLASS_LAYERS: Final = 1 << 10
CLASS_LAYER_MAPS: Final = 1 << 11
CLASS_BASE_MAP: Final = 1 << 12
CLASS_MAPS: Final = 1 << 13
CLASS_MATERIALS: Final = 1 << 14
CLASS_GLASS: Final = 1 << 15
CLASS_BITS: Final = {
  "ies": CLASS_IES,
  "area": CLASS_AREA,
  "headlight": CLASS_HEADLIGHT,
  "culling": CLASS_CULLING,
  "coat": CLASS_COAT,
  "layers": CLASS_LAYERS,
  "layer_maps": CLASS_LAYER_MAPS,
  "base_map": CLASS_BASE_MAP,
  "maps": CLASS_MAPS,
  "materials": CLASS_MATERIALS,
  "glass": CLASS_GLASS,
}
CLASS_ALL: Final = sum(CLASS_BITS.values())
# The geometry bits of an OptiX scene-free build (`tiers.BOUND_FREE`; read from `geo.bits`).
GEO_PLANES: Final = 1 << 16  # the analytic plane loop (`Features.planes`)
GEO_WORLD: Final = 1 << 17  # the per-world IAS is traced (`SceneFlags.has_instances`)
GEO_SHARED: Final = 1 << 18  # the shared IAS is traced (`SceneFlags.has_shared`)
GEO_COMPACT: Final = 1 << 19  # compact pose indexing (`Features.compact_poses`)
GEO_VISIBLE: Final = 1 << 20  # the per-world visibility table (`Features.visibility`)
GEO_SHARED_VISIBLE: Final = 1 << 21  # shared instances' per-world visibility (`SceneFlags.shared_visible`)
GEO_ALL: Final = GEO_PLANES | GEO_WORLD | GEO_SHARED | GEO_COMPACT | GEO_VISIBLE | GEO_SHARED_VISIBLE
# The analytic shape types of the "primitives" class (`tiers.CLASSES`), one bit each (in every copy:
# the trace code reads `geo.bits`, the primitives' texture coordinates the shading's copy): the
# box hit's normal and distance, each type's analytic UV mapping. An OptiX tier build compiles
# every shape type in (`tiers.BOUND_CLASSES`); a module's word has the scene's (`tiers.word`), so
# a scene of one kind folds the others' code (compiled in unused the whole class cost a MuJoCo
# scene's fast raygen +4.1 % SASS and 48 B of stack). The pipeline's custom-primitive and curve
# flags follow the same bits (`trace/optix/device/programs.traced_shapes`).
SHAPE_SPHERE: Final = 1 << 22
SHAPE_CAPSULE: Final = 1 << 23
# Ellipsoids and cylinders share one (the custom-primitive programs; bit 26 is the first half's
# only free bit): a scene of one of them keeps the other's texture mapping.
SHAPE_CUSTOM: Final = 1 << 27
SHAPE_BOX: Final = 1 << 29
# `ShapeType` value -> its bit.
SHAPE_BITS: Final = {
  int(ShapeType.SPHERE): SHAPE_SPHERE,
  int(ShapeType.CAPSULE): SHAPE_CAPSULE,
  int(ShapeType.ELLIPSOID): SHAPE_CUSTOM,
  int(ShapeType.CYLINDER): SHAPE_CUSTOM,
  int(ShapeType.BOX): SHAPE_BOX,
}
SHAPE_ALL: Final = SHAPE_SPHERE | SHAPE_CAPSULE | SHAPE_CUSTOM | SHAPE_BOX
# Scene properties a tier build compiles in and reads from its word (`tiers.BOUND_FIELDS`), each
# measured above 0.2 % of a fast raygen's SASS compiled in unused (a 2-light MuJoCo scene, CPU
# proxy): the camera rays' visible area lights (`Features.visible_lights`, +5.3 %), the spot and
# lobed lights' cone (`Features.has_spot`, +1.5 %), the two-sided instances' culling exception
# (`Features.two_sided`, +1.2 %), the specular curve's F90 (`Features.specular_curve`, +0.8 %) and
# the normal maps' horizon fade (`Features.horizon_fade`, +0.5 %).
SCENE_VISIBLE_LIGHTS: Final = 1 << 6
SCENE_TWO_SIDED: Final = 1 << 7
SCENE_SPECULAR_CURVE: Final = 1 << 8
SCENE_SPOT: Final = 1 << 28
SCENE_HORIZON_FADE: Final = 1 << 30
SCENE_ALL: Final = SCENE_VISIBLE_LIGHTS | SCENE_TWO_SIDED | SCENE_SPECULAR_CURVE | SCENE_SPOT | SCENE_HORIZON_FADE
# The settings bits (read from `env.bits`): GI probes in the ambient terms (`Features.gi`, an OptiX
# scene-free build compiles both ways) and the environment's lighting (`Features.env_lit`), each
# compiled into a tier build and on in a module whose
# word has it (`tiers.BOUND_FIELDS`): folded where a scene or configuration has no lighting
# environment (as a run-time switch the environment's code stayed in the raygen: +15.6 % SASS of a
# fast build without one, bound off +0.26 %).
SETTING_GI: Final = 1 << 24
SETTING_ENV_LIT: Final = 1 << 25
SETTING_ALL: Final = SETTING_GI | SETTING_ENV_LIT
# The bound bits a software build reads from its word copies at run time (`bound_on`), not from
# its modules' literal (`preamble`; `tiers._with_word` leaves them out of `Features.word`): one
# module serves every scene value of them. Those whose code costs a scene without them nothing
# measurable (CPU proxy, sm_80 SASS of a fast / default megakernel of a 2-point-light mesh scene:
# environment lighting +0 (compiled in with the environment class only), two-sided instances +0,
# horizon fade +0.1 %), and the specular curve (+0.9 % / +1.7 %, the probe rays +1 %). The visible
# lights (+4.0 % / +3.0 %, +20-32 B of spills) and the spot cone (+1.8 % / +2.1 %) stay literals:
# the bundle fill lists each software build's modules with each value of them.
RUN_TIME: Final = SETTING_ENV_LIT | SCENE_TWO_SIDED | SCENE_HORIZON_FADE | SCENE_SPECULAR_CURVE

# The word's second half (`bits2` beside `bits` in every copy; `Features.bound` / `word` hold it at
# bit `HIGH` and up, `high`): its constants are declared in-half (gates read `bits2` with
# `bound_on(bound, high=True)`). Bits 0-22 and 30 the fields inside the bound classes a scene
# partly uses (`tiers.FIELD_BITS`; 14: `REFLECTION_SHADOWS`, 15: `TRANSPARENCY`), 23-27 the
# compile-time settings.
HIGH: Final = 32
HALF: Final = (1 << 31) - 1  # a half's bits (bit 31: the sign of the int32 copies)
# The materials class's fields (`tiers.CLASSES` "materials"): mirror reflectance, EON diffuse
# roughness, the specular weight, vertex colours.
MAT_REFLECTION: Final = 1 << 0
MAT_DIFFUSE_ROUGHNESS: Final = 1 << 1
MAT_SPECULAR_WEIGHT: Final = 1 << 2
MAT_VERTEX_COLORS: Final = 1 << 3
# Unlit materials (`Features.unlit_emission`), bit 17.
MAT_UNLIT: Final = 1 << 17
# The maps class's slots and mappings: normal, ORM, emission, opacity, EON roughness, UV sets,
# world UV; the live material UV mappings (`Features.live_material_uv`: the UV set and transform
# of the material each instance shows, MuJoCo's per-world material swaps; an OptiX tier build's,
# `tiers.BOUND_FREE`, off: the instance's own UVs).
MAP_NORMAL: Final = 1 << 4
MAP_ORM: Final = 1 << 5
MAP_EMISSIVE: Final = 1 << 6
MAP_OPACITY: Final = 1 << 7
MAP_DIFFUSE_ROUGHNESS: Final = 1 << 8
MAP_UV_SETS: Final = 1 << 10
MAP_WORLD_UV: Final = 1 << 11
MAP_LIVE_UV: Final = 1 << 16
# The layers class's lobes: specular anisotropy and fuzz, whose inputs' neutral values are the
# lobes off (the coat's are not: the coat layer stays per material). Free in-half bits: 9 and
# 19-22. New ones are taken from bit 19 upward for settings and fields read at run time (zero
# compiles), from bit 30 downward for bits that fold unused code (speed).
LAYER_ANISOTROPY: Final = 1 << 12
LAYER_FUZZ: Final = 1 << 13
# The light caps, the shadow budget and the anisotropic texture taps below a tier build's
# compiled ones (`tiers._RUN_COUNTS`): read from `Toggles` where on (`light_sources.Counts.run`,
# `shadow_budget.make_budget_loop` run_budget, `uv_maps.make_taps`), the compiled values where
# off. In both namespaces as `SHADOWS` (a word bit on OptiX: a module at the tier's counts folds
# the reads; a switch on the software path), bit 18.
RUN_COUNTS: Final = 1 << 18
# The surface class's transmissive layers (glass that reflects and transmits,
# `Features.transmissive`): a scene with translucent layers but no transmissive material folds the
# transmission terms.
# The second half's free bits are allocated from 15 upward and from 30 downward (two allocators
# never meet on one bit).
SURF_TRANSMISSIVE: Final = 1 << 30
# The glass class's solid glass primitives (`Features.glass_shapes`): an OptiX camera ray that
# continues inside a sphere, capsule or box leaves it through its analytic exit face
# (`trace/optix/device/trace.make_leave`), past the layer budget it sees the ambient light ahead; a
# glass scene without them (glass meshes) folds both and their state across the layer loop's traces
# (bound on the whole glass class: a feature-dense room's fast +14 %, default +8 % on RTX).
GLASS_SHAPES: Final = 1 << 29
HIGH_ALL: Final = (
  MAT_REFLECTION
  | MAT_DIFFUSE_ROUGHNESS
  | MAT_SPECULAR_WEIGHT
  | MAT_VERTEX_COLORS
  | MAT_UNLIT
  | MAP_NORMAL
  | MAP_ORM
  | MAP_EMISSIVE
  | MAP_OPACITY
  | MAP_DIFFUSE_ROUGHNESS
  | MAP_LIVE_UV
  | MAP_UV_SETS
  | MAP_WORLD_UV
  | LAYER_ANISOTROPY
  | LAYER_FUZZ
  | RUN_COUNTS
  | SURF_TRANSMISSIVE
  | GLASS_SHAPES
  | SHADOWS
  | REFLECTION_SHADOWS
  | SOFT_SHADOWS
  | FOG
  | ANISOTROPY
  | BACKFACE_CULLING
  | TRANSPARENCY
)


def high(bit: int) -> int:
  """The `Features.bound` / `word` bit of second-half bit `bit`."""
  return bit << HIGH


# Every bit a word can hold (`bound_on`), both halves.
BOUND_ALL: Final = CLASS_ALL | FLEX | FLEX_MATERIALS | GEO_ALL | SHAPE_ALL | SCENE_ALL | SETTING_ALL | high(HIGH_ALL)
# Not a word bit: a software build's word is the `WORD_MACRO` / `WORD2_MACRO` its megakernel
# modules' preamble defines (`Features.word`, `preamble`), a compile-time constant there; the OptiX
# word is a launch-parameter value each module binds.
DEFINED: Final = 1 << 31
WORD_MACRO: Final = "WP_PBR_WORD"
WORD2_MACRO: Final = "WP_PBR_WORD2"


class Word(Protocol):
  """A struct holding a copy of the build word (both halves)."""

  bits: int
  bits2: int


def store(copy: Word, word: int) -> None:
  """Sets the copy `copy` (a struct's `bits` / `bits2`) to build word `word` (both halves)."""
  copy.bits, copy.bits2 = word & HALF, (word >> HIGH) & HALF


def load(copy: Word) -> int:
  """The build word copy `copy` holds (both halves; `store`'s inverse)."""
  return copy.bits | (copy.bits2 << HIGH)


# The most anisotropic taps a build reading them at run time compiles (`toggles.ANISOTROPY`: the
# setting's largest, `surfaces.anisotropy`; also every tier's, `tiers.COMPILED_ON`).
TAPS_CAPACITY: Final = 16
# The capacities of a counted tier build (`Features.run_counts`): the settings' largest values
# (`lighting.max_exact`, `reflections.max_exact`, `shadows.max_casters`).
COUNTS_CAPACITY: Final = {"light_cap": 16, "hit_light_cap": 16, "budget": 16}


def of(f: Features) -> Toggles:
  """The toggles of the specialised Features `f` (the values a build reads at run time)."""
  t = Toggles()
  for name in getattr(Toggles, "vars"):  # noqa: B009  # Warp's Struct (pyrefly sees the class)
    if name != "flags":
      setattr(t, name, int(getattr(f, name)))
  t.flags = sum(bit for name, bit in FLAGS.items() if getattr(f, name))
  return t


@wp.func
def _always(
  # In:
  _flags: int,
  _bit: int,
) -> bool:
  """A specialised build's switch: compiled in means on."""
  return True


def on(f: Features):
  """The switch of `f`: the run-time bit of its switches (`switches_of`), the others constant True
  (`_always` without any)."""
  return switch(switches_of(f))


def sampling_bits(f: Features) -> int:
  """The sampling a `Features.run_sampling` build reads at run time (in-half bits): the soft
  shadows' (`SOFT_SHADOWS`) with shadows, the anisotropic taps (`ANISOTROPY`) with textures; 0
  without `run_sampling`."""
  if not f.run_sampling:
    return 0
  return (SOFT_SHADOWS if f.shadows else 0) | (ANISOTROPY if f.textures else 0)


def count_bits(f: Features) -> int:
  """`RUN_COUNTS` where tier build `f` (scene-free, not a counted one) compiles a light
  cap, a shadow budget or anisotropic texture taps a configuration's lower one can be read
  within, else 0."""
  compiled = f.budget > 0 or f.light_cap > 0 or f.hit_light_cap > 0 or (f.textures and f.anisotropy > 1)
  return RUN_COUNTS if f.scene_free and compiled and not f.run_counts else 0


def switches_of(f: Features) -> int:
  """The run-time switches of build `f`: `f.switches` and the sampling it reads at run time
  (`sampling_bits`) where its word does not bind it (an OptiX tier build's are word bits, a
  software one's are already switches), so `run_sampling` reaches the generated code itself."""
  return f.switches | (sampling_bits(f) & ~((f.bound >> HIGH) & HALF))


def switch(switches: int):
  """The switch of a specialised build whose run-time switches are `switches` (`Features.switches`;
  0: `_always`): for code built from a part of the Features
  (`trace/optix/device/trace.TraceSpec`)."""
  return _switched(switches) if switches else _always


def reads(f: Features, bit: int) -> bool:
  """Whether build `f` reads the compile-time setting of `bit` at run time (`setting`): a switch of
  the software path's tier build, a word bit of an OptiX one."""
  return bool(switches_of(f) & bit) or bool(f.bound & high(bit))


def setting_on(f: Features):
  """`setting` of build `f`."""
  return setting(switches_of(f), f.bound)


@functools.cache
def setting(switches: int, bound: int):
  """on(flags, bits2, bit) of a compile-time setting a build compiles in (`SHADOWS` ...
  `TRANSPARENCY`): its run-time switch where `switches` has it (the software path, `Toggles.flags`)
  and its word bit where `bound` binds it (OptiX: the second half, `bits2`, folded per module);
  compiled in means on otherwise."""
  on, gate = switch(switches), bound_on(bound, high=True)

  @wp.func
  def setting_on(
    # In:
    flags: int,
    bits2: int,
    bit: int,
  ) -> bool:
    """Whether the setting of `bit` is on (see `setting`)."""
    return on(flags, bit) and gate(bits2, bit)

  return setting_on


def bound_on(bound: int, *, high: bool = False):
  """The gate of a build whose bound bits are `bound` (`Features.bound`), for the word's first half
  (`bits`) or with `high` its second (`bits2`, in-half bits): for those, the bit of the word
  (OptiX: the copy passed in, a bound launch-parameter value; with `DEFINED`, software: the
  module's `WORD_MACRO` / `WORD2_MACRO`, but the copy's for the `RUN_TIME` bits), compiled in
  means on for the others (`_always` without any)."""
  half = (bound >> HIGH if high else bound) & HALF
  every = (BOUND_ALL >> HIGH if high else BOUND_ALL) & HALF
  if bound & DEFINED:
    return _defined(half, every, 0 if high else half & RUN_TIME, high)
  return _switched(half, every) if half else _always


def word_on(bound: int, word: int):
  """Host side: whether a build of bound bits `bound` runs bit `bit` in a module of word `word`
  (compiled in means on for the bits it does not bind; `bound_on`'s rule)."""
  return lambda bit: not (bound & bit) or bool(word & bit)


@wp.func_native(f"return ({WORD_MACRO} & (unsigned int)bit) != 0u;")
def _macro_bit(
  # In:
  bit: int,
) -> bool:
  """Whether bit `bit` is set in the module's word (`WORD_MACRO`, a literal: folded)."""
  ...


@wp.func_native(f"return ({WORD2_MACRO} & (unsigned int)bit) != 0u;")
def _macro_bit2(
  # In:
  bit: int,
) -> bool:
  """Whether in-half bit `bit` is set in the module's word's second half (`WORD2_MACRO`)."""
  ...


@functools.cache
def _defined(half: int, every: int, runtime: int, high: bool):
  """A software build's gate of one half: the copy's bit of the `runtime` bits (`RUN_TIME`), the
  macro bit of the other `half` bits, compiled in means on for the others of `every`."""
  fixed = every & ~half
  macro_bit = _macro_bit2 if high else _macro_bit

  @wp.func
  def defined(
    # In:
    bits: int,
    bit: int,
  ) -> bool:
    """Whether bit `bit` is on: always when not bound, else the word copy's (run-time bits) or the
    module word's."""
    if (bit & wp.static(runtime)) != 0:
      return (bits & bit) != 0
    return (bit & wp.static(fixed)) != 0 or macro_bit(bit)

  return defined


def preamble(f: Features, *, outputs: bool) -> str:
  """The CUDA / C++ preamble of a megakernel module of build `f`: its word (`WORD_MACRO`), when
  its gates read it (`DEFINED`), and with `outputs` (a software megakernel's module) the output
  word (`OUT_MACRO`, `HALF_MACRO`: `Features.out_word`, read by `out_on` / `out_half`); empty
  otherwise."""
  text = f"#define {WORD_MACRO} {f.word & HALF}u\n#define {WORD2_MACRO} {f.word >> HIGH}u\n" if f.bound & DEFINED else ""
  if outputs and f.traversal != "optix":
    text += f"#define {OUT_MACRO} {f.out_word & 0xFFFFFFFF}u\n#define {HALF_MACRO} {f.out_word >> 32}u\n"
  return text


# The output word (`kernels/abi/output_layout.py`): its `OUT_*` bits and float16 bits, read where an
# output is stored from the copies of `structs.OutputArrays` (`bits`, `half`): an OptiX module
# binds them (`trace/optix/device/params.bound_words`: folded); a software megakernel module defines
# them (`preamble`: folded, one module per word, `tools/prebuilt/software.py`); the other kernels
# (the deferred display, downsample, sensor) read them at run time.
OUT_MACRO: Final = "WP_PBR_OUT"
HALF_MACRO: Final = "WP_PBR_HALF"


@wp.func_native(f"""
#ifdef {OUT_MACRO}
return ({OUT_MACRO} & (unsigned int)bit) != 0u;
#else
return (bits & bit) != 0;
#endif
""")
def out_on(
  # In:
  bits: int,
  bit: int,
) -> bool:
  """Whether the output word has any of `bit` (`OutputArrays.bits`, or the module's literal)."""
  ...


@wp.func_native(f"""
#ifdef {HALF_MACRO}
return ({HALF_MACRO} & (unsigned int)bit) != 0u;
#else
return (half & bit) != 0;
#endif
""")
def out_half(
  # In:
  half: int,
  bit: int,
) -> bool:
  """Whether the output word stores buffer `bit` as float16 (`OutputArrays.half`, or the module's
  literal)."""
  ...


@functools.cache
def _switched(switches: int, every: int = ALL):
  """A tier build's switch: the run-time bit of the `switches` bits, compiled in means on for the
  others of `every` (the compiler folds both: `bit` is a constant at every call)."""
  fixed = every & ~switches

  @wp.func
  def switched(
    # In:
    flags: int,
    bit: int,
  ) -> bool:
    """Whether switch `bit` is on: always when not among the build's run-time switches."""
    return (bit & wp.static(fixed)) != 0 or (flags & bit) != 0

  return switched
