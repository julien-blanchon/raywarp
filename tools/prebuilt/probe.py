"""The prebuilt kernels' resolution harness, without a GPU: GPU profiles, a Renderer that resolves
Features without compiling anything (`Probe`), the OptiX scene flags of its scene
(`scene_flags`), the fill's synthetic scenes (`load_point`: `feature_space` codes) and the
settings variants a user reaches (`toggles`). Self-contained in the public tree: the kernel-build
pipeline (`fill.py`, `word_reach.py`, `zero.py`, `tools/prebuilt/fill_all.sh`) runs from a public
checkout; the development tree's census adds its registry scenes through `forms` (internal/).

Everything here runs on the CPU device. `Probe` is a Renderer whose construction skips the
background compiles and the software BVH build (the only device work its constructor does besides
small table copies); `Probe.resolution` then resolves any settings for any device profile through
the library's own `resolve.resolve_features` (the path a real Renderer takes), with a `DeviceCaps`
of that GPU. Nothing here is imported by `raywarp`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import types
import typing
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import warp as wp

import raywarp as rw
from raywarp._base.dims import Ids, Mask
from raywarp._base.types import ShapeType
from raywarp.core.build.variants.resolve import Resolution, resolve_features
from raywarp.core.kernels.abi.features import Features
from raywarp.core.pipeline.gpu_scene.gpu_scene import shadow_casters
from raywarp.core.pipeline.platform import device as device_lib
from raywarp.core.pipeline.renderer import Renderer
from raywarp.core.trace.optix import accel
from raywarp.core.trace.software_backend import SoftwareBackend
from raywarp.settings import RenderSettings
from tools.prebuilt import feature_space

ROOT: Final = Path(__file__).resolve().parents[2]
# GPU profiles: (arch, OptiX available). The RTX PRO 6000 (Blackwell RTX: other tuned defaults),
# the L4 (Ada), the A10G (Ampere RTX) and the A100 / H100 (no RT cores: the software traversal):
# one per architecture of the multi-architecture fill (tools/prebuilt/fill_all.sh).
PROFILES: Final = {
  "rtx_pro_6000": (120, True),
  "l4": (89, True),
  "a10g": (86, True),
  "h100": (90, False),
  "a100": (80, False),
  "cpu": (0, False),
}
RESOLUTION: Final = (512, 512)


def device_caps(profile: str) -> device_lib.DeviceCaps:
  """The `DeviceCaps` a Renderer sees on the GPU of `profile` (OptiX availability injected: the
  property would load the driver's OptiX)."""
  arch, optix = PROFILES[profile]
  cuda = arch > 0
  caps = device_lib.DeviceCaps(
    alias="cuda:0" if cuda else "cpu",
    kind="cuda" if cuda else "cpu",
    arch=arch,
    rt_cores=cuda and device_lib.has_rt_cores(arch),
    graphs=cuda,
    pinned=cuda,
    device_textures=cuda,
    parallel_launch=cuda,
    blackwell_rtx=cuda and arch >= 120,
  )
  caps.__dict__["optix"] = device_lib.OptixSupport(available=optix, reason=None if optix else "profile without OptiX")
  return caps


@contextlib.contextmanager
def no_device_work() -> Generator[None]:
  """Renderers built inside skip their background compiles and the software BVH build."""
  start, configure = Renderer._start_compiles, SoftwareBackend.configure
  Renderer._start_compiles = lambda self: None
  SoftwareBackend.configure = lambda self, ctx: None
  try:
    yield
  finally:
    Renderer._start_compiles, SoftwareBackend.configure = start, configure


class LayoutView(Protocol):
  """What `scene_flags` reads of a renderer's sharing layout (`sharing.Layout`)."""

  @property
  def per_world(self) -> Mask[int]:
    """(ninstance,) the instances posed per world."""
    ...

  @property
  def shared_ids(self) -> Ids[int]:
    """The rendered shared instances."""
    ...

  @property
  def visible_slot(self) -> Ids[int]:
    """(ninstance,) each instance's per-world visibility column (-1: none)."""
    ...


class RendererView(Protocol):
  """What `scene_flags` reads of a renderer: its scene and sharing layout."""

  @property
  def scene(self) -> rw.GpuScene:
    """The renderer's scene."""
    ...

  @property
  def layout(self) -> LayoutView:
    """The renderer's sharing layout."""
    ...


class Resolver(Protocol):
  """A scene that resolves settings to builds: `Probe`, or the zero-compile walk's stub scenes
  (`zero.Probe`)."""

  @property
  def name(self) -> str:
    """The scene's name (its census label)."""
    ...

  @property
  def renderer(self) -> RendererView:
    """The renderer (or its stand-in) the scene flags read."""
    ...

  def resolution(self, settings: RenderSettings, profile: str) -> Resolution:
    """The resolution of `settings` on `profile`'s GPU (ValueError: settings the scene cannot
    honor)."""
    ...


@dataclasses.dataclass
class Probe:
  """A scene's Renderer (CPU, nothing compiled) that resolves Features for other settings / GPUs."""

  name: str
  renderer: Renderer
  # The settings' own build specialised on the scene (`performance.kernels="scene"`) instead of
  # the tier build.
  scene_build: bool = False

  def features(self, settings: RenderSettings, profile: str) -> Features:
    """`Features` of `settings` on `profile`'s GPU (ValueError: settings the scene cannot honor)."""
    return self.resolution(settings, profile).features

  def resolution(self, settings: RenderSettings, profile: str) -> Resolution:
    """The resolution of `settings` on `profile`'s GPU: the build's Features and the scene's own
    (`effective`; ValueError: settings the scene cannot honor)."""
    if self.scene_build:
      settings = settings.replace(**{"performance.kernels": "scene"})
    r = self.renderer
    r.settings = settings
    casters = shadow_casters(r._scene_slots(None), r.per_world_fields)
    return resolve_features(settings, r._caps, r._per_world_decl, device_caps(profile), casters=int(casters.sum()))


def probe(
  name: str, scene: rw.GpuScene, cameras: list[rw.CameraSpec] | list[str] | None, per_world: rw.PerWorld | None = None
) -> Probe:
  """The `Probe` of `scene` (RESOLUTION, default; 1 world, or 2 with a `per_world` declaration)."""
  with no_device_work(), wp.ScopedDevice("cpu"):
    r = Renderer(scene, nworld=1 if per_world is None else 2, cameras=cameras, resolution=RESOLUTION, per_world=per_world)
  return Probe(name, r)


def scene_flags(p: Resolver, f: Features) -> types.SimpleNamespace:
  """The `programs.SceneFlags` an `accel.OptixScene` of the probe's renderer would have (host-side;
  no opacity micromaps, no split masks): its shared instances have a per-world visibility when
  the layout declares one for any of them (`OptixScene.shared_visible`, BUG-771: always False
  here left the build word's `GEO_SHARED_VISIBLE` out of the census records)."""
  r = p.renderer
  scene, layout = r.scene, r.layout
  itype = np.asarray(scene.instance_type.numpy(), np.int32)
  per_world = np.asarray(layout.per_world, bool)
  world = accel._members([g for g in scene.rendered if per_world[g]], itype)
  shared = accel._members(list(layout.shared_ids), itype)
  deforms = accel.deforms(scene)
  declared = np.asarray(layout.visible_slot, np.int32) >= 0
  return types.SimpleNamespace(
    has_mesh=any(int(itype[g]) in accel._TRIANGLE_GASES for g in world[0] + shared[0]) or deforms,
    has_box=any(itype[g] == int(ShapeType.BOX) for g in world[0] + shared[0]),
    has_instances=any(world) or deforms,
    has_shared=any(shared),
    translucent_triangles=f.translucent_shadows,
    translucent_curves=(f.translucent_shadows, f.translucent_shadows),
    shared_visible=any(declared[g] for kind in shared for g in kind),
    primitive_types=tuple(sorted({int(itype[g]) for g in world[2]})),
    shared_primitive_types=tuple(sorted({int(itype[g]) for g in shared[2]})),
    has_world_curves=bool(world[1]),
    has_shared_curves=bool(shared[1]),
    has_omm=False,
  )


# Scenes ------------------------------------------------------------------------------------------


def load_mjcf(spec_fn: Callable[[], object], lighting: rw.Lighting | None = None) -> tuple[rw.GpuScene, list[str] | None]:
  """A MuJoCo scene on the CPU device and its cameras, lit by `lighting` (None: its own lights)."""
  import mujoco
  import mujoco_warp as mjw

  import raywarp.integrations.mujoco

  mjm = spec_fn().compile()  # pyrefly: ignore[missing-attribute]
  with wp.ScopedDevice("cpu"):
    scene = raywarp.integrations.mujoco.scene(mjm, mjw.put_model(mjm), lighting=lighting)
  names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mjm.ncam)]
  return scene, [n for n in names if n][:1] or None


def is_point(name: str) -> bool:
  """Whether `name` is a feature-space scene code (`feature_space.Point.code`)."""
  if not name.startswith("fs_"):
    return False
  try:
    return feature_space.parse(name).code() == name
  except ValueError:
    return False


def load_point(code: str) -> tuple[rw.GpuScene, list[rw.CameraSpec] | list[str] | None, dict[str, object]]:
  """(GpuScene on the CPU, cameras, required settings: none) of feature-space scene `code`: its
  MuJoCo copy (`feature_space.MJCF_GEOMETRY`: height fields, aliased poses, deformables; its own
  lights) or its code scene (`feature_space.build`, the first view's field of view)."""
  import mujoco

  p = feature_space.parse(code)
  if p.geometry in feature_space.MJCF_GEOMETRY:
    scene, cams = load_mjcf(lambda: mujoco.MjSpec.from_string(feature_space.build_mjcf(p)))
    return scene, cams, {}
  host = feature_space.build(p)
  with wp.ScopedDevice("cpu"):
    return rw.upload(host), [rw.CameraSpec(fovy=float(fovy)) for _eye, _target, fovy in feature_space.VIEWS[:1]], {}


@dataclasses.dataclass(frozen=True)
class Form:
  """One way a census scene is rendered: its row label, the scene, cameras, required settings,
  per-world declaration (None: shared only) and the recipes it is rendered with (empty: every
  recipe of the census)."""

  label: str
  scene: rw.GpuScene
  cameras: list[rw.CameraSpec] | list[str] | None
  required: dict[str, object]
  per_world: rw.PerWorld | None = None
  recipes: tuple[str, ...] = ()


def point_forms(code: str) -> list[Form]:
  """The census forms of feature-space scene `code`: as it loads, and the per-world optics glass
  point also with its per-world declaration (`feature_space.OPTICS_PER_WORLD`)."""
  scene, cams, required = load_point(code)
  out = [Form(code, scene, cams, required)]
  if feature_space.parse(code) == feature_space.OPTICS_POINT:
    per_world = rw.PerWorld(material=list(feature_space.OPTICS_PER_WORLD["material"]))
    out.append(Form(f"{code}+per_world0", scene, cams, required, per_world))
  return out


# Settings variants -------------------------------------------------------------------------------


def recipe(name: str) -> RenderSettings:
  """The settings of recipe `name` (`tools/recipes.py`)."""
  from tools.recipes import settings

  return settings(name)


def _alternatives(value: object, hint: object, runtime: bool) -> list[object]:
  """Other values a user can pick for a field: every other choice of an enum / Tri / bool, a few
  numbers (runtime numbers: one, to check they stay runtime), an output spec on / off."""
  from raywarp.viewer.controls import annotation, enum_options

  options = enum_options(hint)
  if options:
    return [o for o in options if not (type(o) is type(value) and o == value)]
  if isinstance(value, bool):
    return [not value]
  if isinstance(value, int):
    cands = [value * 2 or 1] if runtime else [0, 1, 2, 4, 8, value * 2]
    return [c for c in dict.fromkeys(cands) if c != value]
  if isinstance(value, float):
    return [value * 2.0 if value else 1.0]
  args = typing.get_args(annotation(hint))
  specs = [a for a in args if isinstance(a, type) and dataclasses.is_dataclass(a)]
  if value is None and specs:
    return [specs[0]()]
  return []


def toggles(base: RenderSettings) -> Iterator[tuple[str, object, RenderSettings | str]]:
  """(field path, new value, settings or the validation error) for every single-field change of
  `base` a user can make (every leaf; outputs on / off; output spec fields)."""
  for gf in dataclasses.fields(base):
    group = getattr(base, gf.name)
    hints = typing.get_type_hints(type(group))
    for f in dataclasses.fields(group):
      value = getattr(group, f.name)
      runtime = bool(f.metadata.get("runtime"))
      path = f"{gf.name}.{f.name}"
      if dataclasses.is_dataclass(value):  # an output spec: off, then each of its fields
        yield path, None, _apply(base, gf.name, {f.name: None})
        sub_hints = typing.get_type_hints(type(value))
        for sf in dataclasses.fields(value):
          for alt in _alternatives(getattr(value, sf.name), sub_hints[sf.name], runtime=False):
            spec = _try(lambda alt=alt, sf=sf, value=value: dataclasses.replace(value, **{sf.name: alt}))
            yield f"{path}.{sf.name}", alt, spec if isinstance(spec, str) else _apply(base, gf.name, {f.name: spec})
        continue
      for alt in _alternatives(value, hints[f.name], runtime):
        yield path, alt, _apply(base, gf.name, {f.name: alt})


def _try(fn: Callable[[], object]) -> object:
  """`fn()`, or its ValueError / TypeError message."""
  try:
    return fn()
  except (ValueError, TypeError) as e:
    return f"invalid: {e}"


def _apply(base: RenderSettings, group: str, changes: dict[str, object]) -> RenderSettings | str:
  """`base` with fields of one group changed, validated (or the error message)."""
  out = _try(lambda: base.replace({f"{group}.{name}": value for name, value in changes.items()}))
  return out if isinstance(out, (RenderSettings, str)) else str(out)
