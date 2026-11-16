"""Shared test helpers: render a MuJoCo scene from inline XML (`make_scene`, the `scene`
fixture) or a host scene on the CPU (`render_host`), and the closed forms the tests compare
against.

`pytest --warp-debug` compiles every Warp module in debug mode (`wp.config.mode = "debug"`):
array accesses are bounds-checked and an out-of-range index aborts the process (code-quality
rule 42 in docs/DEVELOPMENT.md). `--warp-verify-fp` adds Warp's
floating-point verification (`wp.config.verify_fp = True`): a float division by zero or a
non-finite value stored to an array aborts too. It is opt-in while the renderer divides by zero
on purpose (`trace.accel.nodes.reciprocal`: inf components for the slab test of axis-aligned rays).
Warp prints the failing check's line in its own headers (builtin.h `div`, array.h `index`), not
the kernel's: run the failing test alone with `-s -p no:xdist` and one core (`taskset -c 0`, so
CPU launches are not split into band threads) to get the Python stack of the launch. Warp's
store check also reads the destination before writing it, so the uninitialized memory of
`wp.empty` would fail it at random: under `--warp-verify-fp`, `wp.empty` / `wp.empty_like`
allocate zeros. Warp 1.18's debug-mode CPU kernels also crash (segmentation fault) when two
threads launch them at once, which release kernels allow (a 20-line Warp-only kernel reproduces
it): under either option `wp.launch` / `wp.launch_tiled` and the replay of a recorded launch
(`wp.Launch.launch`, e.g. core/pipeline/worlds/validator.py's `record_cmd=True`) hold one
process-wide lock, so the bands of `device.launch_bands` (and the threads of docs/robotics.md) still
run, one launch at a time. Launches made inside Warp itself (warp._src, through its module-level
`launch`) bypass the lock; none of them runs off the main thread in the suite today.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import tempfile
import threading
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, Protocol, final

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline.worlds.tables import InstanceColumns
from raywarp.core.trace.optix.launch import OptixRenderer
from raywarp.core.trace.optix_backend import OptixBackend
from raywarp.settings.fields import DEFAULT_EV100, OutputKind, exposure_from_ev100
from tools import recipes

# Hermetic bake caches (`raywarp.core.pipeline.platform.bake_cache`): each test process starts with
# an empty one.
os.environ["RAYWARP_CACHE_DIR"] = tempfile.mkdtemp(prefix="rw_bakes_")

# Property tests check values, not time: Hypothesis' default 200 ms deadline per example turns one
# stall of a busy xdist worker (a full collection of its heap, a loaded host) into a FlakyFailure.
with contextlib.suppress(ImportError):
  import hypothesis

  hypothesis.settings.register_profile("raywarp", deadline=None)
  hypothesis.settings.load_profile("raywarp")

# The repository root (this conftest sits at the root: it serves tests/ and internal/tests/) and the
# test data directory.
REPO: Final = Path(__file__).resolve().parent
DATA: Final = REPO / "tests" / "data"

WARP_DEBUG_OPTION: Final = "--warp-debug"
WARP_VERIFY_FP_OPTION: Final = "--warp-verify-fp"


def pytest_addoption(parser: pytest.Parser) -> None:
  """Registers `--warp-debug` and `--warp-verify-fp` (see the module docstring)."""
  parser.addoption(WARP_DEBUG_OPTION, action="store_true", help='compile Warp modules with wp.config.mode = "debug"')
  parser.addoption(
    WARP_VERIFY_FP_OPTION,
    action="store_true",
    help="also wp.config.verify_fp = True (division by zero, non-finite stores); implies --warp-debug",
  )


def pytest_configure(config: pytest.Config) -> None:
  """Applies the Warp debug options before any test runs, so before any Warp module compiles
  (modules build on their first launch; mode and verify_fp are part of a module's hash, so
  release and debug binaries live side by side in the kernel cache)."""
  verify_fp = bool(config.getoption(WARP_VERIFY_FP_OPTION))
  if verify_fp or config.getoption(WARP_DEBUG_OPTION):
    wp.config.mode = "debug"
    wp.launch = _serialized(wp.launch)
    wp.launch_tiled = _serialized(wp.launch_tiled)
    wp.Launch.launch = _serialized(wp.Launch.launch)
  if verify_fp:
    wp.config.verify_fp = True
    wp.empty = wp.zeros
    wp.empty_like = wp.zeros_like


# A directory to record the megakernel configurations the session compiles into (one pickle per
# process; `tests/precompile.py` compiles them once before the next suite's workers start).
RECORD_FEATURES_ENV: Final = "RAYWARP_RECORD_FEATURES"


def pytest_sessionstart(session: pytest.Session) -> None:
  """With $RAYWARP_RECORD_FEATURES: records every `pixel.megakernel` call (Features, camera model,
  banded)."""
  del session
  if os.environ.get(RECORD_FEATURES_ENV):
    from raywarp.core.kernels import pixel

    seen: set[object] = set()
    make = pixel.megakernel

    def recording(f, model, *, banded=False):  # the factory's own signature
      seen.add((f, model, banded))
      return make(f, model, banded=banded)

    pixel.megakernel = recording  # pyrefly: ignore[bad-assignment]
    _RECORDED.append(seen)


def pytest_sessionfinish(session: pytest.Session) -> None:
  """Writes the recorded configurations ($RAYWARP_RECORD_FEATURES/features-<pid>.pkl)."""
  del session
  out = os.environ.get(RECORD_FEATURES_ENV)
  if out and _RECORDED:
    import pickle

    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"features-{os.getpid()}.pkl").write_bytes(pickle.dumps(sorted(_RECORDED[0], key=repr)))


_RECORDED: list[set[object]] = []


def _serialized[**P, R](launch: Callable[P, R]) -> Callable[P, R]:
  """`launch` under one process-wide lock (Warp debug kernels crash on concurrent launches)."""
  lock = _LAUNCH_LOCK

  def run(*args: P.args, **kwargs: P.kwargs) -> R:
    """`launch(*args, **kwargs)`, one at a time."""
    with lock:
      return launch(*args, **kwargs)

  return run


_LAUNCH_LOCK: Final = threading.Lock()


# rw.integrations.mujoco.scene keyword arguments, and the `rw.Lighting` fields passed as its
# `lighting`; every other keyword is a `RenderSettings.replace` key (a group,
# `shadows=rw.settings.Shadows(samples=4)`, or "group.field").
SCENE_KWARGS: Final = frozenset({"geom_groups", "use_skybox", "unset_metallic", "unset_roughness"})
LIGHTING_KWARGS: Final = frozenset({"envmap", "envmap_intensity", "envmap_rotation", "ambient", "headlight"})


class _HostReadable(Protocol):
  """A device array (`wp.array.numpy`)."""

  def numpy(self) -> np.ndarray:
    """The array's contents on the host."""
    ...


def _camera(images: _HostReadable | None, kind: str, camera: int) -> np.ndarray:
  """A copy of one camera's images of a (nworld, ncam, ...) output on the host, (nworld, ...).
  A copy: on the CPU, `wp.array.numpy()` is a view of the output buffer, which the next frame
  overwrites (a frame kept for a comparison would silently become the later one).

  Raises:
    ValueError: if the output is off (not in `outputs.kinds`).
  """
  if images is None:
    raise ValueError(f"output {kind!r} is off: add it to outputs.kinds")
  return images.numpy()[:, camera].copy()


def required[T](value: T | None) -> T:
  """`value`, which the test needs set (a requested output, a built table ...): fails if None."""
  if value is None:
    raise AssertionError("expected a value, got None (an output that is off, a table not built)")
  return value


def whole_table(table: wp.array | InstanceColumns) -> wp.array:
  """A per-world table (`Renderer.per_world[field]`) declared whole, not as some instances'
  columns: fails for an `InstanceColumns` view."""
  if not isinstance(table, wp.array):
    raise AssertionError(f"expected the whole per-world table, got {type(table).__name__}")
  return table


def optix_build[S](r: rw.Renderer[S]) -> OptixRenderer:
  """The current OptiX build of `r` (its backend is OptiX and has prepared a frame)."""
  backend = r._backend
  if not isinstance(backend, OptixBackend):
    raise AssertionError(f"the renderer traces with {type(backend).__name__}, not OptiX")
  return required(backend.optix)


@dataclasses.dataclass(slots=True, kw_only=True)
class MjScene:
  """A MuJoCo model and data (host and device), their GpuScene and a Renderer of it."""

  mjm: mujoco.MjModel
  mjd: mujoco.MjData
  m: mjw.Model
  d: mjw.Data
  scene: rw.GpuScene[mjw.Data]
  r: rw.Renderer[mjw.Data]

  def render(self, seed: int = 0) -> MjScene:
    """Renders `d` once and returns self (for chaining)."""
    self.r.render(self.d, seed=seed)
    return self

  @property
  def out(self) -> rw.RenderOutputs:
    """The Renderer's outputs."""
    return self.r.outputs

  def rgb(self, camera: int = 0) -> np.ndarray:
    """Display RGB in [0, 1] of one camera, (nworld, height, width, 3)."""
    return _camera(self.out.rgb, "rgb", camera)[..., :3].astype(np.float32) / 255.0

  def hdr(self, camera: int = 0) -> np.ndarray:
    """Linear HDR radiance of one camera, (nworld, height, width, 3)."""
    return _camera(self.out.hdr, "hdr", camera)

  def depth(self, camera: int = 0) -> np.ndarray:
    """Depth of one camera, (nworld, height, width)."""
    return _camera(self.out.depth, "depth", camera)

  def seg(self, camera: int = 0) -> np.ndarray:
    """Segmentation ids of one camera, (nworld, height, width)."""
    return _camera(self.out.seg, "seg", camera)

  def albedo(self, camera: int = 0) -> np.ndarray:
    """Albedo of one camera, (nworld, height, width, 3)."""
    return _camera(self.out.albedo, "albedo", camera)

  def normal(self, camera: int = 0) -> np.ndarray:
    """World-space normals of one camera, (nworld, height, width, 3)."""
    return _camera(self.out.normal, "normal", camera)

  @property
  def exposure(self) -> float:
    """The linear exposure scale of the Renderer's EV100."""
    return exposure_from_ev100(self.r.settings.exposure.render_ev100())


def split_kwargs(kw: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
  """(`rw.integrations.mujoco.scene` keywords, `RenderSettings.replace` keywords) of `kw`: the
  `SCENE_KWARGS` go into the scene keywords as one `rw.integrations.mujoco.Options` (`adapter`),
  the `LIGHTING_KWARGS` as one `rw.Lighting` (`lighting`)."""
  adapter = {k: v for k, v in kw.items() if k in SCENE_KWARGS}
  scene_kw: dict[str, object] = {"adapter": rw.integrations.mujoco.Options(**adapter)} if adapter else {}  # pyrefly: ignore[bad-argument-type]
  lighting = {k: v for k, v in kw.items() if k in LIGHTING_KWARGS}
  if lighting:
    scene_kw["lighting"] = rw.Lighting(**lighting)  # pyrefly: ignore[bad-argument-type]  # heterogeneous test keywords
  settings_kw = {k: v for k, v in kw.items() if k not in SCENE_KWARGS and k not in LIGHTING_KWARGS}
  return scene_kw, settings_kw


# GI off (sky occlusion sized by the default gi grid fields).
NO_GI: Final = rw.settings.GI(enabled=False)

# Tests build settings without GI unless they ask for it: probe GI is on by default in the library,
# but a GI bake compiles a probe kernel per scene's Features (tens of seconds each on the CPU) and
# bakes on the CPU, which the tests of other features do not need (the suite went from 6.5 to
# more than 43 minutes). GI tests set `gi.enabled` / `GI(enabled=True)` explicitly; `recipe`
# keeps the library's GI. Set here, before any test module is imported.
# GI tests bake the coarse grid they were written for (1 m, 64 rays): a CPU bake at the library's
# 0.5 m / 256 rays costs ~30x more.
_GI_DEFAULTS = required(rw.settings.GI.__init__.__kwdefaults__)
_SETTINGS_DEFAULTS = required(rw.RenderSettings.__init__.__kwdefaults__)
assert "gi" in _SETTINGS_DEFAULTS
LIBRARY_DEFAULTS: Final = (dict(_GI_DEFAULTS), _SETTINGS_DEFAULTS["gi"])
_GI_DEFAULTS.update(enabled=False, spacing=1.0, rays=64)
# RenderSettings' gi default is a GI instance made at import: replaced too.
_SETTINGS_DEFAULTS["gi"] = rw.settings.GI()


# The library's GI (`RenderSettings().gi` outside the test session).
LIBRARY_GI: Final = LIBRARY_DEFAULTS[1]
# Unlit and opaque on the fast field set, without shadows, reflections, light grid or GI: the
# cheapest lit-free configuration (the unlit tier's own settings, opaque).
UNLIT_FAST: Final = (
  rw.RenderSettings(gi=LIBRARY_GI)
  .replace(rw.FAST)
  .replace(
    {
      "surfaces.shading": "unlit",
      "shadows.enabled": False,
      "reflections.enabled": False,
      "transparency.enabled": False,
      "lighting.light_grid": False,
      "gi.enabled": False,
    }
  )
)
# `UNLIT_FAST` as dotted changes of the session's default (a foreign config's overrides).
UNLIT_FAST_CHANGES: Final = rw.RenderSettings().diff(UNLIT_FAST)


# Every name `recipe` takes: the tooling's recipes and `UNLIT_FAST`.
RECIPE_NAMES: Final = (*recipes.RECIPES, "unlit_fast")


def recipe(name: str) -> rw.RenderSettings:
  """The settings of recipe `name` (`tools/recipes.py`: "default", "fast", "unlit", "quality",
  "reference", "rtx_look"; or "unlit_fast", `UNLIT_FAST`) with the library's GI, not the session's
  (GI off: see above)."""
  if name == "unlit_fast":
    return UNLIT_FAST
  changes = rw.RenderSettings().diff(recipes.settings(name))  # over the session's default
  return rw.RenderSettings(gi=LIBRARY_GI).replace(changes)


def library_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  """The library's own GI defaults for the rest of a test (the docs state them)."""
  gi_defaults, settings_gi = LIBRARY_DEFAULTS
  for name, value in gi_defaults.items():
    monkeypatch.setitem(_GI_DEFAULTS, name, value)
  monkeypatch.setitem(_SETTINGS_DEFAULTS, "gi", settings_gi)


def make_scene(  # noqa: PLR0913  # keyword-only fixture options (the test API)
  xml: str,
  *,
  nworld: int = 1,
  cam_res: tuple[int, int] = (32, 32),
  cameras: list[rw.CameraSpec | str | int] | None = None,
  rays: list[np.ndarray] | None = None,
  per_world: rw.PerWorld | None = None,
  **kw: object,
) -> MjScene:
  """A scene of `xml` and its Renderer. `rays`: one (H, W, 2, 3) ray image per scene camera
  (replacing `cameras`); `per_world`: the Renderer's per-world declaration; `kw`: `SCENE_KWARGS`
  for `rw.integrations.mujoco.scene`, `LIGHTING_KWARGS` for its `rw.Lighting`, every other key for
  `default_settings` (`split_kwargs`)."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  scene_kw, settings_kw = split_kwargs(kw)
  scene = rw.integrations.mujoco.scene(mjm, m, d, **scene_kw)  # pyrefly: ignore[bad-argument-type]  # heterogeneous test keywords
  if rays is not None:
    specs = list(scene.cameras.values())
    cameras = [dataclasses.replace(spec, rays=r) for spec, r in zip(specs, rays, strict=True)]
  r = rw.Renderer(
    scene,
    nworld=nworld,
    cameras=cameras,
    resolution=cam_res,
    per_world=per_world,
    settings=default_settings(**settings_kw),
  )
  return MjScene(mjm=mjm, mjd=mjd, m=m, d=d, scene=scene, r=r)


class MakeScene(Protocol):
  """The `scene` fixture: `make_scene`."""

  def __call__(  # noqa: PLR0913  # make_scene's signature
    self,
    xml: str,
    *,
    nworld: int = 1,
    cam_res: tuple[int, int] = (32, 32),
    cameras: list[rw.CameraSpec | str | int] | None = None,
    rays: list[np.ndarray] | None = None,
    per_world: rw.PerWorld | None = None,
    **kw: object,
  ) -> MjScene:
    """See `make_scene`."""
    ...


def default_settings(**kw: object) -> rw.RenderSettings:
  """`RenderSettings().replace(**kw)` (the default) without GI unless `kw` sets a gi field
  (`NO_GI`; else the session's coarse grid), and at the fixed reference exposure (manual,
  `exposure.ev100` = DEFAULT_EV100) unless `kw` sets `exposure.mode` ("auto": automatic
  exposure) or a manual exposure: the tests measure
  direct light, image-based lighting and sky occlusion (which GI replaces) against closed forms
  at a known exposure; GI and automatic exposure have their own tests (test_gi.py,
  test_imaging.py) and the default's pixels the regression suite."""
  base = rw.RenderSettings()
  if not any(k == "gi" or k.startswith("gi.") for k in kw):
    base = base.replace(gi=NO_GI)
  return reference_exposure(base.replace(**kw), kw)


def reference_exposure(settings: rw.RenderSettings, kw: Mapping[str, object]) -> rw.RenderSettings:
  """`settings` at the fixed reference exposure (DEFAULT_EV100) unless `kw` (the test's
  overrides) name `exposure.mode` itself or the settings expose manually (a whole `exposure` group
  in "auto" mode leaves them at the reference); an `exposure.ev100` of `kw` alone is that manual
  exposure."""
  if "exposure.mode" in kw or settings.exposure.mode == "manual":
    return settings
  ev100 = settings.exposure.ev100 if "exposure.ev100" in kw else DEFAULT_EV100
  return settings.replace(**{"exposure.mode": "manual", "exposure.ev100": ev100})


# Camera quaternion (x, y, z, w) looking down -z: the identity.
DOWN: Final = (0.0, 0.0, 0.0, 1.0)


def render_host(  # noqa: PLR0913  # keyword-only after the camera pose
  host: rw.Scene,
  eye: tuple[float, float, float],
  quat: tuple[float, float, float, float] = DOWN,
  *,
  fovy: float,
  res: int | tuple[int, int],
  kinds: frozenset[OutputKind] = frozenset({"hdr"}),
  settings: Mapping[str, object] | None = None,
  seed: int = 0,
) -> rw.Renderer[None]:
  """A Renderer of host scene `host` on the CPU (one world, one free camera at `eye` / `quat`),
  rendered once: software traversal, no environment lighting, no GI, the reference exposure (the
  direct terms the tests check against closed forms), `settings` (`RenderSettings.replace` keys)
  over that."""
  width, height = (res, res) if isinstance(res, int) else res
  rs = rw.RenderSettings(
    gi=NO_GI,
    outputs=rw.Outputs.of(*kinds),
    performance=rw.settings.Performance(traversal="software"),
    environment=rw.settings.Environment(enabled=False),
  ).replace(**(settings or {}))
  rs = reference_exposure(rs, settings or {})
  r = rw.Renderer(rw.upload(host, "cpu"), nworld=1, cameras=[rw.CameraSpec(fovy=fovy)], resolution=(width, height), settings=rs)
  r.camera_pose.assign(np.array([[[*eye, *quat]]], np.float32))
  r.render(seed=seed)
  return r


@final
class _NoDefaultDevice:
  """Stands in for Warp's default device (`no_default_device`): any use of it fails."""

  def __getattr__(self, name: str) -> object:
    """Fails: the code under test relied on Warp's default device."""
    raise AssertionError(f"Warp's default device was used (.{name}): pass device=")


@contextlib.contextmanager
def no_default_device() -> Generator[None]:
  """Within: Warp's default device is a stand-in whose use fails, so an allocation, launch or
  stream lookup without an explicit device fails on any machine (a CPU-only one too, where the
  default and the explicit device are both the CPU). `wp.ScopedDevice` still works inside: it
  saves and restores the attribute (Warp private API: `runtime.default_device`)."""
  from warp._src import context

  wp.init()
  runtime = context.runtime
  assert runtime is not None  # wp.init() created it
  saved = runtime.default_device
  runtime.default_device = _NoDefaultDevice()
  try:
    yield
  finally:
    runtime.default_device = saved


def internal_module(name: str) -> ModuleType:
  """Module `name` of internal/ (the development repository's benchmarks and tools, on the pytest
  pythonpath: `feature_scenes`, `datagen.sinks`, ...), else a skip: the public export has no
  internal/."""
  return pytest.importorskip(name, reason=f"{name}: internal/, the development repository only")


@pytest.fixture
def scene() -> MakeScene:
  """`make_scene`, as a fixture."""
  return make_scene


# The sRGB transfer function's linear segment: threshold and slope (IEC 61966-2-1).
_SRGB_LINEAR_MAX: Final = 0.0031308
_SRGB_LINEAR_SLOPE: Final = 12.92


def srgb(x: np.ndarray) -> np.ndarray:
  """The sRGB encoding of linear `x` in [0, 1] (the closed form the kernels' output is checked
  against)."""
  return np.where(x <= _SRGB_LINEAR_MAX, _SRGB_LINEAR_SLOPE * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def disc_visible_fraction(t: float) -> float:
  """Visible fraction of a uniform disc whose centre is t radii past an occluder's straight edge
  (the circular segment): the penumbra of a sun disc or a disc light."""
  t = min(max(t, -1.0), 1.0)
  return (math.acos(-t) + t * math.sqrt(1.0 - t * t)) / math.pi


def caster_scene(caster: str) -> rw.Scene:
  """A matte floor under a straight-down sun and one translucent caster (base color white,
  opacity 0.4: transmittance 0.6) over its middle: a single-sided closed pane (a thin box), a
  two-sided sheet (one quad, USD doubleSided) or a solid glass cylinder."""
  b = rw.SceneBuilder()
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_roughness=0.1,
      geometry_opacity=0.4,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  matte = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_weight=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=matte)
  at = np.c_[np.eye(3), [0.0, 0.0, 1.0]]
  if caster == "pane":
    b.add_instance("box", size=(0.3, 0.3, 0.02), xform=at, material=glass)
  elif caster == "sheet":
    quad = b.add_mesh(
      np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]]), [[0, 1, 2], [0, 2, 3]]
    )
    b.add_instance(
      "mesh", mesh=quad, xform=at, material=glass, flags=rw.scene.InstanceFlags.DEFAULT | rw.scene.InstanceFlags.DOUBLE_SIDED
    )
  else:
    b.add_instance("cylinder", size=(0.3, 0.1, 0.0), xform=at, material=glass)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0))
  return b.build()


def caster_transmittance(caster: str, traversal: Literal["software", "optix"]) -> float:
  """The floor's light under the caster over its light beside it (the sun's transmittance through
  it), seen from above with `traversal`."""
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("hdr"),
    performance=rw.settings.Performance(traversal=traversal),
    shadows=rw.settings.Shadows(translucent="transmittance"),
    transparency=rw.settings.Transparency(enabled=True),
    gi=rw.settings.GI(enabled=False),  # direct sunlight only
  )
  r = rw.Renderer(
    rw.upload(caster_scene(caster)), nworld=1, cameras=[rw.CameraSpec(fovy=90.0)], resolution=(16, 16), settings=settings
  )
  # Under the caster (z 0.5), looking down: the image centre sees the floor in its shadow, the
  # corners (about 0.5 off axis) the lit floor beside it.
  r.camera_pose.assign(np.array([[[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  r.render()
  img = np.asarray(required(r.outputs.hdr).numpy())[0, 0].reshape(16, 16, -1)[..., :3].mean(-1)
  r.close()
  return float(img[7, 7] / img[0, 0])


def pane_stack_centre(name: str, npane: int, traversal: str, max_layers: int | None = None) -> np.ndarray:
  """The image centre's HDR radiance (3,) of `npane` stacked two-sided closed panes (thin boxes,
  opacity 0.5, red / green / blue: a USD stage's glass) over an emissive floor, seen from above
  with recipe `name` (`recipe`: its layer budget, or `max_layers`), `traversal` and no GI. The
  pane farthest from the camera (nearest the floor) glows."""
  b = rw.SceneBuilder()
  # An emissive orange floor: seen through the panes whatever the shadows, unlike the background.
  floor = b.add_material(
    rw.Material(
      base_color=(0.0, 0.0, 0.0),
      specular_weight=0.0,
      emission_luminance=8.0,
      emission_color=(1.0, 0.6, 0.2),
    )
  )
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=floor)
  for k in range(npane):
    color = np.eye(3)[k % 3] * 0.8 + 0.1
    pane = b.add_material(
      rw.Material(
        base_color=tuple(color),
        specular_roughness=0.1,
        geometry_opacity=0.5,
        # The far pane glows (in the near panes' shadow it would be as dark as the sky).
        emission_luminance=4.0 if k == 0 else 0.0,
        emission_color=tuple(color),
        ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
      )
    )
    b.add_instance(
      "box",
      size=(0.3, 0.3, 0.01),
      xform=np.c_[np.eye(3), [0.0, 0.0, 0.2 + 0.1 * k]],
      material=pane,
      flags=rw.scene.InstanceFlags.DEFAULT | rw.scene.InstanceFlags.DOUBLE_SIDED,
    )
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0))
  changes: dict[str, object] = {
    "outputs": rw.Outputs.of("hdr"),
    "performance.traversal": traversal,
    "gi.enabled": False,
    "transparency.enabled": True,
  }
  if max_layers is not None:
    changes["transparency.max_layers"] = max_layers
  settings = recipe(name).replace(changes)
  with rw.Renderer(
    rw.upload(b.build()), nworld=1, cameras=[rw.CameraSpec(fovy=40.0)], resolution=(16, 16), settings=settings
  ) as r:
    # Above the panes, looking straight down: the centre crosses every pane, then the floor.
    r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
    img = np.asarray(required(r.outputs.hdr).numpy())[0, 0].reshape(16, 16, -1)[..., :3]
  return img[6:10, 6:10].reshape(-1, 3).mean(axis=0)
