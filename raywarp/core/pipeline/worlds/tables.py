"""Per-world tables of a Renderer: the values a Renderer lets each world change (colors, lights,
poses, ...), with the renderer-level defaults and the `InstanceColumns` views of selected instances.

Every per-world field is declared at construction (`sharing.PerWorld`): its (nworld, n) table is
allocated there as a copy of the current value, and the declaration (not the writes) decides what
is compiled in (`resolve.resolve_features`), so writing a table never reconfigures. Writes
are in place: a captured CUDA graph keeps reading the same tables.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, cast, final

import numpy as np
import warp as wp

from raywarp._base.named import construct_named, field_value
from raywarp._base.types import NOPENPBR_TEXROLE, NTEXROLE, SPOT_INNER_FRACTION, TEX_IDENTITY, tex_affine, uv_xform
from raywarp.core.kernels.abi.structs import lens_coefficients
from raywarp.core.kernels.generic import device_generic
from raywarp.core.pipeline.derive import fields as derived_fields
from raywarp.core.pipeline.derive.fields import DERIVED_FIELDS, DERIVED_SLOTS, SHADING_SLOT, SHININESS_SLOT, DerivedTables
from raywarp.core.pipeline.frame import Frame
from raywarp.core.pipeline.gpu_scene import binding as bind
from raywarp.core.pipeline.gpu_scene.binding import SLOTS
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.worlds import sharing
from raywarp.core.pipeline.worlds.optics import OpticsTables, WorldOptics, every_closed, switched
from raywarp.core.pipeline.worlds.validator import Range
from raywarp.core.post.grading.look import LOOK_FIELDS
from raywarp.settings.fields import exposure_from_ev100

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray, AnyHostArray, Ids
  from raywarp.core.pipeline.cameras import CameraRig
  from raywarp.core.pipeline.derive.fields import DerivedSlot
  from raywarp.core.pipeline.gpu_scene.binding import PoseBuffers, Slots
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Renderer-level per-world fields: name -> kernel field (`Frame` field, `Defaults` attribute).
RENDERER_FIELDS: Final = {
  "exposure": "exposure",
  "ambient": "ambient",
  "env.id": "env_id",
  "env.intensity": "env_intensity",
  "env.yaw": "env_yaw",
  "camera.fovy": "camera_fovy",
  "camera.intrinsics": "camera_intrinsics",
  "camera.distortion": "camera_distortion",
  "fog.color": "fog_color",
  "fog.range": "fog_range",
  **{name: name.replace(".", "_") for name in LOOK_FIELDS},
  "sensor.noise": "sensor_noise",
  "sensor.vignette": "sensor_vignette",
  "sensor.response": "sensor_response",
}


# Not slots: `value` reads fields by name through `named.field_value` (`vars`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class Defaults:
  """The renderer-level per-world fields' values when not declared per world (one row each,
  written in place by the Renderer when its settings change).

  Attributes:
    exposure: (1,) photometric exposure of `Exposure.render_ev100()`.
    ambient: (1,) the scene's constant ambient radiance; `no_ambient` (1,) zero (environment-lit
      scenes: the environment is the ambient term).
    env_id, env_intensity, env_yaw: (1,) environment map, its scale and rotation.
    camera_fovy, camera_intrinsics, camera_distortion: (1, ncam) the cameras' (`CameraRig.fovy` /
      `intrinsics` / `coefficients`).
    fog_color, fog_range: (1,) the fog color (linear x exposure) and (start, end, haze) of
      `settings.Atmosphere` (`fog`).
    grading_*, sensor_aberration: (1,) the look's per-world fields (`grading.PER_WORLD_FIELDS`).
    sensor_*: (1,) the sensor pass's per-world fields (`sensor.PER_WORLD_FIELDS`).
  """

  exposure: wp.array[float, [int]]
  ambient: wp.array[wp.vec3, [int]]
  no_ambient: wp.array[wp.vec3, [int]]
  env_id: wp.array[int, [int]]
  env_intensity: wp.array[float, [int]]
  env_yaw: wp.array[float, [int]]
  camera_fovy: wp.array[float, [int, int]]
  camera_intrinsics: wp.array[wp.vec4, [int, int]]
  camera_distortion: wp.array[lens_coefficients, [int, int]]
  fog_color: wp.array[wp.vec3, [int]]
  fog_range: wp.array[wp.vec3, [int]]
  grading_white_balance: wp.array[wp.vec2, [int]]
  grading_saturation: wp.array[float, [int]]
  grading_lift: wp.array[wp.vec3, [int]]
  grading_gamma: wp.array[wp.vec3, [int]]
  grading_gain: wp.array[wp.vec3, [int]]
  sensor_aberration: wp.array[float, [int]]
  sensor_noise: wp.array[wp.vec2, [int]]
  sensor_vignette: wp.array[wp.vec3, [int]]
  sensor_response: wp.array[wp.vec4, [int]]

  @staticmethod
  def of[S](scene: GpuScene[S], rig: CameraRig, *, ev100: float) -> Defaults:
    """The defaults of a renderer of `scene` with cameras `rig` and exposure `ev100`, on the
    scene's device (the fog and look rows zero: the Renderer writes them from its settings)."""
    env = scene.environment
    dev = scene.device
    return Defaults(
      exposure=wp.array([exposure_from_ev100(ev100)], dtype=float, device=dev),
      ambient=wp.array([env.ambient], dtype=wp.vec3, device=dev),
      no_ambient=wp.zeros(1, dtype=wp.vec3, device=dev),
      env_id=wp.zeros(1, dtype=int, device=dev),
      env_intensity=wp.array([env.intensity], dtype=float, device=dev),
      env_yaw=wp.array([env.yaw], dtype=float, device=dev),
      camera_fovy=rig.fovy,
      camera_intrinsics=rig.intrinsics,
      camera_distortion=rig.coefficients,
      fog_color=wp.zeros(1, dtype=wp.vec3, device=dev),
      fog_range=wp.zeros(1, dtype=wp.vec3, device=dev),
      grading_white_balance=wp.zeros(1, dtype=wp.vec2, device=dev),
      grading_saturation=wp.zeros(1, dtype=float, device=dev),
      grading_lift=wp.zeros(1, dtype=wp.vec3, device=dev),
      grading_gamma=wp.zeros(1, dtype=wp.vec3, device=dev),
      grading_gain=wp.zeros(1, dtype=wp.vec3, device=dev),
      sensor_aberration=wp.zeros(1, dtype=float, device=dev),
      sensor_noise=wp.zeros(1, dtype=wp.vec2, device=dev),
      sensor_vignette=wp.zeros(1, dtype=wp.vec3, device=dev),
      sensor_response=wp.zeros(1, dtype=wp.vec4, device=dev),
    )

  def value(self, field: str, *, env_lit: bool) -> AnyDeviceArray:
    """The default of renderer-level field `field` ("ambient": none when environment-lit)."""
    key = RENDERER_FIELDS[field]
    if key == "ambient" and env_lit:
      return self.no_ambient
    return cast("AnyDeviceArray", field_value(self, key))


# Bounds of the float tables of the per-world fields, (low, high) inclusive, checked on the device
# every frame they are declared (`Validator.ranges`; a value outside raises at the next render):
# no kernel clamps them, and a NaN, an inf or a negative value would make the world's image NaN,
# inf or negative radiance. Colors, strengths and intensities are >= 0 (the largest float32: any
# finite), unit quantities in [0, 1] (OpenPBR), a field of view > 0 (degrees, or an orthographic
# camera's extent in metres: no upper bound), angles, positions and offsets finite; a light's
# falloff window `light.range` may be inf (no window: the inverse square law at every distance, the
# upload's value for lights without one).
# The largest and the smallest normal float32.
_MAX_FINITE: Final = 3.4028235e38
_NON_NEGATIVE: Final = (0.0, _MAX_FINITE)
_UNIT: Final = (0.0, 1.0)
_FINITE: Final = (-_MAX_FINITE, _MAX_FINITE)
_POSITIVE: Final = (1.17549435e-38, _MAX_FINITE)
TABLE_RANGES: Final = {
  "material.base_color": _NON_NEGATIVE,
  "instance.color": _NON_NEGATIVE,
  "material.base_diffuse_roughness": _UNIT,
  "material.base_metalness": _UNIT,
  "material.specular_roughness": _UNIT,
  "material.mirror": _UNIT,
  "material.tex_variation": _FINITE,
  "light.strength": _NON_NEGATIVE,
  "light.color": _NON_NEGATIVE,
  "light.radius": _NON_NEGATIVE,
  "light.range": (0.0, math.inf),
  "light.cutoff": (0.0, 180.0),
  "light.cone_inner": _UNIT,
  "light.pos": _FINITE,
  "light.dir": _FINITE,
  "light.size": _NON_NEGATIVE,
  "camera.fovy": _POSITIVE,
  "camera.intrinsics": _FINITE,
  "ambient": _NON_NEGATIVE,
  "env.intensity": _NON_NEGATIVE,
  "env.yaw": _FINITE,
  "fog.color": _NON_NEGATIVE,
}


def _components(table: AnyDeviceArray) -> wp.array[float, [int, int]]:
  """A (rows, n x components) float32 view of a contiguous float-based (rows, ...) table: what
  `check_range` reads (a (rows,) table: (rows, 1)). The view borrows the table's memory: the
  caller keeps the table alive (`PerWorldTables.overrides`).

  Raises:
    TypeError: for a table whose elements are not float32 components.
  """
  inner = _inner_shape(table.dtype)
  scalar = getattr(table.dtype, "_wp_scalar_type_", table.dtype)
  if scalar is not wp.float32 or not table.is_contiguous:
    raise TypeError(f"range check: a contiguous float32-based table, got {table.dtype}")
  rows = table.shape[0]
  width = table.size // max(rows, 1) * math.prod(inner)
  return wp.array(ptr=table.ptr, dtype=float, shape=(rows, width), device=table.device)


def absent_slots(*, nworld: int, nlight: int, ninstance: int, nmaterial: int, device: wp.Device) -> Slots:
  """Placeholders of the optional slots (`frame.Frame`): one row of neutral values (no shading
  bit, visible, no layer, no cut-out, mesh UVs) per instance / material / light, so that a build
  compiling a reader in (a scene-free tier build, `core/build/variants/tiers.py`) reads the neutral
  value of every instance and material the scene has."""
  n, ni, nm = max(nlight, 1), max(ninstance, 1), max(nmaterial, 1)
  return {
    "instance.shading": wp.zeros((1, ni), dtype=int, device=device),
    "instance.visible": wp.ones((1, ni), dtype=int, device=device),
    "shared.pos": wp.zeros(1, dtype=wp.vec3, device=device),
    "shared.rot": wp.zeros(1, dtype=wp.mat33, device=device),
    "material.base_diffuse_roughness": wp.empty((1, 0), dtype=float, device=device),
    "material.specular_weight": wp.empty((1, 0), dtype=float, device=device),
    "material.specular_anisotropy": wp.zeros((1, nm), dtype=wp.vec2, device=device),
    "material.fuzz": wp.zeros((1, nm), dtype=wp.vec2, device=device),
    "material.fuzz_color": wp.ones((1, nm), dtype=wp.vec3, device=device),
    "material.coat_color": wp.ones((1, nm), dtype=wp.vec3, device=device),
    "material.coat_eta": wp.full((1, nm), 1.5, dtype=float, device=device),
    "material.alpha_cutoff": wp.zeros((1, nm), dtype=float, device=device),
    "material.uv_world": wp.zeros((1, nm), dtype=uv_xform, device=device),
    "light.cone_inner": wp.full((1, n), SPOT_INNER_FRACTION, dtype=float, device=device),
    "light.attenuation": wp.array(np.tile([1.0, 0.0, 0.0], (1, n, 1)).astype(np.float32), dtype=wp.vec3, device=device),
    "light.shape": wp.zeros((1, n), dtype=int, device=device),
    "light.axis": wp.array(np.tile([1.0, 0.0, 0.0], (1, n, 1)).astype(np.float32), dtype=wp.vec3, device=device),
    "light.size": wp.zeros((1, n), dtype=wp.vec2, device=device),
    "camera.pos": wp.zeros((nworld, 1), dtype=wp.vec3, device=device),
    "camera.rot": wp.zeros((nworld, 1), dtype=wp.mat33, device=device),
    "flex.vert": wp.zeros((nworld, 0), dtype=wp.vec3, device=device),
  }


# Light fields a simulator binding supplies per call only (MuJoCo: d.light_xpos / d.light_xdir).
_SIMULATOR_LIGHT: Final = frozenset({"light.pos", "light.dir"})


def _declare_hint(field: str) -> str:
  """How to declare per-world `field` (`sharing.PerWorld` group and short name, or instance
  patterns for the geometry fields)."""
  if field in sharing.FIELD_GROUP:
    return f"declare it: Renderer(..., per_world=rw.PerWorld({sharing.FIELD_GROUP[field]}=[...]))"
  group, _, name = field.partition(".")
  if field in ("exposure", "ambient") or group in ("fog", "grading", "sensor"):
    group, name = "env", field
  elif group in ("instance", "material"):
    group = "material"
  return f"declare it: Renderer(..., per_world=rw.PerWorld({group}=[{name!r}]))"


@wp.kernel
@device_generic
def _scatter[T](
  # In:
  values: wp.array2d[T],
  columns: wp.array1d[int],
  # Out:
  table: wp.array2d[T],
) -> None:
  """table[w, columns[k]] = values[w (or row 0), k]."""
  w, k = wp.tid()
  table[w, columns[k]] = values[w % values.shape[0], k]


@wp.kernel
@device_generic
def _gather[T](
  # In:
  table: wp.array2d[T],
  columns: wp.array1d[int],
  # Out:
  values: wp.array2d[T],
) -> None:
  """values[w, k] = table[w, columns[k]]."""
  w, k = wp.tid()
  values[w, k] = table[w, columns[k]]


@wp.kernel
def _keep_hidden(
  # In:
  color: wp.array2d[wp.vec4],
  live: wp.array2d[wp.vec4],
  # Out:
  shown_out: wp.array2d[wp.vec4],
) -> None:
  """Instance i's color in world w: the per-world table's `color`, its alpha 0 where the
  binding's `live` color hides the instance (`Binding.hides_by_color`; rows 1 or nworld)."""
  w, i = wp.tid()
  c = color[wp.min(w, color.shape[0] - 1), i]
  if live[wp.min(w, live.shape[0] - 1), i][3] == 0.0:
    c = wp.vec4(c[0], c[1], c[2], 0.0)
  shown_out[w, i] = c


# The dtypes of the instance tables `InstanceColumns` views (ids, scalars and the derived fields'
# floats, sizes and positions, colors, rotations): `_scatter` / `_gather` are declared for each
# here, so their module is one fixed module (the prebuilt bundle holds it) and a write never
# compiles an overload.
COLUMN_DTYPES: Final = (wp.int32, wp.float32, wp.vec3, wp.vec4, wp.mat33)


def _declare_overloads() -> None:
  """Declares `_scatter` and `_gather` for every dtype of `COLUMN_DTYPES` (at import)."""
  for dtype in COLUMN_DTYPES:
    table, columns = wp.array2d(dtype=dtype), wp.array1d(dtype=int)
    wp.overload(_scatter, {"values": table, "columns": columns, "table": table})
    wp.overload(_gather, {"table": table, "columns": columns, "values": table})


_declare_overloads()


@final
class InstanceColumns[NWorld: IntVar, NColumn: IntVar]:
  """Per-world values of some instances: columns `columns` of the (nworld, n) `array`.

  `assign(values)` writes (nworld or 1, len(columns)) values with one device scatter kernel (a
  Warp array on the renderer's device: graph-capturable, no host round trip; NumPy values are
  copied to the device first); `numpy()` reads them (a gather kernel and a copy of the columns).

  Attributes:
    array: the renderer's per-world table.
    columns: (ncolumn,) its columns of the matched instances; `names` their instance names.
  """

  def __init__(self, array: AnyDeviceArray, columns: Ids[NColumn], names: Sequence[str]) -> None:
    """A view of `columns` of `array`.

    Raises:
      TypeError: for a table dtype outside `COLUMN_DTYPES`.
    """
    if array.dtype not in COLUMN_DTYPES:
      raise TypeError(f"InstanceColumns: no scatter / gather for dtype {array.dtype} (COLUMN_DTYPES)")
    self.array = array
    self.columns: Ids[NColumn] = np.asarray(columns, np.int32)
    self.names = tuple(names)
    # The columns on the device (one placeholder entry when there is none: no empty array).
    host = self.columns if len(self.columns) else np.zeros(1, np.int32)
    self._columns = wp.array(host, dtype=int, device=array.device)

  @property
  def shape(self) -> tuple[int, int]:
    """(nworld, len(columns))."""
    return (self.array.shape[0], len(self.columns))

  def numpy(self) -> AnyHostArray:
    """(nworld, len(columns), ...) values (a host read)."""
    device = self.array.device
    out = wp.empty(self.shape, dtype=self.array.dtype, device=device)
    if len(self.columns):
      wp.launch(_gather, dim=self.shape, inputs=[self.array, self._columns], outputs=[out], device=device)
    return host_copy(out)

  def assign(self, values: AnyDeviceArray | AnyHostArray) -> None:
    """Writes (nworld or 1, len(columns), ...) values (NumPy ones may broadcast to that shape).

    Raises:
      ValueError: for a Warp array of another dtype, device or shape.
    """
    if not len(self.columns):
      return
    device = self.array.device
    source = self._checked(values) if isinstance(values, wp.array) else self._uploaded(values)
    wp.launch(_scatter, dim=self.shape, inputs=[source, self._columns], outputs=[self.array], device=device)

  def _checked(self, values: AnyDeviceArray) -> AnyDeviceArray:
    """`values` when it is a (nworld or 1, ncolumn) array of the table's dtype and device."""
    a = self.array
    same = values.dtype == a.dtype and values.device == a.device
    if same and tuple(values.shape[1:]) == (len(self.columns),) and values.shape[0] in (1, a.shape[0]):
      return values
    raise ValueError(f"InstanceColumns.assign needs a (nworld or 1, {len(self.columns)}) {a.dtype} array on {a.device}")

  def _uploaded(self, values: AnyHostArray) -> AnyDeviceArray:
    """NumPy `values` broadcast to (nworld, ncolumn, ...) on the table's device (flat elements,
    e.g. 9 floats per mat33, are reshaped; (ncolumn, 3) vec3 values are every world's)."""
    inner = _inner_shape(self.array.dtype)
    v = np.asarray(values)
    if inner and v.shape[-len(inner) :] != inner:
      shape: tuple[int, ...] = (*v.shape[:-1], *inner)
      v = v.reshape(shape)  # flat elements
    full = np.ascontiguousarray(np.broadcast_to(v, self.shape + inner))
    return wp.array(full, dtype=self.array.dtype, device=self.array.device)


@final
class LiveCameras:
  """The mounted cameras' "camera.fovy" / "camera.intrinsics" from the binding's live
  scene-camera tables (`Binding.cameras`: MuJoCo's `m.cam_fovy` / `m.cam_intrinsic`, batched per
  world or edited in place), gathered every frame into Renderer-owned (rows, ncam) tables that
  start as the rig's defaults (free cameras keep theirs). One launch per field and frame (two
  with free cameras beside mounted ones); a declared per-world field wins over them.

  Attributes:
    defaults: field -> the rig's (1, ncam) table.
    tables: field -> the gathered (rows, ncam) table of the last `write` (rows: the live table's).
  """

  def __init__(self, defaults: Mapping[str, AnyDeviceArray], mount: Ids[int]) -> None:
    """The gather of the rig cameras mounted on scene cameras `mount` (ncam,) (-1: free)."""
    self.defaults = dict(defaults)
    mount = np.asarray(mount, np.int32)
    device = next(iter(self.defaults.values())).device
    rig = np.flatnonzero(mount >= 0).astype(np.int32)
    self._all_mounted = len(rig) == len(mount)
    self._scene_cols = wp.array(mount[rig], dtype=int, device=device)
    self._rig_cols = wp.array(rig, dtype=int, device=device)
    self.tables: dict[str, AnyDeviceArray] = {}
    self._gathered: dict[str, AnyDeviceArray] = {}

  def write(self, live: Mapping[str, AnyDeviceArray]) -> None:
    """Gathers the mounted columns of each live table (device work; a table of another row count
    than the last allocates its buffers, once)."""
    for field, source in live.items():
      default = self.defaults.get(field)
      if default is None:
        continue
      table, gathered = self._buffers(field, default, int(source.shape[0]))
      dim = gathered.shape
      wp.launch(_gather, dim=dim, inputs=[source, self._scene_cols], outputs=[gathered], device=source.device)
      if gathered is not table:
        wp.launch(_scatter, dim=dim, inputs=[gathered, self._rig_cols], outputs=[table], device=source.device)

  def _buffers(self, field: str, default: AnyDeviceArray, rows: int) -> tuple[AnyDeviceArray, AnyDeviceArray]:
    """The (rows, ncam) table of `field` and the (rows, nmounted) gather target (the table
    itself when every camera is mounted)."""
    table = self.tables.get(field)
    if table is None or table.shape[0] != rows:
      values = np.repeat(host_copy(default)[:1], rows, axis=0)
      table = self.tables[field] = wp.array(values, dtype=default.dtype, device=default.device)
      self._gathered[field] = (
        table if self._all_mounted else wp.empty((rows, self._scene_cols.shape[0]), dtype=default.dtype, device=default.device)
      )
    return table, self._gathered[field]


def _inner_shape(dtype: type) -> tuple[int, ...]:
  """The NumPy trailing shape of one element of `dtype` (() for scalars, (3,) for vec3, ...).

  Warp has no public accessor: its vector / matrix types carry their shape as `_shape_`.
  """
  shape = getattr(dtype, "_shape_", None)
  return tuple(shape) if shape else ()


def _check_overridable(field: str) -> None:
  """Raises ValueError for a pose or an unknown per-world field."""
  if field in bind.FIXED_SLOTS:
    raise ValueError(f"{field} is a pose: write it through the simulator state or camera_pose")
  if field in DERIVED_SLOTS:
    raise ValueError(f"{field} is derived per frame from {DERIVED_SLOTS[field]}: declare and write that")
  if field not in bind.SLOTS and field not in RENDERER_FIELDS and field not in DERIVED_FIELDS:
    known = sorted(set(bind.SLOTS) - bind.FIXED_SLOTS - set(DERIVED_SLOTS) | set(RENDERER_FIELDS) | set(DERIVED_FIELDS))
    raise ValueError(f"unknown per-world field {field!r}; one of {known}")


def _panels_off(scene: GpuScene) -> wp.array[bool, [int, int]] | None:
  """The scene's "light.active" table with the emissive-panel lights off (None: no panel light)."""
  panels = scene.light_instance >= 0
  active = scene.tables.get("light.active")
  if not np.any(panels) or active is None:
    return None
  values = host_copy(active).copy()
  values[:, panels] = False
  return wp.array(values, dtype=bool, device=active.device)


@final
class PerWorldTables:
  """The per-world tables of a Renderer (see the module docstring) and its own slot buffers.

  Attributes:
    overrides: declared field -> its (nworld, n) table (wins over the scene / binding value).
    defaults: the renderer-level fields' values when not declared.
    poses: the renderer's pose buffers of per-world instances (pose-writing bindings).
    visible: the (nworld, nvisible) "instance.visible" table by `Layout.visible_slot` (None when
      not declared).
    shared_table: the shared pose table of a compact layout ("shared.pos", "shared.rot").
    panels_off: the scene's "light.active" with its emissive-panel lights off
      (`lighting.emissive_lights`; None without panel lights).
    derived: the declared derived fields (`derived_fields.py`; their emission tint maps).
  """

  def __init__(  # noqa: PLR0913  # keyword-only: the tables of `build`
    self,
    *,
    nworld: int,
    layout: sharing.Layout,
    names: Sequence[str],
    defaults: Defaults,
    poses: PoseBuffers | None,
    visible: wp.array[int, [int, int]] | None,
    derived: DerivedTables,
  ) -> None:
    """No override yet (`declare` adds them)."""
    self.nworld = nworld
    self.mapped_diffuse = False
    self.derived = derived
    self.layout = layout
    self.names = tuple(names)
    self.defaults = defaults
    self.poses = poses
    self.visible = visible
    self.overrides: dict[str, AnyDeviceArray] = {}
    # The column views `get` made, by (field, patterns) (`bind` replaces a table: cleared).
    self._views: dict[tuple[str, object], InstanceColumns] = {}
    self.shared_table: tuple[wp.array[wp.vec3, [int]], wp.array[wp.mat33, [int]]] | None = None
    self.panels_off: wp.array[bool, [int, int]] | None = None
    self.live_cameras: LiveCameras | None = None
    # The per-world instance optics (`worlds/optics.py`; None: the scene's one row).
    self.optics: WorldOptics | None = None
    # The per-world instance colors with the binding's hiding (`write_colors`; None: not
    # declared, or a binding that hides nothing) and the frame's slots it replaces.
    self.shown_color: wp.array[wp.vec4, [int, int]] | None = None
    self.composed: dict[str, AnyDeviceArray] = {}

  @staticmethod
  def build(  # noqa: PLR0913  # keyword-only
    scene: GpuScene,
    *,
    nworld: int,
    layout: sharing.Layout,
    names: Sequence[str],
    declared: frozenset[str],
    defaults: Defaults,
    mount: Ids[int],
  ) -> PerWorldTables:
    """The tables of a renderer of `scene` on the scene's device: its binding's pose buffers
    (`Binding.allocate`), the shared pose table (compact layouts), the visibility table
    (declared), the panel switch, the derived fields' (`declared`: the per-world fields; `names`:
    the instance names), the live camera tables of the cameras that are scene cameras `mount`
    (ncam,) as authored (-1: none; `LiveCameras`, when the binding has live ones)."""
    binding = scene.binding
    dev = scene.device
    visible = wp.ones((nworld, layout.nvisible), dtype=int, device=dev) if layout.nvisible else None
    poses = binding.allocate(nworld, layout.columns, device=dev)
    derived = DerivedTables(scene, declared)
    t = PerWorldTables(
      nworld=nworld, layout=layout, names=names, defaults=defaults, poses=poses, visible=visible, derived=derived
    )
    t.mapped_diffuse = "material.base_diffuse_roughness" in declared and bool(
      declared & {"material.texture", "material.tex_affine"}
    )
    if layout.compact:
      k = max(len(layout.shared), 1)
      pos, rot = wp.zeros(k, dtype=wp.vec3, device=dev), wp.zeros(k, dtype=wp.mat33, device=dev)
      t.shared_table = (pos, rot)
      if len(layout.shared):
        binding.write_shared(layout.shared, pos=pos, rot=rot, sim=None)
    t.panels_off = _panels_off(scene)
    inputs = switched(scene, declared)
    if inputs is not None:
      t.optics = WorldOptics(every_closed(scene, inputs), nworld, scene.ninstance, dev)
    if "instance.color" in declared and binding.hides_by_color:
      t.shown_color = wp.zeros((nworld, scene.ninstance), dtype=wp.vec4, device=dev)
    if np.any(np.asarray(mount) >= 0) and binding.cameras(None):
      t.live_cameras = LiveCameras(
        {"camera.fovy": defaults.camera_fovy, "camera.intrinsics": defaults.camera_intrinsics}, mount
      )
    return t

  def scene_slots[S](self, scene: GpuScene[S], sim: S | None, *, emissive_lights: bool) -> Slots:
    """The slots of a call without the per-world overrides (scene tables < binding < own
    buffers)."""
    slots: Slots = {**scene.tables, **scene.binding.slots(sim)}
    if self.poses is not None:
      slots["instance.pos"], slots["instance.rot"] = self.poses.pos, self.poses.rot
    if self.shared_table is not None:
      slots["shared.pos"], slots["shared.rot"] = self.shared_table
    if self.visible is not None:
      slots["instance.visible"] = self.visible
    if not emissive_lights and self.panels_off is not None:
      slots["light.active"] = self.panels_off
    if self.derived.emission_maps is not None:
      slots["material.texture"], slots["material.tex_affine"] = self.derived.emission_maps
    return bind.merged(slots, self.derived.material_inputs())

  def instance_material[S](self, scene: GpuScene[S], sim: S | None) -> AnyDeviceArray | None:
    """The call's `instance.material` table (the per-world one when declared; None: none)."""
    over = self.overrides.get("instance.material")
    if over is not None:
      return over
    return self._slot_table(scene, scene.binding.slots(sim), "instance.material")

  def write_colors[S](self, scene: GpuScene[S], sim: S | None) -> None:
    """The call's instance colors: the per-world table with the binding's hiding applied
    (`shown_color`: the pooled objects a world does not show stay hidden whatever the table
    says), when the binding serves colors this call (nothing otherwise)."""
    if self.shown_color is None:
      return
    live = bind.slot(scene.binding.slots(sim), "instance.color")
    if live is None:
      self.composed = {}
      return
    color = self.overrides["instance.color"]
    wp.launch(_keep_hidden, dim=self.shown_color.shape, inputs=[color, live], outputs=[self.shown_color], device=color.device)
    self.composed = {"instance.color": self.shown_color}

  def _slot_table[S](self, scene: GpuScene[S], live: Slots, field: str) -> AnyDeviceArray | None:
    """The call's table of slot `field`: the colors with the binding's hiding (`write_colors`),
    the per-world one when declared, else the binding's `live` slot, else the scene's (None:
    none)."""
    over = self.composed[field] if field in self.composed else self.overrides.get(field)
    if over is not None:
      return over
    found = bind.slot(live, field)
    return found if found is not None else bind.slot(scene.tables, field)

  def write_optics[S](self, scene: GpuScene[S], sim: S | None, translucent: Literal["off", "transmittance", "clear"]) -> None:
    """Recomposes the per-world instance optics from the call's material choice, colors and
    refraction indices (`optics`; nothing without them)."""
    if self.optics is None:
      return
    live = scene.binding.slots(sim)
    material = self._slot_table(scene, live, "instance.material")
    color, base = self._slot_table(scene, live, "instance.color"), self._slot_table(scene, live, "material.base_color")
    if material is None or color is None or base is None:
      return
    ior = self.derived.table("material.specular_ior")
    self.optics.write(OpticsTables(material, color, base, ior), translucent)

  def renderer_default(self, field: str) -> AnyDeviceArray:
    """The (1,) default row of renderer-level field `field` (written by the Renderer)."""
    return self.defaults.value(field, env_lit=False)

  def renderer_field(self, field: str) -> AnyDeviceArray:
    """The table of renderer-level field `field` a frame reads: its per-world override, else its
    default row."""
    return self.overrides.get(field, self.renderer_default(field))

  def camera_field(self, field: str) -> AnyDeviceArray:
    """The table of camera field `field` ("camera.fovy", "camera.intrinsics",
    "camera.distortion") a frame reads: its per-world override, else the live scene cameras'
    (`LiveCameras`), else its default row (as `frame`)."""
    live = self.live_cameras.tables if self.live_cameras is not None else {}
    return self.overrides.get(field, live.get(field, self.renderer_default(field)))

  def frame(self, slots: Slots, *, absent: Slots, env_lit: bool) -> Frame:
    """The kernel fields of a call: its `slots` over the `absent` placeholders, the per-world
    overrides, and the renderer-level fields (override, else default)."""
    over, d = self.overrides, self.defaults
    live = self.live_cameras.tables if self.live_cameras is not None else {}
    # One plain dict read per slot (this runs every render: no per-slot helper call).
    arrays = cast("dict[str, AnyDeviceArray]", {**absent, **slots})
    arrays.update({name: a for name, a in over.items() if name in SLOTS})
    arrays.update(self.composed)
    return construct_named(
      Frame,
      {field: arrays.get(name) for name, field in SLOTS.items()},
      exposure=over.get("exposure", d.exposure),
      ambient=over.get("ambient", d.value("ambient", env_lit=env_lit)),
      env_id=over.get("env.id", d.env_id),
      env_intensity=over.get("env.intensity", d.env_intensity),
      env_yaw=over.get("env.yaw", d.env_yaw),
      camera_fovy=over.get("camera.fovy", live.get("camera.fovy", d.camera_fovy)),
      camera_intrinsics=over.get("camera.intrinsics", live.get("camera.intrinsics", d.camera_intrinsics)),
      camera_distortion=over.get("camera.distortion", d.camera_distortion),
      fog_color=over.get("fog.color", d.fog_color),
      fog_range=over.get("fog.range", d.fog_range),
    )

  def declare(self, field: str, *, scene_slots: Slots, env_lit: bool) -> AnyDeviceArray:
    """Allocates the per-world table of `field`, starting as its current value.

    Raises:
      ValueError: for an unknown or pose field, or a slot the scene does not have.
    """
    _check_overridable(field)
    if field in DERIVED_FIELDS:
      return self._declare_derived(field, scene_slots)
    if field in bind.SLOTS:
      base = bind.slot(scene_slots, field)
      if field == "material.base_diffuse_roughness" and (base is None or base.shape[1] == 0):
        colors = scene_slots.get("material.base_color")
        if colors is None:
          raise ValueError("diffuse roughness requires prepared material rows")
        array = wp.zeros((self.nworld, colors.shape[1]), dtype=float, device=colors.device)
        self.overrides[field] = array
        return array
      if base is None:
        live = " (here it follows the simulator's state: randomise it there, e.g. MuJoCo's batched m.light_pos / m.light_dir)"
        raise ValueError(f"{field} is not available on this scene" + (live if field in _SIMULATOR_LIGHT else ""))
    else:
      base = self.defaults.value(field, env_lit=env_lit)
    array = self.overrides[field] = self._rows(
      base, extra_texture_role=self.mapped_diffuse and field in {"material.texture", "material.tex_affine"}
    )
    if field == "instance.material" and self.derived.select_shading:
      self._declare_slot(SHADING_SLOT, scene_slots)
    return array

  def declare_live(self, scene_slots: Slots) -> None:
    """After the declared fields: the slots derived from the binding's live inputs
    (`derived_fields.reads_shininess`: the roughness from MuJoCo's batched shininess)."""
    if self.derived.shininess:
      self._declare_slot(SHININESS_SLOT, scene_slots)

  def _rows(self, base: AnyDeviceArray, *, extra_texture_role: bool = False) -> AnyDeviceArray:
    """A (nworld, ...) copy of the scene's `base` table: one row repeated, nworld rows as they
    are, K rows (1 < K < nworld: MuJoCo Warp's batched model) tiled as world % K, as the
    kernels read them; more rows than worlds would discard authored worlds, none leaves a world
    without values (both raise, before any read-back)."""
    if not 0 < base.shape[0] <= self.nworld:
      raise ValueError(f"per-world source needs 1 to {self.nworld} world rows; got {base.shape[0]}")
    values = host_copy(base)
    if extra_texture_role and base.shape[2] == NTEXROLE:
      if base.dtype is wp.int32:
        expanded = np.full((base.shape[0], base.shape[1], NOPENPBR_TEXROLE), -1, np.int32)
      elif base.dtype is tex_affine:
        expanded = np.tile(TEX_IDENTITY, (base.shape[0], base.shape[1], NOPENPBR_TEXROLE, 1, 1))
      else:
        raise TypeError("optional texture roles require int32 ids or tex_affine rows")
      expanded[:, :, :NTEXROLE] = values
      values = expanded
    if values.shape[0] != self.nworld:
      values = values[np.arange(self.nworld) % values.shape[0]]
    return wp.array(np.ascontiguousarray(values), dtype=base.dtype, device=base.device)

  def ranges(self) -> list[Range]:
    """The ranges the validation checks: the derived fields' (`DerivedTables.ranges`) and every
    declared table of `TABLE_RANGES` (its float components, `_components`)."""
    ranges = self.derived.ranges()
    for field, (low, high) in TABLE_RANGES.items():
      table = self.overrides.get(field)
      if table is not None and table.size and field not in self.derived.slots:
        ranges.append(Range(field, _components(table), low, high))
    return ranges

  def bind(self, field: str, array: AnyDeviceArray) -> bool:
    """Makes caller storage `array` the declared appearance / environment table of `field`
    (nothing copied or read); whether it replaced the table (False: already bound).

    Raises:
      ValueError: for a geometry or undeclared field, a column view, or an array of another
        shape, dtype or device, or not contiguous.
    """
    if field in sharing.FIELD_GROUP:
      raise ValueError("bind_per_world does not replace geometry storage; write the declared geometry views in place")
    _check_overridable(field)
    current = self.get(field, None)
    if isinstance(current, InstanceColumns):
      raise ValueError("bind_per_world requires a complete appearance or environment table")
    if array.shape != current.shape or array.dtype != current.dtype or array.device != current.device:
      raise ValueError(f"bind_per_world({field!r}) needs shape {current.shape}, dtype {current.dtype} on {current.device}")
    if not array.is_contiguous:
      raise ValueError("bind_per_world requires contiguous storage")
    if current is array:
      return False
    self.overrides[field] = array
    self.derived.bind(field, array)
    self._views.clear()
    return True

  def _declare_slot(self, d: DerivedSlot, scene_slots: Slots) -> None:
    """Allocates derived slot `d` once (shared by material selection and the derived fields'
    inputs)."""
    if d.slot in self.overrides:
      return
    base = bind.slot(scene_slots, d.slot)
    if base is None and d.slot == "instance.shading":  # MuJoCo has no initial shading bits.
      base = wp.zeros((1, len(self.names)), dtype=int, device=self.derived.device)
    if base is None:
      raise ValueError(f"derived {d.slot} is not available on this scene")
    self.overrides[d.slot] = self._rows(base)
    self.derived.add_slot(d, self.overrides[d.slot], base)

  def _declare_derived(self, field: str, scene_slots: Slots) -> AnyDeviceArray:
    """The user table of derived `field`, after the (nworld, n) tables of the slots it feeds."""
    for d in derived_fields.slots_of(field):
      self._declare_slot(d, scene_slots)
    array = self.overrides[field] = self.derived.declare(field, self.nworld)
    return array

  def get(self, field: str, instances: sharing.Patterns | None) -> AnyDeviceArray | InstanceColumns:
    """`Renderer.per_world.columns(field, instances)` (see its docstring). A column view is made
    once per (field, patterns) and kept (its device column ids are allocated and uploaded once:
    a training loop may ask for it every step).

    Raises:
      ValueError: for an undeclared field, or instances given for a non-instance field.
    """
    patterns = (instances,) if instances is None or isinstance(instances, (str, int, np.integer)) else tuple(instances)
    key = (field, patterns)
    view = self._views.get(key)
    if view is None:
      view = self._get(field, instances)
      if isinstance(view, InstanceColumns):
        self._views[key] = view
    return view

  def _get(self, field: str, instances: sharing.Patterns | None) -> AnyDeviceArray | InstanceColumns:
    """`get`, uncached."""
    if field in sharing.FIELD_GROUP:
      return self._geometry(field, instances)
    array = self.overrides.get(field) if field not in DERIVED_SLOTS and field not in self.derived.slots else None
    if array is None:
      _check_overridable(field)
      raise ValueError(f"per_world[{field!r}]: not declared; {_declare_hint(field)}")
    if instances is None:
      return array
    if not field.startswith("instance."):
      raise ValueError(f"per_world.columns({field!r}, instances): only instance fields take instances")
    idx = np.flatnonzero(sharing.match(self.names, instances, f"per_world.columns({field!r})"))
    return InstanceColumns(array, idx.astype(np.int32), [self.names[int(i)] for i in idx])

  def _geometry(self, field: str, instances: sharing.Patterns | None) -> AnyDeviceArray | InstanceColumns:
    """The per-world geometry table of `field` (instances' poses, sizes, meshes, visibility)."""
    if field == "instance.visible":
      return self._visible(instances)
    layout = self.layout
    mask = (
      np.ones(len(self.names), bool)
      if instances is None
      else sharing.match(self.names, instances, f"per_world.columns({field!r})")
    )
    sharing.check_request(layout, self.names, field=field, mask=mask)
    idx = np.flatnonzero(mask)
    names = [self.names[int(i)] for i in idx]
    cols = np.asarray(layout.pose_slot, np.int64)[idx] if layout.compact else idx
    if field in ("instance.pos", "instance.rot"):
      array = self._pose_table(field, idx)
    else:
      array = self.overrides.get(field)
      if array is None:
        raise ValueError(f"per_world[{field!r}]: not declared; {_declare_hint(field)}")
      cols = idx
    if instances is None:
      return array
    return InstanceColumns(array, np.asarray(cols, np.int32), names)

  def _visible(self, instances: sharing.Patterns | None) -> InstanceColumns:
    """The visibility columns of the matched instances (default: every declared one)."""
    field = "instance.visible"
    if self.visible is None:
      raise ValueError(f"per_world[{field!r}]: not declared; {_declare_hint(field)}")
    slot = self.layout.visible_slot
    mask = slot >= 0 if instances is None else sharing.match(self.names, instances, f"per_world.columns({field!r})")
    sharing.check_request(self.layout, self.names, field=field, mask=mask)
    idx = np.flatnonzero(mask)
    return InstanceColumns(self.visible, slot[idx].astype(np.int32), [self.names[int(i)] for i in idx])

  def _pose_table(self, field: str, idx: np.ndarray[[int], np.dtype[np.intp]]) -> AnyDeviceArray:
    """The renderer's position / rotation buffer of static per-world instances `idx`.

    Raises:
      ValueError: for poses that come from the simulator state.
    """
    if self.poses is None:
      raise ValueError(
        f"per_world.columns({field!r}): this scene's poses come from the simulator state (MuJoCo: declare the "
        "geom with PerWorld(pose=[...]) and write d.geom_xpos / d.geom_xmat per world)"
      )
    moved = idx[self.layout.dynamic[idx]]
    if len(moved):
      listed = ", ".join(repr(self.names[int(i)]) for i in moved[:5])
      raise ValueError(f"per_world.columns({field!r}): {listed} {'is' if len(moved) == 1 else 'are'} posed by the simulator")
    return self.poses.pos if field == "instance.pos" else self.poses.rot


@final
class PerWorldView(Mapping[str, "AnyDeviceArray | InstanceColumns"]):
  """`Renderer.per_world`: the declared per-world tables by qualified field name, owned by the
  renderer and written in place (`r.per_world["material.base_color"].assign(colors)`).

  `view[field]` is the field's (nworld, n) table ("instance.visible": the columns of every
  declared instance); `columns(field, instances)` the columns of the instances matching
  `PerWorld` patterns. The keys are the declared tables (per-world poses: `columns`);
  `print(view)` lists them with their shapes. What each field holds: `PerWorldTables.get`.
  """

  def __init__(self, tables: PerWorldTables) -> None:
    """The view of a Renderer's `tables`."""
    self._tables = tables

  def __getitem__(self, field: str) -> AnyDeviceArray | InstanceColumns:
    """The table of `field` (KeyError for an undeclared or unknown field, with what to
    declare)."""
    t = self._tables
    if field in ("instance.size", "instance.mesh") and field in t.overrides:
      return t.overrides[field]  # the whole (nworld, ninstance) table, shared columns included
    try:
      return t.get(field, None)
    except ValueError as e:
      raise KeyError(str(e)) from e

  def columns(self, field: str, instances: sharing.Patterns | None = None) -> InstanceColumns:
    """The columns of instance field `field` of the instances matching `instances` (None: every
    declared one, for "instance.visible").

    Raises:
      ValueError: for an undeclared field, a non-instance field, or patterns matching nothing
        declared.
    """
    view = self._tables.get(field, instances)
    if not isinstance(view, InstanceColumns):
      raise ValueError(f"per_world.columns({field!r}): pass instance patterns (or use per_world[{field!r}])")
    return view

  def __iter__(self) -> Iterator[str]:
    """The declared fields, sorted."""
    t = self._tables
    names = set(t.overrides) - set(DERIVED_SLOTS) - set(t.derived.slots)
    return iter(sorted(names | ({"instance.visible"} if t.visible is not None else set())))

  def __len__(self) -> int:
    """How many fields are declared."""
    return sum(1 for _ in self)

  def __repr__(self) -> str:
    """The declared fields with their table shapes."""
    rows = ", ".join(f"{k!r}: {tuple(self[k].shape)}" for k in self)
    return f"PerWorldView({{{rows}}})"
