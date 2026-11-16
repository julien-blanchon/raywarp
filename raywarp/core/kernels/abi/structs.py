"""The data a render kernel receives, packed into a few structs: render parameters, cameras,
environment, output buffers (one table, `OutputArrays`).

Compile-time settings are not here: they are `Features` constants (`wp.static`) of the
specialised kernel. Continuous settings reach the kernel through `Params` (a field of
`EnvironmentArrays`), so changing them needs no recompilation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Final, Literal, TypedDict

import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.tables import batched1d, in_table
from raywarp.core.kernels.abi.toggles import Toggles
from raywarp.core.kernels.generic import concrete, specialize
from raywarp.core.post.grading.look import Look
from raywarp.core.shade.gi.probes import Volume
from raywarp.core.trace.accel import query

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Constant-bound loops up to this length are unrolled by Warp (e.g. the per-light loop).
MAX_UNROLL: Final = 6
# `OutputTable.wants` and `Lens.wants` bits.
WANT_RGB: Final = 1
WANT_DEPTH: Final = 2
WANT_SEG: Final = 4


@wp.struct
class Params:
  """Runtime (continuous) render settings.

  Attributes:
    headlight_color: camera light color (`lighting.camera_light`).
    headlight_intensity: camera light illuminance times exposure (display-relative: the kernels
      divide it by each world's exposure).
    ambient: `lighting.fill` x `lighting.fill_color`, display-relative (the kernels divide
      it by each world's exposure: a radiance).
    inv_texel_angle: inverse angular size of an environment radiance texel.
    tone_mapper: the display transform (`settings.TONE_MAPPERS` id; `color.display_transform`).
    reflection_max_roughness, reflection_fade_roughness, reflection_min_weight: the glossy
      reflection trigger (`RenderSettings.reflections`; read only with glossy reflections
      compiled in).
    background_map: the environment map rays leaving the scene see (`scene.Environment.background`;
      -1: each world's lighting map); background_scale: its radiance scale (intensity x storage
      scale).
    max_layers: layers a camera ray composites (`transparency.max_layers`; a GI probe ray
      follows `probe._PROBE_LAYERS` hits per layer): the layer loops' bound at run time, so that
      the CUDA compiler keeps them loops instead of unrolling the surface shading per layer, and a
      change compiles nothing.
    spp: samples per pixel axis (`aa.spp`): the software path's per-frame sample loop bound of a
      multi-sample build (`Features.spp` 2), so that any n > 1 shares one kernel.
    skip_display: 0: the render kernels write the display color; 1: a deferred display pass
      (`post/display/deferred.py`) writes it from the HDR image (no coverage alpha to keep).
    relative_emission: 1: material emission is display-relative (MuJoCo: divided by the world's
      exposure, `tables.emission_radiance`); 0: radiance [nits] (host scenes).
    rt: the run-time settings of the build (`kernels/abi/toggles.py`; copied to the
      geometry struct, `GeometryFields.rt`, which every device function reads).
  """

  headlight_color: wp.vec3
  headlight_intensity: float
  ambient: wp.vec3
  inv_texel_angle: float
  background_map: int
  background_scale: float
  tone_mapper: int
  reflection_max_roughness: float
  reflection_fade_roughness: float
  reflection_min_weight: float
  max_layers: int
  spp: int
  skip_display: int
  relative_emission: int
  rt: Toggles


# A camera's per-world lens coefficients (`Cameras.camera_distortion`, the per-world
# "camera.distortion" field): OpenCV's (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4) of a
# perspective camera (`camera.distort`: radial, rational, tangential and thin prism terms), the
# Kannala-Brandt (k1, k2, k3, k4, then zeros) of a fisheye camera; ignored by orthographic cameras.
lens_coefficients = wp.types.vector(length=12, dtype=wp.float32)
# Entries of `lens_coefficients`.
NLENS: Final = 12


@wp.struct
class Lens:
  """One camera's constants (`Cameras.lens`, `cameras.CameraRig`): one table, one launch-parameter
  field for all of them (the OptiX launch parameters stay below
  `trace.optix.device.params.PARAMS_LIMIT`).

  Attributes:
    projection: `ProjectionType`.
    sensorsize: sensor size of cameras with intrinsics ((0, 0): fovy).
    znear: near clip distance.
    zfar: far clip distance (planar depth; distance along the ray for fisheye cameras and
      bundles; <= 0: unbounded).
    exposure: exposure scale of the camera (2^EV compensation; 1: the world's exposure).
    adr: first flat index of the camera's pixels in the output buffers (`render_outputs.Buffers`).
    wants: the `WANT_RGB | WANT_DEPTH | WANT_SEG` outputs the camera writes (`CameraSpec.outputs`;
      AND-ed with the Renderer's `OutputTable.wants`).
    scale: the image's scale n over the camera's resolution: 1, or the OptiX supersampled image's
      `aa.spp` (`CameraRig.scaled_lens`: its n x n pixels per output pixel, run-time data).
  """

  projection: int
  sensorsize: wp.vec2
  znear: float
  zfar: float
  exposure: float
  adr: int
  wants: int
  scale: int


@wp.struct
class Cameras:
  """Camera arrays of the cameras (indexed by camera) and the ray bundle."""

  lens: wp.array1d[Lens]
  camera_fovy: wp.array2d[float]
  camera_intrinsics: wp.array2d[wp.vec4]
  # (1 or nworld, ncam) lens coefficients (read by the lens models, `rays.CameraModel.distortion`,
  # and the reprojection, camera.project).
  camera_distortion: wp.array2d[lens_coefficients]
  camera_pos: wp.array2d[wp.vec3]
  camera_rot: wp.array2d[wp.mat33]
  nrender: int
  # Camera rays per world (every camera's pixels): the ray index range and pixel-seed stride.
  nray: int
  # Units of the render launch (its first dimension): the worlds, or the active list's entries.
  nworld: int
  # The render launch's world per unit (`active.ActiveWorlds.units`: the listed worlds, then -1);
  # empty: unit u renders world u (`camera.launch_world`).
  worlds: wp.array1d[int]
  # The frame's thread-to-pixel order (one entry: RenderOrder, TILED's band height; written by
  # `cameras.CameraRig`).
  order: wp.array1d[wp.vec2i]
  camera_res: wp.array1d[wp.vec2i]
  # Ray bundle (unused by pinhole cameras, see rays.make_camera_rays):
  ray_origin: wp.array1d[wp.vec3]
  ray_direction: wp.array1d[wp.vec3]
  ray_adr: wp.array1d[int]
  ray_res: wp.array1d[wp.vec2i]
  # (angle, distance) to the next bundle ray of the row (trace/rays/bundle.py update_cones).
  ray_cone: wp.array1d[wp.vec2]


@wp.struct
class EnvironmentArrays:
  """Exposure, ambient and environment-map arrays, the GI probe volume and the runtime
  parameters.

  `gi` is a placeholder without `Features.gi` (its lookups are compiled out).
  """

  exposure: wp.array1d[float]
  ambient_radiance: wp.array1d[wp.vec3]
  background: wp.vec3
  dfg_lut: wp.Texture2D
  env_radiance: wp.array1d[wp.Texture2D]
  env_prefiltered: wp.array1d[wp.Texture3D]
  env_sh: wp.array2d[wp.vec3]
  env_scale: wp.array1d[float]
  env_id: wp.array1d[int]
  env_intensity: wp.array1d[float]
  env_yaw: wp.array1d[float]
  # Per map: its sun cut out as a light (`EnvSuns`), drawn in the background
  # (`background.sun_disc`): (direction towards it in the map's frame, the cut-out's angular
  # radius [rad]; 0: none) and the disc's radiance (irradiance / solid angle).
  env_sun: wp.array1d[wp.vec4]
  env_sun_radiance: wp.array1d[wp.vec3]
  # Distance fog (Features.fog): linear color x exposure and (start, end, haze) per world.
  fog_color: wp.array1d[wp.vec3]
  fog_range: wp.array1d[wp.vec3]
  # The display transform's per-world constants (`grading_look.Look`, prepared by `grading.Grading`;
  # read by the graded display and the deferred display, `post/display/deferred.py`).
  look: wp.array1d[Look]
  # Camera 0's metered exposure of the previous frame per world (automatic exposure,
  # `post/exposure/auto.py`; 0 or an empty array: the world's exposure): the display-relative
  # ambient fill's (`ambient.display_exposure`).
  view_exposure: wp.array1d[float]
  gi: Volume
  params: Params
  # The build word's environment copy (`trace/frame_structs.word`: the GI bit and the classes
  # the ambient code reads; an OptiX module binds it, `trace/optix/device/params.bound_words`).
  bits: int
  bits2: int


@wp.struct
class OutputTable:
  """Every output buffer the kernels can write and the tables their stores read
  (`OutputArrays.table`), in the superset layout (`output_layout`: every buffer, every float16
  variant "<kind>16"): one struct for every output layout, so the launch parameters (and the PTX)
  are the same whatever the outputs. A buffer is the address of its (nworld, `OutputArrays.row`)
  contiguous elements (the planes: 3 x the row), its view built where it is stored (`vec3s`, ...:
  the strides follow from the row and the element type; the store reads 8 bytes of the launch
  parameters' constant bank, as a static layout's array field did); a buffer the word does not
  write is never read (0 or a placeholder). Camera c's pixels start at `Lens.adr`."""

  # WANT_RGB | WANT_DEPTH | WANT_SEG: the outputs the Renderer writes (one word; each camera's
  # subset is its `Lens.wants`).
  wants: int
  # Worlds left untouched (indexed world % len; the worlds `Renderer.render(active=...)` does not
  # list, `active.ActiveWorlds.skip`).
  world_skip: wp.array1d[bool]
  # Segment id per (world, instance), (1 or nworld, n) (the instance index without a scene
  # table), and of flex 0; the class id per (world, instance) and the deformables' class (the
  # semantic buffer); the shadow output's light.
  segment_id: wp.array2d[int]
  flex_segment: int
  semantic_id: wp.array2d[int]
  semantic_flex: int
  shadow_light: int
  # The display color: packed RGBA (uint32, its alpha byte the coverage), RGB bytes (vec3ub),
  # channels-first uint8 / float16 planes at 3 x the camera address (kernels/display.py).
  rgb: wp.uint64
  rgb_hwc: wp.uint64
  chw8: wp.uint64
  chw16: wp.uint64
  # float / float16; int (seg, semantic, material); vec3 / vec3h; vec2 / vec2h; uint8 / float16
  # (luminance); uint8 (`output_layout.FlowStatus` of the motion).
  depth: wp.uint64
  depth16: wp.uint64
  seg: wp.uint64
  semantic: wp.uint64
  hdr: wp.uint64
  hdr16: wp.uint64
  albedo: wp.uint64
  albedo16: wp.uint64
  normal: wp.uint64
  normal16: wp.uint64
  position: wp.uint64
  position16: wp.uint64
  uv: wp.uint64
  uv16: wp.uint64
  material: wp.uint64
  roughness: wp.uint64
  roughness16: wp.uint64
  metallic: wp.uint64
  metallic16: wp.uint64
  emission: wp.uint64
  emission16: wp.uint64
  direct: wp.uint64
  direct16: wp.uint64
  indirect: wp.uint64
  indirect16: wp.uint64
  shadow: wp.uint64
  shadow16: wp.uint64
  luminance: wp.uint64
  luminance16: wp.uint64
  flow_status: wp.uint64


@wp.struct
class OutputArrays:
  """The output struct every render kernel takes (an OptiX launch-parameter field): the buffer
  table, the buffers' shape and the output word's copies (`output_layout.OutputLayout`; read
  through `toggles.out_on` / `out_half`: an OptiX module binds them, a software megakernel module
  defines them, other kernels read them at run time).

  Attributes:
    table: the buffers' `OutputTable`.
    rows, row: every buffer's shape, (nworld, every camera's pixels) (the planes 3 x `row`).
    bits, half: the output word.
  """

  table: OutputTable
  rows: int
  row: int
  bits: int
  half: int


# Views of a buffer of the table (`OutputTable`) at its address `ptr`, by element type.
type View[T] = Callable[[OutputArrays, wp.uint64], wp.array2d[T]]
type PlanesView[T] = Callable[[OutputArrays, wp.uint64, int], wp.array2d[T]]


@specialize
def buffer_view[T](dtype: type[T]):
  """view(out, ptr) -> the `dtype` buffer of `out` at address `ptr`, one element per pixel."""

  @wp.func
  @concrete(**{"return": wp.array2d[dtype]})
  def view(
    # In:
    out: OutputArrays,
    ptr: wp.uint64,
  ):
    """The buffer at `ptr`."""
    return wp.array(ptr=ptr, shape=(out.rows, out.row), dtype=dtype)

  return view


@specialize
def planes_view[T](dtype: type[T]):
  """view(out, ptr, planes) -> the `dtype` buffer of `out` at address `ptr`, `planes` planes per
  pixel (the color planes: 3; luminance, status, depth: 1)."""

  @wp.func
  @concrete(**{"return": wp.array2d[dtype]})
  def view(
    # In:
    out: OutputArrays,
    ptr: wp.uint64,
    planes: int,
  ):
    """The buffer at `ptr`."""
    return wp.array(ptr=ptr, shape=(out.rows, out.row * planes), dtype=dtype)

  return view


u32s: Final = buffer_view(wp.uint32)  # the packed RGBA
rgb8s: Final = buffer_view(wp.vec3ub)  # the RGB bytes
u8s: Final = planes_view(wp.uint8)
f16s: Final = planes_view(wp.float16)
f32s: Final = buffer_view(float)
i32s: Final = buffer_view(int)  # seg, semantic, material
vec3s: Final = buffer_view(wp.vec3)
vec3hs: Final = buffer_view(wp.vec3h)
vec2s: Final = buffer_view(wp.vec2)
vec2hs: Final = buffer_view(wp.vec2h)


@wp.func
def skipped(
  # In:
  out: OutputArrays,
  worldid: int,
) -> bool:
  """Whether world `worldid` is skipped (its outputs keep their values)."""
  return batched1d(out.table.world_skip, worldid)


@wp.func
def env_lookup(
  # Model:
  env: EnvironmentArrays,
  # In:
  worldid: int,
) -> tuple[int, float, float]:
  """(environment map id, yaw, radiance scale) of a world.

  The radiance scale is the world's intensity times the map's storage scale (`Environments.scale`:
  maps beyond float16's range are stored divided by it). An out-of-range id reads the nearest map
  (reported by the per-frame validation, `core/pipeline/worlds/validator.py`).
  """
  e = in_table(batched1d(env.env_id, worldid), env.env_scale.shape[0])
  return e, batched1d(env.env_yaw, worldid), batched1d(env.env_intensity, worldid) * env.env_scale[e]


# Warp module options (`wp.set_module_options` / `wp.kernel(module_options=...)`).
type ModuleOptions = dict[str, bool | int | wp.ModuleBuildOptions]


def module_options(f: Features, *, outputs: bool = False) -> ModuleOptions:
  """Warp module options of the render megakernels: with a software build's words as preamble
  literals (`toggles.preamble`: the build word, and with `outputs`, a megakernel's, the output
  word; in the module hash, not the kernels', so each word is a module of its own with the same
  kernel symbols: `tools/prebuilt/software.py` links them all from one LTO-IR)."""
  options: ModuleOptions = {"fast_math": True, "max_unroll": MAX_UNROLL}
  options.update(query.module_options(f.trees, ordered=f.ordered_traversal))
  words = toggles.preamble(f, outputs=outputs)
  if words:
    defined = wp.ModuleBuildOptions(extra_cuda_preamble=words, extra_cpu_preamble=words)
    base = options.get("extra_build_options")
    options["extra_build_options"] = base.merged(defined) if isinstance(base, wp.ModuleBuildOptions) else defined
  return options


def words_free(f: Features) -> Features:
  """`f` without its words (`Features.word`, `out_word`): what its device functions are built from
  (the words are module preamble literals, `toggles.preamble`, never read by a factory)."""
  return dataclasses.replace(f, word=0, out_word=0)


def in_module_of(kernel: wp.Kernel, f: Features, *, outputs: bool) -> wp.Kernel:
  """`kernel`, built for `words_free(f)`, declared again in the unique module of `f`'s words
  (`module_options`, with `outputs`: a megakernel's): one Warp module per word, every word's
  kernel calling the same device functions (one function tree per build, the same module id as a
  kernel built for `f` itself)."""
  options = module_options(f, outputs=outputs)
  return wp.kernel(module="unique", module_options=options, **kernel.options)(kernel.func)


class KernelOptions(TypedDict, total=False):
  """`wp.kernel` keyword options of the render kernels."""

  module: Literal["unique"]
  enable_backward: bool
  grid_stride: bool
  module_options: ModuleOptions
  cuda_max_registers: int


def kernel_options(f: Features) -> KernelOptions:
  """`wp.kernel` options of the software megakernels (unique module, the output word defined,
  CUDA register cap)."""
  options: KernelOptions = {
    "module": "unique",
    "enable_backward": False,
    "grid_stride": False,
    "module_options": module_options(f, outputs=True),
  }
  if f.max_registers:
    options["cuda_max_registers"] = f.max_registers
  return options
