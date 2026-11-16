"""Render outputs: the GPU arrays a Renderer writes, allocated once, and their views.

The output contract: every output is named as requested (`Outputs.rgb` -> `RenderOutputs.rgb`) and
is world-major, (nworld, ncam, H, W) with the per-pixel value as the element type (vec3, vec4ub,
...; NumPy / torch: a trailing channel axis). The one exception is forced by Warp's four-axis limit:
the planar color (`ColorOutput(planar=True)`) is (nworld * ncam, 3, H, W) in Warp, still
world-major (row w * ncam + c), and (nworld, ncam, 3, H, W) from `RenderOutputs.torch()`.

`Buffers` holds the flat kernel buffers (nworld, every camera's pixels) of the requested outputs
(`settings.Outputs`) and fills the kernels' output table and word (`structs.OutputArrays`: the
buffers' addresses in one struct for every output layout); `RenderOutputs` is what the caller
reads: dense (nworld, ncam, H, W) views when every camera has the Renderer's resolution,
per-camera views always. Which buffer backs which output, its dtype and whether a caller array
can be bound in its place come from one table, `buffer_specs`.
"""

from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Final, final

import numpy as np
import warp as wp

from raywarp._base.named import construct_named, field_value
from raywarp._base.types import LightType
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.output_layout import BUFFER_BITS, OUT_HDR_SQRT, BufferKind, FlowStatus, layout_of
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.scene.images.colorspace import REC709_Y, ColorSpace
from raywarp.scene.model.types import LightId

if TYPE_CHECKING:
  from shape_extensions import Int, IntTuple, IntVar
  from torch import Tensor

  from raywarp._base.dims import AnyDeviceArray, Images
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.settings import Outputs, ShadowOutput

# A flat kernel buffer: (nworld, every camera's pixels), or (nworld, 1) for an output that is off.
type Flat[D, NWorld: IntVar] = wp.array[D, [NWorld, int]]
# The channels-first color planes of `ColorOutput(planar=True)`: (nworld * ncam, 3, H, W) (Warp
# arrays have at most 4 dimensions; world-major, the batch axis of a policy).
type Planes = wp.array[wp.uint8, [int, 3, int, int]] | wp.array[wp.float16, [int, 3, int, int]]
# The display color in any `ColorOutput` layout.
type Color[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar] = (
  Images[wp.vec4ub, NWorld, NCam, H, W] | Images[wp.vec3ub, NWorld, NCam, H, W] | Planes
)
# A float output image in either precision.
type Image3[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar] = (
  Images[wp.vec3, NWorld, NCam, H, W] | Images[wp.vec3h, NWorld, NCam, H, W]
)
type Image1[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar] = (
  Images[wp.float32, NWorld, NCam, H, W] | Images[wp.float16, NWorld, NCam, H, W]
)


@final
@dataclasses.dataclass(frozen=True)
class RenderOutputs[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar]:
  """The requested outputs (None for the others), written in place by every render.

  "First hit": the nearest surface (translucent layers included, alpha-cut texels excluded) of the
  pixel-center ray. Geometry, guides and surface values stay aligned when color samples move
  through antialiasing. First-hit outputs are written every frame and never averaged (rgb, hdr
  and luminance are the averaged, anti-aliased ones).

  Attributes:
    rgb: the display color, sRGB, in the `ColorOutput` layout: (nworld, ncam, H, W) vec4ub RGBA
      (alpha 255, or with `ColorOutput(alpha=True)` the coverage: the fraction of the pixel's
      samples whose camera ray hit geometry, 0 where only background is seen) or vec3ub RGB
      (NumPy / torch: a trailing axis of 4 / 3 uint8), or planar (nworld * ncam, 3, H, W) uint8 /
      float16 in [0, 1] (`torch()`: (nworld, ncam, 3, H, W); per camera: (nworld, 3, H_c, W_c)),
      written by the kernels, no conversion pass.
    depth: float32 / float16 planar or radial depth (`DepthOutput`; distance along the ray for
      ray-bundle cameras), its miss value where nothing is hit.
    seg: int32 segment id of the first hit (`GpuScene.segment_id`; MuJoCo: instance id, flex f:
      ninstance + f), -1 on a miss.
    semantic: int32 class id of the first hit (`Renderer.set_semantics`; 1: unlabelled), 0 on a
      miss.
    hdr: vec3 linear radiance before exposure and tone mapping (the mean of the samples).
    albedo: vec3 linear base color of the first hit (zero on a miss).
    normal: vec3 world-space normal of the first hit, facing the camera: shading or geometric
      (`NormalOutput.frame`); zero on a miss.
    position: vec3 world-space point of the first hit [m]; zero on a miss.
    uv: vec2 texture coordinate the first hit's maps read (mesh, flex or primitive UVs with the
      material's tiling, or its world projection); zero without one.
    material: int32 material id of the first hit, -1 without one.
    roughness, metallic: float perceptual roughness and metallic of the first hit as shaded
      (textured; the roughness widened by specular AA at the pixel's footprint); zero on a miss.
    emission: vec3 emission term of the first hit's radiance (before exposure); zero on a miss.
    direct, indirect: vec3 radiance of the first hit towards the camera from the lights
      (shadowed) / from everything else except its emission (ambient, environment, GI,
      reflections); before exposure, zero on a miss. On hits of an opaque, unfogged spp-1 render:
      hdr = direct + indirect + emission (to float rounding).
    shadow: float visibility of the `ShadowOutput.light` at the first hit in [0, 1] (0 facing away
      from it; 1 on a miss or for a light that casts no shadow).
    luminance: uint8 / float16 sRGB-encoded Rec. 709 luminance of the display color.
    motion: vec2 image motion of the first hit to the previous frame [pixels, x right, y down]:
      exact for rigid instances (previous instance and camera poses), zero on the first frame and
      for flex, ray-bundle cameras or misses seen by a moving orthographic camera.
    flow_status: uint8 `FlowStatus` of each pixel's motion (`MotionOutput(status=True)`): VALID
      (the point was visible at the motion's target in the previous frame), MISS (background),
      OUT_OF_FRAME (outside the previous image or beyond its far plane), BEHIND (behind the
      previous camera's near plane), OCCLUDED (hidden by a nearer surface in the previous frame:
      its previous depth at the target, within one pixel, is nearer), UNKNOWN (no reprojection:
      the first frame after a reset, flex hits, moving ray-bundle cameras).
    cameras: the same outputs per camera, (nworld, 1, H_c, W_c) views of camera c's pixels. With
      per-camera resolutions (`CameraSpec.resolution`) the dense fields above are None and these
      are the outputs.
  """

  rgb: Color[NWorld, NCam, H, W] | None = None
  depth: Image1[NWorld, NCam, H, W] | None = None
  seg: Images[wp.int32, NWorld, NCam, H, W] | None = None
  semantic: Images[wp.int32, NWorld, NCam, H, W] | None = None
  hdr: Image3[NWorld, NCam, H, W] | None = None
  albedo: Image3[NWorld, NCam, H, W] | None = None
  normal: Image3[NWorld, NCam, H, W] | None = None
  position: Image3[NWorld, NCam, H, W] | None = None
  uv: Images[wp.vec2, NWorld, NCam, H, W] | Images[wp.vec2h, NWorld, NCam, H, W] | None = None
  material: Images[wp.int32, NWorld, NCam, H, W] | None = None
  roughness: Image1[NWorld, NCam, H, W] | None = None
  metallic: Image1[NWorld, NCam, H, W] | None = None
  emission: Image3[NWorld, NCam, H, W] | None = None
  direct: Image3[NWorld, NCam, H, W] | None = None
  indirect: Image3[NWorld, NCam, H, W] | None = None
  shadow: Image1[NWorld, NCam, H, W] | None = None
  luminance: Images[wp.uint8, NWorld, NCam, H, W] | Images[wp.float16, NWorld, NCam, H, W] | None = None
  motion: Images[wp.vec2, NWorld, NCam, H, W] | None = None
  flow_status: Images[wp.uint8, NWorld, NCam, H, W] | None = None
  cameras: tuple[RenderOutputs[NWorld, int, int, int], ...] = ()
  _torch: dict[str, Tensor] = dataclasses.field(default_factory=dict, repr=False, compare=False)
  # The `flow_status` values (`RenderOutputs.FlowStatus.OCCLUDED`, ...).
  FlowStatus: ClassVar[type[FlowStatus]] = FlowStatus

  def arrays(self) -> dict[str, AnyDeviceArray]:
    """The requested outputs by field name (the fields that are not None; not `cameras`)."""
    out: dict[str, AnyDeviceArray] = {}
    for name in VIEWS:
      a = field_value(self, name)
      if isinstance(a, wp.array):  # None for an output that is off
        out[name] = a
    return out

  @property
  def color_spaces(self) -> dict[str, ColorSpace]:
    """The color space of each requested output by field name (`OUTPUT_COLOR_SPACES`).

    Only the dense fields that are set: with per-camera resolutions they are None (empty map), and
    each camera entry has its own (`cameras[c].color_spaces`)."""
    return {name: OUTPUT_COLOR_SPACES[name] for name in self.arrays()}

  def torch(self) -> dict[str, Tensor]:
    """Zero-copy torch views of the outputs by name (created once, same memory every frame), all
    (nworld, ncam, ...): the planar color (nworld, ncam, 3, H, W).

    Raises:
      ValueError: with per-camera resolutions (no dense outputs: `cameras[c].torch()` each).
    """
    if not self._torch and not self.arrays() and any(c.arrays() for c in self.cameras):
      raise ValueError("the cameras have their own resolutions (no dense outputs): use outputs.cameras[c].torch()")
    if not self._torch:
      ncam = max(len(self.cameras), 1)
      for name, a in self.arrays().items():
        t = wp.to_torch(a)
        planar = name == "rgb" and a.dtype in (wp.uint8, wp.float16)
        self._torch[name] = t.reshape(t.shape[0] // ncam, ncam, *t.shape[1:]) if planar else t
    return self._torch


# The output views by name (`RenderOutputs` fields but `cameras`).
VIEWS: Final = tuple(f.name for f in dataclasses.fields(RenderOutputs) if f.name not in ("cameras", "_torch"))
# The color space of each output view: the display color (and its
# luminance) after exposure and the tone mapper is display-referred sRGB; radiance, its parts and
# albedo are scene-referred linear Rec.709; the geometric and material outputs are data.
OUTPUT_COLOR_SPACES: Final = types.MappingProxyType(
  {
    **dict.fromkeys(("rgb", "luminance"), ColorSpace.SRGB_REC709_DISPLAY),
    **dict.fromkeys(("hdr", "albedo", "emission", "direct", "indirect"), ColorSpace.LIN_REC709),
    **dict.fromkeys(
      (
        "depth",
        "seg",
        "semantic",
        "normal",
        "position",
        "uv",
        "material",
        "roughness",
        "metallic",
        "shadow",
        "motion",
        "flow_status",
      ),
      ColorSpace.DATA,
    ),
  }
)
if set(OUTPUT_COLOR_SPACES) != set(VIEWS):
  raise ImportError(f"OUTPUT_COLOR_SPACES {sorted(OUTPUT_COLOR_SPACES)} and the views {sorted(VIEWS)} differ")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class BufferSpec:
  """One flat kernel buffer: its element type, whether it is full size, what it shows.

  Attributes:
    name: the buffer (its `structs.OutputTable` field is `OutputLayout.field`'s).
    dtype: the element type.
    on: allocated full size (else an (nworld, 1) placeholder of an always-present field).
    view: the `RenderOutputs` field showing it, the name `bind` takes it by (None: none, not
      bindable).
    channels: channel planes per pixel (3 for the "chw" planes, else 1).
  """

  name: BufferKind
  dtype: type
  on: bool
  view: str | None
  channels: int = 1


# The buffers always allocated, as (nworld, 1) placeholders when off (the passes after the render
# read them by name: the coverage, depth, segmentation, the deferred display's image).
_BASE: Final[tuple[BufferKind, ...]] = ("rgb", "depth", "seg", "hdr")


def buffer_specs(outputs: Outputs) -> tuple[BufferSpec, ...]:
  """The kernel buffers of `outputs` (in its output word, `layout_of`): the always-present ones
  (`_BASE`), then the word's other buffers."""
  layout, kinds, rgb = layout_of(outputs), outputs.kinds, outputs.rgb
  # Semantic ids alone are written into the segmentation buffer (`output_layout.OUT_SEG_SEMANTIC`).
  seg_view = "seg" if "seg" in kinds else "semantic" if "semantic" in kinds else None
  views = {
    "rgb": "rgb" if rgb is not None and layout.has("rgb") else None,
    "seg": seg_view,
    # The deferred display's internal image: no view.
    "hdr": None if layout.bits & OUT_HDR_SQRT else "hdr",
  }
  base = tuple(BufferSpec(name=k, dtype=layout.dtype(k), on=layout.has(k), view=views.get(k, k)) for k in _BASE)
  # The other buffers show as their output, the RGB bytes and planes as "rgb" (planes: 3 channels).
  color = {"rgb_hwc": 1, "chw8": 3, "chw16": 3}
  extra = tuple(
    BufferSpec(name=k, dtype=layout.dtype(k), on=True, view="rgb" if k in color else k, channels=color.get(k, 1))
    for k in BUFFER_BITS
    if k not in _BASE and layout.has(k)
  )
  return base + extra


def _alias[D, NWorld: IntVar, N: IntVar](
  array: AnyDeviceArray, dtype: type[D], shape: IntTuple[NWorld, N], device: wp.Device
) -> wp.array[D, [NWorld, N]]:
  """`array`'s memory viewed as `dtype` with `shape` (same element size and count, contiguous).

  `wp.array.view` keeps a reference to `array`, so the view keeps the caller's memory alive.

  Raises:
    ValueError: for an incompatible array (layout, element size, count, device).
  """
  _check_alias(array, dtype=dtype, shape=shape, device=device)
  # An equal-size dtype view keeps the shape, strides and owner: no flat intermediate (a valid
  # multi-axis output can exceed Warp's signed 32-bit extent of a single axis).
  return array.view(dtype).reshape(shape)


# Element types whose bytes a bound output buffer may reinterpret (integers of one size: an RGBA
# uint32 / int32 / vec4ub, int32 / uint32 ids). A float output needs its own dtype: an integer (or
# other float) array of its size would receive its bit patterns.
_INTEGER_ELEMENTS: Final = (wp.bool, wp.int8, wp.uint8, wp.int16, wp.uint16, wp.int32, wp.uint32, wp.vec3ub, wp.vec4ub)


def _integer(dtype: type) -> bool:
  """Whether `dtype` is an integer element of `_INTEGER_ELEMENTS` (by structure: a
  `wp.types.vector(4, wp.uint8)` is a vec4ub)."""
  return any(wp.types.types_equal(dtype, t) for t in _INTEGER_ELEMENTS)


def _check_alias(array: AnyDeviceArray, *, dtype: type, shape: tuple[int, int], device: wp.Device) -> None:
  """Checks a binding before any buffer changes: a contiguous array of `shape`'s element count
  on `device`, of `dtype` (or, for an integer `dtype`, any integer element of its size).

  Raises:
    ValueError: for an incompatible array.
  """
  if not array.is_contiguous:
    raise ValueError("output arrays must be contiguous")
  same = wp.types.types_equal(array.dtype, dtype) or (_integer(array.dtype) and _integer(dtype))
  if not same or wp.types.type_size_in_bytes(array.dtype) != wp.types.type_size_in_bytes(dtype):
    raise ValueError(f"an output array of {wp.types.type_repr(array.dtype)} cannot hold {wp.types.type_repr(dtype)} elements")
  if array.size != math.prod(shape):
    raise ValueError(f"output array of {array.size} elements, the buffer needs {math.prod(shape)}")
  if wp.get_device(array.device) != device:
    raise ValueError(f"output array on {array.device}, the renderer renders on {device}")


def camera_view[D, NWorld: IntVar, H: IntVar, W: IntVar, C: IntVar](
  flat: Flat[D, NWorld], *, adr: int, width: Int[W], height: Int[H], channels: Int[C] = 1
) -> wp.array[D, [NWorld, C, H, W]]:
  """(nworld, 1, height, width) strided view of one camera's pixels at `adr` of a flat buffer;
  with `channels` C > 1, the (nworld, C, height, width) view of its channel planes at C x `adr`.

  The view retains `flat`, including caller-owned storage, even after renderer close/rebinding.

  Raises:
    ValueError: for unallocated/noncontiguous storage or a camera span outside its world row.
  """
  if flat.ptr is None:
    raise ValueError("camera_view needs an allocated buffer")
  if not flat.is_contiguous:
    raise ValueError("camera_view needs contiguous flat storage")
  if adr < 0 or width <= 0 or height <= 0 or channels <= 0 or (adr + width * height) * channels > flat.shape[1]:
    raise ValueError("camera_view dimensions and offset must fit one flat world row")
  size = wp.types.type_size_in_bytes(flat.dtype)
  # Flat buffers are contiguous (allocated here or bound through `_alias`): rows of shape[1]
  # elements.
  row = flat.shape[1] * size
  plane = height * width * size
  shape, strides = (flat.shape[0], channels, height, width), (row, plane, width * size, size)
  view = wp.array(ptr=flat.ptr + adr * channels * size, dtype=flat.dtype, shape=shape, strides=strides, device=flat.device)
  # Warp reshape/slice/view use this same owner link; raw-pointer construction does not.
  view._ref = flat
  return view


@final
class Buffers[NWorld: IntVar]:
  """Flat kernel buffers (nworld, every camera's pixels) behind the `RenderOutputs` views.

  Each buffer is allocated here, or (`bind`) a view of a caller array the kernels then write
  directly. Off base buffers are (nworld, 1) placeholders; the other buffers exist only when
  requested. Camera c's pixels start at `adr[c]` (`kernel_structs.Lens.adr`).

  Attributes:
    layout: the output word of the outputs (`output_layout.layout_of`): what the kernels store.
    device: the device of the buffers.
    sizes: each camera's (width, height).
    adr: each camera's first pixel in a world's row.
    shape: (nworld, ncam, H, W) of the dense outputs, None with per-camera resolutions.
    flat: the kernel buffers by struct field name.
    motion: the motion buffer (written through the temporal state), None when not requested.
    bound: the caller arrays the buffers are views of (`bind`).
    outputs: the views of the requested outputs.
    wants: the outputs the cameras write (`structs.WANT_*` bits).
  """

  def __init__(
    self,
    outputs: Outputs,
    *,
    nworld: Int[NWorld],
    sizes: Sequence[tuple[int, int]],
    device: wp.Device,
    arrays: Mapping[str, AnyDeviceArray] | None = None,
  ) -> None:
    """Allocates (or binds, `arrays`) the buffers of `outputs` for cameras of `sizes`
    ((width, height) each) on `device`, and the kernels' table of them."""
    self.layout = layout_of(outputs)
    self.device = device
    self.sizes = list(sizes)
    self.adr = [sum(w * h for w, h in self.sizes[:c]) for c in range(len(self.sizes))]
    total = sum(w * h for w, h in self.sizes)
    w0, h0 = self.sizes[0]
    self.shape = (nworld, len(self.sizes), h0, w0) if len(set(self.sizes)) == 1 else None
    self._flat = (nworld, total)
    self._specs = {b.name: b for b in buffer_specs(outputs)}
    self._bindable = self.bindable(outputs)
    self._own: dict[str, AnyDeviceArray] = {}
    self._struct: tuple[tuple[object, ...], kernel_structs.OutputArrays] | None = None
    self.motion: Flat[wp.vec2, NWorld] | None = None
    if outputs.motion is not None:
      self.motion = wp.zeros(self._flat, dtype=wp.vec2, device=device)
    seg = outputs.seg is not None or outputs.semantic is not None
    wants = (kernel_structs.WANT_RGB if outputs.shaded else 0) | (kernel_structs.WANT_DEPTH if outputs.depth is not None else 0)
    self.wants = wants | (kernel_structs.WANT_SEG if seg else 0)
    arrays = dict(arrays or {})
    self._check(arrays)
    # Bound buffers are views from the start (no own allocation made and dropped).
    self._assign(arrays)

  @staticmethod
  def bindable(outputs: Outputs) -> frozenset[str]:
    """The names `bind` accepts for `outputs`: every requested output's ("rgb": the color in its
    layout)."""
    return frozenset(b.view for b in buffer_specs(outputs) if b.on and b.view is not None)

  def _buffer(self, spec: BufferSpec, arrays: Mapping[str, AnyDeviceArray]) -> AnyDeviceArray:
    """Buffer `spec`: a view of the caller's array in `arrays` (by its output's name), else an
    own allocation."""
    shape = (self._flat[0], self._flat[1] * spec.channels)
    array = arrays.get(spec.view) if spec.on and spec.view is not None else None
    if array is not None:
      self._own.pop(spec.name, None)
      return _alias(array, spec.dtype, shape, self.device)
    own = self._own.get(spec.name)
    if own is None:
      own = self._own[spec.name] = wp.zeros(shape if spec.on else (self._flat[0], 1), dtype=spec.dtype, device=self.device)
    return own

  def bind(self, arrays: Mapping[str, AnyDeviceArray]) -> bool:
    """Makes the named buffers views of caller arrays, the others own allocations.

    `arrays`: output name (`bindable`: "rgb", "depth", "seg", "hdr", ...) -> array holding the
    (nworld, every camera's pixels) buffer (any shape of that size; a float output's own dtype, an
    integer output any integer element of its size: e.g. a (nworld, ncam, H, W) vec4ub or uint32
    array for an RGBA "rgb", vec3ub for a three-channel one,
    int32 for "seg"; a planar "rgb": 3 x as many uint8 or float16 elements, e.g. a policy's
    (nworld * ncam, 3, H, W) observation tensor). Allocates no device memory for bound buffers
    (an own buffer is allocated again when its output is unbound).

    Returns:
      Whether any buffer changed.

    Raises:
      TypeError: for something else than a Warp array.
      ValueError: for a buffer that is off or not bindable, or an incompatible array.
    """
    self._check(arrays)
    if {k: a.ptr for k, a in arrays.items()} == {k: a.ptr for k, a in self.bound.items()}:
      return False
    self._assign(arrays)
    return True

  def _check(self, arrays: Mapping[str, AnyDeviceArray]) -> None:
    """Raises for a name in `arrays` that is not a full-size buffer of these outputs (`bind`), or
    for something else than a Warp array (before `bind` reads its address)."""
    for name, array in arrays.items():
      if name not in self._bindable:
        raise ValueError(
          f"cannot bind output buffer {name!r}: it is off or not a buffer of these outputs (bindable: {sorted(self._bindable)})"
        )
      if not isinstance(array, wp.array):
        raise TypeError(f"output arrays must be Warp arrays, got {type(array).__name__}")
    for spec in self._specs.values():
      array = arrays.get(spec.view) if spec.on and spec.view is not None else None
      if array is not None:
        _check_alias(array, dtype=spec.dtype, shape=(self._flat[0], self._flat[1] * spec.channels), device=self.device)

  def _assign(self, arrays: Mapping[str, AnyDeviceArray]) -> None:
    """Points every buffer at its array in `arrays` (a view) or an own allocation, then rebuilds
    the output views."""
    self.flat: dict[BufferKind, AnyDeviceArray] = {name: self._buffer(spec, arrays) for name, spec in self._specs.items()}
    self.bound = dict(arrays)
    self.outputs = self._views()

  def flat_of(self, view: str) -> tuple[BufferKind, AnyDeviceArray] | None:
    """(buffer name, flat buffer) showing requested output `view` ("rgb": its color layout's
    buffer; "semantic": the seg buffer when the semantic ids are written alone), None when
    `view` is not requested."""
    for name, spec in self._specs.items():
      if spec.on and spec.view == view:
        return name, self.flat[name]
    return None

  def _flat_views(self) -> dict[str, tuple[AnyDeviceArray, int]]:
    """(flat buffer, channels) of each requested output, by `RenderOutputs` field."""
    flat: dict[str, tuple[AnyDeviceArray, int]] = {}
    for name, spec in self._specs.items():
      if spec.on and spec.view is not None:
        buffer = self.flat[name]
        flat[spec.view] = (buffer.view(wp.vec4ub) if name == "rgb" else buffer, spec.channels)
    if self.motion is not None:
      flat["motion"] = (self.motion, 1)
    return flat

  def _views(self) -> RenderOutputs[NWorld, int, int, int]:
    """The dense (nworld, ncam, H, W) views (uniform resolution) and the per-camera views."""
    flat = self._flat_views()
    cameras = tuple(
      construct_named(RenderOutputs, {k: camera_view(a, adr=adr, width=w, height=h, channels=c) for k, (a, c) in flat.items()})
      for adr, (w, h) in zip(self.adr, self.sizes, strict=True)
    )
    if self.shape is None:
      return RenderOutputs(cameras=cameras)
    nworld, ncam, h, w = self.shape
    dense = {k: a.reshape((nworld * ncam, c, h, w) if c > 1 else self.shape) for k, (a, c) in flat.items()}
    return construct_named(RenderOutputs, dense, cameras=cameras)

  def struct(self, tables: OutputTables) -> kernel_structs.OutputArrays:
    """The kernels' output struct of the buffers over `tables`: their table (`_table_of`), shape
    and output word. The last one while the buffers and tables are the same objects (several calls
    per frame)."""
    # The buffers by address (the table holds their addresses only: a re-bound buffer's view is a
    # new object whose id may be a freed view's), the tables' arrays by identity (the cached struct
    # holds them: no id is reused while it is cached), the other fields by value.
    fields = [getattr(tables, f.name) for f in dataclasses.fields(tables)]
    key = (*(b.ptr for b in self.flat.values()), *(id(v) if isinstance(v, wp.array) else v for v in fields))
    cached = self._struct
    if cached is not None and cached[0] == key:
      return cached[1]
    out = kernel_structs.OutputArrays()
    out.table = self._table_of(tables)
    out.rows, out.row = self._flat
    out.bits, out.half = self.layout.bits, self.layout.half
    # The buffers the table points to stay referenced by their owners (`flat`, the bound arrays).
    self._struct = (key, out)
    return out

  def _table_of(self, tables: OutputTables) -> kernel_structs.OutputTable:
    """The table of the buffers over `tables`: each buffer's address in its dtype's field (the
    buffers are contiguous (nworld, row) arrays: `_alias` checks a bound one).

    Raises:
      ValueError: for a buffer without device memory (the allocated ones always have it).
    """
    t = kernel_structs.OutputTable()
    t.wants = self.wants
    for name, buffer in self.flat.items():
      if buffer.ptr is None:
        raise ValueError(f"output buffer {name!r} has no device memory")
      setattr(t, self.layout.field(name), wp.uint64(buffer.ptr))
    t.world_skip = tables.world_skip
    t.segment_id = tables.segment_id
    t.flex_segment = tables.flex_segment
    t.semantic_id = tables.semantic_id
    t.semantic_flex = tables.semantic_flex
    t.shadow_light = tables.shadow_light
    return t


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutputTables:
  """The per-frame tables the output struct carries besides the buffers (`Buffers.struct`).

  Attributes:
    segment_id: (1 or nworld, ninstance) id the "seg" buffer gets per instance (segment ids, or
      class ids when semantic ids are written alone).
    flex_segment: deformable 0's id in that buffer.
    world_skip: worlds left untouched.
    semantic_id, semantic_flex: the class id table and the deformables' class (semantic buffer).
    shadow_light: the shadow output's light.
  """

  segment_id: wp.array[int, [int, int]]
  flex_segment: int
  world_skip: wp.array[bool, [int]]
  semantic_id: wp.array[int, [int, int]]
  semantic_flex: int
  shadow_light: LightId


def shadow_light[S](scene: GpuScene[S], shadow: ShadowOutput | None) -> LightId:
  """The light of `shadow.light` in `scene` (world 0's light tables, the binding's model tables
  over the scene's, read once, then the environment suns: light `source_lights` + k); light 0
  without a shadow output (never read). "sun": the first directional light, a table's before an
  environment sun; "strongest": by luminous strength (a sun's irradiance at the default
  environment intensity).

  Raises:
    ValueError: for an index out of range, "sun" without a directional light, or no light.
  """
  if shadow is None:
    return LightId(0)
  light, n = shadow.light, scene.nlight
  if n == 0:
    raise ValueError("outputs.shadow needs a scene with at least one light")
  if isinstance(light, int):
    if light >= n:
      raise ValueError(f"outputs.shadow.light {light} is out of range: the scene has {n} lights")
    return LightId(light)
  tables = {**scene.tables, **scene.binding.slots(None)}
  env = scene.environment
  if light == "sun":
    suns = [i for i, t in enumerate(_world0(tables, "light.type")) if t == int(LightType.DIRECTIONAL)]
    suns += range(scene.source_lights, n)
    if not suns:
      raise ValueError('outputs.shadow.light="sun" needs a directional light; the scene has none (pass an index)')
    return LightId(suns[0])
  color = np.asarray(_world0(tables, "light.color"), np.float64).reshape(-1, 3)
  power = np.asarray(_world0(tables, "light.strength"), np.float64) * (color @ np.asarray(REC709_Y))
  power = np.concatenate([power, env.intensity * (np.asarray(env.suns.irradiance, np.float64) @ np.asarray(REC709_Y))])
  return LightId(int(power.argmax()))


def _world0(tables: Mapping[str, object], name: str) -> list[object]:
  """World 0's row of light table `name` (a host read, once per Renderer)."""
  table = tables[name]
  if not isinstance(table, wp.array):
    raise TypeError(f"light table {name!r} is a {type(table).__name__}, not a Warp array")
  return list(host_copy(table)[0])
