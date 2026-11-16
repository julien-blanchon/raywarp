"""The output word: which output stores the render kernels make, and in which precision.

Every kernel compiles every output (the superset) and selects at run time from the output word,
`OutputLayout`: its `bits` (`OUT_*`: one per store, conventions included) and `half` (the outputs
stored as float16, at their `OUT_*` bit). The kernels read the buffers from one table
(`structs.OutputTable`: every buffer and dtype variant's address, in the launch parameters)
through `structs.OutputArrays`, which holds the word's two copies (`bits`, `half`): an OptiX module
binds them (`trace/optix/device/params.bound_words`: the absent stores fold away), a software
megakernel module defines them (`toggles.out_on`). So an output change never changes the launch
parameters, the PTX or a software build's kernels, only the word. `layout_of` is the word of a
settings `Outputs`.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Final, Literal, cast, final

import warp as wp

from raywarp.core.kernels.abi.structs import OutputArrays, PlanesView, View, buffer_view, planes_view
from raywarp.core.kernels.generic import concrete, specialize
from raywarp.settings.fields import LAYOUTS, MISS_DEPTHS
from raywarp.settings.outputs import (
  DepthOutput,
  FloatOutput,
  LuminanceOutput,
  NormalOutput,
  Outputs,
  OutputSpec,
  ShadowOutput,
  is_kind,
)

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.settings import RenderSettings

# The output word's bits (`OutputLayout.bits`): one per store of the kernels. The display color
# (packed RGBA, RGB bytes, uint8 / float16 planes) and its coverage alpha:
OUT_RGBA: Final = 1 << 0
OUT_RGB8: Final = 1 << 1
OUT_CHW8: Final = 1 << 2
OUT_CHW16: Final = 1 << 3
OUT_ALPHA: Final = 1 << 4
# Depth and the segment ids (also per camera: `Lens.wants`), the HDR radiance (OUT_HDR_SQRT: its
# float16 square root, the deferred display's internal image), the guides (the shading normal, or
# OUT_GEOMETRIC the hit's geometric normal, into the "normal" buffer):
OUT_DEPTH: Final = 1 << 5
OUT_SEG: Final = 1 << 6
OUT_HDR: Final = 1 << 7
OUT_HDR_SQRT: Final = 1 << 8
OUT_ALBEDO: Final = 1 << 9
OUT_NORMAL: Final = 1 << 10
OUT_GEOMETRIC: Final = 1 << 11
# The hit outputs (`first_hit.make_write_hit`) and the recorded surface outputs:
OUT_SEMANTIC: Final = 1 << 12
OUT_POSITION: Final = 1 << 13
OUT_UV: Final = 1 << 14
OUT_MATERIAL: Final = 1 << 15
OUT_ROUGHNESS: Final = 1 << 16
OUT_METALLIC: Final = 1 << 17
OUT_EMISSION: Final = 1 << 18
OUT_DIRECT: Final = 1 << 19
OUT_INDIRECT: Final = 1 << 20
OUT_SHADOW: Final = 1 << 21
OUT_LUMINANCE: Final = 1 << 22
OUT_FLOW_STATUS: Final = 1 << 23
# The depth / segmentation conventions: radial depth (else planar), semantic class ids alone in
# "seg" (0 on a miss), the depth of a miss (inf, the camera's zfar; neither: 0).
OUT_DEPTH_RADIAL: Final = 1 << 24
OUT_SEG_SEMANTIC: Final = 1 << 25
OUT_MISS_INF: Final = 1 << 26
OUT_MISS_FAR: Final = 1 << 27
# Groups of bits the kernels test together: the hit outputs, the guides, the recorded surface
# outputs, every first-hit output (the pixel centre's surface is evaluated for them), the
# outputs whose pixel-centre values come from the full shading (direct / indirect).
OUT_HITS: Final = OUT_SEMANTIC | OUT_POSITION | OUT_UV | OUT_MATERIAL | OUT_GEOMETRIC
OUT_GUIDES: Final = OUT_ALBEDO | OUT_NORMAL
OUT_RECORDED: Final = OUT_ROUGHNESS | OUT_METALLIC | OUT_EMISSION | OUT_DIRECT | OUT_INDIRECT | OUT_SHADOW
OUT_FIRST: Final = OUT_HITS | OUT_GUIDES | OUT_RECORDED
OUT_SPLIT: Final = OUT_DIRECT | OUT_INDIRECT
# The color stores the supersampled image leaves to its downsample (`trace/optix/supersample.py`).
OUT_DISPLAY: Final = OUT_RGB8 | OUT_CHW8 | OUT_CHW16 | OUT_LUMINANCE

# An output buffer (`structs.OutputTable` field, `render_outputs.Buffers` name).
type BufferKind = Literal[
  "rgb",
  "rgb_hwc",
  "chw8",
  "chw16",
  "depth",
  "seg",
  "semantic",
  "hdr",
  "albedo",
  "normal",
  "position",
  "uv",
  "material",
  "roughness",
  "metallic",
  "emission",
  "direct",
  "indirect",
  "shadow",
  "luminance",
  "flow_status",
]
# The bits whose stores write each buffer (`OutputLayout.has`).
BUFFER_BITS: Final[dict[BufferKind, int]] = {
  "rgb": OUT_RGBA,
  "rgb_hwc": OUT_RGB8,
  "chw8": OUT_CHW8,
  "chw16": OUT_CHW16,
  "depth": OUT_DEPTH,
  "seg": OUT_SEG,
  "semantic": OUT_SEMANTIC,
  "hdr": OUT_HDR,
  "albedo": OUT_ALBEDO,
  "normal": OUT_NORMAL | OUT_GEOMETRIC,
  "position": OUT_POSITION,
  "uv": OUT_UV,
  "material": OUT_MATERIAL,
  "roughness": OUT_ROUGHNESS,
  "metallic": OUT_METALLIC,
  "emission": OUT_EMISSION,
  "direct": OUT_DIRECT,
  "indirect": OUT_INDIRECT,
  "shadow": OUT_SHADOW,
  "luminance": OUT_LUMINANCE,
  "flow_status": OUT_FLOW_STATUS,
}
# (float32 dtype, float16 dtype) of each buffer; the half bit of a float16-capable one is its
# first `BUFFER_BITS` bit (`HALF_BITS`; luminance: uint8 / float16; one dtype for the others).
DTYPES: Final[dict[BufferKind, tuple[type, type]]] = {
  "rgb": (wp.uint32, wp.uint32),
  "rgb_hwc": (wp.vec3ub, wp.vec3ub),
  "chw8": (wp.uint8, wp.uint8),
  "chw16": (wp.float16, wp.float16),
  "depth": (wp.float32, wp.float16),
  "seg": (wp.int32, wp.int32),
  "semantic": (wp.int32, wp.int32),
  "hdr": (wp.vec3, wp.vec3h),
  "albedo": (wp.vec3, wp.vec3h),
  "normal": (wp.vec3, wp.vec3h),
  "position": (wp.vec3, wp.vec3h),
  "uv": (wp.vec2, wp.vec2h),
  "material": (wp.int32, wp.int32),
  "roughness": (wp.float32, wp.float16),
  "metallic": (wp.float32, wp.float16),
  "emission": (wp.vec3, wp.vec3h),
  "direct": (wp.vec3, wp.vec3h),
  "indirect": (wp.vec3, wp.vec3h),
  "shadow": (wp.float32, wp.float16),
  "luminance": (wp.uint8, wp.float16),
  "flow_status": (wp.uint8, wp.uint8),
}
# The buffers with a float16 variant: their half bit (`OutputLayout.half`).
HALF_BITS: Final[dict[BufferKind, int]] = {
  k: BUFFER_BITS[k] & -BUFFER_BITS[k] for k, (full, half) in DTYPES.items() if full is not half
}
_MISS_BITS: Final = {MISS_DEPTHS["zero"]: 0, MISS_DEPTHS["inf"]: OUT_MISS_INF, MISS_DEPTHS["far"]: OUT_MISS_FAR}


class FlowStatus(enum.IntEnum):
  """Per-pixel status of the "motion" output: `RenderOutputs.flow_status` (uint8), written with
  `MotionOutput(status=True)`.

  UNKNOWN: no reprojection (the first frame after a reset, a deformable (flex) hit, a moving
    ray-bundle camera); the motion is 0.
  VALID: the first hit was visible at the motion's target in the previous frame.
  MISS: the pixel sees the background (no surface).
  OUT_OF_FRAME: the previous position is outside the previous image or beyond its far plane.
  BEHIND: the point was behind the previous camera's near plane (motion 0).
  OCCLUDED: the point was in the previous image but hidden by a nearer surface.
  """

  UNKNOWN = 0
  VALID = 1
  MISS = 2
  OUT_OF_FRAME = 3
  BEHIND = 4
  OCCLUDED = 5


@final
@dataclasses.dataclass(frozen=True, slots=True)
class OutputLayout:
  """An output word (see the module doc).

  Attributes:
    bits: the stores the kernels make (`OUT_*`).
    half: the buffers stored as float16, at their `HALF_BITS` bit.
  """

  bits: int = 0
  half: int = 0

  def has(self, kind: BufferKind) -> bool:
    """Whether buffer `kind` is written."""
    return (self.bits & BUFFER_BITS[kind]) != 0

  def is_half(self, kind: BufferKind) -> bool:
    """Whether buffer `kind` is stored as float16."""
    return (self.half & HALF_BITS.get(kind, 0)) != 0

  def dtype(self, kind: BufferKind) -> type:
    """The element dtype of buffer `kind` (`DTYPES`)."""
    full, half = DTYPES[kind]
    return half if self.is_half(kind) else full

  def field(self, kind: BufferKind) -> str:
    """The `structs.OutputTable` field holding buffer `kind`: its float16 variant's ("depth16",
    ...) when stored as float16."""
    return kind + "16" if self.is_half(kind) else kind

  @property
  def word(self) -> int:
    """The word as one integer (`half` above bit 32): the module records' and the software
    builds' value (`Features.out_word`)."""
    return self.bits | self.half << 32

  @staticmethod
  def of_word(word: int) -> OutputLayout:
    """The layout of word `word` (`word`'s inverse)."""
    return OutputLayout(bits=word & 0xFFFFFFFF, half=word >> 32)


def layout_of(out: Outputs) -> OutputLayout:
  """The output word of `out`: its stores (the packed RGBA of every shaded render but a
  three-channel color, the motion status, the conventions), its float16 buffers."""
  kinds = out.kinds
  rgb = out.rgb
  bits = sum(BUFFER_BITS[k] for k in BUFFER_BITS if k in kinds and k not in ("rgb", "normal"))
  if out.shaded and (rgb is None or rgb.channels == 4):
    bits |= OUT_RGBA  # also the coverage the downsample and the deferred display read
  if rgb is not None:
    color = {LAYOUTS["rgb8"]: OUT_RGB8, LAYOUTS["rgb8_planar"]: OUT_CHW8, LAYOUTS["rgb16f_planar"]: OUT_CHW16}
    bits |= color.get(rgb.color_layout, 0) | (OUT_ALPHA if rgb.alpha else 0)
  if out.normal is not None:
    bits |= OUT_GEOMETRIC if out.normal.frame == "geometric" else OUT_NORMAL
  if out.motion is not None and out.motion.status:
    bits |= OUT_FLOW_STATUS
  if out.semantic is not None and out.seg is None:
    # Semantic ids alone are written into the segmentation buffer (with instance ids: their own).
    bits = (bits & ~OUT_SEMANTIC) | OUT_SEG | OUT_SEG_SEMANTIC
  if out.depth is not None:
    bits |= (OUT_DEPTH_RADIAL if out.depth.convention == "radial" else 0) | _MISS_BITS[MISS_DEPTHS[out.depth.miss]]
  if isinstance(out.hdr, FloatOutput) and out.hdr.sqrt:
    bits |= OUT_HDR_SQRT
  # Every float16-capable buffer is an output's (`is_kind`); an output not requested: float32.
  half = sum(bit for k, bit in HALF_BITS.items() if is_kind(k) and _precision(out.spec(k)) == "float16")
  return OutputLayout(bits=bits, half=half)


def _precision(spec: OutputSpec | None) -> str:
  """The element precision of an output spec ("float32" for specs without one)."""
  match spec:
    case DepthOutput() | FloatOutput() | NormalOutput() | ShadowOutput() | LuminanceOutput():
      return spec.dtype
    case _:
      return "float32"


def check_outputs(s: RenderSettings, scene: GpuScene) -> None:
  """Raises ValueError for outputs the settings or the scene cannot produce."""
  out = s.outputs
  if (out.direct is not None or out.indirect is not None) and s.surfaces.shading == "unlit":
    raise ValueError('outputs.direct / outputs.indirect need lit surfaces (surfaces.shading="pbr"): unlit has no lights')
  if out.shadow is not None and scene.nlight == 0:
    raise ValueError("outputs.shadow needs a scene with at least one light")


def _views(value: type, half: type) -> tuple[View[object], View[object] | PlanesView[object], bool]:
  """(full view, half view, scalar) of the buffers of `value` / `half` elements: a scalar's
  float16 variant is a single-plane view."""
  scalar = value is float
  return buffer_view(value), planes_view(half) if scalar else buffer_view(half), scalar


@specialize
def make_put[V](value: type[V], half: type):
  """put(out, half, full, half_buffer, worldid, adr, v): stores `v` (a `value`) into its output
  (the `half` element type's variant with `half`: the word's bit; `full` / `half_buffer`: the two
  variants' addresses in the table, `structs.OutputTable`)."""
  full_view, half_view, scalar = _views(value, half)
  one, two = cast("PlanesView[object]", half_view), cast("View[object]", half_view)
  half_type = half

  @wp.func
  @concrete(v=value)
  def put(
    # Out:
    out: OutputArrays,
    # In:
    half: bool,
    full: wp.uint64,
    half_buffer: wp.uint64,
    worldid: int,
    adr: int,
    v: V,
  ):
    """Stores `v`."""
    if half:
      if wp.static(scalar):
        one(out, half_buffer, 1)[worldid, adr] = half_type(v)
      else:
        two(out, half_buffer)[worldid, adr] = half_type(v)
    else:
      full_view(out, full)[worldid, adr] = v

  return put


@specialize
def make_copy(value: type, half: type):
  """copy(hi, out, half, src_full, src_half, dst_full, dst_half, worldid, src, dst): copies an
  element of a `value` output from `hi` to `out` (the `half` element type's variant with
  `half`)."""
  full_view, half_view, scalar = _views(value, half)
  one, two = cast("PlanesView[object]", half_view), cast("View[object]", half_view)

  @wp.func
  def copy(
    # In:
    hi: OutputArrays,
    # Out:
    out: OutputArrays,
    # In:
    half: bool,
    src_full: wp.uint64,
    src_half: wp.uint64,
    dst_full: wp.uint64,
    dst_half: wp.uint64,
    worldid: int,
    src: int,
    dst: int,
  ):
    """Copies element `src` of `hi` to element `dst` of `out`."""
    if half:
      if wp.static(scalar):
        one(out, dst_half, 1)[worldid, dst] = one(hi, src_half, 1)[worldid, src]
      else:
        two(out, dst_half)[worldid, dst] = two(hi, src_half)[worldid, src]
    else:
      full_view(out, dst_full)[worldid, dst] = full_view(hi, src_full)[worldid, src]

  return copy


put3: Final = make_put(wp.vec3, wp.vec3h)
put2: Final = make_put(wp.vec2, wp.vec2h)
put1: Final = make_put(float, wp.float16)
