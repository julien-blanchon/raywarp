"""The launch parameters of an OptiX build: one struct per layout, read by every program.

OptiX passes the launch parameters by value (Warp's `external_constant_params` entry point:
`Params p = params;`). Above `PARAMS_LIMIT` bytes NVVM keeps that copy as a local array: every
thread copies the whole struct to local memory, which OptiX then saves across each trace (a
large interior scene went from 281 to 2737 ms at 512^2 x 64 x 3 when two unused array fields took
the struct from 8144 to 8256 bytes). So a build's struct holds only what its programs read
(`ParamsLayout`), and `tests/optix_backend/test_optix_host.py` checks every recipe's layout against
the limit. Upstream (Warp: launch parameters by reference) would lift the constraint.

The motion output's state (`Temporal`, array descriptors) is not a field but a one-element device
array the frame writes before its launches (`write_struct`), as the GI launches' structs: inline it
took a scene-free build past the limit.

`offset` is the first unit (world, ray or lattice point) of a launch chunk: `optixLaunch` takes at
most 2^30 indices, so wider launches are split into chunks of whole units
(`pipeline.Pipeline.chunks`, one parameter copy per chunk).
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import warnings
from typing import Final, Protocol, cast, final

import warp as wp

from raywarp._base.warp_structs import struct_mirror, warp_struct
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.output_layout import OutputLayout
from raywarp.core.kernels.abi.rays import OptixFields, optix_geometry_type
from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays
from raywarp.core.kernels.generic import specialize
from raywarp.core.post.temporal.state import Temporal
from raywarp.core.shade.gi import kernels as gi_kernels
from raywarp.core.shade.lights.flood import Pass
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import material_type
from raywarp.core.shade.shadows.area import AreaLaunch

# The GI programs' launch fields (`ParamsLayout.gi`): field -> launch struct.
GI_LAUNCHES: Final = {
  "gi_probe": gi_kernels.ProbeLaunch,
  "gi_static": gi_kernels.StaticLaunch,
  "gi_survey": gi_kernels.SurveyLaunch,
  "gi_walls": gi_kernels.WallsLaunch,
  "gi_ao": gi_kernels.AoLaunch,
}
# Launch-parameter structs above this size [bytes] run several times slower (see the module doc).
PARAMS_LIMIT: Final = 8192

# A struct field's Warp type: a struct class, an array annotation (`wp.array1d[int]`, not a Python
# `type`) or a scalar type, as `wp.struct` reads `__annotations__`.
type FieldType = object


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ParamsLayout:
  """What a build's launch parameters hold beyond the shading structs (`params_type`)."""

  # lit, env, seed and the material carrier as `geo.cut`: the shading structs (False: a
  # geometry-only build, whose `geo.cut` is a `Cutout`).
  shaded: bool = True
  history: bool = False  # tmp: the temporal state's one-element array (Features.history)
  grid: bool = False  # grid: light_flood.Pass (the light grid's visibility program)
  # GI_LAUNCHES: the GI bake's launches (`shade.gi.kernels`), one-element arrays (a descriptor each:
  # the parameters stay small); its programs write no output (no `out`: their modules bind no
  # output word).
  gi: bool = False
  # area: the soft area shadows' launch (`area_shadows.AreaLaunch`), a one-element array.
  area: bool = False
  world_curves: bool = False  # curve_instance
  world_primitives: bool = False  # prim_code
  shared_curves: bool = False  # shared_curve_instance
  shared_primitives: bool = False  # shared_prim_code
  mesh: bool = False  # face_normal
  live_material_uv: bool = False
  diffuse_roughness: bool = False
  specular_weight: bool = False
  layered: bool = False


class LaunchParams(Protocol):
  """The fields of the launch-parameter structs (`params_type`; the optional ones exist only in the
  layouts that hold them): what the programs and the renderer read and write."""

  geo: OptixFields  # its `cut`: the material carrier in a shaded build (`ParamsLayout.shaded`)
  lit: Lights
  cams: Cameras
  env: EnvironmentArrays
  out: OutputArrays  # the output table and the output word's copies (`structs.OutputArrays`)
  seed: wp.array[wp.int32, [1]]
  nworld: int
  offset: int  # the launch chunk's first unit
  tmp: wp.array[Temporal, [int]]  # one element, written on the device (`write_struct`)
  grid: Pass
  gi_probe: wp.array[gi_kernels.ProbeLaunch, [int]]
  gi_static: wp.array[gi_kernels.StaticLaunch, [int]]
  gi_survey: wp.array[gi_kernels.SurveyLaunch, [int]]
  gi_walls: wp.array[gi_kernels.WallsLaunch, [int]]
  gi_ao: wp.array[gi_kernels.AoLaunch, [int]]
  area: wp.array[AreaLaunch, [int]]
  curve_instance: wp.array[int, [int]]
  prim_code: wp.array[int, [int, int]]  # (nworld, nprim)
  shared_curve_instance: wp.array[int, [int]]
  shared_prim_code: wp.array[int, [int, int]]  # (1, nprim)
  face_normal: wp.array[wp.uint32, [int]]


# The launch parameters' last fields, in their order: layout flag -> (field, type).
_TABLES: Final[dict[str, tuple[str, FieldType]]] = {
  "area": ("area", wp.array1d[AreaLaunch]),
  "world_curves": ("curve_instance", wp.array1d[int]),
  "world_primitives": ("prim_code", wp.array2d[int]),
  "shared_curves": ("shared_curve_instance", wp.array1d[int]),
  "shared_primitives": ("shared_prim_code", wp.array2d[int]),
  "mesh": ("face_normal", wp.array1d[wp.uint32]),
}


def _fields(layout: ParamsLayout) -> dict[str, FieldType]:
  """Field name -> type of the launch parameters of `layout` (the `LaunchParams` it holds)."""
  carrier = None
  if layout.shaded:
    # The material carrier is the geometry's alpha-test tables (`geo.cut`): the texture, UV and
    # mesh-face tables they share are held once (the programs shade with `p.geo.cut`).
    carrier = material_type(
      diffuse_roughness=layout.diffuse_roughness, specular_weight=layout.specular_weight, layered=layout.layered
    )
  fields: dict[str, FieldType] = {"geo": optix_geometry_type(live_uv=layout.live_material_uv, carrier=carrier)}
  if layout.shaded:
    fields["lit"] = Lights
  fields["cams"] = Cameras
  if layout.shaded:
    fields["env"] = EnvironmentArrays
  if not layout.gi:
    fields["out"] = OutputArrays
  if layout.shaded:
    fields["seed"] = wp.array1d[wp.int32]
  fields.update(nworld=int, offset=int)
  if layout.history:
    fields["tmp"] = wp.array1d[Temporal]
  if layout.grid:
    fields["grid"] = Pass
  if layout.gi:
    for name, launch in GI_LAUNCHES.items():
      fields[name] = wp.array1d[launch]
  fields.update({name: kind for flag, (name, kind) in _TABLES.items() if getattr(layout, flag)})
  return fields


@specialize
def params_type(layout: ParamsLayout) -> type[LaunchParams]:
  """The launch-parameter struct of a build: OptiX's launch parameters, read by every program.

  Only what the build's programs read: every field counts towards `PARAMS_LIMIT` (warns above).
  """
  fields = _fields(layout)
  # The program flags in field order: the struct name the code had before the `shaded` field (a
  # new name would change every build's generated code).
  flags = (
    layout.history,
    layout.grid,
    layout.world_curves,
    layout.world_primitives,
    layout.shared_curves,
    layout.shared_primitives,
    layout.mesh,
  )
  name = "Params_" + "".join(str(int(v)) for v in flags)
  # A geometry-only build's name tells the layouts apart.
  name += "" if layout.shaded else "_geometry"
  if layout.diffuse_roughness:
    name += "_eon"
  if layout.specular_weight:
    name += "_weighted"
  if layout.layered:
    name += "_layered"
  if layout.gi:
    name += "_gi"
  if layout.area:
    name += "_area"
  attrs = {"__annotations__": fields, "__module__": __name__, "__qualname__": name, "__doc__": params_type.__doc__}
  # A Warp struct type: called, it makes an instance with the fields of `LaunchParams` it holds.
  struct = cast("type[LaunchParams]", wp.struct(type(name, (), attrs)))
  size = size_of(struct)
  if size > PARAMS_LIMIT:
    warnings.warn(
      f"OptiX launch parameters of {size} bytes (over {PARAMS_LIMIT}): the programs copy them to local memory, "
      "several times slower (params.PARAMS_LIMIT)",
      stacklevel=2,
    )
  return struct


def _array_type() -> type[ctypes.Structure]:
  """Warp's ctypes mirror of an array descriptor (private `warp._src.types.array_t`), checked once:
  the bound words are entries of its `strides` int32 vector, read at its own field offset.

  Raises:
    RuntimeError: when the Warp release lays the descriptor out otherwise.
  """
  from warp._src.types import array_t  # noqa: PLC0415  # private: listed in the Warp fence test

  kind = next((f[1] for f in array_t._fields_ if f[0] == "strides"), None)
  if kind is None or getattr(kind, "_type_", None) is not ctypes.c_int32:
    raise RuntimeError(f"Warp {wp.__version__}: array_t.strides is {kind}, not an int32 vector (trace/optix/device/params.py)")
  return array_t


_ARRAY_T: Final = _array_type()
_STRIDE_BYTES: Final = ctypes.sizeof(ctypes.c_int32)
# (launch-parameter byte offset, value bytes) of each bound word.
type BoundWords = tuple[tuple[int, bytes], ...]
# (launch-parameter byte offset of an array descriptor, its innermost stride's index, element
# bytes, field path) of every array of a launch-parameter struct (`_arrays`).
type _Array = tuple[int, int, int, str]


class _StructType(Protocol):
  """What `_arrays` reads of a Warp struct type (`wp.struct`'s `Struct`)."""

  vars: dict[str, object]
  ctype: type[ctypes.Structure]


def _arrays(struct: _StructType, base: int, path: str, out: list[_Array]) -> None:
  """Appends every array descriptor of Warp struct type `struct` (nested structs included) at byte
  offset `base`, with its element size from the field's type."""
  for name, var in struct.vars.items():
    kind = getattr(var, "type")  # noqa: B009  # a Warp Var
    offset = base + getattr(struct.ctype, name).offset
    if wp.types.is_array(kind):
      ndim, dtype = getattr(kind, "ndim"), getattr(kind, "dtype")  # noqa: B009
      out.append((offset, ndim - 1, wp.types.type_size_in_bytes(dtype), path + name))
    elif hasattr(kind, "vars") and hasattr(kind, "ctype"):
      _arrays(cast("_StructType", kind), offset, f"{path}{name}.", out)


@functools.cache
def _struct_arrays(struct: _StructType) -> tuple[_Array, ...]:
  """The array descriptors of launch-parameter struct type `struct` (`_arrays`)."""
  out: list[_Array] = []
  _arrays(struct, 0, "", out)
  return tuple(out)


def bound_words(params: LaunchParams) -> BoundWords:
  """The launch-parameter words a build compiles in as OptiX bound values: the innermost stride of
  every array descriptor, its element size, every copy of the build word (`word_offsets`; 0 in a
  build without bound bits) and the output word's copies (`OutputArrays.bits` / `half`, a render
  layout's: `output_offsets`). The strides follow from the struct's types alone (an unset or empty
  array's descriptor is never read through); the words take one value per combination of the
  build's bound bits and per output word the scenes and configurations use: one module per value,
  all from one PTX. Nothing sized by the scene or the batch is bound (array extents, nworld, ray
  counts: run time; the output table's buffers are read from the table). Called at every launch
  (the arrays' words cached per struct type, the arrays checked in one vectorised read).

  Raises:
    ValueError: a non-empty array of `params` whose innermost stride is not its element size (a
      strided view: the programs would misread it).
  """
  struct = _struct_of(params)
  _check(params, _struct_arrays(struct))
  words = list(_stride_words(struct))
  for (path, half), offset in word_offsets(struct).items():
    words.append((offset, bytes(ctypes.c_int32(getattr(_copy(params, path), half)))))
  for name, offset in output_offsets(struct).items():
    words.append((offset, bytes(ctypes.c_int32(getattr(params.out, name)))))
  return tuple(sorted(words))


@functools.cache
def _stride_words(struct: _StructType) -> BoundWords:
  """The arrays' bound words of launch-parameter struct type `struct` (`bound_words`): each
  innermost stride's offset and its element size, from the types alone."""
  stride = _ARRAY_T.strides.offset  # a ctypes field descriptor
  return tuple((a + stride + _STRIDE_BYTES * i, bytes(ctypes.c_int32(size))) for a, i, size, _ in _struct_arrays(struct))


def geometry_word(words: BoundWords, struct: type[LaunchParams]) -> int:
  """The build word's geometry copy (every bound bit, both halves) among the bound words `words` of
  launch-parameter struct type `struct` (`bound_words`); 0 when it holds none."""
  offsets = word_offsets(cast("_StructType", struct))
  values = {o: int.from_bytes(v, "little", signed=True) for o, v in words}
  low, high = (values.get(offsets.get((("geo",), half), -1), 0) for half in _HALVES)
  return low | (high << toggles.HIGH)


# The copies of the build word (`kernels/abi/toggles.py`), by struct path, and the bits each holds
# (both halves: `bits`, `bits2`): the geometry's every bound bit
# (`trace/optix/device/programs.build_word`), the light, material and environment copies the bits
# but the geometry's (`tiers.word`, `trace/frame_structs.word`).
_COPIES: Final[dict[tuple[str, ...], int]] = {
  ("geo",): toggles.BOUND_ALL,
  ("geo", "cut"): toggles.BOUND_ALL & ~toggles.GEO_ALL,
  ("lit",): toggles.BOUND_ALL & ~toggles.GEO_ALL,
  ("env",): toggles.BOUND_ALL & ~toggles.GEO_ALL,
}
# A copy's two fields (`toggles.store`): the word's first and second half.
_HALVES: Final = ("bits", "bits2")
# The output word's copies (`structs.OutputArrays`): its stores and its float16 buffers.
_OUTPUT_COPIES: Final = ("bits", "half")


def copy_mask(path: tuple[str, ...]) -> int:
  """The build-word bits copy `path` holds (`_COPIES`)."""
  return _COPIES[path]


def _struct_of(params: LaunchParams) -> _StructType:
  """The Warp struct type of launch parameters `params`."""
  return cast("_StructType", getattr(params, "_cls"))  # noqa: B009  # a Warp struct instance's type


def _copy(params: LaunchParams, path: tuple[str, ...]) -> toggles.Word:
  """The struct at `path` of launch parameters `params` (a copy of the word)."""
  out = params
  for name in path:
    out = getattr(out, name)
  return cast("toggles.Word", out)


def set_words(params: LaunchParams, word: int) -> None:
  """Sets every copy of the build word in launch parameters `params` to word `word` (its geometry
  copy's value, `trace/optix/device/programs.build_word`; each copy its bits, `_COPIES`): what a
  frame's structs hold (`trace/frame_structs.word`), the GI programs' launches
  (`launch.OptixRenderer.gi_params`) and the bundle fill's records (`tools/prebuilt/fill.py`)."""
  for path, half in word_offsets(_struct_of(params)):
    held = word & _COPIES[path]
    setattr(_copy(params, path), half, held & toggles.HALF if half == "bits" else held >> toggles.HIGH)


def set_outputs(params: LaunchParams, outputs: OutputLayout) -> None:
  """Sets the output word `outputs` in launch parameters `params` (a render layout's: the GI
  programs' holds none, `output_offsets`)."""
  if output_offsets(_struct_of(params)):
    params.out.bits, params.out.half = outputs.bits, outputs.half


@functools.cache
def word_offsets(struct: _StructType) -> dict[tuple[tuple[str, ...], str], int]:
  """The byte offset of each half (`_HALVES`) of each copy of the build word (`_COPIES` paths)
  launch-parameter struct type `struct` holds (a geometry-only layout: the geometry's alone)."""
  out: dict[tuple[tuple[str, ...], str], int] = {}
  for path in _COPIES:
    for half in _HALVES:
      kind, offset = struct, 0
      for name in (*path, half):
        var = kind.vars.get(name)
        if var is None:
          break
        offset += getattr(kind.ctype, name).offset
        kind = cast("_StructType", getattr(var, "type"))  # noqa: B009  # a Warp Var
      else:
        out[path, half] = offset
  return out


@functools.cache
def output_offsets(struct: _StructType) -> dict[str, int]:
  """The byte offset of each copy of the output word (`_OUTPUT_COPIES`) launch-parameter struct
  type `struct` holds (none in the GI programs' layout)."""
  if "out" not in struct.vars:
    return {}
  base = getattr(struct.ctype, "out").offset  # noqa: B009  # a ctypes field descriptor
  return {name: base + getattr(warp_struct(OutputArrays).ctype, name).offset for name in _OUTPUT_COPIES}


def _check(params: LaunchParams, arrays: tuple[_Array, ...]) -> None:
  """Raises ValueError when a non-empty array of `params` has another innermost stride than its
  element size (`bound_words`; one vectorised read of the struct's bytes, `strides.misread`)."""
  # Host code: imported here, not at module level, so that this kernel-code module's other code
  # (which every bundled build's digest covers, `build/kernel_cache/kernel_code.py`) stays as it is.
  from raywarp.core.trace.optix.strides import misread  # noqa: PLC0415

  c = struct_mirror(params)
  bad = misread(ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c)), arrays, _ARRAY_T)
  if bad is not None:
    k, stride = bad
    _, _, size, path = arrays[k]
    raise ValueError(
      f"OptiX launch parameter {path}: innermost stride {stride} B, its element is {size} B; the "
      "programs bind the element size (a contiguous array is needed)"
    )


def size_of(struct: type[LaunchParams]) -> int:
  """Bytes of a launch-parameter struct (its ctypes layout)."""
  return wp.types.type_size_in_bytes(struct)


# Launch-parameter words a store copies at most (`store_words`): 16 KiB, twice `PARAMS_LIMIT`
# (a layout above that limit still works, slowly, and warns), within CUDA's 32764-byte kernel
# parameter limit.
STORE_WORDS: Final = 2 * PARAMS_LIMIT // 4
_Words = wp.types.vector(length=STORE_WORDS, dtype=wp.uint32)


@wp.kernel(enable_backward=False, module="unique")
def store_words(
  # In:
  words: _Words,
  nword: int,
  offset_word: int,
  chunk: int,
  # Out:
  out: wp.array2d[wp.uint32],
):
  """Copy i of a struct's `nword` words (`offset_word`: a launch-parameter struct's `offset` field,
  set to i · chunk; -1: none, `write_struct`): one kernel for every layout (the words are the
  struct's bytes), compiled once."""
  i, j = wp.tid()
  if j < nword:
    v = words[j]
    if j == offset_word:
      v = wp.uint32(i * chunk)
    out[i, j] = v


def store_inputs(params: LaunchParams) -> tuple[_Words, int, int]:
  """(words, word count, word index of `offset`) of `params` for `store_words`.

  Raises:
    ValueError: for a struct above `STORE_WORDS` words or not a whole number of words (the
      launches step the copies by the struct's size, `Pipeline.trace`).
  """
  c = struct_mirror(params)
  words, nword = _words(c)
  return words, nword, type(c).offset.offset // 4


def write_struct(value: object, slot: wp.array) -> None:
  """Writes Warp struct instance `value` into the one-element array `slot` of its type on the
  device (`store_words`: the bytes ride in the kernel's arguments, so a CUDA graph capture records
  them: no host copy).

  Raises:
    ValueError: for a struct above `STORE_WORDS` words or not a whole number of words.
  """
  words, nword = _words(struct_mirror(value))
  out = wp.array(ptr=slot.ptr, dtype=wp.uint32, shape=(1, nword), device=slot.device)
  wp.launch(store_words, dim=(1, nword), inputs=[words, nword, -1, 0], outputs=[out], device=slot.device)


def _words(c: ctypes.Structure) -> tuple[_Words, int]:
  """(words, word count) of the ctypes mirror `c` of a Warp struct.

  Raises:
    ValueError: for a struct above `STORE_WORDS` words or not a whole number of words.
  """
  size = ctypes.sizeof(c)
  if size > 4 * STORE_WORDS or size % 4:
    raise ValueError(f"a struct of {size} bytes: not a whole number of words up to {4 * STORE_WORDS}")
  words = _Words()
  # A Warp vector is a ctypes array of its components: the struct's bytes copied in place.
  ctypes.memmove(ctypes.addressof(words), ctypes.addressof(c), size)  # pyrefly: ignore[bad-argument-type]  # a ctypes array
  return words, size // 4
