"""The OptiX 9.0 host API through ctypes: the driver's function table (ABI 105) and its structs.

OptiX ships in the NVIDIA driver (`libnvoptix.so.1`): `table()` opens it once and asks it for the
function table of `ABI_VERSION` (`optixQueryFunctionTable`, as `optix_stubs.h` does). The structs
below mirror `optix_types.h` of the OptiX SDK release `VERSION` field for field (C names), and the
constants its enums; `tests/contracts/test_optix_abi.py` compiles the headers and checks every
size and offset. Only what raywarp calls is typed; the other table entries are opaque pointers
that keep the table's layout.

A foreign call releases the GIL (ctypes), so OptiX's compile tasks run on Python threads in
parallel (`context.DeviceContext.module`), and the log callback (`LogCallback`) re-acquires it.
Every call's result goes through `check`: a failure raises `OptixError` with OptiX's name and text.
Pointers a struct holds (strings, host arrays) are kept alive by ctypes with the struct
(`Structure._objects`); device addresses are plain integers (`CUdeviceptr`).
"""

from __future__ import annotations

import ctypes
import functools
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_size_t, c_uint, c_ulonglong, c_ushort, c_void_p
from typing import TYPE_CHECKING, Final, final

# The OptiX SDK release whose headers these structs mirror, and its function-table ABI.
VERSION: Final = (9, 0, 0)
ABI_VERSION: Final = 105
LIBRARY: Final = "libnvoptix.so.1"

CUdeviceptr = c_ulonglong
TraversableHandle = c_ulonglong
if TYPE_CHECKING:
  from _ctypes import _CDataType  # typeshed's union of the ctypes types

Handle = c_void_p  # OptixDeviceContext, OptixModule, OptixProgramGroup, OptixPipeline, OptixTask
Result = c_int
Enum = c_int

# OptixResult
SUCCESS: Final = 0
# OptixGeometryFlags
GEOMETRY_FLAG_NONE: Final = 0
GEOMETRY_FLAG_DISABLE_ANYHIT: Final = 1 << 0
# OptixIndicesFormat / OptixVertexFormat
INDICES_FORMAT_UNSIGNED_INT3: Final = 0x2103
VERTEX_FORMAT_FLOAT3: Final = 0x2121
# OptixBuildInputType
BUILD_INPUT_TYPE_TRIANGLES: Final = 0x2141
BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES: Final = 0x2142
BUILD_INPUT_TYPE_INSTANCES: Final = 0x2143
BUILD_INPUT_TYPE_CURVES: Final = 0x2145
# OptixBuildFlags
BUILD_FLAG_ALLOW_UPDATE: Final = 1 << 0
BUILD_FLAG_ALLOW_COMPACTION: Final = 1 << 1
BUILD_FLAG_PREFER_FAST_TRACE: Final = 1 << 2
BUILD_FLAG_PREFER_FAST_BUILD: Final = 1 << 3
# OptixBuildOperation
BUILD_OPERATION_BUILD: Final = 0x2161
BUILD_OPERATION_UPDATE: Final = 0x2162
# OptixAccelPropertyType
PROPERTY_TYPE_COMPACTED_SIZE: Final = 0x2181
# The built (or refitted) structure's AABB: 6 floats, min xyz then max xyz (OptixAabb).
PROPERTY_TYPE_AABBS: Final = 0x2182
# OptixInstanceFlags
INSTANCE_FLAG_NONE: Final = 0
INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING: Final = 1 << 0
INSTANCE_FLAG_DISABLE_ANYHIT: Final = 1 << 2
INSTANCE_FLAG_ENFORCE_ANYHIT: Final = 1 << 3
# OptixPrimitiveType / OptixPrimitiveTypeFlags
PRIMITIVE_TYPE_ROUND_LINEAR: Final = 0x2503
PRIMITIVE_TYPE_FLAGS_CUSTOM: Final = 1 << 0
PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR: Final = 1 << 3
PRIMITIVE_TYPE_FLAGS_TRIANGLE: Final = 1 << 31
# OptixTraversableGraphFlags
TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING: Final = 1 << 1
# OptixExceptionFlags
EXCEPTION_FLAG_NONE: Final = 0
# OptixCompileOptimizationLevel / OptixCompileDebugLevel / OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
COMPILE_OPTIMIZATION_DEFAULT: Final = 0
COMPILE_DEBUG_LEVEL_NONE: Final = 0x2350
COMPILE_DEFAULT_MAX_REGISTER_COUNT: Final = 0
# OptixModuleCompileState (`DeviceContext.module` checks for COMPLETED)
MODULE_COMPILE_STATE_COMPLETED: Final = 0x2364
# OptixProgramGroupKind
PROGRAM_GROUP_KIND_RAYGEN: Final = 0x2421
PROGRAM_GROUP_KIND_MISS: Final = 0x2422
PROGRAM_GROUP_KIND_HITGROUP: Final = 0x2424
# OPTIX_SBT_RECORD_HEADER_SIZE
SBT_RECORD_HEADER_SIZE: Final = 32
# OptixOpacityMicromapFlags / Format / ArrayIndexingMode
OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE: Final = 1 << 0
OPACITY_MICROMAP_FORMAT_4_STATE: Final = 2
OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED: Final = 2

# OptixLogCallback: void (*)(unsigned int level, const char* tag, const char* message, void* cbdata)
LogCallback = ctypes.CFUNCTYPE(None, c_uint, c_char_p, c_char_p, c_void_p)


class DeviceContextOptions(ctypes.Structure):
  """OptixDeviceContextOptions."""

  _fields_ = (
    ("logCallbackFunction", LogCallback),
    ("logCallbackData", c_void_p),
    ("logCallbackLevel", c_int),
    ("validationMode", Enum),
  )


class PayloadType(ctypes.Structure):
  """OptixPayloadType."""

  _fields_ = (("numPayloadValues", c_uint), ("payloadSemantics", POINTER(c_uint)))


class ModuleCompileBoundValueEntry(ctypes.Structure):
  """OptixModuleCompileBoundValueEntry."""

  _fields_ = (
    ("pipelineParamOffsetInBytes", c_size_t),
    ("sizeInBytes", c_size_t),
    ("boundValuePtr", c_void_p),
    ("annotation", c_char_p),
  )


class ModuleCompileOptions(ctypes.Structure):
  """OptixModuleCompileOptions."""

  _fields_ = (
    ("maxRegisterCount", c_int),
    ("optLevel", Enum),
    ("debugLevel", Enum),
    ("boundValues", POINTER(ModuleCompileBoundValueEntry)),
    ("numBoundValues", c_uint),
    ("numPayloadTypes", c_uint),
    ("payloadTypes", POINTER(PayloadType)),
  )


class PipelineCompileOptions(ctypes.Structure):
  """OptixPipelineCompileOptions."""

  _fields_ = (
    ("usesMotionBlur", c_int),
    ("traversableGraphFlags", c_uint),
    ("numPayloadValues", c_int),
    ("numAttributeValues", c_int),
    ("exceptionFlags", c_uint),
    ("pipelineLaunchParamsVariableName", c_char_p),
    ("usesPrimitiveTypeFlags", c_uint),
    ("allowOpacityMicromaps", c_int),
    ("allowClusteredGeometry", c_int),
  )


class BuiltinISOptions(ctypes.Structure):
  """OptixBuiltinISOptions."""

  _fields_ = (
    ("builtinISModuleType", Enum),
    ("usesMotionBlur", c_int),
    ("buildFlags", c_uint),
    ("curveEndcapFlags", c_uint),
  )


class ProgramGroupSingleModule(ctypes.Structure):
  """OptixProgramGroupSingleModule."""

  _fields_ = (("module", Handle), ("entryFunctionName", c_char_p))


class ProgramGroupHitgroup(ctypes.Structure):
  """OptixProgramGroupHitgroup."""

  _fields_ = (
    ("moduleCH", Handle),
    ("entryFunctionNameCH", c_char_p),
    ("moduleAH", Handle),
    ("entryFunctionNameAH", c_char_p),
    ("moduleIS", Handle),
    ("entryFunctionNameIS", c_char_p),
  )


class ProgramGroupCallables(ctypes.Structure):
  """OptixProgramGroupCallables (unused: layout only)."""

  _fields_ = (("moduleDC", Handle), ("entryFunctionNameDC", c_char_p), ("moduleCC", Handle), ("entryFunctionNameCC", c_char_p))


class _ProgramGroupUnion(ctypes.Union):
  """The anonymous union of OptixProgramGroupDesc."""

  _fields_ = (
    ("raygen", ProgramGroupSingleModule),
    ("miss", ProgramGroupSingleModule),
    ("exception", ProgramGroupSingleModule),
    ("callables", ProgramGroupCallables),
    ("hitgroup", ProgramGroupHitgroup),
  )


class ProgramGroupDesc(ctypes.Structure):
  """OptixProgramGroupDesc (the union anonymous: `desc.raygen`, `desc.hitgroup`, ...)."""

  _anonymous_ = ("u",)
  _fields_ = (("kind", Enum), ("flags", c_uint), ("u", _ProgramGroupUnion))


class ProgramGroupOptions(ctypes.Structure):
  """OptixProgramGroupOptions."""

  _fields_ = (("payloadType", POINTER(PayloadType)),)


class PipelineLinkOptions(ctypes.Structure):
  """OptixPipelineLinkOptions."""

  _fields_ = (("maxTraceDepth", c_uint),)


class ShaderBindingTable(ctypes.Structure):
  """OptixShaderBindingTable."""

  _fields_ = (
    ("raygenRecord", CUdeviceptr),
    ("exceptionRecord", CUdeviceptr),
    ("missRecordBase", CUdeviceptr),
    ("missRecordStrideInBytes", c_uint),
    ("missRecordCount", c_uint),
    ("hitgroupRecordBase", CUdeviceptr),
    ("hitgroupRecordStrideInBytes", c_uint),
    ("hitgroupRecordCount", c_uint),
    ("callablesRecordBase", CUdeviceptr),
    ("callablesRecordStrideInBytes", c_uint),
    ("callablesRecordCount", c_uint),
  )


_STACK_FIELDS: Final = ("cssRG", "cssMS", "cssCH", "cssAH", "cssIS", "cssCC", "dssDC")


class StackSizes(ctypes.Structure):
  """OptixStackSizes."""

  _fields_ = tuple((name, c_uint) for name in _STACK_FIELDS)


class MotionOptions(ctypes.Structure):
  """OptixMotionOptions."""

  _fields_ = (("numKeys", c_ushort), ("flags", c_ushort), ("timeBegin", c_float), ("timeEnd", c_float))


class AccelBuildOptions(ctypes.Structure):
  """OptixAccelBuildOptions."""

  _fields_ = (("buildFlags", c_uint), ("operation", Enum), ("motionOptions", MotionOptions))


class AccelBufferSizes(ctypes.Structure):
  """OptixAccelBufferSizes."""

  _fields_ = (("outputSizeInBytes", c_size_t), ("tempSizeInBytes", c_size_t), ("tempUpdateSizeInBytes", c_size_t))


class RelocationInfo(ctypes.Structure):
  """OptixRelocationInfo: opaque data of a built acceleration structure (a relocated copy's
  compatibility and its relocation)."""

  _fields_ = (("info", ctypes.c_ulonglong * 4),)


class AccelEmitDesc(ctypes.Structure):
  """OptixAccelEmitDesc."""

  _fields_ = (("result", CUdeviceptr), ("type", Enum))


class OpacityMicromapUsageCount(ctypes.Structure):
  """OptixOpacityMicromapUsageCount (also the layout of OptixOpacityMicromapHistogramEntry)."""

  _fields_ = (("count", c_uint), ("subdivisionLevel", c_uint), ("format", Enum))


OpacityMicromapHistogramEntry = OpacityMicromapUsageCount


class BuildInputOpacityMicromap(ctypes.Structure):
  """OptixBuildInputOpacityMicromap."""

  _fields_ = (
    ("indexingMode", Enum),
    ("opacityMicromapArray", CUdeviceptr),
    ("indexBuffer", CUdeviceptr),
    ("indexSizeInBytes", c_uint),
    ("indexStrideInBytes", c_uint),
    ("indexOffset", c_uint),
    ("numMicromapUsageCounts", c_uint),
    ("micromapUsageCounts", POINTER(OpacityMicromapUsageCount)),
  )


class BuildInputDisplacementMicromap(ctypes.Structure):
  """OptixBuildInputDisplacementMicromap (unused: zeroed, layout only)."""

  _fields_ = (
    ("indexingMode", Enum),
    ("displacementMicromapArray", CUdeviceptr),
    ("displacementMicromapIndexBuffer", CUdeviceptr),
    ("vertexDirectionsBuffer", CUdeviceptr),
    ("vertexBiasAndScaleBuffer", CUdeviceptr),
    ("triangleFlagsBuffer", CUdeviceptr),
    ("displacementMicromapIndexOffset", c_uint),
    ("displacementMicromapIndexStrideInBytes", c_uint),
    ("displacementMicromapIndexSizeInBytes", c_uint),
    ("vertexDirectionFormat", Enum),
    ("vertexDirectionStrideInBytes", c_uint),
    ("vertexBiasAndScaleFormat", Enum),
    ("vertexBiasAndScaleStrideInBytes", c_uint),
    ("triangleFlagsStrideInBytes", c_uint),
    ("numDisplacementMicromapUsageCounts", c_uint),
    ("displacementMicromapUsageCounts", c_void_p),
  )


class BuildInputTriangleArray(ctypes.Structure):
  """OptixBuildInputTriangleArray."""

  _fields_ = (
    ("vertexBuffers", POINTER(CUdeviceptr)),
    ("numVertices", c_uint),
    ("vertexFormat", Enum),
    ("vertexStrideInBytes", c_uint),
    ("indexBuffer", CUdeviceptr),
    ("numIndexTriplets", c_uint),
    ("indexFormat", Enum),
    ("indexStrideInBytes", c_uint),
    ("preTransform", CUdeviceptr),
    ("flags", POINTER(c_uint)),
    ("numSbtRecords", c_uint),
    ("sbtIndexOffsetBuffer", CUdeviceptr),
    ("sbtIndexOffsetSizeInBytes", c_uint),
    ("sbtIndexOffsetStrideInBytes", c_uint),
    ("primitiveIndexOffset", c_uint),
    ("transformFormat", Enum),
    ("opacityMicromap", BuildInputOpacityMicromap),
    ("displacementMicromap", BuildInputDisplacementMicromap),
  )


class BuildInputCurveArray(ctypes.Structure):
  """OptixBuildInputCurveArray."""

  _fields_ = (
    ("curveType", Enum),
    ("numPrimitives", c_uint),
    ("vertexBuffers", POINTER(CUdeviceptr)),
    ("numVertices", c_uint),
    ("vertexStrideInBytes", c_uint),
    ("widthBuffers", POINTER(CUdeviceptr)),
    ("widthStrideInBytes", c_uint),
    ("normalBuffers", POINTER(CUdeviceptr)),
    ("normalStrideInBytes", c_uint),
    ("indexBuffer", CUdeviceptr),
    ("indexStrideInBytes", c_uint),
    ("flag", c_uint),
    ("primitiveIndexOffset", c_uint),
    ("endcapFlags", c_uint),
  )


class BuildInputCustomPrimitiveArray(ctypes.Structure):
  """OptixBuildInputCustomPrimitiveArray."""

  _fields_ = (
    ("aabbBuffers", POINTER(CUdeviceptr)),
    ("numPrimitives", c_uint),
    ("strideInBytes", c_uint),
    ("flags", POINTER(c_uint)),
    ("numSbtRecords", c_uint),
    ("sbtIndexOffsetBuffer", CUdeviceptr),
    ("sbtIndexOffsetSizeInBytes", c_uint),
    ("sbtIndexOffsetStrideInBytes", c_uint),
    ("primitiveIndexOffset", c_uint),
  )


class BuildInputInstanceArray(ctypes.Structure):
  """OptixBuildInputInstanceArray."""

  _fields_ = (("instances", CUdeviceptr), ("numInstances", c_uint), ("instanceStride", c_uint))


class _BuildInputUnion(ctypes.Union):
  """The anonymous union of OptixBuildInput (padded to its 1024 bytes)."""

  _fields_ = (
    ("triangleArray", BuildInputTriangleArray),
    ("curveArray", BuildInputCurveArray),
    ("customPrimitiveArray", BuildInputCustomPrimitiveArray),
    ("instanceArray", BuildInputInstanceArray),
    ("pad", ctypes.c_char * 1024),
  )


class BuildInput(ctypes.Structure):
  """OptixBuildInput (the union anonymous: `bi.triangleArray`, ...)."""

  _anonymous_ = ("u",)
  _fields_ = (("type", Enum), ("u", _BuildInputUnion))


class OpacityMicromapArrayBuildInput(ctypes.Structure):
  """OptixOpacityMicromapArrayBuildInput."""

  _fields_ = (
    ("flags", c_uint),
    ("inputBuffer", CUdeviceptr),
    ("perMicromapDescBuffer", CUdeviceptr),
    ("perMicromapDescStrideInBytes", c_uint),
    ("numMicromapHistogramEntries", c_uint),
    ("micromapHistogramEntries", POINTER(OpacityMicromapHistogramEntry)),
  )


class MicromapBuffers(ctypes.Structure):
  """OptixMicromapBuffers."""

  _fields_ = (
    ("output", CUdeviceptr),
    ("outputSizeInBytes", c_size_t),
    ("temp", CUdeviceptr),
    ("tempSizeInBytes", c_size_t),
  )


class MicromapBufferSizes(ctypes.Structure):
  """OptixMicromapBufferSizes."""

  _fields_ = (("outputSizeInBytes", c_size_t), ("tempSizeInBytes", c_size_t))


class Instance(ctypes.Structure):
  """OptixInstance (80 bytes; `instances.OptixInstance` is its Warp mirror)."""

  _fields_ = (
    ("transform", c_float * 12),
    ("instanceId", c_uint),
    ("sbtOffset", c_uint),
    ("visibilityMask", c_uint),
    ("flags", c_uint),
    ("traversableHandle", TraversableHandle),
    ("pad", c_uint * 2),
  )


def _p(t: type[_CDataType]) -> type[_CDataType]:
  """POINTER(t) (a pointer argument)."""
  return POINTER(t)


_size = POINTER(c_size_t)
# The function table of ABI 105, in order (optix_function_table.h): name, then (argument types) for
# the entries raywarp calls (all return OptixResult except the two error-text getters), None for
# the others (an opaque pointer).
_TABLE: Final = (
  ("optixGetErrorName", "text"),
  ("optixGetErrorString", "text"),
  ("optixDeviceContextCreate", (c_void_p, _p(DeviceContextOptions), _p(Handle))),
  ("optixDeviceContextDestroy", (Handle,)),
  ("optixDeviceContextGetProperty", None),
  ("optixDeviceContextSetLogCallback", None),
  ("optixDeviceContextSetCacheEnabled", (Handle, c_int)),
  ("optixDeviceContextSetCacheLocation", (Handle, c_char_p)),
  ("optixDeviceContextSetCacheDatabaseSizes", (Handle, c_size_t, c_size_t)),
  ("optixDeviceContextGetCacheEnabled", None),
  ("optixDeviceContextGetCacheLocation", None),
  ("optixDeviceContextGetCacheDatabaseSizes", None),
  (
    "optixModuleCreate",
    (Handle, _p(ModuleCompileOptions), _p(PipelineCompileOptions), c_char_p, c_size_t, c_char_p, _size, _p(Handle)),
  ),
  (
    "optixModuleCreateWithTasks",
    (
      Handle,
      _p(ModuleCompileOptions),
      _p(PipelineCompileOptions),
      c_char_p,
      c_size_t,
      c_char_p,
      _size,
      _p(Handle),
      _p(Handle),
    ),
  ),
  ("optixModuleGetCompilationState", (Handle, _p(Enum))),
  ("optixModuleDestroy", (Handle,)),
  (
    "optixBuiltinISModuleGet",
    (Handle, _p(ModuleCompileOptions), _p(PipelineCompileOptions), _p(BuiltinISOptions), _p(Handle)),
  ),
  ("optixTaskExecute", (Handle, _p(Handle), c_uint, _p(c_uint))),
  (
    "optixProgramGroupCreate",
    (Handle, _p(ProgramGroupDesc), c_uint, _p(ProgramGroupOptions), c_char_p, _size, _p(Handle)),
  ),
  ("optixProgramGroupDestroy", (Handle,)),
  ("optixProgramGroupGetStackSize", (Handle, _p(StackSizes), Handle)),
  (
    "optixPipelineCreate",
    (Handle, _p(PipelineCompileOptions), _p(PipelineLinkOptions), _p(Handle), c_uint, c_char_p, _size, _p(Handle)),
  ),
  ("optixPipelineDestroy", (Handle,)),
  ("optixPipelineSetStackSize", (Handle, c_uint, c_uint, c_uint, c_uint)),
  ("optixAccelComputeMemoryUsage", (Handle, _p(AccelBuildOptions), _p(BuildInput), c_uint, _p(AccelBufferSizes))),
  (
    "optixAccelBuild",
    (
      Handle,
      c_void_p,
      _p(AccelBuildOptions),
      _p(BuildInput),
      c_uint,
      CUdeviceptr,
      c_size_t,
      CUdeviceptr,
      c_size_t,
      _p(TraversableHandle),
      _p(AccelEmitDesc),
      c_uint,
    ),
  ),
  ("optixAccelGetRelocationInfo", (Handle, TraversableHandle, _p(RelocationInfo))),
  ("optixCheckRelocationCompatibility", (Handle, _p(RelocationInfo), _p(c_int))),
  (
    "optixAccelRelocate",
    (Handle, c_void_p, _p(RelocationInfo), c_void_p, c_size_t, CUdeviceptr, c_size_t, _p(TraversableHandle)),
  ),
  ("optixAccelCompact", (Handle, c_void_p, TraversableHandle, CUdeviceptr, c_size_t, _p(TraversableHandle))),
  ("optixAccelEmitProperty", None),
  ("optixConvertPointerToTraversableHandle", None),
  ("optixOpacityMicromapArrayComputeMemoryUsage", (Handle, _p(OpacityMicromapArrayBuildInput), _p(MicromapBufferSizes))),
  ("optixOpacityMicromapArrayBuild", (Handle, c_void_p, _p(OpacityMicromapArrayBuildInput), _p(MicromapBuffers))),
  ("optixOpacityMicromapArrayGetRelocationInfo", None),
  ("optixOpacityMicromapArrayRelocate", None),
  ("optixDisplacementMicromapArrayComputeMemoryUsage", None),
  ("optixDisplacementMicromapArrayBuild", None),
  ("optixClusterAccelComputeMemoryUsage", None),
  ("optixClusterAccelBuild", None),
  ("optixSbtRecordPackHeader", (Handle, c_void_p)),
  ("optixLaunch", (Handle, c_void_p, CUdeviceptr, c_size_t, _p(ShaderBindingTable), c_uint, c_uint, c_uint)),
  ("optixCoopVecMatrixConvert", None),
  ("optixCoopVecMatrixComputeSize", None),
  ("optixDenoiserCreate", None),
  ("optixDenoiserDestroy", None),
  ("optixDenoiserComputeMemoryResources", None),
  ("optixDenoiserSetup", None),
  ("optixDenoiserInvoke", None),
  ("optixDenoiserComputeIntensity", None),
  ("optixDenoiserComputeAverageColor", None),
  ("optixDenoiserCreateWithUserModel", None),
)


def _prototype(signature: str | tuple[type[_CDataType], ...] | None) -> type[_CDataType]:
  """The ctypes field type of one table entry."""
  if signature is None:
    return c_void_p
  if isinstance(signature, str):  # "text"
    return ctypes.CFUNCTYPE(c_char_p, Result)
  return ctypes.CFUNCTYPE(Result, *signature)


class FunctionTable(ctypes.Structure):
  """OptixFunctionTable of ABI 105 (`_TABLE`)."""

  _fields_ = tuple((name, _prototype(signature)) for name, signature in _TABLE)


@final
class OptixError(RuntimeError):
  """A failed OptiX call: `code` is its OptixResult."""

  def __init__(self, what: str, code: int, detail: str = "") -> None:
    """`what` failed with `code` (`detail`: OptiX's compile / link log)."""
    self.code = code
    name, text = _error_text(code)
    message = f"OptiX {what} failed: {name} ({code}): {text}"
    super().__init__(f"{message}\n{detail.strip()}" if detail.strip() else message)


def _error_text(code: int) -> tuple[str, str]:
  """OptiX's name and description of `code` (generic when the table is not loaded)."""
  t = _loaded()
  if t is None:
    return "OPTIX_ERROR", "no OptiX function table"
  name, text = t.optixGetErrorName(code), t.optixGetErrorString(code)
  return (name or b"?").decode(errors="replace"), (text or b"").decode(errors="replace")


def check(result: int, what: str, log: ctypes.Array[ctypes.c_char] | None = None) -> None:
  """Raises `OptixError` unless `result` is OPTIX_SUCCESS (`log`: the call's log buffer).

  Raises:
    OptixError: for any other result.
  """
  if result != SUCCESS:
    raise OptixError(what, result, log.value.decode(errors="replace") if log is not None else "")


@functools.cache
def _load() -> FunctionTable | str:
  """The driver's function table, or why it cannot be had."""
  if not sys.platform.startswith("linux"):
    return f"the OptiX ctypes binding supports Linux only (this is {sys.platform})"
  try:
    lib = ctypes.CDLL(LIBRARY, mode=ctypes.RTLD_LOCAL)
  except OSError as e:
    return f"the NVIDIA driver's {LIBRARY} cannot be loaded ({e})"
  query = lib.optixQueryFunctionTable
  query.restype = Result
  query.argtypes = (c_int, c_uint, c_void_p, c_void_p, c_void_p, c_size_t)
  table = FunctionTable()
  result = query(ABI_VERSION, 0, None, None, ctypes.byref(table), ctypes.sizeof(table))
  if result != SUCCESS:
    return f"the driver's OptiX does not provide ABI {ABI_VERSION} (OptiX {'.'.join(map(str, VERSION))}): result {result}"
  _keep.append(lib)
  return table


# The loaded library (never unloaded: the table points into it).
_keep: Final[list[ctypes.CDLL]] = []


def _loaded() -> FunctionTable | None:
  """The table if it was loaded successfully (never loads it)."""
  if _load.cache_info().currsize == 0:
    return None
  t = _load()
  return t if isinstance(t, FunctionTable) else None


def unavailable_reason() -> str | None:
  """None when the driver's OptiX serves `ABI_VERSION`, else why not (worded to follow "OptiX
  cannot run: ")."""
  t = _load()
  return t if isinstance(t, str) else None


def table() -> FunctionTable:
  """The OptiX function table (loaded on first use).

  Raises:
    RuntimeError: when the driver's OptiX cannot be loaded (`unavailable_reason`).
  """
  t = _load()
  if isinstance(t, str):
    raise RuntimeError(f"OptiX cannot run: {t}")
  return t


def accumulate_stack_sizes(groups: list[int], pipeline: int) -> StackSizes:
  """The per-field maximum of the stack sizes of `groups` in `pipeline`
  (`optixUtilAccumulateStackSizes`, optix_stack_size.h)."""
  t = table()
  total, one = StackSizes(), StackSizes()
  for g in groups:
    check(t.optixProgramGroupGetStackSize(g, ctypes.byref(one), pipeline), "optixProgramGroupGetStackSize")
    for name in _STACK_FIELDS:
      setattr(total, name, max(getattr(total, name), getattr(one, name)))
  return total


def compute_stack_sizes(s: StackSizes, max_trace_depth: int, max_cc_depth: int, max_dc_depth: int) -> tuple[int, int, int]:
  """(direct-callable stack from traversal, from state, continuation stack) of a pipeline
  (`optixUtilComputeStackSizes`, optix_stack_size.h)."""
  dc = max_dc_depth * s.dssDC
  cc_tree = max_cc_depth * s.cssCC
  ch_or_ms = max(s.cssCH, s.cssMS) + cc_tree
  continuation = (
    s.cssRG + cc_tree + (max(max_trace_depth, 1) - 1) * ch_or_ms + min(max_trace_depth, 1) * max(ch_or_ms, s.cssIS + s.cssAH)
  )
  return dc, dc, continuation
