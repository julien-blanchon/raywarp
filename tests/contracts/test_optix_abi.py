"""The ctypes OptiX binding (`core/trace/optix/api.py`) matches the OptiX SDK headers it is pinned
to.

A C program compiled against the bundled headers prints the size of every struct `api` mirrors and
the offset of every field (the function table's entries included); each must equal the ctypes
layout. A wrong layout would hand OptiX garbage silently, so this runs on the CPU in every suite.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from raywarp.core.trace.optix import api
from raywarp.core.trace.optix.device import intrinsics

# api struct -> its C type name (OptiX prefix), every struct the binding passes to OptiX.
_STRUCTS = {
  name: f"Optix{name}"
  for name, value in vars(api).items()
  if isinstance(value, type)
  and issubclass(value, ctypes.Structure)
  and not name.startswith("_")
  and value.__module__ == api.__name__
}


def _fields(cls: type[ctypes.Structure]) -> list[str]:
  """The C field names of `cls` (anonymous unions: their members)."""
  out = []
  anonymous = set(getattr(cls, "_anonymous_", ()))
  for field in cls._fields_:
    name, kind = field[0], field[1]
    if name in anonymous:
      out += [f[0] for f in kind._fields_]  # pyrefly: ignore[missing-attribute]  # an anonymous union
    else:
      out.append(name)
  return out


def _probe(inc: Path, tmp: Path) -> dict[str, int]:
  """sizeof / offsetof values of the headers in `inc`, by "Type" and "Type.field"."""
  lines = []
  for name, c_name in sorted(_STRUCTS.items()):
    cls = getattr(api, name)
    lines.append(f'printf("{c_name} %zu\\n", sizeof({c_name}));')
    lines += [f'printf("{c_name}.{f} %zu\\n", offsetof({c_name}, {f}));' for f in _fields(cls)]
  src = tmp / "probe.c"
  src.write_text(
    "#include <stdio.h>\n#include <stddef.h>\n"
    "typedef struct CUctx_st* CUcontext;\ntypedef struct CUstream_st* CUstream;\n"
    "#define OPTIX_DONT_INCLUDE_CUDA\n#include <optix_function_table.h>\n"
    "int main(void) {\n" + "\n".join(lines) + "\nreturn 0;\n}\n"
  )
  exe = tmp / "probe"
  subprocess.run(["gcc", "-std=c11", f"-I{inc}", str(src), "-o", str(exe)], check=True, capture_output=True)
  out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
  return {k: int(v) for k, v in (line.split() for line in out.splitlines())}


def _header_inc() -> Path:
  """The bundled OptiX headers (skips without them or without a C compiler)."""
  inc = intrinsics.include_dir()
  if inc is None or shutil.which("gcc") is None:
    pytest.skip("needs the bundled OptiX SDK headers and gcc")
  return inc


def test_struct_layouts_match_the_headers(tmp_path: Path) -> None:
  """Every struct the binding mirrors has the size and field offsets of the pinned headers."""
  inc = _header_inc()
  c = _probe(inc, tmp_path)
  for name, c_name in _STRUCTS.items():
    cls = getattr(api, name)
    assert ctypes.sizeof(cls) == c[c_name], f"sizeof({c_name})"
    for f in _fields(cls):
      assert getattr(cls, f).offset == c[f"{c_name}.{f}"], f"offsetof({c_name}, {f})"


def test_binding_version_is_the_headers() -> None:
  """The binding's OptiX release is the bundled headers' (`OPTIX_VERSION`) and its ABI the
  function table's (`OPTIX_ABI_VERSION`)."""
  inc = _header_inc()
  assert intrinsics.header_version(inc) == api.VERSION
  table = (inc / "optix_function_table.h").read_text()
  assert f"#define OPTIX_ABI_VERSION {api.ABI_VERSION}\n" in table


def test_stack_sizes_follow_the_sdk_formula() -> None:
  """`compute_stack_sizes` is optixUtilComputeStackSizes (one trace level, no callables)."""
  s = api.StackSizes(cssRG=100, cssMS=8, cssCH=40, cssAH=16, cssIS=30, cssCC=0, dssDC=0)
  assert api.compute_stack_sizes(s, 1, 0, 0) == (0, 0, 100 + max(40, 30 + 16))
  assert api.compute_stack_sizes(s, 2, 0, 0) == (0, 0, 100 + 40 + max(40, 46))


def test_structs_keep_their_host_arrays_alive() -> None:
  """A host array assigned to a nested pointer field lives as long as the build input."""
  bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_TRIANGLES)
  bi.triangleArray.vertexBuffers = (api.CUdeviceptr * 1)(1234)
  bi.triangleArray.flags = (ctypes.c_uint * 1)(api.GEOMETRY_FLAG_DISABLE_ANYHIT)
  import gc

  gc.collect()
  assert bi.triangleArray.vertexBuffers[0] == 1234
  assert bi.triangleArray.flags[0] == api.GEOMETRY_FLAG_DISABLE_ANYHIT


def test_build_input_arrays_keep_their_host_arrays_alive() -> None:
  """Copying a build input into the array OptiX reads (`gas.Accel.inputs`) keeps its nested host
  arrays alive (refits read them every frame)."""
  import gc

  def one() -> ctypes.Array[api.BuildInput]:
    bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_TRIANGLES)
    bi.triangleArray.vertexBuffers = (api.CUdeviceptr * 1)(4321)
    bi.triangleArray.flags = (ctypes.c_uint * 1)(7)
    return (api.BuildInput * 1)(bi)

  inputs = one()
  gc.collect()
  _churn = [(api.CUdeviceptr * 1)(99) for _ in range(10000)]
  assert inputs[0].triangleArray.vertexBuffers[0] == 4321 and inputs[0].triangleArray.flags[0] == 7
