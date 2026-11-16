"""The bundle fill's software megakernel modules: one pre-optimised LTO-IR per group, one link per
word (fill tooling only; nothing here runs in a renderer).

A software build's megakernel modules read their words from their preamble (`toggles.preamble`:
the build word `toggles.WORD_MACRO` of a tier build, the output word `OUT_MACRO` / `HALF_MACRO` of
every build): one Warp module per word value, every one with the same kernels, symbols and device
functions (the preamble is in the module hash, not the kernels'; `structs.in_module_of`). A group
is the modules of one build that differ only in their words (`group_key`). The fill generates the
group's CUDA once (`capture`: Warp's AOT compile of its first module, codegen only) and compiles
it to a pre-optimised LTO-IR with each word an extern function (`lto_source`: NVRTC `-dlto -rdc
--gen-opt-lto`), then per word value links it with a small translation unit returning the words
(`combination_cubin`: nvJitLink `-lto`, with the module's math flags, which nvJitLink does not
take from the LTO-IR) into the Warp cache entry the word's module loads
(`<cache>/<module id>/<module id>.sm<arch>.cubin` + the group's `.meta`: `install`). A renderer
finds that cubin as any cached one; a word outside the bundle is a normal Warp compile of the
literal preamble, which equals the folded build.

`combination_cubin` is the only per-word compile; `nvrtc_cubin` (an NVRTC compile of the word's
own source, what Warp compiles) is the reference it is checked against.

The compiler: NVRTC and nvJitLink of Warp's own CUDA version (an LTO-IR must not be newer than
its linker), through ctypes on the NVIDIA wheels, in the fill environment only:

  pip install "nvidia-cuda-nvrtc==13.4.*" "nvidia-nvjitlink==13.4.*"

(`WPBR_CUDA_LIB`: a directory holding libnvrtc.so.13 and libnvJitLink.so.13, else found in the
wheels' `nvidia/` package directory.)
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import hashlib
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Final

import warp as wp

from raywarp.core.kernels.abi import toggles

# Warp's NVRTC options for a release CUBIN beside the architecture, the include paths and the
# math flags (`wp_cuda_compile_program`, Warp 1.18, CUDA 13, optimisation level 3).
_RELEASE: Final = (
  "--std=c++17",
  "--Ofast-compile=0",
  "--define-macro=NDEBUG",
  "--undefine-macro=WP_VERIFY_FP",
  "--define-macro=WP_ENABLE_MATHDX=1",
  "--device-as-default-execution-space",
  "--extra-device-vectorization",
  "--restrict",
  "--diag-suppress=177,550",
)
# The words a module's preamble defines (`toggles.preamble`), their literals and LTO forms (the
# extern function of each, named after its macro: `wp_pbr_word`, `wp_pbr_out`, `wp_pbr_half`).
_MACROS: Final = (toggles.WORD_MACRO, toggles.OUT_MACRO, toggles.HALF_MACRO)
_LITERAL: Final = re.compile(rf"#define ({'|'.join(_MACROS)}) (\d+)u\n")
_EXTERN: Final = 'extern "C" __device__ unsigned int {fn}();\n#define {macro} {fn}()\n'
_WORD_TU: Final = 'extern "C" __device__ unsigned int {fn}() {{ return {value}u; }}\n'
# A module's words: macro -> value (`words_of`).
type Words = dict[str, int]
_NVJITLINK_LTOIR: Final = 3  # NVJITLINK_INPUT_LTOIR


@dataclasses.dataclass(frozen=True)
class Captured:
  """One word module's Warp AOT output without its binary: Warp's cache entry (source, meta, an
  empty cubin to fill) in `directory`, and what Warp would have compiled it with.

  Attributes:
    directory: the scratch entry (moved into the cache by `install`).
    cubin: the binary's path in it (`<module id>.sm<arch>.cubin`).
    source: the generated CUDA source (the word's literal preamble).
    arch: the SM version.
    fast_math, fuse_fp: the module's math options.
    include_dirs: its extra CUDA include directories.
    ltoirs: Warp's own link inputs of the module (MathDx; usually none).
  """

  directory: Path
  cubin: Path
  source: str
  arch: int
  fast_math: bool
  fuse_fp: bool
  include_dirs: tuple[str, ...]
  ltoirs: tuple[bytes, ...]


def words_of(module: wp.Module) -> Words:
  """The words a module's preamble defines (`toggles.preamble`), by macro; empty: not a word
  module."""
  options = module.options.get("extra_build_options")
  preamble = options.extra_cuda_preamble if isinstance(options, wp.ModuleBuildOptions) else ""
  return {m.group(1): int(m.group(2)) for m in _LITERAL.finditer(preamble)}


def _fn(macro: str) -> str:
  """The extern function standing for word `macro` in the LTO form."""
  return macro.lower()


def group_key(module: wp.Module) -> str:
  """The modules of one LTO-IR: equal kernels (mangled names, which the preamble does not enter)
  and block size; a stable digest (the shard owner of a word module: `fill._owner`)."""
  module.get_module_identifier()  # hashes the kernels
  block = module.options["block_dim"]
  hasher = module.hashers[block]
  names = sorted(k.get_mangled_name(kernel_hash=h) for k, h in hasher.kernel_hashes.items())
  return hashlib.sha1(repr((block, names)).encode()).hexdigest()[:16]


_PATCH: Final = threading.Lock()


def capture(module: wp.Module, arch: int, scratch: Path) -> Captured:
  """Warp's AOT compile of `module` for sm_`arch` into `scratch`, its binary step replaced by
  recording what it would compile (codegen only, seconds; under a lock: it swaps Warp's builder)."""
  import warp._src.build as warp_build  # the AOT compile's binary step

  seen: dict[str, object] = {}

  def record(cu_path: str, _arch: int, output_path: str, **kw: object) -> None:
    """The binary step: the source and options kept, an empty binary written."""
    seen.update(kw, source=Path(cu_path).read_text())
    Path(output_path).write_bytes(b"")

  with _PATCH, _swapped(warp_build, record):
    (cubin,) = wp.compile_aot_module(module, arch=arch, module_dir=str(scratch))
  include_dirs, ltoirs = seen.get("extra_include_dirs") or (), seen.get("ltoirs") or ()
  if not isinstance(include_dirs, list | tuple) or not isinstance(ltoirs, list | tuple):
    raise TypeError(f"Warp's binary step took include dirs {include_dirs!r}, link inputs {type(ltoirs).__name__}")
  return Captured(
    directory=scratch,
    cubin=Path(cubin),
    source=str(seen["source"]),
    arch=arch,
    fast_math=bool(seen["fast_math"]),
    fuse_fp=bool(seen["fuse_fp"]),
    include_dirs=tuple(str(d) for d in include_dirs),
    ltoirs=tuple(bytes(b) for b in ltoirs),
  )


@contextlib.contextmanager
def _swapped(module: object, build: object) -> Generator[None]:
  """Warp's `build_cuda` replaced by `build` for the block."""
  saved = module.build_cuda  # pyrefly: ignore[missing-attribute]
  module.build_cuda = build  # pyrefly: ignore[missing-attribute]
  try:
    yield
  finally:
    module.build_cuda = saved  # pyrefly: ignore[missing-attribute]


def lto_source(c: Captured) -> str:
  """The group's CUDA in its LTO form: each word's literal replaced by its extern function."""
  seen: list[str] = []

  def extern(m: re.Match[str]) -> str:
    """The extern form of literal `m`."""
    seen.append(m.group(1))
    return _EXTERN.format(fn=_fn(m.group(1)), macro=m.group(1))

  source = _LITERAL.sub(extern, c.source)
  if not seen or len(set(seen)) != len(seen):
    raise ValueError(f"{c.cubin.name}: word literals {seen} in the source (expected each defined word once)")
  return source


def nvrtc_options(c: Captured) -> list[str]:
  """Warp's NVRTC options of the captured module (`_RELEASE`, its include paths and math)."""
  native = Path(wp.__file__).parent / "native"
  return [
    f"--gpu-architecture=sm_{c.arch}",
    f"--include-path={native}",
    *_RELEASE,
    *(["--use_fast_math"] if c.fast_math else []),
    f"--fmad={'true' if c.fuse_fp else 'false'}",
    *(f"--include-path={d}" for d in c.include_dirs),
  ]


def link_options(c: Captured) -> list[str]:
  """nvJitLink's options for the captured module: LTO for its architecture and its math (which
  the LTO-IR does not carry: without them the FTZ instructions drop, the numerics change)."""
  fast = ["-ftz=1", "-prec-div=0", "-prec-sqrt=0"] if c.fast_math else ["-ftz=0", "-prec-div=1", "-prec-sqrt=1"]
  return ["-lto", f"-arch=sm_{c.arch}", *fast, f"-fma={int(c.fuse_fp)}"]


def group_ltoir(c: Captured) -> bytes:
  """The group's pre-optimised LTO-IR (once per group: NVVM optimises here, not per link)."""
  return _cuda().compile(
    lto_source(c), [*nvrtc_options(c), "-dlto", "--relocatable-device-code=true", "--gen-opt-lto"], "LTOIR"
  )


def combination_cubin(c: Captured, words: Words, ltoir: bytes) -> bytes:
  """The cubin of the word module `c` (its words `words`, `words_of`): the group's LTO-IR linked
  with the words' translation unit. The one per-word compile (an NVRTC compile of `c.source`
  would replace it: `nvrtc_cubin`)."""
  cuda = _cuda()
  unit = "".join(_WORD_TU.format(fn=_fn(macro), value=value) for macro, value in words.items())
  word_ir = cuda.compile(unit, [*nvrtc_options(c), "-dlto", "--relocatable-device-code=true"], "LTOIR")
  return cuda.link([ltoir, word_ir, *c.ltoirs], link_options(c))


def nvrtc_cubin(c: Captured) -> bytes:
  """The cubin of the word module `c` compiled whole by NVRTC (its literal word): what Warp
  compiles, the reference `combination_cubin` is checked against."""
  return _cuda().compile(c.source, nvrtc_options(c), "CUBIN")


def install(c: Captured, module: wp.Module, cubin: bytes, cache: Path) -> Path:
  """Writes the cache entry of word module `module` of the captured group member `c` into Warp's
  kernel cache `cache` (`<module id>/`): `cubin` and the group's meta (the kernels' shared memory
  by symbol: equal for every word); its directory."""
  name = module.get_module_identifier()
  scratch = Path(tempfile.mkdtemp(prefix="wpbr_sw_entry_"))
  try:
    meta = next(c.directory.glob("*.meta"))
    shutil.copyfile(meta, scratch / f"{name}.meta")
    (scratch / f"{name}.sm{c.arch}.cubin").write_bytes(cubin)
    move_entry(scratch, cache / name)
  finally:
    shutil.rmtree(scratch, ignore_errors=True)
  return cache / name


def move_entry(scratch: Path, entry: Path) -> None:
  """Moves a Warp AOT output `scratch` into cache entry `entry`, the binary last (an entry with
  a binary is whole: a resumed shard skips it, a snapshot may copy it mid-move); no source."""
  entry.mkdir(parents=True, exist_ok=True)
  for f in sorted(scratch.iterdir(), key=lambda f: f.suffix in (".cubin", ".o")):
    if f.suffix != ".cu":
      shutil.move(str(f), entry / f.name)


def fill_group(modules: list[wp.Module], arch: int, cache: Path) -> int:
  """Every word module of one group (`group_key`) into `cache`: one capture (the codegen of its
  first module), one LTO-IR, one link per word (a word is only its translation unit: the module
  of each word is its id); the modules filled. Modules whose entry `cache` holds are skipped (a
  resumed shard)."""
  todo = [m for m in modules if not any((cache / m.get_module_identifier()).glob(f"*.sm{arch}*.cubin"))]
  if not todo:
    return 0
  c = capture(todo[0], arch, Path(tempfile.mkdtemp(prefix="wpbr_sw_")))
  try:
    t0 = time.perf_counter()
    ltoir = group_ltoir(c)
    print(f"FILL warp LTO-IR {len(todo)} words {time.perf_counter() - t0:.0f} s", flush=True)
    for module in todo:
      words = words_of(module)
      if not words:
        raise ValueError(f"{module.name}: not a word module")
      t0 = time.perf_counter()
      entry = install(c, module, combination_cubin(c, words, ltoir), cache)
      print(f"FILL warp link {entry.name} words {words} {time.perf_counter() - t0:.0f} s", flush=True)  # the heartbeat
  finally:
    shutil.rmtree(c.directory, ignore_errors=True)
  return len(todo)


@dataclasses.dataclass(frozen=True)
class _Cuda:
  """NVRTC and nvJitLink (ctypes)."""

  nvrtc: ctypes.CDLL
  jit: ctypes.CDLL

  def compile(self, source: str, options: list[str], output: str) -> bytes:
    """NVRTC of `source` with `options`; its `output` ("CUBIN", "LTOIR")."""
    prog = ctypes.c_void_p()
    _check(self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"wp_pbr.cu", 0, None, None), "nvrtc create")
    try:
      opts = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
      rc = self.nvrtc.nvrtcCompileProgram(prog, len(options), opts)
      if rc != 0:
        raise RuntimeError(f"NVRTC {rc}: {self._log(prog)[-3000:]}")
      return self._read(getattr(self.nvrtc, f"nvrtcGet{output}Size"), getattr(self.nvrtc, f"nvrtcGet{output}"), prog)
    finally:
      self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

  def link(self, inputs: list[bytes], options: list[str]) -> bytes:
    """nvJitLink of the LTO-IR `inputs` with `options`: the cubin."""
    handle = ctypes.c_void_p()
    opts = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
    _check(self.jit.nvJitLinkCreate(ctypes.byref(handle), len(options), opts), f"nvJitLink create {options}")
    try:
      for i, data in enumerate(inputs):
        _check(self.jit.nvJitLinkAddData(handle, _NVJITLINK_LTOIR, data, len(data), f"in{i}".encode()), "nvJitLink add")
      rc = self.jit.nvJitLinkComplete(handle)
      if rc != 0:
        log = self._read(self.jit.nvJitLinkGetErrorLogSize, self.jit.nvJitLinkGetErrorLog, handle)
        raise RuntimeError(f"nvJitLink {rc}: {log.decode(errors='replace')[-3000:]}")
      return self._read(self.jit.nvJitLinkGetLinkedCubinSize, self.jit.nvJitLinkGetLinkedCubin, handle)
    finally:
      self.jit.nvJitLinkDestroy(ctypes.byref(handle))

  def _log(self, prog: ctypes.c_void_p) -> str:
    """NVRTC's log of `prog`."""
    return self._read(self.nvrtc.nvrtcGetProgramLogSize, self.nvrtc.nvrtcGetProgramLog, prog).decode(errors="replace")

  @staticmethod
  def _read(size_of: Callable[..., int], get: Callable[..., int], handle: ctypes.c_void_p) -> bytes:
    """A sized output of `handle` (`size_of`, then `get`)."""
    n = ctypes.c_size_t()
    size_of(handle, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    get(handle, buf)
    return buf.raw


def _check(rc: int, what: str) -> None:
  """Raises RuntimeError on a nonzero CUDA library result."""
  if rc != 0:
    raise RuntimeError(f"{what}: error {rc}")


# The fill's CUDA libraries (`cuda_libs`).
_LIBS: Final = ("libnvrtc.so.13", "libnvJitLink.so.13")


@functools.cache
def _cuda() -> _Cuda:
  """The fill environment's NVRTC and nvJitLink, checked against Warp's CUDA version.

  Raises:
    RuntimeError: the wheels are absent, or of another CUDA version than Warp's.
  """
  libs = cuda_libs()
  if libs is None:
    raise RuntimeError('no NVRTC / nvJitLink 13 libraries: pip install "nvidia-cuda-nvrtc==13.4.*" "nvidia-nvjitlink==13.4.*"')
  cuda = _Cuda(*(ctypes.CDLL(str(path)) for path in libs))
  major, minor = ctypes.c_int(), ctypes.c_int()
  cuda.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
  from warp._src import context as warp_context  # Warp's CUDA toolkit version

  wp.init()
  runtime = warp_context.runtime
  warp_version = tuple(runtime.toolkit_version or ()) if runtime is not None else ()
  if (major.value, minor.value) != warp_version:
    raise RuntimeError(f"NVRTC {major.value}.{minor.value} beside Warp's CUDA {warp_version}: their LTO-IR must match")
  return cuda


def cuda_libs() -> tuple[Path, Path] | None:
  """The fill's libnvrtc.so.13 and libnvJitLink.so.13 (`_LIBS`: in `WPBR_CUDA_LIB`, else the
  NVIDIA wheels' `nvidia/` package); None: not installed."""
  env = os.environ.get("WPBR_CUDA_LIB")
  if env:
    roots = [Path(env)]
  else:
    try:
      import nvidia  # pyrefly: ignore[missing-import]  # the NVIDIA wheels' namespace (fill environment only)
    except ImportError:
      return None
    roots = [Path(r) for r in nvidia.__path__]
  found = [next((p for root in roots for p in root.glob(f"**/{name}")), None) for name in _LIBS]
  nvrtc, jit = found
  return None if nvrtc is None or jit is None else (nvrtc, jit)
