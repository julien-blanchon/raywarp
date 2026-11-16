"""raywarp's native build hook: the OptiX SDK headers and the NVIDIA MDL SDK, bundled in the
installed `raywarp/_native/`.

setuptools runs `BuildNative` as the `build_ext` command (pyproject.toml:
`[tool.setuptools.cmdclass]` and the `_pymdlsdk` entry of `ext-modules`) for wheel builds, sdist
installs and editable installs (`uv sync`). It fetches the pinned NVIDIA artifacts below, checks
every sha256 (a mismatch is fatal), and writes:

  _native/optix/include/       the `include/` directory of NVIDIA/optix-dev at OPTIX_COMMIT, plus
                               its LICENSE.txt and license_info.txt
  _native/mdl/lib/             the MDL SDK release binaries: libmdl_sdk and the plugins
                               nv_openimageio, dds and mdl_distiller (the build platform's archive)
  _native/mdl/mdl/             `::nvidia::core_definitions` (the SDK's example MDL library)
  _native/mdl/licenses/        the SDK's licences
  _native/mdl/pymdlsdk.py      the SDK's SWIG binding, generated here from the SDK's sources at
  _native/mdl/_pymdlsdk.abi3.* MDL_SDK_COMMIT with SWIG `-threads` (every SDK call releases the GIL)
                               and tools/native/mdl/pymdlsdk_logging.patch, compiled against the
                               release's headers for the stable ABI (one binary for CPython >= 3.12)
  _native/mdl/cuda_stub/       (Linux) a libcuda.so.1 stub: the release's libmdl_sdk links the CUDA
                               driver for its GPU baker; the loader preloads the stub only where no
                               driver is installed (raywarp bakes on the CPU)
  _native/BUILD.json           what was built: versions, platform, the build key

Environment:
  RAYWARP_NATIVE_ARCHIVES  a directory of pre-fetched artifacts (file names = `Artifact.name`):
                            every artifact is read from there (checked), nothing is downloaded.
  RAYWARP_NATIVE_CACHE     downloads and compiled bindings (default ~/.cache/raywarp/native).
Needs a C++17 compiler (the binding is compiled on the installing machine) and SWIG (the pinned
`swig` wheel of pyproject's build requirements).

  python tools/native/hook.py fetch DIR [--platform NAME ...]   pre-fetch the artifacts into DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import IO, Final, override

from setuptools import Extension
from setuptools.command.build_ext import build_ext

MDL_SDK_VERSION: Final = "2026.0.2"
MDL_SDK_BUILD: Final = "391700.2276"
# The commit of the MDL SDK's release tag (the binding sources and core_definitions.mdl).
MDL_SDK_COMMIT: Final = "1b9592c1b086b691b44d1a30eee4827d1ff55b8c"
# NVIDIA/optix-dev tag v9.0.0 (OptiX ABI 9.0: driver R580 and newer).
OPTIX_VERSION: Final = "9.0.0"
OPTIX_COMMIT: Final = "fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd"
# Bumped when the build below changes what it produces (part of the build key).
REVISION: Final = 1
# CPython 3.12's stable ABI (requires-python).
LIMITED_API: Final = "0x030C0000"
BINDING: Final = "raywarp._native.mdl._pymdlsdk"
PLUGINS: Final = ("libmdl_sdk", "nv_openimageio", "dds", "mdl_distiller")
CHUNK: Final = 1 << 20
HERE: Final = Path(__file__).resolve().parent
PATCH: Final = HERE / "mdl" / "pymdlsdk_logging.patch"


class NativeBuildError(RuntimeError):
  """A native artifact is missing, fails its checksum, or the platform is unsupported."""


@dataclasses.dataclass(frozen=True, slots=True)
class Artifact:
  """A pinned download: its file name (in RAYWARP_NATIVE_ARCHIVES and the cache), URL, sha256."""

  name: str
  url: str
  sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class Platform:
  """The MDL SDK release of one platform and how the binding is compiled there."""

  release: Artifact
  # Where the release keeps the shared libraries, and their suffix.
  lib_dir: str
  suffix: str
  defines: tuple[str, ...]
  compile_args: tuple[str, ...]
  link_args: tuple[str, ...]
  libraries: tuple[str, ...]
  # The release's libmdl_sdk links the CUDA driver (Linux): ship a stub for driverless machines.
  cuda_stub: bool


def _release(name: str, sha256: str) -> Artifact:
  """A release archive of the MDL SDK."""
  file = f"MDL-SDK-{MDL_SDK_VERSION}-{MDL_SDK_BUILD}-{name}"
  return Artifact(file, f"https://github.com/NVIDIA/MDL-SDK/releases/download/{MDL_SDK_VERSION}/{file}", sha256)


def _source(path: str, sha256: str) -> Artifact:
  """A file of the MDL SDK's repository at MDL_SDK_COMMIT."""
  name = f"MDL-SDK-{MDL_SDK_COMMIT[:12]}-{PurePosixPath(path).name}"
  return Artifact(name, f"https://raw.githubusercontent.com/NVIDIA/MDL-SDK/{MDL_SDK_COMMIT}/{path}", sha256)


OPTIX_HEADERS: Final = Artifact(
  f"optix-dev-{OPTIX_COMMIT}.tar.gz",
  f"https://github.com/NVIDIA/optix-dev/archive/{OPTIX_COMMIT}.tar.gz",
  "d5239da7ad098ae24afc6651d0cd3020ab0346a575ff4935a60e5494fd0bbe28",
)
# The binding's sources (src/prod/bindings/mdl_python), the examples' library loader it includes,
# and the MDL module the SDK's examples ship (not in the release archives).
BINDING_DIR: Final = "src/prod/bindings/mdl_python"
SOURCES: Final = {
  f"{BINDING_DIR}/instantiate_templates_inline.py": "f02cbb9a4a6f185bed6e8e33c729e1a8059aee1fe1c92d17a58d81e276b13323",
  f"{BINDING_DIR}/mdl_python.cpp": "197e7c8a204084d5ec721a9e8c897d20762b10a205e504a0aca2c81284f708f6",
  f"{BINDING_DIR}/mdl_python.h": "57a833b07db88ece9bd4d898e53058a304a7344e7e72f0a30c82b11ee7704ebb",
  f"{BINDING_DIR}/mdl_python_swig.i": "b4ccc79f3bae8fffd18f13b425dc47a18285f6bc0d6081de8fca78bb08c0bb44",
  f"{BINDING_DIR}/post_swig.py": "943ef3eb81910ed1c8015c4eab27af6e7dd43fa9efac425403ca897ee84b7063",
  f"{BINDING_DIR}/swig_library/mi_neuraylib_deprecated.i": "b1ea150ea6de373d8e7f6611fad09e60d2dc6020301f54ce4a810ef45641c61a",
  f"{BINDING_DIR}/swig_library/mi_neuraylib_enums.i": "09a0090116cf27a86dc78bd2fff82978141f6495c3c08cc1c29a717dc7f18576",
  "examples/mdl_sdk/shared/utils/loading.cpp": "2e1da017789efca9e1f7aa140b187d15f01295aa5472d994cf333211b58e0a8c",
  "examples/mdl_sdk/shared/utils/loading.h": "082228a7d9123683f07956c6e6ba75c9a75565c3adbbbe801ba177c6d712eba0",
  "examples/mdl/nvidia/core_definitions.mdl": "f444cf349eb1750aba562ead8a246ac38a7cd018d968342925abfc4cb242cbf9",
}
SOURCE_ARTIFACTS: Final = {path: _source(path, sha) for path, sha in SOURCES.items()}

# The SDK's own compile definitions per platform (its cmake/utilities.cmake), for the binding.
_GCC: Final = ("-std=c++17", "-O2", "-g0", "-fvisibility=hidden", "-fno-strict-aliasing", "-w")
_GCC_LINK: Final = ("-s", "-static-libstdc++", "-static-libgcc", "-Wl,--exclude-libs,ALL")
_CLANG: Final = ("-std=c++17", "-stdlib=libc++", "-O2", "-g0", "-fvisibility=hidden", "-w")
_MSVC: Final = ("/std:c++17", "/O2", "/EHsc", "/bigobj", "/w")
_MSVC_DEFINES: Final = ("MI_PLATFORM_WINDOWS", "WIN_NT", "_CRT_SECURE_NO_WARNINGS", "_SCL_SECURE_NO_WARNINGS")


def _linux(release: Artifact, *defines: str) -> Platform:
  """A Linux release (GCC; the binding links libstdc++ statically)."""
  args = (*_GCC, "-march=nocona") if "HAS_SSE" in defines else _GCC
  return Platform(release, "lib", ".so", ("MI_PLATFORM_UNIX", "LINUX", *defines), args, _GCC_LINK, ("dl", "pthread"), True)


def _macos(release: Artifact, *defines: str) -> Platform:
  """A macOS release (Apple clang, libc++)."""
  return Platform(release, "lib", ".so", ("MI_PLATFORM_MACOSX", "MACOSX", *defines), _CLANG, ("-Wl,-S",), (), False)


# (sys.platform, platform.machine()) -> Platform. The release has no other platforms.
PLATFORMS: Final = {
  ("linux", "x86_64"): _linux(
    _release("linux-x86-64.tgz", "5aa37997647c757d27a0c16d467d5f3f26244a2303e697d4eabf0e3719b8dc19"), "HAS_SSE"
  ),
  ("linux", "aarch64"): _linux(
    _release("linux-aarch64.tgz", "7041a22a81101608f636b1d9496a18c88a8232be5144d0a9c032826eb0469788"), "AARCH64"
  ),
  ("darwin", "arm64"): _macos(
    _release("macosx-aarch64.tgz", "dd9b196008c1fa06ec8dd6dd3d6fbb6621a4b9b92d986199cce2f8ec9793d73e"), "AARCH64"
  ),
  ("darwin", "x86_64"): _macos(
    _release("macosx-x86-64.tgz", "5c5f50b824d9b346b79a3f1d7c42d3915303b9c92a5c583e64e11282bc61b18d")
  ),
  ("win32", "AMD64"): Platform(
    _release("nt-x86-64.zip", "ab1ce5d0621cf222c6dd1ec2eeff4fd71d2efd8126e4251714b1a0171d84f9fd"),
    "bin",
    ".dll",
    _MSVC_DEFINES,
    _MSVC,
    (),
    (),
    False,
  ),
}

# The CUDA driver functions the release's libmdl_sdk imports (bound at load: BIND_NOW). Without a
# driver each one reports CUDA_ERROR_NO_DEVICE; an SDK version importing more fails to load, naming
# the missing symbol.
CUDA_STUB_SYMBOLS: Final = (
  "cuCtxCreate_v2",
  "cuCtxDestroy_v2",
  "cuCtxPopCurrent_v2",
  "cuCtxPushCurrent_v2",
  "cuDeviceGet",
  "cuDeviceGetAttribute",
  "cuInit",
  "cuLaunchKernel",
  "cuLinkAddData_v2",
  "cuLinkComplete",
  "cuLinkCreate_v2",
  "cuLinkDestroy",
  "cuMemAlloc_v2",
  "cuMemFree_v2",
  "cuMemcpyDtoH_v2",
  "cuMemcpyHtoD_v2",
  "cuModuleGetFunction",
  "cuModuleGetGlobal_v2",
  "cuModuleLoadDataEx",
  "cuModuleUnload",
  "cuStreamSynchronize",
)
CUDA_ERROR_NO_DEVICE: Final = 100


def cuda_stub_source() -> str:
  """The libcuda.so.1 stub's C source (cuDeviceGetCount / cuGetErrorString fill their outputs)."""
  lines = [f"int {name}(void) {{ return {CUDA_ERROR_NO_DEVICE}; }}" for name in CUDA_STUB_SYMBOLS]
  lines.append(f"int cuDeviceGetCount(int *count) {{ if (count) *count = 0; return {CUDA_ERROR_NO_DEVICE}; }}")
  lines.append(
    "int cuGetErrorString(int error, const char **text) "
    '{ (void)error; if (text) *text = "no CUDA driver (raywarp libcuda stub)"; return 0; }'
  )
  return "\n".join(lines) + "\n"


def host_platform() -> Platform:
  """The pins of the running platform.

  Raises:
    NativeBuildError: on a platform without an MDL SDK release.
  """
  key = (sys.platform, platform.machine())
  if key not in PLATFORMS:
    supported = ", ".join(f"{os_}/{arch}" for os_, arch in PLATFORMS)
    raise NativeBuildError(f"raywarp bundles the NVIDIA MDL SDK, released for {supported} only; this is {key[0]}/{key[1]}")
  return PLATFORMS[key]


def sha256_of(path: Path) -> str:
  """The file's sha256 (hex)."""
  digest = hashlib.sha256()
  with path.open("rb") as f:
    while block := f.read(CHUNK):
      digest.update(block)
  return digest.hexdigest()


def verify(path: Path, artifact: Artifact) -> Path:
  """`path`, after checking it holds `artifact`.

  Raises:
    NativeBuildError: on a checksum mismatch.
  """
  actual = sha256_of(path)
  if actual != artifact.sha256:
    raise NativeBuildError(
      f"{path}: sha256 {actual}, pinned {artifact.sha256} ({artifact.url}); delete the file if it is a stale download"
    )
  return path


def cache_dir() -> Path:
  """RAYWARP_NATIVE_CACHE, else ~/.cache/raywarp/native (XDG_CACHE_HOME honoured)."""
  if root := os.environ.get("RAYWARP_NATIVE_CACHE"):
    return Path(root)
  return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "raywarp" / "native"


def download(artifact: Artifact, path: Path) -> Path:
  """Downloads `artifact` to `path` (written to a partial file, checked, then renamed).

  Raises:
    NativeBuildError: on a checksum mismatch (the partial file is deleted).
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
  digest = hashlib.sha256()
  print(f"raywarp native: downloading {artifact.url}", file=sys.stderr)
  with urllib.request.urlopen(artifact.url, timeout=60) as response, partial.open("wb") as f:
    while block := response.read(CHUNK):
      digest.update(block)
      f.write(block)
  if digest.hexdigest() != artifact.sha256:
    partial.unlink()
    raise NativeBuildError(f"{artifact.url}: sha256 {digest.hexdigest()}, pinned {artifact.sha256}")
  partial.replace(path)
  return path


def fetch(artifact: Artifact) -> Path:
  """The checked local copy of `artifact`: from RAYWARP_NATIVE_ARCHIVES when set (nothing is
  downloaded then), else from the cache, downloaded on a miss.

  Raises:
    NativeBuildError: when RAYWARP_NATIVE_ARCHIVES lacks it, or on a checksum mismatch.
  """
  if archives := os.environ.get("RAYWARP_NATIVE_ARCHIVES"):
    path = Path(archives) / artifact.name
    if not path.is_file():
      raise NativeBuildError(f"{path} is missing (RAYWARP_NATIVE_ARCHIVES={archives}; source {artifact.url})")
    return verify(path, artifact)
  path = cache_dir() / "archives" / artifact.name
  return verify(path, artifact) if path.is_file() else download(artifact, path)


def _member_path(name: str, prefix: str) -> PurePosixPath | None:
  """The path of archive member `name` below `prefix` (None outside it).

  Raises:
    NativeBuildError: on an absolute path or a `..` component.
  """
  path = PurePosixPath(name)
  if path.is_absolute() or ".." in path.parts:
    raise NativeBuildError(f"unsafe archive member {name!r}")
  if not path.is_relative_to(prefix):
    return None
  return path.relative_to(prefix)


def extract(archive: Path, prefix: str, select: Callable[[PurePosixPath], Path | None]) -> int:
  """Writes the regular files of `archive` (.tar.gz / .tgz / .zip) under `prefix` that `select`
  maps to a destination; returns how many."""
  count = 0

  def write(rel: PurePosixPath, read: Callable[[], Iterator[bytes]]) -> None:
    nonlocal count
    if (dest := select(rel)) is None:
      return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
      for block in read():
        f.write(block)
    count += 1

  if archive.suffix == ".zip":
    with zipfile.ZipFile(archive) as z:
      for info in z.infolist():
        rel = _member_path(info.filename, prefix)
        if rel is not None and not info.is_dir():
          write(rel, lambda info=info: _blocks(z.open(info)))
    return count
  with tarfile.open(archive, "r:gz") as t:
    for member in t:
      rel = _member_path(member.name, prefix)
      if rel is not None and member.isfile():
        write(rel, lambda member=member: _blocks(t.extractfile(member)))
  return count


def _blocks(f: IO[bytes] | None) -> Iterator[bytes]:
  """An archive member's content in blocks (closed at the end)."""
  if f is None:
    raise NativeBuildError("an archive member has no content")
  with f:
    while block := f.read(CHUNK):
      yield block


def apply_patch(patch: str, root: Path) -> None:
  """Applies a unified diff (git format) to the files under `root`; every hunk must match exactly at
  its line.

  Raises:
    NativeBuildError: when a hunk does not match.
  """
  lines = patch.splitlines(keepends=True)
  path: Path | None = None
  text: list[str] = []
  shift = 0
  k = 0
  while k < len(lines):
    line = lines[k]
    k += 1
    if line.startswith("+++ "):
      if path is not None:
        path.write_text("".join(text))
      path = root / line[4:].strip().removeprefix("b/")
      text, shift = path.read_text().splitlines(keepends=True), 0
    elif line.startswith("@@ ") and path is not None:
      old_spec, new_spec = line.split()[1:3]
      start, n_old = _hunk_range(old_spec)
      _, n_new = _hunk_range(new_spec)
      old: list[str] = []
      new: list[str] = []
      while len(old) < n_old or len(new) < n_new:
        tag, body = lines[k][:1], lines[k][1:]
        k += 1
        if tag in " -":
          old.append(body)
        if tag in " +":
          new.append(body)
      at = start - 1 + shift
      if text[at : at + len(old)] != old:
        raise NativeBuildError(f"{PATCH.name}: the hunk {line.strip()} does not match {path}")
      text[at : at + len(old)] = new
      shift += len(new) - len(old)
  if path is not None:
    path.write_text("".join(text))


def _hunk_range(spec: str) -> tuple[int, int]:
  """`-12,3` / `+12` -> (12, 3) / (12, 1)."""
  start, _, count = spec[1:].partition(",")
  return int(start), int(count or 1)


def build_key(plat: Platform) -> str:
  """The digest of every input of the compiled binding (and stub): a cache entry is reused only when
  all of them are equal."""
  inputs = {
    "revision": REVISION,
    "release": plat.release.sha256,
    "sources": SOURCES,
    "patch": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
    "swig": importlib.metadata.version("swig"),
    "limited_api": LIMITED_API,
    "platform": dataclasses.asdict(plat),
    "stub": cuda_stub_source() if plat.cuda_stub else "",
  }
  return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def stage_optix(native: Path) -> None:
  """`native`/optix: optix-dev's include/ and licence files."""
  dest = native / "optix"
  shutil.rmtree(dest, ignore_errors=True)
  licences = {"LICENSE.txt", "license_info.txt"}

  def select(rel: PurePosixPath) -> Path | None:
    if rel.parts[0] == "include" or str(rel) in licences:
      return dest.joinpath(*rel.parts)
    return None

  if not extract(fetch(OPTIX_HEADERS), f"optix-dev-{OPTIX_COMMIT}", select) or not (dest / "include/optix.h").is_file():
    raise NativeBuildError(f"{OPTIX_HEADERS.name} holds no include/optix.h")


def stage_mdl(native: Path, plat: Platform, include: Path) -> None:
  """`native`/mdl: the release's libraries and licences, core_definitions.mdl; the release's
  include/ to `include` (compile only, not installed)."""
  mdl = native / "mdl"
  for sub in ("lib", "mdl", "licenses"):
    shutil.rmtree(mdl / sub, ignore_errors=True)
  libs = {f"{plat.lib_dir}/{name}{plat.suffix}" for name in PLUGINS}

  def select(rel: PurePosixPath) -> Path | None:
    if str(rel) in libs:
      return mdl / "lib" / rel.name
    if len(rel.parts) == 1 and rel.name.startswith("LICENSE"):
      return mdl / "licenses" / rel.name
    if rel.parts[0] == "include":
      return include.joinpath(*rel.parts[1:])
    return None

  extract(fetch(plat.release), plat.release.name.removesuffix(".tgz").removesuffix(".zip"), select)
  missing = [lib for lib in libs if not (mdl / "lib" / PurePosixPath(lib).name).is_file()]
  if missing or not (include / "mi/mdl_sdk.h").is_file():
    raise NativeBuildError(f"{plat.release.name} lacks {missing or 'include/mi/mdl_sdk.h'}")
  core = SOURCE_ARTIFACTS["examples/mdl/nvidia/core_definitions.mdl"]
  (mdl / "mdl" / "nvidia").mkdir(parents=True)
  shutil.copyfile(fetch(core), mdl / "mdl" / "nvidia" / "core_definitions.mdl")


def generate_binding(work: Path, include: Path) -> Path:
  """The SDK's binding generation (template expansion, SWIG -threads, post-processing) in `work`;
  returns the directory holding mdl_python_swig.cpp and pymdlsdk.py."""
  src = work / "src"
  for path, artifact in SOURCE_ARTIFACTS.items():
    (src / path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fetch(artifact), src / path)
  binding = src / BINDING_DIR
  apply_patch(PATCH.read_text(), binding)
  out = work / "generated"
  processed = work / "processed_headers"
  out.mkdir()
  processed.mkdir()
  # The SDK's scripts: pinned by checksum, run isolated (-I) with arguments only.
  run = [sys.executable, "-I"]
  template = [str(binding / "instantiate_templates_inline.py"), str(binding / "mdl_python_swig.i"), ""]
  _check_call([*run, *template, str(include), str(processed), str(work / "processed_headers.d")], work)
  # The SDK's SWIG command line (its CMakeLists.txt) plus -threads.
  flags = ["-c++", "-python", "-threads", "-doxygen", "-w564", "-DSWIGWORDSIZE64", "-DMDL_SOURCE_RELEASE"]
  includes = ["-I-", f"-I{binding}", f"-I{processed}", f"-I{binding / 'swig_library'}", f"-I{include}"]
  outputs = ["-cppext", "cpp", "-o", str(out / "mdl_python_swig.cpp"), "-outdir", str(out), "-module", "pymdlsdk"]
  swig = [_swig(), *flags, *includes, *outputs, str(binding / "mdl_python_swig.i")]
  _check_call(swig, work)
  _check_call([*run, str(binding / "post_swig.py"), "-outdir", str(out), "-module", "pymdlsdk"], work)
  return out


def _swig() -> str:
  """The SWIG executable of the `swig` build requirement (only present in the build environment)."""
  exe = Path(importlib.import_module("swig").BIN_DIR) / ("swig.exe" if sys.platform == "win32" else "swig")
  if not exe.is_file():
    raise NativeBuildError(f"the swig build requirement has no executable at {exe}")
  return str(exe)


def _check_call(command: list[str], cwd: Path) -> None:
  """Runs a build step; its output is shown when it fails.

  Raises:
    NativeBuildError: when it exits non-zero.
  """
  done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
  if done.returncode:
    raise NativeBuildError(f"{Path(command[0]).name} {' '.join(command[1:3])} ... failed:\n{done.stdout}\n{done.stderr}")


class BuildNative(build_ext):
  """setuptools' build_ext, plus the native components (module docstring). The `_pymdlsdk`
  extension's sources are generated here; its compiled form, the SWIG module and the CUDA stub are
  cached by build key in RAYWARP_NATIVE_CACHE."""

  _plat: Platform
  _key: str
  # The installed package's _native/ (the source tree's for an editable install).
  _native: Path
  # The build's temporary directory and the release's include/ there (compile only).
  _temp: Path
  _include: Path

  @override
  def run(self) -> None:
    """Stages `_native/` (into the source tree for an editable install, else the build tree), then
    builds the extension."""
    plat = host_platform()
    self._plat = plat
    self._key = build_key(plat)
    # setuptools builds in the project root (no package_dir mapping: raywarp/ is at the root).
    root = Path.cwd() if self.inplace else Path(self.build_lib)
    self._native = (root / "raywarp" / "_native").resolve()
    self._temp = Path(self.build_temp).resolve()
    self._include = self._temp / "mdl-include"
    record = {
      "mdl_sdk": MDL_SDK_VERSION,
      "mdl_sdk_build": MDL_SDK_BUILD,
      "optix": OPTIX_VERSION,
      "optix_commit": OPTIX_COMMIT,
      "platform": plat.release.name,
      "key": self._key,
    }
    stamp = self._native / "BUILD.json"
    if not (stamp.is_file() and json.loads(stamp.read_text()) == record and self.inplace):
      stage_optix(self._native)
      stage_mdl(self._native, plat, self._include)
      stamp.write_text(json.dumps(record, indent=2) + "\n")
    super().run()

  @override
  def build_extension(self, ext: Extension) -> None:
    """`_pymdlsdk` from the cache, else generated, compiled and cached; pymdlsdk.py and the CUDA
    stub next to the package."""
    if ext.name != BINDING:
      super().build_extension(ext)
      return
    cached = cache_dir() / "build" / self._key
    if not cached.is_dir():
      self._build_binding(ext, cached)
    target = Path(self.get_ext_fullpath(ext.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached / target.name, target)
    mdl = self._native / "mdl"
    shutil.copyfile(cached / "pymdlsdk.py", mdl / "pymdlsdk.py")
    if self._plat.cuda_stub:
      (mdl / "cuda_stub").mkdir(exist_ok=True)
      shutil.copyfile(cached / "libcuda.so.1", mdl / "cuda_stub" / "libcuda.so.1")

  def _build_binding(self, ext: Extension, cached: Path) -> None:
    """Generates and compiles the binding (and the stub) into a new cache entry `cached`."""
    plat = self._plat
    if not (self._include / "mi/mdl_sdk.h").is_file():  # an editable rebuild that skipped staging
      stage_mdl(self._temp / "native", plat, self._include)
    self._temp.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=self._temp, prefix="pymdlsdk-"))
    generated = generate_binding(work, self._include)
    binding = work / "src" / BINDING_DIR
    ext.sources = [str(binding / "mdl_python.cpp"), str(generated / "mdl_python_swig.cpp")]
    ext.include_dirs = [str(binding), str(self._include), str(work / "src/examples/mdl_sdk/shared")]
    ext.define_macros = [("Py_LIMITED_API", LIMITED_API), ("NDEBUG", None), ("MDL_SOURCE_RELEASE", None)]
    ext.define_macros += [(name, None) for name in plat.defines]
    ext.extra_compile_args = list(plat.compile_args)
    ext.extra_link_args = list(plat.link_args)
    ext.libraries = list(plat.libraries)
    ext.language = "c++"
    super().build_extension(ext)
    cached.parent.mkdir(parents=True, exist_ok=True)
    entry = Path(tempfile.mkdtemp(dir=cached.parent, prefix=".partial-"))
    built = Path(self.get_ext_fullpath(ext.name))
    shutil.copyfile(built, entry / built.name)
    shutil.copyfile(generated / "pymdlsdk.py", entry / "pymdlsdk.py")
    if plat.cuda_stub:
      self._build_cuda_stub(work, entry / "libcuda.so.1")
    try:
      entry.replace(cached)
    except OSError:  # another build cached the same key first
      if not cached.is_dir():
        raise

  def _build_cuda_stub(self, work: Path, dest: Path) -> None:
    """libcuda.so.1 (soname libcuda.so.1) from `cuda_stub_source()`."""
    source = work / "cuda_stub.c"
    source.write_text(cuda_stub_source())
    compiler = self.compiler
    if compiler is None or isinstance(compiler, str):
      raise RuntimeError(f"build_ext has no C compiler for the CUDA stub (got {compiler!r})")
    objects = compiler.compile([str(source)], output_dir=str(work), extra_postargs=["-fPIC", "-O2", "-g0"])
    compiler.link_shared_object(objects, str(dest), extra_postargs=["-s", "-Wl,-soname,libcuda.so.1"])


def main() -> None:
  """`fetch DIR [--platform ...]`: the pinned artifacts (default: all platforms) into DIR."""
  names = {plat.release.name.split("-", 4)[-1].split(".")[0]: plat for plat in PLATFORMS.values()}
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)
  p = sub.add_parser("fetch", help="download the pinned artifacts into a directory (RAYWARP_NATIVE_ARCHIVES)")
  p.add_argument("dir", type=Path)
  p.add_argument("--platform", action="append", choices=sorted(names), help="release platforms (default: all)")
  args = parser.parse_args()
  wanted = [names[n].release for n in args.platform or sorted(names)]
  for artifact in [OPTIX_HEADERS, *SOURCE_ARTIFACTS.values(), *wanted]:
    path = args.dir / artifact.name
    if path.is_file():
      verify(path, artifact)
    else:
      download(artifact, path)
    print(f"{artifact.sha256}  {path}")


if __name__ == "__main__":
  main()
