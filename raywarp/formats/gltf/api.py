"""`rw.formats.gltf.scene` / `load` / `stage`: glTF 2.0 (.gltf / .glb) -> GpuScene / Scene / USD
stage.

One path: the glTF converts to a USD stage (`to_usd.convert`: UsdPreviewSurface, UsdLux, instancers)
that the USD importer reads (`rw.formats.usd.load`), so a glTF asset renders exactly as its USD
conversion does in every renderer of a comparison. With a cache, the conversion is kept under
`<cache dir>/gltf/<name>-<digest>/` keyed by the content of the glTF and every file it references
plus the converter's code (a changed texture or converter converts again; renaming or moving the
asset does not), and the stage's host scene goes through the `.wpbr` cache next to it. Without
one, the conversion lives in a temporary directory for the duration of the read.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from raywarp.cache import paths as cache_paths
from raywarp.cache.code_version import code_digest
from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.formats.usd import api as usd_api
from raywarp.formats.usd.options import DEFAULT_OPTIONS, Options
from raywarp.scene.model import wpbr

if TYPE_CHECKING:
  import warp as wp

  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.scene.lighting.rig import Lighting
  from raywarp.scene.model.scene import Scene

# The converted stage's file name inside its directory (textures/ next to it).
STAGE_FILE: Final = "scene.usdc"
_PACKAGE: Final = Path(__file__).parent


# The file suffixes of a glTF 2.0 asset (JSON or binary).
_SUFFIXES: Final = (".gltf", ".glb")


def _source(source: str | os.PathLike[str]) -> Path:
  """The path of glTF file `source`.

  Raises:
    ValueError: for a file that is not a .gltf / .glb (another format's would fail as bad JSON).
  """
  path = Path(source)
  if path.suffix.lower() not in _SUFFIXES:
    raise ValueError(f"{path}: not a glTF file (.gltf / .glb; a USD stage loads with rw.formats.usd.load)")
  return path


@functools.cache
def converter_digest() -> str:
  """The digest of the converter's code (this package)."""
  return code_digest((p.name, p.read_bytes()) for p in sorted(_PACKAGE.glob("*.py")))


def content_digest(source: str | os.PathLike[str]) -> str:
  """The digest of a glTF's content: the document, every file it references (missing ones by name)
  and the converter's code."""
  from raywarp.formats.gltf.reader import Gltf  # noqa: PLC0415  # pygltflib on use

  path = Path(source)
  digest = hashlib.blake2b(converter_digest().encode(), digest_size=16)
  for file in Gltf(path).files():
    name = os.path.relpath(file, path.parent)
    digest.update(f"{name}\0".encode())
    digest.update(file.read_bytes() if file.is_file() else b"\0missing")
  return digest.hexdigest()


def stage(
  source: str | os.PathLike[str],
  *,
  cache: str | os.PathLike[str] | bool = True,  # a location or True (the default one)
) -> Path:
  """The USD stage of a glTF in the cache (converted on a miss): `<cache>/gltf/<name>-<digest>/`.

  Raises:
    ValueError: for cache=False (use `load(cache=None)`), or a file that is not a .gltf / .glb.
  """
  path = _source(source)
  if cache is False:
    raise ValueError("gltf.stage needs a cache directory (cache=True or a path)")
  root = (cache_paths.cache_dir() or Path(cache_paths.DEFAULT_DIR).expanduser()) if cache is True else Path(cache).expanduser()
  folder = root / "gltf" / f"{path.stem}-{content_digest(path)}"
  if not (folder / STAGE_FILE).is_file():
    folder.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    try:
      _convert(path, work / STAGE_FILE)
      work.rename(folder)  # atomic: readers see a complete conversion or none
    except OSError as e:
      shutil.rmtree(work, ignore_errors=True)
      # Another process renamed its conversion in first (ENOTEMPTY / EEXIST): use that one.
      if e.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not (folder / STAGE_FILE).is_file():
        raise
    except BaseException:
      shutil.rmtree(work, ignore_errors=True)
      raise
  return folder / STAGE_FILE


def _convert(source: Path, out: Path) -> None:
  """The glTF -> USD conversion (pxr and pygltflib imported on use)."""
  from raywarp.formats.gltf.to_usd import convert  # noqa: PLC0415

  convert(source, out)


def load(  # noqa: PLR0913  # the source and keyword-only options
  source: str | os.PathLike[str],
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  cache: str | os.PathLike[str] | bool | None = True,  # a location, True (the cache directory) or None
  mips: bool = False,
  threads: int | None = None,
  verify: wpbr.Verify = "files",
) -> Scene:
  """Loads a glTF 2.0 file as a host Scene: its USD conversion read by `rw.formats.usd.load`.

  Args:
    source: the .gltf / .glb file.
    options: the options every adapter shares (`rw.SceneOptions`).
    adapter: the USD options of the converted stage's read (`raywarp.formats.usd.Options`).
    lighting: lights, ambient, environment maps and headlight added to the asset's.
    cache: cache directory, True (the default: the cache directory, as `rw.formats.usd.load`; no
      cache when $RAYWARP_CACHE_DIR is empty) or None (no cache: the conversion is temporary).
      Holds the conversion (`stage`) and the `.wpbr` host scene.
    mips: store full mip chains in the scene.
    threads: image decode threads.
    verify: what a cached scene's freshness compares besides its key, as `rw.formats.usd.load`:
      "files" every file of the conversion it was read from, "source" the converted stage only (the
      conversion itself is keyed by the glTF's content).

  Raises:
    TypeError, ValueError: for invalid options, threads or verify (`rw.formats.usd.load`).
    ValueError: for a file that is not a .gltf / .glb.
  """
  _source(source)
  usd_api.check_arguments(options, adapter, threads=threads, verify=verify)  # before the conversion
  read = {"options": options, "adapter": adapter, "lighting": lighting, "mips": mips, "threads": threads, "verify": verify}
  cache = cache_paths.location(cache)
  if cache is None:
    with tempfile.TemporaryDirectory(prefix="rw-gltf-") as tmp:
      out = Path(tmp) / STAGE_FILE
      _convert(Path(source), out)
      return usd_api.load(out, cache=None, **read)
  converted = stage(source, cache=cache)
  return usd_api.load(converted, cache=converted.parent, **read)  # the .wpbr named by its key


def scene(  # noqa: PLR0913  # the source and keyword-only options
  source: str | os.PathLike[str],
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  device: wp.DeviceLike = None,
  cache: str | os.PathLike[str] | bool | None = True,  # a location, True (the cache directory) or None
  mips: bool = False,
  threads: int | None = None,
  verify: wpbr.Verify = "files",
) -> GpuScene[None]:
  """A glTF 2.0 file as a GpuScene on `device`: `load` (see its arguments), then `rw.upload` with
  `options` (the asset's poses are static)."""
  host = load(
    source, options=options, adapter=adapter, lighting=lighting, cache=cache, mips=mips, threads=threads, verify=verify
  )
  return upload(host, device, options=options)
