"""Host-side scene description: its tables (`tables.py`, `lights.py`), a load report and metadata.

Nothing here knows the source format; adapters fill a `SceneBuilder`, the `.wpbr` cache stores
the arrays as they are, and a load memory-maps them back without any per-object work.
`raywarp.core.pipeline.upload` turns a Scene into a GpuScene.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, final, override

from raywarp.scene.model import wpbr
from raywarp.scene.model.lights import Cameras, Environments, Lights
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.table import Table
from raywarp.scene.model.tables import Images, Instances, MaterialExtras, Materials, Meshes, Textures
from raywarp.scene.model.types import Array, Camera, Json

if TYPE_CHECKING:
  from raywarp.scene.lighting.rig import Lighting

# The tables of a Scene, by attribute name (the `.wpbr` array prefix).
# Final[...]: typed as the base class (the inferred type is a union of the nine).
TABLES: Final[Mapping[str, type[Table]]] = {
  "meshes": Meshes,
  "instances": Instances,
  "materials": Materials,
  "material_extras": MaterialExtras,
  "textures": Textures,
  "images": Images,
  "lights": Lights,
  "environments": Environments,
  "cameras": Cameras,
}


@final
@dataclasses.dataclass
class Scene:
  """Format-neutral host scene: the tables an adapter produced, plus its load report.

  Build one with `SceneBuilder` (or an adapter such as `raywarp.formats.usd.scene`), cache it with
  `save`, and reload it with `Scene.load` (memory-mapped, no per-object work).

  Attributes:
    material_extras: the `OpenPbrExtras` rows of `materials` (`Materials.extras` indexes them).
    meta: JSON metadata (source, adapter options, the cache record of `wpbr.record`).
  """

  meshes: Meshes
  instances: Instances
  materials: Materials
  material_extras: MaterialExtras
  textures: Textures
  images: Images
  lights: Lights
  environments: Environments
  cameras: Cameras
  report: LoadReport = dataclasses.field(default_factory=LoadReport)
  meta: dict[str, Json] = dataclasses.field(default_factory=dict)

  @property
  def nmesh(self) -> int:
    """Number of meshes."""
    return len(self.meshes)

  @property
  def ninstance(self) -> int:
    """Number of instances."""
    return len(self.instances)

  @property
  def nmaterial(self) -> int:
    """Number of materials."""
    return len(self.materials)

  @property
  def ntexture(self) -> int:
    """Number of textures."""
    return len(self.textures)

  @property
  def nimage(self) -> int:
    """Number of images."""
    return len(self.images)

  @property
  def nlight(self) -> int:
    """Number of lights."""
    return len(self.lights)

  @property
  def nenv(self) -> int:
    """Number of environments."""
    return len(self.environments)

  @property
  def ncamera(self) -> int:
    """Number of cameras."""
    return len(self.cameras)

  def camera(self, name: str) -> Camera:
    """Camera named `name`.

    Raises:
      KeyError: for a name no camera has (listing the cameras).
    """
    names = self.cameras.names.tolist()
    if name not in names:
      raise KeyError(f"no camera {name!r} in the scene (cameras: {names})")
    return self.cameras.get(names.index(name))

  def tables(self) -> dict[str, Table]:
    """Every table, by name (`TABLES` order)."""
    fields = vars(self)
    return {name: fields[name] for name in TABLES}

  def arrays(self) -> dict[str, Array]:
    """Every array of the scene, keyed `table.column`."""
    return {f"{t}.{name}": a for t, table in self.tables().items() for name, a in table.arrays().items()}

  @staticmethod
  def from_arrays(arrays: Mapping[str, Array], report: LoadReport | None = None, meta: dict[str, Json] | None = None) -> Scene:
    """Scene viewing `arrays` (keys as `arrays()`; no copy)."""

    def table[T: Table](name: str, cls: type[T]) -> T:
      """Table `name` of `arrays` (keys `<name>.<column>`)."""
      prefix = name + "."
      return cls.from_arrays({k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)})

    return Scene(
      meshes=table("meshes", Meshes),
      instances=table("instances", Instances),
      materials=table("materials", Materials),
      material_extras=table("material_extras", MaterialExtras),
      textures=table("textures", Textures),
      images=table("images", Images),
      lights=table("lights", Lights),
      environments=table("environments", Environments),
      cameras=table("cameras", Cameras),
      report=LoadReport() if report is None else report,
      meta={} if meta is None else meta,
    )

  def with_lighting(self, lighting: Lighting) -> Scene:
    """This scene with `lighting`'s lights, ambient, environments and headlight added (a new
    Scene; `raywarp.scene.lighting.rig.with_lighting`): light a dark scene before `rw.upload`."""
    # Imported here: rig.py builds scenes with SceneBuilder, which imports this module.
    from raywarp.scene.lighting.rig import with_lighting  # noqa: PLC0415 (the import cycle)

    return with_lighting(self, lighting)

  def save(self, path: str | os.PathLike[str], write: Mapping[str, Callable[[wpbr.Writer], None]] | None = None) -> None:
    """Writes the scene as a `.wpbr` file (see `raywarp.scene.model.wpbr`).

    `write`: arrays the caller writes itself, name -> `fn(writer)` (e.g. the texel blob, decoded
    straight into the file); the scene holds them as placeholders of their final dtype and shape.
    """
    arrays = self.arrays() | {f"meshes.{name}": rows for name, rows in self.meshes.stored_rows().items()}
    wpbr.save(path, arrays, lambda: {"report": self.report.to_dict(), "meta": self.meta}, write=write)

  @staticmethod
  def load(path: str | os.PathLike[str]) -> Scene:
    """Memory-maps a `.wpbr` file (one mmap, one view per array; no pxr / format code)."""
    return Scene._of(*wpbr.load(path))

  @staticmethod
  def load_fresh(path: str | os.PathLike[str], key: wpbr.CacheKey, *, verify: wpbr.Verify = "files") -> Scene | None:
    """`load` of a `.wpbr` file fresh for `key` (`wpbr.fresh`, `verify`), else None."""
    found = wpbr.load_fresh(path, key, verify=verify)
    return None if found is None else Scene._of(*found)

  @staticmethod
  def _of(arrays: Mapping[str, Array], header: Mapping[str, Json]) -> Scene:
    """Scene of a loaded `.wpbr` file's arrays and header (its report and metadata)."""
    report, meta = header.get("report"), header.get("meta")
    meshes = Meshes.expanded_rows({k.removeprefix("meshes."): v for k, v in arrays.items() if k.startswith("meshes.")})
    return Scene.from_arrays(
      {**arrays, **{f"meshes.{k}": v for k, v in meshes.items()}},
      LoadReport.from_dict(report if isinstance(report, dict) else {}),
      meta if isinstance(meta, dict) else {},
    )

  def nbytes(self) -> dict[str, int]:
    """Bytes per table."""
    return {name: sum(a.nbytes for a in table.arrays().values()) for name, table in self.tables().items()}

  @override
  def __repr__(self) -> str:
    """Counts of every table."""
    return (
      f"Scene(meshes={self.nmesh}, triangles={len(self.meshes.tris)}, instances={self.ninstance}, "
      f"materials={self.nmaterial}, textures={self.ntexture}, images={self.nimage}, lights={self.nlight}, "
      f"environments={self.nenv}, cameras={self.ncamera})"
    )
