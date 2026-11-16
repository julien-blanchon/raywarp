"""Which cut-out mesh instances get opacity micromaps (`source_of`, for OptiX's
`optix/omm/micromaps.py`): those whose alpha test (`CutSource`) is the same in every world and
frame, read from the scene's static tables only, in a configuration that allows them
(`allowed`)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.records import record_field
from raywarp._base.types import TexRole
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import host_copy

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.trace.context import FrameContext


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CutSource:
  """What the alpha test of a cut-out mesh instance reads, the same in every world and frame.

  Attributes:
    mesh: the instance's mesh.
    texture: its material's opacity texture (index into `GpuScene.textures`), `layout` its texel
      layout (`GpuScene.tex_layout`).
    cutoff: its material's alpha cutoff (> 0).
    repeat: its material's UV tiling (`material.tex_repeat`).
    texcoord: the instance's own-UV address (`MeshInstance.texcoord`; UV_MESH: the mesh's UVs) and
      `uv_xform` their affine transform (row-major 2 x 3).
  """

  mesh: int
  texture: int
  layout: int
  cutoff: float
  repeat: tuple[float, float]
  texcoord: tuple[int, int]
  uv_xform: tuple[float, ...]


# The tables an alpha test reads (`cutout.make_cut_triangle`): a micromap needs all of them
# static (no per-world declaration, no live binding slot).
TABLES: Final = frozenset(
  {
    "instance.mesh",
    "instance.material",
    "material.alpha_cutoff",
    "material.texture",
    "material.tex_repeat",
    "material.uv_world",
  }
)

# One entry of a static table: (rows, ...) every world row's value (one row: a shared table).
type Entry = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def _same_rows(a: Entry) -> bool:
  """Whether every world row of a table entry (axis 0) holds the same value."""
  return bool(np.all(a == a[:1]))


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HostTables:
  """Host copies of what `source_of` reads (`host_tables`: one copy per build).

  Attributes:
    tables: `TABLES` present in the scene's static tables, (rows, ...) each, as float64.
    texcoord: (ninstance, 2) each instance's own-UV address (`MeshInstance.texcoord`).
    uv_xform: (ninstance, 6) its UV transform, row-major.
    tex_layout: (ntex,) texel layout of every texture.
    nmesh, ntex: mesh and texture counts.
  """

  tables: dict[str, Entry]
  texcoord: np.ndarray[[int, 2], np.dtype[np.int32]]
  uv_xform: np.ndarray[[int, 6], np.dtype[np.float32]]
  tex_layout: np.ndarray[[int], np.dtype[np.int32]]
  nmesh: int
  ntex: int

  def entry(self, name: str, *index: int) -> Entry | None:
    """Every world row of table `name`'s entry `index` (None: no table, or out of its range)."""
    a = self.tables.get(name)
    if a is None or any(not 0 <= i < a.shape[1 + k] for k, i in enumerate(index)):
      return None
    return a[(slice(None), *index)]


def host_tables(scene: GpuScene) -> HostTables:
  """The host copies `source_of` reads (the scene's static tables)."""
  records = host_copy(scene.mesh_instances.table)
  return HostTables(
    tables={name: np.asarray(host_copy(a), np.float64) for name in TABLES if isinstance(a := scene.tables.get(name), wp.array)},
    texcoord=np.asarray(record_field(records, "texcoord"), np.int32).reshape(-1, 2),
    uv_xform=np.asarray(record_field(records, "uv_xform"), np.float32).reshape(-1, 6),
    tex_layout=np.asarray(host_copy(scene.tex_layout), np.int32),
    nmesh=len(scene.meshes.faces),
    ntex=len(scene.textures),
  )


def source_of(h: HostTables, gi: int) -> CutSource | None:
  """The alpha test of mesh instance `gi` when it is the same in every world (`h`: the scene's
  static tables), else None (no micromap)."""
  mesh, mat = h.entry("instance.mesh", gi), h.entry("instance.material", gi)
  if mesh is None or mat is None or not _same_rows(mesh) or not _same_rows(mat):
    return None
  m, k = int(mesh[0]), int(mat[0])
  if not 0 <= m < h.nmesh or k < 0:
    return None
  cutoff = h.entry("material.alpha_cutoff", k)
  tex = h.entry("material.texture", k, int(TexRole.OPACITY))
  repeat = h.entry("material.tex_repeat", k)
  if cutoff is None or tex is None or repeat is None or not all(_same_rows(a) for a in (cutoff, tex, repeat)):
    return None
  world = h.entry("material.uv_world", k)
  # World-projected opacity maps read the hit point, not the UVs.
  if world is not None and np.any(world[..., :2] != 0.0):
    return None
  t = int(tex[0])
  if float(cutoff[0]) <= 0.0 or not 0 <= t < h.ntex:
    return None
  return CutSource(
    mesh=m,
    texture=t,
    layout=int(h.tex_layout[t]),
    cutoff=float(cutoff[0]),
    repeat=(float(repeat[0, 0]), float(repeat[0, 1])),
    texcoord=(int(h.texcoord[gi, 0]), int(h.texcoord[gi, 1])),
    uv_xform=tuple(float(x) for x in h.uv_xform[gi]),
  )


def allowed(ctx: FrameContext) -> bool:
  """Whether cut-out meshes may get opacity micromaps (`tuning.Tuning.opacity_micromaps`): the
  configuration tests alpha (a build compiling the test for one that does not reads zero
  cutoffs, `frame_structs.alpha_cutoffs`, which baked micromaps would ignore) and every table the
  alpha test reads is static (no per-world declaration, no live binding slot, no live material
  UVs), so a bake holds in every world and frame."""
  effective = ctx.effective
  if not tuning.current().opacity_micromaps or effective.live_material_uv or not effective.alpha_cutout:
    return False
  return not TABLES & (ctx.per_world_fields | set(ctx.scene.binding.slots(None)))
