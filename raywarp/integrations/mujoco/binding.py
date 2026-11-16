"""MuJoCo Warp binding: the scene's slots alias `mjw.Model` tables and `mjw.Data` poses.

Shared scenery (`raywarp.core.pipeline.worlds.sharing`): a geom is moved per world
(`MjwBinding.dynamic`) when a body up its chain has a joint, is a mocap body posed differently
between worlds when the scene is built (rows of `d.mocap_pos` / `d.mocap_quat`), or when a batched
pose field of `m` (geom_pos, geom_quat, or body_pos / body_quat of a body up its chain) differs
between worlds. The other geoms are static and the same in every world: a Renderer shares them
(world 0's row of `d.geom_xpos` / `d.geom_xmat`), unless its `PerWorld` declares them or a batched
geom_size / geom_dataid / alpha differs (the core's implicit check).

A mocap body posed the same in every world (a room or table mjlab wraps in a mocap body) is
shared scenery: the per-frame validation (`raywarp.core.pipeline.worlds.validator`) refits the
shared trees when it later moves the same in every world, and raises when it moves differently per
world (declare it with `PerWorld(pose=[...])` at construction).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar, Final, final, override

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp._base.types import ColorRule
from raywarp.core.kernels.abi.tables import tile_rows
from raywarp.core.pipeline.gpu_scene.binding import Binding, RowsSource, Slots, merged
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.integrations.mujoco.capabilities import Content, LiveCapabilityCheck
from raywarp.integrations.mujoco.hfield import HeightFields

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray, Ids, Mask
  from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes

# Rank of a batched MuJoCo Warp field without a trailing axis: (nworld or 1, n).
BATCHED_NDIM: Final = 2

# The one map from MuJoCo Warp names to the core's vocabulary, in the core's slot order (keys are
# `core.pipeline.gpu_scene.binding.SLOTS` names): core slot -> its `mjw.Model` table, re-read on
# every call (replacing an attribute of m takes effect) ...
# Final[...]: lambdas infer untyped parameters; the annotation types them.
MODEL_SLOTS: Final[dict[str, Callable[[mjw.Model], AnyDeviceArray]]] = {
  "instance.mesh": lambda m: m.geom_dataid,
  "instance.material": lambda m: m.geom_matid,
  "instance.size": lambda m: m.geom_size,
  "instance.color": lambda m: m.geom_rgba,
  "material.base_color": lambda m: m.mat_rgba,
  "material.emission": lambda m: m.mat_emission,
  "material.texture": lambda m: m.mat_texid,
  "material.tex_repeat": lambda m: m.mat_texrepeat,
  "material.tex_uniform": lambda m: m.mat_texuniform,
  # The reflectance r (F0 = 0.16 r^2) is s: a write to `m.mat_specular` writes r (`materials.py`).
  "material.reflectance": lambda m: m.mat_specular,
  "light.type": lambda m: m.light_type,
  "light.cast_shadow": lambda m: m.light_castshadow,
  "light.active": lambda m: m.light_active,
  "light.color": lambda m: m.light_diffuse,
  "light.cutoff": lambda m: m.light_cutoff,
}
# `mjw.Model` tables MuJoCo Warp's renderer reads per world that raywarp does not read (their
# values at scene creation hold): a batched one (more than one row, i.e. randomised per world)
# warns once, naming the per-world field to randomise instead (`MjwBinding.slots`). A batched
# `m.mat_shininess` is read (`live_inputs`): the roughness of the materials that leave it unset.
UNREAD_BATCHED: Final = {
  "cam_projection": "the projection: one CameraSpec per projection",
  "cam_sensorsize": 'the intrinsics: PerWorld(camera=["intrinsics"])',
  "light_exponent": "the spot falloff: read at scene creation (the inner cone, `lights.classic_cone_inner`)",
}
# ... or its array of the per-call `mjw.Data`.
# Final[...]: as MODEL_SLOTS.
DATA_SLOTS: Final[dict[str, Callable[[mjw.Data], AnyDeviceArray]]] = {
  "instance.pos": lambda d: d.geom_xpos,
  "instance.rot": lambda d: d.geom_xmat,
  "light.pos": lambda d: d.light_xpos,
  "light.dir": lambda d: d.light_xdir,
  "camera.pos": lambda d: d.cam_xpos,
  "camera.rot": lambda d: d.cam_xmat,
  "flex.vert": lambda d: d.flexvert_xpos,
}


def _differs[D](a: wp.array[D, [int, int]], n: int) -> Mask[int]:
  """(n,) whether a batched (nrow, n, ...) field of `m` differs between its rows."""
  if a.ndim < BATCHED_NDIM or a.shape[0] <= 1:
    return np.zeros(n, bool)
  v = host_copy(a)
  return (v != v[:1]).reshape(v.shape[0], v.shape[1], -1).any(axis=(0, 2))


def _mocap_differs(mocapid: Ids[int], d: mjw.Data | None) -> Mask[int]:
  """(nbody,) mocap bodies whose rows of `d.mocap_pos` / `d.mocap_quat` differ between worlds."""
  mocap = mocapid >= 0
  if d is None or not mocap.any():
    return np.zeros(len(mocapid), bool)
  nmocap = int(mocapid.max()) + 1
  moved = _differs(d.mocap_pos, nmocap) | _differs(d.mocap_quat, nmocap)
  return mocap & moved[np.maximum(mocapid, 0)]


def moving_bodies(mjm: mujoco.MjModel, m: mjw.Model, d: mjw.Data | None) -> Mask[int]:
  """(nbody,) bool: bodies the simulation moves, or posed differently per world in `m` or `d`
  (a body moves when it or an ancestor does)."""
  nbody = mjm.nbody
  moving = (np.asarray(mjm.body_jntnum) > 0) | _mocap_differs(np.asarray(mjm.body_mocapid), d)
  moving |= _differs(m.body_pos, nbody) | _differs(m.body_quat, nbody)
  parent = np.asarray(mjm.body_parentid)
  # Parents come before their children.
  for b in range(1, nbody):
    moving[b] |= moving[parent[b]]
  moving[0] = False
  return moving


def dynamic_geoms(mjm: mujoco.MjModel, m: mjw.Model, d: mjw.Data | None) -> Mask[int]:
  """(ngeom,) bool: geoms the simulation moves, or posed differently per world in `m` or `d`
  (module doc)."""
  ngeom = mjm.ngeom
  out = moving_bodies(mjm, m, d)[np.asarray(mjm.geom_bodyid)]
  return out | _differs(m.geom_pos, ngeom) | _differs(m.geom_quat, ngeom)


def moving_lights(mjm: mujoco.MjModel, m: mjw.Model, d: mjw.Data | None) -> tuple[int, ...]:
  """The lights MuJoCo moves: on a moving body (`moving_bodies`), or tracking one (any mode but
  mjCAMLIGHT_FIXED: track, trackcom, targetbody, targetbodycom)."""
  body = moving_bodies(mjm, m, d)[np.asarray(mjm.light_bodyid)]
  tracking = np.asarray(mjm.light_mode) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
  return tuple(int(i) for i in np.flatnonzero(body | tracking))


@final
class MjwBinding(Binding[mjw.Data]):
  """Aliases `m`'s tables and the per-call `d`'s poses (no copy, no kernel).

  `sim` is an `mjw.Data`. Its color table is MuJoCo's (`ColorRule.MUJOCO`: the default geom rgba
  shows the material's color) and segmentation writes the geom id.

  Attributes:
    m: the device model.
    moved: (ngeom,) bool, `dynamic()`.
    hfields: the rendered height fields' BLASes over `m.hfield_*` (`refit_hfields` after a
      terrain change).
    own: slots the scene holds its own table of instead of aliasing `m`'s (e.g.
      "material.texture" when a texture needs a second color space, `textures.py`).
    rendered: (n,) the rendered geom ids on the device (`feature_checks`); `mesh_geoms` the mesh
      ones, `hfield_geoms` the height-field ones; `meshes` the scene's meshes (`Meshes.used`).
  """

  color_rule: ClassVar[ColorRule] = ColorRule.MUJOCO
  instance_segments: ClassVar[bool] = True

  def __init__(  # noqa: PLR0913  # the scene's tables, keyword-only after the model
    self,
    m: mjw.Model,
    *,
    moved: Mask[int],
    hfields: HeightFields,
    rendered: Ids[int],
    mesh_geoms: Ids[int],
    hfield_geoms: Ids[int],
    meshes: Meshes,
    classic: bool,
    scanned: Content,
    own: frozenset[str] = frozenset(),
    moving: tuple[int, ...] = (),
  ) -> None:
    """The binding of `m`'s scene: `moved` geoms are posed per world (`dynamic_geoms`);
    `mesh_geoms` / `hfield_geoms` the rendered mesh / height-field geoms, `meshes` the scene's
    (`Meshes.used`: which have a BLAS); `moving` the lights MuJoCo moves (`moving_lights`);
    `scanned` what the rendered content uses at scene creation (`capabilities.content`)."""
    self.m = m
    self.scanned = scanned
    self.moved = np.asarray(moved, bool)
    self._moving_lights = moving
    self.hfields = hfields
    device = m.geom_type.device
    self.rendered = wp.array(np.asarray(rendered, np.int32), dtype=int, device=device)
    self.mesh_geoms = wp.array(np.asarray(mesh_geoms, np.int32), dtype=int, device=device)
    self.hfield_geoms = wp.array(np.asarray(hfield_geoms, np.int32), dtype=int, device=device)
    self.meshes = meshes
    self.own = own
    self.classic = classic
    # Batched model tables of K rows (1 < K < nworld) tiled to nworld rows (`_tiled`), by (slot,
    # nworld): Renderers of different world counts over one scene each keep their own buffer.
    self._tiles: dict[tuple[str, int], tuple[tuple[int | None, tuple[int, ...]], AnyDeviceArray]] = {}
    self._motion_ids: tuple[Ids[int], Ids[int]] | None = None
    self._warned: set[str] = set()  # `UNREAD_BATCHED` tables already warned about

  @override
  def feature_checks(self) -> tuple[LiveCapabilityCheck, ...]:
    """The per-frame check that `m`'s live tables need no compiled-out feature
    (`core.pipeline.worlds.validator`)."""
    return (LiveCapabilityCheck(self.rendered, self.mesh_geoms, self.meshes, self.hfield_geoms, self.hfields, self.scanned),)

  def refit_hfields(self) -> None:
    """Re-reads `m.hfield_data` into the height-field BLASes (live terrain; device work only).

    MuJoCo keeps height-field data in [0, 1], inside the box the scene BVHs bound a height field
    by (`hfield_size`), so changed terrain needs no scene-BVH refit.
    """
    self.hfields.refit()
    self.meshes.hfield_revision += 1  # the OptiX height-field GASes refit at their next build

  @override
  def dynamic(self, ninstance: int) -> Mask[int]:
    """The geoms the simulation moves or poses per world (see the module docstring)."""
    return self.moved

  @override
  def moving_lights(self) -> tuple[int, ...]:
    """The lights MuJoCo moves (on a moving body or tracking one)."""
    return self._moving_lights

  @override
  def shared_motion(self, sim: mjw.Data | None, ids: Ids[int]) -> RowsSource | None:
    """The shared geoms' poses in `sim` (checked every frame by the Renderer's validation,
    `raywarp.core.pipeline.worlds.validator`)."""
    if sim is None:
      return None
    cached = self._motion_ids
    if cached is None or cached[0] is not ids:
      # The renderer passes the same (layout's) ids every frame: converted once.
      cached = self._motion_ids = (ids, np.asarray(ids, np.int32))
    return RowsSource(sim.geom_xpos, sim.geom_xmat, cached[1])

  @override
  def slots(self, sim: mjw.Data | None) -> Slots:
    """`m`'s tables and, with `sim`, its pose arrays (K-row tables: their nworld-row tilings,
    written by `write_tables`)."""
    self._warn_unread()
    out = self._model_tables()
    if sim is not None:
      out |= {slot: pose(sim) for slot, pose in DATA_SLOTS.items()}
      out = self._tiled(out, sim.nworld)
    return merged({}, out)

  def _model_tables(self) -> dict[str, AnyDeviceArray]:
    """`m`'s tables by slot (re-read on every call: replacing an attribute of m takes effect)."""
    out = {slot: table(self.m) for slot, table in MODEL_SLOTS.items() if slot not in self.own}
    if self.classic:
      out["light.attenuation"] = self.m.light_attenuation
    return out

  @override
  def write_tables(self, sim: mjw.Data | None) -> None:
    """Tiles every batched K-row table of `m` (1 < K < nworld: MuJoCo Warp's domain
    randomisation, read there as world % K; its model and live camera tables) to the nworld rows
    `slots` and `cameras` serve, one copy kernel per table (`kernels/abi/tables.tile_rows`): MuJoCo
    Warp writes such tables in place, so the copy is refreshed every frame."""
    if sim is None:
      return
    live = self._model_tables() | {"camera.fovy": self.m.cam_fovy, "camera.intrinsics": self.m.cam_intrinsic}
    for slot, a in live.items():
      if self._needs_tiling(slot, a, sim.nworld):
        self._tile(slot, a, sim.nworld, write=True)

  def _warn_unread(self) -> None:
    """Warns once per `UNREAD_BATCHED` table of `m` batched per world (host: shapes only)."""
    for name, instead in UNREAD_BATCHED.items():
      table = getattr(self.m, name, None)
      if name in self._warned or table is None or table.ndim < BATCHED_NDIM or table.shape[0] <= 1:
        continue
      self._warned.add(name)
      warnings.warn(
        f"raywarp does not read m.{name} per world ({table.shape[0]} rows): its values at scene creation "
        f"hold in every world; randomise {instead}",
        stacklevel=4,
      )

  @staticmethod
  def _needs_tiling(slot: str, a: AnyDeviceArray, nworld: int) -> bool:
    """Whether table `a` of `slot` is batched with K rows, 1 < K < nworld (tiled to nworld rows).

    Raises:
      ValueError: a K-row table that is not contiguous (its rows cannot be copied as bytes).
    """
    if a.ndim < BATCHED_NDIM or a.shape[0] == 1 or a.shape[0] >= nworld:
      return False
    if not a.is_contiguous:
      raise ValueError(f"MuJoCo Warp table of slot {slot!r}: {a.shape[0]} rows, not contiguous (raywarp tiles K-row tables)")
    return True

  def _tiled(self, arrays: Mapping[str, AnyDeviceArray], nworld: int) -> dict[str, AnyDeviceArray]:
    """`arrays` with every batched K-row table (`_needs_tiling`) replaced by its nworld-row tiling
    (`_tile`'s buffer, written by `write_tables`): the kernels' row read stays min(world, rows - 1).
    Tables of 1 or nworld rows pass as they are."""
    return {slot: self._tile(slot, a, nworld) if self._needs_tiling(slot, a, nworld) else a for slot, a in arrays.items()}

  def _tile(self, slot: str, a: AnyDeviceArray, nworld: int, *, write: bool = False) -> AnyDeviceArray:
    """The nworld-row tiling buffer of the contiguous K-row table `a` of `slot`, reused while `a`
    keeps its address and shape (one per world count); with `write`, `a`'s rows copied into it."""
    key = (a.ptr, tuple(a.shape))
    cached = self._tiles.get((slot, nworld))
    if cached is None or cached[0] != key:
      cached = self._tiles[slot, nworld] = (key, wp.empty((nworld, *a.shape[1:]), dtype=a.dtype, device=a.device))
    dst = cached[1]
    if not write or a.ptr is None or dst.ptr is None:
      return dst  # an empty table: no bytes to copy
    row = math.prod(a.shape[1:]) * wp.types.type_size_in_bytes(a.dtype)  # bytes per row (contiguous)
    src_bytes = wp.array(ptr=a.ptr, dtype=wp.uint8, shape=(a.shape[0], row), device=a.device)
    dst_bytes = wp.array(ptr=dst.ptr, dtype=wp.uint8, shape=(nworld, row), device=a.device)
    wp.launch(tile_rows, dim=(nworld, row), inputs=[src_bytes], outputs=[dst_bytes], device=a.device)
    return dst

  @override
  def cameras(self, sim: mjw.Data | None) -> dict[str, AnyDeviceArray]:
    """`m.cam_fovy` / `m.cam_intrinsic` (MuJoCo Warp's renderer reads them per world: batched
    tables randomise the cameras), K-row tables tiled as `slots` serves them (`write_tables`)."""
    live = {"camera.fovy": self.m.cam_fovy, "camera.intrinsics": self.m.cam_intrinsic}
    return self._tiled(live, sim.nworld) if sim is not None else live

  @override
  def live_inputs(self) -> dict[str, AnyDeviceArray]:
    """`m.mat_shininess`: the roughness 1 - shininess (clipped) of the materials that leave it
    unset, per world when batched (`derived_fields.SHININESS_SLOT`)."""
    return {"material.shininess": self.m.mat_shininess}

  @override
  def sources(self, sim: mjw.Data | None) -> dict[str, AnyDeviceArray]:
    """The live camera tables (`cameras`) and `m.mat_shininess`: a captured graph reads them."""
    return {f"m.{name}": a for name, a in self.cameras(sim).items()} | {"m.mat_shininess": self.m.mat_shininess}

  @override
  def nworld(self, sim: mjw.Data | None) -> int | None:
    """`sim.nworld` (raises without a sim)."""
    if sim is None:
      raise ValueError("a MuJoCo scene renders the poses of a mujoco_warp.Data: pass it to render(d)")
    return sim.nworld
