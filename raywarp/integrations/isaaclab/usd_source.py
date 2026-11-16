"""Isaac clone provenance over an isolated USD session view, without mutating simulation USD."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy
from raywarp.formats.usd.options import Dialect
from raywarp.formats.usd.prim_paths import under

if TYPE_CHECKING:
  import newton
  from pxr import Usd

  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.integrations.newton.binding import IntTable
  from raywarp.integrations.newton.types import NewtonState
  from raywarp.scene.model.scene import Scene


def _alias(label: str, world: int, roots: Sequence[str], homes: Sequence[int], local: dict[int, tuple[str, ...]]) -> str:
  """The prototype path of a body of `world` (-1: global, kept): under its home's root when it is
  that prototype's content, else under world 0's (shared content)."""
  if world < 0:
    return label
  home = homes[world]
  path = rebase(label, roots[world], roots[home])
  if home and any(under(path, k) for k in local[home]):
    return path
  return rebase(label, roots[world], roots[0])


def rebase(path: str, source: str, target: str) -> str:
  """Replace a matching root; preserve unrelated global paths."""
  return target.rstrip("/") + path[len(source.rstrip("/")) :] if under(path, source) else path


# (source path, destination path) of an asset referenced into the prototype view.
type Asset = tuple[str, str]


def world_homes(layout: np.ndarray[[int], np.dtype[np.int32]]) -> tuple[int, ...]:
  """(nworld,) per world, the first world of its prototype (`topology.world_prototype_layout`):
  the root whose prototype content it shows."""
  first: dict[int, int] = {}
  return tuple(first.setdefault(int(p), w) for w, p in enumerate(layout.tolist()))


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CloneAssets:
  """The assets of an Isaac Lab clone plan the prototype view shows.

  Attributes:
    shared: the shared world's (group 0), at world 0's paths: shown in every world.
    local: per home world h (`world_homes`), its prototype's (group layout[h] + 1) at world h's
      paths: shown in the worlds of that prototype.
  """

  shared: tuple[Asset, ...]
  local: dict[int, tuple[Asset, ...]]


def clone_assets(  # noqa: PLR0913, PLR0917  # the clone plan's arrays, in its order
  layout: np.ndarray[[int], np.dtype[np.int32]],
  starts: np.ndarray[[int], np.dtype[np.int64]],
  prototypes: np.ndarray[[int], np.dtype[np.int32]],
  sources: Sequence[str | None],
  templates: Sequence[str],
  homes: Sequence[int],
) -> CloneAssets:
  """The shared assets and each home's prototype assets of an Isaac Lab clone plan.

  The plan's groups: group 0 is the shared world (-1), group p + 1 world prototype p
  (`world_prototype_starts`); a world of no prototype (-1) has no local assets.

  Args:
    layout: (nworld,) `topology.world_prototype_layout`.
    starts: `world_prototype_starts`: group g's members are `prototypes[starts[g]:starts[g + 1]]`.
    prototypes: `topology.world_prototypes` (asset definition per member).
    sources: `get_asset_prototype_paths(plan)` (None: an unused definition).
    templates: `get_world_prototype_asset_templates(plan)[0]` (member path, `{}` the world).
    homes: `world_homes(layout)`.
  """

  def group(g: int, world: int) -> tuple[Asset, ...]:
    """(source, world `world`'s path) of group g's used members."""
    members = range(int(starts[g]), int(starts[g + 1]))
    return tuple((source, templates[i].format(world)) for i in members if (source := sources[int(prototypes[i])]) is not None)

  local = {h: group(int(layout[h]) + 1, h) if int(layout[h]) >= 0 else () for h in sorted(set(homes))}
  return CloneAssets(shared=group(0, 0), local=local)


def _reference_assets(view: Usd.Stage, stage: Usd.Stage, assets: Sequence[tuple[str, str]]) -> None:
  """Reference only missing selected/shared assets; reject cyclic ancestor references."""
  for source, target in assets:
    if not stage.GetPrimAtPath(source):
      raise ValueError(f"Isaac prototype source does not exist: {source}")
    if source != target and under(target, source):
      raise ValueError("an Isaac prototype cannot reference an ancestor of its destination")
    if not view.GetPrimAtPath(target):
      prim = view.DefinePrim(target)
      prim.GetReferences().AddInternalReference(source)
      prim.SetActive(True)


def _keep_only(view: Usd.Stage, root: str, kept: Sequence[str]) -> None:
  """Deactivates every prim under `root` but the `kept` subtrees (and the prims leading to them)."""
  prim = view.GetPrimAtPath(root)
  if not prim:
    return
  for child in prim.GetChildren():
    path = str(child.GetPath())
    if any(under(path, k) for k in kept):
      continue
    if any(under(k, path) for k in kept):
      _keep_only(view, path, kept)
    else:
      child.SetActive(False)


def _private_view(stage: Usd.Stage) -> Usd.Stage:
  """Preserve composition state without first loading excluded payloads or muted assets."""
  from pxr import Sdf, Usd  # noqa: PLC0415

  session = Sdf.Layer.CreateAnonymous("raywarp_isaac_session.usda")
  session.TransferContent(stage.GetSessionLayer())
  view = Usd.Stage.OpenMasked(
    stage.GetRootLayer(), session, stage.GetPathResolverContext(), Usd.StagePopulationMask(), Usd.Stage.LoadNone
  )
  view.MuteAndUnmuteLayers(stage.GetMutedLayers(), [])
  view.SetLoadRules(stage.GetLoadRules())
  view.SetPopulationMask(stage.GetPopulationMask())
  view.SetEditTarget(session)
  return view


def _clone_frames(
  view: Usd.Stage,
  stage: Usd.Stage,
  roots: tuple[str, ...],
  origins: np.ndarray[[int, 3], np.dtype[np.float32]] | None,
) -> None:
  """Place missing roots from metre/Z-up world origins, compensating each USD parent frame."""
  from pxr import Gf, UsdGeom  # noqa: PLC0415

  from raywarp.formats.usd.stage.reader import prim_transforms  # noqa: PLC0415

  prototype = prim_transforms(view, [roots[0]])[0]
  for world, root in enumerate(roots):
    if stage.GetPrimAtPath(root):
      continue  # Authored clone roots remain authoritative.
    frame = UsdGeom.Xform.Define(view, root)
    parent = prim_transforms(view, [str(frame.GetPrim().GetParent().GetPath())])[0]
    placed = prototype.copy()
    if origins is not None:
      placed[:3, 3] = origins[world]
    local = np.linalg.solve(parent, placed)
    frame.MakeMatrixXform().Set(Gf.Matrix4d(*local.T.ravel().tolist()))


# The first Kit whose RTX renderer reads UsdLux lights by the spec (`omni:rtx:usdluxVersion` 2505).
# Earlier Kit (Isaac Sim 5.x: Kit 107) reads lights without that attribute in its legacy
# convention, and Isaac Lab spawns its lights without it.
SPEC_KIT_MAJOR: Final = 109


def kit_lux_dialect(kit_version: str | None) -> Dialect | None:
  """The UsdLux dialect of a live Kit stage's lights from the running Kit's version ("107.3.1+..."):
  RTX legacy before Kit 109, the spec from it; None without Kit (kit-less Isaac Lab: the stage's
  own markers decide, `usd.Options.lux_dialect`)."""
  if kit_version is None:
    return None
  major = int(kit_version.split(".", 1)[0])
  return Dialect.RTX_LEGACY if major < SPEC_KIT_MAJOR else Dialect.SPEC


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UsdSource:
  """Prepared stage view and explicit clone addresses (all host-side preparation data).

  `roots[0]` is the selected prototype, without any required environment naming convention.
  Worlds of different prototypes (Isaac Lab's multi-asset spawning: a different object per
  environment) each show their prototype's content, read once from its first world's root (its
  home, `homes`), placed per world, and hidden in the worlds of the other prototypes (a per-world
  visibility: run-time data). The original stage owns source layers; only this view's session
  layer is modified.

  Attributes:
    homes: per world, the root index whose prototype content it shows (homes[h] == h).
    local: per home, its prototype content's paths (default its whole root); the rest of world
      0's root is shared by every world.
  """

  stage: Usd.Stage
  roots: tuple[str, ...]
  body_paths: tuple[str, ...]
  homes: tuple[int, ...]
  local: dict[int, tuple[str, ...]]
  omitted_environments: tuple[str, ...] = ()
  lux_dialect: Dialect | None = None

  @classmethod
  def create(  # noqa: PLR0913  # the stage, the model, the roots and keyword-only clone data
    cls,
    stage: Usd.Stage,
    model: newton.Model,
    roots: Sequence[str],
    *,
    homes: Sequence[int] | None = None,
    assets: CloneAssets | None = None,
    excluded_sources: Sequence[str] = (),
    origins: np.ndarray[[int, 3], np.dtype[np.float32]] | None = None,
    lux_dialect: Dialect | None = None,
  ) -> UsdSource:
    """Hide duplicate clones/source-only assets in a private USD session layer.

    `homes`: per world, the root whose prototype content it shows (`world_homes`; default: world 0's
    for every world); `assets`: the clone plan's assets referenced into the view (`clone_assets`;
    without, a home's whole root is its prototype content). `lux_dialect`: the UsdLux dialect of the
    stage's lights without a version attribute (`kit_lux_dialect`)."""
    nworld = max(model.world_count, 1)
    root_paths = tuple(root.rstrip("/") or "/" for root in roots)
    if len(root_paths) != nworld or any(not root.startswith("/") for root in root_paths):
      raise ValueError("Isaac USD roots must contain one absolute prim path per world")
    if origins is not None and origins.shape != (nworld, 3):
      raise ValueError("Isaac clone origins must contain one three-dimensional position per world")
    world_home = (0,) * nworld if homes is None else tuple(homes)
    if len(world_home) != nworld or any(not 0 <= h < nworld or world_home[h] != h for h in world_home):
      raise ValueError("Isaac USD homes must give each world a root index that shows its own root (homes[h] == h)")
    home_set = sorted(set(world_home))
    local: dict[int, tuple[str, ...]]
    if assets is None:
      local = {h: (root_paths[h],) for h in home_set}
    else:
      local = {h: tuple(target for _, target in assets.local.get(h, ())) for h in home_set}
    view = _private_view(stage)
    pairs = () if assets is None else (*assets.shared, *(a for h in home_set for a in assets.local.get(h, ())))
    _reference_assets(view, stage, pairs)
    _clone_frames(view, stage, root_paths, origins)
    omitted_environments = cls._omitted_environments(view, [root_paths[h] for h in home_set], nworld)
    shown = [root_paths[h] for h in home_set]
    for path in (*(root for w, root in enumerate(root_paths) if w not in home_set), *excluded_sources):
      if any(under(root, path) for root in shown):
        raise ValueError("cannot exclude an ancestor of a selected Isaac prototype")
      if view.GetPrimAtPath(path):
        view.OverridePrim(path).SetActive(False)
    for h in home_set:
      if h and assets is not None:
        _keep_only(view, root_paths[h], local[h])  # the rest of a home's root repeats world 0's
    worlds = host_copy(model.body_world) if model.body_world is not None else np.full(model.body_count, -1, np.int32)
    aliases = tuple(
      _alias(label, int(world), root_paths, world_home, local) for label, world in zip(model.body_label, worlds, strict=True)
    )
    return cls(
      stage=view,
      roots=root_paths,
      body_paths=aliases,
      homes=world_home,
      local=local,
      omitted_environments=omitted_environments,
      lux_dialect=lux_dialect,
    )

  @staticmethod
  def _omitted_environments(view: Usd.Stage, roots: Sequence[str], nworld: int) -> tuple[str, ...]:
    """Deactivates the dome lights under the shown roots of a stage of several worlds (an
    environment per world is not placed: reported by `visibility`); their paths."""
    from pxr import UsdLux  # noqa: PLC0415  # USD is optional for model-only adapter tests.

    if nworld < 2:
      return ()
    paths = tuple(
      str(prim.GetPath())
      for prim in view.Traverse()
      if prim.IsA(UsdLux.DomeLight) and any(under(str(prim.GetPath()), root) for root in roots)
    )
    for path in paths:
      view.OverridePrim(path).SetActive(False)
    return paths

  def home_of(self, path: str) -> int:
    """The home whose prototype content `path` is (-1: shared by every world)."""
    return next((h for h, kept in self.local.items() if any(under(path, k) for k in kept)), -1)

  def shows(self, path: str, world: int) -> bool:
    """Whether world `world` shows the prim at `path`."""
    home = self.home_of(path)
    return home < 0 or self.homes[world] == home

  def _source_root(self, path: str) -> str:
    """The root `path` is read under (its home's, else world 0's) for rebasing into a world."""
    return next((self.roots[h] for h in sorted(self.local) if under(path, self.roots[h])), self.roots[0])

  @staticmethod
  def _paths(host: Scene) -> tuple[str, ...]:
    """One prototype's source paths, without expanding all cloned world strings."""
    return tuple(host.instances.paths[int(p)] if p >= 0 else "" for p in host.instances.prim)

  def instance_paths(self, host: Scene) -> tuple[tuple[str, ...], ...]:
    """USD visual prim paths addressed in each destination world (diagnostics)."""
    paths = self._paths(host)
    sources = [self._source_root(path) for path in paths]
    return tuple(tuple(rebase(path, src, root) for path, src in zip(paths, sources, strict=True)) for root in self.roots)

  def material_paths(self, host: Scene) -> tuple[tuple[str, ...], ...]:
    """Bound material paths per visual and world, including visuals with no Newton shape."""
    names = [host.materials.names[int(m)] if m >= 0 else "" for m in host.instances.material]
    prototypes = [str(prim.GetPath()) for prim in self.stage.GetPrototypes()]
    instance_paths = self._paths(host)
    for i, name in enumerate(names):
      prototype = next((root for root in prototypes if under(name, root)), None)
      if prototype is None:
        continue
      prim = self.stage.GetPrimAtPath(instance_paths[i])
      while prim and not (prim.IsInstance() and str(prim.GetPrototype().GetPath()) == prototype):
        prim = prim.GetParent()
      if prim:
        names[i] = rebase(name, prototype, str(prim.GetPath()))
      else:
        names[i] = ""
        host.report.drop("isaac.material.prototype_provenance(unsupported)", instance_paths[i])
    sources = {name: self._source_root(name) for name in set(names)}
    # A world binds no material to the visuals it does not show (another prototype's).
    homes = [self.home_of(path) for path in instance_paths]
    bindings: list[tuple[str, ...]] = []
    for w, root in enumerate(self.roots):
      # Keep one string per material/world, not one new string per visual/world.
      addresses = {name: rebase(name, src, root) for name, src in sources.items()}
      shown = [home < 0 or self.homes[w] == home for home in homes]
      bindings.append(tuple(addresses[name] if show else "" for name, show in zip(names, shown, strict=True)))
    return tuple(bindings)

  def shape_rows(self, host: Scene, model: newton.Model) -> IntTable:
    """Exact Newton shape matches; USD-only visuals, and visuals a world does not show, remain -1
    (never the last model shape)."""
    lookup = {path: i for i, path in enumerate(model.shape_label)}
    paths = self._paths(host)
    homes = [self.home_of(path) for path in paths]  # once per visual, not per world
    sources = [self._source_root(path) for path in paths]
    return np.asarray(
      [
        [
          lookup.get(rebase(path, src, root), -1) if home < 0 or self.homes[w] == home else -1
          for path, src, home in zip(paths, sources, homes, strict=True)
        ]
        for w, root in enumerate(self.roots)
      ],
      np.int32,
    )

  def visibility(self, host: Scene) -> np.ndarray[[int, int], np.dtype[np.bool_]]:
    """(nworld, ninstance) whether each world shows each instance: a prototype's content only in
    its worlds, shared content everywhere."""
    for path in self.omitted_environments:
      host.report.drop("isaac.environment.local_placement(unsupported)", path)
    home = np.asarray([self.home_of(path) for path in self._paths(host)], np.int64)
    return (home[None] < 0) | (np.asarray(self.homes, np.int64)[:, None] == home[None])

  def place_lights(self, host: Scene, gpu: GpuScene[NewtonState], model: newton.Model) -> None:
    """Initialize scoped authored/panel light tables once; preserve upload's photometry."""
    from raywarp.integrations.newton import from_usd  # noqa: PLC0415
    from raywarp.scene.model import xform  # noqa: PLC0415
    from raywarp.scene.model.types import LightKind  # noqa: PLC0415

    options = {"body_paths": self.body_paths, "world_local_roots": self.roots, "world_homes": self.homes}
    poses, active = from_usd.static_light_poses(model, host, self.stage, **options)
    tables = from_usd.usd_tables(model, host, host.ninstance, self.stage, **options)
    authored = np.flatnonzero(~np.isin(host.lights.kind, [int(LightKind.CAMERA), int(LightKind.AMBIENT)]))
    nworld = len(self.roots)
    vectors = {key: np.repeat(host_copy(gpu.tables[key]), nworld, axis=0) for key in ("light.pos", "light.dir", "light.axis")}
    enabled = np.repeat(host_copy(gpu.tables["light.active"]), nworld, axis=0)
    for column in range(gpu.source_lights):
      if column < len(authored):
        light = int(authored[column])
        initial = host.lights.xform[light].astype(np.float64)
        path = host.lights.paths[light]
        placed = poses[:, light]
        enabled[:, column] &= active[light]
      elif gpu.light_instance[column] >= 0:
        instance = int(gpu.light_instance[column])
        initial = host.instances.xform[instance].astype(np.float64)
        path = host.instances.paths[int(host.instances.prim[instance])]
        placed = tables.local[:, instance]
        if np.any(tables.body[:, instance] >= 0) or host.instances.source[instance] >= 0:
          enabled[:, column] = False
          host.report.drop("isaac.emissive_light.live_binding(unsupported)", path)
      else:
        continue  # an emitter standing for no instance: nothing moves it
      old_rotation = xform.pose_matrix(xform.rigid(initial[None])[0])[:, :3]
      for world in range(nworld):
        frame = xform.pose_matrix(placed[world].astype(np.float64))
        rotation = frame[:, :3] @ old_rotation.T
        vectors["light.pos"][world, column] = rotation @ (vectors["light.pos"][world, column] - initial[:, 3]) + frame[:, 3]
        for key in ("light.dir", "light.axis"):
          vectors[key][world, column] = rotation @ vectors[key][world, column]
        if not self.shows(path, world):
          enabled[world, column] = False
    gpu.tables["light.pos"] = wp.array(np.ascontiguousarray(vectors["light.pos"]), dtype=wp.vec3, device=gpu.device)
    gpu.tables["light.dir"] = wp.array(np.ascontiguousarray(vectors["light.dir"]), dtype=wp.vec3, device=gpu.device)
    gpu.tables["light.axis"] = wp.array(np.ascontiguousarray(vectors["light.axis"]), dtype=wp.vec3, device=gpu.device)
    gpu.tables["light.active"] = wp.array(np.ascontiguousarray(enabled), dtype=bool, device=gpu.device)
