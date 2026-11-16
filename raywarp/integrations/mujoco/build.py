"""MuJoCo model -> GpuScene (host data from `mjm`, device tables aliased from `m`).

Steps: `capabilities.content` (what the rendered geoms use, over every batched row of `m`), the
static assets (meshes / height fields `meshes.py`, textures `textures.py`, environment
`environment.py`, flex `flex.py`), the tables MuJoCo Warp does not hold (`_tables`: per-material
metalness / roughness, mirror, alpha cutoff derived from the canonical OpenPBR materials
`materials.openpbr_materials`, light strength / range / radius / shape) and the binding aliasing
the rest (`binding.py`).
"""

from __future__ import annotations

import dataclasses
from typing import Final, final

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp._base.dims import Mask
from raywarp._base.types import NTEXROLE, OPTICS_ROWS, TEX_IDENTITY, LightShape, ShapeType, tex_affine
from raywarp.core.pipeline.derive.fields import DeriveInputs
from raywarp.core.pipeline.derive.materials import DerivedMaterials, derive
from raywarp.core.pipeline.gpu_scene.binding import InstancePoses, Slots
from raywarp.core.pipeline.gpu_scene.deformables import Deformables
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, Capabilities, EnvSuns, GpuScene, Meshes, MeshInstances
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.textures.formats import TextureStorage
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.integrations.mujoco import flex as mj_flex
from raywarp.integrations.mujoco import lights as mj_lights
from raywarp.integrations.mujoco import materials as mj_materials
from raywarp.integrations.mujoco import meshes as mj_meshes
from raywarp.integrations.mujoco.binding import BATCHED_NDIM, MjwBinding, dynamic_geoms, moving_lights
from raywarp.integrations.mujoco.capabilities import Content, content
from raywarp.integrations.mujoco.environment import environment
from raywarp.integrations.mujoco.hfield import HeightFields, height_fields
from raywarp.integrations.mujoco.textures import TexturePack, upload_textures
from raywarp.scene.lighting.rig import Lighting
from raywarp.scene.model.material import Material
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import MaterialExtras, Materials
from raywarp.scene.model.types import AlphaMode, Column


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Options:
  """The `scene` arguments the steps read."""

  lighting: Lighting
  use_skybox: bool
  unset_metallic: float | None
  unset_roughness: float | None
  textures: TextureStorage
  env_sun: bool
  data: mjw.Data | None
  flexes: Column[np.bool_]


def _derive(mats: tuple[Material, ...], report: LoadReport) -> tuple[DerivedMaterials, Column[np.float32], DeriveInputs]:
  """`derive` of the canonical MuJoCo materials `mats`, their (nmat,) alpha cutoffs (0: none):
  `ext.alpha_cutoff` itself, the device's test of the opacity layer's texel (`materials.py`), and
  the derived per-world fields' inputs (the emission scales the base color: peak color 1)."""
  table = Materials.from_materials(mats)
  # No volume: the MuJoCo build has no optics rows (`instance_optics` zeros): a transmissive
  # material tints at its surface.
  derived = derive(table, MaterialExtras.from_materials(mats), report, volume=np.zeros(len(mats), bool))
  mask = table.ext_alpha_mode == int(AlphaMode.MASK)
  return (
    derived,
    np.where(mask, table.ext_alpha_cutoff, 0.0).astype(np.float32),
    DeriveInputs.of(table, derived, None),
  )


def _cameras(mjm: mujoco.MjModel) -> dict[str, CameraSpec]:
  """The model's cameras in id order, each mounted on itself (`camera.pos` / `camera.rot` from `d`).

  An unnamed camera i is "camera<i>", suffixed with "#" until no other camera has that name:
  one entry per camera, so `GpuScene.camera_names[i]` is camera i.
  """
  names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(mjm.ncam)]
  taken = set(names)
  cams = {}
  for i in range(mjm.ncam):
    name = names[i]
    if not name:
      name = f"camera{i}"
      while name in taken:
        name += "#"
      taken.add(name)
    ortho = int(mjm.cam_projection[i]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
    fx, fy, cx, cy = (float(v) for v in mjm.cam_intrinsic[i])
    sw, sh = (float(v) for v in mjm.cam_sensorsize[i])
    cams[name] = CameraSpec(
      fovy=float(mjm.cam_fovy[i]),
      intrinsics=(fx, fy, cx, cy),
      sensor_size=(sw, sh),
      projection="orthographic" if ortho else "perspective",
      mount=name,
    )
  return cams


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
  """MuJoCo adapter options (`rw.integrations.mujoco.scene`); the options every adapter shares are
  `rw.SceneOptions`.

  Attributes:
    geom_groups: geom groups to render (geoms and flexes).
    use_skybox: without an envmap or an image light, fill the background with the model's skybox
      texture, if it has one (MuJoCo's skybox is the background only: it lights nothing;
      `<light type="image">` lights the scene, `raywarp.integrations.mujoco.environment`).
    unset_metallic: metallic of materials that leave it unset (-1; default 0, glTF), in [0, 1].
    unset_roughness: roughness of materials that leave it unset (default 1 - shininess), in [0, 1].

  Raises:
    ValueError: for a geom group that is not an integer in [0, mjNGROUP) (a group no geom can be
      in would render nothing of it, silently), or an unset metallic / roughness outside [0, 1].
  """

  geom_groups: tuple[int, ...] = (0, 1, 2)
  use_skybox: bool = True
  unset_metallic: float | None = None
  unset_roughness: float | None = None

  def __post_init__(self) -> None:
    """Checks the values (at construction); stores the geom groups as a tuple."""
    given: object = self.geom_groups
    groups: tuple[object, ...] = (given,) if isinstance(given, str | int | np.integer) else tuple(self.geom_groups)
    ints = [int(np.asarray(g)) for g in groups if isinstance(g, int | np.integer) and not isinstance(g, bool | np.bool_)]
    if len(ints) != len(groups) or any(not 0 <= g < mujoco.mjNGROUP for g in ints):
      raise ValueError(f"mujoco.Options.geom_groups must be integers in [0, {mujoco.mjNGROUP}), got {self.geom_groups!r}")
    object.__setattr__(self, "geom_groups", tuple(ints))
    for name, value in (("unset_metallic", self.unset_metallic), ("unset_roughness", self.unset_roughness)):
      if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"mujoco.Options.{name} must be in [0, 1], got {value}")


# The default options (a shared immutable value: argument defaults).
DEFAULT_OPTIONS: Final = Options()


def scene(  # noqa: PLR0913  # the model, its device copy, its data and keyword-only options
  mjm: mujoco.MjModel,
  m: mjw.Model,
  d: mjw.Data | None = None,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
) -> GpuScene[mjw.Data]:
  """GpuScene of a MuJoCo model, on the device of `m` (the current Warp device by default).

  Instances are the geoms (instance i is geom i, named by the geom name or "geom<i>"; seg
  writes the geom id, flex f writes ngeom + f). `adapter.geom_groups` selects the geoms and
  flexes shown (`geom_group`, `flex_group`). Static geoms (on the world body, welded to it or
  on a mocap body posed the same in every world of `d`, not posed per world by batched mjw fields)
  are shared by every world unless a Renderer's `PerWorld` declares them
  (`raywarp.integrations.mujoco.binding`). The geom, material, light and camera tables of `m` are
  aliased with no copy and re-read on every render, so replacing e.g. `m.geom_rgba`, `m.geom_matid`
  or `m.mat_specular` takes effect (a Renderer's `per_world` overrides win over them); poses come
  from the `mjw.Data` passed to `Renderer.render(d)`. What MuJoCo Warp does not hold (material
  metallic / roughness / reflectance-as-mirror, light intensity / range / bulb radius) is read from
  `mjm` once. Lights follow MuJoCo's photometric or classic model
  (`raywarp.integrations.mujoco.lights`).

  Args:
    mjm: the host model (meshes, textures, materials, lights are read from it).
    m: the device model (`mujoco_warp.put_model(mjm)`).
    d: the device data the scene will render (its mocap poses decide which mocap bodies are
      shared scenery: those posed the same in every world). Without it every mocap body is
      posed by `m` (the same in every world unless a batched body_pos / body_quat differs). A
      shared mocap body later moved per world raises at the next render: declare it with
      `Renderer(per_world=rw.PerWorld(pose=[...]))`.
    options: the options every adapter shares (`rw.SceneOptions`): `textures` (textures bound
      as normal maps use its `normal` format), `env_sun` (the lighting maps' suns become
      shadowed directional lights, shaded after the model's lights:
      `raywarp.integrations.mujoco.environment`). MuJoCo materials are not fitted as panel lights
      (MuJoCo's emission lights nothing: `emissive_lights=True` raises).
    adapter: the MuJoCo options (`raywarp.integrations.mujoco.Options`).
    lighting: ambient, environment maps and headlight added to the model's (`rw.Lighting`;
      `lighting.envmap` replaces the model's image light; a list of maps is selected per world
      with `Renderer.per_world["env.id"]`; `lighting.headlight` replaces `visual/headlight`).
      Its `ev100` is the exposure the conversion of MuJoCo's display-relative quantities is
      calibrated for (classic lights, the headlight ambient, the default envmap intensity; see
      `raywarp.integrations.mujoco.lights`): render with the same `exposure.ev100` (both default to
      14.97) for MuJoCo-like brightness. Extra lights are added to the model itself before compiling
      (`rw.integrations.mujoco.add_lights(spec, lighting)`): the light tables are MuJoCo Warp's,
      aliased.

  Raises:
    ValueError: for a model with nothing to render in the enabled groups, extra lights in
      `lighting` or `options.emissive_lights=True`.
  """
  if options.emissive_lights:
    raise ValueError(
      "rw.integrations.mujoco.scene: MuJoCo emission lights nothing: SceneOptions.emissive_lights=True is not supported"
    )
  rendered = np.isin(mjm.geom_group, list(adapter.geom_groups))
  flexes = mj_flex.shown(mjm, adapter.geom_groups)
  if not np.any(rendered) and not np.any(flexes):
    raise ValueError("the model has no geoms or flexes in the enabled geom groups to render")
  lighting = Lighting() if lighting is None else lighting
  if lighting.lights:
    raise ValueError(
      "rw.integrations.mujoco.scene: MuJoCo lights come from the model: add them to the MjSpec before compiling"
      " (lighting = rw.integrations.mujoco.add_lights(spec, lighting))"
    )
  opts = _Options(
    lighting=lighting,
    use_skybox=adapter.use_skybox,
    unset_metallic=adapter.unset_metallic,
    unset_roughness=adapter.unset_roughness,
    textures=options.textures,
    env_sun=options.env_sun,
    data=d,
    flexes=flexes,
  )
  return _scene(mjm, m, rendered, opts)


def _capabilities(  # noqa: PLR0913  # the scan's parts, keyword-only
  mjm: mujoco.MjModel,
  *,
  used: Content,
  meshes: Meshes,
  flex: Deformables,
  lit: mj_lights.LightTables,
  suns: EnvSuns,
  mirror: Column[np.float32],
  cutoff: Column[np.float32],
  rendered: Column[np.bool_],
) -> Capabilities:
  """What the rendered content uses (compile-time features; `capabilities.content`); the
  environment `suns` are shadowed directional lights with a disc."""
  types = {int(t) for t in mjm.geom_type[rendered]}
  if flex.nprim:
    types.add(int(ShapeType.FLEX))
  cut = cutoff[used.materials] > 0.0
  return Capabilities(
    shape_types=tuple(sorted(types)),
    textures=used.slots,
    has_alpha=used.has_alpha,
    alpha_map=used.alpha_map,
    has_mirror=bool(np.any(mirror[used.materials] > 0.0)),
    has_spot=used.has_spot,
    has_area_lights=bool(np.any(lit.shape != int(LightShape.NONE))),
    mesh_frames=meshes.frames is not None,
    alpha_cutout=bool(np.any(cut & used.opacity)),
    sun_disc=bool(np.any(suns.angle > 0.0)),
  )


def _tables(
  lit: mj_lights.LightTables, pack: TexturePack, derived: DerivedMaterials, cutoff: Column[np.float32], device: wp.Device
) -> Slots:
  """The (1, n) tables on `device` MuJoCo Warp does not hold (shared by every world until
  overridden); `cutoff`: the materials' alpha cutoffs."""

  def row(values: Column[np.float32]) -> wp.array[float, [int, int]]:
    """A (1, n) batched float table of `values`."""
    return wp.array(values.reshape(1, -1), dtype=float, device=device)

  nmat = len(derived.mirror)
  tables: Slots = {
    "material.base_metalness": row(derived.base_metalness),
    "material.specular_roughness": row(derived.specular_roughness),
    "material.mirror": row(derived.mirror),
    "light.strength": row(lit.intensity),
    "light.range": row(lit.range),
    "light.radius": row(lit.radius),
    "light.shape": wp.array(lit.shape.reshape(1, -1), dtype=int, device=device),
    # MuJoCo materials have no texel factors: identity texel maps (per-world overridable).
    "material.tex_affine": wp.array(np.tile(TEX_IDENTITY, (1, nmat, NTEXROLE, 1, 1)), dtype=tex_affine, device=device),
    "material.tex_variation": wp.zeros((1, nmat), dtype=wp.vec4, device=device),
  }
  if lit.cone_inner is not None:
    tables["light.cone_inner"] = row(lit.cone_inner)
  if np.any(cutoff > 0.0):
    tables["material.alpha_cutoff"] = row(cutoff)
  if pack.material_texture is not None:
    tables["material.texture"] = wp.array(pack.material_texture, dtype=int, device=device)
  return tables


def _ambient(mjm: mujoco.MjModel, lighting: Lighting) -> tuple[float, float, float]:
  """The model's ambient (its headlight's unless `lighting` replaces it; MuJoCo's no-light
  fallback unless `lighting` adds an environment or an ambient: then the model is lit, and the
  fallback's flat 0.3 would wash it out) plus `lighting`'s."""
  own = lighting.headlight is None and not lighting.replace
  unlit = not lighting.envmaps and lighting.ambient is None
  r0, g0, b0 = (
    (0.0, 0.0, 0.0) if lighting.replace else mj_lights.ambient(mjm, lighting.exposure, with_headlight=own, fallback=unlit)
  )
  r1, g1, b1 = lighting.ambient_radiance()
  return r0 + r1, g0 + g1, b0 + b1


def used_meshes(mjm: mujoco.MjModel, m: mjw.Model, rendered: Column[np.bool_]) -> Mask[int]:
  """(nmesh,) the meshes a rendered mesh geom shows in some batched row of `m.geom_dataid` (the
  others, e.g. collision-only meshes, get no BLAS: `Meshes.used`; a later `m.geom_dataid` write
  onto one is reported by the validation, `capabilities.LiveCapabilityCheck`)."""
  used = np.zeros(mjm.nmesh, bool)
  if not mjm.nmesh or not mjm.ngeom:
    return used
  dataid = host_copy(m.geom_dataid).reshape(-1, mjm.ngeom)
  # SDF geoms are traced as their mesh (`_base.types.MESH_LIKE`).
  mesh = rendered & np.isin(mjm.geom_type, (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF)))
  ids = dataid[:, mesh]
  used[ids[(ids >= 0) & (ids < mjm.nmesh)]] = True
  return used


def _binding(  # noqa: PLR0913  # the model, its rendered geoms and keyword-only scene parts
  mjm: mujoco.MjModel,
  m: mjw.Model,
  rendered: Column[np.bool_],
  d: mjw.Data | None,
  *,
  hfields: HeightFields,
  meshes: Meshes,
  scanned: Content,
  pack: TexturePack,
) -> MjwBinding:
  """The binding of the `rendered` geoms of `m` (`d`: the data deciding the shared mocap bodies)."""

  def geoms(kind: mujoco.mjtGeom) -> Column[np.int32]:
    """The rendered geoms of type `kind`."""
    return np.flatnonzero(rendered & (mjm.geom_type == kind)).astype(np.int32)

  return MjwBinding(
    m,
    moved=dynamic_geoms(mjm, m, d),
    moving=moving_lights(mjm, m, d),
    hfields=hfields,
    rendered=np.flatnonzero(rendered).astype(np.int32),
    mesh_geoms=geoms(mujoco.mjtGeom.mjGEOM_MESH),
    hfield_geoms=geoms(mujoco.mjtGeom.mjGEOM_HFIELD),
    meshes=meshes,
    scanned=scanned,
    classic=not mj_lights.photometric(mjm),
    own=frozenset({"material.texture"}) if pack.material_texture is not None else frozenset(),
  )


def _scene(mjm: mujoco.MjModel, m: mjw.Model, rendered: Column[np.bool_], opts: _Options) -> GpuScene[mjw.Data]:
  """The scene of `scene`, on the device of `m`."""
  device = m.geom_type.device
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  used = content(mjm, m, rendered)
  pack = upload_textures(mjm, host_copy(m.mat_texid), used.materials, storage=opts.textures, device=device)
  hfields = height_fields(mjm, m, mj_meshes.used_hfields(mjm, m, rendered))
  meshes = mj_meshes.meshes(mjm, m, hfields, device=device)
  meshes.used = used_meshes(mjm, m, rendered)
  flex = mj_flex.deformables(mjm, mjd, opts.flexes, device)
  lit = mj_lights.light_tables(mjm, opts.lighting.exposure, off=opts.lighting.replace)
  extent, ngeom = float(mjm.stat.extent), mjm.ngeom
  geom_ids = np.flatnonzero(rendered).astype(np.int32)
  report = LoadReport(source="mujoco")
  binding = _binding(mjm, m, rendered, opts.data, hfields=hfields, meshes=meshes, scanned=used, pack=pack)
  mats = mj_materials.openpbr_materials(mjm, report, unset_metallic=opts.unset_metallic, unset_roughness=opts.unset_roughness)
  derived, cutoff, inputs = _derive(mats, report)
  inputs = mj_materials.shininess_inputs(mjm, inputs, unset_roughness=opts.unset_roughness)
  env = environment(
    mjm,
    opts.lighting,
    report,
    use_skybox=opts.use_skybox,
    env_sun=opts.env_sun,
    ambient=_ambient(mjm, opts.lighting),
    device=device,
  )
  nlight = mjm.nlight + len(env.suns)
  return GpuScene(
    device=device,
    ninstance=ngeom,
    nmaterial=mjm.nmat,
    nlight=nlight,
    tables=_tables(lit, pack, derived, cutoff, device),
    derive_inputs=inputs,
    instance_type=m.geom_type,
    instance_names=[mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}" for i in range(ngeom)],
    rendered=geom_ids,
    init_pose=InstancePoses(
      np.asarray(mjd.geom_xpos, np.float32).reshape(-1, 3), np.asarray(mjd.geom_xmat, np.float32).reshape(-1, 3, 3)
    ),
    bound_radius=_rbound(m),
    instance_shadow_transmit=np.zeros((ngeom, 3), np.float32),
    instance_optics=np.zeros((ngeom, OPTICS_ROWS, 4), np.float32),
    instance_twosided=np.zeros(ngeom, np.int32),
    optics_inputs=None,
    light_instance=np.full(mjm.nlight, -1, np.int32),
    light_visible=np.zeros((0, 2), np.int32),
    meshes=meshes,
    mesh_instances=MeshInstances.plain(ngeom, device),
    textures=pack.textures,
    texture_array=wp.array(pack.textures, dtype=wp.Texture2D, device=device),
    tex_size=wp.array(pack.sizes, dtype=wp.vec2, device=device),
    tex_layout=wp.array(pack.layouts, dtype=int, device=device),
    environment=env,
    headlight=mj_lights.headlight(mjm, opts.lighting.headlight, off=opts.lighting.replace),
    cameras=_cameras(mjm),
    clip=(float(mjm.vis.map.znear * extent), float(mjm.vis.map.zfar * extent)),
    capabilities=_capabilities(
      mjm,
      used=used,
      meshes=meshes,
      flex=flex,
      lit=lit,
      suns=env.suns,
      mirror=derived.mirror,
      cutoff=cutoff,
      rendered=rendered,
    ),
    binding=binding,
    color_rule=binding.color_rule,
    deformables=flex,
    segment_id=None,  # binding.instance_segments: the geom id
    report=report,
    flex_segment=ngeom,
  )


def _rbound(m: mjw.Model) -> Column[np.float32]:
  """(ngeom,) bounding radius per geom (world 0's row of a batched `geom_rbound`)."""
  radius = host_copy(m.geom_rbound)
  return radius[0] if radius.ndim == BATCHED_NDIM else radius
