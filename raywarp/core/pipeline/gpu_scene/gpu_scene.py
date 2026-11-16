"""The scene as it lives on the GPU: one record of the arrays the kernels read (geometry,
materials, lights, textures, cameras), shared by any number of Renderers.

A `GpuScene` holds

* `tables`: named per-instance / per-material / per-light arrays (the slot names of `binding.SLOTS`,
  e.g. "instance.color"), batched as (1 or nworld, n) and read by the kernels at row
  `min(worldid, shape[0] - 1)` (`kernels/abi/tables.table_row`). A binding may alias some of them
  from a simulator per call (MuJoCo Warp arrays are aliased with no copy); a Renderer's per-world
  overrides win over both.
* instances, named (`instance_names`: what `PerWorld` patterns match). Which ones every world
  shares (static scenery: one pose, one entry in a shared BVH / OptiX IAS) is decided by each
  Renderer (`core/pipeline/worlds/sharing.py`) from the binding and the Renderer's `PerWorld`, not
  by the scene.
* static geometry: `Meshes` (triangle meshes, their BLASes, height fields), `Deformables`
  (optional, e.g. MuJoCo flex), textures, environment maps, cameras authored in the source.
* `capabilities`: what the content uses (shape kinds, texture slots, translucency, mirrors, spot
  lights, ...), which resolves the "auto" settings and specializes the kernels.
* `binding`: how per-step state (poses) reaches the kernels (`binding.Binding[S]`, S the simulator
  state a Renderer of the scene renders).

Contract: a `GpuScene` is a pure typed record. Every field is set and means the same for every
source: per-instance / per-light extras are always present (zeros or -1 if absent), so consumers
never branch on a missing field. Its only constructors are `rw.upload` (`core/pipeline/upload/`: a
host scene, `raywarp.scene.Scene` from `SceneBuilder`, `rw.formats.usd.scene` or a `.rayw` cache;
the Newton adapter passes its binding) and the MuJoCo adapter (`raywarp.integrations.mujoco.scene`,
aliasing MuJoCo Warp).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, final

import numpy as np
import warp as wp

from raywarp._base.records import record_field
from raywarp._base.types import UV_IDENTITY, UV_MESH, ColorRule, MaterialUvMapping, MeshAdr
from raywarp.core.kernels.abi.rays import MeshInstance
from raywarp.core.pipeline.gpu_scene.camera_checks import check_frame, check_lens
from raywarp.core.pipeline.gpu_scene.deformables import Deformables
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.shade.environment import envmap as envmap_lib
from raywarp.core.shade.lights.ies import IesLights
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_ENVIRONMENT, EnvironmentId, MeshId

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import Floats
  from raywarp.core.pipeline.derive.fields import DeriveInputs
  from raywarp.core.pipeline.gpu_scene.binding import Binding, InstancePoses, Slots
  from raywarp.core.pipeline.upload.optics import OpticsInputs
  from raywarp.settings.fields import OutputKind


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class CameraSpec:
  """A camera of a Renderer.

  Attributes:
    fovy: vertical field of view [degrees] (perspective), the view height [m] (orthographic), or
      for a fisheye without intrinsics the field of view the image height spans on an equidistant
      (f-theta) scale, up to 360 [degrees] (the distorted angle theta_d of the lens model).
    intrinsics: (fx, fy, cx, cy) with `sensor_size`: pinhole or fisheye intrinsics instead of fovy
      (same units as the sensor size; MuJoCo's convention).
    sensor_size: (width, height) of the sensor for `intrinsics`; with `conform="fit"` and no
      intrinsics, the window whose height `fovy` spans (perspective: its vertical field of view;
      orthographic: its view height), only its aspect read.
    conform: how a sensor (window) of another aspect than the image meets it: "crop" (MuJoCo's:
      the intrinsics' sensor cropped to the image's aspect; a fovy camera keeps fovy vertical) or
      "fit" (Hydra's CameraUtilFit, USD cameras: the whole window shows, extended along its
      shorter side to the image's aspect, its centre (aperture offsets) kept). The per-world
      "camera.fovy" defaults are the conformed values.
    rays: per-pixel camera-frame rays, (height, width, 2, 3) float (origin, direction) or a
      (height, width, 2) vec3 Warp array, at the Renderer's resolution, replacing the projection
      (measured lens models, LiDAR-like patterns). Either every camera of a Renderer has rays or
      none.
    projection: "perspective", "orthographic" or "fisheye" (OpenCV's `cv::fisheye`
      Kannala-Brandt model, `distortion`; an equidistant f-theta lens without it): fields wider
      than 180 degrees are exact, and the pixels outside the model's image (past its fold or
      beyond 180 degrees from the axis) are misses (depth / seg miss values, the background color
      along the axis). Fisheye, pinhole and orthographic cameras mix in one Renderer; a fisheye
      has no ray bundle.
    znear: near clip (None: the scene's default): camera rays start on the near plane, planar like
      `zfar` (along the ray for fisheye cameras and `rays` bundles): nearer surfaces are not seen
      in any output; depth still counts from the eye; shadow, reflection and GI rays are not
      clipped.
    zfar: far clip, planar depth (along the ray for fisheye cameras and bundles; 0: unbounded;
      None: the scene's default). A fisheye's depth output is always the distance along the ray
      (planar depth is undefined beyond 90 degrees).
    mount: name of a scene camera whose pose the camera follows (None: a free camera, posed by
      `Renderer.camera_pose`).
    local: (position (3,), rotation (3, 3)) offset from the mount (a rotation matrix: orthonormal,
      det +1), or None.
    resolution: (width, height) of this camera (None: the Renderer's resolution). Cameras of
      different resolutions render without the motion output and give
      per-camera outputs (`RenderOutputs.cameras`).
    exposure: exposure compensation [stops] over the world's exposure (USD camera `exposure`):
      the display image is 2^exposure brighter, the HDR output is unchanged.
    distortion: lens coefficients (None: none): OpenCV's distortion vector of a perspective
      camera (`from_opencv`), 4, 5, 8 or 12 of (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4):
      radial, tangential, rational (k4-k6: `CALIB_RATIONAL_MODEL`) and thin prism (s1-s4:
      `CALIB_THIN_PRISM_MODEL`) terms; the Kannala-Brandt (k1, k2, k3, k4) of a fisheye
      (`from_opencv_fisheye`). The defaults of the per-world "camera.distortion" field
      ((nworld, ncam) rows of 12: OpenCV's twelve, a fisheye's four, zero-padded), which
      randomizes them per world without recompiling.
    outputs: the Renderer's outputs this camera writes (None: all), e.g. ("depth",) for a
      depth-only wrist camera next to an RGB head camera. Whole groups are selected (every shaded
      output, depth, seg + semantic; position, uv, material, a geometric normal and motion are
      written for every camera); a camera without the shaded outputs is not shaded (a
      geometry-only cost). Its outputs left out keep their buffers' contents (zeros as allocated,
      or what a bound array holds): the buffers stay (nworld, ncam, H, W), no memory is saved.
      Checked against the Renderer's outputs (ValueError).
  """

  fovy: float = 60.0
  intrinsics: tuple[float, float, float, float] | None = None
  sensor_size: tuple[float, float] | None = None
  conform: Literal["crop", "fit"] = "crop"
  rays: Floats[[int, int, 2, 3]] | wp.array[wp.vec3, [int, int, 2]] | None = None
  projection: Literal["perspective", "orthographic", "fisheye"] = "perspective"
  znear: float | None = None
  zfar: float | None = None
  mount: str | None = None
  local: tuple[Floats[[3]], Floats[[3, 3]]] | None = None
  resolution: tuple[int, int] | None = None
  exposure: float = 0.0
  distortion: tuple[float, ...] | None = None
  outputs: tuple[OutputKind, ...] | None = None

  def __post_init__(self) -> None:
    """Validates every field, also through `dataclasses.replace` (ValueError naming the field:
    `core/pipeline/gpu_scene/camera_checks.py`)."""
    check_lens(self)
    check_frame(self)

  @staticmethod
  def from_opencv(  # noqa: PLR0913  # the class's fields, keyword-only
    k: Sequence[Sequence[float]] | Floats[[3, 3]],
    resolution: tuple[int, int],
    *,
    distortion: Sequence[float] | None = None,
    mount: str | None = None,
    local: tuple[Floats[[3]], Floats[[3, 3]]] | None = None,
    znear: float | None = None,
    zfar: float | None = None,
    exposure: float = 0.0,
  ) -> CameraSpec:
    """A perspective camera of an OpenCV calibration: the 3 x 3 matrix `k` [pixels] and the
    distortion coefficients (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]) of an image of
    `resolution` (width, height);
    `mount`, `local`, `znear`, `zfar`, `exposure` as in the class. OpenCV puts pixel centers at
    integers, the renderer at i + 0.5: the principal point (cx, cy) becomes the offsets
    (width / 2 - cx - 0.5, height / 2 - cy - 0.5) of MuJoCo's intrinsics on a `resolution` sensor.

    Raises:
      ValueError: for a matrix that is not finite 3 x 3 with zero skew and bottom row
        [0, 0, 1], or an invalid field (`__post_init__`).
    """
    intrinsics, sensor = _opencv_intrinsics(k, resolution)
    return CameraSpec(
      intrinsics=intrinsics,
      sensor_size=sensor,
      resolution=resolution,
      distortion=None if distortion is None else tuple(float(c) for c in distortion),
      mount=mount,
      local=local,
      znear=znear,
      zfar=zfar,
      exposure=exposure,
    )

  @staticmethod
  def from_opencv_fisheye(  # noqa: PLR0913  # the class's fields, keyword-only
    k: Sequence[Sequence[float]] | Floats[[3, 3]],
    resolution: tuple[int, int],
    d: Sequence[float] | None = None,
    *,
    mount: str | None = None,
    local: tuple[Floats[[3]], Floats[[3, 3]]] | None = None,
    znear: float | None = None,
    zfar: float | None = None,
    exposure: float = 0.0,
  ) -> CameraSpec:
    """A fisheye camera of an OpenCV `cv::fisheye` calibration: the 3 x 3 matrix `k` [pixels]
    and the Kannala-Brandt coefficients `d` = (k1, k2, k3, k4) (None: an equidistant f-theta
    lens) of an image of `resolution` (width, height); the pixel at angle theta from the axis
    lies at theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) times
    (fx, fy) from the principal point. The other fields and the principal point as `from_opencv`.

    Raises:
      ValueError: for a matrix that is not finite 3 x 3 with zero skew and bottom row
        [0, 0, 1], or an invalid field (`__post_init__`).
    """
    intrinsics, sensor = _opencv_intrinsics(k, resolution)
    return CameraSpec(
      projection="fisheye",
      intrinsics=intrinsics,
      sensor_size=sensor,
      resolution=resolution,
      distortion=None if d is None else tuple(float(c) for c in d),
      mount=mount,
      local=local,
      znear=znear,
      zfar=zfar,
      exposure=exposure,
    )


def _opencv_intrinsics(
  k: Sequence[Sequence[float]] | Floats[[3, 3]], resolution: tuple[int, int]
) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
  """(intrinsics, sensor_size) of OpenCV camera matrix `k` [pixels] for an image of `resolution`
  (`CameraSpec.from_opencv`).

  Raises:
    ValueError: for a matrix that is not finite 3 x 3 with zero skew and bottom row [0, 0, 1].
  """
  m = np.asarray(k, dtype=np.float64)
  if m.shape != (3, 3):
    raise ValueError(f"the OpenCV camera matrix must be 3 x 3, got {m.shape}")
  if not np.isfinite(m).all():
    raise ValueError("the OpenCV camera matrix must contain only finite entries")
  if m[0, 1] != 0 or m[1, 0] != 0 or not np.array_equal(m[2], [0, 0, 1]):
    raise ValueError("the OpenCV camera matrix needs zero skew and bottom row [0, 0, 1]")
  (w, h), (fx, fy, cx, cy) = resolution, (float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]))
  return (fx, fy, 0.5 * w - cx - 0.5, 0.5 * h - cy - 0.5), (float(w), float(h))


@final
@dataclasses.dataclass(frozen=True)
class TextureSlots:
  """Texture slots bound by at least one rendered material."""

  base_color: bool = False
  normal: bool = False
  orm: bool = False
  emissive: bool = False
  opacity: bool = False
  scaled_normal: bool = False  # a normal map's strength / offset (`Features.scaled_normal_map`)
  diffuse_roughness: bool = False
  layered: bool = False  # an anisotropy strength or fuzz map (TexRole.ANISOTROPY / FUZZ)
  coat_normal: bool = False  # a coat normal map (TexRole.COAT_NORMAL)
  detail_normal: bool = False  # a detail normal map (TexRole.DETAIL_NORMAL)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LayerTables:
  """The layered lobes' static host tables (`LayeredMaterials.material_layer`, `mesh_tangent`;
  uploaded by the Renderer only for Features.layered).

  Attributes:
    coat_darkening: (m,) OpenPBR coat darkening (0: none).
    wrap: (m,) cloth wrapped-diffuse width.
    wrap_tint: (m, 3) the cloth's subsurface tint.
    detail_uv: (m, 2, 3) the detail normal map's UV transform on the base UVs.
    flags: (m,) LayerFlag bits.
    mesh_tangent: (nnormal + 1, 4) per-vertex UV tangents (xyz, w handedness) addressed like the
      meshes' normals, or None (no anisotropic material).
  """

  coat_darkening: np.ndarray[[int], np.dtype[np.float32]]
  wrap: np.ndarray[[int], np.dtype[np.float32]]
  wrap_tint: np.ndarray[[int, 3], np.dtype[np.float32]]
  detail_uv: np.ndarray[[int, 2, 3], np.dtype[np.float32]]
  flags: np.ndarray[[int], np.dtype[np.int32]]
  mesh_tangent: np.ndarray[[int, 4], np.dtype[np.float32]] | None = None


@final
@dataclasses.dataclass(frozen=True, slots=True)
class LayerUse:
  """The layered lobes some material uses (`Features.layered`).

  Attributes:
    anisotropy: a nonzero `specular_roughness_anisotropy`.
    fuzz: `fuzz_weight` > 0.
    coat: an OpenPBR coat (its own IOR, color, darkening: LayerFlag.COAT_OPENPBR).
    coat_own_normal: a coated material whose coat does not share its mapped normal
      (LayerFlag.COAT_BASE_NORMAL unset).
  """

  anisotropy: bool = False
  fuzz: bool = False
  coat: bool = False
  coat_own_normal: bool = False

  def union(self, other: LayerUse) -> LayerUse:
    """The lobes either uses."""
    return LayerUse(
      anisotropy=self.anisotropy or other.anisotropy,
      fuzz=self.fuzz or other.fuzz,
      coat=self.coat or other.coat,
      coat_own_normal=self.coat_own_normal or other.coat_own_normal,
    )


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PreparedMaterialCapabilities:
  """What every prepared material uses (`Capabilities.prepared_materials`): read instead of the
  rendered materials' capabilities when "instance.material" is declared per world."""

  textures: TextureSlots
  world_uv: bool
  alpha_cutout: bool
  alpha_map: bool
  has_alpha: bool
  has_mirror: bool
  specular_curve: bool
  clearcoat: bool
  transmissive: bool
  vertex_colors: bool
  layers: LayerUse = LayerUse()
  unlit_emission: bool = False


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Capabilities:
  """What a scene's content uses (derived by the adapter; host scenes:
  `pipeline/upload/capabilities.py`).

  Attributes:
    shape_types: shape kinds of the rendered instances (ShapeType values; FLEX with deformables).
    textures: texture slots bound by rendered materials.
    prepared_materials: what every prepared material uses, read instead of the rendered ones'
      when "instance.material" is declared per world (None: no prepared superset).
    has_alpha: a rendered instance, material or deformable is translucent (0 < alpha < 1).
    alpha_map: a base color or opacity texture of a rendered material has texels with alpha < 1.
    has_mirror: a rendered material has mirror reflectance > 0 (`ext.mirror`, `mat_reflectance`).
    has_spot: the scene has spot lights.
    has_area_lights: a light has a `light.shape` (lighting.area_lights="ltc").
    visible_lights: an area light has no geometry of its own (camera rays see its surface:
      `GpuScene.light_visible`).
    mesh_frames: some mesh BLASes are built in a rotated frame (`Meshes.frames`).
    uv_maps: a rendered mesh instance reads another UV set or a UV transform (`MeshInstances`).
    world_uv: a rendered material projects its UVs from the world position (`material.uv_world`).
    physical_emission: material emission is radiance [nits] (host scenes), not display-relative
      (MuJoCo: emission 1 shows the base color at full brightness whatever the exposure).
    specular_curve: a rendered instance has ShadingFlag.SPECULAR_CURVE (Unreal specular).
    transmissive: a rendered instance has ShadingFlag.TRANSMISSIVE (`transmission_weight` > 0).
    light_panels: a rendered instance's emission is also a light (`light_instance`).
    alpha_cutout: a rendered material has a cutoff on an opacity map ("material.alpha_cutoff").
    vertex_colors: a rendered mesh has vertex colors (`Meshes.coloradr` >= 0).
    unlit_emission: a rendered instance is unlit or has display-relative emission
      (`ShadingFlag.UNLIT` / `DISPLAY_EMISSION`: its color / emission display-relative).
    diffuse_roughness: a rendered material has a base diffuse roughness > 0 (OpenPBR EON).
    clearcoat: a rendered instance has a coat (`instance.shading` coat bits: `coat_weight` > 0).
    sun_disc: a shadow-casting directional light has an angular size (`light.radius` > 0).
    has_ies: a light has an IES profile (`GpuScene.ies`, Features.ies).
    layers: the layered lobes some rendered material uses (`LayerUse`).
  """

  shape_types: tuple[int, ...]
  textures: TextureSlots = TextureSlots()
  prepared_materials: PreparedMaterialCapabilities | None = None
  has_alpha: bool = False
  alpha_map: bool = False
  has_mirror: bool = False
  has_spot: bool = False
  has_area_lights: bool = False
  visible_lights: bool = False
  mesh_frames: bool = False
  uv_maps: bool = False
  world_uv: bool = False
  physical_emission: bool = False
  specular_curve: bool = False
  transmissive: bool = False
  light_panels: bool = False
  alpha_cutout: bool = False
  vertex_colors: bool = False
  unlit_emission: bool = False
  diffuse_roughness: bool = False
  clearcoat: bool = False
  sun_disc: bool = False
  has_ies: bool = False
  layers: LayerUse = LayerUse()


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GpuHeadlight:
  """Camera-attached directional light (display-relative: it follows each world's exposure).

  Attributes:
    active: whether the scene authored it on (lighting.camera_light="auto").
    color: linear color.
    intensity: illuminance times exposure.
  """

  active: bool
  color: tuple[float, float, float] = (0.4, 0.4, 0.4)
  intensity: float = np.pi


@final
@dataclasses.dataclass(kw_only=True)
class Meshes[NMesh: IntVar, NVert: IntVar, NFace: IntVar]:
  """Triangle meshes shared by every world, and their BLASes.

  Device arrays are indexed per mesh with `vertadr` / `faceadr` / `normaladr` / `texcoord_adr`
  (-1: no UVs). `points` / `faces` are the host copies the BLASes are built from; `frames` the
  optional per-mesh rotation R of a re-framed BLAS (built over `R v`, see kernels/abi/rays.py),
  with `framed_points` its vertices. `used`: the meshes a rendered instance can show (None: all); an
  unused mesh (e.g. MuJoCo collision-only) gets an empty placeholder BLAS / GAS instead of its
  own. Dims: `NMesh` meshes, `NVert` vertices and `NFace` triangles of every mesh concatenated;
  normals, UVs and colors have their own counts, and `facetexcoord` has 2 x NFace rows with a
  second UV set. Not frozen: it caches its BLASes.
  """

  vertadr: wp.array[int, [NMesh]]
  faceadr: wp.array[int, [NMesh]]
  normaladr: wp.array[int, [NMesh]]
  vert: wp.array[wp.vec3, [NVert]]
  normal: wp.array[wp.vec3, [int]]
  face: wp.array[wp.vec3i, [NFace]]
  facenormal: wp.array[wp.vec3i, [NFace]]
  facetexcoord: wp.array[wp.vec3i, [int]]
  texcoord: wp.array[wp.vec2, [int]]
  texcoord_adr: wp.array[int, [NMesh]]
  # Vertex colors: per-mesh address into `color` (-1: none), per-face corner indices into it
  # (aligned with `face`; `face` itself without colors), linear RGBA; each at least one entry.
  coloradr: wp.array[int, [int]]
  facecolor: wp.array[wp.vec3i, [NFace]]
  color: wp.array[wp.vec4, [int]]
  points: list[np.ndarray[[int, 3], np.dtype[np.float32]]]
  faces: list[np.ndarray[[int, 3], np.dtype[np.int32]]]
  frames: np.ndarray[[NMesh, 3, 3], np.dtype[np.floating]] | None = None
  framed_points: dict[int, np.ndarray[[int, 3], np.dtype[np.floating]]] = dataclasses.field(default_factory=dict)
  used: np.ndarray[[NMesh], np.dtype[np.bool_]] | None = None
  hfield_bvh_id: wp.array[wp.uint64, [int]] | None = None
  hfield_bounds_size: wp.array[wp.vec3, [int]] | None = None
  # The height fields' wp.Mesh objects by height field (kept alive for the ids in `hfield_bvh_id`;
  # the OptiX backend builds its height-field GASes over their points and indices).
  hfield_registry: dict[int, wp.Mesh] = dataclasses.field(default_factory=dict)
  # Bumped by every terrain change (`MjwBinding.refit_hfields`): the OptiX backend refits its
  # height-field GASes when it differs from the one they were built at.
  hfield_revision: int = 0
  # Packed-node addresses of the height-field BLASes (`trace.accel.packed`; their owner repacks them
  # after a terrain refit), aligned with `hfield_bvh_id`.
  hfield_nodes: wp.array[wp.uint64, [int]] | None = None
  # BLASes by (constructor, framed, every mesh used): their ids and the wp.Mesh objects that keep
  # them alive. The `used` part keeps the tables built with placeholders alive after `use_all`
  # (a Renderer created before it still traces them, also after a constructor change).
  _bvhs: dict[tuple[str, bool, bool], tuple[wp.array[wp.uint64, [NMesh]], list[wp.Mesh]]] = dataclasses.field(
    default_factory=dict, init=False
  )
  _boxes: tuple[wp.array[wp.vec3, [NMesh]], wp.array[wp.vec3, [NMesh]]] | None = dataclasses.field(default=None, init=False)
  # The per-mesh addresses packed for the kernels (`MeshAdr`, one launch-parameter array).
  adr: wp.array[MeshAdr, [int]] = dataclasses.field(init=False)

  def __post_init__(self) -> None:
    """Packs the per-mesh address columns into `adr` (max(nmesh, 1) entries)."""
    n = len(self.vertadr)
    packed = wp.zeros(max(n, 1), dtype=MeshAdr, device="cpu").numpy()
    for name, column in (
      ("vert", self.vertadr),
      ("face", self.faceadr),
      ("normal", self.normaladr),
      ("color", self.coloradr),
      ("texcoord", self.texcoord_adr),
    ):
      values = np.asarray(host_copy(column), np.int32)[:n]
      record_field(packed, name)[: len(values)] = values
    self.adr = wp.array(packed, dtype=MeshAdr, device=self.vertadr.device)

  @property
  def nmesh(self) -> int:
    """Number of meshes."""
    return len(self.points)

  def used_now(self) -> np.ndarray[[NMesh], np.dtype[np.bool_]] | None:
    """A copy of `used` (None: every mesh), the meshes with a BLAS / GAS in the tables of a
    Renderer created now (`RendererTables.mesh_used`): a later `use_all` leaves it unchanged."""
    return None if self.used is None or self.used.all() else self.used.copy()

  def use_all(self) -> None:
    """Every mesh gets its own BLAS in the Renderers created from now on (e.g. one declaring
    per-world meshes, which can show any mesh); the Renderers created before keep their mask
    (`used_now`) and tables (`bvhs`)."""
    self.used = None

  def local_boxes(self) -> tuple[wp.array[wp.vec3, [NMesh]], wp.array[wp.vec3, [NMesh]]]:
    """((nmesh,) centres, (nmesh,) half extents) of the meshes' boxes in their own frame (no BLAS),
    cached: the scene BVH bounds an instance by this box, not by a box centred on the mesh origin
    (vertices far from their origin, as scenes authored in world space, would make it huge)."""
    if self._boxes is None:
      lo = np.array([p.min(axis=0) if len(p) else np.zeros(3) for p in self.points], np.float64).reshape(-1, 3)
      hi = np.array([p.max(axis=0) if len(p) else np.zeros(3) for p in self.points], np.float64).reshape(-1, 3)
      device = self.vert.device
      centre = wp.array((0.5 * (lo + hi)).astype(np.float32), dtype=wp.vec3, device=device)
      half = wp.array((0.5 * (hi - lo)).astype(np.float32), dtype=wp.vec3, device=device)
      self._boxes = (centre, half)
    return self._boxes

  def bvhs(
    self, constructor: str, *, used: np.ndarray[[NMesh], np.dtype[np.bool_]] | None, framed: bool = False
  ) -> wp.array[wp.uint64, [NMesh]]:
    """(nmesh,) uint64 BLAS ids of the meshes, built with `constructor` on first use (cached): a
    BLAS for the `used` meshes (a Renderer's `used_now` mask; None: every mesh), one placeholder
    for the others. With `framed`, meshes that have a frame are built over their re-framed
    vertices, the others share the unframed BLASes."""
    every = used is None or bool(used.all())
    key = (constructor, framed, every)
    if key not in self._bvhs:
      if framed:
        self.bvhs(constructor, used=used)  # the unframed BLASes, shared by the meshes without a frame
      base = self._bvhs[(constructor, False, every)][1] if framed else None
      vertadr, faceadr = host_copy(self.vertadr), host_copy(self.faceadr)
      empty: list[wp.Mesh] = []  # one placeholder BLAS for every unused mesh
      registry = [
        self._placeholder(empty, constructor)
        if used is not None and not used[i]
        else base[i]
        if base is not None and i not in self.framed_points
        else self._blas(MeshId(i), va=int(vertadr[i]), fa=int(faceadr[i]), constructor=constructor, framed=framed)
        for i in range(self.nmesh)
      ]
      ids = wp.array(np.array([m.id for m in registry], dtype=np.uint64), dtype=wp.uint64, device=self.vert.device)
      self._bvhs[key] = (ids, registry)
    return self._bvhs[key][0]

  def _placeholder(self, empty: list[wp.Mesh], constructor: str) -> wp.Mesh:
    """The BLAS of the unused meshes (`used`): one degenerate triangle, never hit; built once into
    `empty`."""
    if not empty:
      device = self.vert.device
      empty.append(
        wp.Mesh(
          points=wp.zeros(3, dtype=wp.vec3, device=device),
          indices=wp.array([0, 1, 2], dtype=wp.int32, device=device),
          bvh_constructor=constructor,
          bvh_leaf_size=2,
        )
      )
    return empty[0]

  def _blas(self, mesh_id: MeshId, *, va: int, fa: int, constructor: str, framed: bool) -> wp.Mesh:
    """The BLAS of mesh `mesh_id` (vertex / face addresses `va`, `fa`): a view of the device
    tables (no copy), or over its re-framed vertices with `framed`."""
    nface = len(self.faces[mesh_id])
    device = self.vert.device
    if framed:
      verts = wp.array(self.framed_points[mesh_id].astype(np.float32), dtype=wp.vec3, device=device)
    else:
      verts = self.vert[va : va + len(self.points[mesh_id])]
    tris = self.face[fa : fa + nface]
    return wp.Mesh(
      points=verts,
      indices=wp.array(ptr=tris.ptr, dtype=wp.int32, shape=(3 * nface,), device=device),
      bvh_constructor=constructor,
      bvh_leaf_size=2,
    )


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MaterialUvTables:
  """Shared authored mappings and UV1 addresses; no per-world or mesh/material product."""

  material: wp.array[MaterialUvMapping, [int]]
  mesh_second: wp.array[wp.vec2i, [int]]
  missing_second: tuple[int, ...]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MeshInstances:
  """How each instance reads its mesh (static, (ninstance,) device arrays).

  A mesh instance's pose may carry a uniform scale (s R); the rest of a host instance's stretch
  (non-uniform, sheared, mirrored) is baked into a copy of its mesh (`pipeline/upload/meshes.py`). A
  material's UV set and UV transform are the instance's (no mesh copies per UV variant).

  Attributes:
    table: the kernels' per-instance record (`kernels.abi.rays.MeshInstance`: own UVs),
      `GeometryFields.mesh_instance`.
  """

  table: wp.array[MeshInstance, [int]]

  @staticmethod
  def build[N: IntVar](
    *,
    texcoord: np.ndarray[[N, 2], np.dtype[np.int32]],
    uv: np.ndarray[[N, 2, 3], np.dtype[np.float32]],
    device: wp.Device,
  ) -> MeshInstances:
    """The device record of per-instance own-UV addresses / transforms."""
    table = wp.zeros(len(texcoord), dtype=MeshInstance, device="cpu").numpy()  # a structured array
    fields = {"uv_xform": uv, "texcoord": texcoord}
    for name, values in fields.items():
      record_field(table, name)[...] = values
    return MeshInstances(table=wp.array(table, dtype=MeshInstance, device=device))

  @staticmethod
  def plain(ninstance: int, device: wp.Device) -> MeshInstances:
    """Instances that read their meshes as they are (no own UVs)."""
    n = max(ninstance, 1)
    uv, texcoord = np.tile(UV_IDENTITY, (n, 1, 1)), np.full((n, 2), UV_MESH, np.int32)
    return MeshInstances.build(texcoord=texcoord, uv=uv, device=device)


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvSuns:
  """The suns cut out of the lighting environment maps (`latlong.extract_sun`, at most one per
  map), shaded as shadowed directional lights after the source's lights (`GpuScene.nlight`): a
  rendering decision, invisible to the light tables. Each one follows its world's environment:
  on where the world shows its map (`env.id`), turned by its `env.yaw`, scaled by its
  `env.intensity` (`shade/lights/pack.py`).

  Attributes:
    env: (nsun,) the map each sun was cut out of.
    direction: (nsun, 3) unit direction towards the sun in its map's frame (before the yaw).
    irradiance: (nsun, 3) irradiance at normal incidence (the map's radiance units x sr).
    angle: (nsun,) angular radius of the sun disc [rad] (its light's).
    extent: (nsun,) angular radius of the region cut out of the map [rad]: the background draws
      the sun over it (`lights.pack.sun_discs`).
  """

  env: np.ndarray[[int], np.dtype[np.int32]]
  direction: np.ndarray[[int, 3], np.dtype[np.float32]]
  irradiance: np.ndarray[[int, 3], np.dtype[np.float32]]
  angle: np.ndarray[[int], np.dtype[np.float32]]
  extent: np.ndarray[[int], np.dtype[np.float32]]

  def __len__(self) -> int:
    """The number of suns."""
    return len(self.env)

  @staticmethod
  def none() -> EnvSuns:
    """No sun."""
    return EnvSuns(
      env=np.zeros(0, np.int32),
      direction=np.zeros((0, 3), np.float32),
      irradiance=np.zeros((0, 3), np.float32),
      angle=np.zeros(0, np.float32),
      extent=np.zeros(0, np.float32),
    )


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class GpuEnvironment:
  """Environment maps (lighting / background) and the constant ambient.

  Attributes:
    maps: GPU resources of the maps (or the placeholder of a scene without maps).
    lit: whether the maps light the scene (environment.enabled="auto").
    has_maps: whether there is any map (else environment.background shows nothing).
    intensity: default radiance scale (per world: `Renderer.per_world["env.intensity"]`).
    yaw: default rotation about +z [rad].
    ambient: constant ambient radiance (display-relative) used when the maps do not light the scene.
    background: the map shown by camera rays leaving the scene, apart from the
      lighting maps (a MuJoCo skybox beside an image light); NO_ENVIRONMENT: the lighting map.
    background_intensity: radiance scale of the `background` map.
    suns: the suns cut out of the lighting maps (`EnvSuns`; none by default).
  """

  maps: envmap_lib.EnvironmentTextures
  lit: bool = False
  has_maps: bool = False
  intensity: float = 1.0
  yaw: float = 0.0
  ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
  background: EnvironmentId = NO_ENVIRONMENT
  background_intensity: float = 1.0
  suns: EnvSuns = dataclasses.field(default_factory=EnvSuns.none)


@final
@dataclasses.dataclass(kw_only=True)
class GpuScene[S]:
  """GPU scene of one source model (see the module docstring).

  Attributes:
    device: Warp device of every array.
    ninstance: number of instances (MuJoCo: geoms; index i is geom i).
    nmaterial: number of materials.
    nlight: number of lights shaded: the source's own (`source_lights`, the columns of its light
      tables, what `Renderer.per_world` light tables index), then the environment suns
      (`environment.suns`).
    tables: slot name -> default array (`binding.Slots`).
    derive_inputs: what the derived per-world material fields start from (`derived_fields.py`).
    instance_type: (ninstance,) ShapeType.
    instance_names: (ninstance,) instance names (MuJoCo geom names, Newton shape labels, USD
      prim paths; "" when unnamed), matched by `PerWorld` patterns.
    rendered: indices of the instances to render (e.g. MuJoCo's enabled geom groups).
    init_pose: the instances' initial poses (the world BVH topology is built from them with the
      scene's "instance.size" / "instance.mesh"), the one host copy of them.
    bound_radius: (ninstance,) bounding radius per instance (light grid extent).
    instance_shadow_transmit: (ninstance, 3) float32 RGB transmittance of translucent instances
      (blend-mode or transmissive materials, e.g. window glass: base color x (1 - opacity)), 0
      for opaque ones; read by translucent shadows, and the GI probes' placement ignores the
      translucent ones.
    instance_optics: (ninstance, `types.OPTICS_ROWS`, 4) float32 optics rows of their
      materials (Beer-Lambert extinction, refraction index; zeros: none).
    instance_twosided: (ninstance,) int32 bits (`kernels.abi.rays.TWO_SIDED`: never back-face
      culled, USD doubleSided, thin sheets seen from both sides; `CLOSED_PANE`: a closed
      see-through pane, its exit face stepped over by the camera rays).
    optics_inputs: what those three tables are composed of (`pipeline/upload/optics.compose`),
      recomposed per world when a renderer chooses materials per world
      (`core/pipeline/worlds/optics.py`); None when no material choice can change them (no solid,
      volume or see-through material or color).
    light_instance: (source_lights,) int32 instance whose emission each light stands for (emissive
      panels converted to rect lights), -1 for none. GI probes skip that emission (the light
      already carries it).
    light_visible: (k, 2) int32 (light, LightShape) of the lights whose emitting surface camera
      rays see (`shade/lights/visible.py`; area lights without geometry of their own).
    ies: the lights' IES profiles (`shade/lights/ies.py`; none by default).
    meshes: triangle meshes and height fields.
    mesh_instances: how each instance reads its mesh (UVs, vertex colors).
    textures: material textures (texture id = index), kept alive for `texture_array`, the kernel
      array of them; their stored sizes `tex_size` (ntex,) vec2 and texel layouts `tex_layout`
      (ntex,) int (`types.TEX_LAYOUT_*`: the stored channel each read channel comes from).
    environment: environment maps and ambient.
    headlight: camera headlight.
    cameras: cameras authored in the source (name -> CameraSpec with `mount` = its name).
    clip: default (znear, zfar) of the cameras.
    capabilities, binding: see `Capabilities`, `binding.Binding`.
    color_rule: how an instance's color combines with its material's base color, the source's
      convention (`ColorRule`: TINT multiplies them; MUJOCO, MuJoCo's default-rgba rule).
    deformables: deformable geometry (`core/pipeline/gpu_scene/deformables.py`; empty without any).
    material_uv: the materials' authored UV mappings, read when "instance.material" is declared
      per world (None: no material reads another UV set or a UV transform).
    segment_id: (1 or nworld, ninstance) int32 id written to the `seg` output per (world,
      instance), or None for the instance index (the renderer binds that identity table);
      deformable d writes `flex_segment + d`. A Renderer can replace it
      (`Renderer.set_segment_table`).
    flex_segment: segment id of deformable 0.
    instance_labels: (ninstance,) authored semantic class of each instance ("" none; UsdSemantics),
      the classes `Renderer.set_semantics` starts from (empty: none authored).
    report: load report.
    layers: the layered lobes' static tables (None: a source without them, e.g. MuJoCo).
  """

  device: wp.Device
  ninstance: int
  nmaterial: int
  nlight: int
  tables: Slots
  derive_inputs: DeriveInputs
  instance_type: wp.array[int, [int]]
  instance_names: list[str]
  rendered: np.ndarray[[int], np.dtype[np.int32]]
  init_pose: InstancePoses[int]
  bound_radius: np.ndarray[[int], np.dtype[np.float32]]
  instance_shadow_transmit: np.ndarray[[int, 3], np.dtype[np.float32]]
  instance_optics: np.ndarray[[int, int, int], np.dtype[np.float32]]
  instance_twosided: np.ndarray[[int], np.dtype[np.int32]]
  optics_inputs: OpticsInputs | None
  light_instance: np.ndarray[[int], np.dtype[np.int32]]
  light_visible: np.ndarray[[int, 2], np.dtype[np.int32]]
  meshes: Meshes
  mesh_instances: MeshInstances
  textures: list[wp.Texture2D]
  texture_array: wp.array[wp.Texture2D, [int]]
  tex_size: wp.array[wp.vec2, [int]]
  tex_layout: wp.array[int, [int]]
  environment: GpuEnvironment
  headlight: GpuHeadlight
  cameras: dict[str, CameraSpec]
  clip: tuple[float, float]
  capabilities: Capabilities
  binding: Binding[S]
  color_rule: ColorRule
  ies: IesLights = dataclasses.field(default_factory=lambda: IesLights.none(0))
  deformables: Deformables
  material_uv: MaterialUvTables | None = None
  segment_id: wp.array[int, [int, int]] | None = None
  flex_segment: int = 0
  instance_labels: list[str] = dataclasses.field(default_factory=list)
  report: LoadReport = dataclasses.field(default_factory=LoadReport)
  layers: LayerTables | None = None

  @property
  def source_lights(self) -> int:
    """The source's own lights (the light tables' columns): `nlight` without the environment
    suns."""
    return self.nlight - len(self.environment.suns)

  @property
  def camera_names(self) -> list[str]:
    """Names of the scene cameras, in order."""
    return list(self.cameras)


def shadow_transmit(scene: GpuScene) -> np.ndarray[[int, 3], np.dtype[np.float32]]:
  """(max(ninstance, 1), 3) float32 RGB shadow transmittance of the instances (a kernel table)."""
  t = scene.instance_shadow_transmit
  return t if len(t) else np.zeros((1, 3), np.float32)


def shadow_casters(slots: Slots, declared: frozenset[str]) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(nlight,) the lights that can cast a shadow: casting in some world, whether active or not (a
  light can be turned on at any time, without a host call); a cast-shadow table declared per world
  counts as on in every light.

  The shadow budget compares its cap with this count, not with the light count: a budget that
  covers every potential caster traces each of them anyway and only adds its selection. A light
  whose live cast-shadow flag is turned on later (an aliased model table) is caught by the
  validation (`validator_kernels.Word.CASTERS`), which re-resolves the Features.
  """
  table = np.asarray(host_copy(slots["light.cast_shadow"]), bool)
  return np.ones(table.shape[1], bool) if "light.cast_shadow" in declared else table.any(axis=0)
