"""SceneBuilder: what adapters and tests fill with meshes, images, materials, instances and lights
to make a `Scene`.

Format-neutral (numpy only): the USD and Newton adapters and the tests use it. Every `add_*`
validates its values (`checks`: finite, in range, ids of rows added before) and fails there with
a clear message. Meshes are deduplicated by content hash (`mesh_pool`), images by (source, size
cap, transfer), textures by (image, color space, wrap) and materials by value. Images are read and
decoded on a thread pool as they are registered, while the adapter still reads its source
(`blob.Ahead`); `build` writes them (and decodes the rest) straight into the texel blob (in
memory, or in the `.wpbr` file).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Unpack, final

import numpy as np
import numpy.typing as npt

from raywarp.scene.build import displacement
from raywarp.scene.build.image_pool import ImagePool
from raywarp.scene.build.instance_pool import InstanceChunk, InstancePool
from raywarp.scene.build.mesh_pool import MeshAttributes, MeshPool
from raywarp.scene.images import blob, mips
from raywarp.scene.images import decode as imaging
from raywarp.scene.images.colorspace import ColorSpace, code, texel_transfer, texture_space
from raywarp.scene.images.mips import Transfer
from raywarp.scene.lighting import checks as light_checks
from raywarp.scene.model import checks, lights, wpbr
from raywarp.scene.model.lights import Cameras, Environments, Lights
from raywarp.scene.model.material import Displacement, Material, OpenPbrExtras
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.table import record_columns
from raywarp.scene.model.tables import Images, MaterialExtras, Materials, Textures, texel_digest
from raywarp.scene.model.types import (
  NO_MATERIAL,
  NO_MESH,
  NO_TEXTURE,
  Array,
  Camera,
  CameraId,
  EnvironmentId,
  ImageId,
  InstanceFlags,
  InstanceId,
  InstanceKind,
  Json,
  LightId,
  MaterialId,
  MeshId,
  Texels,
  TextureId,
  TextureRef,
  Wrap,
)
from raywarp.scene.model.xform import XformLike, as_xform


@final
class SceneBuilder:
  """Host-side scene assembly; `build()` returns the `Scene`.

  Attributes:
    report: load report shared with the adapter filling the builder.
    meta: metadata stored with the scene (and in its cache file).
    dependencies: the files the scene is read from (missing ones included); an adapter adds its
      own (layers, modules), the builder the image files.
    displacement_options: the limits of the displacement preparation at `build`.
  """

  def __init__(
    self, report: LoadReport | None = None, *, displacement_options: displacement.DisplacementOptions | None = None
  ) -> None:
    """An empty builder recording into `report` (a new one by default)."""
    self.report = LoadReport() if report is None else report
    self.meta: dict[str, Json] = {}
    self.dependencies: set[str] = set()
    self.displacement_options = displacement.DisplacementOptions() if displacement_options is None else displacement_options
    self._meshes = MeshPool()
    self._images = ImagePool(self.report, self.dependencies)
    # (image, color space code, wrap s, wrap t) of each texture.
    self._textures: list[tuple[ImageId, int, int, int]] = []
    self._texture_keys: dict[tuple[ImageId, int, int, int], TextureId] = {}
    self._materials: list[Material] = []
    self._material_keys: dict[Material, MaterialId] = {}
    self._instances = InstancePool()
    self._lights: list[lights.Light] = []
    self._light_bodies: list[int] = []  # `Instances.bodies` index of each light's body (-1: static)
    self._envs: list[lights.EnvironmentLight] = []
    self._cameras: list[tuple[str, Camera]] = []

  # Meshes ---------------------------------------------------------------------------------------

  def add_mesh(self, points: npt.ArrayLike, tris: npt.ArrayLike, **attributes: Unpack[MeshAttributes]) -> MeshId:
    """Adds a triangle mesh (deduplicated by content); returns its mesh id.

    Args:
      points: (V, 3) finite vertex positions (local frame).
      tris: (T, 3) vertex indices, counter-clockwise front faces; at least one triangle.
      **attributes: normals, UVs, second UV set, vertex colors (`mesh_pool.MeshAttributes`).

    Raises:
      ValueError: for a mesh without triangles, non-finite values or out-of-range indices.
    """
    (mesh,) = self._meshes.add(points, tris, attributes, report=self.report)
    if mesh == NO_MESH:
      raise ValueError("add_mesh: a mesh needs at least one triangle (add_mesh_parts allows empty parts)")
    return mesh

  def add_mesh_parts(
    self,
    points: npt.ArrayLike,
    tris: npt.ArrayLike,
    *,
    tri_part: npt.ArrayLike | None = None,
    nparts: int = 1,
    **attributes: Unpack[MeshAttributes],
  ) -> list[MeshId]:
    """Adds a mesh split into `nparts` meshes by `tri_part` (per-face materials).

    The parts share one vertex / normal / UV / color range; each gets its own triangle range.
    Returns one mesh id per part (NO_MESH for a part without triangles). Attributes as `add_mesh`.
    Without `tri_part`, all triangles belong to part 0 and the remaining parts are empty.

    Raises:
      ValueError: for `nparts` below 1, a `tri_part` outside [0, nparts), non-finite values or
        out-of-range indices.
    """
    if nparts < 1:
      raise ValueError(f"add_mesh_parts: nparts must be at least 1, got {nparts}")
    return self._meshes.add(points, tris, attributes, report=self.report, tri_part=tri_part, nparts=nparts)

  # Images, textures, materials ----------------------------------------------------------------

  def add_image(
    self, source: str | os.PathLike[str] | Array, *, max_size: int | None = None, transfer: Transfer = "srgb"
  ) -> ImageId:
    """Registers an image (decoded at `build`); returns its id, NO_IMAGE if the file is unusable.

    Keyed by its 8-bit `transfer` too ("linear", "srgb" or a pure gamma's exponent): mips filter
    in linear space. An array is (h, w) or (h, w, 1-4) uint8 or floating point (finite), at
    least 1 x 1.

    Raises:
      ValueError: for a `max_size` below 1 or an array of another shape, dtype or a NaN / inf.
    """
    mips.check_transfer(transfer, "add_image")
    return self._images.add(source, max_size=max_size, transfer=transfer)

  def add_derived_image(self, derived: imaging.Derived) -> ImageId:
    """Registers an image computed at build time from image files (`images.Derived`); returns its
    id, NO_IMAGE if a source file is missing or unsupported (reported).

    Raises:
      ValueError: for a source that is not an image file (an array or a derived image).
    """
    return self._images.add_derived(derived)

  def add_texture(
    self,
    image: str | os.PathLike[str] | Array | ImageId,
    *,
    colorspace: ColorSpace | str = ColorSpace.SRGB_REC709,
    wrap: tuple[int | str, int | str] = (Wrap.REPEAT, Wrap.REPEAT),
    max_size: int | None = None,
  ) -> TextureId:
    """A sampler over an image (a path, an array or an `add_image` id); NO_TEXTURE if unusable.

    Raises:
      ValueError: for an unknown color space or wrap mode, a wrap that is not an (s, t) pair, an
        image id not added (-1 gives NO_TEXTURE) or a bad `max_size` / array (`add_image`).
    """
    cs = texture_space(colorspace, where="add_texture")
    if isinstance(wrap, str | int) or len(wrap) != 2:  # (s, t)
      raise ValueError(f"add_texture: wrap is the (s, t) pair of wrap modes, got {wrap!r}")
    ws, wt = (checks.named("wrap", w, Wrap, where="add_texture") for w in wrap)
    if isinstance(image, (int, np.integer)):
      image_id = ImageId(checks.index("image", image, len(self._images), where="add_texture"))
    else:
      image_id = self.add_image(image, max_size=max_size, transfer=texel_transfer(cs))
    if image_id < 0:
      return NO_TEXTURE
    key = (image_id, code(cs), int(ws), int(wt))
    if key not in self._texture_keys:
      self._texture_keys[key] = TextureId(len(self._textures))
      self._textures.append(key)
    return self._texture_keys[key]

  def add_material(self, material: Material) -> MaterialId:
    """Adds an OpenPBR material (deduped by value, its `name` included: two source materials of
    equal values stay two rows, so a per-material write by name reaches only its own instances;
    all-default extras become None); returns its id.

    Raises:
      ValueError: for a value out of its range or a texture not added (`checks.material`).
    """
    if material.displacement is not None:
      checks.material(material, len(self._textures))
      amount = material.displacement.amount
      if isinstance(amount, TextureRef) and amount.scale[0] == 0.0:
        amount = amount.bias[0]
      material = dataclasses.replace(material, displacement=None if amount == 0.0 else Displacement(amount=amount))
    if material.extras == OpenPbrExtras():  # no extras: one table row less, dedup with `extras=None`
      material = dataclasses.replace(material, extras=None)
    try:
      hash(material)
    except TypeError:  # a list / array where a tuple goes: the check names the field
      checks.material(material, len(self._textures))
      raise
    if material not in self._material_keys:
      checks.material(material, len(self._textures))
      self._material_keys[material] = MaterialId(len(self._materials))
      self._materials.append(material)
    return self._material_keys[material]

  @property
  def materials(self) -> tuple[Material, ...]:
    """The materials added so far (material id = index)."""
    return tuple(self._materials)

  def material(self, material: MaterialId) -> Material:
    """Material `material` (an id `add_material` returned), without copying the list.

    Raises:
      ValueError: for an id not added (NO_MATERIAL included).
    """
    return self._materials[checks.index("material", material, len(self._materials), where="material", optional=False)]

  # Instances, lights, environments, cameras ----------------------------------------------------

  def add_instance(  # noqa: PLR0913 (keyword-only instance columns)
    self,
    kind: int | str,
    *,
    mesh: MeshId = NO_MESH,
    material: MaterialId = NO_MATERIAL,
    size: Sequence[float] = (0.0, 0.0, 0.0),
    xform: XformLike = None,
    color: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    flags: int = InstanceFlags.DEFAULT,
    segment: int | None = None,
    path: str | None = None,
    body: str | None = None,
    source: int = -1,
    label: str | None = None,
  ) -> InstanceId:
    """Adds one instance; returns its index. `segment` defaults to the instance index; `source` is
    its id in what the adapter binds (`Instances.source`: a Newton shape; -1 none); `label` its
    semantic class (`Instances.label`). `xform` is None (identity), (3, 4) or homogeneous (4, 4),
    never a batch. A fourth row must be (0, 0, 0, 1). Validated as `add_instances`."""
    return InstanceId(
      self.add_instances(
        kind,
        xforms=as_xform(xform)[None],
        mesh=mesh,
        material=material,
        size=size,
        color=color,
        flags=flags,
        segment=segment,
        paths=None if path is None else [path],
        body=body,
        source=source,
        labels=None if label is None else [label],
      ).start
    )

  def _instance_kind(self, kind: int | str, mesh: MeshId, material: MaterialId) -> InstanceKind:
    """The validated kind of instances of `mesh` (required by mesh instances) and `material`.

    Raises:
      ValueError: for an unknown kind, or a mesh / material id not added.
    """
    k = checks.named("kind", kind, InstanceKind, where="add_instances")
    where = f"add_instances {k.name.lower()}"
    checks.index("mesh", mesh, len(self._meshes), where=where, optional=k != InstanceKind.MESH)
    checks.index("material", material, len(self._materials), where=where)
    return k

  def add_instances(  # noqa: PLR0913 (keyword-only instance columns)
    self,
    kind: int | str,
    *,
    xforms: npt.ArrayLike,
    mesh: MeshId = NO_MESH,
    material: MaterialId = NO_MATERIAL,
    size: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    flags: int = InstanceFlags.DEFAULT,
    segment: int | None = None,
    paths: Sequence[str] | None = None,
    body: str | None = None,
    source: int = -1,
    labels: Sequence[str] | None = None,
  ) -> range:
    """Adds N instances of one shape at `xforms` (N, 3|4, 4) (vectorised PointInstancer path);
    `labels`: each one's semantic class ("" or None: none). Fourth rows must be (0, 0, 0, 1).

    Raises:
      ValueError: for an unknown kind or flag, a mesh instance without a mesh, a mesh / material
        id not added, transforms with another shape, non-affine or non-finite float32 values,
        a negative or non-finite size or color, a `segment` / `source` below -1, or
        `paths` / `labels` not holding one entry per instance.
    """
    k = self._instance_kind(kind, mesh, material)
    where = f"add_instances {k.name.lower()}"
    for name, value in (("segment", -1 if segment is None else segment), ("source", source)):
      checks.in_range(name, value, where, low=-1)
    transforms = checks.finite_array("xforms", xforms, where)
    if transforms.ndim != 3 or transforms.shape[1:] not in ((3, 4), (4, 4)):
      raise ValueError(f"{where}: xforms must have shape (N, 3, 4) or (N, 4, 4), got {transforms.shape}")
    if transforms.shape[1] == 4 and not np.all(transforms[:, 3, :] == (0.0, 0.0, 0.0, 1.0)):
      raise ValueError(f"{where}: xforms must have affine last rows (0, 0, 0, 1)")
    with np.errstate(over="ignore"):
      stored = transforms[:, :3, :].astype(np.float32)
    if not np.all(np.isfinite(stored)):
      raise ValueError(f"{where}: xforms must contain finite float32 values")
    chunk = InstanceChunk(
      kind=k,
      xforms=stored,
      mesh=mesh,
      material=material,
      size=checks.color("size", size, where),
      color=checks.color("color", color, where, size=4),
      flags=checks.flags("flags", flags, InstanceFlags, where=where),
      segment=segment,
      source=source,
    )
    return self._instances.add(chunk, paths=paths, body=body, labels=labels)

  def add_light(self, light: lights.Light) -> LightId:
    """Adds a canonical UsdLux light; ValueError for an invalid one (`light_checks.check_light`)."""
    light_checks.check_light(
      light, ntexture=len(self._textures), ninstance=len(self._instances), where=f"add_light {light.path!r}"
    )
    self._lights.append(light)
    self._light_bodies.append(self._instances.bodies.add(light.body))
    return LightId(len(self._lights) - 1)

  def add_environment(self, env: lights.EnvironmentLight) -> EnvironmentId:
    """Adds a canonical UsdLux environment; ValueError for an invalid one (`check_environment`)."""
    light_checks.check_environment(env, ntexture=len(self._textures), where=f"add_environment {env.path!r}")
    self._envs.append(env)
    return EnvironmentId(len(self._envs) - 1)

  def add_camera(self, name: str, camera: Camera) -> CameraId:
    """Adds a named camera; returns its index.

    Raises:
      ValueError: for a camera value out of range (`checks.camera`).
    """
    checks.camera(camera, f"add_camera {name!r}")
    self._cameras.append((name, camera))
    return CameraId(len(self._cameras) - 1)

  # Build ----------------------------------------------------------------------------------------

  @property
  def nmesh(self) -> int:
    """Number of meshes."""
    return len(self._meshes)

  @property
  def ninstance(self) -> int:
    """Number of instances."""
    return len(self._instances)

  def read_states(self) -> dict[str, wpbr.Dependency]:
    """The state of every image file read so far (fstat when read: `wpbr.record` reuses them),
    once the reads ahead are done."""
    return self._images.read_states()

  def instance_xforms(self) -> np.ndarray[[int, 3, 4], np.dtype[np.float32]]:
    """The (N, 3, 4) transforms of the instances added so far (float32, as stored)."""
    return self._instances.xforms()

  def _scene(self, images: Images, probed: Sequence[blob.Probed]) -> Scene:
    """The scene of every table (`images`: the image rows; its texel blob comes later)."""
    tex = np.asarray(self._textures, dtype=np.int64).reshape(-1, 4)
    cameras = [c for _, c in self._cameras]
    instances = self._instances.table()
    with self.report.timer("build.displacement"):
      displacement.prepare(
        self._meshes,
        instances,
        self._materials,
        self._textures,
        self._images.sources,
        probed,
        self.displacement_options,
        self.report,
      )
    return Scene(
      meshes=self._meshes.table(),
      instances=instances,
      materials=Materials.from_materials(self._materials),
      material_extras=MaterialExtras.from_materials(self._materials),
      textures=Textures.from_columns({"image": tex[:, 0], "colorspace": tex[:, 1], "wrap": tex[:, 2:4]}),
      images=images,
      lights=Lights.from_records(self._lights, bodies=self._light_bodies),
      environments=Environments.from_records(self._envs),
      cameras=Cameras.from_columns(record_columns(cameras, Cameras.SCHEMA), {"names": [n for n, _ in self._cameras]}),
      report=self.report,
      meta=self.meta,
    )

  def build(self, path: str | os.PathLike[str] | None = None, *, mips: bool = False, threads: int | None = None) -> Scene:
    """Assembles the scene and decodes its images.

    Args:
      path: write the scene to this `.wpbr` file while decoding (images are decoded straight
        into the file) and return it memory-mapped from there.
      mips: store full mip chains (box filter; sRGB filtered in linear space).
      threads: image decode threads, at least 1 (None: the CPU count, at most 32).

    Raises:
      ValueError: for `threads` below 1.
    """
    blob.worker_count(threads)  # validated before any work
    report = self.report
    # Images: the reads ahead (probed, some decoded), then the blob layout.
    layout, images, probed = self._images.layout(mips=mips)
    with report.timer("build.tables"):
      scene = self._scene(images, probed)
    self.meta["mips"] = mips

    digests = scene.images.digest

    def decode(sink: blob.Sink) -> None:
      """Decodes every image into `sink`, recording its content digest (`Images.digest`)."""

      def digested(i: int, levels: list[Texels]) -> None:
        """Records image `i`'s content digest, then hands its levels to `sink`."""
        digests[i] = texel_digest(levels)
        sink(i, levels)

      self._images.decode(layout, probed, digested, threads=threads)

    def compact(move: Callable[[int, int, int], None]) -> int:
      """Stores each image content once (`blob.compaction`: `move(source, destination, bytes)`
      in blob order, the offsets rewritten); the compacted blob's size."""
      plan = blob.compaction(layout, digests)
      for source, destination, nbytes in plan.moves:
        move(source, destination, nbytes)
      scene.images.offset[:] = plan.offsets
      if plan.shared:
        report.count("images.shared_content", plan.shared)
      return plan.total

    if path is None:
      data = np.zeros(layout.total, np.uint8)  # the alignment padding stays 0 (deterministic files)

      def copy(offset: int, level: Texels) -> None:
        """Copies a decoded level into the in-memory blob."""
        data[offset : offset + level.nbytes] = level.reshape(-1).view(np.uint8)

      def move(source: int, destination: int, nbytes: int) -> None:
        """Moves bytes of the in-memory blob down."""
        data[destination : destination + nbytes] = data[source : source + nbytes]

      decode(layout.sink(copy))
      total = compact(move)
      data = data if total == len(data) else data[:total].copy()
      # Read-only, as a loaded scene's: an in-place edit would silently differ from the digests.
      data.setflags(write=False)
      digests.setflags(write=False)
      scene.images.data = data
      return scene

    def write_blob(writer: wpbr.Writer) -> None:
      """Decodes the images straight into the file's blob, then stores each content once."""
      decode(layout.sink(lambda offset, level: writer.write("images.data", level, offset)))
      writer.shrink("images.data", compact(lambda s, d, n: writer.move("images.data", s, d, n)))

    # The blob is written while decoding: the scene holds a zero-stride placeholder of its size.
    scene.images.data = np.broadcast_to(np.zeros(1, np.uint8), (layout.total,))
    scene.save(path, {"images.data": write_blob})
    return Scene.load(path)
