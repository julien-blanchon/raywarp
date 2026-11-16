"""Host environments and cameras -> core environment maps and scene cameras.

Environments: radiance baked into the image; the lookups turn every map by one runtime yaw about
+z (the first map's), another map's own yaw is rolled into its columns, a tilted map is resampled
once; each map's dominant sun is cut out as a directional light (`_cut_suns`; cached by the
maps' content for the MuJoCo adapter, `cut_suns`). Cameras are mounted on their own static pose,
each with its own clipping range; the Renderer's free cameras get `default_near` and no far plane
(`SceneCameras.clip`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import warp as wp

from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, EnvSuns, GpuEnvironment
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.textures.pool import image_digest
from raywarp.core.pipeline.upload.lights import derive_environments
from raywarp.core.post.grading.color import srgb_to_linear
from raywarp.core.shade.environment import envmap as envmap_lib
from raywarp.core.shade.environment.latlong import rotate_latlong
from raywarp.scene.images import colorspace, latlong
from raywarp.scene.images.colorspace import REC709_Y, ColorSpace
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import Matrices, Projection, Rows, TextureId
from raywarp.scene.model.xform import Mat3, polar

# Color channels of an RGB(A) map.
_RGB: Final = 3
# A yaw below this is none [rad].
_YAW_EPS: Final = 1.0e-6
# A rotation within this of a pure yaw (`np.allclose` atol) is one: rolled, not resampled.
_TILT_TOL: Final = 1.0e-4
# The radiance map of an environment without a texture: white, (h, w, 3).
_WHITE_MAP_SHAPE: Final = (4, 8, 3)
# The environment records the processed maps depend on (`_inputs_digest`).
_RECORD_FIELDS: Final = (
  "texture",
  "intensity",
  "exposure",
  "color",
  "enable_color_temperature",
  "color_temperature",
  "rotation",
  "illuminance",
)

# Processed environment maps (RGB float32), their suns and the report of their processing: a
# bake's value (`bake_cache.cached`).
type _Maps = tuple[list[np.ndarray[[int, int, 3], np.dtype[np.float32]]], EnvSuns, LoadReport]


def _yaw_part(rot: Mat3) -> tuple[float, Mat3]:
  """(yaw about +z, the rest Rz(-yaw) @ rot) of a map -> world rotation (identity rest: a pure
  yaw)."""
  yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
  return yaw, _rz(-yaw) @ rot


def _rz(angle: float) -> Mat3:
  """The rotation by `angle` about +z."""
  c, s = math.cos(angle), math.sin(angle)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cut_suns(
  images: Sequence[np.ndarray[[int, int, 3], np.dtype[np.float32]]],
  lit: Sequence[bool],
  paths: Sequence[str],
  report: LoadReport,
  *,
  device: wp.Device,
) -> tuple[list[np.ndarray[[int, int, 3], np.dtype[np.float32]]], EnvSuns]:
  """`_cut_suns`, cached on disk by the maps' content (`bake_cache`, kind "env_sun"): a scene
  built again from the same maps (a MuJoCo model, whose maps are not processed by a cached host
  upload) loads the cut maps and their suns instead of searching them again (15-50 ms per 1k map,
  seconds for a large bank)."""
  key = bake_cache.key("env_sun", device, host=(tuple(bake_cache.array_digest(im) for im in images), tuple(lit), tuple(paths)))

  def cut() -> tuple[_Maps, bake_cache.Bake]:
    local = LoadReport()
    out, suns = _cut_suns(images, lit, paths, local)
    return (out, suns, local), {f"image{i}": out[i] for i in suns.env.tolist()} | _stored(suns, local)

  def restore(stored: bake_cache.Bake) -> _Maps:
    suns, local = _restored(stored)
    return [stored[f"image{i}"] if i in suns.env else image for i, image in enumerate(images)], suns, local

  out, suns, local = bake_cache.cached(key, cut, restore)
  report.merge(local)
  return out, suns


def _stored(suns: EnvSuns, report: LoadReport) -> dict[str, np.ndarray]:
  """The cache arrays of `suns` and the `report` of their cut (`_restored` reads them back)."""
  return {f"sun_{f.name}": getattr(suns, f.name) for f in dataclasses.fields(EnvSuns)} | {
    "report": np.array(json.dumps(report.to_dict()))
  }


def _restored(stored: bake_cache.Bake) -> tuple[EnvSuns, LoadReport]:
  """(the suns, the report) `_stored` wrote into the cache arrays `stored`."""
  suns = EnvSuns(**{f.name: stored[f"sun_{f.name}"] for f in dataclasses.fields(EnvSuns)})
  return suns, LoadReport.from_dict(json.loads(str(stored["report"])))


def _cut_suns(
  images: Sequence[np.ndarray[[int, int, 3], np.dtype[np.float32]]],
  lit: Sequence[bool],
  paths: Sequence[str],
  report: LoadReport,
) -> tuple[list[np.ndarray[[int, int, 3], np.dtype[np.float32]]], EnvSuns]:
  """(the maps without their suns, the suns) of equirectangular radiance maps (h, w, 3) in their
  lookup frame (before the runtime yaw): the sun of each lighting map (`lit`) that has a dominant
  compact one (`latlong.extract_sun`), reported per map (`paths`); background-only maps keep
  theirs."""
  out, rows = [], []
  for i, (image, on) in enumerate(zip(images, lit, strict=True)):
    cut = latlong.extract_sun(image) if on else None
    if cut is None:
      out.append(image)
      continue
    rest, sun = cut
    out.append(rest)
    rows.append((i, sun))
    report.approx(f"gpu.environment.sun({100.0 * sun.share:.0f}% of the facing irradiance -> directional light)", paths[i])
  if not rows:
    return out, EnvSuns.none()
  return out, EnvSuns(
    env=np.array([i for i, _ in rows], np.int32),
    direction=np.array([s.direction for _, s in rows], np.float32).reshape(-1, 3),
    irradiance=np.array([s.irradiance for _, s in rows], np.float32).reshape(-1, 3),
    angle=np.array([s.angle for _, s in rows], np.float32),
    extent=np.array([s.extent for _, s in rows], np.float32),
  )


def _map_radiance(scene: Scene, texture: TextureId, report: LoadReport) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """(h, w, 3) linear radiance texels of an environment texture (a 4 x 8 white map for none);
  a space not stored as is is converted to Rec.709 (reported)."""
  if texture < 0:
    return np.ones(_WHITE_MAP_SHAPE, np.float32)
  img = scene.images.level(int(scene.textures.image[texture]), 0)
  space = colorspace.from_code(int(scene.textures.colorspace[texture]))
  if not colorspace.stored_as_is(space, float_texels=img.dtype != np.uint8):
    report.approx(f"gpu.environment.colorspace:{space.value}(converted to Rec.709 at upload)", f"texture {texture}")
    return colorspace.decode(img, space)[..., :3]
  f = img.astype(np.float32)
  if img.dtype == np.uint8:
    f /= 255.0
    if space == ColorSpace.SRGB_REC709:
      f = srgb_to_linear(f).astype(np.float32)
  return f[..., :3] if f.shape[2] >= _RGB else np.repeat(f[..., :1], 3, axis=2)


def environment(scene: Scene, report: LoadReport, *, env_sun: bool, device: wp.Device) -> GpuEnvironment:
  """The environment maps on `device`, with `env_sun` the sun cut out of each map that has one
  (`GpuEnvironment.suns`, lit as a directional light where a world shows that map).

  The processed maps (decoded, rotated, scaled, downsampled to the radiance texture's width, sun
  cut out) and their suns are a pure function of the maps' texels and records: cached on disk by
  that content (`bake_cache`, kind "environment"), so a warm start only uploads them."""
  envs = scene.environments
  if not len(envs):
    return GpuEnvironment(maps=envmap_lib.empty_environments(device=device))
  yaw0, _ = _yaw_part(envs.rotation[0].astype(np.float64))
  key = bake_cache.key("environment", device, host=(_inputs_digest(scene), env_sun))

  def process() -> tuple[_Maps, bake_cache.Bake]:
    local = LoadReport()
    images, suns = _processed(scene, local, yaw0=yaw0, env_sun=env_sun, device=device)
    return (images, suns, local), {f"image{i}": im for i, im in enumerate(images)} | _stored(suns, local)

  def restore(stored: bake_cache.Bake) -> _Maps:
    return [stored[f"image{i}"] for i in range(len(envs))], *_restored(stored)

  images, suns, local = bake_cache.cached(key, process, restore)
  report.merge(local)
  maps = envmap_lib.build_environments(images, device=device)
  return GpuEnvironment(maps=maps, lit=True, has_maps=True, yaw=yaw0, suns=suns)


def _inputs_digest(scene: Scene) -> str:
  """Digest of what the processed maps depend on: each map's image (its content digest,
  `Images.digest`) and color space, and the environment records.

  Raises:
    ValueError: for a map image without a content digest (every `SceneBuilder` scene has them).
  """
  envs = scene.environments
  h = hashlib.blake2b(digest_size=16)
  for i in range(len(envs)):
    t = int(envs.texture[i])
    if t >= 0:
      h.update(image_digest(scene, int(scene.textures.image[t])))
      h.update(str(int(scene.textures.colorspace[t])).encode())
  for name in _RECORD_FIELDS:
    h.update(np.ascontiguousarray(getattr(envs, name)).data)
  h.update(repr(list(envs.paths)).encode())
  return h.hexdigest()


def _processed(
  scene: Scene, report: LoadReport, *, yaw0: float, env_sun: bool, device: wp.Device
) -> tuple[list[np.ndarray[[int, int, 3], np.dtype[np.float32]]], EnvSuns]:
  """(each map's radiance in the maps' common frame, at most MAX_RADIANCE_WIDTH wide, the suns cut
  out of them with `env_sun`)."""
  envs = scene.environments
  images = []
  scales = derive_environments(envs)  # (nenv, 3) float32 radiance scales
  for i in range(len(envs)):
    rgb = _map_radiance(scene, TextureId(int(envs.texture[i])), report)
    finite = np.isfinite(rgb)
    if not finite.all():
      # NaN / infinite texels (a damaged or overflowed float map) carry no radiance: zeroed and
      # reported once per map, instead of failing the upload (`EnvironmentMap` refuses them).
      report.approx("gpu.environment.nonfinite_radiance_zeroed", f"{envs.paths[i]} ({int((~finite).sum())} components)")
      rgb = np.where(finite, rgb, 0.0).astype(np.float32)
    if (rgb < 0.0).any():
      # Out-of-gamut texels (a converted space, or a float map stored as is): clamped and reported,
      # as the texture upload does (`EnvironmentMap` would clamp them with a warning).
      report.approx("gpu.environment.negative_radiance_clamped", envs.paths[i])
      rgb = np.maximum(rgb, 0.0)
    rgb = _in_common_frame(rgb, envs.rotation[i].astype(np.float64), yaw0=yaw0, device=device) * scales[i]
    target = float(envs.illuminance[i])
    if target > 0.0:
      rgb = rgb * _illuminance_scale(rgb, target, envs.paths[i], report)
    images.append(envmap_lib.downsample(np.ascontiguousarray(rgb, dtype=np.float32), envmap_lib.MAX_RADIANCE_WIDTH))
  if not env_sun:
    return images, EnvSuns.none()
  return _cut_suns(images, [True] * len(images), list(envs.paths), report)


def _in_common_frame(
  rgb: np.ndarray[[int, int, 3], np.dtype[np.float32]], rotation: Mat3, *, yaw0: float, device: wp.Device
) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """A map's radiance `rgb` in the maps' common frame: its rotation after the runtime yaw `yaw0`
  is a further yaw (columns rolled) or anything else (resampled)."""
  rest = _rz(-yaw0) @ rotation
  yaw, tilt = _yaw_part(rest)
  if not np.allclose(tilt, np.eye(3), atol=_TILT_TOL):
    return rotate_latlong(rgb, rest, device=device)
  if abs(yaw) > _YAW_EPS:
    return latlong.roll_yaw(rgb, yaw)
  return rgb


def _illuminance_scale(
  rgb: np.ndarray[[int, int, 3], np.dtype[np.float32]], target: float, path: str, report: LoadReport
) -> float:
  """The factor giving an upward-facing surface `target` lux from latlong radiance `rgb` (map
  frame after its tilt: row 0 is +Z): E = sum of Rec.709 Y x the row's exact cosine-weighted solid
  angle (dphi (sin^2 theta1 - sin^2 theta0) / 2 over its band, clipped at the horizon)."""
  h, w, _ = rgb.shape
  t0 = np.minimum(np.arange(h) * (np.pi / h), np.pi / 2)
  t1 = np.minimum((np.arange(h) + 1) * (np.pi / h), np.pi / 2)
  row = (2.0 * np.pi / w) * 0.5 * (np.sin(t1) ** 2 - np.sin(t0) ** 2)
  y = rgb.astype(np.float64) @ np.array(REC709_Y)
  e = float((y * row[:, None]).sum())
  if not e > 0.0:
    report.drop("PhotometricDomeLightAPI.illuminance(dark upper hemisphere)", path)
    return 1.0
  return target / e


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class SceneCameras:
  """The cameras authored in the host scene.

  Attributes:
    specs: name -> CameraSpec mounted on the camera's own pose.
    pos: (ncamera, 3) positions ("camera.pos").
    rot: (ncamera, 3, 3) rotations ("camera.rot").
    clip: (znear, zfar) of the cameras without their own (free cameras): `default_near` and
      unbounded. An authored camera's clipping range is its own (its spec's), never the free
      cameras' (a USD camera's far plane of 3 m clipped every free camera of the scene).
  """

  specs: dict[str, CameraSpec]
  pos: Rows[np.float32, 3]
  rot: Matrices[np.float32, 3, 3]
  clip: tuple[float, float]


def _unique(name: str, taken: dict[str, CameraSpec]) -> str:
  """`name`, suffixed "#k" when an earlier camera has it (a mount names one pose row)."""
  out, k = name, 1
  while out in taken:
    out, k = f"{name}#{k}", k + 1
  return out


# The default near plane of a scene without cameras: MuJoCo's (`visual.map.znear` 0.01 of the
# model's extent), at most 1 cm (larger scenes keep 1 cm: their close-ups are still seen).
_NEAR_OF_EXTENT: Final = 0.01
_MAX_DEFAULT_NEAR: Final = 0.01


def default_near(scene: Scene) -> float:
  """The near plane of the free cameras: 0.01 of the scene's extent (the diagonal of its instances'
  positions, widened by their sizes), at most 1 cm: a millimetre-scale scene is not clipped."""
  inst = scene.instances
  if not len(inst.xform):
    return _MAX_DEFAULT_NEAR
  pos = np.asarray(inst.xform, np.float64).reshape(-1, 3, 4)[:, :, 3]
  size = np.abs(np.asarray(inst.size, np.float64)).reshape(-1, 3).max(axis=1, keepdims=True)
  span = (pos + size).max(axis=0) - (pos - size).min(axis=0)
  extent = float(np.sqrt(span @ span))
  return min(_MAX_DEFAULT_NEAR, _NEAR_OF_EXTENT * extent) if extent > 0.0 else _MAX_DEFAULT_NEAR


def _rotations(frames: Matrices[np.float64, 3, 3], names: list[str], report: LoadReport) -> Matrices[np.float64, 3, 3]:
  """(n, 3, 3) camera rotations of (n, 3, 3) camera frames: the nearest rotation of each, a
  mirrored frame's with its X axis negated first (the camera keeps its view axis -Z and its up
  +Y; its image is not mirrored, reported)."""
  mirrored = np.linalg.det(frames) < 0.0
  for i in np.flatnonzero(mirrored):
    report.approx("gpu.camera.mirrored(image not mirrored)", names[i] or f"camera{i}")
  return polar(np.where(mirrored[:, None, None], frames * np.array([-1.0, 1.0, 1.0]), frames))[0]


def cameras(scene: Scene, report: LoadReport) -> SceneCameras:
  """The host scene's cameras (`SceneCameras`), each with its own clipping range.

  The apertures meet an image of another aspect by Hydra's rule (`CameraSpec.conform="fit"`,
  CameraUtilFit: the whole aperture shows, extended along its shorter side; the centre, with its
  offsets, kept). A perspective camera with an aperture offset gets pinhole intrinsics (the
  focal length and the offsets on its aperture); orthographic offsets are reported.
  """
  cams = scene.cameras
  n = len(cams)
  specs: dict[str, CameraSpec] = {}
  for i in range(n):
    cam = cams.get(i)
    name = _unique(cams.names[i] or f"camera{i}", specs)
    ortho = cam.projection == int(Projection.ORTHOGRAPHIC)
    offset = cam.horizontal_offset != 0.0 or cam.vertical_offset != 0.0
    if offset and ortho:
      report.drop("gpu.camera.aperture_offset(orthographic)", name)
    intrinsics = None
    if offset and not ortho:
      # The window centre moves by the offset (USD: +x right, +y up; MuJoCo's cy points down).
      f = float(cam.focal_length)
      intrinsics = (f, f, float(cam.horizontal_offset), -float(cam.vertical_offset))
    specs[name] = CameraSpec(
      fovy=float(cam.vertical_aperture) if ortho else cam.fovy,
      intrinsics=intrinsics,
      sensor_size=(float(cam.horizontal_aperture), float(cam.vertical_aperture)),
      conform="fit",
      projection="orthographic" if ortho else "perspective",
      znear=float(cam.near),
      zfar=float(cam.far),
      mount=name,
      exposure=float(cam.exposure),
    )
  pose = np.asarray(cams.xform, np.float64).reshape(-1, 3, 4)
  rot = _rotations(pose[:, :, :3], [cams.names[i] for i in range(n)], report)
  clip = (default_near(scene), 0.0)
  return SceneCameras(specs=specs, pos=pose[:, :, 3].astype(np.float32), rot=rot.astype(np.float32), clip=clip)
