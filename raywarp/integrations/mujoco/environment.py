"""The MuJoCo model's environment: skybox background, image lights, or the caller's maps.

MuJoCo's semantics (its OpenGL renderer and Filament's `model_lights.cc`): the skybox texture is
the background only and lights nothing; image-based lighting comes from a
`<light type="image" texture=...>`, whose texels times its `intensity` are the radiance [nits]
(Filament's IndirectLight intensity). So:

* `lighting.envmap` given: those maps light the scene and fill the background
  (`lighting.envmap_intensity`, default display-relative 1 / exposure).
* else, an image light with intensity > 0 and a texture: its texture (cube -> equirect) lights the
  scene, radiance texel x intensity, and fills the background; a skybox with another texture is
  shown instead (a second, background-only map: display-relative, not prefiltered).
* else, the skybox (`use_skybox`): background only (not prefiltered), display-relative (a texel
  of 1 shows white).

Every map is decoded in its texture's color space (`texture_radiance`). With `env_sun`, the sun
of each lighting map that has a dominant one is cut out as a shadowed directional light
(`core.pipeline.upload.environment.cut_suns`, `GpuEnvironment.suns`): shaded after the model's
lights, following each world's `env.id` / `env.yaw` / `env.intensity`.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from raywarp._base.types import LightType
from raywarp.core.pipeline.gpu_scene.gpu_scene import EnvSuns, GpuEnvironment
from raywarp.core.pipeline.upload.environment import cut_suns
from raywarp.core.post.grading.color import srgb_to_linear
from raywarp.core.shade.environment import envmap as envmap_lib
from raywarp.core.shade.environment.envmap import Image
from raywarp.integrations.mujoco import skybox
from raywarp.integrations.mujoco.textures import MjTextureId, loaded, rgba_texels, srgb
from raywarp.scene.lighting.rig import Lighting, envmap_radiance
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_ENVIRONMENT, EnvironmentId

type Radiance = np.ndarray[[int, int, 3], np.dtype[np.float32]]


def _image_light(mjm: mujoco.MjModel) -> int:
  """The first active image light with a texture and intensity > 0, else -1."""
  ids = np.flatnonzero(
    (np.asarray(mjm.light_type) == int(LightType.IMAGE))
    & (np.asarray(mjm.light_texid) >= 0)
    & (np.asarray(mjm.light_intensity) > 0.0)
    & np.asarray(mjm.light_active, bool)
  )
  return int(ids[0]) if len(ids) else -1


def texture_radiance(mjm: mujoco.MjModel, tex_id: MjTextureId, *, device: wp.Device, width: int = 1024) -> Radiance:
  """Texture `tex_id` as a linear equirect image: cube / skybox textures resampled (on `device`,
  `width` texels wide), a 2D texture read as an equirect map. Display colors are decoded: an
  environment is a color role, so its texels are sRGB unless the texture says linear (its
  `colorspace`, or a by-reference file's resolved one: `textures.loaded`, `textures.srgb`)."""
  pixels, space = loaded(mjm, tex_id)
  encoded = srgb(space, color=True)
  if int(mjm.tex_type[tex_id]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
    return skybox.cube_equirect(pixels, srgb=encoded, device=device, width=width)
  rgb = rgba_texels(pixels)[..., :3].astype(np.float32) / 255.0
  return (srgb_to_linear(rgb) if encoded else rgb).astype(np.float32)


def environment(  # noqa: PLR0913  # the model, its lighting and keyword-only options
  mjm: mujoco.MjModel,
  scene_lighting: Lighting,
  report: LoadReport,
  *,
  use_skybox: bool,
  env_sun: bool,
  ambient: tuple[float, float, float],
  device: wp.Device,
) -> GpuEnvironment:
  """The scene environment on `device` (module docstring); `ambient` lights it without a lit map;
  `env_sun`: the lighting maps' suns cut out (reported in `report`)."""
  images: list[Image] = [envmap_radiance(item) for item in scene_lighting.envmaps]
  paths = [f"lighting.envmap[{i}]" for i in range(len(images))]
  lighting = [True] * len(images)
  intensity = scene_lighting.env_intensity
  background = NO_ENVIRONMENT
  light = -1 if scene_lighting.replace else _image_light(mjm)
  if not images and light >= 0:
    light_tex = MjTextureId(int(mjm.light_texid[light]))
    images = [texture_radiance(mjm, light_tex, device=device)]
    paths = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_LIGHT, light) or f"light{light}"]
    lighting = [True]
    intensity = float(mjm.light_intensity[light])
    sky = skybox.skybox_id(mjm)
    if use_skybox and sky >= 0 and sky != light_tex:
      # The skybox shows behind the scene, the image light lights it (two maps).
      images.append(texture_radiance(mjm, sky, device=device))
      lighting.append(False)
      paths.append("skybox")
      background = EnvironmentId(1)
  elif not images and use_skybox:
    sky = skybox.skybox_id(mjm)
    images = [texture_radiance(mjm, sky, device=device)] if sky >= 0 else []
    lighting = [False] * len(images)
    paths = ["skybox"] * len(images)
  suns = EnvSuns.none()
  if env_sun and any(lighting):
    # At the radiance texture's width (`build_environments` stores no more).
    rgb = [envmap_lib.downsample(np.asarray(im, np.float32)[..., :3], envmap_lib.MAX_RADIANCE_WIDTH) for im in images]
    cut, suns = cut_suns(rgb, lighting, paths, report, device=device)
    images = [*cut]
  return GpuEnvironment(
    maps=envmap_lib.build_environments(images, lighting, device=device)
    if images
    else envmap_lib.empty_environments(device=device),
    lit=any(lighting),
    has_maps=bool(images),
    intensity=intensity,
    yaw=float(np.radians(scene_lighting.envmap_rotation)),
    ambient=ambient,
    background=background,
    background_intensity=1.0 / scene_lighting.exposure,
    suns=suns,
  )
