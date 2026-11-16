"""`SceneOptions`: the options every adapter's scene shares (`rw.SceneOptions`)."""

from __future__ import annotations

import dataclasses
from typing import Final, final

import numpy as np

from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, TextureStorage


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class SceneOptions:
  """How a source becomes a GpuScene, the same for every adapter (`rw.<adapter>.scene`,
  `rw.upload`); what is specific to one source is that adapter's `Options`.

  Attributes:
    textures: how the material textures are stored (`TextureStorage`; the default stores 8-bit
      color as 8-bit sRGB on CUDA, `EXACT_STORAGE` keeps every texel). Its `max_size` is the one
      texture size cap: adapters that decode images (USD, Newton) decode at most that side, and
      the upload stores at most that side.
    emissive_lights: fit the panel lights of emissive surfaces (`scene.lighting.emissive`; a
      renderer switches them with `lighting.emissive_lights`). None: the source format's convention
      (`fit_emissive`): fitted for host scenes (USD, Newton: an emissive surface's radiance
      lights the scene), not for MuJoCo models, whose emission lights nothing (an explicit True
      raises there). False skips the fit: for scenes only ever rendered with
      `lighting.emissive_lights=False`, the setting then has no panels to switch.
    env_sun: cut the sun out of every lighting environment map that has one and light it as a
      directional light with shadow rays and its disc's size, on in the worlds that show its map
      (`env.id`), turned by their `env.yaw` and scaled by their `env.intensity`; the map keeps
      the rest of the light (energy conserved), the background draws the disc back over it. A
      sun is the connected region of texels above 50 x the map's median radiance around its
      brightest texel, within 5 degrees of its centre, holding at least 30 % of the irradiance
      of a surface facing it (`latlong.extract_sun`): sunny skies have one, overcast skies and
      studios with several softboxes none. Every adapter (USD DomeLight, MuJoCo image lights
      and `Lighting.envmap`, glTF and host `Environment` records). A rendering decision: the
      sun is no light of the light tables. False keeps the sun in the map (lit by the SH /
      prefiltered map and the GI, shadowed only by their occlusion), the upload-time switch (a
      run-time one would keep a second copy of every map, its prefiltered stack and SH).

  Raises:
    TypeError: for `textures` that is not a TextureStorage, an `emissive_lights` that is not
      None / True / False (0 would fit the panels: only False skips them) or a non-bool `env_sun`.
  """

  textures: TextureStorage = DEFAULT_STORAGE
  emissive_lights: bool | None = None
  env_sun: bool = True

  def __post_init__(self) -> None:
    """Checks the field types; stores a NumPy bool as the Python bool (`fit_emissive` compares
    with `is`)."""
    if not isinstance(self.textures, TextureStorage):
      raise TypeError(f"SceneOptions.textures must be a TextureStorage, got {type(self.textures).__name__}")
    for name in ("emissive_lights", "env_sun"):
      value = getattr(self, name)
      if isinstance(value, np.bool_):
        object.__setattr__(self, name, bool(value))
      elif not (isinstance(value, bool) or (value is None and name == "emissive_lights")):
        raise TypeError(
          f"SceneOptions.{name} must be True or False{' or None' if name == 'emissive_lights' else ''}, got {value!r}"
        )

  @property
  def fit_emissive(self) -> bool:
    """Whether a host scene's upload fits emissive panel lights (`emissive_lights`, None: yes)."""
    return self.emissive_lights is not False

  @property
  def max_texture_size(self) -> int | None:
    """The texture size cap of image decoding (None: no cap)."""
    return self.textures.max_size or None


# The default options (a shared immutable value: argument defaults).
DEFAULT_SCENE_OPTIONS: Final = SceneOptions()
