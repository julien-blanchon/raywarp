"""The settings groups of the look: `surfaces` (textures, normals, culling, lit or unlit shading)
and `post` (the display transform; the exposure: `settings.Exposure`)."""

from __future__ import annotations

import dataclasses
from typing import Literal, final

from raywarp.settings.fields import ToneMapperName, Tri, runtime_field, setting, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Surfaces:
  """Surface detail.

  Attributes:
    textures: sample material textures; "auto" = on when the scene binds any (C).
    backface_culling: drop hits whose normal faces away from the ray (C).
    anisotropy: texture taps along the footprint's major axis at grazing angles, at most (1:
      one isotropic tap sized by the footprint's geometric mean; N: up to N taps, each sized by
      the minor axis, against moiré towards the horizon; 8 by default) (C).
    shading: "pbr" (lights, shadows, environment, reflections, GI) or "unlit": the base color
      (material x instance x vertex colors x base-color texture) shown as is, display-relative
      (through the tone mapper), plus the emission (lamps stay visible), no light, shadow,
      environment or reflection term: the fastest shading (1.5-13x), for policies that need
      shape and color only (C).

  Meshes with vertex normals are always shaded with them interpolated.
  """

  textures: Tri = "auto"
  backface_culling: bool = True
  anisotropy: int = setting(8, ge=1, le=16)
  shading: Literal["pbr", "unlit"] = "pbr"

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "surfaces")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Post:
  """The display transform (the exposure before it: `settings.Exposure`).

  Attributes:
    tone_mapper: "pbr_neutral" (Khronos PBR Neutral: keeps base colors; the default),
      "aces_rtx" (Omniverse RTX's ACES approximation: `omniverse`'s, close to RTX renders),
      "aces" or "clamp" (R: a uniform branch per pixel; changing it never recompiles).
  """

  tone_mapper: ToneMapperName = runtime_field("pbr_neutral")  # noqa: RUF009  # a dataclasses.field

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "post")
