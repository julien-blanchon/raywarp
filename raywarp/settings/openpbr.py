"""The `openpbr` settings group: which local OpenPBR lobes are compiled in."""

from __future__ import annotations

import dataclasses
from typing import Literal, final

from raywarp.settings.fields import Tri, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenPBR:
  """Authored local lobes; "auto" compiles only the lobes some material uses or a per-world field
  declares.

  Attributes:
    diffuse_roughness: True forces EON preparation, False selects Lambert, "auto" detects authored
      constants and declared per-world roughness (C).
    layers: the layered lobes (anisotropic GGX, fuzz, the OpenPBR coat with its own normal and
      IOR, detail normals: `Features.layered`): "auto" compiles each one some material uses or a
      per-world field declares; False renders the base model only (isotropic, no fuzz, glTF-law
      coat with the base normal, no detail map), e.g. to measure their cost (C). There is no True:
      a lobe no material uses has nothing to render.
  """

  diffuse_roughness: Tri = "auto"
  layers: Literal["auto", False] = "auto"

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "openpbr")
