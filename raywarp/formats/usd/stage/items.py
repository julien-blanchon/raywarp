"""The stage reader's intermediate representation: what one walk of a (sub)tree produces.

A walk turns a subtree into a flat list of `Item`s, each with its transform relative to the
subtree's root (`local`) and its prim-path suffix. Emitting the list under K parent transforms
places every item K times with one batched matrix product (`reader.StageReader.emit`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, final

import numpy as np
from pxr import Usd

from raywarp.formats.usd.stage.camera import StageCamera
from raywarp.formats.usd.stage.physical import PhotometricLight
from raywarp.formats.usd.util import Color, Matrices4, Matrix4
from raywarp.scene.model.lights import EnvironmentLight, Light
from raywarp.scene.model.types import InstanceKind, MaterialId, MeshId

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.context import MeshNeeds  # shade imports this module

# Material id of a prototype item whose binding comes from the instance referencing it.
INHERIT: Final = MaterialId(-2)

type Ids = np.ndarray[[int], np.dtype[np.int64]]


@final
@dataclasses.dataclass(frozen=True)
class Display:
  """The fallback material of an unbound gprim: displayColor / displayOpacity (None: unset).

  `vertex`: the mesh carries per-vertex displayColor as vertex colors (white base color).
  """

  color: Color | None = None
  opacity: float | None = None
  vertex: bool = False


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Shape:
  """A gprim: a mesh part or an analytic primitive."""

  kind: InstanceKind
  mesh: MeshId  # NO_MESH: analytic
  material: MaterialId  # NO_MATERIAL (unbound: `display`) or INHERIT
  display: Display
  size: tuple[float, float, float]
  flags: int
  local: Matrix4
  suffix: str
  body: str | None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Instance:
  """A native instance: its prototype's items placed at `local`."""

  prototype: Usd.Prim
  local: Matrix4
  suffix: str
  material: MaterialId  # the instance's own binding, or INHERIT
  needs: MeshNeeds  # the mesh attributes its binding reads (the prototype's meshes read them)
  body: str | None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PointInstancer:
  """A PointInstancer: prototype `prototypes[indices[i]]` at `xforms[i]` (relative to `local`)."""

  prototypes: list[Usd.Prim]
  indices: Ids
  xforms: Matrices4  # prototype root xform included
  ids: Ids  # instance ids (path suffixes `[id]`)
  local: Matrix4
  suffix: str
  material: MaterialId
  needs: MeshNeeds  # the mesh attributes `material` reads
  body: str | None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StageLight:
  """A UsdLux light or dome, read once."""

  spec: Light | EnvironmentLight | PhotometricLight
  local: Matrix4
  suffix: str
  body: str | None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CameraItem:
  """A UsdGeom.Camera, read once."""

  spec: StageCamera
  local: Matrix4
  suffix: str


type Item = Shape | Instance | PointInstancer | StageLight | CameraItem
