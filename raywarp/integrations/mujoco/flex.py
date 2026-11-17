"""MuJoCo flex -> core deformables (`raywarp.core.pipeline.gpu_scene.deformables`): one `Part` per
flex, from the MjModel (topology, color, material and texture coordinates)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import mujoco
import numpy as np
import warp as wp

from raywarp._base.dims import Mask
from raywarp.core.pipeline.gpu_scene.deformables import Deformables, Part
from raywarp.scene.model.types import MaterialId

# MuJoCo's flex dimensions (`flex_dim`): curves (edges), surfaces (triangles), volumes (tetrahedra).
CURVE: Final = 1
SURFACE: Final = 2
VOLUME: Final = 3


def flex_materials(mjm: mujoco.MjModel) -> np.ndarray[[int], np.dtype[np.int32]]:
  """(nflex,) material id of every flex. MuJoCo 4 stores one per element (`nflexelem` rows, read at
  `flex_elemadr`), MuJoCo 3 one per flex; a flex whose elements differ raises (one material per
  part)."""
  matid = np.asarray(mjm.flex_matid, np.int32)
  if matid.shape[0] == mjm.nflex:
    return matid
  out = np.empty(mjm.nflex, np.int32)
  for f in range(mjm.nflex):
    adr, num = int(mjm.flex_elemadr[f]), int(mjm.flex_elemnum[f])
    ids = np.unique(matid[adr : adr + num])
    if ids.size > 1:
      raise NotImplementedError(f"flex {f}: per-element materials {ids.tolist()} (one material per flex is rendered)")
    out[f] = ids[0] if ids.size else -1
  return out


def _part(mjm: mujoco.MjModel, f: int, material: int, *, shown: bool) -> Part:
  """Flex `f` as a core part: dim 1 its edges; dim 2 its triangles and shell (boundary) edges; dim 3
  its shell (surface) triangles. A flex not `shown` (outside the rendered groups) is a curve with no
  segments: no primitive, and part i stays flex i (segmentation writes ngeom + f)."""
  dim = int(mjm.flex_dim[f]) if shown else CURVE
  edges, faces, borders = np.zeros((0, 2), np.int32), np.zeros((0, 3), np.int32), np.zeros((0, 2), np.int32)
  shell = mjm.flex_shell[int(mjm.flex_shelldataadr[f]) :].astype(np.int32)
  nshell = int(mjm.flex_shellnum[f])
  if dim == CURVE and shown:
    adr, num = int(mjm.flex_edgeadr[f]), int(mjm.flex_edgenum[f])
    edges = mjm.flex_edge[adr : adr + num].astype(np.int32).reshape(-1, 2)
  elif dim == SURFACE:
    adr, nelem = int(mjm.flex_elemdataadr[f]), int(mjm.flex_elemnum[f])
    faces = mjm.flex_elem[adr : adr + 3 * nelem].astype(np.int32).reshape(-1, 3)
    borders = shell[: 2 * nshell].reshape(-1, 2)
  elif dim == VOLUME:
    faces = shell[: 3 * nshell].reshape(-1, 3)
  texcoord, face_texcoord = _texcoords(mjm, f, dim, faces)
  r, g, b, a = (float(c) for c in mjm.flex_rgba[f])
  return Part(
    dim=CURVE if dim == CURVE else SURFACE if dim == SURFACE else VOLUME,
    vertadr=int(mjm.flex_vertadr[f]),
    vertnum=int(mjm.flex_vertnum[f]),
    radius=float(mjm.flex_radius[f]),
    rgba=(r, g, b, a),
    edges=edges,
    faces=faces,
    borders=borders,
    material=MaterialId(material),
    texcoord=texcoord,
    face_texcoord=face_texcoord,
  )


def _texcoords(
  mjm: mujoco.MjModel, f: int, dim: int, faces: np.ndarray[[int, 3], np.dtype[np.int32]]
) -> tuple[np.ndarray[[int, 2], np.dtype[np.float32]], np.ndarray[[int, 3], np.dtype[np.int32]]]:
  """(texcoord, face_texcoord) of flex `f` (empty without texture coordinates): dim 2 corners read
  `flex_elemtexcoord`, dim 3 shell corners their vertex id (`mjv_updateActiveFlex`)."""
  adr = int(mjm.flex_texcoordadr[f])
  if dim == CURVE or adr < 0:
    return np.zeros((0, 2), np.float32), np.zeros((0, 3), np.int32)
  later = [int(a) for a in mjm.flex_texcoordadr if a > adr]
  end = min(later) if later else mjm.nflextexcoord
  texcoord = np.asarray(mjm.flex_texcoord[adr:end], np.float32).reshape(-1, 2)
  if dim == SURFACE:
    eadr, nelem = int(mjm.flex_elemdataadr[f]), int(mjm.flex_elemnum[f])
    return texcoord, mjm.flex_elemtexcoord[eadr : eadr + 3 * nelem].astype(np.int32).reshape(-1, 3)
  return texcoord, faces


def shown(mjm: mujoco.MjModel, geom_groups: Sequence[int]) -> Mask[int]:
  """(nflex,) whether each flex is in the rendered `geom_groups` (by `flex_group`)."""
  return np.isin(mjm.flex_group, list(geom_groups))


def deformables(mjm: mujoco.MjModel, mjd: mujoco.MjData, rendered: Mask[int], device: wp.Device) -> Deformables:
  """The model's flex objects at the vertex positions of `mjd` (after `mj_forward`), on `device`;
  `rendered`: (nflex,) the flexes shown (`shown`; the others have no primitive)."""
  materials = flex_materials(mjm)
  parts = [_part(mjm, f, int(materials[f]), shown=bool(rendered[f])) for f in range(mjm.nflex)]
  return Deformables.create(parts, np.asarray(mjd.flexvert_xpos, np.float32).reshape(-1, 3), device=device)
