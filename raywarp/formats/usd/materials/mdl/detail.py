"""A distilled normal that adds a detail normal map to a base one, split back into the two maps.

MDL materials add a detail normal map by reoriented normal mapping (Barré-Brisebois and Hill's
RNM, `add_detail_normal(nd, n)` of `nvidia::distilling_support` and OmniPBR_ClearCoat, OmniPBR's
`detail_normalmap_texture` with its own `detail_texture_*` tiling):

    t = n + (0, 0, 1),  u = nd x (-1, -1, 1),  result = t dot(t, u) / t.z - u

in the tangent frame. The distiller keeps the call or inlines its arithmetic, so the normal is
recognised by what it computes, not by its shape: among the expression's subexpressions that read
one normal map (a texture's affine RGB with an outward z, `Writer.field`'s convention), a pair
(n, nd) is the split when the expression equals RNM(n, nd) at random texels. The core then
combines `geometry_normal` (n) and `geometry_detail_normal` (nd, its own UV transform) by the
same RNM, and no bake is needed.
"""

from __future__ import annotations

import itertools
from typing import Final

import numpy as np

from raywarp.formats.usd.materials.mdl.bake_eval import Numeric, TexelPlanes, leaves
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.values import (
  RGB,
  NeedsBakeError,
  SlotValue,
  TexTerm,
  UnsupportedError,
  analyse,
  comps,
  pad4,
)
from raywarp.scene.model import normal_map
from raywarp.scene.model.types import CHANNEL_B, CHANNEL_RGB

type Vectors = np.ndarray[tuple[int, int], np.dtype[np.float64]]  # (3, samples)

# Random texels each candidate pair is checked at, and the agreement it needs (float64 sums of
# float32 constants).
_SAMPLES: Final = 16
_MATCH_TOL: Final = 1e-5
# Texels of a plausible normal map: x, y near the middle, z near 1 (RNM divides by 1 + z).
_XY_TEXELS: Final = (0.3, 0.7)
_Z_TEXELS: Final = (0.85, 1.0)
# A base and a detail map.
_MAPS: Final = 2
# Candidate reads of one map at most (the map, and the RNM's own affine steps of it).
_MAX_READS: Final = 8


def rnm(base: Vectors, detail: Vectors) -> Vectors:
  """Reoriented normal mapping of tangent-space normals (3, samples), as MDL's
  `add_detail_normal(detail, base)` before its normalisation."""
  t = base + np.array([[0.0], [0.0], [1.0]])
  u = detail * np.array([[-1.0], [-1.0], [1.0]])
  return t * np.sum(t * u, axis=0) / t[2] - u


def split(dag: Dag, path: str) -> tuple[SlotValue, SlotValue] | None:
  """(base, detail) normal maps whose RNM the normal at `path` is (read as a direction), None
  when it is one map, a constant, or anything else."""
  try:
    cs = comps(dag.value(path, direction=True))
  except UnsupportedError:
    return None
  try:
    analyse(cs, RGB)
  except NeedsBakeError:
    pass  # several maps: a candidate
  except UnsupportedError:
    return None
  else:
    return None  # one map or a constant: `Writer.field` binds it
  reads = _reads(dag, path)
  if len(reads) != _MAPS:
    return None
  try:
    sources = leaves(cs)
  except UnsupportedError:
    return None
  if not all(isinstance(s, TexTerm) for s in sources):
    return None
  planes = _planes([s for s in sources if isinstance(s, TexTerm)])
  value = _directions(np.stack([Numeric(planes, (1, _SAMPLES))(c)[0] for c in cs[:RGB]]))
  for first, second in itertools.product(*reads.values()):
    for base, detail in ((first, second), (second, first)):
      expected = rnm(_decoded(base, planes), _decoded(detail, planes))
      if np.max(np.abs(_directions(expected) - value)) < _MATCH_TOL:
        return base, detail
  return None


def _reads(dag: Dag, path: str) -> dict[tuple[object, ...], list[SlotValue]]:
  """The normal-map reads below `path` by map (texture, UV map, wrap): subexpressions whose
  value is one texture's affine RGB with every scale nonzero and an outward z (the map itself
  sits below the RNM's own affine steps of it: every read is kept)."""
  out: dict[tuple[object, ...], list[SlotValue]] = {}
  seen: set[tuple[int, int, int, int]] = set()
  todo = [path]
  while todo:
    p = todo.pop()
    key = dag.hash(p)
    if key in seen:
      continue
    seen.add(key)
    sv = _read(dag, p)
    if sv is not None and sv.tex is not None:
      found = out.setdefault((sv.tex, sv.uv, sv.wrap), [])
      if sv not in found and len(found) < _MAX_READS:
        found.append(sv)
    todo.extend(dag.node(p)[1].values())
  return out


def _read(dag: Dag, path: str) -> SlotValue | None:
  """The normal-map read at `path` (None: none)."""
  try:
    sv = analyse(dag.value(path, direction=True), RGB)
  except (NeedsBakeError, UnsupportedError):
    return None
  if sv.tex is None or sv.channel != CHANNEL_RGB or len(sv.scale) != RGB or not all(sv.scale):
    return None
  return sv if normal_map.outward(pad4(sv.scale, 1.0), pad4(sv.bias, 0.0)) else None


def _planes(sources: list[TexTerm]) -> TexelPlanes:
  """Random texels of every lookup (seeded: the split is deterministic)."""
  rng = np.random.default_rng(0)
  planes: TexelPlanes = {}
  for s in sorted(sources, key=lambda t: (t.tex.path, t.channel, t.uv.uv_set, t.uv.rows, t.wrap)):
    low, high = _Z_TEXELS if s.channel == CHANNEL_B else _XY_TEXELS
    planes[s] = rng.uniform(low, high, (1, _SAMPLES))
  return planes


def _decoded(sv: SlotValue, planes: TexelPlanes) -> Vectors:
  """A map read's tangent-space normals at the planes' texels (3, samples)."""
  rows: list[np.ndarray[[int], np.dtype[np.float64]]] = []
  for k in range(RGB):
    texel = None if sv.tex is None or sv.uv is None else planes.get(TexTerm(sv.tex, k, sv.uv, sv.wrap))
    if texel is None:
      return np.full((RGB, _SAMPLES), np.nan)
    rows.append(sv.scale[k] * texel[0] + sv.bias[k])
  return np.stack(rows)


def _directions(v: Vectors) -> Vectors:
  """Unit vectors of (3, samples) columns."""
  return v / np.maximum(np.linalg.norm(v, axis=0), 1e-30)
