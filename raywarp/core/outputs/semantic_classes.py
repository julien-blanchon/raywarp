"""Semantic classes of a Renderer's instances: which class id each instance writes to the
"semantic" output (`Outputs.semantic`).

Class ids: 0 background (nothing hit), 1 unlabelled, then one id per label in the order labels
first appear: the scene's authored labels (`GpuScene.instance_labels`, UsdSemantics), then the
labels of `Semantics.assign`'s rules, which override the authored class of the instances they
match. One label list serves every world: a class id means the same label in every world.

The table is (1, ninstance) int32 on the device, shared by every world, until a first per-world
assignment (`assign(rules, worlds)`) replaces it, once, by a (nworld, ninstance) table; it is
written in place afterwards (a captured graph keeps reading it) from its host copy, which only
`assign` writes (no device read: no synchronisation); changing it never recompiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.worlds import sharing

if TYPE_CHECKING:
  from raywarp._base.dims import Ids, Mask

# Class id of instances no label names (0 is the background: nothing hit).
UNLABELLED: Final = 1
# Class names of ids 0 and 1: rules naming them raise, authored labels naming them are unlabelled.
RESERVED: Final = ("background", "unlabelled")


@final
class Semantics:
  """The semantic class of every instance (see the module doc).

  Attributes:
    ids: (1 or nworld, max(ninstance, 1)) class id per (world, instance), read by the kernels
      (`OutputTable.semantic_id`).
    labels: label names by class id ("background", "unlabelled", then the assigned labels).
  """

  def __init__(self, names: Sequence[str], authored: Sequence[str] = (), *, device: wp.DeviceLike) -> None:
    """The instances of `names` with their `authored` labels ("", none or a reserved name:
    unlabelled); the id table on `device`.

    Raises:
      ValueError: for authored labels that are not one per instance (or none).
    """
    if authored and len(authored) != len(names):
      raise ValueError(f"Semantics: {len(authored)} authored labels for {len(names)} instances (one per instance, or none)")
    self._names = list(names)
    self._authored = list(authored)
    # The table's host copy (what `assign` wrote; the device table is never read back).
    self._host = np.full((1, max(len(names), 1)), UNLABELLED, np.int32)
    self.ids: wp.array[int, [int, int]] = wp.array(self._host, dtype=int, device=device)
    self.labels: tuple[str, ...] = RESERVED
    if any(self._authored):
      self.assign({})

  def assign(self, rules: Mapping[str, sharing.Patterns], worlds: Mask[int] | None = None) -> tuple[str, ...]:
    """Gives the instances matched by each label's patterns that label (later labels win).

    `rules`: label -> patterns (`PerWorld` patterns: globs on the instance names or indices);
    instances no pattern matches keep their authored label, else are unlabelled. `worlds`: None
    replaces every world's assignment and restarts the class numbering; a (nworld,) bool mask
    replaces the masked worlds' only, keeps every existing class id and appends new labels (the
    first such call allocates the (nworld, ninstance) table: a new array).

    Returns:
      The labels by class id.

    Raises:
      TypeError: for a label that is not a str.
      ValueError: for a pattern that matches no instance, or an empty or reserved label name.
    """
    labels = list(RESERVED if worlds is None else self.labels)
    row = self._row(rules, labels)
    if worlds is None:
      self._host[:] = row
      self.ids.assign(self._host)
    elif self._host.shape[0] == 1:
      self._host = np.repeat(self._host, len(worlds), axis=0)
      self._host[worlds] = row
      self.ids = wp.array(self._host, dtype=int, device=self.ids.device)
    else:
      self._host[worlds] = row
      self.ids.assign(self._host)
    self.labels = tuple(labels)
    return self.labels

  def _row(self, rules: Mapping[str, sharing.Patterns], labels: list[str]) -> Ids[int]:
    """(max(ninstance, 1),) class ids of the authored labels overridden by `rules`, adding new
    labels to `labels`."""
    ids = np.full(max(len(self._names), 1), UNLABELLED, np.int32)
    for i, label in enumerate(self._authored):
      # Reserved names are never classes: an authored one counts as unlabelled.
      if label and label not in RESERVED:
        ids[i] = _class_id(labels, label)
    for label, patterns in rules.items():
      if not isinstance(label, str):
        raise TypeError(f"semantic labels must be names (str), got {label!r}")
      if label in RESERVED:
        raise ValueError(f"semantic label {label!r} is reserved (class ids 0 and 1)")
      if not label:
        raise ValueError("semantic labels must be non-empty names")
      mask = sharing.match(self._names, patterns, "set_semantics")
      ids[: len(mask)][mask] = _class_id(labels, label)
    return ids


def _class_id(labels: list[str], label: str) -> int:
  """The class id of `label`, appended to `labels` when new."""
  if label not in labels:
    labels.append(label)
  return labels.index(label)


def world_mask(worlds: Sequence[int] | Mask[int], nworld: int) -> Mask[int]:
  """(nworld,) bool mask of `worlds`: world indices, or a (nworld,) bool mask.

  Raises:
    ValueError: for an index out of range, a mask of another length, or no world.
  """
  a = np.asarray(worlds)
  if a.dtype == np.bool_:
    if a.shape != (nworld,):
      raise ValueError(f"a world mask is ({nworld},) bool, got {a.shape}")
    mask = a.copy()
  else:
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
      raise ValueError(f"worlds are world indices or a ({nworld},) bool mask, got {a.dtype} {a.shape}")
    if a.size and (a.min() < 0 or a.max() >= nworld):
      raise ValueError(f"world indices must be in [0, {nworld}), got {a.tolist()}")
    mask = np.zeros(nworld, bool)
    mask[a.astype(np.int64)] = True
  if not mask.any():
    raise ValueError("worlds selects no world")
  return mask


class _Named(Protocol):
  """A named model element (`mujoco.MjModel.body(i)`)."""

  @property
  def name(self) -> str:
    """The element's name ("" without one)."""
    ...


class BodyModel(Protocol):
  """What `body_instances` reads of a MuJoCo model (`mujoco.MjModel`)."""

  @property
  def nbody(self) -> int:
    """Number of bodies."""
    ...

  @property
  def body_parentid(self) -> Ids[int]:
    """(nbody,) parent body of each body (the world body 0 is its own)."""
    ...

  @property
  def geom_bodyid(self) -> Ids[int]:
    """(ngeom,) body of each geom."""
    ...

  def body(self, i: int, /) -> _Named:
    """Body `i`."""
    ...


def body_instances(model: BodyModel, patterns: str | Sequence[str], *, subtree: bool = True) -> list[int]:
  """The instance indices (geom ids: a MuJoCo scene has one instance per geom) of the bodies whose
  names match `patterns` (`fnmatch` globs), with their descendants' unless `subtree` is False: a
  `set_semantics` rule by body, e.g. `r.set_semantics({"robot": body_instances(mjm, "base")})`.

  Raises:
    ValueError: for a pattern that matches no body, or bodies without geoms.
  """
  names = [model.body(b).name for b in range(model.nbody)]
  chosen = np.asarray(sharing.match(names, patterns, "body_instances"), bool)
  parent = np.asarray(model.body_parentid)
  if subtree:
    # Parents come before their children in MuJoCo's body order.
    for b in range(1, model.nbody):
      chosen[b] |= chosen[parent[b]]
  geoms = np.flatnonzero(chosen[np.asarray(model.geom_bodyid)])
  if geoms.size == 0:
    raise ValueError(f"body_instances: the bodies matching {patterns!r} have no geoms")
  return [int(g) for g in geoms]
