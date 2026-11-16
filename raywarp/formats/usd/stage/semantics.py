"""Semantic labels of USD prims: the class an instance shows in semantic segmentation.

Two encodings, the nearest labelled ancestor (the prim included) winning:
* UsdSemantics `LabelsAPI`: `semantics:labels:<taxonomy>` token arrays; taxonomy "class" first,
  else the first taxonomy; the first token.
* Isaac Sim's legacy `SemanticsAPI`: `semantic:<name>:params:semanticType` = "class" with
  `semantic:<name>:params:semanticData`.

Lookups go by the instance's prim path (instance proxies resolve on the stage); a PointInstancer
instance's path ("/PI[3]/...") resolves from the instancer up (labels authored inside its
prototypes are not seen). Results are cached per path.
"""

from __future__ import annotations

import re
from typing import Final, final

from pxr import Sdf, Usd

_LABELS: Final = "semantics:labels:"
_LEGACY: Final = re.compile(r"^semantic:([^:]+):params:semanticType$")
# A PointInstancer instance index in a path.
_INDEX: Final = re.compile(r"\[\d+\]")


def applies(prim: Usd.Prim) -> bool:
  """Whether the prim applies a semantics schema (UsdSemantics' or Isaac's legacy one)."""
  return any(s.startswith(("SemanticsLabelsAPI", "SemanticsAPI")) for s in prim.GetAppliedSchemas())


def prim_label(prim: Usd.Prim) -> str:
  """The prim's own semantic class label ("" for none)."""
  by_taxonomy: dict[str, str] = {}
  for attr in prim.GetAttributes():
    name = attr.GetName()
    if name.startswith(_LABELS):
      value = attr.Get()
      if value:
        by_taxonomy[name[len(_LABELS) :]] = str(value[0])
      continue
    m = _LEGACY.match(name)
    if m and attr.Get() == "class":
      data = prim.GetAttribute(f"semantic:{m.group(1)}:params:semanticData").Get()
      if data:
        by_taxonomy.setdefault("class", str(data))
  if "class" in by_taxonomy:
    return by_taxonomy["class"]
  return next(iter(by_taxonomy.values()), "")


@final
class Labels:
  """Semantic label of instance paths (nearest labelled ancestor; cached)."""

  def __init__(self, stage: Usd.Stage) -> None:
    """Lookups on `stage`."""
    self.stage = stage
    self._cache: dict[str, str] = {}

  def __call__(self, path: str) -> str:
    """The label of the instance at `path` ("" for none)."""
    if path in self._cache:
      return self._cache[path]
    clean = _INDEX.sub("", path)
    label = ""
    prim = self.stage.GetPrimAtPath(clean) if Sdf.Path.IsValidPathString(clean) else None
    if prim and prim.IsValid():
      label = prim_label(prim)
    if not label:
      parent = clean.rsplit("/", 1)[0]
      label = self(parent) if parent else ""
    self._cache[path] = label
    return label
