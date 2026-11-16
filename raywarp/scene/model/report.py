"""Load report: what an adapter read, dropped and approximated, and how long each phase took."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import time
from collections.abc import Callable, Generator, Mapping
from typing import override

from raywarp.scene.model.types import Json


@dataclasses.dataclass
class LoadReport:
  """Counters of one scene load.

  Every attribute an adapter cannot represent exactly increments a named counter, so coverage is
  measurable on real assets. Names are dotted, `<source>.<attribute>`, e.g.
  `Mesh.primvars:st1` (dropped) or `mdl.OmniUe4Base.heuristic` (approximated).

  Attributes:
    source: what was loaded (a path).
    counts: totals (prims by type, meshes, triangles, materials, textures, texels, lights...).
    dropped: attribute name -> occurrences ignored.
    approximated: attribute name -> occurrences converted with a documented approximation.
    examples: counter name -> the first prim path that hit it.
    timings: phase -> seconds.
    describe: names the `where` of a `drop` / `approx` in `examples` (default `str`; an adapter
      sets its own, e.g. USD prims -> their paths). Not stored.
  """

  source: str = ""
  counts: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)
  dropped: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)
  approximated: collections.Counter[str] = dataclasses.field(default_factory=collections.Counter)
  examples: dict[str, str] = dataclasses.field(default_factory=dict)
  timings: dict[str, float] = dataclasses.field(default_factory=dict)
  describe: Callable[[object], str] = dataclasses.field(default=str, compare=False, repr=False)

  def count(self, name: str, n: int = 1) -> None:
    """Adds `n` to counter `name`."""
    self.counts[name] += n

  def drop(self, name: str, where: object = None, n: int = 1) -> None:
    """Records an ignored attribute (`where`: the object or path, kept for the first hit)."""
    self.dropped[name] += n
    self._example(name, where)

  def approx(self, name: str, where: object = None, n: int = 1) -> None:
    """Records an attribute converted approximately (`where` as in `drop`)."""
    self.approximated[name] += n
    self._example(name, where)

  def merge(self, other: LoadReport | None) -> None:
    """Adds the counters of `other` (and its examples of counters this report has none of)."""
    if other is None:
      return
    self.counts.update(other.counts)
    self.dropped.update(other.dropped)
    self.approximated.update(other.approximated)
    for name, where in other.examples.items():
      self.examples.setdefault(name, where)

  def _example(self, name: str, where: object) -> None:
    """Keeps `where` (named by `describe`) as the example of counter `name` (the first one)."""
    if where is not None and name not in self.examples:
      self.examples[name] = self.describe(where)

  @contextlib.contextmanager
  def timer(self, phase: str) -> Generator[None]:
    """Adds the wall time of the `with` block to `timings[phase]`."""
    start = time.perf_counter()
    try:
      yield
    finally:
      self.timings[phase] = self.timings.get(phase, 0.0) + time.perf_counter() - start

  def to_dict(self) -> dict[str, Json]:
    """JSON-serialisable form (without `describe`)."""
    return {
      "source": self.source,
      "counts": dict(self.counts),
      "dropped": dict(self.dropped),
      "approximated": dict(self.approximated),
      "examples": dict(self.examples),
      "timings": dict(self.timings),
    }

  @staticmethod
  def from_dict(data: Mapping[str, Json]) -> LoadReport:
    """Report from `to_dict` output (malformed entries are skipped)."""

    def entries(name: str) -> dict[str, Json]:
      """Mapping `name` of `data` ({} when absent)."""
      value = data.get(name)
      return value if isinstance(value, dict) else {}

    def counter(name: str) -> collections.Counter[str]:
      """The integer entries of mapping `name`."""
      return collections.Counter({k: v for k, v in entries(name).items() if isinstance(v, int)})

    source = data.get("source")
    return LoadReport(
      source=source if isinstance(source, str) else "",
      counts=counter("counts"),
      dropped=counter("dropped"),
      approximated=counter("approximated"),
      examples={k: v for k, v in entries("examples").items() if isinstance(v, str)},
      timings={k: float(v) for k, v in entries("timings").items() if isinstance(v, (int, float))},
    )

  def summary(self, top: int = 40) -> str:
    """Human-readable summary (the `str()` of the report)."""
    lines = [f"LoadReport({self.source})"]
    if self.counts:
      lines.append("  counts: " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())))
    if self.timings:
      lines.append("  timings: " + ", ".join(f"{k}={v * 1e3:.0f}ms" for k, v in self.timings.items()))
    for title, counter in (("dropped", self.dropped), ("approximated", self.approximated)):
      if not counter:
        continue
      lines.append(f"  {title} ({sum(counter.values())} in {len(counter)} kinds):")
      for name, n in counter.most_common(top):
        example = self.examples.get(name)
        lines.append(f"    {n:6d}  {name}" + (f"  (e.g. {example})" if example else ""))
      if len(counter) > top:
        lines.append(f"    ... {len(counter) - top} more")
    return "\n".join(lines)

  @override
  def __str__(self) -> str:
    """The summary."""
    return self.summary()
