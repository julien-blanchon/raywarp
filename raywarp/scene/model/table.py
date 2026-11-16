"""Host-scene tables: numpy structure-of-arrays, memory-mappable (`Table`, `StringTable`,
`StringPool`).

Every table is a dataclass of numpy columns described by its `SCHEMA` (column -> dtype,
trailing shape), one row per object, plus shared pools (mesh data, the texel blob) that rows
address with `*_adr` / `*_num` (MuJoCo's convention). String columns (`STRINGS`) are stored as
one UTF-8 blob plus offsets (`StringTable`) so that they map too. The tables themselves are in
`tables.py` (geometry, materials, images) and `lights.py` (lights, environments, cameras).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import ClassVar, Final, Self, final

import numpy as np
import numpy.typing as npt

from raywarp.scene.model.types import Array, Column

# A column of a table: (dtype, trailing shape).
type ColumnSpec = tuple[np.dtype, tuple[int, ...]]

I32: Final = np.dtype(np.int32)
U32: Final = np.dtype(np.uint32)
U8: Final = np.dtype(np.uint8)
I8: Final = np.dtype(np.int8)
F32: Final = np.dtype(np.float32)
I64: Final = np.dtype(np.int64)
# OpenPBR / UsdLux inputs of non-linear derivations are float64; extra-texture inputs int16.
I16: Final = np.dtype(np.int16)
F64: Final = np.dtype(np.float64)


@final
class StringTable:
  """Immutable list of strings stored as one UTF-8 blob plus offsets (memory-mappable)."""

  def __init__(self, blob: Column[np.uint8], offsets: Column[np.int64]) -> None:
    """Table over `blob` (UTF-8 bytes) and `offsets` (n + 1 byte offsets: string i is [o[i],
    o[i+1]))."""
    self.blob = blob
    self.offsets = offsets

  @staticmethod
  def from_list(strings: Sequence[str]) -> StringTable:
    """Table holding `strings`."""
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
      np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return StringTable(np.frombuffer(b"".join(encoded), dtype=np.uint8).copy(), offsets)

  def __len__(self) -> int:
    """Number of rows."""
    return len(self.offsets) - 1

  def __getitem__(self, i: int) -> str:
    """String `i` (negative indices count from the end; IndexError outside the table)."""
    if i < 0:
      i += len(self)
    if not 0 <= i < len(self):
      raise IndexError("string table index out of range")
    return bytes(self.blob[self.offsets[i] : self.offsets[i + 1]]).decode("utf-8")

  def __iter__(self) -> Iterator[str]:
    """Iterates over the strings."""
    return iter(self.tolist())

  def tolist(self) -> list[str]:
    """All strings (decoded at once)."""
    data = bytes(self.blob)
    o = self.offsets.tolist()
    return [data[o[i] : o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]

  def index(self, s: str) -> int:
    """Index of string `s`."""
    return self.tolist().index(s)


@dataclasses.dataclass
class Table:
  """A table of numpy columns described by `SCHEMA`; each `STRINGS` name is two columns.

  String column `name` is stored as `<name>_blob` / `<name>_offsets` (array keys `<name>.blob` /
  `<name>.offsets`) and read through `strings(name)` (each table names its own properties).
  """

  SCHEMA: ClassVar[dict[str, ColumnSpec]] = {}
  STRINGS: ClassVar[tuple[str, ...]] = ()

  @classmethod
  def from_columns(
    cls, columns: Mapping[str, npt.ArrayLike | None], strings: Mapping[str, Sequence[str]] | None = None
  ) -> Self:
    """Table from column values (missing or empty columns are empty).

    Raises:
      KeyError: for a column or string column the table does not have (a stale name).
    """
    given = {} if strings is None else strings
    unknown = sorted((columns.keys() - cls.SCHEMA.keys()) | (given.keys() - set(cls.STRINGS)))
    if unknown:
      raise KeyError(f"{cls.__name__} has no columns {unknown}")
    kwargs: dict[str, Array] = {}
    for name, (dtype, shape) in cls.SCHEMA.items():
      value = columns.get(name)
      a = np.zeros((0, *shape), dtype) if value is None else np.asarray(value, dtype)
      kwargs[name] = np.zeros((0, *shape), dtype) if a.size == 0 else a.reshape(-1, *shape)
    for name in cls.STRINGS:
      table = StringTable.from_list(given.get(name, []))
      kwargs[f"{name}_blob"], kwargs[f"{name}_offsets"] = table.blob, table.offsets
    return cls(**kwargs)

  @classmethod
  def from_arrays(cls, arrays: Mapping[str, Array]) -> Self:
    """Table viewing `arrays` (keys as `arrays()`; no copy)."""
    kwargs = {name: arrays[name] for name in cls.SCHEMA}
    for name in cls.STRINGS:
      kwargs[f"{name}_blob"], kwargs[f"{name}_offsets"] = arrays[f"{name}.blob"], arrays[f"{name}.offsets"]
    return cls(**kwargs)

  def __len__(self) -> int:
    """Number of rows (the length of the first column)."""
    return len(vars(self)[next(iter(self.SCHEMA))])

  def arrays(self) -> dict[str, Array]:
    """Every array, keyed by column (`<name>.blob` / `<name>.offsets` for strings)."""
    fields = vars(self)
    out = {name: fields[name] for name in self.SCHEMA}
    for name in self.STRINGS:
      out[f"{name}.blob"], out[f"{name}.offsets"] = fields[f"{name}_blob"], fields[f"{name}_offsets"]
    return out

  def strings(self, name: str) -> StringTable:
    """String column `name`."""
    fields = vars(self)
    return StringTable(fields[f"{name}_blob"], fields[f"{name}_offsets"])

  def take(self, rows: Column[np.intp]) -> Self:
    """The rows `rows` of this table, in that order (a table of rows only: the pools of `Meshes` /
    `Images`, which rows address, are not re-addressed)."""
    mine = self.arrays()
    columns = {name: mine[name][rows] for name in self.SCHEMA}
    picked = rows.tolist()
    strings: dict[str, list[str]] = {}
    for name in self.STRINGS:
      values = self.strings(name).tolist()  # decoded once, not once per row
      strings[name] = [values[i] for i in picked]
    return self.from_columns(columns, strings)

  def concat(self, other: Self) -> Self:
    """This table's rows followed by `other`'s (a table of rows only, as `take`)."""
    mine, theirs = self.arrays(), other.arrays()
    columns = {name: np.concatenate([mine[name], theirs[name]]) for name in self.SCHEMA}
    strings = {name: self.strings(name).tolist() + other.strings(name).tolist() for name in self.STRINGS}
    return self.from_columns(columns, strings)


def record_columns(records: Sequence[object], names: Iterable[str]) -> dict[str, npt.ArrayLike]:
  """Column `name` -> each record's field `name` (`vars`), for the `names` of a table's SCHEMA.

  Raises:
    KeyError: for a SCHEMA column the records do not have (a stale column name).
  """
  fields = [vars(r) for r in records]
  return {name: [f[name] for f in fields] for name in names}


@final
class StringPool:
  """Deduplicated string list (prim paths, body paths) a builder fills for a string column."""

  def __init__(self) -> None:
    """An empty list."""
    self.items: list[str] = []
    self.index: dict[str, int] = {}

  def add(self, s: str | None) -> int:
    """Index of `s` (added when new); -1 for None."""
    if s is None:
      return -1
    i = self.index.get(s)
    if i is None:
      i = self.index[s] = len(self.items)
      self.items.append(s)
    return i

  def ids(self, strings: Sequence[str | None] | None, count: int, what: str, *, rows: str) -> Column[np.int32]:
    """The (count,) indices of `strings` (`add`; all -1 for None).

    Raises:
      TypeError: for one string (it would be read as its characters).
      ValueError: when `strings` does not hold `count` entries (one `what` per row of `rows`).
    """
    if strings is None:
      return np.full(count, -1, np.int32)
    if isinstance(strings, str):
      raise TypeError(f"{what} is a sequence of one string for each of the {rows}, got the string {strings!r}")
    if len(strings) != count:
      raise ValueError(f"{len(strings)} {what} for {count} {rows}")
    return np.array([self.add(s) for s in strings], np.int32).reshape(count)
