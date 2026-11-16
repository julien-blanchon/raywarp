"""The compile-trigger inventory holds every compile-key input the code has (CPU, static): the
Features fields (the OptiX PTX key and every megakernel factory's), the PTX key's other parts, the
scene flags, the OptiX module record's fields, the bound build-word bits, and the device-code
factories' arguments that are not a Features field (`tools/prebuilt/triggers.py` reads them from
the code; `trigger_table.py` states each one's domain, bundle coverage and miss cost). A new input
fails here until its row exists: no compile key slips in silently."""

from __future__ import annotations

import pytest

triggers = pytest.importorskip("tools.prebuilt.triggers")
table = pytest.importorskip("tools.prebuilt.trigger_table")


@pytest.fixture(scope="module")
def derived() -> dict[str, set[str]]:
  """The triggers the code has (`triggers.derive`)."""
  return triggers.derive()


def test_every_features_field_is_inventoried(derived: dict[str, set[str]]) -> None:
  """Each Features field has its row (a domain: `FEATURES`, else a bool) and a source the
  resolution reads, or is set by the routing."""
  from raywarp.core.kernels.abi.features import Features

  types = {f.name: f.type for f in __import__("dataclasses").fields(Features)}
  untyped = [n for n in derived["features"] if n not in table.FEATURES and types[n] != "bool"]
  assert not untyped, f"non-bool Features fields without a domain in trigger_table.FEATURES: {untyped}"
  sources = triggers.features_sources()
  assert not [n for n in derived["features"] if not sources[n]], "Features fields no resolution reads"
  assert set(table.ACTIONS) <= derived["features"], sorted(set(table.ACTIONS) - derived["features"])


def test_every_factory_input_is_inventoried(derived: dict[str, set[str]]) -> None:
  """Each device-code factory argument is a Features field, a listed value (`FACTORY_INPUTS`,
  its source stated) or of a listed type (`FACTORY_TYPES`): a factory specialised on a new value
  fails here."""
  assert derived["factory"] <= set(table.FACTORY_INPUTS), sorted(derived["factory"] - set(table.FACTORY_INPUTS))
  assert derived["types"] <= table.FACTORY_TYPES, sorted(derived["types"] - table.FACTORY_TYPES)
  assert not set(table.FACTORY_INPUTS) - derived["factory"], "stale FACTORY_INPUTS rows"


def test_every_key_part_flag_record_field_and_bit_is_inventoried(derived: dict[str, set[str]]) -> None:
  """The PTX key's parts, the scene flags, the module record's fields: one `OTHER` row each; the
  word bits are rows by kind (`bit_row`); no `wp.constant` is made inside a function."""
  ids = set(table.OTHER)
  for kind in ("ptx", "flags", "record"):
    missing = {f"{kind} {k}" for k in derived[kind]} - ids
    assert not missing, sorted(missing)
  assert all(table.bit_row(n) for n in derived["word"])
  assert triggers.nested_constants() == []
