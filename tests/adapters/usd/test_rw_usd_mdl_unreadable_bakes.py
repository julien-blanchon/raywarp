"""MDL disk-cache files that are gone, empty or truncated are misses, never a failed read."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("pxr")

from raywarp.formats.usd.materials.context import ShadeContext
from raywarp.formats.usd.materials.mdl import bake, memo, sdk
from raywarp.scene.model.types import Json

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.mdl import sdk_types as st

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _unreadable(tmp_path: Path, state: str) -> Path:
  """A bake file that is `state`: "gone", "empty" (a crash before its data reached the disk) or
  "truncated" (a header only)."""
  path = tmp_path / f"{state}.npy"
  if state == "empty":
    path.write_bytes(b"")
  elif state == "truncated":
    path.write_bytes(b"\x93NUMPY")
  return path


@pytest.mark.parametrize("state", ["gone", "empty", "truncated"])
def test_a_conversion_entry_whose_bake_file_is_unreadable_does_not_replay(tmp_path: Path, state: str) -> None:
  """`memo.replay` of an entry whose bake file is gone or unreadable is a miss (None: the read
  converts live), not an exception that fails the stage read."""
  npy = _unreadable(tmp_path, state)
  data: dict[str, Json] = {"ops": [["baked", str(npy), "srgb", "srgb_rec709", 1.0]], "material": {}, "uv": []}
  ctx = cast(ShadeContext, SimpleNamespace())
  assert memo.replay(ctx, "/Looks/M", data, {}) is None


def test_an_empty_cached_bake_is_a_miss(tmp_path: Path) -> None:
  """An empty bake cache file is removed and the bake made again (here: a baker the SDK does not
  create, so None), not an EOFError."""
  cached = _unreadable(tmp_path, "empty")
  calls: list[str] = []

  def create_baker(material: object, path: str, resource: object) -> None:
    """Records the bake (the cache missed) and makes no baker."""
    calls.append(path)

  api = cast(sdk.Api, SimpleNamespace(distiller=SimpleNamespace(create_baker=create_baker), bake_on_cpu=None))
  domain = bake.Domain(window=(0.0, 1.0, 0.0, 1.0), uv_set=0, rows=_IDENTITY, size=2, exact=True)
  job = bake._Job(  # the cache read in isolation from the expression planning
    path="p", domain=domain, kind="scalar", mean=False, varies=True, key="k", cached=cached, gamuts=()
  )
  material = cast("st.ICompiled_material", None)
  assert bake._texels(api, material, job, set()) is None  # the cache read
  assert calls == ["p"] and not cached.exists()
