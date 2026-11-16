"""`rw.integrations.newton.scene` reads the model's shape rows once."""

from collections.abc import Collection

import pytest
import warp as wp

newton = pytest.importorskip("newton")

import raywarp.integrations.newton as newton_pbr
from raywarp.integrations.newton import api, build
from raywarp.integrations.newton.binding import IntTable


def test_scene_reads_shape_rows_once(monkeypatch: pytest.MonkeyPatch) -> None:
  """`scene` selects the rendered shapes once (it computed them for `load` and again for the
  per-world report: twice the host copies and the unrendered-type warning)."""
  builder = newton.ModelBuilder()
  body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  builder.add_shape_sphere(body, radius=0.2)
  model = builder.finalize(device="cpu")
  calls: list[int] = []
  original = build.shape_rows

  def counted(model: newton.Model, shapes: Collection[int] | None, *, merge_worlds: bool, non_usd: bool = False) -> IntTable:
    calls.append(1)
    return original(model, shapes, merge_worlds=merge_worlds, non_usd=non_usd)

  monkeypatch.setattr(api, "shape_rows", counted)  # `scene` is api.scene
  scene = newton_pbr.scene(model)
  assert scene.ninstance == 1
  assert len(calls) == 1
