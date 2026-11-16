"""Newton stage ingest: USD rigid bodies matched to Newton bodies, import transforms (host only)."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")
pxr = pytest.importorskip("pxr")

from pxr import UsdGeom, UsdPhysics

import raywarp.integrations.newton as newton_pbr
from raywarp.integrations.newton.from_usd import usd_tables


def _stage(path: Path, bodies: dict[str, tuple[float, float, float]]) -> str:
  """A z-up stage: one rigid body Xform with a cube visual per entry, and a static ground cube."""
  stage = pxr.Usd.Stage.CreateNew(str(path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  for prim, pos in bodies.items():
    body = UsdGeom.Xform.Define(stage, prim)
    UsdGeom.XformCommonAPI(body).SetTranslate(pos)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdGeom.Cube.Define(stage, f"{prim}/Visual").CreateSizeAttr(0.2)
  ground = UsdGeom.Cube.Define(stage, "/World/Ground")
  UsdGeom.XformCommonAPI(ground).SetTranslate((0.0, 0.0, -0.05))
  stage.GetRootLayer().Save()
  return str(path)


def test_a_suffix_naming_two_bodies_of_one_world_raises(tmp_path):
  stage = _stage(tmp_path / "s.usda", {"/Robot/Link": (0.0, 0.0, 1.0)})
  builder = newton.ModelBuilder()
  for prefix in ("/a", "/b"):  # two prefixed replicas in the same (global) world
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label=f"{prefix}/Robot/Link")
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  model = builder.finalize(device="cpu")
  with pytest.raises(ValueError, match="matches several Newton bodies of world"):
    newton_pbr.load_usd(model, stage)


def test_prefixed_replicas_in_different_worlds_bind_one_body_each(tmp_path):
  stage = _stage(tmp_path / "s.usda", {"/Robot/Link": (0.0, 0.0, 1.0)})
  item = newton.ModelBuilder()
  body = item.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="/Robot/Link")
  item.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  builder = newton.ModelBuilder()
  for w in range(2):
    builder.begin_world()
    builder.add_builder(item, xform=wp.transform((2.0 * w, 0.0, 0.0), wp.quat_identity()), label_prefix=f"/env_{w}")
    builder.end_world()
  model = builder.finalize(device="cpu")
  host = newton_pbr.load_usd(model, stage)
  tables = usd_tables(model, host, host.ninstance, stage)
  visual = host.instances.paths.tolist().index("/Robot/Link/Visual")
  assert tables.body[:, visual].tolist() == [0, 1]
  # The static ground is copied per world too: row w's copy is world w's (viewer world offsets).
  ground = host.instances.paths.tolist().index("/World/Ground")
  assert tables.world[:, ground].tolist() == [0, 1]


def test_static_placement_follows_the_import_transform_most_bodies_agree_on(tmp_path):
  bodies = {"/World/A": (0.0, 0.0, 1.0), "/World/B": (1.0, 0.0, 1.0), "/World/C": (2.0, 0.0, 1.0)}
  stage = _stage(tmp_path / "s.usda", bodies)
  offset = np.array([0.0, 0.0, 0.5])  # Newton's import transform (e.g. add_usd(xform=...))
  builder = newton.ModelBuilder()
  for prim, pos in bodies.items():
    moved = (0.0, 3.0, 0.0) if prim == "/World/A" else (0.0, 0.0, 0.0)  # A (the first) moved after import
    start = np.asarray(pos) + offset + np.asarray(moved)
    body = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()), label=prim)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  model = builder.finalize(device="cpu")
  host = newton_pbr.load_usd(model, stage)
  tables = usd_tables(model, host, host.ninstance, stage)
  ground = host.instances.paths.tolist().index("/World/Ground")
  np.testing.assert_allclose(tables.local[0, ground, :3], [0.0, 0.0, 0.45], atol=1e-6)
  assert host.report.approximated["newton.import_transform(bodies_moved_after_import)"] == 1


def test_one_row_tables_of_a_multi_world_model(tmp_path):
  """A 2-world model whose USD body matches in world 0 only (a robot added to one world): the
  tables have one row (no body replicated), `local` included (it had one row per world)."""
  stage = _stage(tmp_path / "s.usda", {"/Robot/Link": (0.0, 0.0, 1.0)})
  builder = newton.ModelBuilder()
  for w in range(2):
    builder.begin_world()
    if w == 0:
      body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="/Robot/Link")
      builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    else:
      builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
    builder.end_world()
  model = builder.finalize(device="cpu")
  host = newton_pbr.load_usd(model, stage)
  tables = usd_tables(model, host, host.ninstance, stage)
  assert tables.body.shape == tables.world.shape == tables.shape.shape == tables.local.shape[:2] == (1, host.ninstance)
  visual = host.instances.paths.tolist().index("/Robot/Link/Visual")
  assert tables.body[0, visual] == 0
