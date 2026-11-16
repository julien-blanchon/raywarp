"""USD adapter stage handling: units, up axis, pruning, instancing, PointInstancer, cache."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import REPO
from rw_helpers import bind, new_stage, preview_material, quad

pytest.importorskip("pxr")

from pxr import Gf, UsdGeom, UsdPhysics

import raywarp as rw
import raywarp.formats.usd
from raywarp.cache import paths
from raywarp.core.pipeline.textures import formats
from raywarp.scene.model import wpbr as cache


def load(stage, **options) -> rw.Scene:
  return rw.formats.usd.read(stage, adapter=rw.formats.usd.Options(**options)).build()


def world_points(s: rw.Scene, i: int) -> np.ndarray:
  m = s.meshes.get(int(s.instances.mesh[i]))
  x = s.instances.xform[i]
  return m.points @ x[:, :3].T + x[:, 3]


def test_centimetre_stage_matches_metres(tmp_path):
  cm = new_stage(tmp_path, "cm.usda", mpu=0.01)
  q = quad(cm, "/Q", 100.0)
  UsdGeom.Xformable(q).AddTranslateOp().Set(Gf.Vec3d(0, 0, 50))
  m = new_stage(tmp_path, "m.usda")
  q = quad(m, "/Q", 1.0)
  UsdGeom.Xformable(q).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
  np.testing.assert_allclose(world_points(load(cm), 0), world_points(load(m), 0), atol=1e-6)


def test_y_up_is_rotated_to_z_up(tmp_path):
  stage = new_stage(tmp_path, up="Y")
  q = quad(stage, "/Q")
  UsdGeom.Xformable(q).AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))  # 2 up
  s = load(stage)
  np.testing.assert_allclose(s.instances.xform[0, :, 3], [0, 0, 2], atol=1e-6)
  assert s.meta["up_axis"] == "Y"


def test_purpose_and_visibility_pruning(tmp_path):
  stage = new_stage(tmp_path)
  quad(stage, "/Visible")
  quad(stage, "/Hidden/Q")
  UsdGeom.Imageable(stage.GetPrimAtPath("/Hidden")).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
  guide = quad(stage, "/Guide")
  guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
  proxy = UsdGeom.Xform.Define(stage, "/Proxy")
  proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
  quad(stage, "/Proxy/Child")  # inherits proxy
  render = quad(stage, "/Render")
  render.CreatePurposeAttr(UsdGeom.Tokens.render)
  s = load(stage)
  assert sorted(s.instances.paths.tolist()) == ["/Render", "/Visible"]
  assert s.report.counts["pruned.invisible"] == 1 and s.report.counts["pruned.purpose.guide"] == 1
  assert sorted(load(stage, purposes=("default", "proxy")).instances.paths.tolist()) == ["/Proxy/Child", "/Visible"]


def test_native_instancing_shares_prototype(tmp_path):
  stage = new_stage(tmp_path)
  stage.CreateClassPrim("/Asset")  # abstract: only referenced, not rendered itself
  quad(stage, "/Asset/Geo")
  red, _ = preview_material(stage, "/Asset/Looks/Red", (1, 0, 0))
  bind(stage.GetPrimAtPath("/Asset/Geo"), red)
  for i in range(20):
    prim = UsdGeom.Xform.Define(stage, f"/World/I{i}").GetPrim()
    prim.GetReferences().AddInternalReference("/Asset")
    prim.SetInstanceable(True)
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(i, 0, 0))
  s = load(stage)
  assert s.nmesh == 1 and s.ninstance == 20 and s.nmaterial == 1
  assert s.report.counts["usd.prototypes"] == 1
  np.testing.assert_allclose(sorted(s.instances.xform[:, 0, 3]), np.arange(20))
  assert "/World/I3/Geo" in s.instances.paths.tolist()
  np.testing.assert_allclose(s.materials.base_color[0], [1, 0, 0])


def test_instance_inherits_binding_from_outside_prototype(tmp_path):
  stage = new_stage(tmp_path)
  stage.CreateClassPrim("/Asset")
  quad(stage, "/Asset/Geo")
  green, _ = preview_material(stage, "/Looks/Green", (0, 1, 0))
  prim = stage.DefinePrim("/World/I")
  prim.GetReferences().AddInternalReference("/Asset")
  prim.SetInstanceable(True)
  bind(prim, green)
  plain = stage.DefinePrim("/World/J")
  plain.GetReferences().AddInternalReference("/Asset")
  plain.SetInstanceable(True)
  s = load(stage)
  by_path = {s.instances.paths[int(s.instances.prim[i])]: i for i in range(s.ninstance)}
  np.testing.assert_allclose(s.materials.base_color[s.instances.material[by_path["/World/I/Geo"]]], [0, 1, 0])
  assert s.materials.names[int(s.instances.material[by_path["/World/J/Geo"]])] == "<fallback>"


def test_a_missing_stage_file_raises_file_not_found(tmp_path):
  """A stage path that does not open raises FileNotFoundError (pxr raises its own Tf error)."""
  with pytest.raises(FileNotFoundError, match="cannot open USD stage"):
    rw.formats.usd.read(tmp_path / "missing.usda")
  with pytest.raises(FileNotFoundError):
    rw.formats.usd.settings(tmp_path / "missing.usda")


def test_point_instancer(tmp_path):
  stage = new_stage(tmp_path)
  pi = UsdGeom.PointInstancer.Define(stage, "/PI")
  UsdGeom.Xformable(pi).AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))
  cube = UsdGeom.Cube.Define(stage, "/PI/Protos/Cube")
  UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))  # prototype root xform applies
  quad(stage, "/PI/Protos/Quad")
  pi.CreatePrototypesRel().SetTargets([cube.GetPath(), "/PI/Protos/Quad"])
  pi.CreateProtoIndicesAttr([0, 1, 0, 0])
  pi.CreatePositionsAttr([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])
  pi.CreateInvisibleIdsAttr([2])
  s = load(stage)
  inst = s.instances
  assert s.ninstance == 3  # 4 points, one invisible; prototypes are not rendered standalone
  cubes = np.nonzero(inst.kind == rw.scene.InstanceKind.BOX)[0]
  np.testing.assert_allclose(sorted(inst.xform[cubes, 0, 3]), [1, 4])
  np.testing.assert_allclose(inst.xform[cubes, 2, 3], 11)
  assert "/PI[3]" in inst.paths.tolist()


def _scattered_instancer(stage, n: int, rng: np.random.Generator) -> UsdGeom.PointInstancer:
  """A PointInstancer of `n` scattered instances of one quad (a prototype root with its own
  transform) at random uniform scales and `quath` orientations, as scattered vegetation is."""
  from pxr import Vt

  pi = UsdGeom.PointInstancer.Define(stage, "/PI")
  root = UsdGeom.Xform.Define(stage, "/PI/Protos/Plant")
  root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
  root.AddRotateXYZOp().Set(Gf.Vec3f(10.0, 20.0, 30.0))
  quad(stage, "/PI/Protos/Plant/Leaf")
  pi.CreatePrototypesRel().SetTargets([root.GetPath()])
  pi.CreateProtoIndicesAttr([0] * n)
  pi.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in rng.normal(size=(n, 3)) * 10.0]))
  pi.CreateScalesAttr(Vt.Vec3fArray([Gf.Vec3f(float(s)) for s in rng.uniform(0.3, 2.0, n)]))
  q = rng.normal(size=(n, 4))
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  pi.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(w), float(x), float(y), float(z)) for x, y, z, w in q]))
  return pi


def _usd_matrices(ms):
  """USD's composed instance transforms (row-vector Gf.Matrix4d) as (N, 4, 4) column-vector
  matrices."""
  return np.ascontiguousarray(np.array(ms, dtype=np.float64).reshape(-1, 4, 4).transpose(0, 2, 1))


def test_point_instancer_transforms_are_usds_with_unit_orientations(tmp_path):
  """The adapter composes a PointInstancer's transforms as USD does (prototype root transform,
  scale, orientation, position), its `quath` orientations normalised: a uniformly scaled instance
  stays a uniform scale of a rotation (USD composes the rounded quaternion unnormalised: a stretch
  of ~1e-3, which made each instance a mesh copy at upload), and loads without a baked copy."""
  from pxr import Usd

  from raywarp.core.pipeline.upload import meshes
  from raywarp.formats.usd.stage.gprims import instance_transforms
  from raywarp.scene.model.report import LoadReport

  stage = new_stage(tmp_path)
  n = 64
  pi = _scattered_instancer(stage, n, np.random.default_rng(3))
  t = Usd.TimeCode.Default()
  protos = [stage.GetPrimAtPath(p) for p in pi.GetPrototypesRel().GetTargets()]
  ours = instance_transforms(pi, protos, np.zeros(n, np.int64), t)
  usd = _usd_matrices(pi.ComputeInstanceTransformsAtTime(t, t))
  assert ours is not None
  np.testing.assert_allclose(ours, usd, atol=5e-3)  # USD's: the unnormalised half quaternions
  linear = ours[:, :3, :3]
  sv2 = np.linalg.eigvalsh(np.swapaxes(linear, 1, 2) @ linear)  # squared singular values
  assert np.all(sv2.max(axis=1) / sv2.min(axis=1) - 1.0 < 2e-6)
  s = load(stage)
  baked = meshes.bake_stretches(s, scaled=True, report=(report := LoadReport()), paths=s.instances.paths.tolist())
  assert len(baked.meshes) == len(s.meshes) and "gpu.meshes_baked_stretch" not in report.counts


def test_point_instancer_orientationsf_takes_precedence(tmp_path):
  """`orientationsf` (float quaternions) wins over `orientations` when both are authored (the
  schema's rule), as in USD's own composition."""
  from pxr import Usd, Vt

  from raywarp.formats.usd.stage.gprims import instance_transforms

  stage = new_stage(tmp_path)
  pi = _scattered_instancer(stage, 4, np.random.default_rng(4))
  pi.CreateOrientationsfAttr(Vt.QuatfArray([Gf.Quatf(0.0, 0.0, 0.0, 1.0)] * 4))
  t = Usd.TimeCode.Default()
  protos = [stage.GetPrimAtPath(p) for p in pi.GetPrototypesRel().GetTargets()]
  ours = instance_transforms(pi, protos, np.zeros(4, np.int64), t)
  assert ours is not None
  np.testing.assert_allclose(ours, _usd_matrices(pi.ComputeInstanceTransformsAtTime(t, t)), atol=1e-6)


def test_point_instancer_with_a_missing_prototype_drops_its_points(tmp_path):
  """A prototype target naming no prim drops the points using it (reported); the others load."""
  stage = new_stage(tmp_path)
  pi = UsdGeom.PointInstancer.Define(stage, "/PI")
  cube = UsdGeom.Cube.Define(stage, "/PI/Protos/Cube")
  pi.CreatePrototypesRel().SetTargets([cube.GetPath(), "/PI/Protos/Missing"])
  pi.CreateProtoIndicesAttr([0, 1, 1])
  pi.CreatePositionsAttr([(1, 0, 0), (2, 0, 0), (3, 0, 0)])
  s = load(stage)
  assert s.ninstance == 1
  assert s.report.dropped["PointInstancer.missing_prototype"] == 2


def _instanced_assets(tmp_path, n: int = 2):
  """A stage of `n` instanceable references to asset.usda (prim /Asset), at y = 0, 10, ...; the
  asset stage is returned to be filled first."""
  from pxr import Sdf

  asset = new_stage(tmp_path, "asset.usda")
  asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Asset").GetPrim())

  def instantiate():
    asset.GetRootLayer().Save()
    stage = new_stage(tmp_path)
    for k in range(n):
      inst = stage.DefinePrim(f"/Inst{k}")
      inst.GetReferences().AddReference(Sdf.Reference(str(tmp_path / "asset.usda")))
      inst.SetInstanceable(True)
      UsdGeom.Xformable(inst).AddTranslateOp().Set(Gf.Vec3d(0, 10 * k, 0))
    assert len(stage.GetPrototypes()) == 1
    return stage

  return asset, instantiate


def test_point_instancer_inside_a_native_prototype(tmp_path):
  """The prototypes of a PointInstancer inside an instanced asset render only through it (they
  were also drawn once per instance at the asset root: `Traverse` skips native prototypes)."""
  asset, instantiate = _instanced_assets(tmp_path)
  UsdGeom.Cube.Define(asset, "/Asset/Protos/Cube")
  pi = UsdGeom.PointInstancer.Define(asset, "/Asset/PI")
  pi.CreatePrototypesRel().SetTargets(["/Asset/Protos/Cube"])
  pi.CreateProtoIndicesAttr([0, 0])
  pi.CreatePositionsAttr([(1, 0, 0), (2, 0, 0)])
  s = load(instantiate())
  assert s.ninstance == 4
  np.testing.assert_allclose(sorted(map(tuple, s.instances.xform[:, :2, 3])), [(1, 0), (1, 10), (2, 0), (2, 10)])


def test_body_binding_hook(tmp_path):
  stage = new_stage(tmp_path)
  body = UsdGeom.Xform.Define(stage, "/Robot/Link")
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
  quad(stage, "/Robot/Link/Visual")
  quad(stage, "/Static")
  s = load(stage)
  inst = s.instances
  by_path = {inst.paths[int(inst.prim[i])]: i for i in range(s.ninstance)}
  assert inst.bodies[int(inst.body[by_path["/Robot/Link/Visual"]])] == "/Robot/Link"
  assert inst.body[by_path["/Static"]] == -1


def test_scene_cache_cold_warm_and_invalidation(tmp_path):
  stage = new_stage(tmp_path)
  quad(stage, "/Q")
  stage.Save()
  src = tmp_path / "scene.usda"
  cache = tmp_path / "cache"
  cold = rw.formats.usd.load(src, cache=cache)
  files = list(cache.glob("*.wpbr"))
  assert len(files) == 1 and "warm_load" not in cold.report.timings
  warm = rw.formats.usd.load(src, cache=cache)
  assert "warm_load" in warm.report.timings and warm.ninstance == 1
  np.testing.assert_array_equal(warm.meshes.points, cold.meshes.points)
  # Editing the layer invalidates the cache.
  quad(stage, "/Q2")
  stage.Save()
  st = os.stat(src)
  os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
  again = rw.formats.usd.load(src, cache=cache)
  assert "warm_load" not in again.report.timings and again.ninstance == 2
  # Options are part of the key.
  assert rw.formats.usd.cache_path(src, cache) != rw.formats.usd.cache_path(
    src, cache, options=rw.SceneOptions(textures=formats.TextureStorage(max_size=512))
  )


def test_load_caches_by_default_and_can_be_switched_off(tmp_path, monkeypatch):
  """`cache=True` is the default (the cache directory: a second load maps the `.wpbr`);
  `cache=None` and an empty RAYWARP_CACHE_DIR convert every time and write nothing."""
  stage = new_stage(tmp_path)
  quad(stage, "/Q")
  stage.Save()
  src = tmp_path / "scene.usda"
  directory = tmp_path / "default_cache"
  monkeypatch.setenv(paths.CACHE_ENV, str(directory))
  cold = rw.formats.usd.load(src)
  assert len(list(directory.glob("*.wpbr"))) == 1 and "warm_load" not in cold.report.timings
  assert "warm_load" in rw.formats.usd.load(src).report.timings
  assert "warm_load" not in rw.formats.usd.load(src, cache=None).report.timings
  monkeypatch.setenv(paths.CACHE_ENV, "")
  off = rw.formats.usd.load(src)
  assert "warm_load" not in off.report.timings and off.ninstance == 1
  assert len(list(directory.glob("*.wpbr"))) == 1


def test_a_default_cache_directory_that_cannot_be_written_warns_and_loads(tmp_path, monkeypatch):
  """The default cache (`cache=True`) in a directory that cannot be made (under a file: also for
  root) loads without it, with a warning; an explicit cache path still raises."""
  stage = new_stage(tmp_path)
  quad(stage, "/Q")
  stage.Save()
  src = tmp_path / "scene.usda"
  (tmp_path / "file").write_text("")
  monkeypatch.setenv(paths.CACHE_ENV, str(tmp_path / "file" / "cache"))
  with pytest.warns(RuntimeWarning, match="cannot be written"):
    s = rw.formats.usd.load(src)
  assert s.ninstance == 1 and "warm_load" not in s.report.timings
  with pytest.raises(OSError):
    rw.formats.usd.load(src, cache=tmp_path / "file" / "cache")


def test_bake_and_load_without_pxr_work(tmp_path):
  """A baked `.wpbr` loads (in a fresh interpreter) without importing pxr."""
  stage = new_stage(tmp_path)
  quad(stage, "/Q")
  stage.Save()
  rw.formats.usd.bake(tmp_path / "scene.usda", tmp_path / "out.wpbr")
  s = rw.Scene.load(tmp_path / "out.wpbr")
  assert s.meta["adapter"] == "usd" and s.ninstance == 1
  assert any(dep.path.endswith("scene.usda") for dep in cache.recorded_dependencies(s.meta, tmp_path))
  code = (
    "import sys, raywarp\n"
    f"s = raywarp.Scene.load({str(tmp_path / 'out.wpbr')!r})\n"
    "assert s.ninstance == 1 and 'pxr' not in sys.modules, sorted(m for m in sys.modules if m.startswith('pxr'))\n"
  )
  subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


@pytest.mark.parametrize(
  "options",
  [
    {"displacement_max_level": 1.5},
    {"displacement_max_triangles": None},
    {"displacement_max_level": -1},
    {"material_contexts": ("mdl", "mdl")},
    {"material_contexts": ("osl",)},
    {"time": float("nan")},
  ],
)
def test_invalid_options_are_rejected(options):
  """USD options are checked when made: whole numbers are ints with a lower bound (floats were
  truncated), render contexts known and distinct, a time finite."""
  with pytest.raises(ValueError, match=r"usd\.Options"):
    rw.formats.usd.Options(**options)


def test_texture_size_cap_is_checked():
  """The texture size cap (`SceneOptions.textures.max_size`) is a whole number >= 0."""
  for size in (-1, 512.5):
    with pytest.raises(ValueError, match="max_size"):
      rw.SceneOptions(textures=formats.TextureStorage(max_size=size))  # pyrefly: ignore[bad-argument-type]


def test_time_sampled_transforms_points_and_instancers_are_reported(tmp_path):
  """A stage's animation (time-sampled transform, visibility, mesh points, PointInstancer
  positions) is read at one time: each animated prim is reported, never silently frozen; a static
  stage reports nothing."""
  stage = new_stage(tmp_path)
  moving = quad(stage, "/W/Moving")
  op = moving.AddTranslateOp()
  op.Set(Gf.Vec3d(0, 0, 0), 0)
  op.Set(Gf.Vec3d(0, 0, 5), 10)
  moving.CreateVisibilityAttr().Set("inherited", 0)
  moving.GetVisibilityAttr().Set("invisible", 10)
  points = quad(stage, "/W/Deforming").GetPointsAttr()
  points.Set([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 0)
  points.Set([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 10)
  quad(stage, "/W/Static")
  pi = UsdGeom.PointInstancer.Define(stage, "/W/PI")
  quad(stage, "/W/PI/Protos/Q")
  pi.CreatePrototypesRel().SetTargets(["/W/PI/Protos/Q"])
  pi.CreateProtoIndicesAttr([0, 0])
  pi.CreatePositionsAttr().Set([(0, 0, 0), (2, 0, 0)], 0)
  pi.GetPositionsAttr().Set([(0, 0, 1), (2, 0, 1)], 10)
  report = rw.formats.usd.read(stage).report
  assert report.approximated["usd.time_samples.xform(one time read)"] == 1
  assert report.approximated["usd.time_samples.visibility(one time read)"] == 1
  assert report.approximated["Mesh.time_samples(one time read)"] == 1
  assert report.approximated["PointInstancer.time_samples(one time read)"] == 1
  static = new_stage(tmp_path, "static.usda")
  quad(static, "/W/Q").AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
  assert not any("time_samples" in k for k in rw.formats.usd.read(static).report.approximated)


def test_joints_and_render_prims_are_data_not_dropped_geometry(tmp_path):
  """A robot's UsdPhysics joints and Kit's UsdRender prims carry no geometry: they are skipped
  without report entries (a Franka reported 14 `prim.*` drops that hid the real ones)."""
  from pxr import UsdRender

  stage = new_stage(tmp_path)
  quad(stage, "/Robot/Link")
  UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Link/joint1")
  UsdPhysics.PrismaticJoint.Define(stage, "/Robot/Link/joint2")
  UsdPhysics.Joint.Define(stage, "/Robot/rootJoint")
  UsdRender.Settings.Define(stage, "/Render/Settings")
  UsdRender.Product.Define(stage, "/Render/Product")
  UsdRender.Var.Define(stage, "/Render/Vars/LdrColor")
  assert not rw.formats.usd.read(stage).report.dropped
