"""USD adapter geometry: meshes (triangulation, primvars, subsets) and primitives."""

from __future__ import annotations

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad

pytest.importorskip("pxr")

from pxr import Sdf, UsdGeom

import raywarp as rw
import raywarp.formats.usd


def load(stage, **options) -> rw.Scene:
  return rw.formats.usd.read(stage, adapter=rw.formats.usd.Options(**options)).build()


def test_quad_and_ngon_triangulation(tmp_path):
  stage = new_stage(tmp_path)
  quad(stage, "/Quad")
  pent = UsdGeom.Mesh.Define(stage, "/Pent")
  pent.CreatePointsAttr([(np.cos(a), np.sin(a), 0) for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)])
  pent.CreateFaceVertexCountsAttr([5])
  pent.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4])
  s = load(stage)
  assert s.nmesh == 2 and s.ninstance == 2
  m = s.meshes.get(0)
  np.testing.assert_array_equal(m.tris, [[0, 1, 2], [0, 2, 3]])
  assert len(s.meshes.get(1).tris) == 3
  assert s.report.approximated["Mesh.ngon_fan_triangulation"] == 1
  assert s.report.approximated["Mesh.subdivisionScheme.catmullClark(cage)"] == 1  # Pent: default scheme
  assert s.meshes.normal_num[1] == 5  # smooth normals computed for the subdivision cage


def test_left_handed_and_holes(tmp_path):
  stage = new_stage(tmp_path)
  mesh = quad(stage, "/Q")
  mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
  two = UsdGeom.Mesh.Define(stage, "/Two")
  two.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
  two.CreateFaceVertexCountsAttr([3, 3])
  two.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
  two.CreateHoleIndicesAttr([0])
  two.CreateSubdivisionSchemeAttr("none")
  s = load(stage)
  np.testing.assert_array_equal(s.meshes.get(0).tris[0], [0, 2, 1])
  np.testing.assert_array_equal(s.meshes.get(1).tris, [[1, 3, 2]])


def test_facevarying_indexed_uvs_and_normals(tmp_path):
  stage = new_stage(tmp_path)
  mesh = quad(stage, "/Q")
  api = UsdGeom.PrimvarsAPI(mesh)
  st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
  st.Set([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
  st.SetIndices([4, 1, 2, 3])  # corner 0 uses the 5th value
  normals = api.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.uniform)
  normals.Set([(0, 0, 1)])
  s = load(stage)
  m = s.meshes.get(0)
  uv_corners = m.uvs[m.uv_idx]  # (T, 3, 2)
  np.testing.assert_allclose(uv_corners[0, 0], [0.5, 0.5])
  np.testing.assert_allclose(uv_corners[1, 2], [0, 1])
  assert len(m.normals) == 1 and np.all(m.normal_idx == 0)


def test_non_finite_normals_and_uvs_are_dropped_per_mesh(tmp_path):
  """A mesh with NaN normals and inf UVs still loads (flat, without UVs), reported; a clean mesh of
  the same stage keeps its own (the builder rejects non-finite pools: the adapter drops them)."""
  stage = new_stage(tmp_path)
  broken = quad(stage, "/Broken")
  broken.CreateNormalsAttr([(0, 0, float("nan"))] * 4)
  broken.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  st = UsdGeom.PrimvarsAPI(broken).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
  st.Set([(0, 0), (float("inf"), 0), (1, 1), (0, 1)])
  clean = quad(stage, "/Clean", size=2.0)
  clean.CreateNormalsAttr([(0, 0, 1)] * 4)
  clean.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  s = load(stage)
  assert s.nmesh == 2
  bad, good = s.meshes.get(0), s.meshes.get(1)
  assert len(bad.normals) == 0 and len(bad.uvs) == 0 and len(good.normals) == 4
  assert s.report.dropped["Mesh.normals.non_finite"] == 1


def test_vertex_normals_and_extra_primvars_reported(tmp_path):
  stage = new_stage(tmp_path)
  mesh = quad(stage, "/Q")
  mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
  mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st1", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0, 0)] * 4)
  s = load(stage)
  np.testing.assert_array_equal(s.meshes.get(0).normal_idx, s.meshes.get(0).tris)
  assert s.report.dropped["Mesh.primvars:st1"] == 1


def test_geom_subsets_split_meshes(tmp_path):
  stage = new_stage(tmp_path)
  mesh = UsdGeom.Mesh.Define(stage, "/M")
  mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)])
  mesh.CreateFaceVertexCountsAttr([4, 4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5, 2])
  red, _ = preview_material(stage, "/Looks/Red", (1, 0, 0))
  blue, _ = preview_material(stage, "/Looks/Blue", (0, 0, 1))
  bind(mesh, red)
  subset = UsdGeom.Subset.CreateGeomSubset(mesh, "right", UsdGeom.Tokens.face, [1], "materialBind")
  bind(subset, blue)
  s = load(stage)
  assert s.nmesh == 2 and s.ninstance == 2
  assert s.meshes.vert_adr[0] == s.meshes.vert_adr[1] and s.meshes.tri_num.tolist() == [2, 2]
  colors = {tuple(s.materials.base_color[m].round(3)) for m in s.instances.material}
  assert colors == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}


def test_mesh_dedupe_across_prims(tmp_path):
  stage = new_stage(tmp_path)
  quad(stage, "/A")
  quad(stage, "/B")
  s = load(stage)
  assert s.nmesh == 1 and s.ninstance == 2 and s.report.counts["meshes.deduplicated"] == 1


def test_primitives(tmp_path):
  stage = new_stage(tmp_path)
  UsdGeom.Cube.Define(stage, "/Cube").CreateSizeAttr(3.0)
  UsdGeom.Sphere.Define(stage, "/Sphere").CreateRadiusAttr(0.5)
  cap = UsdGeom.Capsule.Define(stage, "/Capsule")
  cap.CreateRadiusAttr(0.25)
  cap.CreateHeightAttr(2.0)
  cap.CreateAxisAttr("X")
  UsdGeom.Cylinder.Define(stage, "/Cylinder").CreateHeightAttr(4.0)
  UsdGeom.Cone.Define(stage, "/Cone")
  plane = UsdGeom.Plane.Define(stage, "/Plane")
  plane.CreateWidthAttr(10.0)
  plane.CreateLengthAttr(4.0)
  s = load(stage)
  inst = s.instances
  by_path = {inst.paths[int(inst.prim[i])]: i for i in range(s.ninstance)}
  K = rw.scene.InstanceKind
  assert inst.kind[by_path["/Cube"]] == K.BOX and np.allclose(inst.size[by_path["/Cube"]], 1.5)
  assert inst.kind[by_path["/Sphere"]] == K.SPHERE and inst.size[by_path["/Sphere"], 0] == 0.5
  c = by_path["/Capsule"]
  assert inst.kind[c] == K.CAPSULE and np.allclose(inst.size[c, :2], [0.25, 1.0])
  np.testing.assert_allclose(inst.xform[c, :, 2], [1, 0, 0])  # local Z (capsule axis) -> world X
  assert np.allclose(inst.size[by_path["/Cylinder"], :2], [1.0, 2.0])
  assert inst.kind[by_path["/Cone"]] == K.MESH and s.report.approximated["Cone.tessellated"] == 1
  assert inst.kind[by_path["/Plane"]] == K.PLANE and np.allclose(inst.size[by_path["/Plane"], :2], [5, 2])


def test_primitives_of_invalid_size_are_dropped(tmp_path):
  """A primitive with a negative or non-finite dimension, or a Plane / Cone with a zero one, is
  dropped and reported: the stage still loads (a negative Cube size aborted it, a zero-width
  Plane became an infinite one)."""
  stage = new_stage(tmp_path)
  UsdGeom.Cube.Define(stage, "/Cube").CreateSizeAttr(-2.0)
  UsdGeom.Cylinder.Define(stage, "/Cylinder").CreateHeightAttr(-1.0)
  UsdGeom.Sphere.Define(stage, "/Sphere").CreateRadiusAttr(float("nan"))
  UsdGeom.Cone.Define(stage, "/Cone").CreateRadiusAttr(-1.0)
  UsdGeom.Plane.Define(stage, "/Plane").CreateWidthAttr(0.0)
  UsdGeom.Sphere.Define(stage, "/Kept").CreateRadiusAttr(0.5)
  s = load(stage)
  assert [s.instances.paths[int(i)] for i in s.instances.prim] == ["/Kept"]
  for t in ("Cube", "Cylinder", "Sphere", "Cone", "Plane"):
    assert s.report.dropped[f"{t}.invalid_size"] == 1


@pytest.mark.parametrize(("culling", "expected"), [(None, True), (False, True), (True, False)])
def test_double_sided_follows_the_stage_face_culling(tmp_path, culling, expected):
  """`Options.double_sided=None`: every gprim double-sided (Omniverse RTX) unless the stage's RTX
  settings enable face culling; then `doubleSided` decides (the quad authors none)."""
  stage = new_stage(tmp_path)
  if culling is not None:
    stage.GetRootLayer().customLayerData = {"renderSettings": {"rtx:hydra:faceCulling:enabled": culling}}
  quad(stage, "/A")
  s = load(stage)
  assert bool(s.instances.flags[0] & rw.scene.InstanceFlags.DOUBLE_SIDED) == expected


def test_flags_double_sided_and_shadows(tmp_path):
  stage = new_stage(tmp_path)
  a = quad(stage, "/A")
  a.CreateDoubleSidedAttr(True)
  b = quad(stage, "/B", 2.0)
  UsdGeom.PrimvarsAPI(b).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
  s = load(stage, double_sided=False)
  F = rw.scene.InstanceFlags
  assert s.instances.flags[0] & F.DOUBLE_SIDED and s.instances.flags[0] & F.CAST_SHADOW
  assert not s.instances.flags[1] & F.DOUBLE_SIDED and not s.instances.flags[1] & F.CAST_SHADOW


def test_a_binding_on_a_subset_outside_the_material_bind_family_is_reported(tmp_path):
  """Only materialBind subsets split a mesh by material (USD / Hydra): a bound subset of another
  family (an exporter that left `familyName` unset) is reported, not silently ignored."""
  stage = new_stage(tmp_path)
  mesh = quad(stage, "/Q")
  mat, _ = preview_material(stage, "/Looks/M")
  bind(UsdGeom.Subset.CreateGeomSubset(mesh, "part", "face", [0], ""), mat)
  bind(UsdGeom.Subset.CreateGeomSubset(mesh, "ok", "face", [0], "materialBind"), mat)
  report = rw.formats.usd.read(stage).report
  assert report.dropped["GeomSubset.binding_outside_materialBind_family"] == 1
