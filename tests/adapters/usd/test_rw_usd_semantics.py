"""UsdSemantics labels read at ingest: a host column (kept by the `.wpbr` file) and the default
classes of semantic segmentation; explicit `set_semantics` patterns override them."""

from __future__ import annotations

import numpy as np
import pytest
from rw_helpers import new_stage, quad

pytest.importorskip("pxr")

import warp as wp
from pxr import Sdf, UsdGeom, UsdSemantics

import raywarp as rw
import raywarp.formats.usd


def _stage(tmp_path):
  """Three quads: /Chair/Seat (labelled through its parent, UsdSemantics), /Table (Isaac's legacy
  SemanticsAPI), /Floor (no label)."""
  stage = new_stage(tmp_path)
  UsdGeom.Xform.Define(stage, "/Chair")
  UsdSemantics.LabelsAPI.Apply(stage.GetPrimAtPath("/Chair"), "class").CreateLabelsAttr(["chair"])
  quad(stage, "/Chair/Seat", 0.5)
  table = quad(stage, "/Table", 0.5).GetPrim()
  table.AddAppliedSchema("SemanticsAPI:Semantics")
  table.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
  table.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set("table")
  quad(stage, "/Floor", 2.0)
  stage.GetRootLayer().Save()
  return stage


def test_labels_are_a_host_column_and_the_default_classes(tmp_path):
  """Labels survive the `.wpbr` file; the renderer starts from them; patterns override."""
  s = rw.formats.usd.read(_stage(tmp_path)).build()
  path = tmp_path / "scene.wpbr"
  s.save(path)
  loaded = rw.Scene.load(path)
  paths = loaded.instances.paths.tolist()
  by_path = dict(zip([paths[p] for p in loaded.instances.prim], loaded.instances.label_of(), strict=True))
  assert by_path == {"/Chair/Seat": "chair", "/Table": "table", "/Floor": ""}
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(loaded)
    r = rw.Renderer(gpu, nworld=1, cameras=[rw.CameraSpec()], resolution=(4, 4))
  assert r.semantic_labels == ("background", "unlabelled", "chair", "table")
  ids = r.context().tables.semantics.ids.numpy()[0]
  names = gpu.instance_names
  assert ids[names.index("/Chair/Seat")] == 2 and ids[names.index("/Table")] == 3 and ids[names.index("/Floor")] == 1
  labels = r.set_semantics({"ground": "/Floor", "chair": "/Table"})
  assert labels == ("background", "unlabelled", "chair", "table", "ground")
  ids = r.context().tables.semantics.ids.numpy()[0]
  assert ids[names.index("/Table")] == 2 and ids[names.index("/Floor")] == 4


def test_stage_without_semantics_has_no_labels(tmp_path):
  """No semantics schema on the stage: no label column values, no authored classes."""
  stage = new_stage(tmp_path)
  quad(stage, "/Floor")
  s = rw.formats.usd.read(stage).build()
  assert np.all(np.asarray(s.instances.label) == -1)
  with wp.ScopedDevice("cpu"):
    assert rw.upload(s).instance_labels == []


def test_labels_inside_native_prototypes(tmp_path):
  """Labels authored only inside an instanced asset are read (they were missed: the label scan
  did not visit native prototypes)."""
  from pxr import Gf

  asset = new_stage(tmp_path, "asset.usda")
  asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Asset").GetPrim())
  seat = quad(asset, "/Asset/Seat", 0.5).GetPrim()
  UsdSemantics.LabelsAPI.Apply(seat, "class").CreateLabelsAttr(["chair"])
  asset.GetRootLayer().Save()
  stage = new_stage(tmp_path)
  for k in range(2):
    inst = stage.DefinePrim(f"/Chair{k}")
    inst.GetReferences().AddReference(Sdf.Reference(str(tmp_path / "asset.usda")))
    inst.SetInstanceable(True)
    UsdGeom.Xformable(inst).AddTranslateOp().Set(Gf.Vec3d(k, 0, 0))
  s = rw.formats.usd.read(stage).build()
  paths = s.instances.paths.tolist()
  labels = dict(zip([paths[p] for p in s.instances.prim], s.instances.label_of(), strict=True))
  assert labels == {"/Chair0/Seat": "chair", "/Chair1/Seat": "chair"}
