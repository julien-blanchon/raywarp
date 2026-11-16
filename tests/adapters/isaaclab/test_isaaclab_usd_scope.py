"""Generic USD clone scope used by Isaac: global content stays global, aliases bind bodies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import newton
import numpy as np
import pytest
import warp as wp
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from raywarp.integrations.newton import api, from_usd

ROOTS: Final = ("/Arena/prototype_A", "/Arena/replica_B")


def _stage() -> Usd.Stage:
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  for root in ROOTS:
    UsdGeom.Xform.Define(stage, root)
  placed = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[1]))
  placed.SetTranslate((4.0, 2.0, 0.0))
  placed.SetRotate((0.0, 0.0, 90.0))
  body = UsdGeom.Xform.Define(stage, ROOTS[0] + "/Body")
  UsdGeom.XformCommonAPI(body).SetTranslate((0.0, 0.0, 1.0))
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
  for path in (ROOTS[0] + "/Body/Visual", ROOTS[0] + "/Static", ROOTS[0] + "_extra/Global"):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(0.2)
    UsdGeom.XformCommonAPI(cube).SetTranslate((1.0, 0.0, 0.0))
  for path in (ROOTS[0] + "/Lamp", ROOTS[0] + "/Body/MovingLamp", "/Arena/GlobalLamp"):
    lamp = UsdLux.SphereLight.Define(stage, path)
    lamp.CreateRadiusAttr(0.1)
    lamp.CreateIntensityAttr(10.0)
    UsdGeom.XformCommonAPI(lamp).SetTranslate((1.0, 0.0, 2.0))
  return stage


def _model(*, bodies: bool) -> newton.Model:
  item = newton.ModelBuilder()
  body = item.add_body(label="actual_body_name") if bodies else -1
  item.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, label=ROOTS[0] + "/Body/Visual")
  builder = newton.ModelBuilder()
  builder.replicate(item, 2, spacing=(0.0, 0.0, 0.0))
  model = builder.finalize(device="cpu")
  if bodies:
    assert model.body_q is not None
    model.body_q.assign(np.array([[0, 0, 1, 0, 0, 0, 1], [4, 2, 1, 0, 0, math.sqrt(0.5), math.sqrt(0.5)]], np.float32))
  return model


def test_scoped_clone_bodies_scenery_and_lights() -> None:
  stage, model = _stage(), _model(bodies=True)
  aliases = [ROOTS[0] + "/Body"] * 2
  original_labels = list(model.body_label)
  host = from_usd.load_usd(model, stage, body_paths=aliases, world_local_roots=ROOTS)
  tables = from_usd.usd_tables(model, host, host.ninstance, stage, body_paths=aliases, world_local_roots=ROOTS)
  paths = [host.instances.paths[int(p)] for p in host.instances.prim]
  moving = paths.index(ROOTS[0] + "/Body/Visual")
  local = paths.index(ROOTS[0] + "/Static")
  global_ = paths.index(ROOTS[0] + "_extra/Global")
  np.testing.assert_array_equal(tables.body[:, moving], [0, 1])
  np.testing.assert_allclose(tables.local[:, moving, :3], [[1, 0, 0]] * 2, atol=1e-6)
  np.testing.assert_allclose(tables.local[:, local, :3], [[1, 0, 0], [4, 3, 0]], atol=1e-6)
  np.testing.assert_allclose(tables.local[:, global_, :3], [[1, 0, 0]] * 2, atol=1e-6)
  np.testing.assert_array_equal(tables.world[:, global_], [-1, -1])
  bound = api.binding(host, model, stage=stage, body_paths=aliases, world_local_roots=ROOTS)
  assert bound.dynamic(host.ninstance)[moving] and bound.dynamic(host.ninstance)[local]
  assert not bound.dynamic(host.ninstance)[global_]
  poses, active = from_usd.static_light_poses(model, host, stage, body_paths=aliases, world_local_roots=ROOTS)
  lamp_paths = host.lights.paths.tolist()
  local_lamp = lamp_paths.index(ROOTS[0] + "/Lamp")
  global_lamp = lamp_paths.index("/Arena/GlobalLamp")
  moving_lamp = lamp_paths.index(ROOTS[0] + "/Body/MovingLamp")
  np.testing.assert_allclose(poses[:, local_lamp, :3], [[1, 0, 2], [4, 3, 2]], atol=1e-6)
  np.testing.assert_allclose(poses[:, global_lamp, :3], [[1, 0, 2]] * 2, atol=1e-6)
  np.testing.assert_allclose(np.abs(poses[1, local_lamp, 3:]), [0, 0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-6)
  assert active[local_lamp] and active[global_lamp] and not active[moving_lamp]
  assert "newton.light.body(unsupported)" in str(host.report.to_dict())
  assert list(model.body_label) == original_labels


def test_body_free_worlds_use_authored_root_frames() -> None:
  stage, model = _stage(), _model(bodies=False)
  host = from_usd.load_usd(model, stage, body_paths=[], world_local_roots=ROOTS)
  tables = from_usd.usd_tables(model, host, host.ninstance, stage, body_paths=[], world_local_roots=ROOTS)
  paths = [host.instances.paths[int(p)] for p in host.instances.prim]
  local = paths.index(ROOTS[0] + "/Static")
  global_ = paths.index(ROOTS[0] + "_extra/Global")
  np.testing.assert_allclose(tables.local[:, local, :3], [[1, 0, 0], [4, 3, 0]], atol=1e-6)
  np.testing.assert_allclose(tables.local[:, global_, :3], [[1, 0, 0]] * 2, atol=1e-6)


def test_explicit_clone_scope_rejects_incomplete_mapping() -> None:
  stage, model = _stage(), _model(bodies=True)
  with pytest.raises(ValueError, match="body_paths needs 2"):
    from_usd.load_usd(model, stage, body_paths=[ROOTS[0] + "/Body"])
  with pytest.raises(ValueError, match="world_local_roots needs 2"):
    from_usd.load_usd(model, stage, world_local_roots=ROOTS[:1])
  with pytest.raises(ValueError, match="existing absolute"):
    from_usd.load_usd(model, stage, world_local_roots=(ROOTS[0], "/missing"))


def test_private_stage_view_preserves_simulation_layers() -> None:
  """Native clones need only root metadata; the simulation's original layers stay unchanged."""
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=True)
  stage.RemovePrim(ROOTS[1])  # A native Newton clone has no duplicated USD geometry.
  model.body_label = [ROOTS[0] + "/Body", ROOTS[1] + "/Body"]
  original_root = stage.GetRootLayer().ExportToString()
  original_session = stage.GetSessionLayer().ExportToString()
  origins = np.array([[0, 0, 0], [4, 2, 0]], np.float32)
  source = UsdSource.create(stage, model, ROOTS, origins=origins)
  assert source.body_paths == (ROOTS[0] + "/Body",) * 2
  assert source.stage.GetPrimAtPath(ROOTS[1]) and not source.stage.GetPrimAtPath(ROOTS[1]).IsActive()
  assert stage.GetRootLayer().ExportToString() == original_root
  assert stage.GetSessionLayer().ExportToString() == original_session
  assert not stage.GetPrimAtPath(ROOTS[1])
  host = from_usd.load_usd(model, source.stage, body_paths=source.body_paths, world_local_roots=source.roots)
  visible = source.visibility(host)
  paths = source.instance_paths(host)[0]
  local = paths.index(ROOTS[0] + "/Static")
  assert visible.all()  # one prototype: every world shows it (worlds of several: the test below)
  # No accidental shape[-1] lookup for a visual which has no physics shape.
  shape_rows = source.shape_rows(host, model)
  np.testing.assert_array_equal(shape_rows[:, local], [-1, -1])


def test_source_library_prim_is_referenced_once_in_private_prototype() -> None:
  from raywarp.integrations.isaaclab.usd_source import CloneAssets, UsdSource

  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  UsdGeom.Cube.Define(stage, "/Library/Asset/Visual")
  model = _model(bodies=False)
  source = UsdSource.create(
    stage,
    model,
    ROOTS,
    origins=np.array([[0, 0, 0], [4, 2, 0]], np.float32),
    assets=CloneAssets(shared=(), local={0: (("/Library/Asset", ROOTS[0] + "/Asset"),)}),
    excluded_sources=["/Library/Asset"],
  )
  assert source.stage.GetPrimAtPath(ROOTS[0] + "/Asset/Visual").IsActive()
  assert not source.stage.GetPrimAtPath("/Library/Asset").IsActive()
  assert stage.GetPrimAtPath("/Library/Asset").IsActive()
  host = from_usd.load_usd(model, source.stage, body_paths=[], world_local_roots=source.roots)
  assert host.ninstance == 1


def test_stage_adapter_prepares_authored_light_tables_and_visual_provenance() -> None:
  """The adapter keeps global USD lighting and places local lights in each cloned world."""
  from fixtures import isaaclab_fake as fk

  from raywarp.integrations.isaaclab.cameras import Cameras
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=True)
  model.body_label = [ROOTS[0] + "/Body", ROOTS[1] + "/Body"]
  source = UsdSource.create(stage, model, ROOTS)
  cameras = Cameras(fk.Config())
  cameras.initialize(fk.Simulation(model, model.state(), source=source))
  gpu = cameras.scene
  # USD-only lights are retained; no default key/fill is injected over authored lighting.
  assert gpu.nlight == 3
  pos = gpu.tables["light.pos"].numpy()
  active = gpu.tables["light.active"].numpy()
  assert pos.shape == (2, 3, 3)
  assert active.sum() == 4  # The body-attached light is explicitly omitted in both worlds.
  assert any(np.allclose(pos[:, i], [[1, 0, 2], [4, 3, 2]]) for i in range(3))
  assert any(np.allclose(pos[:, i], [[1, 0, 2]] * 2) for i in range(3))
  assert gpu.report.dropped["newton.light.body(unsupported)"] == 1
  cameras.close()


def test_initial_body_randomization_does_not_move_static_scene() -> None:
  stage, model = _stage(), _model(bodies=True)
  assert model.body_q is not None
  body = model.body_q.numpy()
  body[1, :3] += np.array([9, 7, 3], np.float32)
  model.body_q.assign(body)
  aliases = [ROOTS[0] + "/Body"] * 2
  host = from_usd.load_usd(model, stage, body_paths=aliases, world_local_roots=ROOTS)
  tables = from_usd.usd_tables(model, host, host.ninstance, stage, body_paths=aliases, world_local_roots=ROOTS)
  paths = [host.instances.paths[int(p)] for p in host.instances.prim]
  local = paths.index(ROOTS[0] + "/Static")
  np.testing.assert_allclose(tables.local[:, local, :3], [[1, 0, 0], [4, 3, 0]], atol=1e-6)
  poses, _ = from_usd.static_light_poses(model, host, stage, body_paths=aliases, world_local_roots=ROOTS)
  lamp = host.lights.paths.tolist().index(ROOTS[0] + "/Lamp")
  np.testing.assert_allclose(poses[:, lamp, :3], [[1, 0, 2], [4, 3, 2]], atol=1e-6)


def test_usd_camera_world_pose_and_material_updates() -> None:
  """Expected: a lit red cube in both views; world-one body/color edits affect only that view."""
  import warp as wp
  from fixtures import isaaclab_fake as fk
  from pxr import UsdShade

  from raywarp.integrations.isaaclab.cameras import Cameras
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=True)
  # A multi-pixel face makes the physically expected material/light response visible at 32x24.
  UsdGeom.Cube(stage.GetPrimAtPath(ROOTS[0] + "/Body/Visual")).GetSizeAttr().Set(1.0)
  # 314 cd after sphere-area conversion: both visible faces lie inside the upload's 1-lux window.
  for path in (ROOTS[0] + "/Lamp", "/Arena/GlobalLamp"):
    UsdLux.SphereLight(stage.GetPrimAtPath(path)).GetIntensityAttr().Set(10000.0)
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[0] + "/Lamp")).SetTranslate((-2.0, -1.0, 3.0))
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/Arena/GlobalLamp")).SetTranslate((-3.0, 3.0, 3.0))
  material_path = ROOTS[0] + "/Looks/Paint"
  material = UsdShade.Material.Define(stage, material_path)
  shader = UsdShade.Shader.Define(stage, material_path + "/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  from pxr import Sdf

  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.7, 0.0, 0.0))
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
  material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ROOTS[0] + "/Body/Visual")).Bind(material)
  model.body_label = [ROOTS[0] + "/Body", ROOTS[1] + "/Body"]
  source = UsdSource.create(stage, model, ROOTS)
  state = model.state()
  config = fk.Config(
    settings={"performance.traversal": "software", "aa.spp": 1, "exposure.mode": "manual", "exposure.ev100": 5.0},
    use_cuda_graph=False,
  )
  cameras = Cameras(config)
  cameras.initialize(fk.Simulation(model, state, source=source))
  colors = wp.array([(0.7, 0.0, 0.0), (0.0, 0.0, 0.7)], dtype=wp.vec3, device="cpu")
  writer = cameras.material_writer([material_path, ROOTS[1] + "/Looks/Paint"], colors)
  session = cameras.create_render_data(fk.RenderSpec(fk.CameraCfg(32, 24, ["rgb_hdr", "depth"]), 2))
  data = fk.CameraData.allocate(["rgb_hdr", "depth"], 2, 32, 24)
  cameras.set_outputs(session, data.output)
  pos, rot = fk.poses([(-3.0, 0.0, 1.0), (4.0, -1.0, 1.0)])
  orientations = rot.numpy()
  orientations[1] = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
  rot.assign(orientations)
  cameras.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(2, 32, 24)))
  cameras.render(session)
  depth = data.output["depth"].warp.numpy().copy()
  hdr = data.output["rgb_hdr"].warp.numpy().copy()
  np.testing.assert_allclose(depth[:, 12, 16, 0], [3.5, 3.5], atol=1e-4)
  assert np.all(hdr[:, 12, 16, 0] > hdr[:, 12, 16, 2])
  writer()
  assert state.body_q is not None
  moved = state.body_q.numpy()
  moved[1, 1] += 1.0
  state.body_q.assign(moved)
  cameras.update_transforms()  # the renderer keeps its own poses: Isaac calls this every frame
  cameras.render(session)
  after = data.output["depth"].warp.numpy()
  changed = data.output["rgb_hdr"].warp.numpy()
  np.testing.assert_array_equal(after[0], depth[0])
  np.testing.assert_allclose(after[:, 12, 16, 0], [3.5, 4.5], atol=1e-4)
  assert changed[1, 12, 16, 2] > changed[1, 12, 16, 0]
  cameras.close()


def test_equal_leaf_materials_on_subsets_keep_distinct_usd_addresses() -> None:
  """Identical constants and leaf names retain each subset's exact material path for DR."""
  from pxr import Sdf, UsdShade

  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=False)
  mesh = UsdGeom.Mesh.Define(stage, ROOTS[0] + "/Split")
  mesh.CreatePointsAttr([(-1, 0, 0), (0, 0, 0), (-1, 0, 1), (1, 0, 0), (1, 0, 1), (0, 0, 1)])
  mesh.CreateFaceVertexCountsAttr([3, 3])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
  material_paths = (ROOTS[0] + "/Looks/A/Paint", ROOTS[0] + "/Looks/B/Paint")
  for i, path in enumerate(material_paths):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.5, 0.5, 0.5))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    subset = UsdGeom.Subset.Define(stage, ROOTS[0] + f"/Split/part_{i}")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr([i])
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
  source = UsdSource.create(stage, model, ROOTS)
  host = from_usd.load_usd(model, source.stage, body_paths=[], world_local_roots=ROOTS)
  names = host.materials.names.tolist()
  ids = [names.index(path) for path in material_paths]
  assert ids[0] != ids[1]
  np.testing.assert_array_equal(host.materials.base_color[ids[0]], host.materials.base_color[ids[1]])
  bindings = source.material_paths(host)
  for world, root in enumerate(ROOTS):
    assert root + "/Looks/A/Paint" in bindings[world]
    assert root + "/Looks/B/Paint" in bindings[world]


def test_instance_prototype_material_address_resolves_to_its_visual() -> None:
  """An internal USD prototype name is translated through the actual instance metadata."""
  from pxr import Sdf, UsdShade

  from raywarp.integrations.isaaclab.usd_source import CloneAssets, UsdSource

  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  cube = UsdGeom.Cube.Define(stage, "/Library/Asset/Visual")
  material = UsdShade.Material.Define(stage, "/Library/Asset/Looks/Paint")
  shader = UsdShade.Shader.Define(stage, "/Library/Asset/Looks/Paint/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.5, 0.5, 0.5))
  material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
  model = _model(bodies=False)
  source = UsdSource.create(
    stage,
    model,
    ROOTS,
    origins=np.array([[0, 0, 0], [4, 2, 0]], np.float32),
    assets=CloneAssets(shared=(), local={0: (("/Library/Asset", ROOTS[0] + "/Robot"),)}),
    excluded_sources=["/Library/Asset"],
  )
  source.stage.GetPrimAtPath(ROOTS[0] + "/Robot").SetInstanceable(True)
  host = from_usd.load_usd(model, source.stage, body_paths=[], world_local_roots=ROOTS)
  assert host.ninstance == 1
  assert source.material_paths(host) == tuple((root + "/Robot/Looks/Paint",) for root in ROOTS)


def test_native_clones_without_positions_preserve_authored_root_placement() -> None:
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=False)
  stage.RemovePrim(ROOTS[1])
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[0])).SetTranslate((5.0, 0.0, 0.0))
  source = UsdSource.create(stage, model, ROOTS)
  host = from_usd.load_usd(model, source.stage, body_paths=[], world_local_roots=ROOTS)
  tables = from_usd.usd_tables(model, host, host.ninstance, source.stage, body_paths=[], world_local_roots=ROOTS)
  paths = [host.instances.paths[int(p)] for p in host.instances.prim]
  local = paths.index(ROOTS[0] + "/Static")
  np.testing.assert_allclose(tables.local[:, local, :3], [[6, 0, 0]] * 2, atol=1e-6)


def test_private_view_preserves_composition_state(tmp_path: Path) -> None:
  """Muted, unloaded and unpopulated content must not reappear during renderer preparation."""
  from pxr import Ar

  from raywarp.integrations.isaaclab.usd_source import UsdSource

  original = _stage()
  context = Ar.ResolverContext(Ar.DefaultResolverContext([str(tmp_path)]))
  stage = Usd.Stage.Open(original.GetRootLayer(), original.GetSessionLayer(), context)
  muted = Usd.Stage.CreateInMemory()
  UsdGeom.Cube.Define(muted, "/Arena/Muted")
  stage.GetRootLayer().subLayerPaths.append(muted.GetRootLayer().identifier)
  stage.MuteLayer(muted.GetRootLayer().identifier)
  payload = Usd.Stage.CreateInMemory()
  UsdGeom.Xform.Define(payload, "/Payload")
  UsdGeom.Cube.Define(payload, "/Payload/Visual")
  prim = stage.DefinePrim("/Arena/Payload")
  prim.GetPayloads().AddPayload(payload.GetRootLayer().identifier, "/Payload")
  stage.Unload("/Arena/Payload")
  UsdGeom.Cube.Define(stage, "/Excluded")
  stage.SetPopulationMask(Usd.StagePopulationMask(["/Arena"]))
  source = UsdSource.create(stage, _model(bodies=False), ROOTS)
  assert source.stage.GetPathResolverContext() == context
  assert source.stage.GetMutedLayers() == stage.GetMutedLayers()
  assert source.stage.GetLoadRules() == stage.GetLoadRules()
  assert source.stage.GetPopulationMask() == stage.GetPopulationMask()
  assert not source.stage.GetPrimAtPath("/Arena/Muted")
  assert not source.stage.GetPrimAtPath("/Arena/Payload/Visual")
  assert not source.stage.GetPrimAtPath("/Excluded")


@pytest.mark.parametrize("positions", [False, True])
def test_missing_clone_uses_canonical_origin_with_stage_units_and_parent(positions: bool) -> None:
  from raywarp.formats.usd.stage.reader import prim_transforms
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage = _stage()
  UsdGeom.SetStageMetersPerUnit(stage, 0.01)
  UsdGeom.XformCommonAPI(UsdGeom.Xform.Define(stage, "/Arena")).SetTranslate((100.0, 200.0, 0.0))
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[0])).SetTranslate((50.0, 0.0, 0.0))
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[0])).SetRotate((0.0, 0.0, 90.0))
  second_parent = UsdGeom.Xform.Define(stage, "/Other")
  UsdGeom.XformCommonAPI(second_parent).SetTranslate((-100.0, 0.0, 0.0))
  UsdGeom.XformCommonAPI(second_parent).SetRotate((0.0, 0.0, -90.0))
  roots = (ROOTS[0], "/Other/Replica")
  origins = np.array([[99.0, 99.0, 99.0], [4.0, 2.0, 0.0]], np.float32) if positions else None
  source = UsdSource.create(stage, _model(bodies=False), roots, origins=origins)
  frames = prim_transforms(source.stage, list(roots))
  np.testing.assert_allclose(frames[0, :3, 3], [1.5, 2.0, 0.0], atol=1e-6)  # Existing roots override positions.
  expected = [4.0, 2.0, 0.0] if positions else [1.5, 2.0, 0.0]
  np.testing.assert_allclose(frames[1, :3, 3], expected, atol=1e-6)
  np.testing.assert_allclose(frames[1, :3, :3], frames[0, :3, :3], atol=1e-6)


def test_local_environment_is_reported_and_omitted_while_global_dome_is_retained() -> None:
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=False)
  local = ROOTS[0] + "/LocalSky"
  UsdLux.DomeLight.Define(stage, local)
  UsdLux.DomeLight.Define(stage, "/Arena/GlobalSky")
  source = UsdSource.create(stage, model, ROOTS)
  host = from_usd.load_usd(model, source.stage, body_paths=[], world_local_roots=ROOTS)
  source.visibility(host)
  assert host.environments.paths.tolist() == ["/Arena/GlobalSky"]
  assert host.report.dropped["isaac.environment.local_placement(unsupported)"] == 1
  assert stage.GetPrimAtPath(local).IsActive()


def test_clone_assets_are_the_shared_group_and_each_homes_prototype() -> None:
  """Every world shows the shared assets (group 0, at world 0's paths); each prototype's assets
  (group p + 1) are read once, at its first world's paths (its home)."""
  import numpy as np

  from raywarp.integrations.isaaclab.usd_source import CloneAssets, clone_assets, world_homes

  layout = np.array([1, 0, 2, 1], np.int32)
  # Groups: shared [ground], prototype 0 [A], prototype 1 [B, cube], prototype 2 [C].
  starts = np.array([0, 1, 2, 4, 5], np.int64)
  prototypes = np.array([0, 1, 2, 3, 4], np.int32)
  sources = ["/Lib/Ground", "/Lib/RobotA", "/Lib/RobotB", None, "/Lib/RobotC"]
  templates = ["/World/Ground", "/World/env_{}/RobotA", "/World/env_{}/RobotB", "/World/env_{}/Cube", "/World/env_{}/RobotC"]
  homes = world_homes(layout)
  assert homes == (0, 1, 2, 0)
  assert clone_assets(layout, starts, prototypes, sources, templates, homes) == CloneAssets(
    shared=(("/Lib/Ground", "/World/Ground"),),
    local={
      0: (("/Lib/RobotB", "/World/env_0/RobotB"),),  # the unused cube definition (None) is skipped
      1: (("/Lib/RobotA", "/World/env_1/RobotA"),),
      2: (("/Lib/RobotC", "/World/env_2/RobotC"),),
    },
  )


MULTI: Final = tuple(f"/World/env_{w}" for w in range(3))


def _multi_asset_stage() -> Usd.Stage:
  """Three environments 4 m apart; a library of a table (shared) and two objects (a cube, a ball):
  Isaac Lab's multi-asset spawning, prototype 0 (the cube) in worlds 0 and 2, 1 (the ball) in
  world 1."""
  stage = Usd.Stage.CreateInMemory()
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  for w, root in enumerate(MULTI):
    UsdGeom.XformCommonAPI(UsdGeom.Xform.Define(stage, root)).SetTranslate((4.0 * w, 0.0, 0.0))
  table = UsdGeom.Cube.Define(stage, "/Lib/Table")
  table.CreateSizeAttr(1.0)
  UsdGeom.XformCommonAPI(table).SetScale((2.0, 2.0, 0.1))
  for name in ("Cube", "Ball"):
    body = UsdGeom.Xform.Define(stage, f"/Lib/{name}")
    UsdGeom.XformCommonAPI(body).SetTranslate((0.0, 0.0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
  UsdGeom.Cube.Define(stage, "/Lib/Cube/Visual").CreateSizeAttr(0.4)
  UsdGeom.Sphere.Define(stage, "/Lib/Ball/Visual").CreateRadiusAttr(0.5)
  return stage


def _multi_asset_model() -> newton.Model:
  """Each world's object body (at its root + (0, 0, 1)) and shape, labelled by prim path."""
  scene = newton.ModelBuilder()
  for w, root in enumerate(MULTI):
    item = newton.ModelBuilder()
    body = item.add_body(xform=wp.transform((4.0 * w, 0.0, 1.0), wp.quat_identity()), label=root + "/Object")
    if w == 1:
      item.add_shape_sphere(body, radius=0.5, label=root + "/Object/Visual")
    else:
      item.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2, label=root + "/Object/Visual")
    scene.add_world(item)
  return scene.finalize(device="cpu")


def test_worlds_of_different_prototypes_show_their_own_object() -> None:
  """Isaac Lab's multi-asset spawning (a different object per environment): each prototype's
  content is read once (at its home), bound to every world's own body and shown only in its
  worlds; the shared table is shown everywhere. Rendered: each world sees its own object."""
  from fixtures import isaaclab_fake as fk

  from raywarp.integrations.isaaclab.cameras import Cameras
  from raywarp.integrations.isaaclab.usd_source import UsdSource, clone_assets, world_homes

  stage, model = _multi_asset_stage(), _multi_asset_model()
  layout = np.array([0, 1, 0], np.int32)
  homes = world_homes(layout)
  starts, prototypes = np.array([0, 1, 2, 3], np.int64), np.array([0, 1, 2], np.int32)
  templates = ["/World/env_{}/Table", "/World/env_{}/Object", "/World/env_{}/Object"]
  assets = clone_assets(layout, starts, prototypes, ["/Lib/Table", "/Lib/Cube", "/Lib/Ball"], templates, homes)
  source = UsdSource.create(stage, model, MULTI, homes=homes, assets=assets, excluded_sources=["/Lib"])
  assert source.homes == (0, 1, 0)
  assert source.body_paths == (MULTI[0] + "/Object", MULTI[1] + "/Object", MULTI[0] + "/Object")
  scope = {"body_paths": source.body_paths, "world_local_roots": source.roots, "world_homes": source.homes}
  host = from_usd.load_usd(model, source.stage, **scope)
  paths = [host.instances.paths[int(p)] for p in host.instances.prim]
  cube, ball, table = (paths.index(p) for p in (MULTI[0] + "/Object/Visual", MULTI[1] + "/Object/Visual", MULTI[0] + "/Table"))
  assert host.ninstance == 3  # the table once, each object once (not world 1's table, not world 2's cube)
  np.testing.assert_array_equal(source.visibility(host)[:, [cube, ball, table]], [[1, 0, 1], [0, 1, 1], [1, 0, 1]])
  np.testing.assert_array_equal(source.shape_rows(host, model)[:, [cube, ball]], [[0, -1], [-1, 1], [2, -1]])
  tables = from_usd.usd_tables(model, host, host.ninstance, source.stage, **scope)
  np.testing.assert_array_equal(tables.body[:, cube], [0, -1, 2])
  np.testing.assert_array_equal(tables.body[:, ball], [-1, 1, -1])
  np.testing.assert_allclose(tables.local[:, table, :3], [[0, 0, 0], [4, 0, 0], [8, 0, 0]], atol=1e-6)

  cameras = Cameras(fk.Config(settings={"performance.traversal": "software", "aa.spp": 1}, use_cuda_graph=False))
  cameras.initialize(fk.Simulation(model, model.state(), source=source))
  session = cameras.create_render_data(fk.RenderSpec(fk.CameraCfg(9, 9, ["distance_to_image_plane"]), 3))
  data = fk.CameraData.allocate(["distance_to_image_plane"], 3, 9, 9)
  cameras.set_outputs(session, data.output)
  # Above each object, looking down (+X forward turned to -Z).
  pos, rot = fk.poses([(4.0 * w, 0.0, 4.0) for w in range(3)], (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)))
  cameras.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(3, 9, 9)))
  cameras.render(session)
  depth = np.asarray(data.output["distance_to_image_plane"].warp.numpy()).reshape(3, 9, 9)
  np.testing.assert_allclose(depth[:, 4, 4], [2.8, 2.5, 2.8], atol=1e-3)  # cube top 1.2 m, ball top 1.5 m
  # Unlabelled: the table in each world (no Newton shape); not the objects where they are hidden.
  cameras._segments_of(3)  # noqa: SLF001  # the segment tables a segmentation output reads
  assert cameras.scene.report.dropped["isaac.segmentation.USD_only_visual(unlabelled)"] == 3
  cameras.close()


def test_kit_version_decides_the_lux_dialect_of_the_live_stage() -> None:
  from raywarp.formats.usd.options import Dialect
  from raywarp.integrations.isaaclab.usd_source import kit_lux_dialect

  assert kit_lux_dialect("107.3.1+release.1234") == Dialect.RTX_LEGACY  # Isaac Sim 5.x
  assert kit_lux_dialect("109.0.0") == Dialect.SPEC
  assert kit_lux_dialect(None) is None  # kit-less: the stage's markers decide


def test_the_source_dialect_reaches_the_stage_lights() -> None:
  """A DistantLight without a version attribute reads by the source's dialect (Kit's)."""
  from fixtures import isaaclab_fake as fk

  from raywarp._base.types import LightType
  from raywarp.formats.usd.options import Dialect
  from raywarp.integrations.isaaclab.cameras import Cameras
  from raywarp.integrations.isaaclab.usd_source import UsdSource

  strengths = []
  for dialect in (Dialect.RTX_LEGACY, Dialect.SPEC):
    stage, model = _stage(), _model(bodies=True)
    UsdLux.DistantLight.Define(stage, "/Arena/Sun").CreateIntensityAttr(3000.0)
    model.body_label = [ROOTS[0] + "/Body", ROOTS[1] + "/Body"]
    cameras = Cameras(fk.Config())
    cameras.initialize(fk.Simulation(model, model.state(), source=UsdSource.create(stage, model, ROOTS, lux_dialect=dialect)))
    sun = cameras.scene.tables["light.type"].numpy()[0].tolist().index(int(LightType.DIRECTIONAL))
    strengths.append(float(cameras.scene.tables["light.strength"].numpy()[0, sun]))
  legacy, spec = strengths
  assert legacy == pytest.approx(3000.0, rel=1e-4)  # RTX legacy: an unnormalised DistantLight intensity is lux
  assert spec != pytest.approx(legacy, rel=1e-2)
