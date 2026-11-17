"""rw.integrations.newton.scene: Newton model -> host Scene, uploaded bound to the bodies (Warp CPU
device)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from conftest import required

newton = pytest.importorskip("newton")

import raywarp as rw
import raywarp.integrations.newton as rw_newton
from raywarp._base.types import ShapeType
from raywarp.core.pipeline.worlds.sharing import Layout
from raywarp.integrations.newton.binding import NewtonBinding
from raywarp.integrations.newton.types import NewtonState
from raywarp.settings.fields import DEFAULT_EV100, exposure_from_ev100


def _gpu(model: newton.Model, lighting: rw.Lighting | None = None) -> rw.GpuScene[NewtonState]:
  """The uploaded scene of `rw.integrations.newton.load(model, lighting=lighting)`, bound to the
  model's bodies."""
  host = rw_newton.load(model, lighting=lighting)
  return rw.upload(host, model.device, binding=rw_newton.binding(host, model))


def _tetra() -> newton.Mesh:
  """A unit tetrahedron."""
  verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
  faces = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32)
  return newton.Mesh(verts, faces)


def test_lighting_takes_lights_only():
  with pytest.raises(TypeError, match=r"rw\.Light"):
    rw.Lighting(["area"])  # pyrefly: ignore[bad-argument-type]  # not a Light on purpose


def _object_worlds(objects: list[str], *, ground: bool = True) -> newton.Model:
  """One world per object kind (on a body at the origin, colored by kind), plus a global ground."""
  colors = {"sphere": (0.9, 0.1, 0.1), "box": (0.1, 0.9, 0.1), "capsule": (0.1, 0.1, 0.9)}

  def world(kind: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    if kind == "sphere":
      builder.add_shape_sphere(body, radius=0.5, color=colors[kind])
    elif kind == "box":
      builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.2, color=colors[kind])
    else:
      builder.add_shape_capsule(body, radius=0.2, half_height=0.3, color=colors[kind])
    return builder

  scene = newton.ModelBuilder()
  if ground:
    scene.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -1.0), wp.quat_identity()), hx=5.0, hy=5.0, hz=0.1)
  for kind in objects:
    scene.add_world(world(kind))
  return scene.finalize(device="cpu")


def test_worlds_of_different_objects_pool_them_and_show_their_own():
  """A different object per world: the scene pools the distinct shapes (equal ones share an
  instance), each world shows its own (the others hidden there), with its own color; remapping
  the worlds (`StateView.world_indices`) shows the other objects with no recompile."""
  kinds = ["sphere", "box", "capsule", "sphere"]
  model = _object_worlds(kinds)
  gpu = _gpu(model)
  assert gpu.ninstance == 4  # sphere, box, capsule (world 3's sphere is world 0's column) + ground
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth", "albedo"), gi=rw.settings.GI(enabled=False))
  r = rw.Renderer(gpu, nworld=4, cameras=[rw.CameraSpec(fovy=30.0)], resolution=(9, 9), settings=settings)
  pose = r.camera_pose.numpy()
  pose[:] = (0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0)  # above the origin, looking down -Z
  r.camera_pose.assign(pose)
  state = model.state()

  def centre(sim: NewtonState) -> tuple[np.ndarray, np.ndarray]:
    out = r.render(sim)
    return required(out.depth).numpy()[:, 0, 4, 4], required(out.albedo).numpy()[:, 0, 4, 4]

  from raywarp.core.post.grading.color import srgb_to_linear as linear

  top = {"sphere": 3.5, "box": 3.8, "capsule": 3.5}  # camera height 4 - the top (capsule along +Z)
  red, green, blue = linear(np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]))
  albedo = {"sphere": red, "box": green, "capsule": blue}
  depth, color = centre(state)
  np.testing.assert_allclose(depth, [top[k] for k in kinds], atol=1e-3)
  np.testing.assert_allclose(color, [albedo[k] for k in kinds], atol=1e-4)
  features, executables = r.features, _loaded_executables()
  order = [2, 1, 0, 1]
  view = rw_newton.StateView(state, world_indices=wp.array(order, dtype=wp.int32, device="cpu"))
  depth, color = centre(view)
  np.testing.assert_allclose(depth, [top[kinds[w]] for w in order], atol=1e-3)
  np.testing.assert_allclose(color, [albedo[kinds[w]] for w in order], atol=1e-4)
  assert r.features is features and _loaded_executables() == executables


def test_per_world_colors_keep_the_pooled_objects_hidden():
  """Colour randomisation (`PerWorld(material=["color"])`) over pooled objects: each world shows
  its own object in the colour of its table row, the others stay hidden (the binding's hiding
  applied to the table)."""
  kinds = ["sphere", "box", "capsule", "sphere"]
  model = _object_worlds(kinds)
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth", "albedo"), gi=rw.settings.GI(enabled=False))
  r = rw.Renderer(
    _gpu(model),
    nworld=4,
    cameras=[rw.CameraSpec(fovy=30.0)],
    resolution=(9, 9),
    settings=settings,
    per_world=rw.PerWorld(material=["color"]),
  )
  pose = r.camera_pose.numpy()
  pose[:] = (0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0)  # above the origin, looking down -Z
  r.camera_pose.assign(pose)
  table, state = r.per_world["instance.color"], model.state()
  tint = np.array([[0.2, 0.4, 0.6], [0.6, 0.2, 0.4], [0.4, 0.6, 0.2], [0.5, 0.5, 0.5]], np.float32)
  top = {"sphere": 3.5, "box": 3.8, "capsule": 3.5}
  albedo = []
  for rgb in (np.ones_like(tint), tint):
    colors = table.numpy()
    colors[:, :, :3] = rgb[:, None]
    colors[:, :, 3] = 1.0  # every object opaque in the table: the binding still hides the others
    table.assign(colors)
    out = r.render(state)
    np.testing.assert_allclose(required(out.depth).numpy()[:, 0, 4, 4], [top[k] for k in kinds], atol=1e-3)
    albedo.append(required(out.albedo).numpy()[:, 0, 4, 4, :3].copy())
  np.testing.assert_allclose(albedo[1], tint * albedo[0], rtol=1e-4, atol=1e-6)  # the table tints each world's object
  r.close()


def test_pooled_columns_keep_world_0s_order_and_share_equal_shapes():
  """`build.pooled`: world 0's shapes first in its order, a world's j-th shape of a kind in that
  kind's j-th column, -1 where a world has none; the binding finds every column from the scene's
  representative shapes (`source_rows`)."""
  from raywarp.integrations.newton import build

  model = _object_worlds(["sphere", "box", "box", "capsule"], ground=False)
  rows = build.shape_rows(model, None, merge_worlds=False)
  np.testing.assert_array_equal(rows, [[0, -1, -1], [-1, 1, -1], [-1, 2, -1], [-1, -1, 3]])
  np.testing.assert_array_equal(build.representatives(rows), [0, 1, 3])
  np.testing.assert_array_equal(build.source_rows(model, build.representatives(rows), merge_worlds=False), rows)


def test_worlds_differing_only_in_unrendered_shapes_build_and_bind():
  """Worlds whose rendered shapes match but one carries an extra site and an invisible shape
  (scene and binding use one rule: the rendered shapes); the binding poses world 1's sphere from
  world 1's body."""

  def world(*, extra: bool) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    if extra:
      builder.add_site(body, label="tip")
      invisible = newton.ModelBuilder.ShapeConfig(is_visible=False)
      builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=invisible)
    builder.add_shape_sphere(body, radius=0.2)
    return builder

  scene = newton.ModelBuilder()
  scene.add_world(world(extra=False))
  scene.add_world(world(extra=True))
  model = scene.finalize(device="cpu")
  gpu = _gpu(model)
  assert gpu.ninstance == 1
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"))
  r = rw.Renderer(gpu, nworld=2, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=settings)
  state = model.state()
  q = state.body_q.numpy()
  q[1, :3] = (0.0, 0.0, 3.0)  # world 1's body
  state.body_q.assign(q)
  r.render(state)
  pos = required(r.poses).pos.numpy()  # (nworld, ninstance, 3)
  np.testing.assert_allclose(pos[:, 0], [[0.0, 0.0, 0.5], [0.0, 0.0, 3.0]], atol=1e-6)


def _worlds_model(meshes: list[newton.Mesh]) -> newton.Model:
  """One world per mesh (a body with a sphere and the mesh), interleaved with a global box."""
  scene = newton.ModelBuilder()
  scene.add_shape_box(-1, hx=1.0, hy=1.0, hz=0.1)  # global (world -1)
  for mesh in meshes:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.2)
    builder.add_shape_mesh(body, mesh=mesh)
    scene.add_world(builder)
  return scene.finalize(device="cpu")


def test_world_rows_group_shapes_per_world():
  from raywarp.integrations.newton import build

  model = _worlds_model([_tetra()] * 4)
  rows = build.shape_rows(model, None, merge_worlds=False)
  world = model.shape_world.numpy()
  visible = np.isin(np.arange(len(world)), rows)
  # The per-world scans this grouping replaced: world w's shapes, then the global ones.
  expected = np.stack([np.concatenate([np.flatnonzero(visible & (world == w)), np.flatnonzero(world < 0)]) for w in range(4)])
  np.testing.assert_array_equal(rows, expected)
  bound = build.source_rows(model, rows[0], merge_worlds=False)
  np.testing.assert_array_equal(bound, rows)


def test_scales_rounded_differently_pool_into_one_column():
  """Worlds whose sphere radii differ by float rounding (1 ulp) are the same shapes: one column."""
  from raywarp.integrations.newton import build

  radii = np.array([0.2, 0.2, 0.2], np.float32)
  radii[1] = np.nextafter(radii[0], radii[0] + 1)  # one float32 ulp more
  scene = newton.ModelBuilder()
  for r in radii.tolist():
    world = newton.ModelBuilder()
    world.add_shape_sphere(world.add_body(), radius=r)
    scene.add_world(world)
  rows = build.shape_rows(scene.finalize(device="cpu"), None, merge_worlds=False)
  assert rows.shape == (3, 1) and (rows >= 0).all()


def test_a_scene_missing_a_pooled_shape_does_not_bind():
  """A (stale) scene without an instance for a world's shape raises instead of hiding it."""
  from raywarp.integrations.newton import build

  model = _object_worlds(["sphere", "box"], ground=False)
  with pytest.raises(ValueError, match="has no instance of shape 1"):
    build.source_rows(model, np.array([0], np.int32), merge_worlds=False)


def test_worlds_with_different_meshes_pool_them():
  """Worlds whose meshes differ by value get a column per distinct mesh; equal meshes (distinct
  objects of equal vertices) share one."""
  from raywarp.integrations.newton import build

  other = newton.Mesh(np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.float32), _tetra().indices)
  model = _worlds_model([_tetra(), _tetra(), other])
  rows = build.shape_rows(model, None, merge_worlds=False)
  assert rows.shape == (3, 4)  # sphere, tetrahedron, the other mesh, the global box
  np.testing.assert_array_equal(rows[:, 2] >= 0, [False, False, True])
  np.testing.assert_array_equal(rows[:, 1] >= 0, [True, True, False])


def test_shared_meshes_share_one_core_mesh():
  mesh = _tetra()
  builder = newton.ModelBuilder()
  for x in (0.0, 1.5):
    builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
  builder.add_shape_mesh(-1, mesh=mesh, scale=(2.0, 2.0, 2.0))
  scene = _gpu(builder.finalize(device="cpu"))
  assert scene.ninstance == 3
  assert scene.meshes.nmesh == 2  # one core mesh for both unscaled shapes, the scaled one's copy
  np.testing.assert_allclose(scene.meshes.points[1], 2.0 * scene.meshes.points[0], rtol=1e-6)


def test_binding_poses_follow_bodies_per_world_and_state_swaps():
  """render(state) writes body_q * local into the instance poses of each world, every call."""
  item = newton.ModelBuilder()
  body = item.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="box")
  item.add_shape_box(body, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1)
  builder = newton.ModelBuilder()
  builder.replicate(item, 2, spacing=(0.0, 0.0, 0.0))
  builder.add_ground_plane()
  model = builder.finalize(device="cpu")
  scene = _gpu(model)
  assert scene.ninstance == 2  # the box of world 0 (bound per world) and the shared ground
  r = rw.Renderer(
    scene,
    nworld=2,
    cameras=[rw.CameraSpec()],
    resolution=(8, 8),
    settings=rw.RenderSettings(outputs=rw.Outputs.of("depth")),
  )
  state_0, state_1 = model.state(), model.state()
  q = state_1.body_q.numpy()
  q[1, :3] = (1.0, 2.0, 3.0)  # world 1's box
  state_1.body_q.assign(q)
  pos = required(r.poses).pos
  r.render(state_0)
  np.testing.assert_allclose(pos.numpy()[:, 0], [[0.1, 0.0, 1.0], [0.1, 0.0, 1.0]], atol=1e-6)
  r.render(state_1)  # a swapped state is resolved on the call
  np.testing.assert_allclose(pos.numpy()[:, 0], [[0.1, 0.0, 1.0], [1.1, 2.0, 3.0]], atol=1e-6)
  r.render(rw_newton.StateView(state_1, world_indices=wp.array([1, 1], dtype=wp.int32)))
  np.testing.assert_allclose(pos.numpy()[:, 0], [[1.1, 2.0, 3.0], [1.1, 2.0, 3.0]], atol=1e-6)
  assert r.layout.nshared == 0  # planes stay per world
  assert np.allclose(pos.numpy()[:, 1], 0.0)  # the ground stays put


def test_pose_buffers_of_two_renderers_keep_their_own_columns():
  """Two Renderers of one scene with different pose layouts (one also poses the ground per
  world): each pose buffer is written with its own columns, whichever was allocated last."""
  item = newton.ModelBuilder()
  body = item.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="box")
  item.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  builder = newton.ModelBuilder()
  builder.replicate(item, 2, spacing=(0.0, 0.0, 0.0))
  builder.add_ground_plane()
  model = builder.finalize(device="cpu")
  binding = _gpu(model).binding
  assert isinstance(binding, NewtonBinding)
  device = wp.get_device("cpu")
  box_only = binding.allocate(2, np.array([0]), device=device)
  ground_and_box = binding.allocate(2, np.array([1, 0]), device=device)
  state = model.state()
  q = state.body_q.numpy()
  q[:, :3] = [(1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
  state.body_q.assign(q)
  binding.write(box_only, state)
  np.testing.assert_allclose(box_only.pos.numpy()[:, 0], [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], atol=1e-6)
  binding.write(ground_and_box, state)
  np.testing.assert_allclose(ground_and_box.pos.numpy()[:, 1], [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], atol=1e-6)
  np.testing.assert_allclose(ground_and_box.pos.numpy()[:, 0], 0.0, atol=1e-6)


def test_usd_visuals_bind_to_bodies_by_prim_path(tmp_path):
  pxr = pytest.importorskip("pxr")
  from pxr import UsdGeom, UsdPhysics

  stage = pxr.Usd.Stage.CreateNew(str(tmp_path / "body.usda"))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  UsdPhysics.Scene.Define(stage, "/physics")
  body = UsdGeom.Xform.Define(stage, "/World/Body")
  UsdGeom.XformCommonAPI(body).SetTranslate((0.0, 0.0, 1.0))
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
  cube = UsdGeom.Cube.Define(stage, "/World/Body/Visual")
  cube.CreateSizeAttr(0.2)
  UsdGeom.XformCommonAPI(cube).SetTranslate((0.3, 0.0, 0.0))
  UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
  ground = UsdGeom.Cube.Define(stage, "/World/Ground")
  UsdGeom.XformCommonAPI(ground).SetScale((5.0, 5.0, 0.1))
  stage.GetRootLayer().Save()

  # The model Newton's USD importer builds: bodies labelled with their prim paths.
  builder = newton.ModelBuilder()
  b = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="/World/Body")
  builder.add_shape_box(b, xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1)
  model = builder.finalize(device="cpu")
  stage = str(tmp_path / "body.usda")
  host = rw_newton.load_usd(model, stage)
  scene = rw.upload(host, model.device, binding=rw_newton.binding(host, model, stage=stage))
  r = rw.Renderer(
    scene,
    nworld=1,
    cameras=[rw.CameraSpec()],
    resolution=(8, 8),
    settings=rw.RenderSettings(outputs=rw.Outputs.of("depth")),
  )
  state = model.state()
  q = state.body_q.numpy()
  q[b, :3] += (0.0, 2.0, 0.0)
  state.body_q.assign(q)
  r.render(state)
  pos = required(r.poses).pos.numpy()[0]  # the body's cube (per world); the ground is shared
  np.testing.assert_allclose(pos[0], [0.3, 2.0, 1.0], atol=1e-5)
  assert r.layout.nshared == 1 and r.layout.compact
  np.testing.assert_allclose(r.debug.frame().shared_pos.numpy()[0], [0.0, 0.0, 0.0], atol=1e-5)


def _scenery_model(nworld: int) -> tuple[newton.Model, newton.State]:
  """Per world: a body box and two static props; global scenery: box, mesh, cylinder, sphere."""
  item = newton.ModelBuilder()
  body = item.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  item.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  item.add_shape_sphere(-1, xform=wp.transform((0.4, 0.3, 0.15), wp.quat_identity()), radius=0.15)
  item.add_shape_capsule(-1, xform=wp.transform((-0.4, 0.3, 0.2), wp.quat_identity()), radius=0.05, half_height=0.2)
  builder = newton.ModelBuilder()
  builder.add_shape_box(-1, xform=wp.transform((0.0, 1.0, 0.5), wp.quat_identity()), hx=1.0, hy=0.05, hz=0.5)
  tetra = wp.transform((-0.5, -0.4, 0.0), wp.quat_identity())
  builder.add_shape_mesh(-1, mesh=_tetra(), xform=tetra, scale=(0.3, 0.3, 0.3))
  builder.add_shape_cylinder(-1, xform=wp.transform((0.6, -0.5, 0.3), wp.quat_identity()), radius=0.1, half_height=0.3)
  builder.add_shape_sphere(-1, xform=wp.transform((-0.7, 0.6, 0.2), wp.quat_identity()), radius=0.12)
  builder.replicate(item, nworld, spacing=(0.0, 0.0, 0.0))
  builder.add_ground_plane()
  model = builder.finalize(device="cpu")
  state = model.state()
  q = state.body_q.numpy()
  q[:, :3] = [(0.05 * w, 0.0, 0.5 + 0.1 * w) for w in range(nworld)]
  state.body_q.assign(q)
  return model, state


def _lights() -> rw.Lighting:
  """A directional and a point light."""
  lights = [rw.Light.sun((-0.4, 0.5, -0.75)), rw.Light.point((0.5, -0.5, 1.5))]
  return rw.Lighting(lights)


def _render(  # noqa: PLR0913  # the scene and keyword options
  scene: rw.GpuScene[NewtonState],
  state: NewtonState,
  nworld: int,
  prepare: Callable[[rw.Renderer[NewtonState]], None] | None = None,
  *,
  per_world: rw.PerWorld | None = None,
  layout: Callable[[Layout], None] | None = None,
) -> np.ndarray:
  """(nworld, pixels) rgba and depth of one camera per world."""
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("rgb", "depth"),
    performance=rw.settings.Performance(traversal="software"),
    shadows=rw.settings.Shadows(enabled=True),
    # No light grid: instances declared per world do not occlude its rays (they may move), so
    # culling differs between shared scenery and per-world copies by design.
    lighting=rw.settings.Lights(light_grid=False),
    exposure=rw.settings.Exposure(mode="manual", ev100=DEFAULT_EV100),  # worlds are compared pixel for pixel
    gi=rw.settings.GI(enabled=False),  # per-world copies vs shared scenery: the direct terms
  )
  r = rw.Renderer(
    scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 16), settings=settings, per_world=per_world
  )
  if layout is not None:
    layout(r.layout)
  rot = wp.quat_from_matrix(wp.mat33(1.0, 0.0, 0.0, 0.0, 0.5, -0.866, 0.0, 0.866, 0.5))
  r.camera_pose.assign(np.tile(np.array([0.0, -2.5, 1.5, *rot], np.float32), (nworld, 1, 1)))
  if prepare is not None:
    prepare(r)
  r.render(state)
  rgba = required(r.outputs.rgb).numpy().reshape(nworld, -1).astype(np.float64)
  return np.concatenate([rgba, required(r.outputs.depth).numpy().reshape(nworld, -1)], axis=1)


def test_shared_scenery_renders_as_per_world_copies():
  """Global static shapes are stored once (layout.nshared) and render as per-world copies."""
  nworld = 3
  model, state = _scenery_model(nworld)
  scene = _gpu(model, lighting=_lights())

  def shared_layout(layout: Layout) -> None:
    assert (layout.nshared, layout.ncolumn) == (4, 4)  # 4 global shapes; box, 2 props and the ground plane

  def copied_layout(layout: Layout) -> None:
    assert layout.nshared == 0

  copies = rw.PerWorld(pose=["*"])
  np.testing.assert_array_equal(
    _render(scene, state, nworld, layout=shared_layout), _render(scene, state, nworld, per_world=copies, layout=copied_layout)
  )

  def randomise(r: rw.Renderer[NewtonState]) -> None:  # per-world appearance of the (shared) scenery and lights
    base = r.per_world["material.base_color"]
    b = base.numpy()
    b[1, :, :3] *= 0.4
    base.assign(b)
    li = r.per_world["light.strength"]
    v = li.numpy()
    v[2] *= 0.5
    li.assign(v)

  looks = {"material": ["base_color"], "light": ["strength"]}
  shared, copied = rw.PerWorld(**looks), rw.PerWorld(pose=["*"], **looks)
  np.testing.assert_array_equal(
    _render(scene, state, nworld, randomise, per_world=shared), _render(scene, state, nworld, randomise, per_world=copied)
  )


def test_moved_global_shape_raises():
  """A global shape moved in model.shape_transform is detected a frame later and raises (shared
  scenery is static)."""
  nworld = 2
  model, state = _scenery_model(nworld)
  scene = _gpu(model, lighting=_lights())
  cylinder = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.CYLINDER))[0])
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software"))
  r = rw.Renderer(scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 16), settings=settings)
  rot = wp.quat_from_matrix(wp.mat33(1.0, 0.0, 0.0, 0.0, 0.5, -0.866, 0.0, 0.866, 0.5))
  r.camera_pose.assign(np.tile(np.array([0.0, -2.5, 1.5, *rot], np.float32), (nworld, 1, 1)))
  r.render(state)
  tf = model.shape_transform.numpy()
  tf[cylinder, 0] -= 0.5
  model.shape_transform.assign(tf)
  r.render(state)
  with pytest.raises(ValueError, match=r"shared scenery .* moved or changed after the first frame"):
    r.render(state)


def test_domain_randomisation_on_shared_scenery():
  """Per-world lights (intensity, color, position) and prop poses leave world 0 unchanged."""
  nworld = 4
  model, state = _scenery_model(nworld)
  cylinder = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.CYLINDER))[0])
  scene = _gpu(model, lighting=_lights())
  binding = scene.binding
  assert isinstance(binding, NewtonBinding)
  inst = int(np.flatnonzero(binding.shape0 == cylinder)[0])
  name = inst  # an instance index (Newton labels may repeat)
  per_world = rw.PerWorld(pose=[name], light=["strength", "color", "pos"])
  base = _render(scene, state, nworld, per_world=per_world)

  def randomise(r: rw.Renderer[NewtonState]) -> None:
    intensity = r.per_world["light.strength"]  # (nworld, nlight): only this field gets per-world rows
    i = intensity.numpy()
    i[1] *= 0.5
    intensity.assign(i)
    color = r.per_world["light.color"]
    c = color.numpy()
    c[2] = (1.0, 0.3, 0.3)
    color.assign(c)
    pos = r.per_world["light.pos"]
    p = pos.numpy()
    p[3, 1] += np.array([0.0, 1.0, 0.0])
    pos.assign(p)
    assert intensity.shape == (nworld, 2) and pos.shape == (nworld, 2)
    # The cylinder was declared per world: move it in world 3.
    assert r.layout.nshared == 3
    cyl = r.per_world.columns("instance.pos", name)
    c = cyl.numpy()
    c[3, 0, 0] += 0.2
    cyl.assign(c)

  out = _render(scene, state, nworld, randomise, per_world=per_world)
  np.testing.assert_array_equal(out[0], base[0])
  assert all(np.any(out[w] != base[w]) for w in range(1, nworld))


def test_host_scene_instances_are_shared_unless_posed_per_world():
  """A host scene's instances are shared unless the Renderer's PerWorld poses them per world."""
  builder = rw.SceneBuilder()
  mat = builder.add_material(rw.Material(base_color=(0.8, 0.8, 0.8)))
  floor = np.eye(4)[:3]
  floor[2, 3] = -0.3
  builder.add_instance("box", material=mat, size=(2.0, 2.0, 0.1), xform=floor)
  builder.add_instance("box", material=mat, size=(0.2, 0.2, 0.2), xform=np.eye(4)[:3])
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0, xform=np.eye(4)[:3]))
  host = builder.build()
  scene = rw.upload(host, device="cpu")
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software"))
  cams = [rw.CameraSpec(fovy=40.0)]
  assert rw.Renderer(scene, nworld=2, cameras=cams, resolution=(8, 8), settings=settings).layout.nshared == 2
  r = rw.Renderer(scene, nworld=2, cameras=cams, resolution=(8, 8), settings=settings, per_world=rw.PerWorld(pose=[1]))
  assert (r.layout.nshared, r.layout.ncolumn) == (1, 1)
  pose = np.zeros((2, 1, 7), np.float32)
  pose[..., 2] = 3.0
  pose[..., 6] = 1.0
  r.camera_pose.assign(pose)
  pos = r.per_world.columns("instance.pos", 1)
  assert pos.shape == (2, 1)
  p = pos.numpy()
  p[1, 0] = (5.0, 0.0, 0.0)  # world 1's box moves out of view
  pos.assign(p)
  depth = required(r.render().depth).numpy().reshape(2, -1)
  assert depth[0].min() < depth[1].min()  # world 0 sees the box, world 1 the floor behind it


def test_png_envmap_is_srgb_decoded(tmp_path):
  """B29: an 8-bit (PNG) environment map holds sRGB values: decoded to linear radiance."""
  from PIL import Image

  path = tmp_path / "sky.png"
  Image.fromarray(np.full((8, 16, 3), 128, np.uint8)).save(path)
  builder = newton.ModelBuilder()
  builder.add_shape_sphere(-1, xform=wp.transform((0.0, 0.0, -50.0), wp.quat_identity()), radius=0.1)
  model = builder.finalize(device="cpu")
  scene = _gpu(model, lighting=rw.Lighting(envmap=str(path), envmap_intensity=1.0))
  # Exposure 1 (ev100 = log2(1 / 1.2)): the HDR output is the environment's radiance.
  settings = rw.RenderSettings(outputs=rw.Outputs.of("hdr")).replace(
    **{"exposure.mode": "manual", "exposure.ev100": float(np.log2(1.0 / 1.2))}
  )
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=30.0)], resolution=(8, 8), settings=settings)
  hdr = required(r.render(model.state()).hdr).numpy()
  expected = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4  # 0.216, not 0.502
  np.testing.assert_allclose(hdr, expected, rtol=0.02)


def test_opacity_zero_shapes_are_not_rendered():
  """B36: a shape of opacity 0 neither shows nor blocks rays."""
  builder = newton.ModelBuilder()
  builder.add_shape_sphere(-1, radius=0.2)
  builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5, opacity=0.0)
  scene = _gpu(builder.finalize(device="cpu"))
  assert scene.ninstance == 1
  assert scene.instance_type.numpy().tolist() == [int(ShapeType.SPHERE)]


def _replicated_usd_model(tmp_path: Path, nworld: int) -> tuple[newton.Model, str]:
  """One body's visual cube in a stage; Newton replicates the body (label = its prim path)."""
  pxr = pytest.importorskip("pxr")
  from pxr import UsdGeom, UsdPhysics

  path = str(tmp_path / "robot.usda")
  stage = pxr.Usd.Stage.CreateNew(path)
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  body = UsdGeom.Xform.Define(stage, "/World/Body")
  UsdGeom.XformCommonAPI(body).SetTranslate((0.0, 0.0, 1.0))
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
  cube = UsdGeom.Cube.Define(stage, "/World/Body/Visual")
  cube.CreateSizeAttr(0.2)
  stage.GetRootLayer().Save()
  item = newton.ModelBuilder()
  b = item.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="/World/Body")
  item.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1, label="/World/Body/Visual")  # imported from the stage
  builder = newton.ModelBuilder()
  builder.replicate(item, nworld, spacing=(0.0, 0.0, 0.0))
  return builder.finalize(device="cpu"), path


def test_merged_usd_scene_draws_every_world(tmp_path):
  """B28: with merge_worlds (one render world), the USD visuals of every Newton world are drawn."""
  model, path = _replicated_usd_model(tmp_path, 3)
  merged = rw_newton.Options(merge_worlds=True)
  host = rw_newton.load_usd(model, path, adapter=merged)
  with pytest.raises(ValueError, match="stage="):
    rw_newton.binding(host, model, adapter=merged)
  bound = rw_newton.binding(host, model, stage=path, adapter=merged)
  scene = rw.upload(host, model.device, binding=bound)
  assert scene.ninstance == 3  # the stage's cube, once per Newton world
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"))
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=settings)
  state = model.state()
  q = state.body_q.numpy()
  q[:, 0] = [0.0, 1.0, 2.0]  # world w's body at x = w
  state.body_q.assign(q)
  r.render(state)
  poses = r.poses
  assert poses is not None
  np.testing.assert_allclose(sorted(poses.pos.numpy()[0, :, 0]), [0.0, 1.0, 2.0], atol=1e-5)


def test_scene_is_a_host_scene_with_the_headlight_as_lights():
  """Adapter Protocol: `scene` ingests only; the headlight is a CAMERA light (its diffuse
  color) and an AMBIENT light (its ambient term at the scene's exposure), uploaded as the
  GpuScene's headlight and constant ambient."""
  builder = newton.ModelBuilder()
  builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5)
  model = builder.finalize(device="cpu")
  ev100 = 2.0
  headlight = rw.Headlight(ambient=(0.1, 0.2, 0.3), diffuse=(0.4, 0.5, 0.6))
  host = rw_newton.load(model, lighting=rw.Lighting(headlight=headlight, ev100=ev100))
  assert isinstance(host, rw.Scene)
  kinds = host.lights.kind.tolist()
  assert kinds == [int(rw.scene.LightKind.CAMERA), int(rw.scene.LightKind.AMBIENT)]
  gpu = rw.upload(host, model.device, binding=rw_newton.binding(host, model))
  assert gpu.nlight == 0  # display lights have no light slot
  assert gpu.headlight.active
  np.testing.assert_allclose(gpu.headlight.color, (0.4, 0.5, 0.6), rtol=1e-6)
  exposure = exposure_from_ev100(ev100)
  np.testing.assert_allclose(gpu.environment.ambient, np.array([0.1, 0.2, 0.3]) / exposure, rtol=1e-6)
  dark = rw.upload(rw_newton.load(model), model.device)
  assert not dark.headlight.active and dark.environment.ambient == (0.0, 0.0, 0.0)


def test_short_world_indices_raise_instead_of_reading_past_them():
  """StateView.world_indices with fewer entries than render worlds: a ValueError (the pose kernel
  read view_world[v] past the array)."""
  model, state = _scenery_model(2)
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"))
  r = rw.Renderer(_gpu(model), nworld=2, cameras=[rw.CameraSpec()], resolution=(4, 4), settings=settings)
  with pytest.raises(ValueError, match="world_indices has 1 entries for 2 render worlds"):
    r.render(rw_newton.StateView(state, world_indices=wp.array([1], dtype=wp.int32)))


def test_merged_scene_bound_per_world_raises():
  """A merge_worlds scene (shapes of every world) bound without merge_worlds: a ValueError, not a
  KeyError from the shape lookup."""
  model, _ = _scenery_model(2)
  host = rw_newton.load(model, adapter=rw_newton.Options(merge_worlds=True))
  with pytest.raises(ValueError, match="merge_worlds=True"):
    rw_newton.binding(host, model)


def test_front_end_options_are_render_settings_keys():
  """The front ends' keyword options are RenderSettings keys; lighting goes through `lighting=`."""
  from raywarp.integrations.newton.options import render_options

  with pytest.raises(TypeError, match="lighting="):
    render_options({"envmap": "sky.hdr"}, (), "test")
  with pytest.raises(TypeError, match="sets \\['outputs'\\] itself"):
    render_options({"outputs": None}, ("outputs",), "test")
  assert render_options({"exposure.ev100": 9.0, "aa": rw.settings.AntiAliasing()}, (), "test")["exposure.ev100"] == 9.0


def test_compose_many_is_bit_equal_to_warp():
  from raywarp.integrations.newton import xform32

  rng = np.random.default_rng(0)

  def poses(n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return np.concatenate([rng.normal(size=(n, 3)) * 10.0, q / np.linalg.norm(q, axis=1, keepdims=True)], axis=1)

  a, b = poses(2000), poses(2000)
  many = xform32.compose_many(a, b)
  one = np.array([xform32.compose(x, y) for x, y in zip(a, b, strict=True)], np.float32)
  np.testing.assert_array_equal(many.view(np.uint32), one.view(np.uint32))


def test_a_scene_of_explicit_shapes_binds_with_the_same_shapes():
  """A scene built with explicit `shapes` (here an invisible box too) binds with the same `shapes`
  (world w's copies found among them); without them the binding says so."""

  def world() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(is_visible=False))
    builder.add_shape_sphere(body, radius=0.2)
    return builder

  scene = newton.ModelBuilder()
  scene.add_world(world())
  scene.add_world(world())
  model = scene.finalize(device="cpu")
  every = range(model.shape_count)
  host = rw_newton.load(model, adapter=rw_newton.Options(shapes=every))
  assert host.ninstance == 2
  bound = rw_newton.binding(host, model, adapter=rw_newton.Options(shapes=every))
  assert rw.upload(host, model.device, binding=bound).ninstance == 2
  with pytest.raises(ValueError, match="bind with the `shapes` and merge_worlds the scene was built with"):
    rw_newton.binding(host, model)


def test_body_quaternions_are_normalised():
  """A non-unit simulator quaternion poses its instances by the rotation it stands for (a zero one
  stays NaN: no silent identity)."""
  builder = newton.ModelBuilder()
  body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
  model = builder.finalize(device="cpu")
  r = rw.Renderer(
    _gpu(model),
    nworld=1,
    cameras=[rw.CameraSpec()],
    resolution=(8, 8),
    settings=rw.RenderSettings(outputs=rw.Outputs.of("depth")),
  )
  state = model.state()
  q = state.body_q.numpy()
  unit = np.array([0.0, 0.0, np.sin(0.4), np.cos(0.4)], np.float32)
  q[body, 3:] = 3.0 * unit  # a quaternion of norm 3
  state.body_q.assign(q)
  r.render(state)
  rot = required(r.poses).rot.numpy()[0, 0]
  expected = np.array(wp.quat_to_matrix(wp.quatf(*unit.tolist())), np.float32).reshape(3, 3)
  np.testing.assert_allclose(rot, expected, atol=1e-6)
  q[body, 3:] = 0.0
  state.body_q.assign(q)
  r.render(state)
  assert np.isnan(required(r.poses).rot.numpy()[0, 0]).all()


def test_shape_indices_out_of_range_raise():
  """A negative index in `Options(shapes=...)` selected a shape from the end, silently; one past
  the shapes failed with a bare NumPy IndexError."""
  from raywarp.integrations.newton import build

  model = _worlds_model([_tetra()])
  for shapes in ([-1], [model.shape_count]):
    with pytest.raises(ValueError, match="out-of-range shape index"):
      build.shape_rows(model, shapes, merge_worlds=True)


def _loaded_executables() -> int:
  """How many module executables Warp holds for raywarp's modules (a recompile adds one)."""
  import sys

  return sum(len(wp.get_module(name).execs) for name in list(sys.modules) if name.startswith("raywarp"))


def test_worlds_render_their_own_shape_colors():
  """Each world shows its shapes' own `shape_color` (it rendered world 0's everywhere), re-read
  on every render: a color written into `model.shape_color` shows on the next render, with no
  recompile (one device kernel re-reads the table)."""

  def world(color: tuple[float, float, float]) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, color=color)
    return builder

  scene = newton.ModelBuilder()
  scene.add_world(world((0.9, 0.1, 0.1)))
  scene.add_world(world((0.1, 0.1, 0.9)))
  model = scene.finalize(device="cpu")
  gpu = rw_newton.scene(model)
  assert gpu.report.dropped.get("newton.shape_color(per world: world 0's shown)", 0) == 0
  settings = rw.RenderSettings(outputs=rw.Outputs.of("albedo"), gi=rw.settings.GI(enabled=False))
  cam = rw.CameraSpec(fovy=30.0)
  r = rw.Renderer(gpu, nworld=2, cameras=[cam], resolution=(8, 8), settings=settings)
  pose = r.camera_pose.numpy()
  pose[:] = (0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0)  # above the boxes, looking down -Z
  r.camera_pose.assign(pose)
  state = model.state()

  def centre() -> np.ndarray:
    return required(r.render(state).albedo).numpy()[:, 0, 4, 4]  # (nworld, 3) linear albedo

  from raywarp.core.post.grading.color import srgb_to_linear as linear

  np.testing.assert_allclose(centre(), linear(np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])), atol=1e-4)
  features, executables = r.features, _loaded_executables()
  colors = model.shape_color.numpy()
  colors[1] = (0.1, 0.8, 0.1)  # world 1's box turns green
  model.shape_color.assign(colors)
  np.testing.assert_allclose(centre(), linear(np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.1]])), atol=1e-4)
  assert r.features is features and _loaded_executables() == executables


def test_world_colors_off_keeps_world_0s_and_reports_it():
  """`Options(world_colors=False)` (Isaac Lab, which colors per world itself) keeps world 0's
  material colors and reports the difference."""

  def world(color: tuple[float, float, float]) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, color=color)
    return builder

  scene = newton.ModelBuilder()
  scene.add_world(world((1.0, 0.0, 0.0)))
  scene.add_world(world((0.0, 0.0, 1.0)))
  model = scene.finalize(device="cpu")
  adapter = rw_newton.Options(world_colors=False)
  gpu = rw_newton.scene(model, adapter=adapter)
  assert gpu.report.dropped["newton.shape_color(per world: world 0's shown)"] == 1
  assert rw_newton.build.WORLD_COLORS_META not in rw_newton.load(model, adapter=adapter).meta


def test_renders_the_bodies_newtons_mujoco_solver_steps():
  """Newton's SolverMuJoCo (MuJoCo Warp) steps the bodies' free joints and writes `state.body_q`:
  a box dropped from 1 m in each of two worlds (its own start height per world, set in the
  state's `joint_q`, what the solver steps from) lands on the ground, seen from above."""
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  for _ in range(2):
    world = newton.ModelBuilder()
    body = world.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    world.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
    scene.add_world(world)
  model = scene.finalize(device="cpu")
  s0, s1, control = model.state(), model.state(), model.control()
  q = s0.joint_q.numpy().reshape(2, 7)
  q[1, 2] = 1.5
  s0.joint_q.assign(q.reshape(-1))
  newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), gi=rw.settings.GI(enabled=False))
  r = rw.Renderer(_gpu(model), nworld=2, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(5, 5), settings=settings)
  r.camera_pose.assign(np.array([[(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0)]] * 2, np.float32))

  def top() -> np.ndarray:
    r.render(s0)
    return 4.0 - required(r.outputs.depth).numpy().reshape(2, 5, 5)[:, 2, 2]

  np.testing.assert_allclose(top(), (1.2, 1.7), atol=1e-3)
  solver = newton.solvers.SolverMuJoCo(model)
  for _ in range(300):  # 1.5 s
    solver.step(s0, s1, control, None, 0.005)
    s0, s1 = s1, s0
  np.testing.assert_allclose(top(), 0.4, atol=0.02)
