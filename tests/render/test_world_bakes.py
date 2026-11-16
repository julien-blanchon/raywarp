"""Independent worlds must never inherit another world's baked geometry or light culling."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.pipeline.baked import _bake_dynamic
from raywarp.core.shade import gi as gi_lib
from raywarp.core.shade.lights import flood
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights.punctual import Light


def _room() -> rw.Scene:
  """Static floor/wall, a movable red block and shared illumination."""
  b = rw.SceneBuilder()
  gray = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6), specular_roughness=1.0))
  red = b.add_material(rw.Material(base_color=(0.9, 0.02, 0.02), specular_roughness=1.0))
  b.add_instance("box", material=gray, size=(1.0, 1.0, 0.05), path="floor")
  b.add_instance("box", material=gray, size=(1.0, 0.05, 0.7), xform=np.c_[np.eye(3), [0.0, 1.0, 0.7]], path="wall")
  b.add_instance("box", material=red, size=(0.3, 0.3, 0.3), xform=np.c_[np.eye(3), [0.0, 0.0, 0.35]], path="moving")
  b.add_environment(rw.scene.EnvironmentLight(intensity=20.0))
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=100.0, radius=0.05, xform=np.c_[np.eye(3), [0.0, 0.0, 1.8]]))
  return b.build()


def _renderer() -> rw.Renderer[None]:
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "performance.traversal": "software",
      "gi.enabled": True,
      "gi.spacing": 0.5,
      "gi.max_probes": 32,
      "gi.rays": 16,
      "gi.bake_updates": 2,
      "gi.update_every": 1,
      "reflections.enabled": False,
      "lighting.light_grid": False,
    }
  )
  r = rw.Renderer(
    rw.upload(_room(), device="cpu"),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=60.0)],
    resolution=(8, 8),
    settings=settings,
    per_world=rw.PerWorld(pose=["moving"]),
  )
  r.camera_pose.assign(np.tile(np.array([0, 0, 2, 0, 0, 0, 1], np.float32), (2, 1, 1)))
  return r


def test_static_bakes_ignore_other_world_pose() -> None:
  """Fresh bake bounds, relocation and radiance exclude a distant world-0-only block."""
  images: list[np.ndarray[[int, int, int, 3], np.dtype[np.float32]]] = []
  origins: list[tuple[float, ...]] = []
  for x in (0.0, 20.0):
    with _renderer() as r:
      r.per_world.columns("instance.pos", "moving").assign(np.array([[[x, 0, 0.35]], [[0.6, 0, 0.35]]], np.float32))
      r.render()
      ctx = r.context()
      assert ctx.gi is not None
      origins.append(tuple(ctx.gi.struct.origin))
      # Also compare a periodic update after moving world 0 again.
      r.per_world.columns("instance.pos", "moving").assign(np.array([[[x + 2, 0, 0.35]], [[0.6, 0, 0.35]]], np.float32))
      out = r.render().hdr
      assert out is not None
      images.append(out.numpy()[1].copy())
  assert origins[0] == origins[1]
  np.testing.assert_array_equal(images[0], images[1])


def test_texture_uniform_variation_excludes_shared_occluders() -> None:
  """Texture projection variation can move alpha cutouts even when geometry stays shared."""
  with _renderer() as r:
    ctx, frame = r.context(), r.debug.frame()
    assert not _bake_dynamic(ctx, frame).all()
    declared = dataclasses.replace(ctx, per_world_fields=ctx.per_world_fields | {"material.tex_uniform"})
    assert _bake_dynamic(declared, frame).all()
    varying = wp.zeros((2, frame.material_tex_uniform.shape[1]), dtype=bool, device="cpu")
    live = dataclasses.replace(frame, material_tex_uniform=varying)
    assert _bake_dynamic(ctx, live).all()


def test_plane_only_grid_uses_light_bounds() -> None:
  """MuJoCo's zero-radius plane must not collapse the grid to an unrelated origin cell."""
  boxes = flood.Boxes(
    active=np.ones(1, bool),
    directional=np.zeros(1, bool),
    centre=np.array([[8.0, 0.0, 1.0]], np.float64),
    extent=np.zeros(1, np.float64),
  )
  # Baked._build_grid converts all plane bounds (including radius=0) to infinity.
  lo, hi = light_grid._bounds(np.zeros((1, 3), np.float32), np.full(1, np.inf, np.float32), boxes, 1.0)
  assert lo[0] < 8.0 < hi[0]
  assert np.all(hi > lo)


def test_live_light_grid_validity_is_per_row_and_reversible() -> None:
  """Simulator writes need no declaration: a moved row falls back, then becomes valid on return."""
  light = Light()
  light.active, light.pos, light.range = True, wp.vec3(0.0, 0.0, 1.0), 3.0
  snapshot = wp.array([light], dtype=Light, device="cpu").reshape((1, 1))
  current = wp.array([light, light], dtype=Light, device="cpu").reshape((2, 1))
  grid = light_grid.assemble(
    origin=wp.vec3(0.0), cell=1.0, dims=wp.vec3i(1, 1, 1), start=np.zeros(2, np.int32), lights=np.zeros(0, np.int32),
    dynamic=np.zeros(0, np.int32), snapshot=snapshot, far=np.zeros((0, 3), np.float32),
    far_members=np.zeros(1, np.int32), nlight=1,
  )  # fmt: skip
  moved = Light()
  moved.active, moved.pos, moved.range = True, wp.vec3(4.0, 0.0, 1.0), 3.0
  wp.copy(current, wp.array([light, moved], dtype=Light, device="cpu").reshape((2, 1)))
  grid = light_grid.checked(grid, current, 1)
  np.testing.assert_array_equal(grid.valid.numpy()[:2], [1, 0])
  wp.copy(current, wp.array([light, light], dtype=Light, device="cpu").reshape((2, 1)))
  grid = light_grid.checked(grid, current, 1)
  np.testing.assert_array_equal(grid.valid.numpy()[:2], [1, 1])
  # A recoloured row keeps its reach (its lists), not its far fields (built from the bake's colors).
  tinted = Light()
  tinted.active, tinted.pos, tinted.range, tinted.color = True, wp.vec3(0.0, 0.0, 1.0), 3.0, wp.vec3(1.0, 0.0, 0.0)
  wp.copy(current, wp.array([light, tinted], dtype=Light, device="cpu").reshape((2, 1)))
  grid = light_grid.checked(grid, current, 1)
  np.testing.assert_array_equal(grid.valid.numpy(), [1, 1, 1, 0])


def test_gi_inputs_include_live_simulator_tables() -> None:
  """Per-world GI tracks the radiance inputs (lights, materials, environment), not poses."""
  assert gi_lib.gi_slot("light.pos") and gi_lib.gi_slot("env.id") and gi_lib.gi_slot("material.specular_roughness")
  assert not gi_lib.gi_slot("instance.pos") and not gi_lib.gi_slot("camera.pos")


@pytest.mark.parametrize("source", ["declared", "live_initial", "live_later"])
def test_renderer_keeps_gi_per_world(source: str) -> None:
  """Declarations and live batched bindings keep GI per world (`shade.gi.groups`; two distinct
  worlds: a relit volume, `gi.Relight`), including sources that appear after the first bake
  (re-baked)."""
  gpu = rw.upload(_room(), device="cpu")
  positions = wp.array([[(0.0, 0.0, 1.8)], [(1.0, 0.0, 1.8)]], dtype=wp.vec3, device="cpu")
  if source == "live_initial":
    gpu.tables["light.pos"] = positions
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "performance.traversal": "software",
      "gi.enabled": True,
      "lighting.light_grid": False,
      "reflections.enabled": False,
    }
  )
  per_world = rw.PerWorld(light=["strength"]) if source == "declared" else None
  with rw.Renderer(
    gpu, nworld=2, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(4, 4), settings=settings, per_world=per_world
  ) as r:
    assert r.features.gi
    r.render()
    if source == "live_later":
      vol = r.debug.gi_volume
      assert vol is not None and not vol.groups.tracked
      gpu.tables["light.pos"] = positions
      r.render()
    r.update_gi()
    vol = r.debug.gi_volume
    assert vol is not None and vol.ngroup == 1
    # Declared but equal rows: one tracked group; two distinct live rows: relit.
    assert vol.groups.tracked if source == "declared" else vol.relight is not None
    assert "gpu.gi.per_world_inputs" not in gpu.report.dropped


def test_live_light_motion_matches_unculled_render() -> None:
  """Moved simulator-style light arrays are handled in this frame without a per-world override."""
  gpu = rw.upload(_room(), device="cpu")
  positions = wp.array([[(0.0, 0.0, 1.8)], [(0.0, 0.0, 1.8)]], dtype=wp.vec3, device="cpu")
  gpu.tables["light.pos"] = positions
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "performance.traversal": "software",
      "gi.enabled": False,
      "lighting.light_grid": True,
      "lighting.light_grid_cell": 0.5,
      "shadows.max_casters": None,
      "reflections.enabled": False,
    }
  )
  with rw.Renderer(gpu, nworld=2, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(8, 8), settings=settings) as r:
    r.camera_pose.assign(np.tile(np.array([0, 0, 2, 0, 0, 0, 1], np.float32), (2, 1, 1)))
    initial = r.render().hdr
    assert initial is not None
    grid = r.context().light_grid
    assert r.features.culling and grid is not None  # Explicit True enables even a one-light grid.
    np.testing.assert_array_equal(grid.valid.numpy()[:2], [1, 1])
    world1 = initial.numpy()[1].copy()
    positions.assign(np.array([[(20.0, 0.0, 1.8)], [(0.0, 0.0, 1.8)]], np.float32))
    moved = r.render().hdr
    assert moved is not None
    np.testing.assert_array_equal(grid.valid.numpy()[:2], [0, 1])
    np.testing.assert_array_equal(moved.numpy()[1], world1)
    positions.assign(np.array([[(20.0, 0.0, 1.8)], [(0.7, -0.3, 1.2)]], np.float32))
    moved = r.render().hdr
    assert moved is not None
    with_grid = moved.numpy().copy()
    r.update_settings(**{"lighting.light_grid": False})
    exact = r.render().hdr
    assert exact is not None
    np.testing.assert_allclose(with_grid, exact.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("field", ["material.base_color", "instance.color"])
def test_color_randomization_keeps_shared_occluders(field: str) -> None:
  """BUG-002: per-world colors (plain color DR) used to make every instance dynamic, so the
  bake traced an empty scene; colors no longer count, cut-out inputs still do."""
  with _renderer() as r:
    ctx, frame = r.context(), r.debug.frame()
    colored = dataclasses.replace(ctx, per_world_fields=ctx.per_world_fields | {field})
    assert not _bake_dynamic(colored, frame).all()
    cutoff = dataclasses.replace(ctx, per_world_fields=ctx.per_world_fields | {"material.alpha_cutoff"})
    assert _bake_dynamic(cutoff, frame).all()
