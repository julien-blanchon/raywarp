"""The per-world instance optics (`core/pipeline/worlds/optics.py`): the device composition equals
the upload's (`pipeline/upload/optics.compose`) for every world's material choice and refraction
indices."""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp

import raywarp as rw
from raywarp._base.types import CLOSED_PANE, TWO_SIDED
from raywarp.core.pipeline.upload.optics import compose, see_through
from raywarp.core.pipeline.worlds.optics import OpticsTables, WorldOptics, every_closed
from raywarp.scene.model.types import AlphaMode, InstanceFlags


def _scene() -> rw.Scene:
  """Volume glass (closed box and ball: chords), a thin pane material, an alpha-blended tint, an
  opaque one; a doubled closed box (a pane candidate), a doubled quad, a see-through tinted
  instance color."""
  b = rw.SceneBuilder()
  glass = b.add_material(
    rw.Material(transmission_weight=1.0, transmission_color=(0.5, 0.8, 0.6), specular_ior=1.5, specular_roughness=0.0)
  )
  thin = b.add_material(rw.Material(transmission_weight=1.0, transmission_color=(0.9, 0.9, 0.7), geometry_thin_walled=True))
  blend = b.add_material(
    rw.Material(base_color=(0.3, 0.6, 0.9), geometry_opacity=0.4, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.BLEND))
  )
  opaque = b.add_material(rw.Material(base_color=(0.8, 0.2, 0.1)))
  eye = np.eye(3)
  doubled = int(InstanceFlags.DEFAULT | InstanceFlags.DOUBLE_SIDED)
  b.add_instance("box", material=glass, size=(0.3, 0.2, 0.1), xform=np.c_[eye, [0.0, 0.0, 0.3]])
  b.add_instance("sphere", material=glass, size=(0.25, 0.25, 0.25), xform=np.c_[eye, [1.0, 0.0, 0.3]])
  b.add_instance("box", material=blend, size=(0.2, 0.2, 0.2), xform=np.c_[eye, [2.0, 0.0, 0.3]], flags=doubled)
  b.add_instance("plane", material=thin, size=(0.5, 0.5, 0.0), xform=np.c_[eye, [3.0, 0.0, 0.0]], flags=doubled)
  b.add_instance("box", material=opaque, size=(0.2, 0.2, 0.2), xform=np.c_[eye, [4.0, 0.0, 0.3]], color=(0.5, 1.0, 0.5, 0.5))
  b.add_instance("box", material=opaque, size=(0.2, 0.2, 0.2), xform=np.c_[eye, [5.0, 0.0, 0.3]])
  return b.build()


def test_device_composition_equals_the_upload_s_in_every_world() -> None:
  """Every material (and none) on every instance across worlds, two refraction indices: the
  shadow transmittance, optics rows and two-sided bits of each world equal `compose`'s, for each
  `shadows.translucent` mode; one world with the authored materials equals the upload's tables."""
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(_scene())
    inputs = gpu.optics_inputs
    assert inputs is not None
    n, m = gpu.ninstance, gpu.nmaterial
    authored = gpu.tables["instance.material"].numpy()[0]
    choices = np.stack([authored, *[np.roll(np.arange(m + 1) - 1, k)[np.arange(n) % (m + 1)] for k in range(m + 1)]]).astype(
      np.int32
    )
    ior = np.stack([np.full(m, 1.5, np.float32), np.full(m, 1.33, np.float32)])[np.arange(len(choices)) % 2]
    ior[0] = np.asarray(gpu.derive_inputs.values["material.specular_ior"], np.float32)
    optics = WorldOptics(inputs, len(choices), n, gpu.device)
    color, base = gpu.tables["instance.color"], gpu.tables["material.base_color"]
    for mode in ("transmittance", "clear", "off"):
      optics.write(OpticsTables(wp.array(choices, dtype=int), color, base, wp.array(ior, dtype=float)), mode)
      w_, n_ = len(choices), n
      shadow = optics.shadow.numpy().reshape(w_, n_, 3)
      rows = optics.optics.numpy().reshape(w_, n_, -1, 4)
      bits = optics.twosided.numpy().reshape(w_, n_)
      for w, (ids, eta) in enumerate(zip(choices, ior, strict=True)):
        expected = compose(inputs, ids, eta)
        t = {"transmittance": expected.shadow, "off": 0.0 * expected.shadow}.get(
          mode, (expected.shadow > 0).any(1, keepdims=True).repeat(3, 1).astype(np.float32)
        )
        np.testing.assert_allclose(shadow[w], t, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(rows[w], expected.rows, rtol=1e-6)
        np.testing.assert_array_equal(bits[w], expected.twosided)
    optics.write(OpticsTables(wp.array(choices[:1], dtype=int), color, base, wp.array(ior[:1], dtype=float)), "transmittance")
    np.testing.assert_allclose(optics.shadow.numpy()[:n], gpu.instance_shadow_transmit, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(optics.optics.numpy()[:n], gpu.instance_optics, rtol=1e-6)
    np.testing.assert_array_equal(optics.twosided.numpy()[:n], gpu.instance_twosided)
    # The scene exercises every bit: solid (two-sided), a closed pane, a doubled open quad.
    every = np.concatenate([compose(inputs, ids, eta).twosided for ids, eta in zip(choices, ior, strict=True)])
    assert {1, 2} <= set(every.tolist())


def test_device_composition_follows_each_world_s_colors() -> None:
  """Per-world `instance.color` (alpha) and `material.base_color` (tint, opacity): each world's
  tables equal `compose` of that world's colors (the blended materials' see-through part
  recomputed from the world's base color), not the upload's."""
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(_scene())
    inputs = gpu.optics_inputs
    assert inputs is not None
    n, m = gpu.ninstance, gpu.nmaterial
    rng = np.random.default_rng(3)
    color = np.repeat(inputs.color[None], 3, 0)
    color[1, :, 3] = rng.uniform(0.2, 1.0, n)
    color[2, 4] = 1.0  # the tinted instance turns white: the material's own transmittance
    base = np.repeat(inputs.base[None], 3, 0)
    base[1:, :, :3] = rng.uniform(0.1, 1.0, (2, m, 3))
    base[2, :, 3] = rng.uniform(0.1, 1.0, m)
    material = np.repeat(inputs.material[None], 3, 0)
    ior = np.repeat(np.asarray(gpu.derive_inputs.values["material.specular_ior"], np.float32)[None], 3, 0)
    optics = WorldOptics(inputs, 3, n, gpu.device)
    tables = OpticsTables(
      wp.array(material, dtype=int), wp.array(color, dtype=wp.vec4), wp.array(base, dtype=wp.vec4), wp.array(ior, dtype=float)
    )
    optics.write(tables, "transmittance")
    shadow, bits = optics.shadow.numpy().reshape(3, n, 3), optics.twosided.numpy().reshape(3, n)
    for w in range(3):
      world = dataclasses.replace(inputs, color=color[w], base=base[w], see=see_through(base[w], inputs.see_alpha))
      expected = compose(world, material[w], ior[w])
      np.testing.assert_allclose(shadow[w], expected.shadow, rtol=1e-5, atol=1e-7)
      np.testing.assert_array_equal(bits[w], expected.twosided)
    assert not np.allclose(shadow[1], shadow[0]) and not np.allclose(shadow[2], shadow[0])


def _cube() -> tuple[np.ndarray, np.ndarray]:
  """A closed unit cube mesh (points, triangles)."""
  p = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32)
  quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
  return p, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int32)


def test_a_doubled_mesh_switched_to_see_through_is_a_closed_pane() -> None:
  """A doubled closed mesh authored opaque (the upload does not test its shape): switched per
  world to an alpha-blended material, it is a closed pane (`every_closed`), not a two-sided
  sheet; kept opaque, two-sided."""
  b = rw.SceneBuilder()
  opaque = b.add_material(rw.Material(base_color=(0.8, 0.2, 0.1)))
  blend = b.add_material(
    rw.Material(base_color=(0.3, 0.6, 0.9), geometry_opacity=0.4, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.BLEND))
  )
  mesh = b.add_mesh(*_cube())
  doubled = int(InstanceFlags.DEFAULT | InstanceFlags.DOUBLE_SIDED)
  b.add_instance("mesh", mesh=mesh, material=opaque, xform=np.c_[np.eye(3) * 2.0, [0.0, 0.0, 0.5]], flags=doubled)
  b.add_instance("box", material=blend, size=(0.2, 0.2, 0.2), xform=np.c_[np.eye(3), [2.0, 0.0, 0.3]])
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(b.build())
    inputs = gpu.optics_inputs
    assert inputs is not None and not inputs.closed[0]
    closed = every_closed(gpu, inputs)
    assert closed.closed[0] and closed.closed[1]
    optics = WorldOptics(closed, 2, gpu.ninstance, gpu.device)
    material = wp.array(np.array([[opaque, blend], [blend, blend]], np.int32), dtype=int)
    ior = wp.array(np.asarray(gpu.derive_inputs.values["material.specular_ior"], np.float32)[None], dtype=float)
    color, base = gpu.tables["instance.color"], gpu.tables["material.base_color"]
    optics.write(OpticsTables(material, color, base, ior), "transmittance")
    bits = optics.twosided.numpy().reshape(2, -1)
  assert bits[0, 0] == TWO_SIDED and bits[1, 0] == CLOSED_PANE


def _room(*, glass: bool, blend: bool = False) -> rw.Scene:
  """A floor plane and three boxes ("table", "cup", "vase") of opaque materials; with `glass` the
  vase is solid glass, with `blend` the cup an alpha-blended tint (see-through, not
  transmissive)."""
  b = rw.SceneBuilder()
  red = b.add_material(rw.Material(base_color=(0.8, 0.2, 0.1)))
  blue = b.add_material(rw.Material(base_color=(0.1, 0.2, 0.8)))
  cup = blue
  if blend:
    cup = b.add_material(
      rw.Material(base_color=(0.3, 0.6, 0.9), geometry_opacity=0.4, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.BLEND))
    )
  vase = b.add_material(rw.Material(transmission_weight=1.0, specular_ior=1.5)) if glass else red
  eye = np.eye(3)
  b.add_instance("plane", material=red, size=(3.0, 3.0, 0.1), path="floor")
  for k, (name, m) in enumerate((("table", red), ("cup", cup), ("vase", vase))):
    b.add_instance("box", material=m, size=(0.2, 0.2, 0.2), xform=np.c_[eye, [float(k), 0.0, 0.3]], path=name)
  b.add_light(rw.Light.sun((0.4, 0.2, -0.9), illuminance=5.0))
  return b.build()


def _sharing(scene: rw.Scene, material: list[str]) -> tuple[bool, list[bool]]:
  """(Features.world_optics, the per-world mask by instance) of a 2-world renderer declaring
  `material` per world."""
  with wp.ScopedDevice("cpu"):
    r = rw.Renderer(
      rw.upload(scene),
      nworld=2,
      cameras=[rw.CameraSpec(fovy=60.0)],
      resolution=(8, 8),
      per_world=rw.PerWorld(material=material),
    )
    try:
      return r.features.world_optics, r.layout.per_world.tolist()
    finally:
      r.close()


def test_opaque_material_choice_keeps_the_shared_scenery() -> None:
  """Randomising among opaque materials (`instance.material`, colours, roughness) switches no
  optics: the base's build and full sharing."""
  assert _sharing(_room(glass=False), ["instance.material", "base_color", "specular_roughness", "color"]) == (
    False,
    [False] * 4,
  )


def test_authored_glass_with_appearance_randomisation_keeps_the_shared_scenery() -> None:
  """An Office-like scene with authored glass and see-through tints whose other objects get
  per-world colours and roughness (no material choice, no refraction index): base behaviour."""
  assert _sharing(_room(glass=True, blend=True), ["base_color", "base_metalness", "specular_roughness", "color"]) == (
    False,
    [False] * 4,
  )


def test_material_choice_with_glass_switches_every_instance() -> None:
  """`instance.material` in a scene with a glass material: any instance can turn glass in any
  world (two-sided, a translucent caster): per-world optics, every non-plane instance per world."""
  assert _sharing(_room(glass=True), ["instance.material", "base_color"]) == (True, [False, True, True, True])


def test_refraction_index_alone_switches_the_solid_instances() -> None:
  """`specular_ior` per world, the authored materials kept: only the solid glass vase's flags can
  differ between worlds (per world); the alpha-blended cup's tables cannot change, the opaque
  objects' neither."""
  assert _sharing(_room(glass=True, blend=True), ["specular_ior"]) == (True, [False, False, False, True])


def test_refraction_index_without_glass_switches_nothing() -> None:
  """`specular_ior` per world in a scene whose only see-through surface is an alpha-blended tint
  (no solid or transmissive material): it changes no optics, base behaviour."""
  assert _sharing(_room(glass=False, blend=True), ["specular_ior"]) == (False, [False] * 4)


def _glass_on(shape: str) -> rw.Scene:
  """One solid glass instance of type `shape` (a primitive, or "mesh": a closed tetrahedron) over
  an opaque box."""
  b = rw.SceneBuilder()
  glass = b.add_material(rw.Material(transmission_weight=1.0, specular_ior=1.5, specular_roughness=0.0))
  opaque = b.add_material(rw.Material(base_color=(0.8, 0.2, 0.1)))
  eye = np.eye(3)
  at = np.c_[eye, [0.0, 0.0, 0.3]]
  if shape == "mesh":
    tet = b.add_mesh([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    b.add_instance("mesh", mesh=tet, material=glass, xform=at)
  else:
    b.add_instance(shape, material=glass, size=(0.25, 0.25, 0.25), xform=at)
  b.add_instance("box", material=opaque, size=(1.0, 1.0, 0.05), xform=np.c_[eye, [0.0, 0.0, -0.05]])
  b.add_light(rw.Light.sun((0.4, 0.2, -0.9), illuminance=5.0))
  return b.build()


def test_glass_shapes_resolve_from_solid_glass_spheres_and_boxes_on_optix() -> None:
  """`Features.glass_shapes` (the OptiX layer loop's analytic exit, `Tracer.leave`, and the
  ambient past the layer budget): on with a solid glass primitive, off with solid glass meshes
  only; a tier build's word bit carries the scene's value (OptiX: bound by the module; software:
  its modules' literal)."""
  from conftest import internal_module

  import tools.prebuilt.fill  # noqa: F401  # puts the development tree's compile lab (`common`) on the path
  from raywarp.core.build.variants import tiers
  from raywarp.core.kernels.abi import toggles

  common = internal_module("common")
  s = rw.RenderSettings(performance=rw.settings.Performance(traversal="optix"))
  for shape, expect in (("sphere", True), ("box", True), ("capsule", True), ("ellipsoid", True), ("mesh", False)):
    with wp.ScopedDevice("cpu"):
      gpu = rw.upload(_glass_on(shape))
    res = common.probe(shape, gpu, [rw.CameraSpec()]).resolution(s, "rtx_pro_6000")
    assert res.effective.refraction and res.effective.glass_shapes is expect, shape
    assert res.features.glass_shapes and res.features.bound & toggles.high(toggles.GLASS_SHAPES)
    assert bool(tiers.word(res.features, res.effective) & toggles.high(toggles.GLASS_SHAPES)) is expect, shape
    soft = common.probe(shape, gpu, [rw.CameraSpec()]).resolution(s.replace(**{"performance.traversal": "software"}), "a100")
    # Software: a module literal of the tier build's word (`tiers._with_word`), the scene's value.
    assert soft.effective.refraction and soft.effective.glass_shapes is expect and soft.features.glass_shapes
    assert bool(soft.features.word & toggles.high(toggles.GLASS_SHAPES)) is expect, shape
