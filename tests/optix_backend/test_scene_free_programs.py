"""The OptiX programs of a scene-free build (`programs.scene_free_flags`): one PTX per geometry
kinds class (the triangle kinds, or with boxes, curves and custom primitives:
`tiers.geometry_kinds`) for every scene, the pipeline linking the scene's own program groups
only (GPU-free: keys and generated source)."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest
import warp as wp
from conftest import RECIPE_NAMES, recipe
from fixtures.optix_codegen import cuda_source, preset_features, scene_flags

import raywarp as rw
from raywarp.core.kernels.abi.features import PLANES_SOME
from raywarp.core.trace.context import FrameContext
from raywarp.core.trace.optix.device import intrinsics as optix_device
from raywarp.core.trace.optix.device import programs
from raywarp.core.trace.optix.device import trace as rt_trace

# The geometry kinds of three scenes: boxes only; meshes, cylinders and cut-outs; every kind.
SCENES = (
  scene_flags(has_mesh=False, primitive_types=(), has_world_curves=False),
  scene_flags(has_box=False, primitive_types=(5,), has_world_curves=False, has_omm=True),
  scene_flags(),
)


@pytest.fixture(scope="module")
def features():
  """The fast field set's Features of the parity scene as a scene-free OptiX build (the alpha test
  compiled in, as every tier build's: `tiers.SCENE_FREE` "alpha_cutout")."""
  return dataclasses.replace(preset_features(recipe("fast")), scene_free=True, alpha_cutout=True)


@pytest.fixture
def _headers(tmp_path, monkeypatch):
  """`device.build_options` only checks that the SDK header exists: codegen never reads it."""
  (tmp_path / "optix_device.h").write_text("")
  monkeypatch.setattr(optix_device, "INCLUDE_DIR", tmp_path)


def test_one_ptx_key_for_every_geometry_kind(features):
  """A scene-free build's PTX key is the same whatever geometry kinds the scene has; a specialised
  build's follows them."""
  keys = {programs.ptx_key(features, programs.program_flags(features, s), supersampled=False, arch=120) for s in SCENES}
  assert len(keys) == 1
  own = dataclasses.replace(features, scene_free=False)
  assert len({programs.ptx_key(own, programs.program_flags(own, s), supersampled=False, arch=120) for s in SCENES}) == 3


def test_scene_free_module_holds_every_program(features, _headers):
  """The module holds the closest-hit, any-hit, curve and every custom primitive's intersection
  programs, so that any scene's pipeline links from it."""
  flags = programs.program_flags(features, SCENES[0])
  with wp.ScopedDevice("cpu"):
    module = programs.build(features, flags)
    cuda_source(module)
  names = {k.key for k in module._get_live_kernels()}  # Warp's module contents
  want = {"__closesthit__triangle", "__closesthit__custom", "__closesthit__curve", "__anyhit__triangle"}
  want |= {programs.intersection_name(t) for t in rt_trace.CUSTOM_TYPES}
  assert want <= names, want - names


def test_alpha_test_runs_in_every_hit_program(features, _headers):
  """The alpha test runs where each kind is found, so one PTX serves cut-outs on any kind: the
  triangles' and curves' any-hit programs, the custom primitives' intersection programs (testing
  both crossings unculled on occlusion rays) and the plane queries; the flags hold no form of it."""
  assert "cut_anyhit" not in programs.SceneFlags.__annotations__
  f = dataclasses.replace(features, planes=PLANES_SOME)
  flags = programs.program_flags(f, SCENES[2])
  with wp.ScopedDevice("cpu"):
    module = programs.build(f, flags)
    source = cuda_source(module)
  names = {k.key for k in module._get_live_kernels()}  # Warp's module contents
  assert {"__anyhit__triangle", "__anyhit__curve"} <= names
  assert "mjpbr_optix::first_hit_ray(" in source
  assert "make_word_cut_away" in source  # the planes' test, behind the opacity-map bit


def test_triangle_kinds_build_leaves_the_primitives_out(features, _headers):
  """A build of the triangle kinds (no primitive in its shape types) compiles no box, curve or
  custom-primitive code and has a PTX key of its own, the same for every triangle scene."""
  from raywarp.core.build.variants import tiers

  f = dataclasses.replace(
    features, shape_types=tiers.geometry_kinds((7,), features.shape_types, every=False, flex=False, inlined=False)
  )
  assert not tiers.PRIMITIVE_TYPES & set(f.shape_types)
  triangles = (scene_flags(has_box=False, primitive_types=(), has_world_curves=False), scene_flags(has_mesh=False, **_NONE))
  flags = [programs.program_flags(f, s) for s in triangles]
  assert not flags[0].has_box and not flags[0].has_world_curves and not flags[0].primitive_types
  keys = {programs.ptx_key(f, fl, supersampled=False, arch=120) for fl in flags}
  every = programs.ptx_key(features, programs.program_flags(features, SCENES[2]), supersampled=False, arch=120)
  assert len(keys) == 1
  assert every not in keys
  with wp.ScopedDevice("cpu"):
    module = programs.build(f, flags[0])
    cuda_source(module)
  names = {k.key for k in module._get_live_kernels()}  # Warp's module contents
  assert "__closesthit__triangle" in names
  assert not {n for n in names if n.startswith(("__intersection__", "__closesthit__curve"))}


def test_pipeline_flags_follow_the_build_s_classes(features):
  """Opacity micromaps are allowed and the translucent casters' any-hit compiled only in a build
  of the class that has them (the alpha test, translucent shadows: the surface class), whatever
  the scene's structures; an opaque build allows no micromaps (W6-S: every build allowed them), nor
  one over a scene without micromaps (a pipeline option outside the PTX key)."""
  cut = programs.scene_free_flags(features, SCENES[1])
  assert cut.has_omm and not cut.translucent_triangles
  plain = programs.scene_free_flags(features, SCENES[2])
  assert not plain.has_omm
  key = {
    programs.ptx_key(features, flags, supersampled=False, arch=89) for flags in (cut, dataclasses.replace(cut, has_omm=False))
  }
  assert len(key) == 1
  shadows = dataclasses.replace(features, translucent_shadows=True)
  assert programs.scene_free_flags(shadows, SCENES[0]).translucent_triangles
  opaque = dataclasses.replace(features, alpha_cutout=False)
  assert {programs.scene_free_flags(opaque, s).has_omm for s in SCENES} == {False}


# A scene of planes only: no box, curve or custom primitive.
_NONE: dict[str, object] = {"has_box": False, "primitive_types": (), "has_world_curves": False}


def _tier_build(recipe_name: str):
  """The tier build of recipe `recipe_name` on the parity scene (scene-free), as an OptiX
  build's Features."""
  import mujoco
  import mujoco_warp as mjw
  from fixtures.optix_codegen import PARITY_XML

  import raywarp.integrations.mujoco  # noqa: F401

  mjm = mujoco.MjModel.from_xml_string(PARITY_XML)
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  s = recipe(recipe_name).replace(**{"performance.traversal": "software"})
  r = rw.Renderer(scene, nworld=1, resolution=(8, 8), settings=s)
  try:
    return dataclasses.replace(r.context().features, traversal="optix", ordered_traversal=False, mesh_frames=False)
  finally:
    r.close()


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
def test_scene_free_launch_parameters_fit(recipe_name):
  """Each scene-free tier's launch parameters (every geometry table, the layered material carrier,
  the light grid, the temporal state's slot) fit under `params.PARAMS_LIMIT`: every tier is
  scene-free."""
  from raywarp.core.trace.optix.device import params

  f = _tier_build(recipe_name)
  assert f.scene_free
  every = scene_flags(has_shared=True, shared_primitive_types=(4, 5), has_shared_curves=True)
  layout = programs.layout(dataclasses.replace(f, culling=f.any_rgb), programs.program_flags(f, every))
  assert params.size_of(params.params_type(layout)) <= params.PARAMS_LIMIT


def test_moving_material_ids_cut_out_nothing_without_a_cutoff():
  """`launch._cut_all`: live material ids (MuJoCo's geom_matid) make every instance a cut-out
  candidate only when some material has an alpha cutoff; per-world or live cutoffs always do (a
  tier build with the alpha test enforced the any-hit program on every instance of a MuJoCo scene
  without cut-outs: 15x its own build)."""
  import types

  from raywarp.core.trace.optix import launch

  def ctx(
    cutoffs: list[float], *, live: tuple[str, ...] = ("instance.material",), declared: frozenset[str] = frozenset()
  ) -> FrameContext[None]:
    tables = {"material.alpha_cutoff": wp.array(np.asarray([cutoffs], np.float32), device="cpu")}
    binding = types.SimpleNamespace(slots=lambda _state: dict.fromkeys(live))
    scene = types.SimpleNamespace(tables=tables, binding=binding)
    return cast(FrameContext[None], types.SimpleNamespace(per_world_fields=declared, scene=scene))

  assert not launch._cut_all(ctx([0.0, 0.0]))
  assert launch._cut_all(ctx([0.0, 0.5]))
  assert not launch._cut_all(ctx([0.0, 0.5], live=()))
  assert launch._cut_all(ctx([0.0], live=("material.alpha_cutoff",)))
  assert launch._cut_all(ctx([0.0], live=(), declared=frozenset({"material.alpha_cutoff"})))


def test_the_ptx_key_ignores_what_no_program_reads(features):
  """block_dim, max_registers and ordered_traversal (`programs.KEY_UNREAD`) are not in the PTX key
  (a tuning study keeps the bundle); a field the programs read is."""
  flags = programs.program_flags(features, SCENES[2])
  key = programs.ptx_key(features, flags, supersampled=False, arch=120)
  for name, value in (("block_dim", 128), ("max_registers", 96), ("ordered_traversal", True)):
    assert programs.ptx_key(dataclasses.replace(features, **{name: value}), flags, supersampled=False, arch=120) == key
  assert programs.ptx_key(dataclasses.replace(features, fog=not features.fog), flags, supersampled=False, arch=120) != key
