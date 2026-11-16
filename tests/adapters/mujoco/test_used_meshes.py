"""Meshes no rendered geom shows (MuJoCo collision-only meshes) get no BLAS (`Meshes.used`): the
same pixels, and a later `m.geom_dataid` write onto one is reported by the validation."""

from __future__ import annotations

import warnings

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, optix_build, reference_exposure, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import device_caps

# A tetrahedron (visual, group 0) and a cube hull (collision only, group 3), each its own mesh.
_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.4 0 0  0 0.4 0  0 0 0.4"/>
    <mesh name="hull" vertex="-.2 -.2 -.2  .2 -.2 -.2  -.2 .2 -.2  .2 .2 -.2  -.2 -.2 .2  .2 -.2 .2  -.2 .2 .2  .2 .2 .2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    <geom type="plane" size="3 3 0.1"/>
    <geom name="visual" type="mesh" mesh="tet" pos="-0.3 0 0.2"/>
    <geom name="collision" type="mesh" mesh="hull" pos="0.4 0 0.2" group="3"/>
  </worldbody>
</mujoco>
"""

# At the reference exposure: renderers with different frame histories compare equal (a metered,
# locked EV is one frame's float reduction, its last bits in the order the GPU sums them).
_SETTINGS = reference_exposure(
  rw.RenderSettings(gi=NO_GI, outputs=rw.Outputs.of("rgb", "depth"), performance=rw.settings.Performance(traversal="software")),
  {},
)


def _model() -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of `_XML`, two worlds."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=2)


def test_collision_only_meshes_get_no_blas() -> None:
  """The hull (group 3) is unused: a placeholder BLAS; the image equals one with every BLAS."""
  mjm, m, d = _model()
  scene = rw.integrations.mujoco.scene(mjm, m)
  assert scene.meshes.used is not None
  assert scene.meshes.used.tolist() == [True, False]
  r = rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(32, 24), settings=_SETTINGS)
  r.render(d)
  ids = scene.meshes.bvhs("cubql", used=scene.meshes.used_now())
  full = rw.integrations.mujoco.scene(mjm, m)
  full.meshes.use_all()
  ref = rw.Renderer(full, nworld=2, cameras=["cam"], resolution=(32, 24), settings=_SETTINGS)
  ref.render(d)
  np.testing.assert_array_equal(required(r.outputs.rgb).numpy(), required(ref.outputs.rgb).numpy())
  np.testing.assert_array_equal(required(r.outputs.depth).numpy(), required(ref.outputs.depth).numpy())
  assert ids.numpy()[1] != full.meshes.bvhs("cubql", used=None).numpy()[1]  # the placeholder, not the hull's BLAS


def test_per_world_meshes_build_every_blas() -> None:
  """A Renderer declaring per-world meshes can show any mesh: every mesh gets its BLAS."""
  mjm, m, _ = _model()
  scene = rw.integrations.mujoco.scene(mjm, m)
  rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(8, 8), settings=_SETTINGS, per_world=rw.PerWorld(mesh=["visual"]))
  assert scene.meshes.used is None


def test_a_geom_switched_onto_an_unused_mesh_is_reported() -> None:
  """`m.geom_dataid` written onto the collision-only mesh: the validation reports it."""
  mjm, m, d = _model()
  r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=["cam"], resolution=(8, 8), settings=_SETTINGS)
  r.render(d)
  visual = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "visual")
  dataid = m.geom_dataid.numpy()
  dataid[..., visual] = 1
  m.geom_dataid.assign(dataid)
  r.render(d)  # its check flags the mesh
  wp.synchronize()
  with pytest.raises(ValueError, match="geom_dataid"):
    r.render(d)


def test_a_per_world_mesh_renderer_keeps_an_earlier_renderers_blases() -> None:
  """Two Renderers on one scene: the second declares per-world meshes (`Meshes.use_all`). The
  first keeps tracing its own BLAS table (alive, same image) and still reports a geom switched
  onto the mesh its table has no BLAS of."""
  mjm, m, d = _model()
  scene = rw.integrations.mujoco.scene(mjm, m)
  first = rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(32, 24), settings=_SETTINGS)
  first.render(d)
  rgba, depth = required(first.outputs.rgb).numpy().copy(), required(first.outputs.depth).numpy().copy()
  table = scene.meshes.bvhs("cubql", used=scene.meshes.used_now())
  per_world = rw.PerWorld(mesh=["visual"])
  second = rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(32, 24), settings=_SETTINGS, per_world=per_world)
  second.render(d)
  assert scene.meshes.used is None
  assert scene.meshes.bvhs("cubql", used=None).numpy()[1] != table.numpy()[1]  # the second has the hull's BLAS
  wp.synchronize()
  first.render(d)  # the first still traces its own (alive) BLAS table: the same image
  np.testing.assert_array_equal(required(first.outputs.rgb).numpy(), rgba)
  np.testing.assert_array_equal(required(first.outputs.depth).numpy(), depth)
  visual = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "visual")
  dataid = m.geom_dataid.numpy()
  dataid[..., visual] = 1
  m.geom_dataid.assign(dataid)
  second.render(d)  # every mesh has its BLAS in the second's table: nothing to report
  first.render(d)  # the first's check flags the mesh
  wp.synchronize()
  second.render(d)
  with pytest.raises(ValueError, match="geom_dataid"):
    first.render(d)


@pytest.mark.parametrize("traversal", ["software", "optix"])
def test_an_earlier_renderer_keeps_its_mesh_mask_across_a_rebuild(traversal: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """After another Renderer's `use_all`, the first one changes the mesh BVH builder
  (`tuning.MESH_BVH`; software: new BLASes; OptiX: a rebuilt GAS set). Its tables keep its
  construction mask (the hull keeps its placeholder), matching its validation: no false report,
  the image of a fresh lbvh Renderer, and a geom switched onto the hull is still reported."""
  if traversal == "optix" and not (wp.get_device().is_cuda and device_caps(wp.get_device()).optix.available):
    pytest.skip("needs OptiX")
  mjm, m, d = _model()
  scene = rw.integrations.mujoco.scene(mjm, m)
  settings = _SETTINGS.replace(**{"performance.traversal": traversal})
  first = rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(32, 24), settings=settings)
  first.render(d)
  rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(8, 8), settings=settings, per_world=rw.PerWorld(mesh=["visual"]))
  monkeypatch.setattr(tuning, "MESH_BVH", "lbvh")
  first.update_settings(**{"shadows.samples": 2})  # new BLASes (lbvh) and a rebuild
  first.update_settings(**{"shadows.samples": 1})
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    for _ in range(3):  # a false report would raise on the render after the flagged one
      first.render(d)
  wp.synchronize()
  ctx = first.context()
  assert ctx.tables.mesh_used is not None and ctx.tables.mesh_used.tolist() == [True, False]
  hull = ctx.software.mesh_bvh_id.numpy()[1]
  assert hull != scene.meshes.bvhs("lbvh", used=None).numpy()[1]  # its placeholder, as its check assumes
  if traversal == "optix":
    mesh_used = optix_build(first).scene.mesh_used
    assert mesh_used is not None and mesh_used.tolist() == [True, False]
  fresh = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=["cam"], resolution=(32, 24), settings=settings)
  fresh.render(d)
  np.testing.assert_array_equal(required(first.outputs.rgb).numpy(), required(fresh.outputs.rgb).numpy())
  visual = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "visual")
  dataid = m.geom_dataid.numpy()
  dataid[..., visual] = 1
  m.geom_dataid.assign(dataid)
  first.render(d)
  wp.synchronize()
  with pytest.raises(ValueError, match="geom_dataid"):
    first.render(d)


def test_options_mujoco_cannot_honour_raise():
  """An explicit SceneOptions MuJoCo cannot honour raises (its emission lights nothing)."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  with wp.ScopedDevice("cpu"):
    m = mjw.put_model(mjm)
    with pytest.raises(ValueError, match="not supported"):
      rw.integrations.mujoco.scene(mjm, m, options=rw.SceneOptions(emissive_lights=True))
