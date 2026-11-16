"""MuJoCo flex on every available device: a cloth with edge equalities (eq_data must match the
MuJoCo build: mjNEQDATA values per row) behind analytic planes and shared static scenery."""

import numpy as np
import pytest
import warp as wp

import raywarp as rw

_XML = """
  <mujoco>
    <worldbody>
      <light type="directional" dir="0 0.3 -1"/>
      {extra}
      <flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.15 0.15 0.15" pos="0 0 0.5"
        radius="0.02" dim="2" rgba="0.2 0.7 0.3 1">
        <edge equality="true"/>
      </flexcomp>
      <camera name="cam" pos="0 -1.1 1.6" xyaxes="1 0 0 0 0.7 0.7" fovy="45"/>
    </worldbody>
  </mujoco>
"""

# Each layout puts the flex in a different tree: alone, with an analytic plane, next to a shared
# static box (world tree = flex, shared tree = box), and both (plane hits bound the BVH queries).
_EXTRA = {
  "alone": "",
  "plane": '<geom type="plane" size="2 2 0.1"/>',
  "shared": '<geom type="box" size="0.1 0.1 0.1" pos="1 1 0.1"/>',
  "plane_shared": '<geom type="plane" size="2 2 0.1"/><geom type="box" size="0.1 0.1 0.1" pos="1 1 0.1"/>',
}


_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("layout", list(_EXTRA))
def test_flex_cloth_renders(scene, layout, device):
  with wp.ScopedDevice(device):
    s = scene(_XML.format(extra=_EXTRA[layout]), cam_res=(48, 48), outputs=rw.Outputs.of("rgb", "seg")).render()
    assert s.mjm.neq > 0 and s.mjm.nflex == 1
    seg = s.seg()[0]
    flex = seg == s.scene.flex_segment
    assert flex.sum() > 200  # the cloth covers the image centre
    assert flex[24, 24]
    rgb = s.rgb()[0][flex]
    assert np.all(rgb[:, 1] > rgb[:, 0]) and np.all(rgb[:, 1] > rgb[:, 2])  # green


def test_flex_outside_the_rendered_groups_is_hidden(scene):
  """`geom_groups` hides a flex whose `flex_group` is not rendered, like a geom (mjlab and MuJoCo
  Warp group masks); a second, shown flex keeps its own segment id (ngeom + f)."""
  hidden = _XML.format(extra='<geom type="box" size="0.05 0.05 0.05" pos="0.6 0 0.1"/>').replace('dim="2"', 'dim="2" group="3"')
  s = scene(hidden, cam_res=(48, 48), outputs=rw.Outputs.of("rgb", "seg")).render()
  assert not (s.seg()[0] >= s.scene.flex_segment).any()
  shown = scene(hidden, cam_res=(48, 48), geom_groups=(0, 3), outputs=rw.Outputs.of("seg")).render()
  assert (shown.seg()[0] == shown.scene.flex_segment).sum() > 200
  rope = '<flexcomp name="rope" type="grid" count="6 1 1" spacing="0.12 0.12 0.12" pos="0 0 0.6" radius="0.04" dim="1"/>'
  both = hidden.replace("{extra}", "").replace("</flexcomp>", "</flexcomp>" + rope)
  two = scene(both, cam_res=(48, 48), outputs=rw.Outputs.of("seg")).render()
  ids = set(np.unique(two.seg()[0]).tolist())
  assert two.scene.flex_segment + 1 in ids and two.scene.flex_segment not in ids
  with pytest.raises(ValueError, match="no geoms or flexes"):
    scene(_XML.format(extra="").replace('dim="2"', 'dim="2" group="3"'), outputs=rw.Outputs.of("seg"))


_ROPE_XML = """
  <mujoco>
    <worldbody>
      <light type="directional" dir="0 0.3 -1"/>
      <flexcomp name="rope" type="grid" count="6 1 1" spacing="0.12 0.12 0.12" pos="0 0 0.5"
        radius="0.04" dim="1" rgba="0.2 0.7 0.3 1"/>
      <camera name="cam" pos="0 -1.1 1.2" xyaxes="1 0 0 0 0.6 0.8" fovy="45"/>
    </worldbody>
  </mujoco>
"""


def test_flex_vert_one_row_is_read_by_every_world(scene):
  """A one-row `d.flexvert_xpos` (a batched table: 1 or nworld rows) is world 0's rope in every
  world (the capsule intersection read row `worldid`, past the end of a one-row array)."""
  s = scene(_ROPE_XML, nworld=2, cam_res=(48, 48), outputs=rw.Outputs.of("rgb", "seg"))
  s.d.flexvert_xpos = wp.clone(s.d.flexvert_xpos[:1])
  s.render()
  seg = s.seg()
  assert (seg[0] == s.scene.flex_segment).sum() > 50  # the rope is in view
  np.testing.assert_array_equal(seg[1], seg[0])
  np.testing.assert_array_equal(s.rgb()[1], s.rgb()[0])


# A flex with a material: `{flex}` is the flexcomp (`_CLOTH` / `_BOX` with its attributes).
_MATERIAL_XML = """
  <mujoco>
    <asset>
      <texture name="checker" type="2d" builtin="checker" width="64" height="64" rgb1="0.9 0.1 0.1"
        rgb2="0.1 0.1 0.9"/>
      <material name="red" rgba="0.9 0.1 0.1 1" specular="0.2" shininess="0.1"/>
      <material name="glow" rgba="0.9 0.1 0.1 1" specular="0.2" shininess="0.1" emission="1"/>
      <material name="shiny" rgba="0.9 0.1 0.1 1" specular="1" shininess="0.1"/>
      <material name="checker" texture="checker" texrepeat="2 2"/>
      <material name="hidden" rgba="1 1 1 0"/>
    </asset>
    <worldbody>
      <light type="directional" dir="0 0.3 -1"/>
      {flex}
      <camera name="cam" pos="0 -1.1 1.6" xyaxes="1 0 0 0 0.7 0.7" fovy="45"/>
    </worldbody>
  </mujoco>
"""
_CLOTH = '<flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.15 0.15 0.15" pos="0 0 0.5" radius="0.02" dim="2" {}/>'
# A soft box (dim 3: its shell triangles, texture coordinates per vertex).
_BOX = '<flexcomp name="box" type="box" count="3 3 3" spacing="0.3 0.3 0.3" pos="0 0 0.5" dim="3" {}/>'


def _cloth_rgb(scene, attrs: str, flex: str = _CLOTH) -> np.ndarray:
  """(n, 3) display colors (0..1) of the flex's pixels (the image centre must be flex)."""
  s = scene(_MATERIAL_XML.format(flex=flex.format(attrs)), cam_res=(48, 48), outputs=rw.Outputs.of("rgb", "seg"))
  s.render()
  flex_px = s.seg()[0] == s.scene.flex_segment
  assert flex_px.sum() > 200 and flex_px[24, 24]
  return s.rgb()[0][flex_px].astype(np.float32)


@pytest.mark.parametrize("device", _DEVICES)
def test_flex_material_color(scene, device):
  """A flex of the default rgba shows its material's color (it rendered the default grey); a
  flex rgba of its own wins over the material's (MuJoCo's `setMaterial`)."""
  with wp.ScopedDevice(device):
    red = _cloth_rgb(scene, 'material="red"')
    assert np.all(red[:, 0] > red[:, 1] + 0.15) and np.all(red[:, 0] > red[:, 2] + 0.15)
    own = _cloth_rgb(scene, 'rgba="0.2 0.7 0.3 1" material="red"')
    assert np.all(own[:, 1] > own[:, 0]) and np.all(own[:, 1] > own[:, 2])


def test_flex_material_emission_and_specular(scene):
  """The material's emission lights the flex up, its specular raises the reflectance."""
  red = _cloth_rgb(scene, 'material="red"').mean(axis=0)
  glow = _cloth_rgb(scene, 'material="glow"').mean(axis=0)
  shiny = _cloth_rgb(scene, 'material="shiny"').mean(axis=0)
  assert glow[0] > red[0] + 0.05, (glow, red)
  assert shiny.sum() > red.sum() + 0.02, (shiny, red)


@pytest.mark.parametrize("device", _DEVICES)
def test_flex_material_texture(scene, device):
  """A textured flex material maps its texture through the flex's texture coordinates: the cloth
  shows both checker colors (red and blue squares)."""
  with wp.ScopedDevice(device):
    rgb = _cloth_rgb(scene, 'material="checker"')
    reds = (rgb[:, 0] > rgb[:, 2] + 0.15).mean()
    blues = (rgb[:, 2] > rgb[:, 0] + 0.15).mean()
    assert reds > 0.2 and blues > 0.2, (reds, blues)


def test_flex_material_texture_on_a_closed_surface(scene):
  """A dim-3 flex reads its texture through its shell corners' texture coordinates."""
  rgb = _cloth_rgb(scene, 'material="checker"', flex=_BOX)
  reds = (rgb[:, 0] > rgb[:, 2] + 0.15).mean()
  blues = (rgb[:, 2] > rgb[:, 0] + 0.15).mean()
  assert reds > 0.05 and blues > 0.05, (reds, blues)


def test_flex_hidden_by_its_material_alpha(scene):
  """A default-rgba flex whose material has alpha 0 is invisible (skipped like a flex rgba of alpha
  0): no flex pixel in the segmentation."""
  s = scene(
    _MATERIAL_XML.format(flex=_CLOTH.format('material="hidden"')), cam_res=(48, 48), outputs=rw.Outputs.of("rgb", "seg")
  )
  s.render()
  assert not np.any(s.seg()[0] == s.scene.flex_segment)


def test_flex_surface_refit_between_renders_matches_a_fresh_scene(scene):
  """A soft box's surface BLAS is refit and repacked every frame: rendering it, moving its
  vertices and rendering again gives the image of a fresh scene built where they moved to (its
  BLAS built there, not refit)."""
  box = '<flexcomp name="box" type="box" count="3 3 3" spacing="0.3 0.3 0.3" pos="{}" dim="3" rgba="0.8 0.3 0.2 1"/>'
  kw = {"cam_res": (32, 32), "outputs": rw.Outputs.of("depth")}
  live = scene(_MATERIAL_XML.format(flex=box.format("0 0 0.5")), **kw)
  before = live.render().depth()[0].copy()
  vert = live.d.flexvert_xpos.numpy()
  live.d.flexvert_xpos.assign((vert + np.array([0.15, 0.0, -0.25])).astype(np.float32))
  after = live.render().depth()[0]
  fresh = scene(_MATERIAL_XML.format(flex=box.format("0.15 0 0.25")), **kw)
  expected = fresh.render().depth()[0]
  assert np.abs(after - before).max() > 0.05  # the surface moved in view
  np.testing.assert_allclose(after, expected, atol=1e-5)


# Two soft tetrahedra in one flex; the camera sits inside the first and looks at the second.
_TWO_TETS = """
  <mujoco>
    <worldbody>
      <camera name="cam" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="20"/>
      <flexcomp name="two" type="direct" dim="3" rgba="0.2 0.7 0.3 1"
        point="-2 -2 -2  2 -2 -2  0 2 -2  0 0 2   -0.5 4 -0.5  0.5 4 -0.5  0 4 0.5  0 5 0"
        element="0 1 2 3  4 5 6 7">
        <edge equality="false"/>
      </flexcomp>
    </worldbody>
  </mujoco>
"""


@pytest.mark.parametrize("culling", [True, False])
def test_flex_back_face_does_not_hide_the_front_face_behind_it(scene, culling):
  """Back-face culling skips a flex's back-facing triangles inside the query, like a mesh's: the
  inner side of the tetrahedron around the camera is culled and the second one, behind it in the
  same flex, is the hit (the nearest back face used to drop the whole flex)."""
  s = scene(_TWO_TETS, cam_res=(16, 16), outputs=rw.Outputs.of("seg", "depth"), **{"surfaces.backface_culling": culling})
  s.render()
  assert s.seg()[0][8, 8] == s.scene.flex_segment
  # Culling sees the second tetrahedron (y = 4), not the first one's far side (y < 2).
  assert (s.depth()[0][8, 8] > 3.5) == culling
