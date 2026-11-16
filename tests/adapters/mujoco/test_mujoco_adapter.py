"""MuJoCo adapter regressions: textures, skybox, height fields, live materials / lights."""

from __future__ import annotations

import ast
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import REPO, recipe, required
from PIL import Image

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline.textures import formats

CPU = wp.get_device("cpu")

# A 2-texel-wide red / blue checker on a box seen face on.
_TEXTURED = """
<mujoco>
  {compiler}
  <asset>
    <texture name="t" type="2d" file="{png}" {embed}/>
    <material name="m" texture="t"/>
  </asset>
  <visual><headlight active="1" ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <camera name="cam" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <geom type="box" size="0.3 0.3 0.3" material="m"/>
  </worldbody>
</mujoco>
"""


def _checker_png(tmp_path: Path) -> Path:
  img = np.zeros((4, 8, 3), np.uint8)
  img[:, :4] = (255, 0, 0)
  img[:, 4:] = (0, 0, 255)
  path = tmp_path / "checker.png"
  Image.fromarray(img).save(path)
  return path


# By-reference textures (`embed`, `passthrough`) exist from MuJoCo 3.17.
_BY_REFERENCE = pytest.mark.skipif(tuple(int(v) for v in mujoco.__version__.split(".")[:2]) < (3, 17), reason="MuJoCo < 3.17")


@_BY_REFERENCE
def test_b35_file_reference_textures_render_like_embedded_ones(scene, tmp_path):
  """B35: MuJoCo >= 3.17 by-reference textures (tex_adr -1, tex_width 0) are decoded, not sliced."""
  png = _checker_png(tmp_path)
  images = []
  for compiler, embed in (("", ""), ("", 'embed="false"'), ('<compiler passthrough="textures"/>', "")):
    s = scene(_TEXTURED.format(png=png, compiler=compiler, embed=embed), cameras=["cam"], cam_res=(24, 24))
    if compiler or embed:
      assert s.mjm.tex_adr[0] == -1  # the repro: no payload in the model
    images.append(s.render().rgb())
  assert images[0][0, 12, 4, 0] > 0.5 > images[0][0, 12, 4, 2]  # red half
  for image in images[1:]:
    np.testing.assert_array_equal(image, images[0])


@_BY_REFERENCE
@pytest.mark.parametrize("chunk", [False, True])
def test_by_reference_textures_take_the_files_color_space(tmp_path, chunk):
  """A by-reference PNG reads in the color space MuJoCo resolves from the file when compiling it
  embedded (no sRGB chunk: linear; an sRGB chunk: sRGB), not the model's "auto": the embedded and
  by-reference copies store alike (sRGB color: float16 linear, linear: uint8)."""
  from PIL import PngImagePlugin

  from raywarp.integrations.mujoco import textures

  info = PngImagePlugin.PngInfo()
  if chunk:
    info.add(b"sRGB", b"\x00")
  png = tmp_path / "checker.png"
  Image.open(_checker_png(tmp_path)).save(png, pnginfo=info)
  dtypes = []
  for embed in ("", 'embed="false"'):
    mjm = mujoco.MjModel.from_xml_string(_TEXTURED.format(png=png, compiler="", embed=embed))
    want = int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if chunk else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
    assert textures.loaded(mjm, textures.MjTextureId(0))[1] == want
    with wp.ScopedDevice(CPU):
      gpu = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), options=rw.SceneOptions(textures=formats.EXACT_STORAGE))
    dtypes.append(gpu.textures[0].dtype)
  assert dtypes[0] == dtypes[1] == (wp.float16 if chunk else wp.uint8)


_SKYBOX = """
<mujoco>
  {compiler}
  <asset><texture name="sky" type="skybox" file="{png}" {embed}/></asset>
  <worldbody><geom type="sphere" size="0.1"/></worldbody>
</mujoco>
"""


@_BY_REFERENCE
def test_b35_by_reference_skybox_resamples_like_an_embedded_one(tmp_path):
  """B35: a by-reference skybox (tex_width 0) is resampled from its decoded faces (no 1 / 0)."""
  from raywarp.integrations.mujoco import environment, skybox

  png = _checker_png(tmp_path)
  square = np.asarray(Image.open(png))[:, :4]
  Image.fromarray(np.ascontiguousarray(square)).save(png)
  images = []
  for embed in ("", 'embed="false"'):
    mjm = mujoco.MjModel.from_xml_string(_SKYBOX.format(png=png, compiler="", embed=embed))
    images.append(environment.texture_radiance(mjm, skybox.skybox_id(mjm), device=CPU, width=64))
  assert np.all(np.isfinite(images[1]))
  np.testing.assert_array_equal(images[1], images[0])


# A 5 x 5 height field seen straight down.
_HFIELD = """
<mujoco>
  <asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{elevation}"/></asset>
  <worldbody>
    <camera name="down" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="20"/>
    <geom type="hfield" hfield="h"/>
  </worldbody>
</mujoco>
"""


def test_hfield_blas_is_built_on_the_device_with_mujocos_triangles_and_refits_live(scene):
  """The height field is triangulated by a Warp kernel from m.hfield_data (MuJoCo's cell
  split, 2 triangles per cell), and a terrain change in m reaches the render after refit_hfields."""
  elevation = " ".join(["0 1"] * 12 + ["0"])  # normalised by MuJoCo to [0, 1]
  s = scene(_HFIELD.format(elevation=elevation), cameras=["down"], cam_res=(8, 8), outputs=rw.Outputs.of("rgb", "depth"))
  hfields = s.scene.binding.hfields
  assert hfields.ids == [0]
  assert hfields.meshes[0].indices.shape[0] == 6 * 4 * 4
  data = s.mjm.hfield_data.reshape(5, 5)
  points = hfields.meshes[0].points.numpy()
  np.testing.assert_allclose(points[:, 2], data.ravel() * 0.5, atol=1e-6)
  np.testing.assert_allclose(points[[0, 4, 20, 24], :2], [[-1, -1], [1, -1], [-1, 1], [1, 1]], atol=1e-6)
  # Live terrain: raise it flat to 1 x size[2] and refit; the depth seen from above follows.
  s.m.hfield_data.assign(np.ones_like(s.mjm.hfield_data))
  s.scene.binding.refit_hfields()
  depth = s.render().depth()[0]
  np.testing.assert_allclose(depth, 3.0 - 0.5, atol=1e-3)


def test_a_height_field_shown_in_one_world_only_gets_its_blas():
  """A per-world `m.geom_dataid` row showing another height field builds that field's BLAS too
  (every batched row is scanned): world 1 sees the raised terrain, not nothing."""
  from conftest import default_settings

  flat, raised = " ".join(["0"] * 25), " ".join(["0"] * 12 + ["1"] + ["0"] * 12)  # MuJoCo normalises to [0, 1]
  xml = f"""
<mujoco>
  <asset>
    <hfield name="low" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{flat}"/>
    <hfield name="high" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{raised}"/>
  </asset>
  <worldbody>
    <camera name="down" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="1"/>
    <geom type="hfield" hfield="low"/>
  </worldbody>
</mujoco>
"""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  m.geom_dataid = wp.array(np.array([[0], [1]], np.int32), dtype=int)
  s = raywarp.integrations.mujoco.scene(mjm, m)
  assert isinstance(s.binding, raywarp.integrations.mujoco.MjwBinding) and s.binding.hfields.ids == [0, 1]
  settings = default_settings(outputs=rw.Outputs.of("depth"))
  r = rw.Renderer(s, nworld=2, cameras=["down"], resolution=(2, 2), settings=settings)
  out = r.render(mjw.put_data(mjm, mjd, nworld=2)).depth
  assert out is not None
  depth = out.numpy()
  assert np.all(depth[0] > 2.9) and np.all(depth[1] < 2.6)  # the centre vertex raised to 0.5


def test_a_live_write_onto_a_height_field_without_its_blas_is_reported():
  """A per-world height-field write after scene creation onto a height field no batched row of
  `m.geom_dataid` showed (built without a BLAS: a placeholder that traces nothing) is reported
  like the mesh case (`LiveCapabilityCheck`, BUG-268)."""
  from conftest import default_settings

  flat = " ".join(["0"] * 25)
  xml = f"""
<mujoco>
  <asset>
    <hfield name="low" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{flat}"/>
    <hfield name="high" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{flat}"/>
  </asset>
  <worldbody>
    <camera name="down" pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="1"/>
    <geom name="terrain" type="hfield" hfield="low"/>
  </worldbody>
</mujoco>
"""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  m.geom_dataid = wp.array(np.array([[0], [0]], np.int32), dtype=int)
  s = raywarp.integrations.mujoco.scene(mjm, m)
  assert isinstance(s.binding, raywarp.integrations.mujoco.MjwBinding) and s.binding.hfields.ids == [0]
  settings = default_settings(outputs=rw.Outputs.of("depth"))
  per_world = rw.PerWorld(mesh=["terrain"])  # per-world terrain: not shared scenery
  r = rw.Renderer(s, nworld=2, cameras=["down"], resolution=(2, 2), settings=settings, per_world=per_world)
  d = mjw.put_data(mjm, mjd, nworld=2)
  r.render(d)
  r.render(d)  # nothing to report
  table = r.per_world["instance.mesh"]
  ids = table.numpy()
  ids[1] = 1
  table.assign(ids)
  r.render(d)  # its check flags the height field
  wp.synchronize()
  with pytest.raises(ValueError, match="height field"):
    r.render(d)


# A 5 x 5 height field seen from the side (its relief in view).
_HFIELD_SIDE = """
<mujoco>
  <asset><hfield name="h" nrow="5" ncol="5" size="1 1 0.5 0.1" elevation="{elevation}"/></asset>
  <worldbody>
    <camera name="side" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.5 1" fovy="45"/>
    <geom type="hfield" hfield="h"/>
  </worldbody>
</mujoco>
"""


def test_hfield_refit_after_a_render_repacks_the_traced_nodes(scene):
  """Render, change the terrain, refit, render: the image is that of a scene built on the new
  terrain (the queries read packed copies of the BLAS nodes, repacked by the refit)."""
  terrain = " ".join(str(v) for v in np.random.default_rng(0).uniform(0, 1, 25).round(3))
  kw = {"cameras": ["side"], "cam_res": (24, 24), "outputs": rw.Outputs.of("depth")}
  fresh = scene(_HFIELD_SIDE.format(elevation=terrain), **kw)  # BLAS built on the new terrain
  expected = fresh.render().depth()[0]
  live = scene(_HFIELD_SIDE.format(elevation=" ".join(["0"] * 24 + ["1"])), **kw)
  before = live.render().depth()[0].copy()
  live.m.hfield_data.assign(fresh.mjm.hfield_data)  # MuJoCo's normalised terrain
  live.scene.binding.refit_hfields()
  after = live.render().depth()[0]
  assert np.abs(after - before).max() > 0.1  # the relief changed in view
  np.testing.assert_allclose(after, expected, atol=1e-5)


def test_no_private_mujoco_warp_api():
  """The adapter uses MuJoCo Warp's public Model / Data only (no mujoco_warp._src)."""
  root = REPO / "raywarp" / "integrations" / "mujoco"
  offenders = []
  for path in sorted(root.glob("*.py")):
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
      if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
      elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
      else:
        continue
      offenders += [f"{path.name}: {n}" for n in names if n.startswith("mujoco_warp._src")]
    assert "SimpleNamespace" not in text, path.name
  assert not offenders, offenders


# A sphere lit from the front by a directional light; two materials, rough plastic and metal.
_MATERIALS = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <material name="plastic" rgba="0.8 0.8 0.8 1" metallic="0" roughness="1" specular="0.5"/>
    <material name="metal" rgba="0.8 0.8 0.8 1" metallic="1" roughness="0.2"/>
  </asset>
  <worldbody>
    <light type="directional" dir="0 1 0" diffuse="1 1 1" castshadow="false"/>
    <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="20"/>
    <geom name="ball" type="sphere" size="0.5" material="plastic"/>
  </worldbody>
</mujoco>
"""


def test_b27_material_switch_and_specular_through_m_take_effect(scene):
  """B27: metallic / roughness / reflectance follow the live m.geom_matid (not baked per geom at
  load), and m.mat_specular (MuJoCo Warp's) is aliased."""
  s = scene(_MATERIALS, cameras=["front"], cam_res=(16, 16), outputs=rw.Outputs.of("rgb", "hdr"))
  plastic = s.render().hdr()[0].copy()
  metal = (
    scene(
      _MATERIALS.replace('material="plastic"/>\n  </worldbody>', 'material="metal"/>\n  </worldbody>'),
      cameras=["front"],
      cam_res=(16, 16),
      outputs=rw.Outputs.of("rgb", "hdr"),
    )
    .render()
    .hdr()[0]
  )
  assert np.abs(metal - plastic).max() > 0.05 * plastic.max()
  matid = s.m.geom_matid.numpy()
  matid[..., 0] = 1  # the ball is metal now
  s.m.geom_matid.assign(matid)
  np.testing.assert_allclose(s.render().hdr()[0], metal, rtol=1e-5, atol=1e-6)
  # Back to plastic, without its specular lobe: m.mat_specular is read every render.
  matid[..., 0] = 0
  s.m.geom_matid.assign(matid)
  specular = s.m.mat_specular.numpy()
  specular[..., 0] = 0.0
  s.m.mat_specular.assign(specular)
  assert s.render().hdr()[0].sum() < plastic.sum()


# A white Lambertian floor under one classic (intensity 0) light 1 m or 3 m above it.
_GL_LIGHT = """
<mujoco>
  <statistic center="0 0 {center}" extent="1"/>
  <visual><headlight active="0"/></visual>
  <asset><material name="white" rgba="1 1 1 1" metallic="0" roughness="1" specular="0"/></asset>
  <worldbody>
    <light type="{kind}" pos="0 0 {height}" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false" cutoff="80"/>
    <camera name="down" pos="0 0 0.5" xyaxes="1 0 0 0 1 0" fovy="10"/>
    <geom type="plane" size="5 5 0.1" material="white"/>
  </worldbody>
</mujoco>
"""


def test_classic_lights_have_no_falloff_and_show_their_diffuse_color(scene):
  """MuJoCo's GL lights have no distance falloff: a point / spot light at any height (and
  wherever the model centre is: no stat.center calibration) shows its diffuse color on white."""
  for kind in ("point", "spot"):
    seen = []
    for height, center in ((1.0, 0.0), (3.0, 0.0), (1.0, 5.0)):
      xml = _GL_LIGHT.format(kind=kind, height=height, center=center)
      s = scene(xml, cameras=["down"], cam_res=(8, 8), post=rw.settings.Post(tone_mapper="clamp"))
      seen.append(s.render().rgb()[0, 4, 4])
    np.testing.assert_allclose(seen, np.broadcast_to(seen[0], (3, 3)), atol=1.0 / 255)
    np.testing.assert_allclose(seen[0], 0.735, atol=2.0 / 255)  # sRGB of 0.5


def test_moving_classic_light_keeps_its_brightness(scene):
  """A GL point light moved by the simulation (light.pos from d) keeps its brightness."""
  xml = (
    _GL_LIGHT.format(kind="point", height=1.0, center=0.0)
    .replace(
      '<light type="point" pos="0 0 1.0"', '<body name="lamp" pos="0 0 1.0" mocap="true"><light type="point" pos="0 0 0"'
    )
    .replace('cutoff="80"/>', 'cutoff="80"/></body>')
  )
  s = scene(xml, cameras=["down"], cam_res=(8, 8), post=rw.settings.Post(tone_mapper="clamp"))
  near = s.render().rgb()[0, 4, 4].copy()
  xpos = s.d.light_xpos.numpy()
  xpos[..., 2] = 4.0
  s.d.light_xpos.assign(xpos)
  np.testing.assert_allclose(s.render().rgb()[0, 4, 4], near, atol=1.0 / 255)
  # Moved aside (2 m off, 1 m up), it lights the floor at a slant: the position is read per frame.
  xpos[..., 0], xpos[..., 2] = 2.0, 1.0
  s.d.light_xpos.assign(xpos)
  assert (s.render().rgb()[0, 4, 4] < near - 0.1).all()


def test_photometric_models_follow_filament(scene):
  """A model with a light of intensity > 0 is photometric (MuJoCo Filament): color =
  diffuse x intensity, intensity-0 lights dark, headlight off, no ambient."""
  floor = '<asset><material name="white" rgba="1 1 1 1" metallic="0" roughness="1" specular="0"/></asset>'
  body = (
    '<camera name="down" pos="0 0 0.5" xyaxes="1 0 0 0 1 0" fovy="10"/><geom type="plane" size="5 5 0.1" material="white"/>'
  )
  head = '<visual><headlight active="1" ambient="0.3 0.3 0.3" diffuse="0.5 0.5 0.5"/></visual>'
  lights = '<light type="point" pos="0 0 1" intensity="2" diffuse="0.5 0.5 0.5" range="100" castshadow="false"/><light type="point" pos="0 0 1"/>'
  xml = f"<mujoco>{head}{floor}<worldbody>{lights}{body}</worldbody></mujoco>"
  s = scene(xml, cameras=["down"], cam_res=(8, 8), outputs=rw.Outputs.of("rgb", "hdr"))
  assert not s.scene.headlight.active
  assert s.scene.environment.ambient == (0.0, 0.0, 0.0)
  # Lambertian white: radiance = 1 / pi x 0.5 x 2 cd / (1 m)^2 (the intensity-0 light adds nothing).
  np.testing.assert_allclose(s.render().hdr()[0, 4, 4], 0.5 * 2.0 / np.pi, rtol=3e-3)


_TWO_ROLES = """
<mujoco>
  <asset>
    <texture name="t" type="2d" builtin="checker" rgb1="0.8 0.2 0.2" rgb2="0.2 0.2 0.8" width="8" height="8"/>
    <material name="color" texture="t"/>
    <material name="bump"><layer texture="t" role="normal"/></material>
    <material name="hidden"><layer texture="t" role="orm"/></material>
  </asset>
  <worldbody>
    <geom type="sphere" size="0.1" material="color"/>
    <geom type="sphere" size="0.1" pos="0.3 0 0" material="{second}"/>
    <geom type="sphere" size="0.1" pos="0.6 0 0" material="hidden" group="4"/>
  </worldbody>
</mujoco>
"""


def test_color_space_per_texture_and_role_over_rendered_geoms():
  """A texture bound as color (sRGB) and as data (linear) by rendered geoms gets two GPU
  textures and the material table points the data role at the linear copy; data roles of geoms
  that are not rendered (group 4) do not count, and m.mat_texid stays aliased then."""
  import mujoco
  import mujoco_warp as mjw

  for second, split in (("bump", True), ("color", False)):
    mjm = mujoco.MjModel.from_xml_string(_TWO_ROLES.format(second=second))
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), options=rw.SceneOptions(textures=formats.EXACT_STORAGE))
    assert len(scene.textures) == mjm.ntex + split
    own = "material.texture" in scene.tables
    assert own == split and ("material.texture" in scene.binding.slots(None)) != split
    if split:
      texid = scene.tables["material.texture"].numpy()[0]
      assert texid[0, 1] == 0 and texid[1, 5] == mjm.ntex  # rgb: the sRGB texture; normal: the copy


def test_capabilities_scan_every_batched_row(scene):
  """A translucent geom color or a spot light in any world of a batched m compiles them in;
  data written later that needs a compiled-out feature fires the binding's feature check."""
  import warp as wp

  xml = '<mujoco><worldbody><light type="point" pos="0 0 2"/><geom type="sphere" size="0.1"/><camera pos="0 -1 0" xyaxes="1 0 0 0 0 1"/></worldbody></mujoco>'
  s = scene(xml, nworld=2, cam_res=(8, 8))
  assert not s.scene.capabilities.has_alpha and not s.scene.capabilities.has_spot
  (check,) = s.scene.binding.feature_checks()  # core.pipeline.gpu_scene.binding.FeatureCheck
  flags = wp.zeros(1, dtype=wp.int32)

  def fired() -> int:
    """Runs the check on the Renderer's current frame."""
    flags.zero_()
    check.enqueue(s.r.features, s.r.debug.frame(s.d), flags)
    return int(flags.numpy()[0] & check.bit != 0)

  assert fired() == 0
  m = s.m
  rgba = np.tile(m.geom_rgba.numpy(), (2, 1, 1))
  rgba[1, 0, 3] = 0.5
  m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  assert fired() == 1  # a translucent geom in world 1, compiled without translucency
  m.geom_rgba = wp.array(rgba[:1], dtype=wp.vec4)
  m.light_type = wp.array(np.array([[2], [0]], np.int32), dtype=int)  # world 1: a spot light
  assert fired() == 1
  assert "compiled without" in check.explain(s.r.features)
  m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  caps = rw.integrations.mujoco.scene(s.mjm, m).capabilities  # scanned over every batched row
  assert caps.has_alpha and caps.has_spot


@pytest.mark.parametrize("off", ["transparency.enabled", "surfaces.textures"])
def test_settings_that_turn_a_scanned_feature_off_render(off):
  """A scene with a translucent geom and a texture, rendered with transparency or textures off:
  the live check does not take the settings' choice for a live table change (it raised at the
  first frame)."""
  import mujoco
  import mujoco_warp as mjw
  import warp as wp

  from raywarp.core.pipeline.platform.device import host_copy

  xml = """
  <mujoco>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".6 .6 .6" width="16" height="16"/>
      <material name="grid" texture="grid"/>
    </asset>
    <worldbody>
      <light type="directional" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" material="grid"/>
      <geom type="box" pos="0 0 0.2" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 0.5"/>
      <camera name="c" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=1)
  mjw.forward(m, d)
  scene = rw.integrations.mujoco.scene(mjm, m)
  assert scene.capabilities.has_alpha and scene.capabilities.textures.base_color
  settings = rw.RenderSettings().replace(**{off: False})
  with rw.Renderer(scene, nworld=1, cameras="c", resolution=(16, 12), settings=settings) as r:
    r.render(d)
    r.render(d)  # the check of the first frame is read at the next
    wp.synchronize_device(r.device)
    assert host_copy(required(r.outputs.rgb))[..., :3].mean() > 5


def test_flex_material_alpha_scans_every_batched_row():
  """A default-rgba flex shows its material's alpha: translucency compiles in when only world 1's
  `m.mat_rgba` row is translucent (the scan read the host model's single row)."""
  import mujoco_warp as mjw
  import warp as wp

  xml = """
  <mujoco>
    <asset><material name="cloth" rgba="1 0 0 1"/></asset>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 1" dim="2" material="cloth"/>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mjw.put_model(mjm)
  assert not rw.integrations.mujoco.scene(mjm, m).capabilities.has_alpha
  rgba = np.tile(m.mat_rgba.numpy().reshape(1, -1, 4), (2, 1, 1))
  rgba[1, 0, 3] = 0.5
  m.mat_rgba = wp.array(rgba, dtype=wp.vec4)
  assert rw.integrations.mujoco.scene(mjm, m).capabilities.has_alpha


def test_one_slot_table_in_the_core_vocabulary():
  """The MuJoCo Warp map is one table keyed by the core's slots (no slot outside `SLOTS`): model
  and data slots are disjoint, each in the core's slot order."""
  from raywarp.core.pipeline.gpu_scene.binding import SLOTS
  from raywarp.integrations.mujoco import binding as bind

  assert set(bind.MODEL_SLOTS) | set(bind.DATA_SLOTS) <= set(SLOTS)
  assert not set(bind.MODEL_SLOTS) & set(bind.DATA_SLOTS)
  order = list(SLOTS)
  for table in (bind.MODEL_SLOTS, bind.DATA_SLOTS):  # in the core's slot order
    assert list(table) == sorted(table, key=order.index)


def test_unlit_render_of_a_spot_light_does_not_flag_it():
  """Unlit shading (`UNLIT_FAST`: no lights compiled in) of a model with a spot light (MJCF's
  default light type) renders frame after frame: the live capability check skips the lights of a
  build that reads none (it raised from the second frame on)."""
  xml = (
    '<mujoco><worldbody><light pos="0 0 3" dir="0 0 -1"/><geom type="plane" size="3 3 .1"/>'
    '<camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 .5 1"/></worldbody></mujoco>'
  )
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  r = rw.Renderer(
    rw.integrations.mujoco.scene(mjm, m), nworld=1, cameras=["cam"], resolution=(8, 8), settings=recipe("unlit_fast")
  )
  for _ in range(3):
    r.render(d)


def test_textures_prepared_on_threads_upload_in_order():
  """More bound textures than the preparation window: every texture keeps its id and size, an
  unbound one stays a placeholder (the texels are prepared on a pool, uploaded in order)."""
  from raywarp.integrations.mujoco import textures

  n = 2 * 2 * textures.PREPARE_THREADS + 3
  assets = "".join(
    f'<texture name="t{i}" type="2d" builtin="checker" width="{4 + 2 * i}" height="{2 + i}" rgb1="1 0 0" rgb2="0 0 1"/>'
    f'<material name="m{i}" texture="t{i}"/>'
    for i in range(n)
  )
  assets += '<texture name="unused" type="2d" builtin="flat" width="8" height="8"/>'
  geoms = "".join(f'<geom type="box" size="0.1 0.1 0.1" pos="{i} 0 0" material="m{i}"/>' for i in range(n))
  mjm = mujoco.MjModel.from_xml_string(f"<mujoco><asset>{assets}</asset><worldbody>{geoms}</worldbody></mujoco>")
  materials = np.arange(mjm.nmat).astype(np.int64)
  pack = textures.upload_textures(mjm, mjw.put_model(mjm).mat_texid.numpy(), materials, device=CPU)
  np.testing.assert_array_equal(pack.sizes[:n], np.stack([mjm.tex_width[:n], mjm.tex_height[:n]], 1))
  np.testing.assert_array_equal(pack.sizes[n], [1, 1])


def test_textures_are_capped_like_the_other_adapters():
  """MuJoCo textures arrive decoded in the model: the default storage halves a 2048 x 1024 one to
  `formats.MAX_TEXTURE_SIZE` (the Newton / USD decode cap: the one `SceneOptions` knob),
  `max_size=0` keeps it."""
  from raywarp.integrations.mujoco import textures

  assert formats.MAX_TEXTURE_SIZE == 1024 == rw.SceneOptions().max_texture_size == formats.DEFAULT_STORAGE.max_size
  xml = (
    '<mujoco><asset><texture name="t" type="2d" builtin="checker" width="2048" height="1024" rgb1="1 0 0" rgb2="0 1 0"/>'
    '<material name="m" texture="t"/></asset><worldbody><geom type="box" size=".1 .1 .1" material="m"/></worldbody></mujoco>'
  )
  mjm = mujoco.MjModel.from_xml_string(xml)
  texid, materials = mjw.put_model(mjm).mat_texid.numpy(), np.arange(mjm.nmat).astype(np.int64)
  capped = textures.upload_textures(mjm, texid, materials, device=CPU)
  np.testing.assert_array_equal(capped.sizes[0], [1024, 512])
  full = textures.upload_textures(mjm, texid, materials, storage=formats.TextureStorage(max_size=0), device=CPU)
  np.testing.assert_array_equal(full.sizes[0], [2048, 1024])


_NORMAL_MAPPED = """
<mujoco>
  <asset>
    <texture name="n" type="2d" file="{png}"/>
    <texture name="c" type="2d" builtin="checker" rgb1="0.8 0.3 0.2" rgb2="0.2 0.4 0.8" width="16" height="16"/>
    <material name="m"><layer texture="c" role="rgb"/><layer texture="n" role="normal"/></material>
  </asset>
  <visual><headlight active="1" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8"/></visual>
  <worldbody>
    <light pos="0.4 -1 0.6" dir="-0.4 1 -0.6"/>
    <camera name="cam" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <geom type="box" size="0.3 0.3 0.3" material="m"/>
  </worldbody>
</mujoco>
"""


# A uniform grey skybox behind a small sphere, in the color space `{space}`.
_GREY_SKY = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" width="16" height="16" rgb1=".5 .5 .5" rgb2=".5 .5 .5" colorspace="{space}"/>
  </asset>
  <worldbody>
    <geom type="sphere" size="0.05"/>
    <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="40"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(("space", "shown"), [("auto", 0.5), ("sRGB", 0.5), ("linear", 0.7354)])
def test_skybox_is_decoded_in_its_color_space(scene, space, shown):
  """The skybox background is decoded in its texture's color space (it was always sRGB): a texel
  of 0.5 shows 0.5 when sRGB-encoded ("auto" for a color), the sRGB encoding of 0.5 when linear."""
  s = scene(_GREY_SKY.format(space=space), post=rw.settings.Post(tone_mapper="clamp")).render()
  np.testing.assert_allclose(s.rgb()[0, 1, 1], shown, atol=0.01)


# A textured box: a color texture and a normal map.
_TEXTURED_BOX = """
<mujoco>
  <asset>
    <texture name="c" type="2d" builtin="checker" width="64" height="64" rgb1=".9 .2 .1" rgb2=".1 .2 .9"/>
    <texture name="n" type="2d" builtin="gradient" width="64" height="64" rgb1=".5 .5 1" rgb2=".6 .5 .9"/>
    <material name="m">
      <layer role="rgb" texture="c"/>
      <layer role="normal" texture="n"/>
    </material>
  </asset>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <geom type="box" size=".3 .3 .3" material="m"/>
    <camera name="cam" pos="0 -1.6 0.9" xyaxes="1 0 0 0 0.5 0.86"/>
  </worldbody>
</mujoco>
"""


def test_textures_are_remembered_and_a_repeat_upload_builds_no_texel(scene, tmp_path, monkeypatch):
  """A texture's disk entry (keyed by its texels and storage) lets a repeat upload rebuild it from
  its cached levels without building a texel; the image is the same."""
  from raywarp.integrations.mujoco import textures

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  first = scene(_TEXTURED_BOX).render().rgb()
  assert len(list((tmp_path / "textures").glob("mjcf-*.json"))) == 2

  def no_texels(*args, **kwargs):
    raise AssertionError("a remembered texture was built from its texels")

  monkeypatch.setattr(textures, "host_texture", no_texels)
  np.testing.assert_array_equal(scene(_TEXTURED_BOX).render().rgb(), first)


# An emissive ball (material emission 1, a black light: none of MuJoCo's no-light ambient) seen
# face on by an unlit camera.
_GLOW = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <asset><material name="glow" rgba="0.8 0.2 0.1 1" emission="1" specular="0"/></asset>
  <worldbody>
    <light type="directional" dir="0 1 0" diffuse="0 0 0" castshadow="false"/>
    <camera name="cam" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <geom type="sphere" size="0.2" material="glow" rgba="{rgba}"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("rgba", ["0.5 0.5 0.5 1", "0.1 0.9 0.2 1"])
def test_emission_follows_live_geom_and_material_colors(scene, rgba):
  """MuJoCo's emission is emission x the geom's color: a live write of m.geom_rgba (or, for a
  geom with the default rgba, of m.mat_rgba) retints the emitted light on the next render."""
  s = scene(_GLOW.format(rgba=rgba), cam_res=(8, 8), outputs=rw.Outputs.of("rgb", "hdr"))
  before = s.render().hdr()[0, 4, 4, :3].copy()
  shown = "mat_rgba" if rgba == "0.5 0.5 0.5 1" else "geom_rgba"
  table = getattr(s.m, shown)
  values = table.numpy()
  values[..., 0, :3] = (0.1, 0.3, 0.9)
  table.assign(values)
  after = s.render().hdr()[0, 4, 4, :3]
  # Emission alone lights the ball: its radiance is proportional to the color shown.
  np.testing.assert_allclose(after / after.max(), np.array([0.1, 0.3, 0.9]) / 0.9, atol=0.02)
  assert before.argmax() != after.argmax()


def test_a_k_row_model_table_is_tiled_to_every_world(scene):
  """A batched model table of K rows (MuJoCo Warp's world % K) reaches the kernels as its
  nworld-row tiling, rewritten from the table once per frame (`write_tables`: in-place writes
  included); `slots`, which a frame calls several times, serves the buffer without device work."""
  import warp as wp

  xml = '<mujoco><worldbody><light pos="0 0 2"/><geom type="sphere" size="0.1"/><camera pos="0 -1 0" xyaxes="1 0 0 0 0 1"/></worldbody></mujoco>'
  s = scene(xml, nworld=5, cam_res=(8, 8))
  binding = s.scene.binding
  rgba = np.tile(s.m.geom_rgba.numpy(), (2, 1, 1))
  rgba[1, :, 0] = 0.25
  s.m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  binding.write_tables(s.d)
  np.testing.assert_array_equal(binding.slots(s.d)["instance.color"].numpy(), rgba[np.arange(5) % 2])
  before = rgba.copy()
  rgba[0, :, 1] = 0.5
  s.m.geom_rgba.assign(rgba)  # in place
  np.testing.assert_array_equal(binding.slots(s.d)["instance.color"].numpy(), before[np.arange(5) % 2])  # no copy
  s.r.render(s.d)  # the frame's write
  np.testing.assert_array_equal(binding.slots(s.d)["instance.color"].numpy(), rgba[np.arange(5) % 2])
  # A second world count over the same scene (another Renderer) gets its own buffer: the first
  # keeps its 5 rows (a captured graph's address), the second has 7 (not 7 rows written into 5).
  first = binding.slots(s.d)["instance.color"]
  mjd = mujoco.MjData(s.mjm)
  mujoco.mj_forward(s.mjm, mjd)
  wide = mjw.put_data(s.mjm, mjd, nworld=7)
  binding.write_tables(wide)
  np.testing.assert_array_equal(binding.slots(wide)["instance.color"].numpy(), rgba[np.arange(7) % 2])
  assert binding.slots(s.d)["instance.color"].ptr == first.ptr and first.shape[0] == 5


_NO_LIGHT = """
<mujoco>
  <visual><headlight active="{head}" ambient="0 0 0" diffuse="0 0 0"/></visual>
  <worldbody><geom type="sphere" size="0.2"/></worldbody>
</mujoco>
"""


@pytest.mark.parametrize(("head", "seen"), [("0", 0.3), ("1", 0.0)])
def test_a_model_without_lights_shows_mujocos_fallback_ambient(head: str, seen: float) -> None:
  """A classic model without lights and with its headlight off is lit by MuJoCo's fallback
  ambient (0.3: its OpenGL renderer's global ambient then, MuJoCo Warp's
  `NO_LIGHT_AMBIENT_FALLBACK`), display-relative; with the headlight on, the headlight's own."""
  import mujoco

  from raywarp.integrations.mujoco import lights

  mjm = mujoco.MjModel.from_xml_string(_NO_LIGHT.format(head=head))
  np.testing.assert_allclose(lights.ambient(mjm, 0.5), (seen / 0.5,) * 3, rtol=1e-6)
  np.testing.assert_allclose(lights.ambient(mjm, 0.5, with_headlight=False), (0.0,) * 3)
