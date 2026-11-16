"""`rw.Lighting`: lights, ambient, envmaps and headlight added to any scene before upload."""

import dataclasses

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import recipe, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.integrations.mujoco.lights import NO_LIGHT_AMBIENT
from raywarp.scene.model.types import NO_TEXTURE
from raywarp.settings.fields import DEFAULT_EV100, exposure_from_ev100

_XML = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <worldbody>
    <camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="sphere" size="0.4" pos="0 0 0.4" rgba="0.8 0.2 0.2 1"/>
  </worldbody>
</mujoco>
"""


def _host(textured: bool = False) -> rw.Scene:
  """A floor and a sphere, no light (a textured floor: one image already in the blob)."""
  b = rw.SceneBuilder()
  material = rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0)
  if textured:
    tex = b.add_texture(np.full((4, 4, 4), 200, np.uint8))
    material = material.with_texture(rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(tex))
  floor = b.add_material(material)
  b.add_instance("box", size=(3.0, 3.0, 0.05), material=floor, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
  b.add_instance("sphere", size=(0.4, 0.0, 0.0), material=floor, xform=np.c_[np.eye(3), [0.0, 0.0, 0.4]])
  return b.build()


def _mean_rgb(scene: rw.GpuScene, sim: object = None) -> float:
  """Mean display value of a 16 x 12 view of `scene` (unlit by GI: the default recipe)."""
  pose = wp.transformf((0.0, -3.0, 1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.3))
  settings = recipe("fast").replace(**{"outputs": rw.Outputs.of("rgb")})
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(16, 12), settings=settings)
  r.camera_pose.fill_(pose)
  return float(required(r.render(sim).rgb).numpy()[..., :3].mean())


def test_with_lighting_adds_lights_ambient_envmap_and_headlight():
  host = _host(textured=True)
  sky = np.full((8, 16, 3), 0.25, np.float32)
  lighting = rw.Lighting(
    lights=[rw.Light.point((0.0, 0.0, 2.0))],
    ambient=(0.1, 0.1, 0.1),
    envmap=sky,
    headlight=rw.Headlight(ambient=(0.05, 0.05, 0.05), diffuse=(0.3, 0.3, 0.3)),
  )
  lit = host.with_lighting(lighting)
  assert host.nlight == 0 and host.nenv == 0  # a new Scene
  kinds = [rw.scene.LightKind(int(k)) for k in lit.lights.kind]
  assert kinds == [rw.scene.LightKind.SPHERE, rw.scene.LightKind.CAMERA, rw.scene.LightKind.AMBIENT]
  np.testing.assert_allclose(lit.lights.color[2], np.full(3, 0.15) / lighting.exposure, rtol=1e-6)  # AMBIENT: color = radiance
  # The environment's texture is the appended one, its image after the floor's in the blob.
  assert lit.nenv == 1 and int(lit.environments.texture[0]) == host.ntexture
  image = int(lit.textures.image[host.ntexture])
  assert image == host.nimage and int(lit.images.offset[image]) % 256 == 0
  np.testing.assert_allclose(lit.images.level(image)[..., :3], 0.25)
  np.testing.assert_array_equal(lit.images.level(0), host.images.level(0))
  gpu = rw.upload(lit, "cpu")
  assert gpu.headlight.active and gpu.environment.lit


def test_with_lighting_appends_without_copying_the_mapped_blob(tmp_path):
  """Lighting a memory-mapped scene appends the rig's images after its texel blob without copying
  it (a multi-GB scene would double in RAM), twice in a row; the result still saves and loads."""
  host = _host(textured=True)
  host.save(tmp_path / "host.rayw")
  mapped = rw.Scene.load(tmp_path / "host.rayw")
  lit = mapped.with_lighting(rw.Lighting(envmap=np.full((8, 16, 3), 0.25, np.float32)))
  lit = lit.with_lighting(rw.Lighting(envmap=np.full((4, 8, 3), 0.5, np.float32), replace=True))
  assert lit.images.data is mapped.images.data and lit.nimage == 3
  images = [int(lit.textures.image[int(t)]) for t in lit.environments.texture]
  assert images == [2]  # replace: only the second rig's dome, its texture shifted past the others
  np.testing.assert_array_equal(lit.images.level(0), host.images.level(0))
  np.testing.assert_allclose(lit.images.level(1)[..., :3], 0.25)
  np.testing.assert_allclose(lit.images.level(2)[..., :3], 0.5)
  lit.save(tmp_path / "lit.rayw")
  again = rw.Scene.load(tmp_path / "lit.rayw")
  for i in range(3):
    np.testing.assert_array_equal(again.images.level(i), lit.images.level(i))


def test_missing_envmap_file_raises(tmp_path):
  """An envmap path that names no readable image raises (never a constant environment)."""
  with pytest.raises(ValueError, match="missing or unsupported"):
    _host().with_lighting(rw.Lighting(envmap=tmp_path / "sky.hdr"))


def test_with_lighting_brightens_a_dark_scene():
  host = _host()
  dark = _mean_rgb(rw.upload(host, "cpu"))
  ambient = _mean_rgb(rw.upload(host.with_lighting(rw.Lighting(ambient=(0.5, 0.5, 0.5))), "cpu"))
  assert dark < 1.0 < ambient


def test_point_light_display_relative_distance_uses_the_scene_bounds():
  """A display-relative point light placed from a built scene's instances matches the builder's."""
  light = rw.Light.point((0.0, 0.0, 3.0))
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0))
  b.add_instance("sphere", size=(0.4, 0.0, 0.0), material=floor, xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]])
  from raywarp.scene.lighting.rig import add_lighting

  add_lighting(b, rw.Lighting([light]))
  built = b.build()
  b2 = rw.SceneBuilder()
  b2.add_instance(
    "sphere",
    size=(0.4, 0.0, 0.0),
    material=b2.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0)),
    xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]],
  )
  later = b2.build().with_lighting(rw.Lighting([light]))
  np.testing.assert_array_equal(built.lights.intensity, later.lights.intensity)


def _mujoco(lighting: rw.Lighting | None) -> tuple[rw.GpuScene, mjw.Data]:
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  return rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), lighting=lighting), mjw.put_data(mjm, mjd)


def test_mujoco_scene_takes_ambient_envmap_and_headlight():
  ev100 = 10.0
  lighting = rw.Lighting(
    ambient=(0.2, 0.2, 0.2), headlight=rw.Headlight(ambient=(0.1, 0.1, 0.1), diffuse=(0.5, 0.4, 0.3)), ev100=ev100
  )
  scene, _ = _mujoco(lighting)
  assert scene.headlight.active
  np.testing.assert_allclose(scene.headlight.color, (0.5, 0.4, 0.3), rtol=1e-6)
  np.testing.assert_allclose(scene.environment.ambient, np.full(3, 0.3) / exposure_from_ev100(ev100), rtol=1e-6)
  # No light, headlight off: MuJoCo's no-light ambient (its renderers' fallback), unless Python
  # lighting adds an environment or an ambient (the model is lit then).
  dark, _ = _mujoco(None)
  assert not dark.headlight.active
  fallback = np.full(3, NO_LIGHT_AMBIENT) / exposure_from_ev100(DEFAULT_EV100)
  np.testing.assert_allclose(dark.environment.ambient, fallback, rtol=1e-6)
  lit, _ = _mujoco(rw.Lighting(envmap=np.ones((8, 16, 3), np.float32)))
  assert lit.environment.lit and lit.environment.has_maps and lit.environment.ambient == (0.0, 0.0, 0.0)


def test_mujoco_envmap_files_of_any_format(tmp_path):
  """An 8-bit PNG envmap is sRGB-decoded (it used to need a .hdr file)."""
  from PIL import Image

  Image.fromarray(np.full((8, 16, 3), 128, np.uint8)).save(tmp_path / "sky.png")
  from raywarp.scene.lighting.rig import envmap_radiance

  np.testing.assert_allclose(envmap_radiance(tmp_path / "sky.png"), 0.2158605, rtol=1e-4)
  scene, _ = _mujoco(rw.Lighting(envmap=str(tmp_path / "sky.png")))
  assert scene.environment.lit


def test_mujoco_scene_rejects_extra_lights():
  with pytest.raises(ValueError, match="MjSpec"):
    _mujoco(rw.Lighting([rw.Light.sun((-0.3, 0.4, -1.0))]))


def test_lighting_validates_its_lights():
  with pytest.raises(TypeError, match=r"rw\.Light"):
    rw.Lighting(["area"])  # pyrefly: ignore[bad-argument-type]


def test_usd_scene_lighting_keeps_the_cache_key(tmp_path):
  pytest.importorskip("pxr")
  from pxr import Usd, UsdGeom

  from raywarp.formats import usd as rw_usd

  path = tmp_path / "ball.usda"
  stage = Usd.Stage.CreateNew(str(path))
  UsdGeom.Sphere.Define(stage, "/ball")
  stage.Save()
  plain = rw_usd.load(path, cache=tmp_path)
  lit = rw_usd.load(path, cache=tmp_path, lighting=rw.Lighting(ambient=(0.2, 0.2, 0.2)))
  assert plain.nlight == 0 and lit.nlight == 1
  assert "warm_load" in lit.report.timings  # the cached stage, lighting added after


def test_studio_rig_follows_the_up_axis():
  z_up, y_up = rw.Lighting.studio(), rw.Lighting.studio(up="y", fill=False)
  assert len(z_up.lights) == 2 and len(y_up.lights) == 1 and not z_up.replace
  key_z, key_y = z_up.lights[0], y_up.lights[0]
  dz, dy = key_z.direction, key_y.direction
  assert key_z.shadow_enable and dz[2] < -0.7  # 50 deg down along -z
  np.testing.assert_allclose(dy, (dz[0], dz[2], -dz[1]), atol=1e-12)  # +z -> +y
  # The dome is brighter above than below the horizon, about the up axis.
  from raywarp.scene.images.latlong import latlong_directions

  for axis, lighting in ((2, z_up), (1, y_up)):
    dome = np.asarray(lighting.envmaps[0])
    up = latlong_directions(*dome.shape[:2])[..., axis]
    assert dome[up > 0.5].mean() > 2.0 * dome[up < -0.5].mean()


def test_studio_exposes_a_dark_scene_at_the_default_exposure():
  """A white floor under the studio rig alone renders mid-to-bright (not black, not clipped)."""
  host = _host()
  lit = rw.upload(host.with_lighting(rw.Lighting.studio()), "cpu")
  assert 90.0 < _mean_rgb(lit) < 235.0


def test_studio_replace_drops_the_scene_lighting():
  b = rw.SceneBuilder()
  b.add_instance(
    "sphere",
    size=(0.4, 0.0, 0.0),
    material=b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0)),
    xform=np.c_[np.eye(3), np.zeros(3)],
  )
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.SPHERE, intensity=5.0, radius=0.0, treat_as_point=True, xform=np.c_[np.eye(3), [0.0, 0.0, 2.0]]
    )
  )
  b.add_environment(rw.scene.EnvironmentLight(texture=NO_TEXTURE, intensity=0.1))
  host = b.build()
  added = host.with_lighting(rw.Lighting.studio())
  replaced = host.with_lighting(rw.Lighting.studio(replace=True))
  assert (added.nlight, added.nenv) == (3, 2) and (replaced.nlight, replaced.nenv) == (2, 1)


def test_mujoco_add_lights_keeps_the_light_model():
  """`rw.integrations.mujoco.add_lights` adds MJCF lights: classic ones in a classic model,
  display-relative photometric ones in a photometric model; `replace` deletes the model's lights
  first."""
  studio = rw.Lighting.studio()
  classic = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2"/>'))
  rest = rw.integrations.mujoco.add_lights(classic, studio)
  assert rest.lights == () and rest.envmaps and not rest.replace
  mjm = classic.compile()
  assert mjm.nlight == 3 and float(mjm.light_intensity.sum()) == 0.0
  photometric = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2" intensity="50"/>'))
  rw.integrations.mujoco.add_lights(photometric, studio)
  np.testing.assert_allclose(photometric.compile().light_intensity[1:], np.pi / studio.exposure, rtol=1e-6)
  replaced = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2" intensity="50"/>'))
  rw.integrations.mujoco.add_lights(replaced, rw.Lighting.studio(replace=True))
  mjm = replaced.compile()
  assert mjm.nlight == 2 and float(mjm.light_intensity.sum()) == 0.0  # a classic model now
  with pytest.raises(ValueError, match="display-relative"):
    lit = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2"/>'))
    rw.integrations.mujoco.add_lights(
      lit, rw.Lighting([rw.Light.sun((-0.3, 0.4, -1.0), illuminance=3.0)])
    )  # into a classic model


def test_mujoco_replace_switches_the_model_lights_off():
  mjm_spec = mujoco.MjSpec.from_string(
    _XML.replace('<headlight active="0"/>', "").replace("<worldbody>", '<worldbody><light pos="0 0 2"/>')
  )
  mjm = mjm_spec.compile()
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), lighting=rw.Lighting(replace=True, ambient=(0.1, 0.1, 0.1)))
  assert not scene.headlight.active
  assert float(scene.tables["light.strength"].numpy().sum()) == 0.0


def test_mujoco_add_lights_to_a_model_without_lights():
  """A spec without lights (none authored, or deleted by `replace`) takes the added lights' light
  model: a physical light makes it photometric (it raised: the photometric test ran after the
  deletion)."""
  physical = rw.Lighting(lights=[rw.Light.sun((-0.3, 0.4, -1.0), illuminance=1000.0)], replace=True)
  spec = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2"/>'))
  rw.integrations.mujoco.add_lights(spec, physical)
  np.testing.assert_allclose(spec.compile().light_intensity, [1000.0])
  bare = mujoco.MjSpec.from_string(_XML)
  rw.integrations.mujoco.add_lights(
    bare, rw.Lighting(lights=[rw.Light.sun((-0.3, 0.4, -1.0), illuminance=1000.0), rw.Light.sun((-0.3, 0.4, -1.0))])
  )
  intensities = bare.compile().light_intensity
  np.testing.assert_allclose(intensities, [1000.0, np.pi / rw.Lighting().exposure], rtol=1e-6)
  classic = mujoco.MjSpec.from_string(_XML)
  rw.integrations.mujoco.add_lights(classic, rw.Lighting.studio())
  assert float(classic.compile().light_intensity.sum()) == 0.0  # display-relative only: classic


@pytest.mark.parametrize(
  "change", [{"exposure": 1.0}, {"normalize": True}, {"specular": 0.5}, {"ext": rw.scene.LightExt(range=3.0)}]
)
def test_mujoco_add_lights_raises_on_fields_mjcf_cannot_hold(change):
  """A light field MJCF has no slot for raises instead of being dropped."""
  light = dataclasses.replace(rw.Light.sun((-0.3, 0.4, -1.0), illuminance=1000.0), **change)
  with pytest.raises(ValueError, match="cannot hold"):
    rw.integrations.mujoco.add_lights(mujoco.MjSpec.from_string(_XML), rw.Lighting(lights=[light], replace=True))


def test_mujoco_add_lights_keep_inverse_square_and_hard_cones():
  """Lights added to a photometric model keep their UsdLux falloff and cone: MJCF's defaults
  (range 10 m: a window darkening a light 12 % at 5 m; softness 0.2: an inner cone at 0.8 of a
  hard cone) are not theirs."""
  spec = mujoco.MjSpec.from_string(_XML.replace("<worldbody>", '<worldbody><light pos="0 0 2" intensity="50"/>'))
  spots = [rw.Light.point((0.0, 0.0, 3.0), intensity=20.0), rw.Light.spot((0.0, 0.0, 3.0), (0.0, 0.0, -1.0), intensity=20.0)]
  rw.integrations.mujoco.add_lights(spec, rw.Lighting(lights=spots))
  mjm = spec.compile()
  assert np.all(np.isinf(mjm.light_range[1:]))
  tables = raywarp.integrations.mujoco.lights.light_tables(mjm, 1.0)
  assert np.all(np.isinf(tables.range[1:]))
  assert tables.cone_inner is not None and float(tables.cone_inner[2]) == 1.0
