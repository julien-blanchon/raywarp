"""White furnace: spheres under a uniform environment, and the GI probe placement of scenes
smaller than the probe spacing."""

import numpy as np
import pytest
import warp as wp
from conftest import recipe

import raywarp as rw
from raywarp.settings.fields import exposure_from_ev100

# Camera 3 m from a 0.5 m sphere at the origin, looking along +y; at 32 x 32 and fovy 20 deg the
# sphere covers a disc of about 15 px radius around the centre.
_RES = 32
_FOVY_DEG = 20.0
_INNER_PX = 8  # the disc where NoV > ~0.85 (away from the Fresnel rim)
_GOLD = (1.0, 0.78, 0.34)
_TABLE_TOP_M = 0.45


def _uniform_sky(scene: rw.Scene) -> rw.Scene:
  """`scene` under a constant display-relative environment of 1."""
  return scene.with_lighting(rw.Lighting(envmap=np.ones((8, 16, 3), np.float32)))


def _sphere(material: rw.Material) -> rw.Scene:
  """One 0.5 m sphere at the origin under a uniform sky."""
  b = rw.SceneBuilder()
  b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=b.add_material(material), xform=np.c_[np.eye(3), np.zeros(3)])
  return _uniform_sky(b.build())


def _inner_disc(settings: rw.RenderSettings, material: rw.Material) -> np.ndarray:
  """(n, 3) display-relative linear radiance of the pixels of the sphere's inner disc."""
  settings = settings.replace(outputs=rw.Outputs.of("hdr"))
  r = rw.Renderer(
    rw.upload(_sphere(material), "cpu"),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=_FOVY_DEG)],
    resolution=(_RES, _RES),
    settings=settings,
  )
  r.camera_pose.fill_(wp.transformf((0.0, -3.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)))
  hdr = r.render().hdr
  assert hdr is not None
  yy, xx = np.mgrid[:_RES, :_RES] + 0.5 - 0.5 * _RES
  inside = yy * yy + xx * xx < _INNER_PX * _INNER_PX
  return hdr.numpy()[0, 0][inside] * exposure_from_ev100(settings.exposure.render_ev100())


# balanced (probe GI, the test session's 1 m grid) and balanced with GI off (the environment
# unoccluded).
_BALANCED_GI = recipe("default").replace(gi=rw.settings.GI(enabled=True))
_RECIPES = {
  "default": rw.RenderSettings(),  # GI off
  "default_gi": _BALANCED_GI,
  "fast": recipe("fast"),
}


@pytest.mark.parametrize("recipe_name", sorted(_RECIPES))
def test_white_furnace_diffuse_sphere(recipe_name: str):
  """A grey-white rough dielectric under a uniform sky shows about its albedo everywhere (diffuse
  plus a few % of specular), with GI probes and without: not black (probes inside it)."""
  material = rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=0.5)
  disc = _inner_disc(_RECIPES[recipe_name], material)
  assert disc.min() > 0.76 and disc.max() < 0.86, (disc.min(), disc.max())
  assert np.ptp(disc) < 0.03


@pytest.mark.parametrize("recipe_name", sorted(_RECIPES))
def test_white_furnace_metal_sphere(recipe_name: str):
  """A smooth gold sphere under a uniform sky shows F0 x E (E ~ 1 at NoV ~ 1, DFG table)."""
  material = rw.Material(base_color=_GOLD, base_metalness=1.0, specular_roughness=0.1)
  disc = _inner_disc(_RECIPES[recipe_name], material)
  np.testing.assert_allclose(disc, np.broadcast_to(_GOLD, disc.shape), atol=0.04)


def _table() -> rw.Scene:
  """The robotics review's 0.9 x 1.2 m pick table (top at 0.45 m, 4 legs) on a floor plane."""
  b = rw.SceneBuilder()
  wood = b.add_material(rw.Material(base_color=(0.6, 0.5, 0.4), specular_roughness=1.0))
  b.add_instance("plane", size=(5.0, 5.0, 0.0), material=wood)
  b.add_instance("box", size=(0.45, 0.6, 0.02), material=wood, xform=np.c_[np.eye(3), [0.0, 0.0, _TABLE_TOP_M - 0.02]])
  for x in (-0.4, 0.4):
    for y in (-0.55, 0.55):
      b.add_instance("box", size=(0.02, 0.02, 0.205), material=wood, xform=np.c_[np.eye(3), [x, y, 0.205]])
  return _uniform_sky(b.build())


def _active_probes(scene: rw.Scene, settings: rw.RenderSettings) -> np.ndarray:
  """(n, 3) positions of the active probes of the volume a render of `scene` places."""
  r = rw.Renderer(rw.upload(scene, "cpu"), nworld=1, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=settings)
  r.render()
  vol = r.debug.gi_volume
  assert vol is not None
  s = vol.struct
  nx, ny, _ = vol.counts
  idx = np.arange(vol.nprobe)
  coords = np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)], axis=1)
  pos = np.asarray(s.origin) + coords * np.asarray(s.spacing) + s.offset.numpy()
  return pos[s.active.numpy() == 1]


def test_probe_grid_covers_the_space_above_a_small_table():
  """The default's 1 m probe grid (GI opt-in) over a table smaller than the spacing has active
  probes above the table top (one cell per axis put its 2 probes under the top: the robotics-fit
  review's bug)."""
  pos = _active_probes(_table(), _BALANCED_GI)
  above = (pos[:, 2] > _TABLE_TOP_M) & (np.abs(pos[:, 0]) < 1.0) & (np.abs(pos[:, 1]) < 1.0)
  assert above.any(), pos


def test_probe_grid_surrounds_a_lone_sphere():
  """A 1 m sphere under balanced's 1 m grid gets active probes on every side of it (one cell put
  the only probe at its centre: inactive, the sphere rendered black)."""
  pos = _active_probes(_sphere(rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0)), _BALANCED_GI)
  assert len(pos) and np.all(np.linalg.norm(pos, axis=1) > 0.5)
  for axis in range(3):
    assert pos[:, axis].min() < 0.0 < pos[:, axis].max(), pos
