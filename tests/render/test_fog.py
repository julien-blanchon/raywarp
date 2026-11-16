"""Distance fog and haze (`settings.Atmosphere`): off by default, runtime and per world."""

import dataclasses

import numpy as np
from fixtures.conformance import module_loads

import raywarp as rw

# A floor receding to the horizon under a sky (misses in the top rows).
XML = """
<mujoco>
  <visual><map znear="0.01" zfar="0"/></visual>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.8 0.3 0.2 1"/>
    <geom name="ball" type="sphere" pos="0 2 0.5" size="0.5" rgba="0.2 0.6 0.9 1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.3 1" fovy="70"/>
  </worldbody>
</mujoco>
"""
RES = (32, 24)
GREY = (0.6, 0.6, 0.6)


def test_fog_fades_to_its_color_with_distance(scene):
  """Misses show the fog color; the fogged floor moves towards it more with distance."""
  plain = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", "depth")).render()
  fogged = scene(
    XML,
    cam_res=RES,
    outputs=rw.Outputs.of("rgb", "depth"),
    atmosphere=rw.settings.Atmosphere(color=GREY, haze=0.3),
  ).render()
  assert fogged.r.features.fog and not plain.r.features.fog
  depth = plain.depth()[0]
  rgb, base = fogged.rgb()[0], plain.rgb()[0]
  sky = depth == 0.0
  assert sky.any()
  np.testing.assert_allclose(rgb[sky], np.broadcast_to(GREY, rgb[sky].shape), atol=1.5 / 255)
  # Fraction of the way to the fog color (red channel of the red floor), by depth.
  floor = (~sky) & (base[..., 0] > base[..., 2])
  t = (base[..., 0] - rgb[..., 0])[floor] / (base[..., 0] - GREY[0])[floor]
  near, far = depth[floor] < np.median(depth[floor]), depth[floor] >= np.median(depth[floor])
  assert 0.0 < t[near].mean() < t[far].mean() < 1.0


def test_fog_color_is_relative_to_the_lens_exposure(scene):
  """The display-relative fog color is the color shown at any camera exposure: a lens exposure of
  +1 EV (`CameraSpec.exposure`) keeps the misses at the fog color, as the world exposure does."""
  base = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", "depth"))
  spec = dataclasses.replace(next(iter(base.scene.cameras.values())), exposure=1.0)
  fogged = scene(
    XML,
    cam_res=RES,
    cameras=[spec],
    outputs=rw.Outputs.of("rgb", "depth"),
    atmosphere=rw.settings.Atmosphere(color=GREY, haze=0.3),
  ).render()
  rgb, sky = fogged.rgb()[0], fogged.depth()[0] == 0.0
  assert sky.any()
  np.testing.assert_allclose(rgb[sky], np.broadcast_to(GREY, rgb[sky].shape), atol=1.5 / 255)


def test_linear_fog_window():
  """MuJoCo's linear fog: clear before start, opaque after end (haze 0)."""
  import warp as wp

  from raywarp.core.shade.environment.fog import fog_amount

  @wp.kernel
  def amounts(d: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = fog_amount(wp.vec3(2.0, 6.0, 0.0), d[i])

  d = wp.array([0.0, 2.0, 4.0, 6.0, 1.0e10], dtype=float)
  out = wp.zeros(5, dtype=float)
  wp.launch(amounts, dim=5, inputs=[d], outputs=[out])
  np.testing.assert_allclose(out.numpy(), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)


def test_per_world_fog_never_recompiles(scene):
  """Per-world fog tables ("fog.color", "fog.range"): world 0 without fog renders as no fog at
  all, world 1 with haze; writing them keeps the Features."""
  fog = rw.settings.Atmosphere(color=GREY)
  decl = rw.PerWorld(env=["fog.color", "fog.range"])
  s = scene(XML, nworld=2, cam_res=RES, per_world=decl, atmosphere=fog).render()
  features = s.r.features
  with module_loads() as loads:
    s.r.per_world["fog.range"].assign(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], np.float32))
    s.r.per_world["fog.color"].assign(np.array([[1.0, 0.0, 0.0], [0.2, 0.2, 0.2]], np.float32))
    rgb = s.render().rgb()
  assert s.r.features is features and not loads, loads
  plain = scene(XML, cam_res=RES).render().rgb()[0]
  np.testing.assert_allclose(rgb[0], plain, atol=1.0 / 255)
  assert np.abs(rgb[1] - plain).max() > 0.1
