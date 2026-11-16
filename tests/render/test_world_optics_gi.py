"""Per-world instance optics in the GI probes: a shade under the room's light is glass in some
worlds and opaque in others (`instance.material` per world, translucent shadows); the probes'
shadow rays read each world's transmittance, so each world's image equals the render of a
renderer of that world alone."""

from __future__ import annotations

import numpy as np
import warp as wp

import raywarp as rw

_SIZE = 16
_SHADE = 5  # the shade's instance index (SceneBuilder order)


def _room() -> rw.Scene:
  """A white box room lit by a sphere light under the ceiling, a thin shade below the light
  (opaque white as authored), and a glass material for the shade to switch to (row 1)."""
  b = rw.SceneBuilder()
  white = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_material(rw.Material(transmission_weight=1.0, transmission_color=(0.9, 0.9, 0.9), specular_ior=1.5))
  h, t = 1.0, 0.05
  boxes = {
    "floor": ((h, h, t), (0.0, 0.0, -t)),
    "ceiling": ((h, h, t), (0.0, 0.0, 2 * h + t)),
    "back": ((h, t, h), (0.0, h + t, h)),
    "left": ((t, h, h), (-h - t, 0.0, h)),
    "right": ((t, h, h), (h + t, 0.0, h)),
    "shade": ((0.6, 0.6, 0.02), (0.0, 0.0, 1.5)),
  }
  for path, (size, at) in boxes.items():
    b.add_instance("box", material=white, size=size, xform=np.c_[np.eye(3), at], path=path)
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=20.0, radius=0.05, xform=np.c_[np.eye(3), [0.0, 0.0, 1.8]]))
  return b.build()


def _settings() -> rw.RenderSettings:
  """HDR, software traversal, translucent shadows, a coarse fast probe bake."""
  return rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("hdr"),
      "performance.traversal": "software",
      "shadows.translucent": "transmittance",
      "gi.enabled": True,
      "gi.spacing": 0.5,
      "gi.rays": 32,
      "gi.bake_updates": 4,
      "reflections.enabled": False,
      "lighting.light_grid": False,
      "shadows.samples": 1,
    }
  )


def _render(shades: list[int]) -> np.ndarray:
  """(nworld, H, W, 3) HDR of a renderer whose world w shows the shade in material `shades[w]`."""
  r = rw.Renderer(
    rw.upload(_room(), device="cpu"),
    nworld=len(shades),
    cameras=[rw.CameraSpec(fovy=80.0)],
    resolution=(_SIZE, _SIZE),
    settings=_settings(),
    per_world=rw.PerWorld(material=["instance.material"]),
  )
  with r:
    assert r.features.world_optics
    q = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))  # looking along +y, from the front
    r.camera_pose.assign(np.tile(np.array([0.0, -0.95, 0.9, *q], np.float32), (len(shades), 1, 1)))
    table = r.per_world["instance.material"]
    values = table.numpy()
    values[:, _SHADE] = shades
    table.assign(values)
    out = r.render().hdr
    assert out is not None
    return out.numpy().reshape(len(shades), _SIZE, _SIZE, -1)[..., :3].astype(np.float64)


def test_probe_shadows_read_each_world_s_shade() -> None:
  """Glass and opaque shades over four worlds: each world renders as a renderer of that world
  alone (the probes' shadow rays read the world's own transmittance, not world 0's); the glass
  worlds are brighter."""
  with wp.ScopedDevice("cpu"):
    image = _render([1, 0, 1, 0])
    glass, opaque = _render([1])[0], _render([0])[0]
  for w, alone in enumerate((glass, opaque, glass, opaque)):
    np.testing.assert_allclose(image[w], alone, rtol=1e-4, atol=1e-6)
  assert image[0].mean() > 1.05 * image[1].mean()
