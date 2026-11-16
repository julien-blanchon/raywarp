"""Renderer comparison on one simulation and one camera: raywarp, MuJoCo Warp's own batch renderer
and MuJoCo's OpenGL renderer (when an OpenGL context is available: `MUJOCO_GL=egl` on a GPU
machine). Writes one synchronised raw clip per renderer (`<out>_<renderer>.mp4`) and the panels
side by side (`<out>.mp4`).

  python examples/demos/compare_renderers.py --size 1920x360 --out compare.mp4
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

import _common
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco

type Panel = Callable[[], _common.Image]


def with_camera(path: Path, view: _common.View) -> mujoco.MjModel:
  """The model of `path` with a camera "gallery" at `view` (every renderer renders that one)."""
  spec = mujoco.MjSpec.from_file(str(path))
  eye, target = np.asarray(view[0]), np.asarray(view[1])
  forward = (target - eye) / np.linalg.norm(target - eye)
  x = np.cross(forward, [0.0, 0.0, 1.0])
  x /= np.linalg.norm(x)
  spec.worldbody.add_camera(name="gallery", pos=eye, xyaxes=[*x, *np.cross(x, forward)], fovy=view[2])
  return spec.compile()


def raywarp_panel(mjm: mujoco.MjModel, m: mjw.Model, d: mjw.Data, size: tuple[int, int]) -> Panel:
  """The image of world 0 by raywarp (default settings)."""
  r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m, d), nworld=d.nworld, cameras="gallery", resolution=size)

  def panel() -> _common.Image:
    r.render(d)
    return _common.rgb_of(r).numpy()[0, 0, ..., :3]

  return panel


def mujoco_warp_panel(mjm: mujoco.MjModel, m: mjw.Model, d: mjw.Data, size: tuple[int, int]) -> Panel:
  """MuJoCo Warp's renderer (shadows on)."""
  cam = mjm.camera("gallery").id
  rc = mjw.create_render_context(mjm, nworld=d.nworld, cam_res=size, render_rgb=True, use_shadows=True, cam_active=[cam])
  rgb = wp.zeros((d.nworld, size[1], size[0]), dtype=wp.vec3)

  def panel() -> _common.Image:
    mjw.render(m, d, rc)
    mjw.get_rgb(rc, cam, rgb)
    return (np.clip(rgb.numpy()[0], 0.0, 1.0) * 255).astype(np.uint8)

  return panel


def opengl_panel(mjm: mujoco.MjModel, d: mjw.Data, size: tuple[int, int]) -> Panel | None:
  """MuJoCo's OpenGL renderer of world 0 (None without an OpenGL context)."""
  try:
    gl = mujoco.Renderer(mjm, size[1], size[0])
  except Exception as e:  # noqa: BLE001  # no display / EGL / OSMesa: the panel is left out
    print(f"MuJoCo OpenGL renderer unavailable ({e}): skipped")
    return None
  host = mujoco.MjData(mjm)

  def panel() -> _common.Image:
    mjw.get_data_into(host, mjm, d)  # world 0 to the host
    gl.update_scene(host, "gallery")
    return gl.render()

  return panel


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1920x360",
  frames: int = 240,
  out: str = "compare.mp4",
  scene: str = "franka",
) -> None:
  """Steps the simulation and records every renderer's view of world 0."""
  width, height = _common.parse_size(size)
  path, view = _common.scene_path(scene)
  with wp.ScopedDevice(device):
    mjm = with_camera(path, view or _common.mjcf_view(mujoco.MjModel.from_xml_path(str(path))))
    m, d = mjw.put_model(mjm), mjw.make_data(mjm, nworld=worlds)
    tile = (width // 3 // 2 * 2 - 4, height // 2 * 2)
    panels = {"raywarp": raywarp_panel(mjm, m, d, tile), "mujoco_warp": mujoco_warp_panel(mjm, m, d, tile)}
    if (gl := opengl_panel(mjm, d, tile)) is not None:
      panels["mujoco_opengl"] = gl
    stem = Path(out).with_suffix("")
    canvases = {name: _common.Canvas(*tile) for name in panels}
    side = _common.Canvas(width, height)
    with contextlib.ExitStack() as stack:
      raw = {name: stack.enter_context(_common.record(f"{stem}_{name}.mp4", c.rgb)) for name, c in canvases.items()}
      both = stack.enter_context(_common.record(out, side.rgb))
      for _ in range(frames):
        mjw.step(m, d)
        images = {name: panel() for name, panel in panels.items()}
        for name, image in images.items():
          canvases[name].show(image)
          raw[name].write(canvases[name].rgb)
        side.show(_common.row(list(images.values())))
        both.write(side.rgb)
    print(f"{out}: {', '.join(panels)}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
