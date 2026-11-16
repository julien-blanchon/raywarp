"""The display transform with the grading fused in (`settings.Grading`): exposed scene-linear HDR
-> display-linear [0, 1].

`make_display(graded)` is the one display function of every pass that writes a pixel (the megakernel
bodies, the OptiX downsample, the deferred display pass). Only the deferred display pass grades
(`settings.imaging.deferred`: a look never compiles a render kernel); without the grading it is
exactly `color.display_transform` (no other code, no extra read). Order (Filament's colour grading):
white balance (Bradford), lift / gamma / gain (ASC CDL in LogC), saturation, tone mapper.

Per-world constants: one `Look` row per world (`EnvironmentArrays.look`), prepared every frame by
`Grading.prepare` (one launch of nworld threads) from the settings and the per-world fields
(`PER_WORLD_FIELDS`: white balance, saturation, lift, gamma, gain and the chromatic aberration
strength of the deferred display), so randomising them never recompiles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.core.kernels.abi.structs import EnvironmentArrays
from raywarp.core.kernels.abi.tables import batched1d
from raywarp.core.kernels.generic import specialize
from raywarp.core.post.grading import color
from raywarp.core.post.grading import colorimetry as gc
from raywarp.core.post.grading.look import LOOK_FIELDS, Look
from raywarp.scene.images.colorspace import REC709_Y

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.settings import RenderSettings

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Per-world fields of the look (`PerWorld(env=[...])`): name -> dtype (rows: `defaults`).
PER_WORLD_FIELDS: Final = LOOK_FIELDS

# display(env, worldid, exposed) -> display-linear color (`make_display`).
type Display = Callable[[EnvironmentArrays, int, wp.vec3], wp.vec3]


@wp.func
def grade(
  # In:
  look: Look,
  c: wp.vec3,
) -> wp.vec3:
  """The scene-referred grading of linear `c`: white balance, the ASC CDL in LogC, saturation."""
  c = wp.max(look.balance @ c, wp.vec3(0.0))
  v = gc.linear_to_logc(c)
  v = wp.cw_mul(v, look.slope) + look.offset
  for k in range(3):
    if v[k] > 0.0:
      v[k] = wp.pow(v[k], look.power[k])
  c = gc.logc_to_linear(v)
  y = wp.dot(c, wp.static(wp.vec3(*REC709_Y)))
  return wp.max(wp.vec3(y) + (c - wp.vec3(y)) * look.saturation, wp.vec3(0.0))


@specialize
def make_display(graded: bool):
  """display(env, worldid, exposed) -> display-linear color, with the grading (`graded`)."""

  @wp.func
  def display(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    exposed: wp.vec3,
  ) -> wp.vec3:
    """Exposed scene-linear HDR -> display-linear [0, 1] (see the module doc)."""
    if wp.static(not graded):
      return color.display_transform(exposed, env.params.tone_mapper)
    c = grade(batched1d(env.look, worldid), wp.max(exposed, wp.vec3(0.0)))
    return color.display_transform(c, env.params.tone_mapper)

  return display


@wp.kernel
def _prepare(
  # In:
  base: Look,
  white_balance: wp.array1d[wp.vec2],
  saturation: wp.array1d[float],
  lift: wp.array1d[wp.vec3],
  gamma: wp.array1d[wp.vec3],
  gain: wp.array1d[wp.vec3],
  aberration: wp.array1d[float],
  # Out:
  look: wp.array1d[Look],
) -> None:
  """World w's look from its per-world fields (white balance matrix, CDL, aberration)."""
  w = wp.tid()
  out = base
  wb = batched1d(white_balance, w)
  out.balance = gc.adaptation_matrix(wb[0], wb[1])
  out.saturation = batched1d(saturation, w)
  out.offset = batched1d(lift, w)
  g = batched1d(gamma, w)
  out.power = wp.vec3(1.0 / wp.max(g[0], 1.0e-6), 1.0 / wp.max(g[1], 1.0e-6), 1.0 / wp.max(g[2], 1.0e-6))
  out.slope = batched1d(gain, w)
  out.aberration = batched1d(aberration, w)
  look[w] = out


def defaults(settings: RenderSettings) -> dict[str, object]:
  """The settings' row of each per-world field (`PER_WORLD_FIELDS`)."""
  g = settings.grading
  return {
    "grading.white_balance": wp.vec2(g.kelvin, g.tint),
    "grading.saturation": float(g.saturation),
    "grading.lift": wp.vec3(*g.lift),
    "grading.gamma": wp.vec3(*g.gamma),
    "grading.gain": wp.vec3(*g.gain),
    "sensor.aberration": float(settings.sensor.aberration),
  }


@final
class Grading:
  """The look of a Renderer: the per-world `Look` rows and the per-frame prepare.

  Attributes:
    look: (nworld,) prepared rows (`EnvironmentArrays.look`).
    active: whether anything reads the rows (`configure`): `prepare` launches only then.
  """

  def __init__(self, device: wp.Device, nworld: int) -> None:
    """Empty rows for `nworld` worlds on `device` (`configure` turns them on)."""
    self.device = device
    self.nworld = nworld
    self.look: wp.array[Look, [int]] = wp.zeros(nworld, dtype=Look, device=device)
    self.active = False

  def configure(self, settings: RenderSettings) -> None:
    """Whether the rows are read: the grading, or the deferred display (its aberration
    strength)."""
    self.active = settings.grading.enabled or settings.deferred_display()

  def prepare(self, fields: Mapping[str, AnyDeviceArray]) -> None:
    """Writes every world's row from the per-world `fields` (`PER_WORLD_FIELDS`, 1 or nworld rows
    each); no launch when nothing reads them. Device work only (graph-capturable)."""
    if not self.active:
      return
    inputs = [Look(), *(fields[name] for name in PER_WORLD_FIELDS)]
    wp.launch(_prepare, dim=self.nworld, inputs=inputs, outputs=[self.look], device=self.device)
