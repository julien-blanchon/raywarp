"""The deferred display pass (`RenderSettings.deferred_display`): the frame's scene-linear HDR
image through automatic exposure, chromatic aberration and the grading into the display outputs,
after the render.

Off (the default) nothing here is allocated or launched. On, the render also writes the "hdr"
output (`RenderSettings.written_outputs`); per frame:

1. automatic exposure (`exposure.AutoExposure`): metering + solve, one launch per camera + one;
2. per camera, `make_display`'s kernel (on the CPU in bands of pixels over the cores): each
   pixel's exposed color (per-channel positions where the world's chromatic aberration strength
   is not 0: a run-time branch) times its view's exposure scale, with the contact AO on its
   ambient share occluded by the dynamic objects' proxies (`contact.apply`; off: the kernel
   variant without that code), through `transform.make_display`, written by `display.write_display`
   in the output word's layout (the coverage alpha kept).

The launches cover the frame's launch units (`Source.worlds`: every world, or the entries of
`Renderer.render(active=...)`'s list, holes returning at once): the worlds not listed launch
nothing and keep their outputs and exposure state. Every launch has a fixed shape (CUDA-graph
capturable), and nothing is read back to the host.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Final, cast, final

import warp as wp

from raywarp.core.kernels.abi.output_layout import OUT_ALPHA, OUT_HDR_SQRT, BufferKind, OutputLayout
from raywarp.core.kernels.abi.structs import WANT_RGB, EnvironmentArrays, Lens, OutputArrays, u32s
from raywarp.core.kernels.abi.tables import table_row
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.kernels.display import write_display
from raywarp.core.pipeline.platform.device import launch_bands
from raywarp.core.post.display import lens
from raywarp.core.post.exposure import auto as exposure_lib
from raywarp.core.post.exposure import source as exposure_source
from raywarp.core.post.exposure.source import Source
from raywarp.core.post.grading import color, transform
from raywarp.core.shade.ao import contact
from raywarp.core.shade.shadows import area

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.settings import RenderSettings

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Fewest pixels per CPU band of a display launch (`device.launch_bands`; one launch on a GPU).
DISPLAY_BAND: Final = 4096


@functools.cache
def make_display(kind: int, *, graded: bool, ao: bool):
  """The deferred display kernel of source `kind`, with the grading (`graded`) or without, with
  the contact AO (`ao`: a kernel of its own, reading the depth output and the frame's proxies, so
  that off it costs nothing) or without (the outputs are the word's, the rgb buffer's coverage
  kept with `ColorOutput.alpha`)."""
  fetch = exposure_source.make_fetch(kind)
  sample = exposure_source.make_sample(kind)
  display = transform.make_display(graded)

  @wp.func
  def exposed(
    # In:
    src: Source,
    env: EnvironmentArrays,
    scale: wp.array2d[float],
    world: int,
    cam: int,
    x: int,
    y: int,
    i: int,
    width: int,
    height: int,
  ) -> wp.vec3:
    """Pixel i's (x, y) exposed color (its chromatic aberration's channel positions)."""
    c = fetch(src, world, cam, i)
    strength = env.look[table_row(world, env.look.shape[0])].aberration
    if strength != 0.0:
      ab = lens.channel_positions(x, y, width, height, strength)
      red = sample(src, world, cam, width, height, wp.vec2(ab[0, 0], ab[0, 1]))
      blue = sample(src, world, cam, width, height, wp.vec2(ab[1, 0], ab[1, 1]))
      c = wp.vec3(red[0], c[1], blue[2])
    return c * scale[world, cam]

  @wp.func
  def write(
    # In:
    src: Source,
    env: EnvironmentArrays,
    world: int,
    cam: int,
    i: int,
    width: int,
    height: int,
    c: wp.vec3,
    # Out:
    out: OutputArrays,
  ):
    """Writes pixel i's displayed color `c` in the output word's layout."""
    adr = src.lens[cam].adr + i
    a = float(1.0)
    if out_on(out.bits, wp.static(OUT_ALPHA)):
      a = color.unpack_rgba(u32s(out, out.table.rgb)[world, adr])[3]
    write_display(out, world, adr, src.lens[cam].adr, width * height, display(env, world, c), a)

  if ao:

    @wp.kernel(module="unique", enable_backward=False)
    def deferred_display_ao(
      # In:
      src: Source,
      env: EnvironmentArrays,
      scale: wp.array2d[float],
      fovy: wp.array2d[float],
      proxies: contact.Proxies,
      strength: float,
      cam: int,
      width: int,
      height: int,
      first: int,
      # Out:
      out: OutputArrays,
    ):
      """`deferred_display` with the contact AO of `strength` over the frame's `proxies`."""
      unit, j = wp.tid()
      i = first + j
      world = exposure_source.unit_world(src, unit)
      if world < 0 or (src.lens[cam].wants & wp.static(WANT_RGB)) == 0:
        return
      y = i // width
      x = i - y * width
      c = exposed(src, env, scale, world, cam, x, y, i, width, height)
      view_fovy = fovy[table_row(world, fovy.shape[0]), cam]
      c = contact.apply(c, out, proxies, world, cam, src.lens[cam].adr, x, y, width, height, view_fovy, strength)
      write(src, env, world, cam, i, width, height, c, out)

    return deferred_display_ao

  @wp.kernel(module="unique", enable_backward=False)
  def deferred_display(
    # In:
    src: Source,
    env: EnvironmentArrays,
    scale: wp.array2d[float],
    cam: int,
    width: int,
    height: int,
    first: int,
    # Out:
    out: OutputArrays,
  ):
    """Pixel `first` + j of camera `cam` of launch unit `unit`'s world (see the module doc;
    `first`: the launch's first pixel, `Imaging.run`'s bands)."""
    unit, j = wp.tid()
    i = first + j
    world = exposure_source.unit_world(src, unit)
    if world < 0 or (src.lens[cam].wants & wp.static(WANT_RGB)) == 0:
      return
    y = i // width
    x = i - y * width
    write(src, env, world, cam, i, width, height, exposed(src, env, scale, world, cam, x, y, i, width, height), out)

  return deferred_display


def source_kind(layout: OutputLayout) -> int:
  """The source the deferred display reads under this output layout (`exposure_source` kinds:
  the "hdr" output's dtype)."""
  if layout.bits & OUT_HDR_SQRT:
    return exposure_source.HDR_SQRT16
  return exposure_source.HDR16 if layout.is_half("hdr") else exposure_source.HDR


# Every source `source_kind` returns: the render's "hdr" image (its dtype is the output word's:
# one more output or dtype changes the source, not the kernels).
SOURCE_KINDS: Final = (exposure_source.HDR, exposure_source.HDR16, exposure_source.HDR_SQRT16)


def first_frame_kernels(settings: RenderSettings) -> list[wp.Kernel]:
  """The deferred display's kernels under `settings` (`warmup`): listed whether or not the
  settings defer the display, and for every source of the render's "hdr" image, so that turning
  automatic exposure on or changing the outputs later compiles nothing."""
  graded, ao = settings.grading.enabled, settings.contact_ao() > 0.0
  out = [k for kind in SOURCE_KINDS for k in (make_display(kind, graded=graded, ao=ao), exposure_lib.make_meter(kind))]
  gather = [contact.gather_kernel, contact.group_kernel, contact.bin_kernel] if ao else []
  return [*out, *exposure_lib.STATE_KERNELS, *gather]


def reachable_kernels() -> list[wp.Kernel]:
  """Every deferred display kernel a run-time setting change reaches, whatever the settings: each
  source's display with and without the grading and the contact AO, its meter, the exposure
  state's kernels, the soft area shadows' filter (`area.apply_kernel`), the contact AO's
  proxy gather, grouping and binning (`contact.gather_kernel`, `group_kernel`, `bin_kernel`).
  The prebuilt fill lists them (the warm-up only the settings' own, `first_frame_kernels`)."""
  flags = (False, True)
  displays = [make_display(kind, graded=graded, ao=ao) for kind in SOURCE_KINDS for graded in flags for ao in flags]
  meters = [exposure_lib.make_meter(kind) for kind in SOURCE_KINDS]
  state = list(exposure_lib.STATE_KERNELS)
  ao = (contact.gather_kernel, contact.group_kernel, contact.bin_kernel)
  extra = [area.apply_kernel, *ao]
  return [*displays, *meters, *state, *extra]


@final
class Imaging:
  """The deferred display of a Renderer: its exposure state and the per-frame launches (see the
  module doc). Built only when `deferred_display()` is on."""

  def __init__(self, device: wp.Device, nworld: int, sizes: list[tuple[int, int]]) -> None:
    """Empty state for `nworld` worlds of cameras of `sizes` ((W, H) each)."""
    self.device, self.nworld, self.sizes = device, nworld, list(sizes)
    self.auto: exposure_lib.AutoExposure | None = None
    self.ones: wp.array[float, [int, int]] = wp.ones((nworld, len(sizes)), dtype=float, device=device)
    self.placeholders = Placeholders(device)
    self.settings: RenderSettings | None = None

  def configure(self, settings: RenderSettings) -> None:
    """(Re)allocates what the settings use; kept state survives runtime changes."""
    if not settings.auto_exposure():
      self.auto = None
    elif self.auto is None:
      self.auto = exposure_lib.AutoExposure(self.device, self.nworld, len(self.sizes))
    if self.auto is not None:
      self.auto.configure(settings.exposure)
    self.settings = settings

  def reset(self, worlds: wp.array[bool, [int]] | None) -> None:
    """The masked worlds' cameras (None: all) meter their next frame without adaptation."""
    if self.auto is not None:
      self.auto.reset(worlds)

  def run(  # noqa: PLR0913  # the frame's inputs
    self,
    src: Source,
    kind: int,
    env: EnvironmentArrays,
    out: OutputArrays,
    fovy: wp.array[float, [int, int]],
    ao: tuple[float, contact.Proxies] | None,
  ) -> None:
    """The frame's deferred display (see the module doc) into the outputs `out`; `fovy`: the
    cameras' vertical fields of view [deg] (1 or nworld, ncam), `ao`: the contact AO's strength
    and the frame's proxies (None: off)."""
    s = self.settings
    if s is None:
      raise RuntimeError("Imaging.run before configure")
    scale = self.ones
    nunit = src.worlds.shape[0] or self.nworld
    if self.auto is not None:
      self.auto.run(src, kind, self.sizes, nunit)
      scale = self.auto.scale
    kernel = make_display(kind, graded=s.grading.enabled, ao=ao is not None)
    for cam, (width, height) in enumerate(self.sizes):

      def band(a: int, b: int, cam: int = cam, width: int = width, height: int = height) -> None:
        """Pixels [a, b) of camera `cam`."""
        args = [src, env, scale] if ao is None else [src, env, scale, fovy, ao[1], ao[0]]
        inputs = [*args, cam, width, height, a]
        wp.launch(kernel, dim=(nunit, b - a), inputs=inputs, outputs=[out], device=self.device)

      launch_bands(kernel.module, device=self.device, n=width * height, launch=band, min_band=DISPLAY_BAND)


def source(
  *,
  kind: int,
  flat: dict[BufferKind, AnyDeviceArray],
  exposure: wp.array[float, [int]],
  lens_table: wp.array[Lens, [int]],
  worlds: wp.array[int, [int]],
  placeholders: Placeholders,
) -> Source:
  """The `Source` struct of a frame over the output buffers `flat` (`render_outputs.Buffers.flat`;
  unused arrays: `placeholders`) and the launch units' `worlds` (`Source.worlds`)."""
  src = Source()
  # The hdr output's dtype follows its spec (float32 or float16): one of the two fields.
  src.hdr = flat["hdr"] if kind == exposure_source.HDR else placeholders.vec3
  half = kind in (exposure_source.HDR16, exposure_source.HDR_SQRT16)
  src.hdr16 = cast("wp.array[wp.vec3h, [int, int]]", flat["hdr"]) if half else placeholders.vec3h
  src.exposure, src.lens, src.worlds = exposure, lens_table, worlds
  return src


@final
class Placeholders:
  """Small arrays standing for a `Source`'s unused fields."""

  def __init__(self, device: wp.Device) -> None:
    """The placeholders on `device`."""
    self.vec3: wp.array[wp.vec3, [int, int]] = wp.zeros((1, 1), dtype=wp.vec3, device=device)
    self.vec3h: wp.array[wp.vec3h, [int, int]] = wp.zeros((1, 1), dtype=wp.vec3h, device=device)
