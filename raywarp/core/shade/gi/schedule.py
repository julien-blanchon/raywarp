"""The GI bake / update policy a Renderer runs on its frames (`Schedule`: gi.bake_updates,
gi.update_every): the bake on the first render, then the periodic updates, gated on the device
(`cadence.Cadence`).
"""

from __future__ import annotations

from typing import Final, NamedTuple, final

import warp as wp

from raywarp.core.pipeline.platform.cadence import Cadence
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.shade.gi.volume import ProbeInputs, ProbeVolume, UpdateStep
from raywarp.settings import GI

# The weight of the previous maps in a periodic update (RTXGI's default).
HYSTERESIS: Final = 0.97


@final
class Schedule:
  """The GI bake and periodic-update policy of a Renderer (gi.* settings).

  Args:
    vol: the placed volume.
    settings: the renderer's GI settings.
  """

  def __init__(self, vol: ProbeVolume, settings: GI) -> None:
    """Keeps the policy and the device cadence (on the volume's device)."""
    self.vol = vol
    self.settings = settings
    # Periodic cadence (device): frames since the bake, and the gate of this frame's update.
    self.cadence = Cadence(vol.device)
    # The closed gate of frozen bakes (a relit volume still recomputes its changed worlds).
    self._closed = wp.zeros(1, dtype=int, device=vol.device)

  def bake(self, kernel: wp.Kernel, inputs: ProbeInputs) -> None:
    """The initial bake (first render): `bake_updates` updates of every source, each adding a
    bounce, from one update's rays (`ProbeVolume.bake`)."""
    self.vol.bake(kernel, inputs, inputs.lights(), self.settings.bake_updates)

  def update(self, kernels: ProbeKernels, inputs: ProbeInputs, count: int) -> None:
    """`count` updates now (`Renderer.update_gi`): every source at once, bounced without bound;
    a relit volume's shared maps are updated untinted (world 0's current inputs), then its
    reference and every world's tint are recomputed against them."""
    vol = self.vol
    lit = inputs.lights()
    if vol.relight is not None:
      vol.relight.restart(vol.always)
    for _ in range(count):
      vol.update(kernels.trace, inputs, lit, step=UpdateStep(hysteresis=HYSTERESIS))
    if vol.relight is not None:
      vol.relight.step(relight_kernel_of(kernels), relit_inputs(inputs), vol.always)

  def step(self, kernels: ProbeKernels, inputs: ProbeInputs) -> None:
    """A frame after the bake: one update every gi.update_every frames, decided on the device, and
    the tint of a relit volume's worlds whose inputs changed (`relight.Relight.track` ran on this
    frame first; after an update its reference and every world's tint).

    No host value or allocation: a CUDA graph capturing the frame replays the cadence, and
    launches no update on the frames without one (`wp.capture_if`; eagerly, and with the
    backend's hardware rays, the kernels test the gate: a host read of it would synchronise).
    The update's end (`ProbeVolume.count`: the resample and the texture uploads) follows the
    conditional body, gated by its kernels as in an eager frame: CUDA graphs take no CUDA array
    copy in a conditional body.
    """
    every = self.settings.update_every
    vol = self.vol
    if every <= 0 and vol.relight is None:
      return
    due = self.cadence.tick(every) if every > 0 else self._closed
    if every > 0:
      step = UpdateStep(hysteresis=HYSTERESIS)
      lit = inputs.lights()

      def run() -> None:
        """The frame's update of the maps (traced and blended)."""
        vol.update(kernels.trace, inputs, lit, step=step, gate=due, counted=False)

      # A relit volume's tint reset first (gated, its texture upload outside the conditional
      # body): the shared maps are updated untinted, then the reference and every tint follow.
      if vol.relight is not None:
        vol.relight.restart(due)
      # A capture records the software rays' update as a conditional body; the backend's
      # hardware rays are cut out of the graph (no conditional body holds a cut): the update then
      # runs at every replay as eagerly, each launch testing the gate.
      if device_caps(vol.device).capturing() and inputs.rays is None:
        wp.capture_if(due, on_true=run)
      else:
        run()
      vol.count(due)
    if vol.relight is not None:
      vol.relight.step(relight_kernel_of(kernels), relit_inputs(inputs), due)


def relight_kernel_of(kernels: ProbeKernels) -> wp.Kernel:
  """The relight kernel of a relit volume's frame.

  Raises:
    RuntimeError: the frame's kernels were built without it.
  """
  if kernels.relight is None:
    raise RuntimeError("a relit GI volume needs the relight kernel (ProbeKernels.relight)")
  return kernels.relight


def relit_inputs(inputs: ProbeInputs) -> ProbeInputs:
  """The relight kernel's inputs of a relit volume's frame (`ProbeInputs.relit`).

  Raises:
    RuntimeError: the frame's inputs were built without them.
  """
  if inputs.relit is None:
    raise RuntimeError("a relit GI volume needs the relight kernel's inputs (ProbeInputs.relit)")
  return inputs.relit


class ProbeKernels(NamedTuple):
  """The probe kernels of a frame's Features (`kernels/probe.py`): the probe-ray update, and the
  per-world relight (`relight.Relight`)."""

  trace: wp.Kernel
  relight: wp.Kernel | None
