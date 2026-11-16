"""The Renderer's background warm-up beside the caller's own Warp work (stress test).

`Renderer()` returns while its first-frame modules load on background threads; the caller then
steps its simulation (other Warp modules loading and launching on the main thread, inside a
`wp.ScopedDevice`), reads arrays back and builds a second Renderer on the same scene (its
constructor launches renderer kernels whose modules the first warm-up may be loading). Every
module is unloaded before each round, so the warm-up and the simulation really load at the same
time. Checks: no crash, the caller's default device and stream unchanged, pixels equal to a
serial reference.
"""

from __future__ import annotations

import sys

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline import warmup

DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
NWORLD = 2
ROUNDS = 3
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.1 1"/>
    </body>
    <camera name="cam" pos="1.2 -1.2 1.1" xyaxes="1 1 0 -0.4 0.4 1"/>
  </worldbody>
</mujoco>
"""


@wp.kernel
def _scale(
  # In:
  x: wp.array[float],
  # Out:
  y: wp.array[float],
) -> None:
  """y = 2 x (the caller's own kernel, launched while the warm-up runs)."""
  i = wp.tid()
  y[i] = 2.0 * x[i]


def _simulator_modules() -> list[wp.Module]:
  """The loaded Warp modules of mujoco_warp (the caller's simulator)."""
  names = [n for n in sys.modules if n.startswith("mujoco_warp.")]
  return [m for m in (wp.get_module(n) for n in names) if m.kernels]


def _renderer_modules(r: rw.Renderer[mjw.Data]) -> list[wp.Module]:
  """The modules a Renderer's warm-up loads."""
  default, render = warmup.first_frame(r.context(), r._backend).modules()
  return default + render


def _caller_work(m: mjw.Model, d: mjw.Data, device: wp.Device) -> None:
  """The caller's work during the warm-up: simulation steps, its own kernel, host reads."""
  x = wp.array(np.arange(1024).astype(np.float32), device=device)
  y = wp.empty_like(x)
  for _ in range(3):
    mjw.forward(m, d)
    wp.launch(_scale, dim=x.shape[0], inputs=[x], outputs=[y])
    np.testing.assert_array_equal(y.numpy(), 2.0 * np.arange(1024).astype(np.float32))
    assert d.xpos.numpy().shape[0] == NWORLD


@pytest.mark.parametrize("device", DEVICES)
def test_warmup_beside_the_simulation_and_a_second_renderer(device: str) -> None:
  dev = wp.get_device(device)
  mjm = mujoco.MjModel.from_xml_string(XML)
  with wp.ScopedDevice(dev):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=NWORLD)
    mjw.forward(m, d)
    gpu = rw.integrations.mujoco.scene(mjm, m)
    ref = rw.Renderer(gpu, nworld=NWORLD, cameras=[0], resolution=(32, 32))
    ref.render(d)
    want = required(ref.outputs.rgb).numpy()
  modules = [*_renderer_modules(ref), *_simulator_modules()]
  stream = dev.stream if dev.is_cuda else None
  default = wp.get_device()
  for _ in range(ROUNDS):
    wp.synchronize_device(dev)
    for module in modules:
      module.unload()  # loaded again below, on the warm-up threads and the main thread at once
    with wp.ScopedDevice("cpu"), wp.ScopedDevice(dev):
      a = rw.Renderer(gpu, nworld=NWORLD, cameras=[0], resolution=(32, 32))
      _caller_work(m, d, dev)
      b = rw.Renderer(gpu, nworld=NWORLD, cameras=[0], resolution=(32, 32))
      assert wp.get_device() == dev
      a.render(d)
      b.render(d)
      np.testing.assert_array_equal(required(a.outputs.rgb).numpy(), want)
      np.testing.assert_array_equal(required(b.outputs.rgb).numpy(), want)
      a.close()
      b.close()
    assert wp.get_device() == default
    if stream is not None:
      assert dev.stream is stream


def test_packing_warmup_lists_every_packing_kernel() -> None:
  # A probe / grid bake's packing launches every kernel of trace/accel/packed.py (depth and triangle
  # counts first): the warm-up list must hold them all, or the first bake compiles one.
  from raywarp.core.trace.accel import packed

  defined = {k for k in vars(packed).values() if isinstance(k, wp.Kernel)}
  assert defined == set(packed.KERNELS)
