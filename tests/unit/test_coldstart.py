"""Cold-start helpers: texture mip chains, first-frame warm-up, the OptiX programs' one PTX
compile."""

import numpy as np
import pytest
import warp as wp
from conftest import make_scene

import raywarp as rw
from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.textures import uploader

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <light pos="1 1 2" dir="0 0 -1" castshadow="true"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
"""


def _warp_chain(img: np.ndarray) -> list[np.ndarray]:
  """Warp's own host mip chain of `img`."""
  t = wp.Texture2D.__new__(wp.Texture2D)
  t._ndim = 2
  t._mip_shapes = wp.Texture2D._compute_mip_shapes((img.shape[1], img.shape[0], 1), 2, 0)
  t._num_mip_levels = len(t._mip_shapes)
  return wp.Texture2D._generate_mipmap_chain(t, img)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16])
@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (32, 64), (256, 128)])
def test_mip_chain_matches_warp(dtype, shape):
  """Power-of-two mip filters retain Warp's texels bit for bit, including 1-texel axes."""
  rng = np.random.default_rng(0)
  img = (
    rng.integers(0, 256, (*shape, 4)).astype(np.uint8)
    if dtype == np.uint8
    else (rng.random((*shape, 4)) * 3.0).astype(np.float16)
  )
  fast = uploader.box_chain(img)  # pyrefly: ignore[bad-argument-type]  # a uint8 or float16 image
  ref = _warp_chain(img)
  assert len(ref) == len(fast)
  for a, b in zip(ref, fast, strict=True):
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")
def test_warmup_lists_the_first_frame_modules():
  """The warm-up compiles the feature kernels the first frame launches (GI: the probe rays and the
  free-path survey that relocates them; light grid, the motion output's history)."""
  s = make_scene(
    _XML,
    cam_res=(16, 16),
    gi=rw.settings.GI(enabled=True),
    lighting=rw.settings.Lights(light_grid=True),
    outputs=rw.Outputs.of("rgb", "motion"),
    performance=rw.settings.Performance(traversal="software"),
  )
  ctx, backend = s.r.context(), s.r._backend
  default, render = warmup.first_frame(ctx, backend).modules()
  names = {m.name for m in default + render}
  assert "raywarp.core.pipeline.cameras" in names  # the camera poses, every frame
  for part in ("trace_probe_rays", "survey_rays", "visibility", "history", "megakernel"):
    assert any(part in n for n in names), (part, names)
  warmup.start(ctx, backend).wait()
  assert np.isfinite(s.render().rgb()).all()


def test_warmup_lists_the_validation_kernels():
  """A renderer with shared scenery validates every frame: the warm-up compiles that module (it
  listed `core.pipeline.worlds.validator`, which holds no kernel: the first frame compiled the check
  serially)."""
  s = make_scene(_XML, cam_res=(16, 16), performance=rw.settings.Performance(traversal="software"))
  assert s.r.layout.nshared
  default, _ = warmup.first_frame(s.r.context(), s.r._backend).modules()
  assert "raywarp.core.pipeline.worlds.validator_kernels" in {m.name for m in default}


def test_warmup_lists_the_relight_and_rebuild_modules():
  """Per-world radiance inputs (a relit GI bake on the first re-randomisation) and periodic BVH
  rebuilds: the warm-up compiles their modules, so no later frame does; none without them."""
  kw = {"cam_res": (8, 8), "nworld": 2, "gi": rw.settings.GI(enabled=True)}
  sw = rw.settings.Performance(traversal="software")
  dr = make_scene(_XML, per_world=rw.PerWorld(material=["base_color"], pose=["*"]), performance=sw, **kw)
  default, _ = warmup.first_frame(dr.r.context(), dr.r._backend).modules()
  names = {m.name for m in default}
  assert "raywarp.core.shade.gi.relight" in names and any("relight_rays" in n for n in names), names
  assert "raywarp.core.trace.accel.lbvh" in names  # the default rebuild cadence
  plain = make_scene(_XML, performance=sw, **kw)  # no per-world input, no world BVH
  names = {m.name for m in warmup.first_frame(plain.r.context(), plain.r._backend).modules()[0]}
  assert not any("relight" in n for n in names) and "raywarp.core.trace.accel.lbvh" not in names


def _optix_available() -> bool:
  if not wp.is_cuda_available():
    return False
  from raywarp.core.pipeline.platform.device import device_caps

  return device_caps("cuda:0").optix.available


@pytest.mark.skipif(not _optix_available(), reason="OptiX backend unavailable (CUDA / driver OptiX / SDK headers)")
def test_cold_optix_build_compiles_one_ptx(tmp_path, monkeypatch):
  """P21: a cold build (empty Warp kernel cache) compiles the programs' PTX once, for the device's
  architecture capped at sm_89, off the render thread, and renders as the software traversal."""
  from raywarp.core.trace.optix import pipeline
  from raywarp.core.trace.optix.context import MAX_PTX_ARCH

  archs = []
  real = pipeline.PtxCompile._compile

  def compile_ptx(module, arch, path):
    archs.append(arch)
    return real(module, arch, path)

  monkeypatch.setattr(pipeline.PtxCompile, "_compile", staticmethod(compile_ptx))
  monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path))
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "wpbr"))  # no cached PTX either
  monkeypatch.setattr(pipeline, "read_ptx", lambda *_a, **_k: None)  # nor the prebuilt bundle's
  kw = {"cam_res": (24, 16), "outputs": rw.Outputs.of("rgb")}
  optix = make_scene(_XML, performance=rw.settings.Performance(traversal="optix"), **kw).render()
  assert archs == [min(wp.get_device("cuda:0").arch, MAX_PTX_ARCH)]
  software = make_scene(_XML, performance=rw.settings.Performance(traversal="software"), **kw).render()
  np.testing.assert_allclose(optix.rgb(), software.rgb(), atol=3 / 255)


def _warm_names(**kw) -> list[str]:
  """The warm-up's module names (with repeats) for a software renderer of `_XML` with `kw`."""
  s = make_scene(_XML, performance=rw.settings.Performance(traversal="software"), **kw)
  default, render = warmup.first_frame(s.r.context(), s.r._backend).modules()
  return [m.name for m in default + render]


def test_warmup_lists_each_module_once_and_only_modules_with_kernels():
  """Q4: every warm-up entry is a module holding kernels, listed once (`trace.accel.bounds` was
  listed by the backend and again for the light grid: two threads compiled it; an entry naming a
  module without kernels warmed nothing)."""
  names = _warm_names(cam_res=(128, 128), lighting=rw.settings.Lights(light_grid=True))
  assert len(names) == len(set(names)), names
  assert all(wp.get_module(n).kernels or "megakernel" in n or "visibility" in n for n in names), names
  for module in ("trace.accel.bounds", "shade.lights.pack", "shade.lights.flood"):
    assert f"raywarp.core.{module}" in names, (module, names)


def test_warmup_skips_the_mesh_frames_kernel_without_mesh_frames():
  """The meshes' trace-frame kernel (`kernels.abi.rays`) is warmed only when the frame launches
  it."""
  names = _warm_names(cam_res=(16, 16))
  assert "raywarp.core.kernels.abi.rays" not in names


def test_warmup_module_kernels_rejects_a_module_without_kernels():
  """A warm-up entry naming a module without module-level kernels fails loudly."""
  from raywarp.core.post.temporal import store

  with pytest.raises(RuntimeError, match="no module-level Warp kernel"):
    warmup.module_kernels(store)


def test_warmup_compile_error_is_raised_by_wait(monkeypatch):
  """A compile error on a warm-up thread is raised by `wait` (it was dropped)."""
  s = make_scene(_XML, cam_res=(16, 16), performance=rw.settings.Performance(traversal="software"))

  def failing(device, mods, block_dim):
    raise RuntimeError("compile failed")

  monkeypatch.setattr(warmup.Warmup, "_run", staticmethod(failing))
  w = warmup.start(s.r.context(), s.r._backend)
  with pytest.raises(RuntimeError, match="compile failed"):
    w.wait()


@pytest.mark.parametrize("backend_name", ["software", "optix"])
def test_backend_close_drains_owned_warmup_threads(monkeypatch, backend_name):
  """Closing either backend waits for actual warmup workers and their predecessor chain."""
  import threading
  from concurrent.futures import ThreadPoolExecutor

  from raywarp.core.trace.optix_backend import OptixBackend
  from raywarp.core.trace.software_backend import SoftwareBackend

  with wp.ScopedDevice("cpu"):
    s = make_scene(
      _XML, cam_res=(2, 2), gi=rw.settings.GI(enabled=False), performance=rw.settings.Performance(traversal="software")
    )
  s.r._backend.prepare(s.r.context(), wait=True)
  backend = SoftwareBackend() if backend_name == "software" else OptixBackend()
  entered, release, close_started = threading.Event(), threading.Event(), threading.Event()

  def blocked(device, mods, block_dim):
    entered.set()
    if not release.wait(3):
      raise RuntimeError("test did not release compilation")

  monkeypatch.setattr(warmup.Warmup, "_run", staticmethod(blocked))
  first = warmup.start(s.r.context(), s.r._backend)
  last = warmup.start(s.r.context(), s.r._backend, after=first)
  backend.warm = last

  def close():
    close_started.set()
    backend.close()

  try:
    assert entered.wait(3)
    with ThreadPoolExecutor(max_workers=1) as pool:
      future = pool.submit(close)
      try:
        assert close_started.wait(3)
        with pytest.raises(TimeoutError):
          future.result(timeout=0.05)
      finally:
        release.set()
      future.result(timeout=3)
    assert backend.warm is None
    assert all(not worker.is_alive() for pending in (first, last) for worker in pending._threads)
    backend.close()
  finally:
    release.set()
    last.join()
    s.r.close()
