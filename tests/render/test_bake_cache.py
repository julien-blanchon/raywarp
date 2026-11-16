"""Bake disk caches (W7, `core/pipeline/platform/bake_cache.py`): a warm build loads the light grid,
the GI volume, the sky visibility and the environment prefilter / SH from disk without recomputing
them, and renders bit-identical images to an uncached build; a changed input misses."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import recipe, required

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.shade import gi as gi_lib
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.lights import grid
from raywarp.core.trace.software_backend import SoftwareBackend
from raywarp.scene.images import latlong

# A box room with an opening and two point lights, under a gradient environment.
_XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/></visual>
  <asset><material name="mat" rgba="0.8 0.6 0.4 1" roughness="0.5"/></asset>
  <worldbody>
    <light pos="0 0 0.8" diffuse="1 1 1" intensity="20"/>
    <light pos="1.5 0 0.8" diffuse="1 0.5 0.5" intensity="20"/>
    <geom type="plane" size="3 3 0.1" material="mat"/>
    <geom type="box" pos="0 0 1.0" size="0.6 0.6 0.025" material="mat"/>
    <geom type="box" pos="0.6 0 0.5" size="0.025 0.6 0.5" material="mat"/>
    <geom type="sphere" pos="-0.2 0.1 0.2" size="0.2" material="mat"/>
    <camera name="cam" pos="-1.8 -1.2 1.2" xyaxes="0.55 -0.83 0 0.3 0.2 0.93" fovy="60"/>
  </worldbody>
</mujoco>
"""

_SKY = np.linspace(0.2, 2.0, 32 * 64 * 3, dtype=np.float32).reshape(32, 64, 3)

_GI = {"gi": rw.settings.GI(enabled=True, max_probes=64, rays=32, bake_updates=4)}
_GI_UPDATES = {"gi": rw.settings.GI(enabled=True, max_probes=64, rays=32, bake_updates=4, update_every=1)}


def _render(scene, **settings) -> np.ndarray:
  """The HDR images of two frames of the scene with the light grid on and `settings` (the
  second one after a periodic GI update when they have them)."""
  s = scene(
    _XML,
    cam_res=(24, 16),
    envmap=_SKY,
    outputs=rw.Outputs.of("hdr"),
    lighting=rw.settings.Lights(light_grid=True),
    **settings,
  )
  first = s.render().hdr(0)
  return np.stack([first, s.render().hdr(0)])


def _traversal(scene) -> str:
  """The traversal `_render`'s renderers resolve to (OptiX on an RT-core GPU)."""
  return scene(_XML, cam_res=(8, 8), envmap=_SKY, outputs=rw.Outputs.of("hdr")).r.features.traversal


def _forbid(monkeypatch: pytest.MonkeyPatch, *, bvh: bool) -> None:
  """Makes every bake computation raise (a warm cache must not run one); with `bvh`, also the
  software BVH builds of the bake rays (the keys are looked up before them: a warm start of a
  frozen bake builds none)."""

  def boom(*_args: object, **_kw: object) -> None:
    """A recomputation."""
    raise AssertionError("a cached bake was recomputed")

  monkeypatch.setattr(grid, "build", boom)
  monkeypatch.setattr(gi_lib.ProbeVolume, "place", boom)
  monkeypatch.setattr(gi_lib.Schedule, "bake", boom)
  monkeypatch.setattr(envmap, "_launch_prefilter", boom)
  monkeypatch.setattr(latlong, "extract_sun", boom)
  if bvh:
    monkeypatch.setattr(SoftwareBackend, "software_bvh", boom)
    monkeypatch.setattr(SoftwareBackend, "software_meshes", boom)


@pytest.mark.parametrize("settings", [_GI, _GI_UPDATES], ids=["gi", "gi_updates"])
def test_warm_cache_is_bit_identical_and_skips_the_bakes(scene, monkeypatch, tmp_path, settings):
  """Uncached, cold (fills the cache) and warm (loads it, no bake runs) renders are identical."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")
  uncached = _render(scene, **settings)
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  cold = _render(scene, **settings)
  kinds = sorted(p.parent.name for p in tmp_path.rglob("*.npz"))
  # OptiX builds also store their static mesh GASes (`trace/optix/gas.py`, relocated on a warm
  # start).
  gas = ["gas"] if _traversal(scene) == "optix" else []
  # The MuJoCo maps' sun search (`pipeline.upload.environment.cut_suns`; this sky has none).
  assert kinds == sorted(["env_sun", "envmap", "light_grid", "gi" if "gi" in settings else "sky", *gas])
  with monkeypatch.context() as m:
    # Periodic GI updates trace the scene after the bake: they build the BVH.
    _forbid(m, bvh=settings is not _GI_UPDATES)
    warm = _render(scene, **settings)
  np.testing.assert_array_equal(cold, uncached)
  np.testing.assert_array_equal(warm, uncached)


def test_changed_input_misses(scene, monkeypatch, tmp_path):
  """A different light intensity is a different light grid / GI key (a new entry, no stale hit)."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  _render(scene, **_GI)
  before = {p.name for p in tmp_path.rglob("*.npz")}
  _render(scene, **_GI, **{"exposure.mode": "manual", "exposure.ev100": 9.0})
  after = {p.name for p in tmp_path.rglob("*.npz")}
  new = after - before
  assert len(new) == 1  # the GI volume (its scale is the exposure); grid and envmap reused


def test_background_only_map_is_not_prefiltered(scene, monkeypatch):
  """A MuJoCo skybox (background only, B56) uploads its radiance without the prefilter."""
  xml = _XML.replace(
    "<asset>", '<asset><texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="32" height="32"/>'
  )

  def boom(*_args: object, **_kw: object) -> None:
    """A prefilter."""
    raise AssertionError("background-only map prefiltered")

  monkeypatch.setattr(envmap, "_launch_prefilter", boom)
  s = scene(xml, cam_res=(16, 16), outputs=rw.Outputs.of("hdr"))
  assert s.scene.environment.has_maps and not s.scene.environment.lit
  s.render()
  assert np.isfinite(s.hdr(0)).all()
  with pytest.raises(ValueError, match="background map"):
    scene(xml, cam_res=(16, 16), environment=rw.settings.Environment(enabled=True))


def _count_compiles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """The names of the Warp modules compiled from here on (Warp's kernel cache missed)."""
  import warp._src.context as warp_context

  compiled: list[str] = []
  original = warp_context.Module._compile

  def counted(self: object, *args: object, **kwargs: object) -> object:
    """Warp's compile, recorded when it compiled (False: the cached binary)."""
    done = original(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]
    if done:
      compiled.append(str(getattr(self, "name", "?")))
    return done

  monkeypatch.setattr(warp_context.Module, "_compile", counted)
  return compiled


def _mjcf_scene(xml: str, nworld: int) -> tuple[rw.GpuScene, object]:
  """(GpuScene, MuJoCo Warp data of `nworld` worlds) of an MJCF string on the CPU."""
  import mujoco
  import mujoco_warp as mjw

  import raywarp.integrations.mujoco

  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with wp.ScopedDevice("cpu"):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    return raywarp.integrations.mujoco.scene(mjm, m, d), d


def test_warm_leaves_a_renderer_nothing_to_compile_or_bake(monkeypatch, tmp_path):
  """`Renderer.warm` with the constructor's arguments prepares that renderer: a second warm and a
  renderer constructed after it compile no Warp module and run no bake; the second warm's frame
  adds no cache entry."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  gpu, d = _mjcf_scene(_XML, 2)
  settings = recipe("fast").replace(lighting=rw.settings.Lights(light_grid=True), **_GI)
  config = {"nworld": 2, "resolution": (24, 16), "settings": settings}
  warmed = rw.Renderer.warm(gpu, sim=d, **config).result()
  assert [p.name for p in warmed.parts] == ["renderer", "frame"]
  assert {p.parent.name for p in tmp_path.rglob("*.npz")} >= {"light_grid", "gi"}
  compiled = _count_compiles(monkeypatch)
  with monkeypatch.context() as m:
    _forbid(m, bvh=True)
    again = rw.Renderer.warm(gpu, sim=d, **config).result()
    rw.Renderer(gpu, **config).render(d)
  assert compiled == []
  assert again.parts[1].added == 0, again  # the frame: every bake a cache hit


def test_concurrent_warms_share_their_work(monkeypatch, tmp_path):
  """Warms started together: an identical request returns the running warm's future, the others
  (other settings, another scene) finish without races, and no temporary file is left."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  gpu, d = _mjcf_scene(_XML, 2)
  other, d1 = _mjcf_scene(_XML.replace('rgba="0.8 0.6 0.4 1"', 'rgba="0.2 0.6 0.4 1"'), 1)
  fast = recipe("fast")
  gi = fast.replace(**_GI)  # a GI bake beside the plain build
  futures = [
    rw.Renderer.warm(gpu, sim=d, nworld=2, resolution=(16, 16), settings=fast),
    rw.Renderer.warm(gpu, sim=d, nworld=2, resolution=(16, 16), settings=gi),
    rw.Renderer.warm(gpu, sim=d, nworld=2, resolution=(16, 16), settings=fast),
    rw.Renderer.warm(other, sim=d1, nworld=1, resolution=(16, 16), settings=gi),
  ]
  assert futures[2] is futures[0]
  assert all(len(f.result().parts) == 2 for f in futures)
  assert not [p for p in tmp_path.rglob("*") if ".tmp" in p.name or p.suffix == ".partial"]


def test_concurrent_saves_of_one_bake_never_collide(monkeypatch, tmp_path):
  """Threads storing the same bake at once (CPU workers rendering one scene) each write their own
  temporary file: every save succeeds, the stored file is complete, no temporary file is left."""
  import threading

  from raywarp.core.pipeline.platform import bake_cache

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  key = bake_cache.key("test", wp.get_device("cpu"), host=("concurrent",))
  arrays = {"a": np.arange(1 << 18).astype(np.float32)}
  nthread, nsave = 8, 16
  barrier = threading.Barrier(nthread)
  errors: list[BaseException] = []

  def worker() -> None:
    """Saves the bake `nsave` times right after the other threads start."""
    barrier.wait()
    try:
      for _ in range(nsave):
        bake_cache.save(key, arrays)
    except Exception as e:  # noqa: BLE001  # reported by the assertion below
      errors.append(e)

  threads = [threading.Thread(target=worker) for _ in range(nthread)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert not errors, errors[:1]
  stored = bake_cache.load(key)
  assert stored is not None
  np.testing.assert_array_equal(stored["a"], arrays["a"])
  # One bake file and no temporary left (the cache directory also holds the code-version memo).
  bakes = [p.name for p in tmp_path.rglob("*") if p.is_file() and not p.name.startswith("code-digests")]
  assert bakes == [f"{required(key).hexdigest()}.npz"]


def test_bake_keys_ignore_struct_padding():
  """Struct records equal field by field hash alike whatever their padding bytes hold (a kernel
  storing a local struct, e.g. the packed lights, copies its stack there): the light-grid key
  missed at random before."""

  from raywarp.core.pipeline.platform import bake_cache
  from raywarp.core.shade.lights.punctual import Light

  records = wp.zeros(3, dtype=Light, device="cpu").numpy()
  record_field(records, "active")[...] = True
  assert records.dtype.itemsize > sum(records.dtype.fields[n][0].itemsize for n in records.dtype.names)  # pyrefly: ignore[missing-attribute]  # padded; stub gap NS6
  garbage = records.copy()
  raw = garbage.view(np.uint8).reshape(3, -1)
  padding = np.ones(records.dtype.itemsize, bool)  # pyrefly: ignore[missing-attribute]  # stub gap NS6
  for name in records.dtype.names:  # pyrefly: ignore[missing-attribute]  # stub gap NS6
    field, offset = records.dtype.fields[name][:2]  # pyrefly: ignore[missing-attribute]  # stub gap NS6
    padding[offset : offset + field.itemsize] = False
  raw[:, padding] = 0xAB

  def digest(a: np.ndarray) -> str:
    return bake_cache.Digest("t", None).array(wp.array(a, dtype=Light, device="cpu")).hexdigest()

  assert digest(garbage) == digest(records)
  record_field(records, "intensity")[1] = 2.0
  assert digest(records) != digest(garbage)


def test_bake_keys_hash_values_not_addresses(monkeypatch, tmp_path):
  """A bake key hashes Warp matrices by their entries (their repr is an object address: two
  different matrices could share a key, equal ones never did), refuses a value with no value repr
  (no key: caching off for that bake) and changes with the Warp version (its code generation)."""

  from raywarp.core.pipeline.platform import bake_cache

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  cpu = wp.get_device("cpu")

  def digest(*host: object) -> str | None:
    """The key of a bake reading only `host`."""
    key = bake_cache.key("test", cpu, host=host)
    return None if key is None else key.hexdigest()

  one, other = wp.mat33(*range(9)), wp.mat33(*range(1, 10))
  assert digest(one) == digest(wp.mat33(*range(9)))
  assert digest(one) != digest(other)
  with pytest.warns(RuntimeWarning, match="the test bake is not cached"):
    assert digest(object()) is None
  # Arrays inside a tuple are hashed by content (a large array's repr elides its middle).
  a, b = np.zeros(4096, np.float32), np.zeros(4096, np.float32)
  b[2048] = 1.0
  assert digest((a, 1)) != digest((b, 1))
  assert digest((one,)) != digest((other,))
  before = digest(1.0)
  monkeypatch.setattr(wp.config, "version", "0.0.0")
  assert digest(1.0) != before


def test_bake_keys_refuse_functions_and_hash_containers_by_value(monkeypatch, tmp_path):
  """A function's repr is its address too (`<function f at 0x...>`: no key; it was hashed); a
  dataclass is hashed field by field (its repr elides a large array's middle: two values shared a
  key); a dict item by item, in an order independent of insertion (equal dicts share a key)."""
  import dataclasses

  from raywarp.core.pipeline.platform import bake_cache

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))

  def digest(*host: object) -> str | None:
    """The key of a bake reading only `host`."""
    key = bake_cache.key("test", wp.get_device("cpu"), host=host)
    return None if key is None else key.hexdigest()

  def shade() -> None:
    """A function (no value repr)."""

  with pytest.warns(RuntimeWarning, match="the test bake is not cached"):
    assert digest(shade) is None
    assert digest(lambda: 0) is None

  @dataclasses.dataclass(frozen=True)
  class Holder:
    values: np.ndarray

  a, b = np.zeros(4096, np.float32), np.zeros(4096, np.float32)
  b[2048] = 1.0
  assert digest(Holder(a)) != digest(Holder(b))
  assert digest({"v": a}) != digest({"v": b})
  assert digest({"x": 1, "y": 2}) == digest({"y": 2, "x": 1})
  assert digest({"x": 1, "y": 2}) != digest({"x": 2, "y": 1})


def test_empty_struct_arrays_have_a_key():
  """An empty array of a Warp struct is hashed by its type's name (its repr is an address: the light
  grid's key, whose world-0 light struct holds an empty hot-record table, had none, so the grid was
  traced again on every start)."""

  from raywarp.core.pipeline.platform import bake_cache
  from raywarp.core.shade.lights.pack import LightHot

  empty = wp.zeros((1, 0), dtype=LightHot, device="cpu")
  assert (
    bake_cache.Digest("x", None).array(empty).hexdigest()
    == bake_cache.Digest("x", None).array(wp.zeros((1, 0), dtype=LightHot, device="cpu")).hexdigest()
  )


def test_bake_keys_of_sets_are_the_same_in_every_process():
  """A set's repr follows the per-process string hash (PYTHONHASHSEED): its key did too, so a
  warm cache missed in the next process. Sets are hashed in a fixed order."""
  import os
  import subprocess
  import sys

  code = (
    "import warp as wp\n"
    "from raywarp.core.pipeline.platform import bake_cache\n"
    "d = bake_cache.Digest('test', wp.get_device('cpu'))\n"
    "print(d.add(frozenset({'rgb', 'depth', 'seg', 'hdr', 'albedo', 'normal'})).hexdigest())\n"
  )
  out = set()
  for seed in ("1", "2", "3", "4"):
    env = {**os.environ, "PYTHONHASHSEED": seed}
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    out.add(run.stdout.strip().splitlines()[-1])
  assert len(out) == 1, out


def test_probe_geometry_builds_the_mesh_blases_only_for_software_traversal(scene):
  """`Baked._probe_geometry(traverse=False)` (hardware GI rays, no relight kernel: they read the
  geometry's colors and the scene BVH's boxes only) builds no software mesh BLAS; `traverse`
  builds them."""
  s = scene(_XML, cam_res=(8, 8), envmap=_SKY, outputs=rw.Outputs.of("hdr"), **_GI)
  s.render()
  r = s.r
  calls: list[str] = []

  class Spy:
    """A backend recording the software structure builds."""

    def software_bvh(self, ctx: object, frame: object) -> None:
      calls.append("bvh")

    def software_meshes(self, ctx: object) -> None:
      calls.append("meshes")

  ctx, frame = r.context(), r._frame(s.d)
  r._baked._probe_geometry(ctx, frame, Spy(), traverse=False, refit=False)
  assert "meshes" not in calls
  r._baked._probe_geometry(ctx, frame, Spy(), traverse=True)
  assert calls[-2:] == ["bvh", "meshes"]
