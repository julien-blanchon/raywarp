"""The bake caches' keys (`core/pipeline/bake_keys.py`): what the light grid and the sky visibility
read, and nothing else. N2: two meshes differing only in their vertices shared one entry (the
geometry struct holds BLAS ids, not vertices). N3: the key hashed every world's rows although only
world 0 is traced, so a run with another world count missed a warm cache."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp._base.warp_structs import warp_struct
from raywarp.core.pipeline.platform import bake_cache


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], list[str]]:
  """The bake cache events of a kind ("miss", "hit", "save"), in order: a second run that is not
  cached at all (no key) or bakes again also leaves one file, so the tests check that it read the
  first run's bake and stored nothing."""
  seen: list[tuple[str, str]] = []
  load, save = bake_cache.load, bake_cache.save

  def loaded(key: bake_cache.Digest | None) -> bake_cache.Bake | None:
    stored = load(key)
    if key is not None:
      seen.append((key.kind, "miss" if stored is None else "hit"))
    return stored

  def saved(key: bake_cache.Digest | None, arrays: Mapping[str, np.ndarray]) -> None:
    if key is not None:
      seen.append((key.kind, "save"))
    save(key, arrays)

  monkeypatch.setattr(bake_cache, "load", loaded)
  monkeypatch.setattr(bake_cache, "save", saved)
  return lambda kind: [event for k, event in seen if k == kind]


def _scene(turned: bool) -> rw.Scene:
  """A floor, a two-sided square at z = 1 (turned 45 degrees about z: same vertex count, faces,
  UVs and bounding radius), a point light above it, a constant sky."""
  b = rw.SceneBuilder()
  quad = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
  if turned:
    c = np.sqrt(0.5)
    quad = np.array([[0.0, -c, 0.0], [c, 0.0, 0.0], [0.0, c, 0.0], [-c, 0.0, 0.0]])
  tris = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
  mat = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  b.add_instance("plane", material=mat, size=(3.0, 3.0, 0.0))
  b.add_instance("mesh", mesh=b.add_mesh(quad, tris), material=mat, xform=np.c_[np.eye(3), [0.0, 0.0, 1.0]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=1.0e4, radius=0.01, xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]]))
  b.add_environment(rw.scene.EnvironmentLight(intensity=0.5))
  return b.build()


def _bake(host: rw.Scene, nworld: int, **settings: object) -> None:
  """One render of `host` (software traversal, light grid on, `settings`), which bakes."""
  with wp.ScopedDevice("cpu"):
    s = rw.RenderSettings().replace(
      **{
        "outputs": rw.Outputs.of("hdr"),
        "environment.enabled": False,
        "lighting.light_grid": True,
        "lighting.light_grid_cell": 0.25,
        "performance.traversal": "software",
        **settings,
      }
    )
    r = rw.Renderer(rw.upload(host), nworld=nworld, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(8, 8), settings=s)
    r.camera_pose.assign(np.array([[[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0]]] * nworld, np.float32))
    r.render()


@pytest.mark.parametrize(
  ("kind", "settings"),
  [
    ("light_grid", {}),
    ("gi", {"environment.enabled": True, "gi.enabled": True, "gi.max_probes": 32, "gi.rays": 16, "gi.bake_updates": 2}),
  ],
)
def test_keys_hash_mesh_vertices(kind, settings, tmp_path, monkeypatch):
  """N2: a mesh turned in place is another bake."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  _bake(_scene(False), 1, **settings)
  _bake(_scene(True), 1, **settings)
  assert len(list((tmp_path / "bakes" / kind).glob("*.npz"))) == 2


def test_grid_key_ignores_the_world_count(tmp_path, monkeypatch, lookups):
  """N3: the grid of world 0 is one bake whatever the world count."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  _bake(_scene(False), 1)
  _bake(_scene(False), 3)
  assert len(list((tmp_path / "bakes" / "light_grid").glob("*.npz"))) == 1
  assert lookups("light_grid") == ["miss", "save", "hit"]


# Every per-world declaration a bake struct reads: per-world poses (no shared scenery), material
# and environment tables.
_EVERY_FIELD = rw.PerWorld(
  pose=["*"],
  material=[
    "base_color",
    "emission_luminance",
    "mirror",
    "base_metalness",
    "specular_roughness",
    "specular_weight",
    "tex_affine",
    "tex_variation",
    "tex_repeat",
  ],
  env=["intensity", "yaw", "exposure", "ambient"],
)
_ROOM = """
<mujoco>
  <visual><headlight active="0"/></visual>
  <asset><material name="mat" rgba="0.8 0.6 0.4 1" roughness="0.5"/></asset>
  <worldbody>
    <light pos="0 0 0.8" diffuse="1 1 1" intensity="20"/>
    <geom type="plane" size="3 3 0.1" material="mat"/>
    <geom type="box" pos="0 0 1.0" size="0.6 0.6 0.025" material="mat"/>
    <geom type="sphere" pos="-0.2 0.1 0.2" size="0.2" material="mat"/>
    <camera name="cam" pos="-1.8 -1.2 1.2" xyaxes="0.55 -0.83 0 0.3 0.2 0.93" fovy="60"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(
  ("kinds", "settings"),
  [
    (("light_grid",), {"gi": rw.settings.GI(enabled=True, max_probes=32, rays=16, bake_updates=2)}),
  ],
  ids=["gi"],
)
def test_keys_ignore_the_world_count_with_per_world_tables(scene, kinds, settings, tmp_path, monkeypatch, lookups):
  """N3 for every per-world table: the material / environment tables, the per-world BVH roots and
  the light-cost counters were hashed whole, so a declared field made every bake miss in a run
  with another world count."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  sky = np.full((8, 16, 3), 0.5, np.float32)
  for nworld in (1, 3):
    s = scene(
      _ROOM,
      nworld=nworld,
      cam_res=(8, 8),
      envmap=sky,
      per_world=_EVERY_FIELD,
      outputs=rw.Outputs.of("hdr"),
      lighting=rw.settings.Lights(light_grid=True),
      **settings,
    )
    s.render()
  for kind in kinds:
    assert len(list((tmp_path / "bakes" / kind).glob("*.npz"))) == 1, kind
    assert lookups(kind) == ["miss", "save", "hit"], kind  # the 3-world run read the bake


def test_key_row_names_are_struct_fields():
  """The names `bake_keys` cuts to world 0 or leaves out of keys are fields of the hashed structs
  (a renamed field would silently be hashed whole again)."""
  from raywarp.core.kernels.abi import rays as trace
  from raywarp.core.kernels.abi import structs
  from raywarp.core.pipeline import bake_keys
  from raywarp.core.shade.lights import punctual
  from raywarp.core.shade.materials import material

  names = set()
  for struct in (
    trace.SoftwareGeometry,
    trace.Cutout,
    punctual.Lights,
    material.Materials,
    structs.EnvironmentArrays,
    structs.Params,
  ):
    names |= set(warp_struct(struct).vars)
  assert bake_keys.UNHASHED <= names
  # The derived per-world rows (not Frame fields) by their struct names.
  assert {"lights", "ambient_radiance", "instance_rot_trace"} <= bake_keys.WORLD_ROWS & names


def test_gi_key_ignores_the_user_ambient(tmp_path, monkeypatch, lookups):
  """`lighting.fill` is added at camera and reflection hits only: another value reuses the GI
  bake (it is a runtime fill, never a re-bake)."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  gi = {"gi.enabled": True, "gi.max_probes": 64, "gi.rays": 16, "gi.bake_updates": 2}
  _bake(_scene(False), 1, **gi)
  _bake(_scene(False), 1, **gi, **{"lighting.fill": 0.08, "lighting.fill_color": (1.0, 0.5, 0.2)})
  assert len(list((tmp_path / "bakes" / "gi").glob("*.npz"))) == 1
  assert lookups("gi") == ["miss", "save", "hit"]


def test_registered_arrays_key_by_content_digest() -> None:
  """A device array registered with its content digest (`register_array`) is keyed by it, without
  a host copy: equal contents give equal keys, a changed content a different one, and an
  unregistered array (or one whose registration died with it) is hashed by content."""
  from raywarp.core.pipeline.platform import bake_cache

  host = np.arange(12).astype(np.float32).reshape(4, 3)
  changed = host.copy()
  changed[2, 1] += 1.0

  def key(a: object) -> str:
    return bake_cache.Digest("t", None).add(a).hexdigest()

  def registered(values: np.ndarray) -> wp.array:
    return bake_cache.register_array(wp.array(values, dtype=wp.vec3), bake_cache.array_digest(values))

  a, b, c = registered(host), registered(host), registered(changed)
  assert key(a) == key(b) != key(c)
  plain = wp.array(host, dtype=wp.vec3)
  assert key(plain) == key(host) != key(a)  # unregistered: by content, as a host array
  assert bake_cache._registered(a) == bake_cache.array_digest(host)
  del a
  assert len([v for v in bake_cache._arrays.values() if v[0]() is None]) == 0  # forgotten


def test_upload_registers_the_mesh_tables() -> None:
  """The upload registers its mesh tables with the content digests of their host copies."""
  from raywarp.core.pipeline.platform import bake_cache

  gpu = rw.upload(_scene(turned=False))
  m = gpu.meshes
  for table in (m.vert, m.face, m.normal, m.texcoord):
    assert bake_cache._registered(table) == bake_cache.array_digest(table.numpy())


def test_probe_key_follows_the_exposure_of_display_relative_emission() -> None:
  """The GI volume is re-baked for a new exposure (`probe_key`) in a scene of display-relative
  emission (the probes light it at the renderer's exposure), and not with physical emission."""
  import dataclasses

  from raywarp.core.pipeline import bake_keys
  from raywarp.core.trace.context import FrameContext

  settings = rw.RenderSettings().replace(**{"gi.enabled": True, "exposure.mode": "manual"})
  with (
    wp.ScopedDevice("cpu"),
    rw.Renderer(rw.upload(_scene(False)), nworld=1, cameras=[rw.CameraSpec()], resolution=(8, 8), settings=settings) as r,
  ):
    ctx = r.context()
  other = dataclasses.replace(ctx, settings=ctx.settings.replace(**{"exposure.ev100": ctx.settings.exposure.ev100 + 2.0}))
  assert ctx.scene.capabilities.physical_emission
  assert bake_keys.probe_key(other) == bake_keys.probe_key(ctx)

  def relative(c: FrameContext[None]) -> FrameContext[None]:
    caps = dataclasses.replace(c.scene.capabilities, physical_emission=False)
    return dataclasses.replace(c, scene=dataclasses.replace(c.scene, capabilities=caps))

  assert bake_keys.probe_key(relative(other)) != bake_keys.probe_key(relative(ctx))
