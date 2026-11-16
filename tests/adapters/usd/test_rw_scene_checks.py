"""SceneBuilder boundary checks, decode thread / memory bounds, int64 blob offsets."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np
import pytest

import raywarp as rw
from raywarp.scene.images import blob, mips
from raywarp.scene.images import decode as imaging
from raywarp.scene.model import checks
from raywarp.scene.model.material import Material, OpenPbrExtras, RaywarpExt
from raywarp.scene.model.types import (
  AlphaMode,
  Camera,
  ImageId,
  MaterialFlags,
  MaterialId,
  PixelType,
  Slot,
  TextureId,
  TextureRef,
)

TRI = np.array([[0, 1, 2]])
POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)


# Threads -------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [0, -1, -8])
def test_threads_below_one_are_rejected(threads):
  """0 is not "the CPU count" (it used to be, while negatives failed later): None is."""
  with pytest.raises(ValueError, match="at least 1"):
    blob.worker_count(threads)
  b = rw.SceneBuilder()
  b.add_texture(np.zeros((2, 2, 4), np.uint8))
  with pytest.raises(ValueError, match="at least 1"):
    b.build(threads=threads)


def test_threads_none_is_the_cpu_count():
  """None means the default (the CPU count, at most MAX_THREADS); a positive count is kept."""
  assert 1 <= blob.worker_count(None) <= blob.MAX_THREADS
  assert blob.worker_count(3) == 3


# Byte budget ---------------------------------------------------------------------------------


def test_byte_budget_bounds_concurrent_holders():
  """Holders never hold more than the budget at once; an oversized request runs alone."""
  budget = blob.ByteBudget(100)
  held = [0]
  most = [0]
  lock = threading.Lock()

  def work(n: int) -> None:
    with budget.hold(n):
      with lock:
        held[0] += min(n, 100)
        most[0] = max(most[0], held[0])
      time.sleep(0.002)
      with lock:
        held[0] -= min(n, 100)

  threads = [threading.Thread(target=work, args=(n,)) for n in [30, 40, 50, 60, 250, 10] * 4]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert most[0] <= 100 and budget.peak <= 100
  assert not budget.try_acquire(101) and budget.try_acquire(100)


def test_byte_budget_rejects_an_empty_budget():
  """A budget of 0 bytes could never run anything."""
  with pytest.raises(ValueError, match="at least 1 byte"):
    blob.ByteBudget(0)


def _mip_decode_peak(nbytes: int | None, threads: int) -> int:
  """Peak traced memory [bytes] of decoding 16 512 x 512 RGBA8 arrays with mips into a no-op sink
  under a `nbytes` budget (None: the default, large)."""
  rng = np.random.default_rng(0)
  arrays = [rng.integers(0, 255, (512, 512, 4), dtype=np.uint8) for _ in range(16)]
  sources = [imaging.ImageSource(array=a) for a in arrays]
  probed = [blob.Probed(info=imaging.probe(s)) for s in sources]
  levels = [mips.mip_count(512, 512)] * len(sources)
  budget = None if nbytes is None else blob.ByteBudget(nbytes)
  tracemalloc.start()
  tracemalloc.reset_peak()
  base = tracemalloc.get_traced_memory()[0]
  blob.decode_all(sources, probed, levels, sink=lambda _i, _levels: None, threads=threads, budget=budget)
  peak = tracemalloc.get_traced_memory()[1] - base
  tracemalloc.stop()
  return peak


def test_decode_memory_is_bounded_by_the_budget():
  """Workers building mip chains hold at most about the budget (the estimate covers them):
  without a budget 8 workers hold several images' working sets at once."""
  one = blob.decode_bytes(imaging.probe(imaging.ImageSource(array=np.zeros((512, 512, 4), np.uint8))), 10)
  bounded = _mip_decode_peak(one, threads=8)
  unbounded = _mip_decode_peak(None, threads=8)
  assert bounded <= one, (bounded, one)
  assert unbounded > 2 * one, (unbounded, one)


# Peak RSS above the baseline while one image decodes, sampled every millisecond (ru_maxrss would
# include the import-time peak), and its reserved bytes.
_PEAK_SCRIPT = """
import sys, threading, time
from raywarp.scene.images import blob
from raywarp.scene.images import decode as imaging
def rss():
  with open("/proc/self/statm") as f:
    return int(f.read().split()[1]) * 4096
source = imaging.ImageSource(path=sys.argv[1], max_size=int(sys.argv[2]))
info = imaging.probe(source)
warm = imaging.ImageSource(path=sys.argv[3])  # loads the decoders (their code is not the budget)
imaging.decode_base(warm, imaging.probe(warm))
base, peak, done = rss(), [0], threading.Event()
def sample():
  while not done.is_set():
    peak[0] = max(peak[0], rss())
    time.sleep(0.001)
thread = threading.Thread(target=sample)
thread.start()
blob.decode_all([source], [blob.Probed(info=info)], [1], sink=lambda _i, _l: None, threads=1)
done.set()
thread.join()
print(peak[0] - base, blob.decode_bytes(info, 1))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/statm")
def test_decode_budget_covers_downscaled_images(tmp_path):
  """A downscaled image decodes at its source resolution (Pillow's full-size output, conversions):
  its reserved bytes cover the measured peak (they used to be the stored size: 2 MB for a 4096^2
  PNG peaking at 200 MB)."""
  from PIL import Image

  path = tmp_path / "big.png"
  rng = np.random.default_rng(3)
  Image.fromarray(rng.integers(0, 255, (2048, 2048, 4), dtype=np.uint8)).save(path)
  Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(tmp_path / "warm.png")
  argv = [sys.executable, "-c", _PEAK_SCRIPT, str(path), "128", str(tmp_path / "warm.png")]
  out = subprocess.run(argv, capture_output=True, text=True, check=True)
  peak, estimate = (int(v) for v in out.stdout.split())
  stored = 128 * 128 * 4
  assert estimate > 100 * stored  # from the 2048^2 source, not the 128^2 result
  assert peak <= estimate, (peak, estimate)


def test_ahead_budget_covers_decodes_in_flight(tmp_path, monkeypatch):
  """`Ahead` reserves a decode's working memory, not only the retained top level: with room for
  one decode only, at most one image is decoded ahead, the rest at build (same pixels)."""
  from rw_helpers import write_png

  files = [write_png(tmp_path / f"{i}.png", np.full((32, 32, 3), 10 * i, np.uint8)) for i in range(6)]
  info = imaging.probe(imaging.ImageSource(path=str(files[0])))
  monkeypatch.setattr(blob, "AHEAD_BYTES", blob.decode_bytes(info, 1))
  b = rw.SceneBuilder()
  for f in files:
    b.add_texture(f)
  results = b._images._ahead.results()
  assert sum(p.base is not None for p in results) <= 1
  assert b._images._ahead.budget.peak <= blob.decode_bytes(info, 1)
  scene = b.build()
  assert b._images._ahead.budget.held == 0  # the kept top levels left the budget with them
  for i in range(6):
    assert int(scene.images.level(i)[0, 0, 0]) == 10 * i


# int64 offsets -------------------------------------------------------------------------------


def test_blob_offsets_beyond_2gib_are_exact():
  """Image offsets past 2^31 bytes are stored exactly (int64), not wrapped."""
  big = imaging.ImageInfo(width=16384, height=16384, channels=4, pixel=PixelType.F32)  # 4 GiB
  small = imaging.ImageInfo(width=4, height=4, channels=4, pixel=PixelType.U8)
  layout = blob.BlobLayout.of([big, small, small], mips=False)
  table = layout.table([big, small, small], ["big", "a", "b"])
  assert table.offset.dtype == np.int64
  assert int(table.offset[1]) == 16384 * 16384 * 16 > 2**31
  assert int(table.offset[2]) == int(table.offset[1]) + 256
  assert layout.total == int(table.offset[2]) + 256


def test_mesh_pools_beyond_int32_addresses_fail(monkeypatch):
  """Pool rows past the int32 addresses of `Meshes` fail at build instead of wrapping (the limit
  lowered to 2 rows here)."""
  from raywarp.scene.build import mesh_pool

  monkeypatch.setattr(mesh_pool, "_INT32_MAX", 2)
  b = rw.SceneBuilder()
  b.add_mesh(POINTS, TRI)
  with pytest.raises(ValueError, match="beyond int32"):
    b.build()


def test_scene_keeps_the_meta_dict_it_is_given():
  """An empty `meta` (or report) given to `Scene.from_arrays` is kept, not replaced (falsy)."""
  scene = rw.SceneBuilder().build()
  meta: dict = {}
  report = rw.scene.LoadReport()
  again = rw.Scene.from_arrays(scene.arrays(), report, meta)
  assert again.meta is meta and again.report is report


# Builder boundary checks -----------------------------------------------------------------------


def test_bad_lights_fail_at_add_light():
  """Physical light values are checked where they are given."""
  b = rw.SceneBuilder()
  cases = [
    ({"color": (math.nan, 1.0, 1.0)}, "color must be finite"),
    ({"color": (-1.0, 1.0, 1.0)}, "non-negative"),
    ({"color": (1.0, 1.0)}, "shape"),
    ({"intensity": -1.0}, "intensity"),
    ({"exposure": math.inf}, "exposure"),
    ({"color_temperature": 0.0}, "color_temperature"),
    ({"radius": -0.1}, "radius"),
    ({"shaping": rw.scene.Shaping(cone_angle=200.0)}, "cone_angle"),
    ({"shaping": rw.scene.Shaping(cone_softness=1.5)}, "cone_softness"),
    ({"xform": np.full((3, 4), math.inf)}, "xform"),
    ({"xform": np.c_[2.0 * np.eye(3), np.zeros(3)]}, "orthonormal"),
    ({"texture": 0}, "texture 0 is not one of the 0"),
    ({"meters_per_unit": 0.0}, "meters_per_unit"),
    ({"ext": rw.scene.LightExt(range=0.0)}, "ext.range"),
  ]
  for kwargs, message in cases:
    with pytest.raises(ValueError, match=message):
      b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, **kwargs))  # pyrefly: ignore[bad-argument-type] (bad values on purpose)
  with pytest.raises(ValueError, match="positive width and height"):
    b.add_light(rw.Light(kind=rw.scene.LightKind.RECT, intensity=1.0, width=0.0, height=1.0))
  with pytest.raises(ValueError, match="not a LightKind"):
    b.add_light(rw.Light(kind=42))  # pyrefly: ignore[bad-argument-type] (bad value on purpose)
  assert (
    b.add_light(
      rw.Light(
        kind=rw.scene.LightKind.SPHERE, intensity=1.0, radius=0.0, shaping=rw.scene.Shaping(cone_angle=math.degrees(0.5))
      )
    )
    == 0
  )


def test_bad_materials_fail_at_add_material():
  """Every OpenPBR input out of its range (canonical_design.md §1.5), bad extensions, the
  combinations the derivation relies on and unbound textures fail at add_material, naming the
  field."""
  b = rw.SceneBuilder()
  blend = RaywarpExt(alpha_mode=AlphaMode.BLEND)
  cases = [
    (Material(specular_roughness=1.5), "specular_roughness"),
    (Material(base_metalness=-0.1), "base_metalness"),
    (Material(base_weight=1.1), "base_weight"),
    (Material(geometry_opacity=math.nan, ext=blend), "geometry_opacity"),
    (Material(base_color=(1.0, math.inf, 1.0)), "base_color"),
    (Material(base_color=(1.2, 0.5, 0.5)), "base_color"),
    (Material(specular_color=(0.5, -0.1, 0.5)), "specular_color"),
    (Material(emission_color=(-1.0, 0.0, 0.0)), "emission_color"),
    (Material(emission_luminance=-1.0), "emission_luminance"),
    (Material(specular_weight=-0.5), "specular_weight"),
    (Material(specular_ior=0.0), "specular_ior"),
    (Material(coat_ior=-1.0), "coat_ior"),
    (Material(ext=RaywarpExt(alpha_mode=7)), "alpha_mode"),  # pyrefly: ignore[bad-argument-type]  # deliberately invalid
    (Material(ext=RaywarpExt(alpha_cutoff=0.0)), "alpha_cutoff"),
    (Material(ext=RaywarpExt(mirror=2.0)), "mirror"),
    (Material(ext=RaywarpExt(occlusion_strength=-0.5)), "occlusion_strength"),
    (Material(ext=RaywarpExt(flags=1 << 20)), "flags"),
    (Material(geometry_opacity=0.5), "BLEND or MASK"),
    (Material(ext=RaywarpExt(flags=MaterialFlags.SPECULAR_CURVE), specular_ior=1.4), "specular_ior 1.5"),
    (Material(coat_weight=0.5, ext=RaywarpExt(flags=MaterialFlags.COAT_GLTF)), "coat_ior 1.5"),
    (Material(extras=OpenPbrExtras(fuzz_weight=2.0)), "fuzz_weight"),
    (Material(extras=OpenPbrExtras(subsurface_scatter_anisotropy=-1.5)), "subsurface_scatter_anisotropy"),
    (Material(extras=OpenPbrExtras(thin_film_ior=0.0)), "thin_film_ior"),
    (Material(extras=OpenPbrExtras(subsurface_radius_scale=(1.0, -1.0, 1.0))), "subsurface_radius_scale"),
    (Material(extras=OpenPbrExtras(textures=(("base_color", TextureRef(TextureId(0))),))), "without a Slot"),
    (Material(extras=OpenPbrExtras(textures=(("nope", TextureRef(TextureId(0))),))), "without a Slot"),
    (Material().with_texture(Slot.BASE_COLOR, TextureRef(TextureId(3))), "texture 3"),
  ]
  for mat, message in cases:
    with pytest.raises(ValueError, match=message):
      b.add_material(mat)
  tex = b.add_texture(np.zeros((2, 2, 4), np.uint8))
  with pytest.raises(ValueError, match="channel"):
    b.add_material(Material().with_texture(Slot.BASE_METALNESS, TextureRef(tex, channel=5)))
  with pytest.raises(ValueError, match="BLEND or MASK"):
    b.add_material(Material().with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3)))
  with pytest.raises(ValueError, match="sorted by slot"):
    b.add_material(Material(textures=((Slot.SPECULAR_ROUGHNESS, TextureRef(tex)), (Slot.BASE_COLOR, TextureRef(tex)))))
  assert b.add_material(Material().with_texture(Slot.BASE_COLOR, TextureRef(tex))) == 0
  coated = Material(coat_weight=0.5, coat_ior=1.5, ext=RaywarpExt(flags=MaterialFlags.COAT_GLTF))
  extras = OpenPbrExtras(fuzz_weight=0.5, textures=(("geometry_coat_normal", TextureRef(tex)),))
  assert b.add_material(coated) == 1 and b.add_material(Material(extras=extras)) == 2


def test_bad_instances_fail_at_add_instance():
  """Mesh / material ids not added, NaN transforms and negative sizes fail at add_instance."""
  b = rw.SceneBuilder()
  mesh = b.add_mesh(POINTS, TRI)
  cases = [
    ({"kind": "mesh"}, "mesh -1 is not one of the 1"),
    ({"kind": "mesh", "mesh": 5}, "mesh 5"),
    ({"kind": "box", "material": 0}, "material 0"),
    ({"kind": "box", "size": (-1.0, 1.0, 1.0)}, "size"),
    ({"kind": "box", "xform": np.full((3, 4), math.nan)}, "xform"),
    ({"kind": "box", "color": (1.0, 1.0, 1.0)}, "color"),
    ({"kind": "box", "flags": 1 << 12}, "flags"),
    ({"kind": "blob"}, "not one of"),
  ]
  for kwargs, message in cases:
    with pytest.raises(ValueError, match=message):
      b.add_instance(**kwargs)  # pyrefly: ignore[bad-argument-type] (bad values on purpose)
  assert b.add_instance("mesh", mesh=mesh) == 0


def test_bad_meshes_fail_at_add_mesh():
  """Empty meshes and NaN points fail at add_mesh (empty parts stay allowed in add_mesh_parts)."""
  b = rw.SceneBuilder()
  with pytest.raises(ValueError, match="at least one triangle"):
    b.add_mesh(POINTS, np.zeros((0, 3), np.int32))
  with pytest.raises(ValueError, match="NaN"):
    b.add_mesh(np.full((3, 3), math.nan), TRI)
  with pytest.raises(ValueError, match="NaN"):
    b.add_mesh(POINTS, TRI, normals=np.full((3, 3), math.inf))
  with pytest.raises(ValueError, match="nparts"):
    b.add_mesh_parts(POINTS, TRI, nparts=0)
  assert b.add_mesh_parts(POINTS, TRI, tri_part=[1], nparts=2)[0] == -1


def test_finite_attributes_drops_broken_pools_for_adapters():
  """The adapters' helper drops a non-finite pool with its index (reported), keeps the rest: the
  mesh then builds (flat normals) where the raw attributes fail at add_mesh."""
  from raywarp.scene.build.mesh_pool import MeshAttributes, finite_attributes

  report = rw.scene.LoadReport()
  attrs = MeshAttributes(normals=np.full((3, 3), math.nan), uvs=np.zeros((3, 2)))
  kept = finite_attributes(attrs, report, "/M", source="newton.Mesh")
  assert kept.get("normals") is None and kept.get("normal_idx") is None and kept.get("uvs") is not None
  assert report.dropped["newton.Mesh.normals.non_finite"] == 1
  b = rw.SceneBuilder()
  with pytest.raises(ValueError, match="NaN"):
    b.add_mesh(POINTS, TRI, **attrs)
  assert b.add_mesh(POINTS, TRI, **kept) == 0


def test_bad_cameras_and_environments_fail_at_add():
  """Camera and environment values are checked where they are given."""
  b = rw.SceneBuilder()
  cam = Camera(xform=np.eye(4)[:3])
  for change, message in [
    ({"focal_length": 0.0}, "focal_length"),
    ({"vertical_aperture": -1.0}, "vertical_aperture"),
    ({"near": 1.0, "far": 0.5}, "far"),
    ({"xform": np.full((3, 4), math.nan)}, "xform"),
    ({"projection": 4}, "projection"),
    ({"fstop": -2.0}, "fstop"),
  ]:
    with pytest.raises(ValueError, match=message):
      b.add_camera("cam", dataclasses.replace(cam, **change))  # pyrefly: ignore[bad-argument-type] (bad values on purpose)
  assert b.add_camera("cam", cam) == 0
  with pytest.raises(ValueError, match="color"):
    b.add_environment(rw.scene.EnvironmentLight(color=(1.0, -1.0, 1.0)))
  with pytest.raises(ValueError, match="texture 2"):
    b.add_environment(rw.scene.EnvironmentLight(texture=TextureId(2)))
  with pytest.raises(ValueError, match="rotation"):
    b.add_environment(rw.scene.EnvironmentLight(rotation=np.full((3, 3), math.nan)))


def test_bad_textures_and_images_fail_at_add():
  """Unknown wraps / color spaces, size limits below 1 and unusable arrays fail at add_texture."""
  b = rw.SceneBuilder()
  img = np.zeros((2, 2, 4), np.uint8)
  with pytest.raises(ValueError, match="wrap"):
    b.add_texture(img, wrap=("repeat", "spiral"))
  with pytest.raises(ValueError, match="colorspace"):
    b.add_texture(img, colorspace="no_such_space")
  with pytest.raises(ValueError, match="outputs only"):
    b.add_texture(img, colorspace="srgb_rec709_display")
  with pytest.raises(ValueError, match="max_size"):
    b.add_texture(img, max_size=0)
  with pytest.raises(ValueError, match="image 4"):
    b.add_texture(ImageId(4))
  for bad, message in [
    (np.zeros((2, 2, 4), np.uint16), "uint8 or floating"),
    (np.full((2, 2, 3), math.nan, np.float32), "finite"),
    (np.zeros((2, 2, 5), np.uint8), "shape"),
    (np.zeros((0, 2, 4), np.uint8), "shape"),
  ]:
    with pytest.raises(ValueError, match=message):
      b.add_texture(bad)
  assert b.add_texture(img, wrap=("clamp", "mirror")) == 0


@pytest.mark.parametrize(
  ("value", "ok"), [(0.0, True), (1.0, True), (0.5, True), (-1e-9, False), (1.0 + 1e-9, False), (math.nan, False)]
)
def test_in_range_accepts_exactly_finite_values_in_range(value, ok):
  """`checks.in_range` accepts a value iff it is finite and in [low, high] (the hypothesis
  property test, `test_rw_scene_checks_props.py`, covers random values)."""
  if ok:
    assert checks.in_range("v", value, "test", low=0.0, high=1.0) == value
  else:
    with pytest.raises(ValueError, match="must be in"):
      checks.in_range("v", value, "test", low=0.0, high=1.0)


@pytest.mark.parametrize(
  ("kwargs", "message"),
  [
    ({"intensity": math.nan}, "intensity"),
    ({"intensity": -1.0}, "intensity"),
    ({"color": (1.0, -0.5, 1.0)}, "color"),
    ({"cone_angle": 200.0}, "cone_angle"),
    ({"radius": -0.1}, "radius"),
    ({"position": (0.0, math.inf, 0.0)}, "xform"),
  ],
)
def test_lighting_rejects_bad_lights_when_made(kwargs, message):
  """A `Lighting` light with a NaN / negative intensity, a bad cone, radius or position fails when
  the rig is made (not deep in an upload)."""
  args = {"position": (0.0, 0.0, 3.0), "direction": (0.0, 0.0, -1.0)} | kwargs
  with pytest.raises(ValueError, match=message):
    rw.Lighting([rw.Light.spot(**args)])
  shaping = rw.Lighting([rw.Light.spot((0.0, 0.0, 3.0), (0.0, 0.0, -1.0), cone_angle=30.0)]).lights[0].shaping
  assert shaping is not None and shaping.cone_angle == 30.0


@pytest.mark.parametrize("name", ["transmission_scatter_anisotropy", "subsurface_scatter_anisotropy"])
def test_scatter_anisotropies_take_openpbr_s_signed_range(name):
  """OpenPBR 1.1.1 (spec §subsurface / §transmission tables, the reference nodedef's uimin /
  uimax): a Henyey-Greenstein g in [-1, 1], back scattering included; the roughness
  anisotropies stay in [0, 1]."""
  b = rw.SceneBuilder()

  def extras(g: float) -> OpenPbrExtras:
    """The extras with anisotropy `name` at g."""
    if name == "transmission_scatter_anisotropy":
      return OpenPbrExtras(transmission_scatter_anisotropy=g)
    return OpenPbrExtras(subsurface_scatter_anisotropy=g)

  for g in (-1.0, -0.5, 1.0):
    b.add_material(Material(extras=extras(g)))
  for g in (-1.01, 1.01):
    with pytest.raises(ValueError, match=name):
      b.add_material(Material(extras=extras(g)))
  with pytest.raises(ValueError, match="specular_roughness_anisotropy"):
    b.add_material(Material(extras=OpenPbrExtras(specular_roughness_anisotropy=-0.5)))


def test_material_lookup_rejects_an_id_not_added():
  """`SceneBuilder.material` refuses NO_MATERIAL (-1) and ids past the table: a negative id must
  not silently name the last material."""
  b = rw.scene.SceneBuilder()
  m = b.add_material(Material(name="a"))
  assert b.material(m).name == "a"
  for bad in (-1, 1):
    with pytest.raises(ValueError, match="material"):
      b.material(MaterialId(bad))


def test_table_columns_reject_unknown_names():
  """`Table.from_columns` refuses a column or string column the table does not have (a stale name
  would otherwise leave that column empty)."""
  from raywarp.scene.model.tables import Textures

  Textures.from_columns({"image": [0], "colorspace": [0], "wrap": [[0, 0]]})
  with pytest.raises(KeyError, match="wraps"):
    Textures.from_columns({"image": [0], "colorspace": [0], "wraps": [[0, 0]]})
  with pytest.raises(KeyError, match="names"):
    Textures.from_columns({"image": [0]}, {"names": ["t"]})


def test_scene_camera_names_the_cameras_on_a_miss():
  """`Scene.camera` of an unknown name raises a KeyError listing the cameras."""
  b = rw.scene.SceneBuilder()
  b.add_camera("/cam", Camera(xform=np.eye(4, dtype=np.float32)[:3]))
  scene = b.build()
  assert scene.camera("/cam").focal_length == Camera(xform=np.eye(4)[:3]).focal_length
  with pytest.raises(KeyError, match="/cam"):
    scene.camera("/other")
