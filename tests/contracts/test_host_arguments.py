"""Host-side argument checks of the public entry points (W9 input-validation hunt): each value
failed late, with an unrelated message, or was accepted and silently did something else."""

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.outputs import pointcloud
from raywarp.core.outputs.render_outputs import Buffers
from raywarp.scene.model.material import RaywarpExt


def test_a_list_color_names_the_field_not_unhashable() -> None:
  """`Material(base_color=[...])` failed with "unhashable type: 'list'" (the dedup lookup ran
  before the check)."""
  b = rw.SceneBuilder()
  for material in (rw.Material(base_color=[0.5, 0.5, 0.5]), rw.Material(base_color=np.full(3, 0.5))):  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(ValueError, match="base_color must be an RGB tuple"):
      b.add_material(material)


@pytest.mark.parametrize("ext", [RaywarpExt(wrap=5.0), RaywarpExt(wrap=-0.1), RaywarpExt(anisotropy_rotation=float("nan"))])
def test_material_extension_ranges(ext: RaywarpExt) -> None:
  """`ext.wrap` (documented in [0, 1]) and `ext.anisotropy_rotation` were never checked: a NaN
  rotation reached the device layer table."""
  with pytest.raises(ValueError, match=r"ext\.(wrap|anisotropy_rotation)"):
    rw.SceneBuilder().add_material(rw.Material(ext=ext))


def test_one_string_is_not_a_sequence_of_strings() -> None:
  """`paths="abc"` / `labels="cup"` named three instances "a", "b", "c"; `wrap="clamp"` failed
  with "wrap 'c' is not one of"; a failed call left the builder's path rows ahead of its labels."""
  b = rw.SceneBuilder()
  xforms = np.tile(np.eye(4)[:3], (3, 1, 1))
  for kwargs in ({"paths": "abc"}, {"labels": "cup"}):
    with pytest.raises(TypeError, match="got the string"):
      b.add_instances("box", xforms=xforms, size=(1.0, 1.0, 1.0), **kwargs)  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(ValueError, match="1 labels for 3 instances"):
    b.add_instances("box", xforms=xforms, size=(1.0, 1.0, 1.0), paths=["a", "b", "c"], labels=["x"])
  assert b.ninstance == 0
  b.add_instances("box", xforms=xforms, size=(1.0, 1.0, 1.0), paths=["a", "b", "c"], labels=["x", "", "y"])
  assert b.ninstance == 3
  image = np.zeros((2, 2, 3), np.uint8)
  for wrap in ("clamp", ("clamp",)):
    with pytest.raises(ValueError, match=r"wrap is the \(s, t\) pair"):
      b.add_texture(image, wrap=wrap)  # pyrefly: ignore[bad-argument-type]


def test_scene_options_types() -> None:
  """`SceneOptions(emissive_lights=0)` fitted the panels (only `False` skips them) and
  `textures=None` failed at upload with an AttributeError."""
  assert rw.SceneOptions(emissive_lights=np.False_).fit_emissive is False  # pyrefly: ignore[bad-argument-type]
  for kwargs in ({"emissive_lights": 0}, {"env_sun": 1}, {"textures": None}):
    with pytest.raises(TypeError, match="SceneOptions"):
      rw.SceneOptions(**kwargs)  # pyrefly: ignore[bad-argument-type]


def test_colors_given_as_lists_are_stored_as_tuples() -> None:
  """A list color was stored as is: the settings were unhashable (the Renderer's tier memo keys on
  them) and `diff` reported a change against the equal tuple."""
  changes = {"lighting.fill_color": [1, 1, 1], "environment.background_color": [0, 0, 0], "atmosphere.color": [0.5] * 3}
  s = rw.RenderSettings().replace(**changes)
  hash(s)
  assert rw.RenderSettings().diff(s) == {}
  assert s.lighting.fill_color == (1.0, 1.0, 1.0)


def test_colors_given_as_numpy_arrays_are_stored_as_tuples() -> None:
  """NumPy colors (as `sensor.vignette` takes) were rejected; they are stored as plain float
  tuples, the settings hashable and equal to the tuple form."""
  changes = {
    "lighting.fill_color": np.ones(3),
    "environment.background_color": np.zeros(3, np.float32),
    "atmosphere.color": np.full(3, 0.5),
    "grading.gain": np.array([1.0, 1.0, 1.0]),
  }
  s = rw.RenderSettings().replace(**changes)
  hash(s)
  assert rw.RenderSettings().diff(s) == {}
  assert type(s.environment.background_color[0]) is float
  for bad in (np.ones(4), np.ones((3, 1)), np.array([0.0, np.nan, 0.0]), (True, False, True)):
    with pytest.raises(ValueError, match="background_color"):
      rw.RenderSettings().replace(**{"environment.background_color": bad})


def test_usd_load_checks_its_options_first() -> None:
  """Swapped `options` / `adapter` (both "Options") failed with an AttributeError; `threads=0`
  raised after the stage read (never on a cache hit); an unknown `verify` meant "files"."""
  usd = pytest.importorskip("raywarp.formats.usd")
  for kwargs, error in (
    ({"options": usd.Options()}, TypeError),
    ({"adapter": rw.SceneOptions()}, TypeError),
    ({"threads": 0}, ValueError),
    ({"verify": "nope"}, ValueError),
  ):
    with pytest.raises(error):
      usd.load("missing.usda", **kwargs)


def test_point_cloud_request_counts_and_seeds() -> None:
  """`npoints=True` gave one point per world, a float seed failed in the kernel launch and a seed
  past int32 wrapped (two seeds, one cloud)."""
  for kwargs in (
    {"npoints": True},
    {"npoints": 0},
    {"npoints": 2.0},
    {"npoints": 2**31},
    {"seed": 1.5},
    {"seed": 2**31},
    {"keep": [2**40]},
  ):
    with pytest.raises(ValueError, match="point_cloud"):
      pointcloud.Request(**kwargs)  # pyrefly: ignore[bad-argument-type]
  assert pointcloud.Request(npoints=np.int64(4), seed=np.int32(-3)).npoints == 4  # pyrefly: ignore[bad-argument-type]


def test_bound_outputs_keep_their_element_kind() -> None:
  """Any element of the output's byte size was accepted: an int32 array bound as "depth" got the
  float bits, a float32 one as "seg" the id bits. Integer outputs still take any integer of their
  size (an RGBA uint32 / int32 / vec4ub, int32 / uint32 ids)."""
  cpu = wp.get_device("cpu")
  buffers = Buffers(rw.Outputs.of("rgb", "depth", "seg"), nworld=2, sizes=[(4, 3)], device=cpu)
  for name, dtype in (("depth", wp.int32), ("seg", wp.float32), ("rgb", wp.float32), ("depth", wp.float16)):
    with pytest.raises(ValueError, match="cannot hold"):
      buffers.bind({name: wp.zeros((2, 12), dtype=dtype, device=cpu)})
  integer = (wp.int32, wp.vec4ub, wp.types.vector(4, wp.uint8))
  for name, dtype in (*(("rgb", t) for t in integer), ("seg", wp.uint32), ("depth", wp.float32)):
    buffers.bind({name: wp.zeros((2, 12), dtype=dtype, device=cpu)})


def test_camera_local_offset_is_a_rotation() -> None:
  """`CameraSpec.local` took any nine finite numbers: a scaled matrix changed the field of view and
  the depth, a mirrored one flipped the image."""
  for rot in (
    2.0 * np.eye(3),
    np.diag(np.array([1.0, 1.0, -1.0])),
    np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
  ):
    with pytest.raises(ValueError, match="local rotation must be a rotation matrix"):
      rw.CameraSpec(mount="cam", local=(np.zeros(3), rot))
  c, s = np.cos(0.3), np.sin(0.3)
  rw.CameraSpec(mount="cam", local=(np.zeros(3), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)))
