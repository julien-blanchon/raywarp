"""The per-frame calls of a dataset generator read nothing back from the device (a host read is a
synchronisation on CUDA): `update_settings` of runtime fields, per-world `set_semantics`."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import warp as wp
from conftest import NO_GI

import raywarp as rw
from raywarp.core.pipeline.platform import device


def _renderer() -> rw.Renderer[None]:
  """Two worlds of a floor and two labelled boxes, a light."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6)))
  b.add_instance("box", material=grey, size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], path="floor")
  for i in range(2):
    b.add_instance("box", material=grey, size=(0.2,) * 3, xform=np.c_[np.eye(3), [0.5 * i, 0.0, 0.2]], path=f"crate_{i}")
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=10.0, radius=0.05, xform=np.c_[np.eye(3), [0, 0, 2.0]]))
  settings = rw.RenderSettings(
    gi=NO_GI, outputs=rw.Outputs.of("rgb", "semantic"), performance=rw.settings.Performance(traversal="software")
  )
  r = rw.Renderer(
    rw.upload(b.build(), device="cpu"), nworld=2, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(16, 12), settings=settings
  )
  pose = np.array([0.0, -3.0, 1.0, np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)], np.float32)
  r.camera_pose.assign(np.tile(pose, (2, 1, 1)))
  return r


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """The library's device-to-host reads (`device.host_copy` in every raywarp module), by caller;
  not the per-frame polls of pinned words (`validate.poll`, `groups.overflow_code`: pinned on CUDA,
  no synchronisation)."""
  calls: list[str] = []
  original = device.host_copy

  def counted(a: wp.array) -> np.ndarray:
    caller = sys._getframe(1).f_code.co_name
    if caller not in ("poll", "overflow_code"):
      calls.append(caller)
    return original(a)

  for name, module in list(sys.modules.items()):
    if name.startswith("raywarp") and getattr(module, "host_copy", None) is original:
      monkeypatch.setattr(module, "host_copy", counted)
  return calls


def test_runtime_settings_read_nothing_back_once_seen(reads: list[str]) -> None:
  """`update_settings` of runtime fields: the shadow casters are not counted again (no light slot
  changed), a background / fog colour and tone mapper seen before are not solved again, an
  unchanged call does nothing (the captured graphs stay valid)."""
  with _renderer() as r:
    r.render()
    variants = [
      {"post.tone_mapper": t, "environment.background_color": (0.1 * i, 0.2, 0.3)} for i, t in enumerate(("aces", "clamp"))
    ]
    for changes in variants:  # each colour / tone mapper solved once
      r.update_settings(**changes, **{"lighting.fill": 0.05})
      r.render()
    reads.clear()
    for changes in variants * 2:
      r.update_settings(**changes, **{"sensor.bits": 8, "reflections.max_roughness": 0.4})
      r.render()
    assert "shadow_casters" not in reads and "inverse_display_transform" not in reads, reads
    version = r.debug.graph_version()
    r.update_settings(**variants[-1])
    assert r.debug.graph_version() == version


def test_per_world_semantics_read_nothing_back(reads: list[str]) -> None:
  """`set_semantics(worlds=...)` writes the masked worlds' rows from its host copy: no device read;
  only the first per-world call makes captured graphs stale (a new table)."""
  with _renderer() as r:
    r.set_semantics({"crate": "crate_*"})
    r.render()
    r.set_semantics({"left": "crate_0"}, worlds=[1])
    version = r.debug.graph_version()
    reads.clear()
    r.set_semantics({"right": "crate_1"}, worlds=np.array([True, False]))
    r.set_semantics({"crate": "crate_*"}, worlds=[1])
    out = r.render().semantic
    assert reads == [] and r.debug.graph_version() == version
    assert out is not None
    ids = r._tables.semantics.ids.numpy()
  labels = r.semantic_labels
  names = [[labels[int(i)] for i in row[1:].tolist()] for row in ids]
  assert names == [["unlabelled", "right"], ["crate", "crate"]]
