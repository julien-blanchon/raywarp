"""Trace backend choice by device capability (`backend.select`): OptiX on RT-core GPUs,
the software traversal elsewhere and for the geometry OptiX does not trace; no fallback."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

import pytest

from raywarp._base.types import ShapeType
from raywarp.core.pipeline.platform.device import DeviceCaps, OptixSupport, has_rt_cores
from raywarp.core.trace import backend

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene


def _caps(arch: int, *, optix: bool) -> DeviceCaps:
  """The caps of a device of SM version `arch` (0: the CPU), its OptiX support injected."""
  cuda = arch > 0
  caps = DeviceCaps(
    alias="cuda:0" if cuda else "cpu",
    kind="cuda" if cuda else "cpu",
    arch=arch,
    rt_cores=cuda and has_rt_cores(arch),
    graphs=cuda,
    pinned=cuda,
    device_textures=cuda,
    parallel_launch=cuda,
    blackwell_rtx=arch >= 120,
  )
  caps.__dict__["optix"] = OptixSupport(available=optix, reason=None if optix else "driver OptiX is not available (test)")
  return caps


def _scene(*shape_types: ShapeType, flex: tuple[int, ...] = ()) -> GpuScene:
  """What `scene_reason` reads of a GpuScene: its shape types and deformable dimensions."""
  return cast(
    "GpuScene",
    types.SimpleNamespace(
      capabilities=types.SimpleNamespace(shape_types=[int(t) for t in shape_types]),
      deformables=types.SimpleNamespace(count=len(flex), dim=list(flex)),
    ),
  )


_RT_ARCHS = [
  (0, False),
  (70, False),
  (75, True),
  (80, False),
  (86, True),
  (87, False),
  (89, True),
  (90, False),
  (100, False),
  (101, False),
  (103, False),
  (110, False),
  (120, True),
]


@pytest.mark.parametrize(("arch", "rt"), _RT_ARCHS)
def test_rt_cores_follow_the_sm_version(arch: int, rt: bool) -> None:
  """Turing and later have RT cores except the compute-only GPUs (A100, Orin, H100, B200, B300)."""
  assert has_rt_cores(arch) is rt


@pytest.mark.parametrize(
  "scene",
  [
    _scene(ShapeType.MESH),
    _scene(ShapeType.BOX, ShapeType.SPHERE, ShapeType.PLANE),
    _scene(ShapeType.MESH, ShapeType.HFIELD),
    _scene(ShapeType.SDF),
    _scene(ShapeType.MESH, flex=(2, 3)),
  ],
)
def test_auto_is_optix_on_rt_core_gpus_for_every_traceable_scene(scene: GpuScene) -> None:
  """Primitive-only scenes, height fields, SDFs and deformable surfaces too: the device decides,
  not the scene's content."""
  assert backend.select("auto", scene, _caps(89, optix=True)).name == "optix"


@pytest.mark.parametrize("arch", [0, 80, 90])
def test_auto_is_software_without_rt_cores(arch: int) -> None:
  """CPU, A100 and H100 take the software traversal even where OptiX could run."""
  assert backend.select("auto", _scene(ShapeType.MESH), _caps(arch, optix=arch > 0)).name == "software"


def test_auto_takes_software_for_geometry_optix_cannot_trace() -> None:
  """Dim-1 deformables (segments) are traced by the software backend only."""
  assert backend.select("auto", _scene(ShapeType.MESH, flex=(2, 1)), _caps(120, optix=True)).name == "software"


def test_auto_without_optix_still_renders_what_optix_would_not_trace() -> None:
  """The device prerequisite only matters for scenes the pick would trace."""
  assert backend.select("auto", _scene(flex=(1,)), _caps(89, optix=False)).name == "software"


def test_auto_on_an_rt_core_gpu_without_optix_raises() -> None:
  """No silent downgrade: the preferred backend's missing prerequisite is an error naming it."""
  with pytest.raises(ValueError, match=r"driver OptiX is not available.*traversal=\"software\""):
    backend.select("auto", _scene(ShapeType.MESH), _caps(89, optix=False))


@pytest.mark.parametrize(
  ("scene", "arch", "optix", "match"),
  [
    (_scene(ShapeType.MESH), 0, False, "driver OptiX"),
    (_scene(ShapeType.MESH, flex=(1,)), 89, True, "dim-1 flex"),
  ],
)
def test_named_optix_that_cannot_render_raises(scene: GpuScene, arch: int, optix: bool, match: str) -> None:
  with pytest.raises(ValueError, match=match):
    backend.select("optix", scene, _caps(arch, optix=optix))


@pytest.mark.parametrize("arch", [0, 80, 89])
def test_named_backends_run_where_they_can(arch: int) -> None:
  """ "software" everywhere; "optix" on any device where OptiX runs (an A100 included)."""
  scene = _scene(ShapeType.MESH, ShapeType.HFIELD)
  assert backend.select("software", scene, _caps(arch, optix=arch > 0)).name == "software"
  if arch:
    assert backend.select("optix", _scene(ShapeType.MESH), _caps(arch, optix=True)).name == "optix"
