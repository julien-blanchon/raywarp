"""UsdGeom.Camera -> core camera (pinhole intrinsics, clipping, focus; -Z forward, +Y up).

Focal length and apertures keep their authored units (tenths of a scene unit by convention):
only their ratio matters for a perspective camera. Orthographic apertures and the clipping /
focus distances are converted to metres. A camera prim is read once (`read_camera`) and placed
under each of its instancing transforms (`emit_camera`). Authored values out of UsdGeom's ranges
are replaced and reported, and a camera that cannot be placed is dropped and reported: neither
aborts the stage.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, final

import numpy as np
from pxr import Gf, Usd, UsdGeom

from raywarp.formats.usd.util import Matrix4, Time
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import Camera, Projection
from raywarp.scene.model.xform import orthonormalize


@final
@dataclasses.dataclass(frozen=True)
class StageCamera:
  """A camera read once: intrinsics in authored units, distances in local units."""

  orthographic: bool
  focal_length: float
  aperture: tuple[float, float]  # horizontal, vertical
  offset: tuple[float, float]  # horizontal, vertical
  clip: tuple[float, float]  # near, far
  fstop: float
  focus_distance: float
  exposure: float  # stops


# Authored-value defaults (UsdGeom.Camera's own: tenths of a scene unit, the clipping range in
# scene units) that replace an invalid value.
_FOCAL_LENGTH: Final = 50.0
_APERTURE: Final = (20.955, 15.2908)
_NEAR: Final = 1.0


def _valid(report: LoadReport, prim: Usd.Prim, attr: str, value: float, *, ok: bool, default: float) -> float:
  """`value` when `ok`, else `default`, the replacement reported (`Camera.<attr>.clamped`)."""
  if ok:
    return value
  report.approx(f"Camera.{attr}.clamped", f"{prim.GetPath()}: {value} (read as {default})")
  return default


def _positive(report: LoadReport, prim: Usd.Prim, attr: str, value: float, default: float) -> float:
  """A finite positive value, else `default` (reported)."""
  return _valid(report, prim, attr, value, ok=math.isfinite(value) and value > 0.0, default=default)


def _non_negative(report: LoadReport, prim: Usd.Prim, attr: str, value: float) -> float:
  """A finite value >= 0, else 0 (reported)."""
  return _valid(report, prim, attr, value, ok=math.isfinite(value) and value >= 0.0, default=0.0)


def _finite(report: LoadReport, prim: Usd.Prim, attr: str, value: float) -> float:
  """A finite value, else 0 (reported)."""
  return _valid(report, prim, attr, value, ok=math.isfinite(value), default=0.0)


def _clip(report: LoadReport, prim: Usd.Prim, cam: Gf.Camera) -> tuple[float, float]:
  """(near, far) of the clipping range: near finite and >= 0 (else 1), far beyond it (else
  unbounded), the replacements reported."""
  near = float(cam.clippingRange.min)
  near = _valid(report, prim, "clippingRange.near", near, ok=math.isfinite(near) and near >= 0.0, default=_NEAR)
  far = float(cam.clippingRange.max)
  return near, _valid(report, prim, "clippingRange.far", far, ok=far > near, default=math.inf)


def _exposure(report: LoadReport, prim: Usd.Prim, schema: UsdGeom.Camera) -> float:
  """USD's camera exposure as stops (exposure, exposure:iso / time / fStop / responsivity); a
  non-positive or non-finite linear scale reads as 0 stops (reported)."""
  scale = float(schema.ComputeLinearExposureScale())
  if math.isfinite(scale) and scale > 0.0:
    return float(np.log2(scale))
  report.approx("Camera.exposure.clamped", f"{prim.GetPath()}: linear scale {scale} (read as 0 stops)")
  return 0.0


def _focal_length(report: LoadReport, prim: Usd.Prim, cam: Gf.Camera) -> float:
  """The focal length: validated for a perspective camera; an orthographic one does not use it
  (an invalid value becomes the default silently: the core camera still needs one > 0)."""
  value = float(cam.focalLength)
  if cam.projection == Gf.Camera.Orthographic:
    return value if math.isfinite(value) and value > 0.0 else _FOCAL_LENGTH
  return _positive(report, prim, "focalLength", value, _FOCAL_LENGTH)


def read_camera(report: LoadReport, prim: Usd.Prim, time: Time) -> StageCamera:
  """The camera of a UsdGeom.Camera prim (unsupported inputs reported once).

  Authored nonsense does not abort the stage: an invalid value (UsdGeom's ranges: focal length
  and apertures > 0, fStop and focusDistance >= 0, 0 <= near < far, all finite) is replaced by
  its default and reported (`Camera.<attr>.clamped`; the focal length of an orthographic camera
  is unused, not validated). A negative fStop is "no depth of field", as fStop 0 (UsdGeom: fStop
  <= 0 disables it), and is still reported.
  """
  schema = UsdGeom.Camera(prim)
  cam = schema.GetCamera(time)
  fstop = _non_negative(report, prim, "fStop", float(cam.fStop))
  if fstop > 0.0:
    report.approx("Camera.fStop(pinhole)", prim)
  if schema.GetClippingPlanesAttr().HasAuthoredValue() and len(cam.clippingPlanes):
    report.drop("Camera.clippingPlanes", prim)
  return StageCamera(
    orthographic=cam.projection == Gf.Camera.Orthographic,
    focal_length=_focal_length(report, prim, cam),
    aperture=(
      _positive(report, prim, "horizontalAperture", float(cam.horizontalAperture), _APERTURE[0]),
      _positive(report, prim, "verticalAperture", float(cam.verticalAperture), _APERTURE[1]),
    ),
    offset=(
      _finite(report, prim, "horizontalApertureOffset", float(cam.horizontalApertureOffset)),
      _finite(report, prim, "verticalApertureOffset", float(cam.verticalApertureOffset)),
    ),
    clip=_clip(report, prim, cam),
    fstop=fstop,
    focus_distance=_non_negative(report, prim, "focusDistance", float(cam.focusDistance)),
    exposure=_exposure(report, prim, schema),
  )


def _core_camera(spec: StageCamera, world: Matrix4) -> Camera:
  """The core camera of `spec` at `world` (4, 4), which includes the stage's metersPerUnit scale.

  Raises:
    ValueError: for a non-finite `world` (numpy's LinAlgError).
  """
  xform, scales = orthonormalize(world)
  scale = float(np.cbrt(np.prod(scales))) or 1.0
  # Orthographic apertures are tenths of a scene unit; perspective ones only form a ratio.
  aperture_scale = 0.1 * scale if spec.orthographic else 1.0
  return Camera(
    xform=xform.astype(np.float32),
    projection=Projection.ORTHOGRAPHIC if spec.orthographic else Projection.PERSPECTIVE,
    focal_length=spec.focal_length,
    horizontal_aperture=spec.aperture[0] * aperture_scale,
    vertical_aperture=spec.aperture[1] * aperture_scale,
    horizontal_offset=spec.offset[0] * aperture_scale,
    vertical_offset=spec.offset[1] * aperture_scale,
    near=spec.clip[0] * scale,
    far=spec.clip[1] * scale if math.isfinite(spec.clip[1]) else 0.0,  # 0: unbounded
    fstop=spec.fstop,
    focus_distance=spec.focus_distance * scale,
    exposure=spec.exposure,
  )


def emit_camera(builder: SceneBuilder, report: LoadReport, spec: StageCamera, world: Matrix4, name: str) -> None:
  """Adds camera `name` at `world` (4, 4); a camera that cannot be placed or that the builder
  rejects (a non-finite transform, a degenerate scale, ...) is dropped and reported
  (`camera.invalid`): the stage still loads."""
  try:
    builder.add_camera(name, _core_camera(spec, world))
  except ValueError as e:  # numpy's LinAlgError is a ValueError
    report.drop("camera.invalid", f"{name}: {e}")
    return
  report.count("cameras")
