"""Analytic HDR calibration: directional light units and the common exposure convention."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from PIL import Image, ImageDraw
from pxr import Sdf, Usd, UsdLux

import raywarp as rw
from raywarp.core.post.grading.color import tonemap_aces_rtx
from raywarp.formats.usd.stage.lux import read_light
from raywarp.scene.model.report import LoadReport
from raywarp.settings.fields import exposure_from_ev100


def _distant(version: int | None, *, normalize: bool, angle: float, kit: bool = False) -> tuple[rw.Light, LoadReport]:
  """A USD light read through the adapter (`kit`: in a layer saved by Omniverse Kit), with its
  diagnostic report."""
  stage = Usd.Stage.CreateInMemory()
  if kit:
    stage.GetRootLayer().customLayerData = {"omni_layer": {}, "renderSettings": {}}
  schema = UsdLux.DistantLight.Define(stage, "/Light")
  schema.CreateIntensityAttr(1000.0)
  schema.CreateAngleAttr(angle)
  schema.CreateNormalizeAttr(normalize)
  if version is not None:
    schema.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(version)
  report = LoadReport()
  light = read_light(
    rw.SceneBuilder(), report, schema.GetPrim(), time=Usd.TimeCode.Default(), up_z=True, mpu=1.0, lux_dialect=None
  )
  assert isinstance(light, rw.Light)
  return light, report


@pytest.mark.parametrize("kit", [False, True], ids=["usd", "kit-layer"])
@pytest.mark.parametrize("version", [None, 2411, 2505, 2600])
@pytest.mark.parametrize("normalize,angle", [(True, 0.53), (False, 0.53), (False, 0.0), (False, 10.0)])
def test_distant_version_and_normalization(version: int | None, normalize: bool, angle: float, kit: bool) -> None:
  """A1-A3: an unnormalized distant light's intensity is a radiance scaled by its projected
  angular area (UsdLux), except in RTX's legacy dialect (version < 2505, or a Kit layer without a
  version), where it stays lux. An authored version wins over the layer."""
  light, report = _distant(version, normalize=normalize, angle=angle, kit=kit)
  legacy = version < 2505 if version is not None else kit
  expected = 1000.0
  if not legacy and not normalize and angle > 0.0:
    expected *= math.pi * math.sin(math.radians(angle) / 2.0) ** 2
  assert light.intensity == pytest.approx(expected, rel=2e-6)
  assert not report.approximated


def _plane(light: rw.Light) -> rw.Renderer[None]:
  """A normal-facing, 50%-reflecting Lambertian plane, without other light sources."""
  builder = rw.SceneBuilder()
  material = builder.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_weight=0.0, specular_roughness=1.0))
  builder.add_instance("plane", material=material, size=(2.0, 2.0, 0.0))
  builder.add_light(light)
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("rgb", "hdr"),
      "performance.traversal": "software",
      "post.tone_mapper": "clamp",
      "exposure.mode": "manual",
      "exposure.ev100": 10.0,
      "environment.enabled": False,
      "gi.enabled": False,
      "lighting.light_grid": False,
      "lighting.emissive_lights": False,
      "shadows.enabled": False,
      "reflections.enabled": False,
    }
  )
  renderer = rw.Renderer(
    rw.upload(builder.build(), device="cpu"),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=10.0)],
    resolution=(8, 8),
    settings=settings,
  )
  renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
  return renderer


def _evidence(name: str, hdr: np.ndarray[[int, int, 3], np.dtype[np.float32]], expected: float) -> None:
  """Optional annotated calibration image from the tested HDR, for a coordinated evidence run."""
  directory = os.environ.get("RAYWARP_CALIBRATION_OUTPUT")
  if directory is None:
    return
  path = Path(directory)
  path.mkdir(parents=True, exist_ok=True)
  # Use a declared per-image display scale so both tiny and bright analytic values are visible.
  linear = np.clip(hdr * (0.18 / expected), 0.0, 1.0)
  srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055)
  tile = Image.fromarray(np.round(srgb * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)
  image = Image.new("RGB", (512, 310), (24, 24, 24))
  image.paste(tile, (128, 0))
  draw = ImageDraw.Draw(image)
  draw.text((8, 265), f"{name}: expected {expected:.6g} nits; measured {float(hdr.mean()):.6g}", fill="white")
  draw.text((8, 285), f"Display scale: 0.18 / expected = {0.18 / expected:.6g}", fill="white")
  image.save(path / f"{name}.png")
  np.save(path / f"{name}.npy", hdr)


@pytest.mark.parametrize("normalize,angle", [(True, 0.53), (False, 0.53), (False, 10.0)])
def test_distant_lambertian_hdr(normalize: bool, angle: float) -> None:
  """A1-A3 integration: 50% Lambertian radiance is 0.5 * illuminance / pi, within 1%."""
  light, _ = _distant(2505, normalize=normalize, angle=angle)
  expected = 500.0 / math.pi if normalize else 500.0 * math.sin(math.radians(angle) / 2.0) ** 2
  with _plane(light) as renderer:
    output = renderer.render()
    assert output.hdr is not None
    hdr = output.hdr.numpy()[0, 0]
    np.testing.assert_allclose(hdr, expected, rtol=0.01)
    _evidence(f"distant_normalized_{normalize}_angle_{angle}", hdr, expected)


def test_clamp_exposure_matches_hdr_and_is_independent_of_tone_mapper() -> None:
  """A18: EV applies the same 2^-EV radiometric scale before every selected display curve."""
  light, _ = _distant(2505, normalize=True, angle=0.53)
  with _plane(light) as renderer:
    for ev in (10.0, 11.0):
      renderer.update_settings(**{"exposure.mode": "manual", "exposure.ev100": ev, "post.tone_mapper": "clamp"})
      output = renderer.render()
      assert output.hdr is not None and output.rgb is not None
      hdr = output.hdr.numpy().copy()
      linear = hdr * 2.0**-ev
      expected = 1.055 * linear ** (1.0 / 2.4) - 0.055  # all pixels are above the sRGB toe
      np.testing.assert_allclose(output.rgb.numpy()[..., :3] / 255.0, expected, atol=1.0 / 255.0)
      assert exposure_from_ev100(ev) == 2.0**-ev
      for curve in ("pbr_neutral", "aces", "aces_rtx"):
        renderer.update_settings(**{"post.tone_mapper": curve})
        changed = renderer.render()
        assert changed.hdr is not None
        np.testing.assert_array_equal(changed.hdr.numpy(), hdr)
        assert renderer.debug.frame().exposure.numpy()[0] == pytest.approx(2.0**-ev)


@wp.kernel
def _aces_curve(
  # In:
  values: wp.array1d[wp.vec3],
  # Out:
  output: wp.array1d[wp.vec3],
) -> None:
  """Evaluate the RTX display curve on already-exposed values."""
  i = wp.tid()
  output[i] = tonemap_aces_rtx(values[i])


def test_rtx_curve_does_not_apply_a_second_exposure() -> None:
  """Narkowicz's published curve receives exposed values without another 1.2 multiplier."""
  values = np.array([[0.0, 0.01, 0.18], [0.5, 1.0, 4.0]], np.float32)
  source = wp.array(values, dtype=wp.vec3, device="cpu")
  output = wp.empty_like(source)
  wp.launch(_aces_curve, dim=2, inputs=[source], outputs=[output], device="cpu")
  expected = values * (2.51 * values + 0.03) / (values * (2.43 * values + 0.59) + 0.14)
  np.testing.assert_allclose(output.numpy(), expected, rtol=2e-6)
