"""`rw.formats.usd.settings`: a stage's Omniverse RTX render settings as RenderSettings."""

from __future__ import annotations

import math

import pytest
from rw_helpers import new_stage

pytest.importorskip("pxr")

from conftest import recipe
from pxr import Gf, Sdf, UsdRender

import raywarp as rw
import raywarp.formats.usd
from raywarp.formats.usd.render_settings import _RTX_LOOK


def test_layer_and_render_product_keys_map_to_settings(tmp_path):
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = {
    "renderSettings": {
      "rtx:rendermode": "RaytracedLighting",
      "rtx:shadows:enabled": False,
      "rtx:post:tonemap:op": 6,
      "rtx:post:tonemap:fNumber": 8.0,
      "rtx:fog:fogColor": Gf.Vec3f(0.75, 0.75, 0.75),
    }
  }
  product = UsdRender.Product.Define(stage, "/Render/Product")
  product.GetPrim().CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("color")
  product.GetPrim().CreateAttribute("omni:rtx:reflections:enabled", Sdf.ValueTypeNames.Bool).Set(False)
  report = rw.scene.LoadReport()
  s = rw.formats.usd.settings(stage, report=report)
  assert s.aa == recipe("default").aa  # RaytracedLighting -> the default
  assert not s.shadows.enabled and not s.reflections.enabled and not s.environment.background
  assert s.post.tone_mapper == "aces_rtx"
  assert s.exposure.mode == "manual" and s.exposure.ev100 == pytest.approx(math.log2(8.0 * 8.0 * 50.0))  # f/8, 1/50 s, ISO 100
  assert report.dropped["usd.settings.fog:fogColor"] == 1
  assert s.outputs == recipe("default").outputs and s.performance == recipe("default").performance


def test_numeric_keys_convert_or_are_reported(tmp_path):
  """Copied keys convert to their field's type (an int field truncates); an unconvertible value is
  reported and leaves the field at the base's."""
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = {
    "renderSettings": {
      "rtx:rtpt:maxBounces": 3.7,
      "rtx:rtpt:maxRoughness": "0.25",
      "rtx:directLighting:sampledLighting:maxLights": "many",
      "rtx:hydra:faceCulling:enabled": 0,  # an int: False (the base culls)
    }
  }
  report = rw.scene.LoadReport()
  assert recipe("quality").surfaces.backface_culling is True
  s = rw.formats.usd.settings(stage, base=recipe("quality"), report=report)
  assert report.dropped["usd.settings.rtpt:maxBounces"] == 1
  assert s.reflections.max_roughness == pytest.approx(0.25)
  assert s.surfaces.backface_culling is False
  assert s.shadows.max_casters == recipe("quality").shadows.max_casters
  assert report.dropped["usd.settings.directLighting:sampledLighting:maxLights"] == 1


def test_explicit_base_and_defaults(tmp_path):
  stage = new_stage(tmp_path)
  assert rw.formats.usd.settings(stage) == rw.RenderSettings().replace(_RTX_LOOK)  # the RTX look recipe
  assert rw.formats.usd.settings(stage, base=recipe("quality")) == recipe("quality")


def test_unusable_exposure_keys_are_reported(tmp_path):
  """An exposure key that is not a positive number is reported and left at RTX's default (a
  token raised ValueError, a zero f-number a math domain error)."""
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = {
    "renderSettings": {
      "rtx:post:tonemap:fNumber": 0.0,
      "rtx:post:tonemap:filmIso": "fast",
      "rtx:post:tonemap:cameraShutter": 100.0,
    }
  }
  report = rw.scene.LoadReport()
  s = rw.formats.usd.settings(stage, report=report)
  assert s.exposure.mode == "manual" and s.exposure.ev100 == pytest.approx(
    math.log2(5.0 * 5.0 * 100.0)
  )  # f/5, 1/100 s, ISO 100
  assert report.dropped["usd.settings.post:tonemap:fNumber"] == 1
  assert report.dropped["usd.settings.post:tonemap:filmIso"] == 1


def test_infinite_exposure_key_is_reported(tmp_path):
  """An infinite exposure key is not a usable camera value (EV100 would be inf)."""
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = {"renderSettings": {"rtx:post:tonemap:filmIso": float("inf")}}
  report = rw.scene.LoadReport()
  s = rw.formats.usd.settings(stage, report=report)
  assert s.exposure == recipe("rtx_look").exposure
  assert report.dropped["usd.settings.post:tonemap:filmIso"] == 1
