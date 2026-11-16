"""Checker-only shader factory probes; included in the ordinary Pyrefly tests gate.

The functions are deliberately not executed: real factory construction and compiled output
decomposition are covered by test_outputs.py and test_unlit.py. Exact ignore codes below are
negative probes: Pyrefly's configured unused-type-ignore gate fails if an input becomes permissive.
"""

from __future__ import annotations

from typing import assert_type

from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.layer import ShadeLayer, make_shade_layer
from raywarp.core.kernels.lighting import DirectLighting
from raywarp.core.kernels.surface import ShadeSurface, ShadeSurfaceSplit, make_indirect, make_shade_surface
from raywarp.core.shade.materials.evaluate import EvalSurface


def _surface_returns(f: Features, lighting: DirectLighting, split: bool) -> None:
  assert_type(make_shade_surface(f, lighting, None), ShadeSurface)
  assert_type(make_shade_surface(f, lighting, None, split=False), ShadeSurface)
  assert_type(make_shade_surface(f, lighting, None, split=True), ShadeSurfaceSplit)
  assert_type(make_shade_surface(f, lighting, None, split=split), ShadeSurface | ShadeSurfaceSplit)


def _layer_returns(f: Features, evaluate: EvalSurface, shade: ShadeSurface) -> None:
  layer: ShadeLayer = make_shade_layer(f, eval_surface=evaluate, shade_surface=shade)  # inferred: fits its consumers
  del layer


def _wrong_callback(value: float) -> float:
  return value


def _invalid_callbacks(f: Features, lighting: DirectLighting, shade: ShadeSurface) -> None:
  make_indirect(f, _wrong_callback)  # type: ignore[bad-argument-type]  # not the Glossy interface
  make_shade_surface(f, lighting, _wrong_callback)  # type: ignore[bad-argument-type]  # wrong Glossy input
  # Not the EvalSurface interface:
  make_shade_layer(f, eval_surface=_wrong_callback, shade_surface=shade)  # type: ignore[bad-argument-type]
