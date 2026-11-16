"""Canonical `Features` (W6 / P2): one kernel per distinct codegen.

* Settings that compile the same code resolve to equal Features (one module, one cold compile):
  resolution and camera count, compiled-out shadow / layer / budget values, the tone mapper, the
  shading settings of a depth / segmentation-only render, the CPU register cap.
* Every field that is not collapsed changes the megakernel's Warp module hash: it reaches the
  kernel through `wp.static` or the kernel's arguments (a field read as a plain closure value is
  invisible to Warp's hash and could load a stale cached binary).
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Final

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp._base.named import replace_named
from raywarp.core.build.variants import canonical, resolve
from raywarp.core.kernels import pixel
from raywarp.core.kernels.abi.features import PLANES_NONE, PLANES_SOME, Features, small_form
from raywarp.core.pipeline.derive import fields as derived_fields
from raywarp.core.pipeline.gpu_scene import binding
from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.worlds import tables
from raywarp.core.trace.rays.primary import PINHOLE

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <light pos="1 0 3" dir="0 0 -1" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <camera name="cam2" pos="0 3 1.2" xyaxes="-1 0 0 0 0.35 1"/>
  </worldbody>
</mujoco>
"""
SHADED = {"performance.traversal": "software", "lighting.light_grid": False}
DEPTH = {**SHADED, "outputs": rw.Outputs.of("depth", "seg")}
# Groups of settings (with resolution and cameras) that compile the same kernel.
EQUIVALENT = {
  "resolution / cameras": [(SHADED, (24, 16), ["cam"]), (SHADED, (32, 24), ["cam"]), (SHADED, (24, 16), ["cam", "cam2"])],
  "shadows off": [
    ({**SHADED, "shadows.enabled": False}, None, None),
    ({**SHADED, "shadows.enabled": False, "shadows.samples": 4, "shadows.max_casters": 1}, None, None),
  ],
  "no transparency": [
    ({**SHADED, "transparency.enabled": False, "transparency.max_layers": 2}, None, None),
    ({**SHADED, "transparency.enabled": False, "transparency.max_layers": 8}, None, None),
  ],
  "budget >= lights": [
    ({**SHADED, "shadows.max_casters": None}, None, None),
    ({**SHADED, "shadows.max_casters": 2}, None, None),
    ({**SHADED, "shadows.max_casters": 9}, None, None),
  ],
  "tone mapper": [({**SHADED, "post.tone_mapper": t}, None, None) for t in ("pbr_neutral", "aces", "clamp", "aces_rtx")],
  "depth / seg only": [
    (DEPTH, None, None),
    (
      {**DEPTH, "reflections.enabled": False, "shadows.samples": 4, "lighting.area_lights": "ltc"},
      None,
      None,
    ),
    (
      {
        **DEPTH,
        "shadows.enabled": False,
        "lighting.light_grid": True,
        "transparency.enabled": True,
      },
      None,
      None,
    ),
  ],
}


def _renderer(
  scene: GpuScene[mjw.Data], settings: Mapping[str, object], res: tuple[int, int] | None, cams: list[str] | None
) -> rw.Renderer:
  """A CPU renderer of the scene (nothing rendered: Features only)."""
  return rw.Renderer(
    scene,
    nworld=1,
    cameras=cams or ["cam"],
    resolution=res or (24, 16),
    settings=rw.RenderSettings().replace(**settings),
  )


@pytest.fixture(scope="module")
def mj_model() -> mujoco.MjModel:
  """The test model."""
  return mujoco.MjModel.from_xml_string(XML)


@pytest.fixture(scope="module")
def mj_scene(mj_model: mujoco.MjModel) -> GpuScene[mjw.Data]:
  """The test scene on the CPU."""
  with wp.ScopedDevice("cpu"):
    return rw.integrations.mujoco.scene(mj_model, mjw.put_model(mj_model))


@pytest.fixture(scope="module")
def mj_data(mj_model: mujoco.MjModel) -> mjw.Data:
  """The simulator state of the test scene (one world, on the CPU)."""
  mjd = mujoco.MjData(mj_model)
  mujoco.mj_forward(mj_model, mjd)
  with wp.ScopedDevice("cpu"):
    return mjw.put_data(mj_model, mjd, nworld=1)


@pytest.mark.parametrize("group", list(EQUIVALENT))
def test_equivalent_settings_share_one_kernel(mj_scene: GpuScene[mjw.Data], group: str) -> None:
  """Each group resolves to one Features, hence one megakernel module (one cold compile)."""
  with wp.ScopedDevice("cpu"):
    features = [_renderer(mj_scene, *v).features for v in EQUIVALENT[group]]
  assert all(f == features[0] for f in features), group
  hashes = {pixel.megakernel(f, PINHOLE).module.hash_module() for f in features}
  assert len(hashes) == 1


def test_cpu_register_cap_is_collapsed(mj_scene: GpuScene[mjw.Data]) -> None:
  """The register cap (`tuning.max_registers`) is a CUDA option: every value is one CPU kernel."""
  with wp.ScopedDevice("cpu"):
    with tuning.override(max_registers=64):
      a = _renderer(mj_scene, SHADED, None, None).features
    with tuning.override(max_registers=0):
      b = _renderer(mj_scene, SHADED, None, None).features
  assert a == b and a.max_registers == 0


def test_tier_routing_follows_the_tuning(mj_scene: GpuScene[mjw.Data]) -> None:
  """The per-Renderer tier memo is keyed by the tuning too: settings resolved again under another
  tuning route to the tier as a fresh Renderer under it does (not outside every tier)."""
  with wp.ScopedDevice("cpu"):
    r = _renderer(mj_scene, {}, None, None)
    exposure = r.settings.exposure
    r.update_settings(**{"exposure.mode": "manual", "exposure.ev100": 9.0})
    with tuning.override(block_dim=64):
      fresh = _renderer(mj_scene, {}, None, None)
      r.update_settings(exposure=exposure)  # back to settings memoised under the old tuning
  tier = _tier(fresh)
  assert tier is not None and _tier(r) == tier and r.features.block_dim == 64


def _tier(r: rw.Renderer) -> str | None:
  """The kernel tier serving `r` (None: its own build)."""
  return r._choice.tier if r._choice is not None else None  # the routing itself


def _other(v: int) -> int:
  """Another value of a small enum-like field: 1, or 2 when it is 1."""
  return 1 if v != 1 else 2


# The flip of each field that is not a bool (bools are negated).
_FLIPS: Final[Mapping[str, Callable[[Features], Features | None]]] = {
  "shape_types": lambda f: replace(f, shape_types=tuple(sorted({*f.shape_types, 6}))),  # + box
  "spp": lambda f: replace(f, spp=f.spp + 1),
  "nlight": lambda f: replace(f, nlight=f.nlight + 1),
  "shadow_samples": lambda f: replace(f, shadow_samples=f.shadow_samples + 3),
  "budget": lambda f: replace(f, budget=_other(f.budget)),
  "light_cap": lambda f: replace(f, light_cap=_other(f.light_cap)),
  "hit_light_cap": lambda f: replace(f, hit_light_cap=_other(f.hit_light_cap)),
  "anisotropy": lambda f: replace(f, anisotropy=f.anisotropy + 3),
  "glossy": lambda f: replace(f, glossy=1 - f.glossy),
  "reflection_shadows": lambda f: replace(f, reflection_shadows=1 - f.reflection_shadows),
  "planes": lambda f: replace(f, planes=PLANES_NONE if f.planes == PLANES_SOME else PLANES_SOME),
  "block_dim": lambda f: replace(f, block_dim=f.block_dim * 2),
  # A counted tier build never takes the small light form (it shades every light: no count to read).
  "run_counts": lambda f: None if small_form(f.light_forms) else replace(f, run_counts=not f.run_counts),
}
# Fields that are not part of a software megakernel's codegen by design: the backend's name
# (another backend builds other programs), the register cap (a CUDA option, 0 on the CPU) and the
# block size (a launch argument: Warp keys its compiled binaries by it, `Module.get_module_hash`).
# Not code generation: the backend, the tuning, and whether a build is scene-free (its routing:
# `tiers.prebuilt`; the kernels of a tier build and its scene's own build with equal fields agree).
_NOT_CODEGEN = {"traversal", "max_registers", "block_dim", "scene_free"}


# The richest base: every effect on, so that each field has code to switch.
_RICH = {
  **SHADED,
  "lighting.area_lights": "ltc",
  "shadows.max_casters": 1,
  "transparency.enabled": True,
  "reflections.enabled": True,
  "reflections.shadows": "sun",
  "outputs": rw.Outputs.of("depth", "seg", "hdr", "normal", rgb=rw.settings.ColorOutput(alpha=True)),
}


def _flipped(f: Features, name: str) -> Features | None:
  """`f` with field `name` changed (None: no other value applies)."""
  if name in _FLIPS:
    return _FLIPS[name](f)
  value = getattr(f, name)
  if isinstance(value, bool):
    return replace_named(f, {name: not value})
  return None


@pytest.mark.slow
def test_every_codegen_field_changes_the_module_hash(mj_scene: GpuScene[mjw.Data]) -> None:
  """Flipping any field of a rich Features that canonicalisation keeps changes the module hash
  (it reaches the kernel through wp.static or an argument type)."""
  with wp.ScopedDevice("cpu"):
    base = _renderer(mj_scene, _RICH, None, None).features
    base_hash = pixel.megakernel(base, PINHOLE).module.hash_module()
    unseen = []
    for field in dataclasses.fields(Features):
      if field.name in _NOT_CODEGEN:
        continue
      variant = _flipped(base, field.name)
      if variant is None:
        continue
      variant = canonical.canonical(variant)
      if variant == base:
        continue  # collapsed: the base's kernel cannot see the field
      if pixel.megakernel(variant, PINHOLE).module.hash_module() == base_hash:
        unseen.append(field.name)
  assert not unseen, f"fields invisible to the module hash: {unseen}"


def test_depth_only_render_unchanged_by_shading_settings(mj_scene: GpuScene[mjw.Data], mj_data: mjw.Data) -> None:
  """The collapsed shading fields do not change a depth / segmentation render."""
  out = []
  with wp.ScopedDevice("cpu"):
    for settings, res, cams in EQUIVALENT["depth / seg only"]:
      r = _renderer(mj_scene, settings, res, cams)
      o = r.render(mj_data)
      assert o.depth is not None and o.seg is not None
      out.append((o.depth.numpy().copy(), o.seg.numpy().copy()))
  for depth, seg in out[1:]:
    np.testing.assert_array_equal(depth, out[0][0])
    np.testing.assert_array_equal(seg, out[0][1])


def test_light_grid_without_lights_is_not_built() -> None:
  """A scene without lights resolves to no light grid: no visibility test is traced (OptiX
  refuses a launch over zero lights: "width is 0"), whatever `lighting.light_grid` says."""
  xml = "\n".join(line for line in XML.splitlines() if "<light" not in line)
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
    r = _renderer(scene, {**SHADED, "lighting.light_grid": True}, None, None)
    assert r.scene.nlight == 0 and not r.features.culling
    r.render(mjw.put_data(mjm, mjd, nworld=1))
    assert r._baked.grid is None


# Unlit settings with the shadow fields away from their collapsed values.
_UNLIT_CASES = [
  {**_RICH, "surfaces.shading": "unlit", "shadows.samples": s, "shadows.max_casters": c} for s, c in ((1, 4), (4, 4), (4, None))
]


@pytest.mark.parametrize("settings", [_RICH, DEPTH, *_UNLIT_CASES])
def test_canonical_is_idempotent(
  mj_scene: GpuScene[mjw.Data], settings: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
  """A resolved Features is already canonical: collapsing it again changes nothing (each rule
  sees the fields the earlier rules collapsed)."""
  monkeypatch.setattr(rw.Renderer, "_start_compiles", lambda self: None)  # Features only
  with wp.ScopedDevice("cpu"):
    f = _renderer(mj_scene, settings, None, None).features
  assert canonical.canonical(f) == f


def test_unlit_shadow_settings_share_one_kernel(mj_scene: GpuScene[mjw.Data], monkeypatch: pytest.MonkeyPatch) -> None:
  """An unlit render traces no shadow ray: its shadow settings resolve to one Features."""
  monkeypatch.setattr(rw.Renderer, "_start_compiles", lambda self: None)  # Features only
  with wp.ScopedDevice("cpu"):
    features = [_renderer(mj_scene, s, None, None).features for s in _UNLIT_CASES]
  assert all(f == features[0] for f in features)


def test_declared_names_read_by_resolve_are_slots() -> None:
  """The per-world field names `resolve` looks up in the declaration are real slot, derived or
  renderer-level field names (a stale or misspelt name would silently never compile its feature
  in)."""
  names = set(re.findall(r'"([a-z_]+\.[a-z_]+)" (?:not )?in declared', inspect.getsource(resolve)))
  known = set(binding.Slots.__annotations__) | set(derived_fields.DERIVED_FIELDS) | set(tables.RENDERER_FIELDS)
  assert names and names <= known, names - known


# Prints the module ids a software frame of the test scene loads (`warmup.first_frame`) and the
# PTX key of its Features as an OptiX build's: one JSON line after "IDENTITY " (Warp logs too).
_IDENTITY_SCRIPT = """
import dataclasses, json, sys
sys.path[:0] = ["tests", "tests/unit"]
import mujoco, mujoco_warp as mjw, warp as wp, raywarp as rw, raywarp.integrations.mujoco
from fixtures.optix_codegen import scene_flags
from test_canonical_features import XML, _RICH
from raywarp.core.pipeline import warmup
from raywarp.core.trace.software_backend import SoftwareBackend
from raywarp.core.trace.optix.device import programs
m = mujoco.MjModel.from_xml_string(XML)
with wp.ScopedDevice("cpu"):
  scene = rw.integrations.mujoco.scene(m, mjw.put_model(m))
  r = rw.Renderer(scene, nworld=2, cameras=["cam"], resolution=(24, 16), settings=rw.RenderSettings().replace(**_RICH))
  default, render = warmup.first_frame(r.context(), SoftwareBackend).modules()
  ids = sorted(module.get_module_identifier() for module in (*default, *render))
  key = programs.ptx_key(dataclasses.replace(r.features, traversal="optix"), scene_flags(), supersampled=False, arch=89)
print("IDENTITY " + json.dumps({"modules": ids, "ptx": key}))
"""


@pytest.mark.slow  # two fresh processes importing raywarp and hashing a rich frame's modules (~6 s each)
def test_module_ids_and_ptx_keys_hold_across_processes() -> None:
  """A frame's Warp module ids and its PTX key are the same in processes of different string-hash
  seeds (`PYTHONHASHSEED`: set and dict iteration orders): a bundle filled in one process serves
  another (no id or key depends on an iteration order, an object id or a path)."""
  import json
  import os
  import subprocess
  import sys
  from pathlib import Path

  root = Path(__file__).resolve().parents[2]
  out = []
  for seed in ("1", "2"):
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(root)}
    run = subprocess.run(
      [sys.executable, "-c", _IDENTITY_SCRIPT], cwd=root, env=env, capture_output=True, text=True, check=True
    )
    (line,) = [s for s in run.stdout.splitlines() if s.startswith("IDENTITY ")]
    # raw_decode: a background warm-up's "Module ... load" message may follow on the same line.
    out.append(json.JSONDecoder().raw_decode(line.removeprefix("IDENTITY "))[0])
  assert len(out[0]["modules"]) > 10
  assert out[0] == out[1]
