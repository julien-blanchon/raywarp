"""The derive kernels of the per-world OpenPBR fields (`derived_fields.py`): the upload's
formulas (`core/pipeline/derive/materials.py`) per world, in float64 from the float32 tables.

Every input is read through `_read`: a value that is not finite or outside its range (`RANGES`)
derives from the scene's value instead (the validation reports it:
`validator_kernels.Word.DERIVED`).
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import COAT_ROUGHNESS_SHIFT, COAT_WEIGHT_SHIFT
from raywarp.core.pipeline.derive.materials import GLTF_COAT_IOR, MAX_REFLECTANCE

# fuse_fp off: no FMA contraction on CUDA, so the float64 derivation rounds as the host
# upload's does (bit for bit).
wp.set_module_options({"enable_backward": False, "default_grid_stride": False, "fuse_fp": False})

# The coat bytes of `instance.shading` (`types.pack_clearcoat`).
_COAT_BITS: Final = (255 << COAT_WEIGHT_SHIFT) | (255 << COAT_ROUGHNESS_SHIFT)
_BYTE: Final = 255.0
# Valid per-world values, (low, high): an IOR is > 0 (the smallest normal float32), a weight or a
# roughness in [0, 1], a specular weight or a luminance >= 0 (the largest float32: any finite).
_MIN_POSITIVE: Final = 1.17549435e-38
_MAX_FINITE: Final = 3.4028235e38
_UNIT: Final = (0.0, 1.0)
_NON_NEGATIVE: Final = (0.0, _MAX_FINITE)
_POSITIVE: Final = (_MIN_POSITIVE, _MAX_FINITE)
# A luminance [nits] at most 1e30: luminance x peak color stays a finite float32.
_LUMINANCE: Final = (0.0, 1.0e30)
RANGES: Final = {
  "material.specular_weight": _NON_NEGATIVE,
  "material.specular_ior": _POSITIVE,
  "material.coat_weight": _UNIT,
  "material.coat_ior": _POSITIVE,
  "material.coat_roughness": _UNIT,
  "material.emission_luminance": _LUMINANCE,
}


@wp.func
def _read(
  # In:
  table: wp.array2d[float],
  scene: wp.array1d[float],
  w: int,
  m: int,
  low: float,
  high: float,
) -> wp.float64:
  """World w's value of material m in a (1 or nworld, nmat) table, the scene's where it is not
  finite or outside [low, high] (`RANGES`; the validation reports it: `Word.DERIVED`)."""
  x = table[w % table.shape[0], m]
  if not (wp.isfinite(x) and x >= low and x <= high):
    x = scene[m]
  return wp.float64(x)


@wp.func
def _coupled(
  # In:
  coat_weight: wp.float64,
  gltf: bool,
) -> bool:
  """An OpenPBR coat (`derive._coupled_coat`): coat_weight > 0, not a glTF clearcoat."""
  return coat_weight > wp.float64(0.0) and not gltf


@wp.kernel
def derive_reflectance(
  # In:
  weight: wp.array2d[float],
  ior: wp.array2d[float],
  coat_weight: wp.array2d[float],
  coat_ior: wp.array2d[float],
  weight_scene: wp.array1d[float],
  ior_scene: wp.array1d[float],
  coat_weight_scene: wp.array1d[float],
  coat_ior_scene: wp.array1d[float],
  gltf: wp.array1d[bool],
  thin: wp.array1d[float],
  # Out:
  reflectance: wp.array2d[float],
) -> None:
  """reflectance[w, m] = min(sqrt(weight / 4) x ior_scale, 2.5), ior_scale = 5 |eta - 1| / (eta + 1)
  of the base IOR under a coupled coat, with a thin-walled pane's two interfaces over its share
  `thin` (`derive._ior_scale` / `thin_wall_scale` / `_reflectance`, float64, rounded once). The
  inputs are (1 or nworld, nmat) tables (`_read`: the scene's value where invalid)."""
  w, m = wp.tid()
  eta = _read(ior, ior_scene, w, m, wp.static(_POSITIVE[0]), wp.static(_POSITIVE[1]))
  cw = _read(coat_weight, coat_weight_scene, w, m, wp.static(_UNIT[0]), wp.static(_UNIT[1]))
  ratio = eta / _read(coat_ior, coat_ior_scene, w, m, wp.static(_POSITIVE[0]), wp.static(_POSITIVE[1]))
  if ratio < wp.float64(1.0):
    ratio = wp.float64(1.0) / ratio
  if _coupled(cw, gltf[m]):
    eta = eta + (ratio - eta) * cw
  scale = wp.float64(5.0) * wp.abs(eta - wp.float64(1.0)) / (eta + wp.float64(1.0))
  q = scale / wp.float64(5.0)
  f0 = q * q
  one = wp.float64(1.0)
  scale = scale * wp.sqrt(one + wp.float64(thin[m]) * (one - f0) / (one + f0))
  sw = _read(weight, weight_scene, w, m, wp.static(_NON_NEGATIVE[0]), wp.static(_NON_NEGATIVE[1]))
  r = wp.sqrt(sw / wp.float64(4.0)) * scale
  reflectance[w, m] = wp.float32(wp.min(r, wp.float64(MAX_REFLECTANCE)))


@wp.func
def roughen_under_coat(
  # In:
  roughness: wp.float64,
  coat_weight: wp.float64,
  coat_roughness: wp.float64,
) -> wp.float64:
  """Canonical coupled-coat roughening; shared by declared coat and material-input updates."""
  four = wp.float64(4.0)
  under = wp.pow(
    wp.min(wp.float64(1.0), wp.float64(2.0) * wp.pow(coat_roughness, four) + wp.pow(roughness, four)), wp.float64(0.25)
  )
  return roughness + (under - roughness) * coat_weight


@wp.kernel
def derive_roughness(
  # In:
  coat_weight: wp.array2d[float],
  coat_roughness: wp.array2d[float],
  coat_weight_scene: wp.array1d[float],
  coat_roughness_scene: wp.array1d[float],
  gltf: wp.array1d[bool],
  roughenable: wp.array1d[bool],
  base: wp.array1d[float],
  # Out:
  roughness: wp.array2d[float],
) -> None:
  """roughness[w, m]: the pre-coat roughness r, under a coupled coat over an untextured roughness
  r + (min(1, 2 c^4 + r^4)^(1/4) - r) x coat_weight (`derive._roughness`, float64)."""
  w, m = wp.tid()
  r = wp.float64(base[m])
  cw = _read(coat_weight, coat_weight_scene, w, m, wp.static(_UNIT[0]), wp.static(_UNIT[1]))
  if roughenable[m] and _coupled(cw, gltf[m]):
    c = _read(coat_roughness, coat_roughness_scene, w, m, wp.static(_UNIT[0]), wp.static(_UNIT[1]))
    r = roughen_under_coat(r, cw, c)
  roughness[w, m] = wp.float32(r)


@wp.kernel
def roughness_of_shininess(
  # In:
  shininess: wp.array2d[float],
  shininess_scene: wp.array1d[float],
  from_shininess: wp.array1d[bool],
  base: wp.array1d[float],
  # Out:
  roughness: wp.array2d[float],
) -> None:
  """roughness[w, m]: 1 - shininess, clipped to [0, 1], of a material whose roughness is unset
  (`from_shininess`; MuJoCo's `materials._metalness_roughness`, float64 rounded once), the live
  shininess a (1, K or nworld, nmat) table read as world % rows (MuJoCo Warp's batched model; a
  non-finite value: the scene's); `base` for the others."""
  w, m = wp.tid()
  r = base[m]
  if from_shininess[m]:
    s = shininess[w % shininess.shape[0], m]
    if not wp.isfinite(s):
      s = shininess_scene[m]
    one = wp.float64(1.0)
    r = wp.float32(wp.clamp(one - wp.float64(s), wp.float64(0.0), one))
  roughness[w, m] = r


@wp.func
def _byte(
  # In:
  x: wp.float64,
) -> wp.int32:
  """The byte of x in [0, 1] (clipped), rounded to the nearest even (`types.pack_clearcoat`)."""
  return wp.int32(wp.rint(wp.clamp(x, wp.float64(0.0), wp.float64(1.0)) * wp.float64(_BYTE)))


@wp.kernel
def derive_shading(
  # In:
  coat_weight: wp.array2d[float],
  coat_roughness: wp.array2d[float],
  coat_weight_scene: wp.array1d[float],
  coat_roughness_scene: wp.array1d[float],
  gltf: wp.array1d[bool],
  material_shading: wp.array1d[int],
  instance_material: wp.array2d[int],
  base: wp.array2d[int],
  # Out:
  shading: wp.array2d[int],
) -> None:
  """shading[w, i]: selected material bits (or scene instance bits with an empty canonical array),
  with the coat bytes of its material in world w: `coat_weight` (its IOR is the coat's own
  Fresnel law, `derive_coat_eta`) and `coat_roughness`
  (`pipeline/upload/materials.instance_materials`)."""
  w, i = wp.tid()
  bits = base[w % base.shape[0], i] & wp.static(~_COAT_BITS)
  m = instance_material[w % instance_material.shape[0], i]
  if material_shading.shape[0] > 0:
    bits = wp.int32(0)
    if m >= 0 and m < material_shading.shape[0]:
      bits = material_shading[m] & wp.static(~_COAT_BITS)
  if m >= 0 and m < gltf.shape[0]:
    k = _read(coat_weight, coat_weight_scene, w, m, wp.static(_UNIT[0]), wp.static(_UNIT[1]))
    if k > wp.float64(0.0):
      cr = _read(coat_roughness, coat_roughness_scene, w, m, wp.static(_UNIT[0]), wp.static(_UNIT[1]))
      bits = bits | (_byte(k) << wp.static(COAT_WEIGHT_SHIFT)) | (_byte(cr) << wp.static(COAT_ROUGHNESS_SHIFT))
  shading[w, i] = bits


@wp.kernel
def derive_coat_eta(
  # In:
  coat_ior: wp.array2d[float],
  coat_ior_scene: wp.array1d[float],
  gltf: wp.array1d[bool],
  # Out:
  eta: wp.array2d[float],
) -> None:
  """eta[w, m]: the coat IOR of material m's own Fresnel law in world w (Features.coat_layer):
  `coat_ior`, or 1.5 for a glTF clearcoat (`derive._layers`)."""
  w, m = wp.tid()
  ci = _read(coat_ior, coat_ior_scene, w, m, wp.static(_POSITIVE[0]), wp.static(_POSITIVE[1]))
  if gltf[m]:
    ci = wp.float64(GLTF_COAT_IOR)
  eta[w, m] = wp.float32(ci)


@wp.kernel
def derive_emission(
  # In:
  luminance: wp.array2d[float],
  luminance_scene: wp.array1d[float],
  peak_color: wp.array1d[float],
  # Out:
  emission: wp.array2d[float],
) -> None:
  """emission[w, m] = luminance x peak_color[m] (the host's float64 product of the two float32
  values, rounded once, is this float32 product; `_read`: the scene's luminance where invalid)."""
  w, m = wp.tid()
  lum = _read(luminance, luminance_scene, w, m, wp.static(_LUMINANCE[0]), wp.static(_LUMINANCE[1]))
  emission[w, m] = wp.float32(lum) * peak_color[m]


@wp.kernel
def weight_of(
  # In:
  reflectance: wp.array2d[float],
  ior_scale: wp.array1d[wp.float64],
  scene: wp.array1d[float],
  # Out:
  weight: wp.array2d[float],
) -> None:
  """The `specular_weight` 4 (r / ior_scale)^2 a live reflectance r comes from (at the 2.5 cap: the
  smallest weight reaching it; IOR scale 0, no reflectance at any weight: the scene's)."""
  w, m = wp.tid()
  s = ior_scale[m]
  if s > wp.float64(0.0):
    q = wp.float64(reflectance[w, m]) / s
    weight[w, m] = wp.float32(wp.float64(4.0) * (q * q))
  else:
    weight[w, m] = scene[m]


@wp.kernel
def luminance_of(
  # In:
  emission: wp.array2d[float],
  peak_color: wp.array1d[float],
  scene: wp.array1d[float],
  # Out:
  luminance: wp.array2d[float],
) -> None:
  """The `emission_luminance` emission / peak_color a live emission comes from (peak 0: the
  scene's)."""
  w, m = wp.tid()
  peak = peak_color[m]
  if peak > 0.0:
    luminance[w, m] = wp.float32(wp.float64(emission[w, m]) / wp.float64(peak))
  else:
    luminance[w, m] = scene[m]
