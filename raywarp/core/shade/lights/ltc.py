"""Linearly transformed cosines (Heitz, Dupuy, Hill, Neubelt 2016) of the area lights: the fitted
tables and the clamped-cosine integrals of the light shapes.

The 64 x 64 table (`raywarp/data/ltc_ggx.npz`, fitted to the renderer's GGX by
tools/fit_ltc.py) gives, per (view angle, roughness), the matrix M^-1 that turns the GGX
lobe into a clamped cosine, plus the lobe's norm and Fresnel term. Applying M^-1 to a light's shape
makes the BRDF integral a form factor with a closed form:

* `quad_integral`: a polygon (rect, the tube's silhouette rect), exact horizon clipping;
* `ellipse_integral`: an ellipse (disk, the sphere's facing disk), Heitz & Hill 2017's form
  factor, horizon-clipped as the equivalent sphere (`sphere_horizon`, Snyder's closed form).

`Frame` is the shading point's frame of the evaluation (shared by every light of the point).
"""

from __future__ import annotations

import functools
import importlib.resources
from typing import Final, Protocol

import numpy as np
import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.shade.materials import brdf, texture
from raywarp.core.shade.materials.material import COAT_F0, Lobes

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The tables' host layout: (rows: sqrt(1 - cos theta_v), columns: perceptual roughness, channels).
type LtcMinv = np.ndarray[[int, int, 4], np.dtype[np.float32]]
type LtcMag = np.ndarray[[int, int, 2], np.dtype[np.float32]]

# The fitted table (tools/fit_ltc.py) shipped in the package data, and its size per axis.
LTC_FILE: Final = "ltc_ggx.npz"
LTC_SIZE: Final = 64
# Ellipses smaller than this (|v1 x v2| / |c|^2, ~ the squared angular radius: ~3.6 degrees) take
# the small-source form factor: the cubic of `ellipse_integral` is ill-conditioned for them (its
# coefficients grow as (distance / size)^2), and CUDA's fast math made a 2 cm MuJoCo bulb at 1 m
# 4.9 % too bright.
SMALL_ELLIPSE: Final = 4.0e-3


@functools.cache
def _ltc_arrays() -> tuple[LtcMinv, LtcMag]:
  """(M^-1 (m00, m02, m11, m20), (norm, fresnel)) tables, (LTC_SIZE, LTC_SIZE, 4 / 2) float32."""
  with importlib.resources.as_file(importlib.resources.files("raywarp.data") / LTC_FILE) as path, np.load(path) as t:
    return np.ascontiguousarray(t["minv"], np.float32), np.ascontiguousarray(t["mag"], np.float32)


def ltc_textures(device: wp.Device) -> tuple[wp.Texture2D, wp.Texture2D]:
  """(M^-1 (m00, m02, m11, m20), (norm, fresnel)) bilinear textures on `device` (cached).

  Rows: sqrt(1 - cos theta_v), columns: perceptual roughness (texel centres at the fitted values).
  """
  minv = bake_cache.table_texture("ltc_minv", lambda: _ltc_arrays()[0], device)
  return minv, bake_cache.table_texture("ltc_mag", lambda: _ltc_arrays()[1], device)


@wp.struct
class Frame:
  """Shading-point frame of the LTC evaluation (shared by every light of the point).

  `t1, t2, n`: tangent frame with the view in the (t1, n) plane; `minv`: the LTC's M^-1 in that
  frame; `spec`: specular weight (F0 norm + (F90 - F0) fresnel) x multiscatter compensation.
  """

  t1: wp.vec3
  t2: wp.vec3
  n: wp.vec3
  minv: wp.mat33
  spec: wp.vec3
  # Clearcoat (Features.clearcoat): the coat lobe's M^-1 and weight (coat x its LTC albedo).
  coat_minv: wp.mat33
  coat_spec: float
  # The specular lobe's own axes about the anisotropic bent normal (Features.specular_anisotropy:
  # an approximation, the LTC fits being isotropic; the base's (t1, t2, n) for an isotropic lobe)
  # and the coat's about its own normal (Features.coat_normal: set, and read, only where a layered
  # coat is on).
  spec_t1: wp.vec3
  spec_t2: wp.vec3
  spec_n: wp.vec3
  coat_t1: wp.vec3
  coat_t2: wp.vec3
  coat_n: wp.vec3
  # The view (Features.fuzz: the fuzz lobe's evaluation toward an area light).
  view: wp.vec3


class LtcTables(Protocol):
  """The light struct fields of the LTC tables (`ltc_textures`, area-light structs)."""

  ltc_minv: wp.Texture2D
  ltc_mag: wp.Texture2D


@wp.func
def view_axes(
  # In:
  normal: wp.vec3,
  view: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(t1, t2) of an LTC frame about `normal` with the view in its (t1, n) plane."""
  t1 = view - normal * wp.dot(view, normal)
  if wp.length_sq(t1) < 1.0e-10:
    b1, _b2 = texture.orthonormal_basis(normal)
    t1 = b1
  t1 = wp.normalize(t1)
  return t1, wp.cross(normal, t1)


@wp.func
def with_axes(
  # In:
  fr: Frame,
  t1: wp.vec3,
  t2: wp.vec3,
  n: wp.vec3,
) -> Frame:
  """`fr` with the axes (t1, t2, n): a lobe's own frame (`local_shape` reads the axes)."""
  out = fr
  out.t1 = t1
  out.t2 = t2
  out.n = n
  return out


@specialize
def make_frame_func(
  *,
  specular_curve: bool,
  clearcoat: bool,
  specular_weight: bool = False,
  anisotropy: bool = False,
  coat_normal: bool = False,
  coat_layer: bool = False,
  fuzz: bool = False,
  bound: int = 0,
):
  """make_frame(lit, lobes, normal, view, nov) -> Frame (`lit`: any light struct with the LTC
  tables).

  `specular_curve` / `specular_weight`: F90 from the lobes; `clearcoat`: the coat lobe's LTC too;
  `anisotropy`, `coat_normal`, `coat_layer`, `fuzz`: the layered lobes' Features (the specular
  lobe's bent axes, the coat's own normal and IOR, the view for the fuzz lobe). `bound`
  (Features.bound): the coat and layers class bits read the build word's material copy
  (`Lobes.bits`).
  """
  gate = toggles.bound_on(bound)

  @wp.func
  @device_generic
  def make_frame[L: LtcTables](
    # Model:
    lit: L,
    # In:
    lobes: Lobes,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ):
    """The LTC frame of a shading point (lobe tables looked up at its roughness and view angle)."""
    fr = Frame()
    layered = False
    if wp.static(anisotropy or coat_normal or coat_layer or fuzz):
      layered = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
    t1, t2 = view_axes(normal, view)
    fr.t1 = t1
    fr.t2 = t2
    fr.n = normal
    scale = wp.static(float(LTC_SIZE - 1) / float(LTC_SIZE))
    bias = wp.static(0.5 / float(LTC_SIZE))
    uv = wp.vec2(lobes.perceptual_roughness * scale + bias, wp.sqrt(wp.max(1.0 - nov, 0.0)) * scale + bias)
    m = wp.texture_sample(lit.ltc_minv, uv, dtype=wp.vec4)
    mag = wp.texture_sample(lit.ltc_mag, uv, dtype=wp.vec2)
    fr.minv = wp.mat33(m[0], 0.0, m[1], 0.0, m[2], 0.0, m[3], 0.0, 1.0)
    f0 = lobes.f0
    if wp.static(specular_curve or specular_weight):
      fr.spec = wp.cw_mul(f0 * mag[0] + (lobes.f90 - f0) * mag[1], lobes.energy_comp)
    else:
      fr.spec = wp.cw_mul(f0 * mag[0] + (wp.vec3(1.0) - f0) * mag[1], lobes.energy_comp)
    if wp.static(fuzz):
      fr.view = view
    if wp.static(anisotropy):
      fr.spec_t1 = fr.t1
      fr.spec_t2 = fr.t2
      fr.spec_n = normal
      if layered and lobes.anisotropy > 0.0:
        bent = brdf.anisotropic_bent_normal(
          view, normal, lobes.tangent, lobes.bitangent, lobes.anisotropy, lobes.perceptual_roughness
        )
        s1, s2 = view_axes(bent, view)
        fr.spec_t1 = s1
        fr.spec_t2 = s2
        fr.spec_n = bent
    if wp.static(clearcoat):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT)) and lobes.coat > 0.0:
        uvc = wp.vec2(lobes.coat_roughness * scale + bias, uv[1])
        f0c = wp.static(COAT_F0)
        if wp.static(coat_normal or coat_layer):
          if layered:
            f0c = lobes.coat_f0
            uvc = wp.vec2(uvc[0], wp.sqrt(wp.max(1.0 - lobes.coat_nov, 0.0)) * scale + bias)
        if wp.static(coat_normal):
          if layered:
            c1, c2 = view_axes(lobes.coat_normal, view)
            fr.coat_t1 = c1
            fr.coat_t2 = c2
            fr.coat_n = lobes.coat_normal
        mc = wp.texture_sample(lit.ltc_minv, uvc, dtype=wp.vec4)
        magc = wp.texture_sample(lit.ltc_mag, uvc, dtype=wp.vec2)
        fr.coat_minv = wp.mat33(mc[0], 0.0, mc[1], 0.0, mc[2], 0.0, mc[3], 0.0, 1.0)
        fr.coat_spec = lobes.coat * (f0c * magc[0] + (1.0 - f0c) * magc[1])
    return fr

  return make_frame


@wp.func
def to_local(
  # In:
  fr: Frame,
  v: wp.vec3,
) -> wp.vec3:
  """World vector v in the frame's tangent space (t1, t2, n)."""
  return wp.vec3(wp.dot(v, fr.t1), wp.dot(v, fr.t2), wp.dot(v, fr.n))


@wp.func
def edge_integral(
  # In:
  a: wp.vec3,
  b: wp.vec3,
) -> float:
  """Z of the edge integral of unit vectors a -> b, over 2 pi (ltc_code IntegrateEdgeVec)."""
  x = wp.dot(a, b)
  y = wp.abs(x)
  fa = 0.8543985 + (0.4965155 + 0.0145206 * y) * y
  fb = 3.4175940 + (4.1616724 + y) * y
  v = fa / fb
  theta_sintheta = wp.where(x > 0.0, v, 0.5 / wp.sqrt(wp.max(1.0 - x * x, 1.0e-7)) - v)
  return (a[0] * b[1] - a[1] * b[0]) * theta_sintheta


@wp.func
def _unit(
  # In:
  v: wp.vec3,
) -> wp.vec3:
  """V normalised (0 for a zero vector)."""
  return v / wp.max(wp.length(v), 1.0e-20)


@wp.func
def clipped_edge(
  # In:
  a: wp.vec3,
  b: wp.vec3,
  ua: wp.vec3,
  ub: wp.vec3,
) -> tuple[float, int, wp.vec3]:
  """(integral of the part of edge a -> b above the horizon, 1 exits / 2 enters / 0, unit crossing).

  `ua` / `ub`: a and b normalised (by the caller, once per vertex: the edges share them).
  """
  if a[2] >= 0.0 and b[2] >= 0.0:
    return edge_integral(ua, ub), 0, ua
  if a[2] < 0.0 and b[2] < 0.0:
    return 0.0, 0, ua
  m = a + (b - a) * (a[2] / (a[2] - b[2]))
  um = _unit(wp.vec3(m[0], m[1], 0.0))
  if a[2] >= 0.0:
    return edge_integral(ua, um), 1, um
  return edge_integral(um, ub), 2, um


@wp.func
def quad_integral(
  # In:
  p0: wp.vec3,
  p1: wp.vec3,
  p2: wp.vec3,
  p3: wp.vec3,
) -> float:
  """Clamped-cosine integral (form factor) of a quad (shading point at the origin, normal +z).

  Exact horizon clipping: the clipped polygon is the parts of the edges above z = 0 plus the
  horizon segment from the exit to the entry point (a convex quad crosses at most twice).
  """
  # Each vertex normalised once (the edges share them).
  u0 = _unit(p0)
  u1 = _unit(p1)
  u2 = _unit(p2)
  u3 = _unit(p3)
  s = float(0.0)
  exit_p = wp.vec3(0.0)
  entry_p = wp.vec3(0.0)
  crossings = int(0)
  e, flag, m = clipped_edge(p0, p1, u0, u1)
  s += e
  if flag == 1:
    exit_p = m
  elif flag == 2:
    entry_p = m
  crossings += flag
  e, flag, m = clipped_edge(p1, p2, u1, u2)
  s += e
  if flag == 1:
    exit_p = m
  elif flag == 2:
    entry_p = m
  crossings += flag
  e, flag, m = clipped_edge(p2, p3, u2, u3)
  s += e
  if flag == 1:
    exit_p = m
  elif flag == 2:
    entry_p = m
  crossings += flag
  e, flag, m = clipped_edge(p3, p0, u3, u0)
  s += e
  if flag == 1:
    exit_p = m
  elif flag == 2:
    entry_p = m
  crossings += flag
  if crossings == 3:
    # The horizon segment from the exit to the entry crossing (unit vectors in z = 0): its signed
    # angle over 2 pi, exactly (the edge fit's 1 / sin(theta) loses float32 precision when the
    # crossings are nearly opposite: a glossy receiver millimetres from the light's plane).
    s += wp.atan2(exit_p[0] * entry_p[1] - exit_p[1] * entry_p[0], wp.dot(exit_p, entry_p)) * wp.static(0.5 / wp.pi)
  return wp.abs(s)


@wp.func
def sphere_horizon(
  # In:
  cos_theta: float,
  form_factor: float,
) -> float:
  """Horizon-clipped form factor of a sphere of unclipped form factor sin^2(sigma).

  At elevation cos_theta (Snyder 1996, as Frostbite's illuminanceSphereOrDisk, divided by pi).
  """
  sin2 = wp.clamp(form_factor, 1.0e-7, 1.0)
  sin_sigma = wp.sqrt(sin2)
  if cos_theta >= sin_sigma:
    return sin2 * cos_theta
  if cos_theta <= -sin_sigma:
    return 0.0
  sin_theta = wp.sqrt(wp.max(1.0 - cos_theta * cos_theta, 1.0e-12))
  x = wp.sqrt(wp.max(1.0 / sin2 - 1.0, 1.0e-12))
  y = wp.clamp(-x * cos_theta / sin_theta, -1.0, 1.0)
  sin_theta_sqrt_y = sin_theta * wp.sqrt(wp.max(1.0 - y * y, 0.0))
  e = (cos_theta * wp.acos(y) - x * sin_theta_sqrt_y) * sin2 + wp.atan(sin_theta_sqrt_y / x)
  return wp.max(e, 0.0) * wp.static(1.0 / np.pi)


@wp.func
def solve_cubic(
  # In:
  c0: float,
  c1: float,
  c2: float,
) -> wp.vec3:
  """Real roots of c0 + c1 x + c2 x^2 + x^3 (three real roots), the smallest second.

  Blinn 2007, "How to solve a cubic equation", as ltc_code's SolveCubic (monic: its c3 is 1 here).
  """
  # Divide the middle coefficients by three.
  cx = c0
  cy = c1 / 3.0
  cz = c2 / 3.0
  a = float(1.0)
  b = cz
  c = cy
  d = cx
  delta_x = -cz * cz + cy
  delta_y = -cy * cz + cx
  delta_z = cz * cx - cy * cy
  disc = 4.0 * delta_x * delta_z - delta_y * delta_y
  sq = wp.sqrt(wp.max(disc, 0.0))
  # Algorithm A (largest root).
  d_a = -2.0 * b * delta_x + delta_y
  theta = wp.atan2(sq, -d_a) / 3.0
  r = 2.0 * wp.sqrt(wp.max(-delta_x, 0.0))
  x1a = r * wp.cos(theta)
  x3a = r * wp.cos(theta + wp.static(2.0 / 3.0 * np.pi))
  xl = wp.where(x1a + x3a > 2.0 * b, x1a, x3a)
  xlc_x = xl - b
  xlc_y = a
  # Algorithm D (smallest root).
  d_d = -d * delta_y + 2.0 * c * delta_z
  theta = wp.atan2(d * sq, -d_d) / 3.0
  r = 2.0 * wp.sqrt(wp.max(-delta_z, 0.0))
  x1d = r * wp.cos(theta)
  x3d = r * wp.cos(theta + wp.static(2.0 / 3.0 * np.pi))
  xs = wp.where(x1d + x3d < 2.0 * c, x1d, x3d)
  xsc_x = -d
  xsc_y = xs + c
  e = xlc_y * xsc_y
  f = -xlc_x * xsc_y - xlc_y * xsc_x
  g = xlc_x * xsc_x
  xmc_x = c * f - b * g
  xmc_y = -b * f + c * e
  r0 = xsc_x / xsc_y
  r1 = xmc_x / xmc_y
  r2 = xlc_x / xlc_y
  if r0 < r1 and r0 < r2:
    return wp.vec3(r1, r0, r2)
  if r2 < r0 and r2 < r1:
    return wp.vec3(r0, r2, r1)
  return wp.vec3(r0, r1, r2)


@wp.func
def ellipse_integral(
  # In:
  c: wp.vec3,
  v1: wp.vec3,
  v2: wp.vec3,
) -> float:
  """Clamped-cosine integral of the ellipse c + cos(t) v1 + sin(t) v2.

  Shading point at the origin, normal +z: exact form factor of the elliptical cone, horizon
  clipped as the equivalent sphere (Heitz & Hill 2017, ltc_code `ltc_disk`). Below SMALL_ELLIPSE:
  the small-source limit, solid angle x clamped cosine / pi (relative error under SMALL_ELLIPSE).
  """
  n = wp.cross(v1, v2)
  c_sq = wp.dot(c, c)
  if wp.length(n) < wp.static(SMALL_ELLIPSE) * c_sq:
    c_len = wp.sqrt(c_sq)
    return wp.abs(wp.dot(n, c)) / (c_sq * c_len) * wp.max(c[2] / c_len, 0.0)
  d11 = wp.dot(v1, v1)
  d22 = wp.dot(v2, v2)
  d12 = wp.dot(v1, v2)
  a = float(0.0)
  b = float(0.0)
  w1 = v1
  w2 = v2
  if wp.abs(d12) / wp.sqrt(wp.max(d11 * d22, 1.0e-30)) > 1.0e-4:
    tr = d11 + d22
    det = wp.sqrt(wp.max(-d12 * d12 + d11 * d22, 0.0))
    u = 0.5 * wp.sqrt(wp.max(tr - 2.0 * det, 0.0))
    v = 0.5 * wp.sqrt(tr + 2.0 * det)
    e_max = (u + v) * (u + v)
    e_min = (u - v) * (u - v)
    if d11 > d22:
      w1 = v1 * d12 + v2 * (e_max - d11)
      w2 = v1 * d12 + v2 * (e_min - d11)
    else:
      w1 = v2 * d12 + v1 * (e_max - d22)
      w2 = v2 * d12 + v1 * (e_min - d22)
    a = 1.0 / e_max
    b = 1.0 / wp.max(e_min, 1.0e-30)
    w1 = wp.normalize(w1)
    w2 = wp.normalize(w2)
  else:
    a = 1.0 / d11
    b = 1.0 / d22
    w1 = v1 * wp.sqrt(a)
    w2 = v2 * wp.sqrt(b)
  w3 = wp.cross(w1, w2)
  if wp.dot(c, w3) < 0.0:
    w3 = -w3
  l = wp.dot(w3, c)
  if l <= 1.0e-12:
    return 0.0
  x0 = wp.dot(w1, c) / l
  y0 = wp.dot(w2, c) / l
  a = a * l * l
  b = b * l * l
  c0 = a * b
  c1 = a * b * (1.0 + x0 * x0 + y0 * y0) - a - b
  c2 = 1.0 - a * (1.0 + x0 * x0) - b * (1.0 + y0 * y0)
  roots = solve_cubic(c0, c1, c2)
  e1 = roots[0]
  e2 = roots[1]
  e3 = roots[2]
  avg = wp.vec3(a * x0 / (a - e2), b * y0 / (b - e2), 1.0)
  avg = wp.normalize(w1 * avg[0] + w2 * avg[1] + w3 * avg[2])
  l1 = wp.sqrt(wp.max(-e2 / e3, 0.0))
  l2 = wp.sqrt(wp.max(-e2 / e1, 0.0))
  form_factor = l1 * l2 / wp.sqrt((1.0 + l1 * l1) * (1.0 + l2 * l2))
  return sphere_horizon(avg[2], form_factor)
