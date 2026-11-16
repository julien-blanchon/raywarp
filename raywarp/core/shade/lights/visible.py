"""Camera-visible area lights: camera rays see the emitting surface of rect, disk, sphere and
cylinder lights flagged visible at its radiance, as a path tracer does (UsdLux lights have no
geometry of their own, so a light panel is seen only through this).

`Lights.visible` lists the lights (`upload/punctual.visible_lights`: every area light with a size
flagged `Light.visible_in_primary_ray`, whatever `lighting.area_lights` and its ShapingAPI, except
the lights standing for an emissive instance, whose own geometry shows their emission). UsdLux
lights are unseen unless flagged, as Omniverse RTX renders them (`visibleInPrimaryRay` false by
default). A camera ray, and its continuation through
translucent layers, is intersected with each of them analytically before the surface it hit
(`closest`, inlined, reading the light arrays): a run-time loop over `Lights.nvisible` (0:
nothing traced, nothing compiled per scene; `lighting.visible_lights` off sets it to 0).

What the camera sees is the light's uniform surface radiance, the one its point and LTC forms
light the scene with (`pack.area_radiance`: color x intensity / projected area; with an IES
tilt when the light has one), times its ShapingAPI cone toward the camera (`punctual.cone_factor`).
Rects and disks emit from their front face only (along `Light.dir`); seen from behind or edge on
they are transparent. Cylinders are their side surface (no end caps, as their LTC form). The IES
profile's angular shape is not drawn on the surface. An inactive light (`Light.active` False) is
not seen.

The lights stay out of every other ray: reflections already carry them through the shading
point's specular light term (LTC or the point lobe, `kernels/reflections.py`), GI probes through
their light loop, and shadow rays must reach them. The aux outputs (depth, normals, segmentation)
keep the surface behind a light: it is not geometry.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.kernels.abi.tables import batched
from raywarp.core.shade.lights import area, punctual
from raywarp.core.shade.lights.pack import area_radiance
from raywarp.core.shade.lights.punctual import Light

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Distance `_distance` reports when the ray misses the light.
NO_HIT: Final = -1.0
# Rays this close to a rect's or disk's plane (|cos|), and a cylinder's axis (sin^2), miss it.
_GRAZING: Final = 1.0e-8


@wp.func
def _plane_distance(
  # Model:
  light: Light,
  shape: int,
  # In:
  origin: wp.vec3,
  direction: wp.vec3,
) -> float:
  """Distance to a rect's or disk's front face along the ray, NO_HIT outside it or from behind."""
  facing = wp.dot(direction, light.dir)
  if facing > -wp.static(_GRAZING):
    return wp.static(NO_HIT)
  t = wp.dot(light.pos - origin, light.dir) / facing
  q = origin + direction * t - light.pos
  if shape == wp.static(area.SHAPE_RECT):
    if wp.abs(wp.dot(q, light.axis_x)) > light.half[0] or wp.abs(wp.dot(q, light.axis_y)) > light.half[1]:
      return wp.static(NO_HIT)
  elif wp.length_sq(q) > light.radius * light.radius:
    return wp.static(NO_HIT)
  return t


@wp.func
def _sphere_distance(
  # Model:
  light: Light,
  # In:
  origin: wp.vec3,
  direction: wp.vec3,
) -> float:
  """Distance to a sphere light's surface along the ray (its far side from inside), NO_HIT on a
  miss."""
  oc = origin - light.pos
  b = wp.dot(oc, direction)
  disc = b * b - (wp.dot(oc, oc) - light.radius * light.radius)
  if disc < 0.0:
    return wp.static(NO_HIT)
  s = wp.sqrt(disc)
  return wp.where(-b - s > 0.0, -b - s, -b + s)


@wp.func
def _tube_distance(
  # Model:
  light: Light,
  # In:
  origin: wp.vec3,
  direction: wp.vec3,
) -> float:
  """Distance to a cylinder light's side surface (axis `Light.axis_x`, half length `half[0]`)
  along the ray (its far side from inside), NO_HIT on a miss."""
  axis = light.axis_x
  oc = origin - light.pos
  dp = direction - axis * wp.dot(direction, axis)
  op = oc - axis * wp.dot(oc, axis)
  a = wp.dot(dp, dp)
  b = wp.dot(op, dp)
  disc = b * b - a * (wp.dot(op, op) - light.radius * light.radius)
  if a < wp.static(_GRAZING) or disc < 0.0:
    return wp.static(NO_HIT)
  s = wp.sqrt(disc)
  t = wp.where(-b - s > 0.0, (-b - s) / a, (-b + s) / a)
  if wp.abs(wp.dot(oc + direction * t, axis)) > light.half[0]:
    return wp.static(NO_HIT)
  return t


@wp.func
def _distance(
  # Model:
  light: Light,
  shape: int,
  # In:
  origin: wp.vec3,
  direction: wp.vec3,
) -> float:
  """Distance along the ray to the emitting surface of `light` of LightShape `shape`, NO_HIT (< 0)
  when the ray misses it or starts past it."""
  if shape == wp.static(area.SHAPE_SPHERE):
    return _sphere_distance(light, origin, direction)
  if shape == wp.static(area.SHAPE_TUBE):
    return _tube_distance(light, origin, direction)
  return _plane_distance(light, shape, origin, direction)


# Inlined: with it out of line (`__noinline__`, an ABI call in the OptiX raygen: 664 instructions
# outside the entry), three scenes with visible lights (a furnished room, a small appliance studio,
# a large office) rendered 26 %, 9 % and 2 % slower than inlined, the raygen otherwise within 0.2 %
# of its instructions (fast, RTX PRO 6000, 512^2 x 512 worlds); a scene without one pays
# the branch on `nvisible` (`kernels/pixel._make_see_light`).
@wp.func
def closest(
  # Model:
  table: wp.array2d[Light],
  visible: wp.array[wp.vec2i],
  nvisible: int,
  # In:
  worldid: int,
  origin: wp.vec3,
  direction: wp.vec3,
  max_dist: float,
) -> wp.vec4:
  """(the radiance the unit ray from `origin` sees on a light's surface before `max_dist`, 1 when
  it sees one else 0) (module doc), from the arrays it reads (`Lights.lights`, `Lights.visible`
  and its count)."""
  best = max_dist
  seen = float(0.0)
  radiance = wp.vec3(0.0)
  for k in range(nvisible):
    entry = visible[k]
    light = batched(table, worldid, entry[0])
    if light.active:
      t = _distance(light, entry[1], origin, direction)
      if t > 0.0 and t < best:
        best = t
        seen = 1.0
        emitted = area_radiance(entry[1], light.radius, light.half, light.color, light.intensity)
        radiance = emitted
        if light.cos_outer > wp.static(punctual.NO_CONE):
          radiance = emitted * punctual.cone_factor(light, -wp.dot(direction, light.dir))
  return wp.vec4(radiance[0], radiance[1], radiance[2], seen)
