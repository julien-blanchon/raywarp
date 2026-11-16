"""Global illumination with an irradiance probe grid (DDGI / RTXGI style, gi.enabled).

One regular grid of probes over the scene's bounds (world 0's static instance boxes, planes
excluded, padded by one spacing), shared by every world, with octahedral irradiance and distance
maps per probe (`probes.py`: the grid, the maps' layout and the lookups).

Placement (`volume.ProbeVolume.place`, static geometry only, once): a relocation pass (fixed
rays) moves probes that see mostly back faces towards the closest back face (at most 0.45
spacing) and away from front faces closer than 0.1 spacing; probes that still see > 25 % back
faces are disabled. Then the ambient cube's walls and probe sight (`cube.py`) and the static bake:
`volume.STATIC_RAYS` geometry-only rays per probe for the distance moments (frozen afterwards).

Update (`volume.ProbeVolume.update`, kernels in `kernels.py`): `rays` rays per probe (spherical
Fibonacci, randomly rotated each update) traced over world 0's static geometry (dynamic and
per-world instances hidden); hits are shaded with the megakernel's direct lighting (same light
loop: light grid, shadow budget, area lights, shadows), the volume's irradiance at the hit read
from the ambient cube (one more bounce per update), and emission; misses return the environment
(or the ambient fill without environment lighting). Translucent layers are composited as by
camera rays. The ray results are then blended into the irradiance maps with hysteresis. The bake
(`volume.ProbeVolume.bake`) traces one update's rays and keeps their hits (light without the
bounce, the bounce's weight, where it is read); each further bake update only lights those hits
again from the previous one's irradiance (one bounce more, no ray traced, no light loop). The
bake and the periodic updates (`schedule.Schedule`) run entirely on the device: graph-capturable.

Shading: after each update the probes' lookup is resampled into an ambient-cube 3D texture
(`cube.py`: validity, dilation, wall-aware filtering) that camera, reflection and probe-ray hits
read in a few fetches (a walls fetch and three filtered fetches: no light through walls of any
thickness).
The development repository's GI debug view draws the volume's lookups, probes and sky
visibility over a rendered frame.

Groups of worlds (`groups.py`): worlds whose radiance inputs (lights, materials, instance
appearance, environment) are equal share one set of irradiance maps; the probe grid, placement
and distance maps (static geometry) are shared by every group. From the data: one group when the
worlds agree (the volume of a scene without randomisation), K stacked sets for a few variants
(each baked with its representative world's inputs; a world's switch to another known variant
moves it to that group). Continuous randomisation (most worlds distinct, or new inputs met after
the bake: one re-bake) gets one shared volume tinted per world (`relight.py`: a coarse
directional RGB ratio field per world from fixed probe rays shaded in that world, recomputed in
the frame a world's inputs change).
"""

from __future__ import annotations

from raywarp.core.shade.gi.dynamic import StaticLights
from raywarp.core.shade.gi.groups import Groups, digest, gi_slot, per_world_inputs
from raywarp.core.shade.gi.kernels import RayHit, back_face, make_relight_kernel, make_trace_kernel, ray_hit
from raywarp.core.shade.gi.probes import MISS_DISTANCE, Volume, beyond, empty_volume, irradiance
from raywarp.core.shade.gi.rebakes import GIRebakes, Reason, RebakeLog
from raywarp.core.shade.gi.relight import Relight
from raywarp.core.shade.gi.schedule import ProbeKernels, Schedule, relight_kernel_of, relit_inputs
from raywarp.core.shade.gi.slots import assign as assign_slots
from raywarp.core.shade.gi.volume import (
  GiRays,
  PlacementRays,
  ProbeInputs,
  ProbeVolume,
  cluster_bounds,
  free_path,
  scene_bounds,
  target_spacing,
)

__all__ = [
  "MISS_DISTANCE",
  "GIRebakes",
  "GiRays",
  "Groups",
  "PlacementRays",
  "ProbeInputs",
  "ProbeKernels",
  "ProbeVolume",
  "RayHit",
  "Reason",
  "RebakeLog",
  "Relight",
  "Schedule",
  "StaticLights",
  "Volume",
  "assign_slots",
  "back_face",
  "beyond",
  "cluster_bounds",
  "digest",
  "empty_volume",
  "free_path",
  "gi_slot",
  "irradiance",
  "make_relight_kernel",
  "make_trace_kernel",
  "per_world_inputs",
  "ray_hit",
  "relight_kernel_of",
  "relit_inputs",
  "scene_bounds",
  "target_spacing",
]
