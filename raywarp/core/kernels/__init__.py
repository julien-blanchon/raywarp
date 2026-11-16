"""The render megakernel, assembled per `Features` from device-func factories.

  abi            the kernel ABI: `Features`, the output word, the kernel input structs, tables,
                 queries, toggles and counters
  generic        the general-kernel specialisation; `native`: the native builtins (BVH node reads,
                 OptiX intrinsics)
  lighting       direct lighting: one light's BRDF, the light loops (shadow rays:
                 `shade.shadows.visibility`)
  ambient        the light from everything but the lights (environment, fill, GI probes)
  reflections    traced glossy reflections
  surface        a hit's shading: ambient, glossy reflection, lights, emission
  layer          one camera-ray layer shaded, its mirror reflection blended in
  first_hit      the first-hit outputs beyond depth / seg (hit and surface outputs)
  center         pixel-center observations
  display        the display color's writes (packed RGBA, channels-first layouts)
  geometry       the geometry-only body and kernel (no shaded output)
  pixel          the per-pixel body and the software megakernels
  probe          the GI probe rays' shading and kernel

Modules are imported by path, like the other core packages.
"""

from __future__ import annotations
