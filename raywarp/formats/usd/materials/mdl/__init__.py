"""MDL materials through the NVIDIA MDL SDK (bundled: `raywarp._native.mdl`): `SdkBaker`, the
default `context.MaterialBaker`.

`sdk` (the bindings, typed; the process's SDK instance), `load` (search paths, modules, authored
arguments, compile and distill to `transmissive_pbr`), `steps` (a conversion's steps, apart for
the parallel prefetch), `dag` and `ops` (distilled expressions as `values.py` expressions),
`detail` (a base + detail normal split back into two maps), `keys` (what a bake depends on),
`bake` (the SDK's CPU baker, disk cache), `writer` and `surface` (distilled lobes -> the canonical
material), `memo` (the disk cache of conversions), `baker` (per-shader conversion and caching).
`library/`: the Omniverse OmniPBR / OmniGlass / OmniSurface sources (BSD-3, library/NOTICE.md),
searched after the stage's own.
"""

from raywarp.formats.usd.materials.mdl.baker import SdkBaker as SdkBaker
