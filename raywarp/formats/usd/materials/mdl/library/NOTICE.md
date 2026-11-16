# NVIDIA Omniverse Core Materials 2.1.4 (BSD-3-Clause)

This directory holds the MDL library that Omniverse RTX (Kit, ovrtx 0.5.1) loads by default. It
is copied unmodified from the ovrtx 0.5.1 wheel, `ovrtx/bin/library/mdl/`
(https://pypi.nvidia.com/ovrtx/; `tools/mdl_corpus/manifest.json`, source `kit_core`). The wheel
itself carries NVIDIA's software licence, but every file here is BSD-3-Clause: each header has
`SPDX-License-Identifier: BSD-3-Clause`, and the package licence is `LICENSE.md`, copied from the
wheel's `PACKAGE-LICENSES/omni-core-materials-LICENSE.md`. The Apache-2.0 MaterialX part of that
package (`materialx/`) is not included.

| directory | modules | search root |
| --- | --- | --- |
| `Base/` | OmniPBR (and _ClearCoat, _Opacity, _ClearCoat_Opacity, Base), OmniGlass (and _Opacity), OmniSurface, OmniSurfaceLite, OmniSurfaceBlend, OmniSurfacePresets, OmniHair (and Presets), OmniEmissive, SimPBR (and _Model, _Translucent), baking_annotations | yes |
| `Ue4/` | OmniUe4Base, OmniUe4Translucent, OmniUe4Subsurface, OmniUe4Function (and Extension17): Unreal exports | yes |
| `mdl/OmniSurface/` | OmniSurfaceBase, OmniSurfaceLiteBase, OmniSurfaceBlendBase, OmniHairBase, OmniShared, OmniImage (`::OmniSurface::*`) | `mdl/` |
| `mdl/nvidia/` | support_definitions, aux_definitions (`::nvidia::*`; `::nvidia::core_definitions` is the MDL SDK wheel's) | `mdl/` |
| `mdl/materialx/` | MaterialX 1.39.5's MDL implementation (`::materialx::core`, `stdlib_*`, `pbrlib_*`, `noise`, `sampling`, `hsv`, `flake`, `hextile`): the modules `MaterialXGenMdl` output imports (Apache-2.0, see below) | `mdl/` |

The USD adapter (`warp_pbr/usd/mdl/load.py`) searches these roots after the stage's own layer
directories and `MDL_USER_PATH` / `MDL_SYSTEM_PATH`. A stage that ships its own copy of a module
uses that copy. When a module resolves here, the load report says so
(`mdl.library_module:<file>`).

## MaterialX 1.39.5 MDL modules (Apache-2.0)

`mdl/materialx/*.mdl` are copied unmodified from the MaterialX repository at tag `v1.39.5`,
`source/MaterialXGenMdl/mdl/materialx/` (https://github.com/AcademySoftwareFoundation/MaterialX,
Apache License 2.0, Copyright Contributors to the MaterialX Project / NVIDIA Corporation, as each
file's header states). They match the `materialx` Python package version the USD adapter pins:
the MDL its shader generator writes imports them. Update both together.
