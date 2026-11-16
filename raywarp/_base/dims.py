"""The one list of what the renderer's array dimensions are called, and the shaped array aliases
built from them (type-only: never imported at runtime).

Shapes are checked by Pyrefly through pyrefly-warp-stubs (`wp.array[dtype, [dims]]`). A named
dimension is a PEP 695 type parameter bound to `IntVar`, declared by the function or class that
uses it, so that two arguments sharing a name must agree:

```python
from __future__ import annotations  # every module with shaped annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # shape_extensions and this module exist only for the type checker
  from shape_extensions import IntVar

  from raywarp._base.dims import Table


def per_world[NWorld: IntVar, N: IntVar](
  a: wp.array[wp.vec3, [NWorld, N]],
) -> wp.array[wp.vec3, [N]]: ...
def light_rows[NRow: IntVar, NLight: IntVar](radius: Table[float, NRow, NLight]) -> None: ...
```

Rules: the names below are the only names of scene / renderer dimensions (a new one is added
here first, with its meaning; helpers generic over any length use single letters, `N`, `C`);
device code (`@wp.kernel`, `@wp.func`, `@wp.struct`) keeps plain Warp annotations
(`wp.array2d[wp.vec3]`): Warp evaluates them, and the typed host launcher carries the shapes.

Dimensions:

* `NWorld`: worlds of a Renderer. `NCam`: its cameras. `H`, `W`: image height and width.
* `NRow`: rows of a batched table, 1 (shared by every world) or `NWorld`; kernels read row
  `min(worldid, NRow - 1)` (`kernels/abi/tables.table_row`).
* `NInstance`: instances of a GpuScene (rendered shapes). `NColumn`: per-world pose columns of a
  Renderer (`sharing.Layout.columns`: every instance, or the per-world ones of a compact layout).
  `NShared`: shared instances (`Layout.shared`).
* `NMaterial`, `NLight`, `NMesh`, `NTexture`: materials, lights, meshes, textures of a GpuScene.
* `NVert`, `NFace`: vertices and triangles of every mesh of a GpuScene, concatenated (`Meshes`).
* `NSceneCam`: cameras authored in the source (`GpuScene.cameras`). `NFlexVert`: deformable
  vertices. `NDeformable`: deformables (`core/pipeline/gpu_scene/deformables.py`); `NFlexPrim`:
  their scene-BVH primitives (a capsule per segment, a box per surface).
* `NImage`: images of a channels-first batch, `NWorld * NCam` (`render_outputs.Planes`).
* `NPixel`: camera pixels of a Renderer's frame, flattened (`NCam * H * W`: temporal buffers).
* `NProbe`, `NRay`, `NTexel`: GI probes, rays per probe and update, texels of a probe map.
* `NEnv`: environment maps of a GpuScene. `NZ`, `NY`, `NX`, `C`: cells of a 3D grid texture and its
  channels.

Aliases: shaped type aliases are PEP 695 `type` statements (required by Pyrefly) with the
dimensions as parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
  from shape_extensions import IntTuple, IntVar

# Host floats of shape S, float32 or float64. A union of arrays, not `np.dtype[np.floating]`:
# pyrefly-numpy-stubs declares `np.dtype` invariant (stub gap NS5), so a float32 array is not an
# array of `np.floating`.
type Floats[S: IntTuple] = np.ndarray[S, np.dtype[np.float32]] | np.ndarray[S, np.dtype[np.float64]]

# A batched table: (1 or nworld, n) values read at row `min(worldid, NRow - 1)`
# (`gpu_scene.binding.Slots`).
type Table[D, NRow: IntVar, N: IntVar] = wp.array[D, [NRow, N]]

# A per-world array: (nworld, n), one row per world (instance poses, overrides).
type PerWorld[D, NWorld: IntVar, N: IntVar] = wp.array[D, [NWorld, N]]

# An output image batch: (nworld, ncam, height, width).
type Images[D, NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar] = wp.array[D, [NWorld, NCam, H, W]]

# Host index lists and masks (NumPy).
type Ids[N: IntVar] = np.ndarray[[N], np.dtype[np.int32]]
type Mask[N: IntVar] = np.ndarray[[N], np.dtype[np.bool_]]

# An array of any dtype and rank, for heterogeneous containers only (slot tables by name, bake
# cache entries, graph inputs, caller output arrays before validation): no element type or shape
# can be named there (Warp arrays are invariant in their dtype). Everything else is shaped.
type AnyDeviceArray = wp.array
type AnyHostArray = np.ndarray[tuple[int, ...]]
# Host values of element type D and any rank, for element-wise helpers ((..., 3) colors,
# directions).
type AnyRank[D: np.generic] = np.ndarray[tuple[int, ...], np.dtype[D]]
