"""glTF 2.0 adapter: `.gltf` / `.glb` assets (Khronos samples, Objaverse, YCB, Sketchfab exports).

```python
import raywarp as rw
import raywarp.formats.gltf

scene = rw.formats.gltf.scene("DamagedHelmet.glb", lighting=rw.Lighting.studio(), cache=True)
```

* `scene(path)`: the GpuScene; `load(path)`: the host Scene (`rw.upload` it, or add it to a
  builder); `stage(path)`: the cached USD conversion both read (`raywarp.formats.gltf.api`).
* Conversion: `raywarp.formats.gltf.to_usd` (meshes, PBR materials and their KHR extensions,
  lights, cameras, instancing; what is not converted is listed there).

Needs the `gltf` extra (`pip install "raywarp[gltf]"`: pygltflib and the `usd` extra).
"""

from raywarp.formats.gltf.api import load as load
from raywarp.formats.gltf.api import scene as scene
from raywarp.formats.gltf.api import stage as stage
