"""Newton adapter: renders a Newton model's shapes with raywarp, bound to its bodies. `[newton]`
extra.

* `scene(model)`: the GpuScene of the model's shapes, bound to its bodies (body_q -> instance
  poses, one kernel); `load(model)` (Ingest: the host `Scene`) and `binding(scene, model)` (Bind:
  its `NewtonBinding`) are its two steps, for a host scene cached or edited in between.
* `load_usd(model, stage)`: host `Scene` of the stage the model was loaded from (`[usd]` extra),
  bound by `binding(scene, model, stage=stage)` (its rigid-body prims).
* Options: `rw.SceneOptions` (shared by every adapter) and `raywarp.integrations.newton.Options`.
* Drop-ins, imported from their modules (they need Newton >= 1.7; this package also imports with
  Newton 1.6, the one Isaac Lab 3.0 pins): `raywarp.integrations.newton.sensor.SensorCamera` (for
  `newton.sensors.SensorCamera`) and `raywarp.integrations.newton.headless.ViewerRaywarpHeadless` (a
  ViewerNull, no window, no OpenGL).

```python
import raywarp as rw
import raywarp.integrations.newton

scene = rw.integrations.newton.scene(model)
r = rw.Renderer(
  scene, nworld=model.world_count, cameras=[rw.CameraSpec(fovy=60)], resolution=(128, 128)
)
r.render(state_0)  # one body_q -> instance pose kernel, then the frame; state swaps are fine
```
"""

from raywarp.integrations.newton.api import binding as binding
from raywarp.integrations.newton.api import scene as scene
from raywarp.integrations.newton.binding import NewtonBinding as NewtonBinding
from raywarp.integrations.newton.build import load as load
from raywarp.integrations.newton.from_usd import load_usd as load_usd
from raywarp.integrations.newton.types import Checkerboard as Checkerboard
from raywarp.integrations.newton.types import Hemisphere as Hemisphere
from raywarp.integrations.newton.types import Options as Options
from raywarp.integrations.newton.types import StateView as StateView
from raywarp.integrations.newton.types import default_lights as default_lights
