"""Isaac Lab adapter: raywarp as the renderer of Isaac Lab's camera sensors (an integration
on the Newton adapter).

```python
from raywarp.integrations.isaaclab.cfg import RaywarpRendererCfg

camera = TiledCameraCfg(..., data_types=["rgb", "depth"], renderer_cfg=RaywarpRendererCfg())
```

* Ingest + Bind: authored USD visuals/materials/lights through a private prototype stage view,
  bound to the Newton model Isaac builds for its renderers (PhysX or Newton physics). Body poses
  come from Isaac's scene data provider each frame (no host copy). Explicit model-only sources
  remain available without a USD stage.
* Integration: `cfg.RaywarpRendererCfg` / `renderer.RaywarpRenderer`
  (`isaaclab.renderers.BaseRenderer`), on `cameras.Cameras` (Isaac Lab's renderer calls without
  importing Isaac Lab: tested with fakes).
* Output contract (`contract.OUTPUTS`): Isaac's (N, H, W, C) arrays written in place.

`cfg` and `renderer` import Isaac Lab, so they are imported from their modules; this package and
the rest do not.
"""

from raywarp.integrations.isaaclab.cameras import Cameras as Cameras
from raywarp.integrations.isaaclab.contract import OUTPUTS as OUTPUTS
