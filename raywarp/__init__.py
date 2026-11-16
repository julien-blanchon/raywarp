"""Batched physically based renderer written in NVIDIA Warp (`import raywarp as rw`).

The public surface is small: these names, three submodules and the adapters.

* Scene: `Scene`, `SceneBuilder`, `Material`, `Light` (host, format-neutral) and the lighting rig
  added to any scene (`Lighting`, `Headlight`: an adapter's `lighting=`, `Scene.with_lighting`);
  `upload` builds the `GpuScene` the kernels read, with `SceneOptions` (texture storage, emissive
  panels, environment sun). Enums, flags, texture references, IES profiles, `read_hdr` and the
  `.wpbr` cache: `rw.scene`.
* Rendering: `Renderer` (`render`, `capture` -> `RenderGraph`, `explain`, ...), `RenderSettings`
  (`RenderSettings()`: the one default; `FAST`: the maximum-throughput field set,
  `RenderSettings().replace(rw.FAST)`), `Outputs` (what is written) ->
  `RenderOutputs` (world-major arrays), `CameraSpec`, `PerWorld` (per-world randomisable tables),
  `StaleBinding`; `Renderer.warm` prepares a configuration ahead, in the background (every cache
  its first frame reads). Settings
  groups and output specs: `rw.settings`. Per-world texture randomization of the material
  tables: `rw.randomize`.
* Adapters, one import each (they need their extras): `import raywarp.integrations.mujoco` then
  `rw.integrations.mujoco.scene(mjm, m)` (MuJoCo Warp, aliased); `import raywarp.formats.usd` (USD
  stages, `[usd]` extra); `import raywarp.integrations.newton` (Newton models; the drop-ins
  `raywarp.integrations.newton.sensor.SensorCamera`,
  `raywarp.integrations.newton.headless.ViewerRaywarpHeadless`); `raywarp.integrations.isaaclab`
  (Isaac Lab's camera renderer).
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING

from raywarp import scene as scene
from raywarp import settings as settings
from raywarp.core.pipeline.gpu_scene.binding import StaleBinding as StaleBinding
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec as CameraSpec
from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene as GpuScene
from raywarp.core.pipeline.upload.build import upload as upload
from raywarp.core.pipeline.upload.options import SceneOptions as SceneOptions
from raywarp.core.pipeline.worlds.sharing import PerWorld as PerWorld
from raywarp.scene import Light as Light
from raywarp.scene import Material as Material
from raywarp.scene import Scene as Scene
from raywarp.scene import SceneBuilder as SceneBuilder
from raywarp.scene.lighting.rig import Headlight as Headlight
from raywarp.scene.lighting.rig import Lighting as Lighting
from raywarp.settings import FAST as FAST
from raywarp.settings import Outputs as Outputs
from raywarp.settings import RenderSettings as RenderSettings

if _TYPE_CHECKING:
  from raywarp import kernels as kernels
  from raywarp import randomize as randomize
  from raywarp.core.outputs.render_outputs import RenderOutputs as RenderOutputs
  from raywarp.core.pipeline.graph import RenderGraph as RenderGraph
  from raywarp.core.pipeline.renderer import Renderer as Renderer

# The renderer and what imports its kernels, imported on first use (`__getattr__`): defining
# its Warp kernels is most of the package's import time, and a warm start reads and uploads the
# scene first (`upload` imports them on a thread meanwhile, `core.pipeline.upload.build`).
_LAZY: dict[str, tuple[str, str | None]] = {
  "Renderer": ("raywarp.core.pipeline.renderer", "Renderer"),
  "RenderGraph": ("raywarp.core.pipeline.graph", "RenderGraph"),
  "RenderOutputs": ("raywarp.core.outputs.render_outputs", "RenderOutputs"),
  "randomize": ("raywarp.randomize", None),
  "kernels": ("raywarp.kernels", None),
}


def __getattr__(name: str) -> object:
  """The lazily imported names of `_LAZY` (module attribute access, PEP 562)."""
  found = _LAZY.get(name)
  if found is None:
    raise AttributeError(f"module 'raywarp' has no attribute {name!r}")
  module = _importlib.import_module(found[0])
  value = module if found[1] is None else getattr(module, found[1])
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  """The module's names with the lazily imported ones (`dir(rw)`, completion; PEP 562)."""
  return sorted({*globals(), *_LAZY})
