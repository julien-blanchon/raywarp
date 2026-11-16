"""`RaywarpRendererCfg`: the Isaac Lab renderer config selecting raywarp
(`CameraCfg.renderer_cfg`)."""

from __future__ import annotations

from typing import Literal

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass

from raywarp.integrations.isaaclab.contract import OUTPUTS
from raywarp.scene.lighting.rig import Lighting


@configclass
class RaywarpRendererCfg(RendererCfg):
  """raywarp's renderer for Isaac Lab cameras
  (`TiledCameraCfg(renderer_cfg=RaywarpRendererCfg())`).

  Attributes:
    settings: dotted overrides of the default settings (`{"shadows.enabled": False}`;
      docs/settings.md); `dict(rw.FAST)` for maximum training throughput (still PBR: point
      lights, no reflection rays, 2 shadow casters; 20-67 % less frame time than the default).
    lighting: lights, ambient, environment and headlight of the scene (`rw.Lighting`;
      Newton models carry no lights); None: preserve authored USD lighting. An explicit
      model-only source uses a default key/fill when no lighting is supplied.
    load_textures: load model mesh textures. The USD path always prepares authored textures;
      False on that path is reported as an unsupported override.
    depth_clipping_behavior: depth of rays that hit nothing or beyond the far clip: "none" inf
      (Isaac RTX's), "zero" 0, "max" the far clip.
    semantic_filter: which UsdSemantics labels the segmentation outputs keep (Isaac's grammar,
      `NewtonWarpRendererCfg.semantic_filter`; "*:*": every type and label).
    use_cuda_graph: replay each camera's frame as a CUDA graph (CUDA devices).
  """

  class_type: type | str = "raywarp.integrations.isaaclab.renderer:RaywarpRenderer"
  renderer_type: str = "raywarp"
  # Isaac's Newton cloner builds the Newton model the scene is read from (as for Newton Warp).
  cloning_contexts: tuple[type | str, ...] = ("isaaclab_newton.cloner:NewtonReplicateContext",)
  settings: dict[str, object] = {}  # noqa: RUF012  # configclass copies mutable defaults per instance
  lighting: Lighting | None = None
  load_textures: bool = True
  depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
  semantic_filter: str | list[str] = "*:*"
  use_cuda_graph: bool = True

  def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
    """The outputs written (`contract.OUTPUTS`), known without creating the renderer."""
    return {RenderBufferKind(name): RenderBufferSpec(c.channels, c.dtype) for name, c in OUTPUTS.items()}
