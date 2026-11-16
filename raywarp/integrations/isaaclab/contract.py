"""Isaac Lab's camera output contract, as raywarp renders it.

Isaac Lab allocates every output as an (N, H, W, C) array (`CameraData.allocate`, one row per
environment) from the renderer's `supported_output_types`; `OUTPUTS` lists the ones this adapter
writes (`RenderBufferKind` values -> channels and Warp dtype). `"rgb"` is Isaac's strided view of
`"rgba"` (written through it). Not supported: `motion_vectors` (Isaac's sign convention is not
copied), colorized segmentation (ids only: `idToLabels` maps them), the simple shading modes
and `instance_id_segmentation_fast`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection
from typing import Final, final

import warp as wp

from raywarp.settings import ColorOutput, DepthOutput, FloatOutput, IdOutput, NormalOutput, Outputs
from raywarp.settings.fields import MissDepth

# Isaac Lab's planar-depth names (`depth` is its alias of `distance_to_image_plane`) and radial one.
PLANAR: Final = ("depth", "distance_to_image_plane")
RADIAL: Final = "distance_to_camera"
INSTANCE: Final = "instance_segmentation"
SEMANTIC: Final = "semantic_segmentation"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Channels:
  """One output's per-pixel layout: `channels` values of the Warp scalar `dtype`."""

  channels: int
  dtype: type


OUTPUTS: Final = {
  "rgba": Channels(4, wp.uint8),
  "rgb": Channels(3, wp.uint8),
  "rgb_hdr": Channels(3, wp.float32),
  "albedo": Channels(4, wp.uint8),
  "depth": Channels(1, wp.float32),
  "distance_to_image_plane": Channels(1, wp.float32),
  "distance_to_camera": Channels(1, wp.float32),
  "normals": Channels(3, wp.float32),
  INSTANCE: Channels(1, wp.int32),
  SEMANTIC: Channels(1, wp.int32),
}

# Isaac Lab's `depth_clipping_behavior` (Isaac RTX's) -> raywarp's miss depth: "none" leaves misses
# at inf (as Isaac RTX does), "zero" writes 0, "max" the far clip (`spawn.clipping_range`).
# Final[...]: the values must stay `MissDepth` literals (`DepthOutput.miss`), not str.
MISS_DEPTH: Final[dict[str, MissDepth]] = {"none": "inf", "zero": "zero", "max": "far"}


def miss_value(miss: MissDepth, zfar: float) -> float:
  """The depth the renderer writes for a miss under `DepthOutput.miss` = `miss`, for a camera
  whose resolved far clip is `zfar` (its own, else the scene's default; 0: unbounded, so inf)."""
  far = zfar if zfar > 0.0 else math.inf
  return {"inf": math.inf, "zero": 0.0, "far": far}[miss]


def check_names(names: Collection[str]) -> None:
  """Raises ValueError for a requested output this adapter does not write."""
  unknown = sorted(set(names) - set(OUTPUTS))
  if unknown:
    raise ValueError(f"raywarp's Isaac Lab renderer does not write {unknown}; it writes {sorted(OUTPUTS)}")


def outputs(names: Collection[str], clipping: str) -> Outputs:
  """Raywarp's `Outputs` for Isaac outputs `names` and Isaac's `depth_clipping_behavior`.

  Depth is planar when a planar name is requested (a radial one is then derived per pixel), radial
  otherwise; segmentation renders segment ids (semantic classes are mapped from them).
  """
  check_names(names)
  if clipping not in MISS_DEPTH:
    raise ValueError(f"depth_clipping_behavior must be one of {sorted(MISS_DEPTH)}, got {clipping!r}")
  wanted = set(names)
  return Outputs(
    rgb=ColorOutput() if wanted & {"rgba", "rgb"} else None,
    depth=DepthOutput(convention="planar" if wanted & set(PLANAR) else "radial", miss=MISS_DEPTH[clipping])
    if wanted & {*PLANAR, RADIAL}
    else None,
    # Semantic classes are mapped from the instance ids (`segments.py`).
    seg=IdOutput() if wanted & {INSTANCE, SEMANTIC} else None,
    hdr=FloatOutput() if "rgb_hdr" in wanted else None,
    albedo=FloatOutput() if "albedo" in wanted else None,
    normal=NormalOutput() if "normals" in wanted else None,
  )
