"""Bind: a USD stage is static; its scene's instances keep their authored poses.

`binding` returns the core's `StaticPoses` over the scene's instance transforms.
"""

from __future__ import annotations

from raywarp.core.pipeline.gpu_scene.binding import StaticPoses
from raywarp.core.pipeline.upload import meshes as mesh_upload
from raywarp.formats.usd.api import StageSource
from raywarp.scene.model.scene import Scene


def binding(scene: Scene, stage: StageSource | None = None) -> StaticPoses:
  """The binding of a USD host scene: its instances' poses (`StaticPoses`, what `upload(scene)`
  binds by default); `stage` (the Adapter protocol's source) is not read: nothing moves a stage."""
  del stage  # a static source: the poses are the scene's
  return StaticPoses(mesh_upload.instance_poses(scene))
