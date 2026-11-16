"""`Frame`: every array one render call reads, gathered by the Renderer before it launches the
kernels (one field per kernel input: instance, material, light, camera tables and poses).

Contract:

* One field per slot (`binding.Slots`, the slot name with "." -> "_") and per Renderer per-world
  field (`exposure`, `ambient`, `env_id`, `env_intensity`, `env_yaw`, `camera_fovy`, `fog_color`,
  `fog_range`, `camera_intrinsics`, `camera_distortion`), holding the
  array of the call: per-world override > binding > scene table > Renderer default
  (`PerWorldTables.frame`).
* Every field is an array: an optional slot the scene, binding and overrides leave out holds the
  Renderer's placeholder (a zero / neutral table read only by kernels compiled with the matching
  feature, e.g. `material_alpha_cutoff` with `Features.alpha_cutout`).
* Readers: the struct builders (`trace/frame_structs.py`, the OptiX backend), the scene BVHs
  (`trace/accel/bvh.py`), the temporal history and the light grid. None of them allocates per frame
  from it: the Renderer rebuilds its kernel structs only when a field's identity changes
  (`same`), and a captured graph compares their fingerprints (address, shape, strides, dtype,
  device: `graph.fingerprint`, `RenderGraph.launch`).

Shapes: `NWorld` worlds and `NColumn` per-world pose columns (`sharing.Layout.columns`); tables are
(1 or nworld, n) (`dims.Table`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, final

import warp as wp

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray, Table
  from raywarp._base.types import tex_affine, uv_xform
  from raywarp.core.kernels.abi.structs import lens_coefficients
  from raywarp.core.pipeline.worlds.sharing import Layout


# Not slots: `arrays` / `same` read the instance `__dict__` (the per-call identity compare of ~50
# fields; `getattr` per field is slower). eq=False: arrays are compared by identity (`same`).
@final
@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class Frame[NWorld: IntVar, NColumn: IntVar]:
  """The arrays of one render call, by kernel field (see the module docstring)."""

  # Instances.
  instance_mesh: Table[int, int, int]
  instance_material: Table[int, int, int]
  instance_size: Table[wp.vec3, int, int]
  instance_color: Table[wp.vec4, int, int]
  instance_shading: Table[int, int, int]
  instance_pos: wp.array[wp.vec3, [NWorld, NColumn]]
  instance_rot: wp.array[wp.mat33, [NWorld, NColumn]]
  instance_visible: Table[int, int, int]
  shared_pos: wp.array[wp.vec3, [int]]
  shared_rot: wp.array[wp.mat33, [int]]
  # Materials.
  material_base_color: Table[wp.vec4, int, int]
  material_emission: Table[float, int, int]
  material_mirror: Table[float, int, int]
  material_texture: wp.array[int, [int, int, int]]
  material_tex_affine: wp.array[tex_affine, [int, int, int]]
  material_tex_variation: Table[wp.vec4, int, int]
  material_tex_repeat: Table[wp.vec2, int, int]
  material_tex_uniform: Table[bool, int, int]
  material_uv_world: Table[uv_xform, int, int]
  material_alpha_cutoff: Table[float, int, int]
  material_base_diffuse_roughness: Table[float, int, int]
  material_base_metalness: Table[float, int, int]
  material_specular_roughness: Table[float, int, int]
  material_specular_weight: Table[float, int, int]
  material_reflectance: Table[float, int, int]
  material_specular_anisotropy: Table[wp.vec2, int, int]
  material_fuzz: Table[wp.vec2, int, int]
  material_fuzz_color: Table[wp.vec3, int, int]
  material_coat_color: Table[wp.vec3, int, int]
  material_coat_eta: Table[float, int, int]
  # Lights.
  light_type: Table[int, int, int]
  light_cast_shadow: Table[bool, int, int]
  light_active: Table[bool, int, int]
  light_color: Table[wp.vec3, int, int]
  light_cutoff: Table[float, int, int]
  light_cone_inner: Table[float, int, int]
  light_strength: Table[float, int, int]
  light_range: Table[float, int, int]
  light_attenuation: Table[wp.vec3, int, int]
  light_radius: Table[float, int, int]
  light_pos: Table[wp.vec3, int, int]
  light_dir: Table[wp.vec3, int, int]
  light_shape: Table[int, int, int]
  light_axis: Table[wp.vec3, int, int]
  light_size: Table[wp.vec2, int, int]
  # Scene cameras (mounts) and deformables.
  camera_pos: Table[wp.vec3, int, int]
  camera_rot: Table[wp.mat33, int, int]
  flex_vert: Table[wp.vec3, int, int]
  # Renderer per-world fields: (1 or nworld,) or (1 or nworld, ncam).
  exposure: wp.array[float, [int]]
  ambient: wp.array[wp.vec3, [int]]
  env_id: wp.array[int, [int]]
  env_intensity: wp.array[float, [int]]
  env_yaw: wp.array[float, [int]]
  camera_fovy: Table[float, int, int]
  camera_intrinsics: Table[wp.vec4, int, int]
  camera_distortion: Table[lens_coefficients, int, int]
  fog_color: wp.array[wp.vec3, [int]]
  fog_range: wp.array[wp.vec3, [int]]

  def arrays(self) -> dict[str, AnyDeviceArray]:
    """Kernel field -> array, in field order (arrays of every dtype and rank)."""
    return {name: self.__dict__[name] for name in _FIELDS}

  def shared_poses(self, layout: Layout) -> tuple[wp.array[wp.vec3, [int]], wp.array[wp.mat33, [int]]]:
    """The poses of the shared instances, by their pose slot: the shared pose table (compact
    layout), else world 0's row of the per-world poses (shared instances are the same in every
    world). Without shared instances, the table (a placeholder)."""
    if layout.compact or not layout.nshared:
      return self.shared_pos, self.shared_rot
    return self.instance_pos[0], self.instance_rot[0]

  def same[OtherWorld: IntVar, OtherColumn: IntVar](self, other: Frame[OtherWorld, OtherColumn] | None) -> bool:
    """Whether `other` holds the same array objects in every field (the struct-cache key)."""
    if other is None:
      return False
    mine, theirs = self.__dict__, other.__dict__
    return all(mine[name] is theirs[name] for name in _FIELDS)


# Frame's field names, in order (read per call by `same`: dataclasses.fields is slow).
_FIELDS: Final = tuple(f.name for f in dataclasses.fields(Frame))
