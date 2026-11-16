"""Per-world instance optics: the shadow transmittance, optics rows and two-sided / closed-pane bits
of every (world, instance), composed per frame on the device from each world's material choice.

The upload composes them once from the authored materials (`pipeline/upload/optics.compose`, one row
every world shares). A renderer that chooses materials per world (`instance.material`, or a
per-world `material.specular_ior`) in a scene where that choice can change them
(`GpuScene.optics_inputs`) owns world-major (nworld x ninstance) tables instead, recomposed by one
launch per frame (`WorldOptics.write`) with the same formulas (`compose_optics`); its kernels are
compiled with `Features.world_optics` (`switched`), whose readers take world w's entry
(`trace.world_entry`). Renderers without such a declaration keep the one-row tables, launch nothing
and compile the per-instance reads.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np
import warp as wp

from raywarp._base.types import CLOSED_PANE, OPTICS_ROWS, TWO_SIDED, ShapeType
from raywarp.core.kernels.abi.rays import world_entry
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.platform.launches import CachedLaunch
from raywarp.core.pipeline.upload.chords import closed_triangles, triangle_measures
from raywarp.core.shade.materials.brdf import fresnel_average

if TYPE_CHECKING:
  from collections.abc import Set

  from raywarp._base.dims import AnyDeviceArray, Mask
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.upload.optics import OpticsInputs

wp.set_module_options({"enable_backward": False})

# `shadows.translucent` as the composing kernel reads it.
TRANSLUCENT_MODES: Final = {"off": 0, "transmittance": 1, "clear": 2}


def switched(scene: GpuScene, declared: Set[str]) -> OpticsInputs | None:
  """The optics inputs a renderer declaring `declared` per world recomposes per world: the scene's
  (`GpuScene.optics_inputs`: a solid, volume or see-through material or color) where it declares
  `instance.material` (any material on any instance), or `material.specular_ior` alone with an
  instance of a solid or transmissive material (`OpticsInputs.ior_dependent`), else None (one-row
  tables, `Features.world_optics` off)."""
  inputs = scene.optics_inputs
  if inputs is None or "instance.material" in declared:
    return inputs
  return inputs if "material.specular_ior" in declared and inputs.ior_dependent else None


def switched_instances(scene: GpuScene, declared: Set[str]) -> Mask[int] | None:
  """(ninstance,) the instances whose OptiX flags (culling, any-hit) can differ between worlds,
  per world in the layout (`sharing.decide`): every instance under `instance.material`, those of a
  solid material under `material.specular_ior` alone (`OpticsInputs.solid`); None when `switched`
  is off. The others keep the authored flags in the shared scenery (their tables are still read
  per world)."""
  inputs = switched(scene, declared)
  if inputs is None:
    return None
  return np.ones(scene.ninstance, bool) if "instance.material" in declared else inputs.solid


@wp.func
def shadow_mode(
  # In:
  t: wp.vec3,
  mode: int,
) -> wp.vec3:
  """A shadow transmittance under `shadows.translucent` (`TRANSLUCENT_MODES`: 0 off, 2 clear)."""
  if mode == 0:
    return wp.vec3(0.0)
  if mode == 2:
    return wp.where(wp.max(t[0], wp.max(t[1], t[2])) > 0.0, wp.vec3(1.0), wp.vec3(0.0))
  return t


@wp.kernel
def compose_optics(
  # Model:
  medium: wp.array1d[wp.vec4],
  volume: wp.array1d[wp.int32],
  see_alpha: wp.array1d[float],
  transmissive: wp.array1d[wp.int32],
  doubled: wp.array1d[wp.int32],
  closed: wp.array1d[wp.int32],
  chord: wp.array1d[float],
  # Data in:
  instance_material: wp.array2d[int],
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  specular_ior: wp.array2d[float],
  # In:
  mode: int,
  # Out:
  shadow_out: wp.array1d[wp.vec3],
  optics_out: wp.array2d[wp.vec4],
  twosided_out: wp.array1d[int],
):
  """The tables of instance i in world w (`pipeline/upload/optics.compose`, one world there): its
  material's medium row (the world's refraction index where solid), the shadow transmittance (the
  faces of a closed volume absorb over half its chord; the material's own from the world's base
  color (`optics.see_through`), or the tinted instance color's), `shadows.translucent` applied
  (`mode`), and the TWO_SIDED / CLOSED_PANE bits."""
  w, i = wp.tid()
  m = instance_material[wp.min(w, instance_material.shape[0] - 1), i]
  color = instance_color[wp.min(w, instance_color.shape[0] - 1), i]
  has = m >= 0 and m < medium.shape[0]
  row = wp.vec4(0.0)
  own = wp.vec3(0.0)
  tint = color
  absorbing = False
  passed = float(1.0)
  if has:
    row = medium[m]
    ior = specular_ior[wp.min(w, specular_ior.shape[0] - 1), m]
    base = material_base_color[wp.min(w, material_base_color.shape[0] - 1), m]
    if row[3] > 0.0:
      row[3] = ior
    if transmissive[m] != 0:
      passed = 1.0 - fresnel_average(ior)
    if see_alpha[m] >= 0.0:
      part = wp.clamp(1.0 - base[3] * see_alpha[m], 0.0, 1.0)
      own = wp.vec3(wp.clamp(base[0], 0.0, 1.0), wp.clamp(base[1], 0.0, 1.0), wp.clamp(base[2], 0.0, 1.0)) * part * passed
    tint = wp.cw_mul(tint, base)
    absorbing = volume[m] != 0 and closed[i] != 0
  half = wp.where(absorbing, 0.5 * chord[i], 0.0)
  face = wp.vec3(wp.exp(-row[0] * half), wp.exp(-row[1] * half), wp.exp(-row[2] * half))
  white = color == wp.vec4(1.0)
  t = own
  if not (has and white):
    rgb = wp.vec3(wp.clamp(tint[0], 0.0, 1.0), wp.clamp(tint[1], 0.0, 1.0), wp.clamp(tint[2], 0.0, 1.0))
    t = rgb * wp.clamp(1.0 - tint[3], 0.0, 1.0)
  t = wp.cw_mul(face, t)
  solid = row[3] > 0.0
  pane = doubled[i] != 0 and wp.max(t[0], wp.max(t[1], t[2])) > 0.0 and not (has and transmissive[m] != 0)
  pane = pane and not solid and closed[i] != 0
  two = doubled[i] != 0 or solid
  bits = int(0)
  if two and not pane:
    bits = wp.static(TWO_SIDED)
  if pane:
    bits = wp.static(CLOSED_PANE)
  e = world_entry(instance_material.shape[1], w, i)
  shadow_out[e] = shadow_mode(t, mode)
  for k in range(wp.static(OPTICS_ROWS)):
    optics_out[e, k] = row
  twosided_out[e] = bits


def every_closed(scene: GpuScene, inputs: OpticsInputs) -> OpticsInputs:
  """`inputs` with the closed bit of every mesh instance and, in a scene with a volume material,
  their mean chords: the upload tests only the meshes whose authored material needs it
  (`pipeline/upload/optics.optics_inputs`), a per-world material choice can make any of them a pane
  or a volume. Each mesh tested once; two host reads (the instances' kinds and meshes)."""
  kind = host_copy(scene.instance_type)
  todo = np.flatnonzero((kind == int(ShapeType.MESH)) & ~inputs.closed)
  if not len(todo):
    return inputs
  mesh = host_copy(scene.tables["instance.mesh"])[0]
  points, faces = scene.meshes.points, scene.meshes.faces
  tested = {m: closed_triangles(points[m], faces[m]) for m in {int(mesh[i]) for i in todo} if 0 <= m < len(points)}
  closed, chord = inputs.closed.copy(), inputs.chord.copy()
  for i in todo:
    m = int(mesh[i])
    closed[i] = tested.get(m, False)
    if closed[i] and np.any(inputs.volume):
      # The pose's uniform scale (s R); the rest of the stretch is baked into the mesh.
      s = float(np.cbrt(abs(np.linalg.det(scene.init_pose.rot[i].astype(np.float64)))))
      volume, area = triangle_measures(points[m].astype(np.float64), faces[m].astype(np.int64), s * np.eye(3))
      chord[i] = 4.0 * volume / area if area > 0.0 and volume > 0.0 else 0.0
  return dataclasses.replace(inputs, closed=closed, chord=chord)


def _ints(values: np.ndarray, device: wp.Device) -> wp.array:
  """`values` as a device int32 array (at least one entry)."""
  v = np.asarray(values, np.int32)
  return wp.array(v if len(v) else np.zeros(1, np.int32), dtype=wp.int32, device=device)


@final
@dataclasses.dataclass(frozen=True, slots=True)
class OpticsTables:
  """A frame's per-world inputs of the instance optics (each (1 or nworld, n)).

  Attributes:
    material: "instance.material" (n instances).
    color: "instance.color" (n instances).
    base_color: "material.base_color" (n materials).
    specular_ior: "material.specular_ior" (n materials).
  """

  material: AnyDeviceArray
  color: AnyDeviceArray
  base_color: AnyDeviceArray
  specular_ior: AnyDeviceArray


@final
class WorldOptics:
  """A renderer's per-world instance optics (module doc): the composing inputs on the device and
  the (nworld, ninstance) tables they write."""

  def __init__(self, inputs: OpticsInputs, nworld: int, ninstance: int, device: wp.Device) -> None:
    """The tables of `nworld` worlds of `ninstance` instances, composed from `inputs` by `write`."""
    n = max(ninstance, 1)
    self.device = device

    def floats(values: np.ndarray, dtype: type) -> wp.array:
      """`values` (at least one row) as a device array of `dtype`."""
      v = np.asarray(values, np.float32)
      return wp.array(v if len(v) else np.zeros((1, *v.shape[1:]), np.float32), dtype=dtype, device=device)

    self.inputs = [
      floats(inputs.medium, wp.vec4),
      _ints(inputs.volume, device),
      floats(inputs.see_alpha, float),
      _ints(inputs.transmissive, device),
      _ints(inputs.doubled, device),
      _ints(inputs.closed, device),
      floats(inputs.chord, float),
    ]
    # An empty material table: no instance has a material (the inputs above are placeholders).
    self.nmaterial = len(inputs.medium)
    # World-major: world w's entries at w x n (`trace.world_entry`).
    self.nworld = nworld
    self.shadow = wp.zeros(nworld * n, dtype=wp.vec3, device=device)
    self.optics = wp.zeros((nworld * n, OPTICS_ROWS), dtype=wp.vec4, device=device)
    self.twosided = wp.zeros(nworld * n, dtype=int, device=device)
    self._launch = CachedLaunch()

  def write(self, tables: OpticsTables, translucent: Literal["off", "transmittance", "clear"]) -> None:
    """Recomposes the tables from the frame's material choice, colors and refraction indices
    `tables` under `shadows.translucent`."""
    if tables.material.shape[1] == 0:
      return
    data = [tables.material, tables.color, tables.base_color, tables.specular_ior]
    self._launch.launch(
      compose_optics,
      (self.nworld, tables.material.shape[1]),
      inputs=[*self.inputs, *data, TRANSLUCENT_MODES[translucent]],
      outputs=[self.shadow, self.optics, self.twosided],
      device=self.device,
    )
