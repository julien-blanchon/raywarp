"""JAX bridge (MJX, MuJoCo Playground): a JAX physics state rendered by a MuJoCo Warp scene, and
the renders as JAX arrays, without host copies.

```python
import raywarp.integrations.mujoco.mjx as rw_mjx

r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=n, ...)  # bound to the mjw.Data `d`
rw_mjx.feed(d, mjx_data)  # the poses MJX computed (batched: one row per world)
r.render(d)  # or a captured graph's replay: `d`'s arrays never change
rgb = rw_mjx.to_jax(r.outputs)["rgb"]  # (nworld, ncam, H, W, 4) uint8, a view
```

`feed` copies the pose fields the renderer reads (`binding.DATA_SLOTS`: geom poses, and the light,
camera and flex vertex poses the JAX state has; an `mjx.Data` holds no light or flex vertex poses:
those keep the `mjw.Data`'s, i.e. lights on moving bodies stay where `put_data` put them) into
the `mjw.Data` the renderer is bound to: one device copy per field (JAX arrays are immutable:
every step allocates new ones, so a renderer reading them in place would see new pointers every
frame and no captured graph could replay).
`to_jax` views the requested outputs as JAX arrays. Both cross through DLPack with its stream
exchange (`wp.from_jax` / `wp.to_jax`): Warp's stream waits for the JAX computation producing a
state, JAX's for the render writing an output; nothing synchronises the host. Needs `jax`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import warp as wp

if TYPE_CHECKING:
  import jax
  import mujoco_warp as mjw

  from raywarp.core.outputs.render_outputs import RenderOutputs

# The `mjw.Data` pose fields the renderer reads (`binding.DATA_SLOTS`), and the ones every state
# has.
FIELDS: Final = ("geom_xpos", "geom_xmat", "light_xpos", "light_xdir", "cam_xpos", "cam_xmat", "flexvert_xpos")
REQUIRED: Final = ("geom_xpos", "geom_xmat")


def feed(d: mjw.Data, state: object) -> None:
  """Copies the pose fields of a JAX state (an `mjx.Data`, batched over worlds or one world: any
  object with those attributes) into `d`, on Warp's stream after JAX's (see the module doc).

  Raises:
    ValueError: for a state without geom poses, or a field of another size than `d`'s.
  """
  for name in FIELDS:
    source: jax.Array | None = getattr(state, name, None)
    if source is None:
      if name in REQUIRED:
        raise ValueError(f"mjx.feed: the state has no {name}")
      continue
    dest: wp.array = getattr(d, name)
    if str(source.dtype) != "float32":
      raise ValueError(f"mjx.feed: {name} is {source.dtype}, the mjw.Data float32 (jax_enable_x64 states: cast first)")
    view = wp.from_jax(source, dtype=dest.dtype)
    batch = tuple(view.shape)
    if batch != tuple(dest.shape) and (dest.shape[0], *batch) != (1, *dest.shape[1:]):
      raise ValueError(f"mjx.feed: {name} is {tuple(source.shape)}, the mjw.Data's {tuple(dest.shape)} {dest.dtype}")
    wp.copy(dest, view.reshape(dest.shape))


def to_jax(outputs: RenderOutputs) -> dict[str, jax.Array]:
  """The requested outputs of a render (`Renderer.outputs`) as JAX arrays (views of the
  renderer's buffers, overwritten by its next render: copy what must outlive it)."""
  out: dict[str, jax.Array] = {}
  for field in dataclasses.fields(outputs):
    value = getattr(outputs, field.name)
    if isinstance(value, wp.array):
      out[field.name] = wp.to_jax(value)
  return out
