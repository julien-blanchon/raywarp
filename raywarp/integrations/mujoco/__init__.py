"""MuJoCo / MuJoCo Warp adapter: `scene(mjm, m)` -> GpuScene rendering `mjw.Data` poses, and MuJoCo
Warp's renderer API on raywarp (`create_render_context`, `render`, `get_rgb`, `get_depth`,
`get_segmentation`: the one-import switch, `render_context.py`; `bind_outputs` / `capture` for
the policy path)."""

from raywarp.integrations.mujoco.binding import MjwBinding as MjwBinding
from raywarp.integrations.mujoco.build import Options as Options
from raywarp.integrations.mujoco.build import scene as scene
from raywarp.integrations.mujoco.lights import add_lights as add_lights
from raywarp.integrations.mujoco.render_context import RenderContext as RenderContext
from raywarp.integrations.mujoco.render_context import bind_outputs as bind_outputs
from raywarp.integrations.mujoco.render_context import capture as capture
from raywarp.integrations.mujoco.render_context import context_renderer as context_renderer
from raywarp.integrations.mujoco.render_context import create_render_context as create_render_context
from raywarp.integrations.mujoco.render_context import get_depth as get_depth
from raywarp.integrations.mujoco.render_context import get_rgb as get_rgb
from raywarp.integrations.mujoco.render_context import get_segmentation as get_segmentation
from raywarp.integrations.mujoco.render_context import render as render
