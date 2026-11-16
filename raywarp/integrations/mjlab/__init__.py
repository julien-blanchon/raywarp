"""mjlab adapter: raywarp as an RGB backend of mjlab's camera sensors (the ``mjlab.rgb_renderers``
entry point `create_groups`; sensors configured with `RaywarpCameraSensorCfg`, `rgb.py`)."""

from raywarp.integrations.mjlab.rgb import MjlabRgbRenderer as MjlabRgbRenderer
from raywarp.integrations.mjlab.rgb import RaywarpCameraSensorCfg as RaywarpCameraSensorCfg
from raywarp.integrations.mjlab.rgb import create_groups as create_groups
