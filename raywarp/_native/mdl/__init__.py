"""The NVIDIA MDL SDK bundled with raywarp.

The release binaries of `raywarp._native.build()["mdl_sdk"]` (BSD-3-Clause, licences in
`licenses/`): libmdl_sdk, the distiller and image plugins, and the SDK's SWIG binding `pymdlsdk`,
built by raywarp's native build hook with the GIL released in every SDK call and
`ILogging_configuration`'s log level (`set_log_level_value` / `get_log_level_value`,
Message_severity values). `load_neuray` loads the SDK and the plugins from this package:

  neuray = raywarp._native.mdl.load_neuray()  # log_level "warning": no info lines on stderr
  with neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
    cfg.add_mdl_path(my_modules)
  assert neuray.start() == 0
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Final, Literal

from . import pymdlsdk

PACKAGE_DIR: Final = Path(__file__).resolve().parent
# libmdl_sdk and the plugins.
LIB_DIR: Final = PACKAGE_DIR / "lib"
# MDL modules shipped with the SDK that are not built into libmdl_sdk (::nvidia::core_definitions).
MDL_DIR: Final = PACKAGE_DIR / "mdl"
# Linux: the release's libmdl_sdk links the CUDA driver (its GPU baker); on a machine without one,
# this stub (every call: CUDA_ERROR_NO_DEVICE) stands in for it. raywarp bakes on the CPU.
CUDA_STUB: Final = PACKAGE_DIR / "cuda_stub" / "libcuda.so.1"
# The plugins `load_neuray` loads: image formats (OpenImageIO: PNG, JPEG, EXR, TIFF, HDR, TGA, ...;
# DDS) and the distiller (`IMdl_distiller_api`: distilling to transmissive_pbr, baking on the CPU).
PLUGINS: Final = ("nv_openimageio", "dds", "mdl_distiller")

type LogLevel = Literal["fatal", "error", "warning", "info", "verbose", "debug"]


def set_log_level(neuray: pymdlsdk.INeuray, log_level: LogLevel) -> None:
  """Set the SDK's log level (messages less severe are dropped; they go to stderr; before or after
  `start()`).

  Raises:
    RuntimeError: if the SDK refuses the level.
  """
  severity = getattr(pymdlsdk._pymdlsdk, f"MESSAGE_SEVERITY_{log_level.upper()}")
  with neuray.get_api_component(pymdlsdk.ILogging_configuration) as logging:
    if logging.set_log_level_value(severity) != 0:
      raise RuntimeError(f"the MDL SDK refused the log level {log_level!r}")


def _cuda_driver() -> None:
  """Linux: makes libcuda.so.1 resolvable for libmdl_sdk: the driver's when installed, else
  `CUDA_STUB` (loaded globally, so libmdl_sdk's dependency binds to it)."""
  if not sys.platform.startswith("linux"):
    return
  try:
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
  except OSError:
    ctypes.CDLL(str(CUDA_STUB), mode=ctypes.RTLD_GLOBAL)


def load_neuray(log_level: LogLevel = "warning") -> pymdlsdk.INeuray:
  """The SDK's `INeuray`, plugins loaded and `MDL_DIR` on the MDL search path, not started yet
  (add search paths, then `start()`).

  The log level is set right after libmdl_sdk loads, before the plugins: only the two info lines
  the library logs while it loads (its path and version) come before it.

  Raises:
    RuntimeError: if libmdl_sdk or a plugin fails to load.
  """
  _cuda_driver()
  neuray = pymdlsdk.load_and_get_ineuray(str(LIB_DIR / "libmdl_sdk"))
  if neuray is None or not neuray.is_valid_interface():
    raise RuntimeError(f"failed to load libmdl_sdk from {LIB_DIR}")
  set_log_level(neuray, log_level)
  for plugin in PLUGINS:
    if not pymdlsdk.load_plugin(neuray, str(LIB_DIR / plugin)):
      raise RuntimeError(f"failed to load the MDL SDK plugin {plugin} from {LIB_DIR}")
  with neuray.get_api_component(pymdlsdk.IMdl_configuration) as cfg:
    if cfg.add_mdl_path(str(MDL_DIR)) != 0:
      raise RuntimeError(f"failed to add the MDL search path {MDL_DIR}")
  return neuray
