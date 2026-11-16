"""Backend tuning that never changes pixels: measured defaults, not user settings.

The values are read where they are used (the kernel launch shape and register cap by `resolve`,
the OptiX register caps by `core/build/variants/registers.py`, the micromaps by the OptiX launch,
the per-frame validation cadence). Benchmarks and tests that
study them swap them with `override` (process-global, for the duration of the `with` block:
Renderers built and rendered inside it use the override). The mesh BVH builder is one constant
(`MESH_BVH`, read by the trace backends).
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Generator
from typing import Final, Literal, final

from raywarp._base.named import replace_named

# CUDA's per-thread register limit: the largest register cap a module can take.
MAX_REGISTERS: Final = 255


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Tuning:
  """The tuning values (None: the measured default of the architecture and scene).

  Attributes:
    block_dim: threads per block of the render kernel (None: 256 for primitive-only scenes
      except on Blackwell RTX GPUs, else 64).
    max_registers: register cap per thread of the Warp render kernel (None: 96; 0: the
      compiler's default).
    ordered_traversal: nearest-first scene-BVH traversal (None: on for shaded outputs except
      primitive-only scenes on Blackwell RTX; never with flex).
    opacity_micromaps: OptiX resolves alpha cut-outs ahead of time with micromaps where their
      alpha test is the same in every world (any-hit calls left on cut-out edges only).
    validate_every: run the per-frame validation (`core/pipeline/worlds/validator.py`) every this
      many eager frames; errors are then reported up to that many frames late (a captured graph
      validates every replay).
    optix_max_registers, optix_light_registers: the OptiX register caps (shaded, light; None:
      the build's tier family's, `OPTIX_REGISTERS`; light 0: the shaded cap).
    counters: debug builds counting, per world, what each frame's shading does (camera samples,
      shaded layers, lights evaluated exactly / cheaply, reflection and shadow rays:
      `Renderer.counters`); slower (atomics), same pixels, compiled by design. OptiX builds also
      count triangle and shared-curve any-hit callback candidates and rejections.
  """

  block_dim: int | None = None
  max_registers: int | None = None
  ordered_traversal: bool | None = None
  opacity_micromaps: bool = True
  validate_every: int = 1
  optix_max_registers: int | None = None
  optix_light_registers: int | None = None
  counters: bool = False

  def __post_init__(self) -> None:
    """Checks the counts (ValueError): a block size and cadence >= 1, a register cap in [0, 255]
    (0: the compiler's default; 255: CUDA's per-thread limit)."""
    counts = {"block_dim": self.block_dim, "validate_every": self.validate_every}
    for name, value in counts.items():
      if value is not None and value < 1:
        raise ValueError(f"tuning {name} must be >= 1, got {value}")
    if self.max_registers is not None and not 0 <= self.max_registers <= MAX_REGISTERS:
      raise ValueError(f"tuning max_registers must be in [0, {MAX_REGISTERS}], got {self.max_registers}")
    for name in ("optix_max_registers", "optix_light_registers"):
      value = getattr(self, name)
      if value is not None and not 0 <= value <= MAX_REGISTERS:
        raise ValueError(f"tuning {name} must be in [0, {MAX_REGISTERS}], got {value}")


# The Warp BVH builder of the mesh BLASes (read at each build: tests swap it).
MESH_BVH: Final[Literal["cubql", "sah", "lbvh"]] = "cubql"

# The tuning stack (`override` pushes, the last one is in effect).
_CURRENT: Final[list[Tuning]] = [Tuning()]


def current() -> Tuning:
  """The tuning in effect."""
  return _CURRENT[-1]


@contextlib.contextmanager
def override(**changes: object) -> Generator[Tuning]:
  """The current tuning with `changes` while the block runs (benchmarks and tests only; one
  thread at a time: the overrides are a process-wide stack).

  Raises:
    TypeError: for an unknown field.
    ValueError: for an invalid value (`Tuning`).
  """
  _CURRENT.append(replace_named(current(), changes))
  try:
    yield _CURRENT[-1]
  finally:
    _CURRENT.pop()
