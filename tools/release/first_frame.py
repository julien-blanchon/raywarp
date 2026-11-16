"""The release smoke: one frame of a scene from the installed wheel, its prebuilt kernels downloaded
on first use (`raywarp/core/build/kernel_cache/prebuilt.py`), timed, every compile counted.

  python -I first_frame.py --out DIR --repo REPO [--scene franka_showroom] [--base fast]
    [--size 512] [--worlds 4] [--expect-zero-compiles]

REPO: a development checkout, for the scene registry only (an MJCF scene with
MuJoCo Warp, or a host / code scene; assets under $MJPBR_ASSET_DIR); raywarp is the installed one
(-I, run from another directory). Fresh caches are the caller's (first_frame.sh). Writes
DIR/<scene>-<base>.json (stage times, the kernel store's bytes and files after the frame, the
compiles by kind, OptiX module creations, mean RGB) and a PNG; exits non-zero on a black image, or,
with `--expect-zero-compiles`, on any compile of a bundled kernel (a Warp module of raywarp or
Warp's own, a module-level one of MuJoCo Warp's, "simulator": a miss means another mujoco-warp
release than the fill's; an OptiX programs' PTX). MuJoCo Warp's per-model kernels ("warp_other")
are not bundled. An OptiX module created here is OptiX's compile of a bundled PTX (no pack for
this driver), counted and timed apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import threading

  from raywarp.core.build.kernel_cache.prebuilt import ModuleRecord
  from raywarp.core.outputs.render_outputs import RenderOutputs
  from raywarp.core.pipeline.renderer import Renderer
  from raywarp.core.trace.optix.context import DeviceContext, OptixObject
  from raywarp.settings import RenderSettings

# (kind, name) of every compile: "warp" (a module of raywarp's or Warp's own code), "simulator"
# (a module-level one of MuJoCo Warp's, bundled: `tools/prebuilt/fill.py` `SIMULATOR`, named after
# its Python module, but its tests' and CLI's), "warp_other" (MuJoCo Warp's per-model kernels),
# "ptx" (an OptiX programs' PTX); OPTIX_SECONDS: each OptiX module created.
COMPILES: list[tuple[str, str]] = []
OPTIX_SECONDS: list[float] = []
KINDS = ("warp", "simulator", "warp_other", "ptx")
_BUNDLED = ("raywarp", "warp.")
_SIMULATOR = "mujoco_warp._src."
_NOT_SIMULATION = ("_test", ".cli")


def _warp_kind(module: object) -> str:
  """The compile kind of Warp module `module` (`COMPILES`): by its name, then its kernels' Python
  modules."""
  name = str(getattr(module, "name", ""))
  if name.startswith(_SIMULATOR) and not name.endswith(_NOT_SIMULATION):
    return "simulator"
  owners = {getattr(getattr(k, "func", None), "__module__", "") for k in getattr(module, "kernels", {}).values()} or {name}
  return "warp" if any(o.startswith(_BUNDLED) for o in owners) else "warp_other"


def _count_compiles() -> None:
  """Wraps Warp's module compile, raywarp's OptiX PTX compile and its OptiX module creation."""
  import warp._src.context as warp_context

  from raywarp.core.trace.optix import modules, pipeline

  compile_module = warp_context.Module._compile

  def counted(self, *args, **kwargs):
    compiled = compile_module(self, *args, **kwargs)
    if compiled:
      COMPILES.append((_warp_kind(self), str(getattr(self, "name", "?"))))
    return compiled

  warp_context.Module._compile = counted
  ptx = pipeline.PtxCompile._compile

  def counted_ptx(module, arch, path):
    COMPILES.append(("ptx", str(path)))
    return ptx(module, arch, path)

  pipeline.PtxCompile._compile = staticmethod(counted_ptx)
  create = modules.create

  def timed_create(ctx: DeviceContext, ptx: bytes, record: ModuleRecord, stop: threading.Event | None = None) -> OptixObject:
    start = time.perf_counter()
    module = create(ctx, ptx, record, stop)
    OPTIX_SECONDS.append(time.perf_counter() - start)
    return module

  modules.create = timed_create


def _renderer(name: str, settings: RenderSettings, size: int, worlds: int) -> tuple[Renderer, Callable[[], RenderOutputs]]:
  """(renderer, render-one-frame) of registry scene `name`."""
  import numpy as np
  import scenes  # the development checkout's scene registry

  import raywarp as rw

  entry = scenes.SCENES[name][1]
  if isinstance(entry, scenes.MjcfScene):
    import mujoco_warp as mjw

    from raywarp.integrations import mujoco as pbr_mujoco

    mjm = entry().compile()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=worlds)
    # The quickstart's order: the renderer first (its seed places the bundled kernels, MuJoCo
    # Warp's module-level ones among them), then the simulator's first launch.
    r = rw.Renderer(pbr_mujoco.scene(mjm, m), nworld=worlds, cameras=["cam"], resolution=(size, size), settings=settings)
    mjw.forward(m, d)
    return r, lambda: r.render(d)
  if not isinstance(entry, (scenes.HostScene, scenes.CodeScene)):
    raise SystemExit(f"{name}: a {type(entry).__name__}, not a MuJoCo, USD or code scene")
  host = entry.load()
  r = rw.Renderer(rw.upload(host), nworld=worlds, cameras=entry.camera_specs()[:1], resolution=(size, size), settings=settings)
  r.camera_pose.assign(np.broadcast_to(entry.camera_poses()[:1], (worlds, 1, 7)).copy())
  return r, r.render


def _store_size() -> tuple[int, int]:
  """(bytes, files) of the kernel store."""
  from raywarp.core.build.kernel_cache import store

  root = store.store_root()
  files = [p for p in root.rglob("*") if p.is_file()] if root is not None and root.is_dir() else []
  return sum(p.stat().st_size for p in files), len(files)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--repo", type=Path, required=True, help="a checkout: the scene registry")
  p.add_argument("--scene", default="franka_showroom")
  p.add_argument("--base", choices=("default", "fast"), default="fast", help="the default or rw.FAST")
  p.add_argument("--size", type=int, default=512)
  p.add_argument("--worlds", type=int, default=4)
  p.add_argument("--expect-zero-compiles", action="store_true")
  a = p.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  sys.path.append(str(a.repo / "internal" / "benchmarks"))
  times: dict[str, float] = {}
  t = time.perf_counter()

  def lap(name: str) -> None:
    nonlocal t
    now = time.perf_counter()
    times[name] = round(now - t, 3)
    t = now
    print(f"first_frame {name}: {times[name]:.2f} s", flush=True)

  import warp as wp

  import raywarp as rw

  _count_compiles()
  wp.init()
  lap("imports")
  settings = rw.RenderSettings().replace(rw.FAST) if a.base == "fast" else rw.RenderSettings()
  r, frame = _renderer(a.scene, settings, a.size, a.worlds)
  lap("scene_and_renderer")
  frame()
  wp.synchronize()
  lap("first_frame")
  rgb = frame().rgb
  wp.synchronize()
  lap("warm_frame")
  assert rgb is not None
  rgba = rgb.numpy()[0, 0]
  from PIL import Image

  tag = f"{a.scene}-{a.base}"
  Image.fromarray(rgba[..., :3].copy()).save(a.out / f"{tag}.png")
  size, files = _store_size()
  mean = [float(v) for v in rgba[..., :3].reshape(-1, 3).mean(0)]
  record = {
    "scene": a.scene,
    "base": a.base,
    "device": str(r.device),
    "traversal": r.features.traversal,
    "size": a.size,
    "worlds": a.worlds,
    "times_s": times,
    "store_bytes": size,
    "store_files": files,
    "compiles": {k: sum(1 for kind, _ in COMPILES if kind == k) for k in KINDS},
    "compiled": [f"{k}:{n}" for k, n in COMPILES if k != "warp_other"][:50],
    "optix_modules": {"count": len(OPTIX_SECONDS), "seconds": round(sum(OPTIX_SECONDS), 2)},
    "mean_rgb": mean,
  }
  (a.out / f"{tag}.json").write_text(json.dumps(record, indent=1))
  print("first_frame " + json.dumps(record), flush=True)
  if max(mean) < 2.0:
    raise SystemExit(f"first_frame: the image is (nearly) black: mean RGB {mean}")
  bundled = sum(n for k, n in record["compiles"].items() if k != "warp_other")
  if a.expect_zero_compiles and bundled:
    raise SystemExit(f"first_frame: {bundled} compiles of bundled kernels: {record['compiled']}")


if __name__ == "__main__":
  main()
