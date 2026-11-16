"""glTF import: a Khronos glTF sample asset, downloaded on first run into the cache (never
committed), turning on a turntable under a studio HDRI. The default, ToyCar (CC0), shows the
clear coat, sheen and transmission extensions; ABeautifulGame (a chess set, CC BY 4.0, 43 MB) works
too.

  python examples/demos/gltf_asset.py --size 1280x720 --out gltf_asset.mp4
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import _common
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.formats.gltf
from raywarp.cache.paths import cache_dir
from raywarp.viewer.camera import bounds

URL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/{0}/glTF-Binary/{0}.glb"
SHA256 = {
  "ToyCar": "01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c",
  "ABeautifulGame": "bd7133b4b322aae97c589b8839dae8155ad2546acb35ae32a127e722a959d007",
}


def fetch(model: str) -> Path:
  """The local copy of sample asset `model` (downloaded once, checked against its pinned hash)."""
  root = cache_dir() or Path.home() / ".cache" / "raywarp"
  path = root / "gallery" / f"{model}.glb"
  if not path.is_file():
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL.format(model), timeout=60) as response:
      data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256[model]:
      raise RuntimeError(f"{model}.glb: unexpected sha256 (the upstream file changed)")
    path.write_bytes(data)
  return path


def turntable_view(host: rw.Scene) -> _common.View:
  """A view of the asset's centre from the front, a little above, framing its bounds."""
  box = bounds(host)
  if box is None:
    raise ValueError("an empty asset")
  centre, radius = (box[0] + box[1]) / 2, float(np.linalg.norm(box[1] - box[0])) / 2
  return tuple(centre + radius * np.array([0.0, -2.4, 0.9])), tuple(centre), 40.0


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "gltf_asset.mp4",
  model: str = "ToyCar",
) -> None:
  """Records a full turn around the asset."""
  studio = rw.Lighting(envmap=str(_common.FIXTURES / "textures" / "studio.hdr"), envmap_intensity=1.0, replace=True)
  host = raywarp.formats.gltf.load(fetch(model), lighting=studio)
  print(host.report)  # what the importer kept, dropped or approximated
  with wp.ScopedDevice(device):
    stage = _common.Stage(rw.upload(host), turntable_view(host))
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution, rw.RenderSettings().replace({"aa.spp": 2}))
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        turns = [360.0 * (f / frames + w / worlds) for w in range(worlds)]
        r.camera_pose.assign(np.stack([_common.orbit(stage.view, a) for a in turns])[:, None, :])
        r.render()
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
