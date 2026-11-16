"""What a surface looks like at a hit (device code): its material, textures and shading normal,
and how it reflects light.

- `evaluate.py`: `make_eval_surface`, the surface and shading normal at a hit, from
  `shape_normal.py` (vertex colors, smooth normals) and `maps.py` (texture maps: `uv_frame.py`,
  `texture.py`, `uv.py`, `uv_maps.py`), into `material.py`'s `Surface` / `Lobes`; `layers.py`: the
  layered lobes' inputs and frames; `flex.py`: a flex hit's.
- `brdf.py`: Filament's BRDF; `openpbr_lobes.py`: the EON rough diffuse lobe.
- `octahedral.py`: unit-vector encodings.
"""

from __future__ import annotations
