"""Computes the split-sum DFG table shipped as `raywarp/data/dfg_lut.npz` (`ibl.dfg_texture`).

    python tools/make_dfg_lut.py [--out raywarp/data/dfg_lut.npz]

Filament's multiscatter DFG table (Karis 2013, Filament "Image based lights"), GGX importance
sampled with the renderer's height-correlated Smith visibility (`brdf.v_smith_ggx_correlated`) on
Warp's CPU device: (DFG_LUT_SIZE, DFG_LUT_SIZE, 2) float32 indexed [perceptual roughness, NoV],
channel x = int V Fc, channel y = int V (texel centres span (0, 1)); and the cloth table
(DFG_LUT_SIZE, DFG_LUT_SIZE) indexed [sqrt(Charlie α), NoV]: the directional albedo
int D_Charlie V_Neubelt NoL of the sheen lobe (Filament `DFV_Charlie_Uniform`, uniform hemisphere
sampling of the half vector, CLOTH_LUT_SAMPLES), the "DG" term of its image-based lighting and
of OpenPBR's fuzz albedo scaling. The renderer loads the file instead of computing it in every
process (W6: no IBL module compile at `Renderer()`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import warp as wp

from raywarp.core.shade.environment import ibl
from raywarp.core.shade.materials import brdf

wp.set_module_options({"enable_backward": False})

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "raywarp" / "data" / ibl.DFG_FILE


@wp.kernel
def dfg_texel(
  # In:
  nsample: int,
  rows: wp.array[int],
  # Out:
  lut_out: wp.array3d[float],
):
  """Texel (roughness rows[r], NoV i) of the DFG table, stored at [r, i]: (int V Fc, int V) by GGX
  importance sampling."""
  r, i = wp.tid()
  j = rows[r]
  n = lut_out.shape[1]
  nov = (float(i) + 0.5) / float(n)
  pr = (float(j) + 0.5) / float(n)
  alpha = pr * pr
  v = wp.vec3(wp.sqrt(1.0 - nov * nov), 0.0, nov)
  inv_n = 1.0 / float(nsample)
  x = float(0.0)
  y = float(0.0)
  for s in range(nsample):
    h = ibl.importance_sample_ggx(ibl.hammersley(s, inv_n), alpha)
    voh = wp.dot(v, h)
    l = 2.0 * voh * h - v
    nol = l[2]
    noh = h[2]
    if nol > 0.0 and voh > 0.0:
      # f · NoL / pdf with pdf = D NoH / (4 VoH): D cancels, leaving V · 4 VoH NoL / NoH.
      gv = brdf.v_smith_ggx_correlated(nov, nol, alpha) * 4.0 * voh * nol / noh
      fc = wp.pow(1.0 - voh, 5.0)
      x += gv * fc
      y += gv
  lut_out[r, i, 0] = x * inv_n
  lut_out[r, i, 1] = y * inv_n


@wp.kernel
def cloth_texel(
  # In:
  nsample: int,
  # Out:
  lut_out: wp.array2d[float],
):
  """Texel (sqrt(α) j, NoV i) of the cloth table: int D_Charlie V_Neubelt NoL over the
  hemisphere (half vectors uniform on the hemisphere: pdf(l) = 1 / (2 pi 4 VoH))."""
  j, i = wp.tid()
  n = lut_out.shape[0]
  nov = (float(i) + 0.5) / float(n)
  pr = (float(j) + 0.5) / float(n)
  alpha = wp.max(pr * pr, brdf.MIN_CHARLIE_ALPHA)
  v = wp.vec3(wp.sqrt(1.0 - nov * nov), 0.0, nov)
  inv_n = 1.0 / float(nsample)
  acc = float(0.0)
  for s in range(nsample):
    u = ibl.hammersley(s, inv_n)
    phi = 2.0 * wp.pi * u[0]
    cos_t = 1.0 - u[1]
    sin_t = wp.sqrt(wp.max(1.0 - cos_t * cos_t, 0.0))
    h = wp.vec3(sin_t * wp.cos(phi), sin_t * wp.sin(phi), cos_t)
    voh = wp.dot(v, h)
    l = 2.0 * voh * h - v
    nol = l[2]
    if nol > 0.0 and voh > 0.0:
      acc += brdf.d_charlie(alpha, h[2]) * brdf.v_neubelt(nov, nol) * nol * voh
  lut_out[j, i] = acc * (8.0 * wp.pi) * inv_n


def cloth_table(size: int = ibl.DFG_LUT_SIZE, nsample: int = ibl.CLOTH_LUT_SAMPLES) -> np.ndarray:
  """The (size, size) float32 cloth DG table computed on the CPU device."""
  with wp.ScopedDevice("cpu"):
    lut = wp.zeros((size, size), dtype=float)
    wp.launch(cloth_texel, dim=(size, size), inputs=[nsample], outputs=[lut])
    return lut.numpy()


def dfg_table(
  size: int = ibl.DFG_LUT_SIZE, nsample: int = ibl.DFG_LUT_SAMPLES, rows: Sequence[int] | None = None
) -> np.ndarray:
  """The (size, size, 2) float32 DFG table computed on the CPU device; `rows`: only these
  roughness rows, (len(rows), size, 2) (the test's spot check)."""
  picked = list(range(size)) if rows is None else list(rows)
  with wp.ScopedDevice("cpu"):
    lut = wp.zeros((len(picked), size, 2), dtype=float)
    wp.launch(dfg_texel, dim=(len(picked), size), inputs=[nsample, wp.array(picked, dtype=int)], outputs=[lut])
    return lut.numpy()


def main() -> None:
  """Writes the table to --out."""
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()
  np.savez_compressed(args.out, dfg=dfg_table(), cloth=cloth_table())
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
