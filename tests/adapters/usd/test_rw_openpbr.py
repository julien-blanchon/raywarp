"""OpenPBR inputs of the canonical material: the coverage table and `clamp_material` (MaterialX
networks: test_rw_materialx.py)."""

from __future__ import annotations

import raywarp as rw
from raywarp._base.named import replace_named
from raywarp.scene.model import checks, openpbr


def test_coverage_table_names_every_input():
  assert (
    set(openpbr.COVERAGE) == set(openpbr.INPUTS) and len(openpbr.INPUTS) == 42
  )  # 41 OpenPBR inputs + geometry_detail_normal


def test_clamp_material_covers_every_input():
  """`clamp_material` brings every float / color input of `Material` and `OpenPbrExtras` into its
  range (a non-positive IOR to its default) and the `ext` fractions into theirs, each reported;
  an in-range material is returned as is."""
  floats, colors = checks.MATERIAL_INPUTS
  x_floats, x_colors = checks.EXTRAS_INPUTS

  def bad(fs: tuple[str, ...], cs: tuple[str, ...]) -> dict[str, float | tuple[float, float, float]]:
    """Every float input `fs` and color input `cs` out of its range."""
    return {**dict.fromkeys(fs, -2.0), **dict.fromkeys(cs, (-1.0, 2.0, 0.5))}

  ext = rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND, alpha_cutoff=2.0, occlusion_strength=-1.0, mirror=3.0)
  extras = replace_named(rw.scene.OpenPbrExtras(), bad(x_floats, x_colors))
  report = rw.scene.LoadReport()
  m = openpbr.clamp_material(replace_named(rw.Material(extras=extras, ext=ext), bad(floats, colors)), report=report)
  for record, names in ((m, floats), (m.extras, x_floats)):
    for name in names:
      want = openpbr.FLOAT_DEFAULTS[name] if name in checks.POSITIVE else checks.FLOAT_RANGES[name][0]
      assert getattr(record, name) == want, name
  for record, names in ((m, colors), (m.extras, x_colors)):
    for name in names:
      assert getattr(record, name)[0] == 0.0 and getattr(record, name)[2] == 0.5, name
  assert (m.ext.alpha_cutoff, m.ext.occlusion_strength, m.ext.mirror) == (1.0, 0.0, 1.0)
  assert sum(report.approximated.values()) == len(floats) + len(colors) + len(x_floats) + len(x_colors) + 3
  good = rw.Material(base_color=(0.2, 0.3, 0.4), specular_roughness=0.7, extras=rw.scene.OpenPbrExtras(fuzz_weight=0.5))
  assert openpbr.clamp_material(good, report=rw.scene.LoadReport()) is good
