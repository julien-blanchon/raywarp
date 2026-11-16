"""`.mtlx` documents a stage sublayers, references or loads are reported dropped, not read."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pxr")

import raywarp as rw
import raywarp.formats.usd  # registers rw.formats.usd

_STAGE = """#usda 1.0
(
    defaultPrim = "World"
    subLayers = [@./looks.mtlx@]
)

def Xform "World"
{
    def Scope "Doc" (
        prepend references = @./materials.mtlx@
    )
    {
    }

    def Scope "Lazy" (
        prepend payload = @./lazy.mtlx@
    )
    {
    }

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </World/Doc/Materials/Gold>
    }
}
"""
_DOCUMENT = """<?xml version="1.0"?>
<materialx version="1.39">
  <open_pbr_surface name="S" type="surfaceshader" />
  <surfacematerial name="Gold" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S" /></surfacematerial>
</materialx>
"""


def test_mtlx_arcs_are_reported_dropped(tmp_path: Path) -> None:
  """Each `.mtlx` arc (sublayer, reference, payload) is one `usd.mtlx_layer` drop; a binding to a
  material of such a document is unresolved (reported), the mesh still loads unbound."""
  for name in ("looks.mtlx", "materials.mtlx", "lazy.mtlx"):
    (tmp_path / name).write_text(_DOCUMENT)
  (tmp_path / "scene.usda").write_text(_STAGE)
  scene = rw.formats.usd.read(str(tmp_path / "scene.usda")).build()
  assert scene.report.dropped["usd.mtlx_layer"] == 3
  assert scene.report.dropped["formats.usd.binding.unresolved"] == 1
  assert len(scene.instances) == 1
  assert not [k for k in scene.report.counts if k.startswith("mtlx.")]
