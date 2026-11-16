"""`.wpbr` freshness (B26): key (source, options, code version) and dependencies, missing files
included."""

from __future__ import annotations

import concurrent.futures
import errno
import os
from types import SimpleNamespace

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad, write_png

pytest.importorskip("pxr")

import raywarp as rw
import raywarp.formats.usd
from raywarp.core.pipeline.textures import formats
from raywarp.scene.model import wpbr


def _scene(tmp_path, name="scene.usda", nquad=1):
  """A stage of `nquad` quads in tmp_path/assets, the cache directory tmp_path/cache."""
  assets = tmp_path / "assets"
  assets.mkdir(exist_ok=True)
  stage = new_stage(assets, name)
  for i in range(nquad):
    quad(stage, f"/Q{i}")
  stage.Save()
  return stage, assets / name, tmp_path / "cache"


def _warm(scene) -> bool:
  return "warm_load" in scene.report.timings


def test_in_place_edit_rebuilds(tmp_path):
  _, src, cache = _scene(tmp_path)
  rw.formats.usd.load(src, cache=cache)
  assert _warm(rw.formats.usd.load(src, cache=cache))
  folder = os.stat(src.parent)
  with open(src, "a") as f:
    f.write("\n# edited in place\n")
  st = os.stat(src)
  # A distinct stamp before the rebuild starts: a future one reads as edited during the read
  # (`cache.record` since_ns), recorded unknown, and the next load would rebuild again.
  os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
  os.utime(src.parent, ns=(folder.st_atime_ns, folder.st_mtime_ns))  # the directory looks unchanged
  assert not _warm(rw.formats.usd.load(src, cache=cache))
  assert _warm(rw.formats.usd.load(src, cache=cache))


def test_same_size_edit_rebuilds(tmp_path):
  """An edit that keeps the file's size (a changed digit) is seen by its modification time alone."""
  _, src, cache = _scene(tmp_path)
  rw.formats.usd.load(src, cache=cache)
  assert _warm(rw.formats.usd.load(src, cache=cache))
  st = os.stat(src)
  text = src.read_text()
  edited = text.replace("(0, 0, 0)", "(0, 0, 1)", 1)
  assert edited != text and len(edited) == len(text)
  src.write_text(edited)
  os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))  # distinct and in the past (see above)
  assert os.stat(src).st_size == st.st_size
  assert not _warm(rw.formats.usd.load(src, cache=cache))


def test_explicit_cache_file_is_keyed_by_source_and_options(tmp_path):
  """B26: an explicit `cache="x.wpbr"` was reused whatever the source and options."""
  _, src, _ = _scene(tmp_path)
  path = tmp_path / "x.wpbr"
  rw.formats.usd.load(src, cache=path)
  assert _warm(rw.formats.usd.load(src, cache=path))
  assert not _warm(rw.formats.usd.load(src, cache=path, options=rw.SceneOptions(textures=formats.TextureStorage(max_size=64))))
  _, other, _ = _scene(tmp_path, "other.usda", nquad=2)
  again = rw.formats.usd.load(other, cache=path)
  assert not _warm(again) and again.ninstance == 2


def test_missing_texture_is_a_dependency(tmp_path):
  """B26: a texture missing at build time did not invalidate the cache once it appeared."""
  stage, src, cache = _scene(tmp_path)
  from pxr import Sdf, UsdShade

  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex/base.png")
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI().CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
  bind(stage.GetPrimAtPath("/Q0"), mat)
  stage.Save()
  cold = rw.formats.usd.load(src, cache=cache)
  assert cold.nimage == 0
  assert _warm(rw.formats.usd.load(src, cache=cache))
  (src.parent / "tex").mkdir()
  write_png(src.parent / "tex" / "base.png", np.full((2, 2, 3), 200, np.uint8))
  again = rw.formats.usd.load(src, cache=cache)
  assert not _warm(again) and again.nimage == 1


def test_converter_change_rebuilds(tmp_path, monkeypatch):
  """The converter version is part of the key (B24's grey caches would otherwise stay)."""
  from raywarp.formats.usd import version

  _, src, cache = _scene(tmp_path)
  rw.formats.usd.load(src, cache=cache)
  assert _warm(rw.formats.usd.load(src, cache=cache))
  monkeypatch.setattr(version, "CONVERTER", version.CONVERTER + 1)
  assert not _warm(rw.formats.usd.load(src, cache=cache))


def test_stat_files_records_missing_files(tmp_path):
  (tmp_path / "a.png").write_bytes(b"x")
  deps = wpbr.stat_files([str(tmp_path / "b.png"), str(tmp_path / "a.png"), str(tmp_path)])
  assert [(os.path.relpath(d.path, tmp_path), d.size) for d in deps] == [(".", -1), ("a.png", 1), ("b.png", -1)]


def test_asset_cache_resolves_as_usd(tmp_path):
  """Cached asset-path resolution gives USD's resolved paths (and falls back where it cannot)."""
  from pxr import Sdf

  from raywarp.formats.usd import util

  (tmp_path / "tex").mkdir()
  (tmp_path / "tex" / "a.png").write_bytes(b"x")
  (tmp_path / "b.png").write_bytes(b"x")
  stage = new_stage(tmp_path)
  prim = stage.DefinePrim("/P")
  values = ["./tex/a.png", "tex/a.png", "b.png", "./missing.png", str(tmp_path / "b.png"), "../nope/c.png", ""]
  attrs = []
  for i, v in enumerate(values):
    a = prim.CreateAttribute(f"inputs:f{i}", Sdf.ValueTypeNames.Asset)
    a.Set(Sdf.AssetPath(v))
    attrs.append(a)
  with util.asset_cache():
    for a in attrs:
      for _ in range(2):  # resolved, then cached
        got, ref = util.get(a), a.Get()
        assert isinstance(got, Sdf.AssetPath), a.GetName()
        assert (got.path, got.resolvedPath) == (ref.path, ref.resolvedPath), a.GetName()


def test_asset_cache_resolves_backslash_paths(tmp_path):
  """A Windows-separator path (3ds Max exports) resolves like its `/` form; a missing one stays a
  dependency under its `/` form."""
  from pxr import Sdf

  from raywarp.formats.usd import util

  (tmp_path / "textures").mkdir()
  (tmp_path / "textures" / "wall.png").write_bytes(b"x")
  stage = new_stage(tmp_path)
  prim = stage.DefinePrim("/P")
  found = prim.CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset)
  found.Set(Sdf.AssetPath("textures\\wall.png"))
  gone = prim.CreateAttribute("inputs:gone", Sdf.ValueTypeNames.Asset)
  gone.Set(Sdf.AssetPath("textures\\gone.png"))
  with util.asset_cache() as cache:
    for _ in range(2):  # resolved, then cached
      got = util.get(found)
      assert isinstance(got, Sdf.AssetPath)
      assert (got.path, got.resolvedPath) == ("textures\\wall.png", str(tmp_path / "textures" / "wall.png"))
      missing = util.get(gone)
      assert isinstance(missing, Sdf.AssetPath) and not missing.resolvedPath
  layer = gone.GetPropertyStack()[0].layer
  assert cache.missing == {os.path.normpath(layer.ComputeAbsolutePath("textures/gone.png"))}


def test_replaced_file_rebuilds(tmp_path):
  """A source saved through a new file (most editors) is seen by the warm load."""
  stage, src, cache = _scene(tmp_path)
  rw.formats.usd.load(src, cache=cache)
  quad(stage, "/Q2")
  tmp = src.parent / "scene.tmp.usda"
  stage.GetRootLayer().Export(str(tmp))
  os.replace(tmp, src)
  again = rw.formats.usd.load(src, cache=cache)
  assert not _warm(again) and again.ninstance == 2


def test_rebuild_prefetches_the_stale_layers(tmp_path, monkeypatch):
  """A rebuild reads the stale file's layers ahead (not its images), records the same file states
  as a stat, and builds the same scene."""
  from raywarp.formats.usd import api, version
  from raywarp.scene.images import prefetch

  stage, src, cache = _scene(tmp_path)
  from pxr import Sdf, UsdShade

  write_png(src.parent / "base.png", np.full((4, 4, 3), 90, np.uint8))
  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./base.png")
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI().CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
  bind(stage.GetPrimAtPath("/Q0"), mat)
  stage.Save()
  first = rw.formats.usd.load(src, cache=cache)
  fetched: list[str] = []

  def spy(files):
    """Records the prefetched paths."""
    files = list(files)
    fetched.extend(d.path for d in files)
    return prefetch.Prefetch(files)

  monkeypatch.setattr(api, "Prefetch", spy)
  monkeypatch.setattr(version, "CONVERTER", version.CONVERTER + 1)
  again = rw.formats.usd.load(src, cache=cache)
  assert not _warm(again)
  assert fetched == [str(src)]
  recorded = wpbr.recorded_dependencies(again.meta, src.parent)
  assert recorded == wpbr.stat_files(d.path for d in recorded)
  assert {k: v.tobytes() for k, v in again.arrays().items()} == {k: v.tobytes() for k, v in first.arrays().items()}


def test_record_reuses_known_states(tmp_path):
  """`record` takes the states read during the build and stats only the others."""
  (tmp_path / "a").write_bytes(b"x")
  (tmp_path / "b").write_bytes(b"yy")
  known = {str(tmp_path / "a"): wpbr.Dependency(path=str(tmp_path / "a"), mtime_ns=7, size=1)}
  meta: dict = {}
  wpbr.record(
    meta,
    wpbr.CacheKey(adapter="usd", source=str(tmp_path / "s"), options={}, code="c"),
    [str(tmp_path / "b"), str(tmp_path / "a")],
    known=known,
  )
  b = wpbr.stat_files([str(tmp_path / "b")])[0]
  assert meta["cache"]["dependencies"] == [["a", 7, 1], ["b", b.mtime_ns, b.size]]


def test_prefetch_leaves_failed_reads_out(tmp_path, monkeypatch):
  """An I/O failure is not recorded; a missing file is recorded missing. Programming errors
  propagate (test_rw_prefetch.py)."""
  from raywarp.scene.images import prefetch

  (tmp_path / "ok").write_bytes(b"x")
  (tmp_path / "bad").write_bytes(b"y" * 10)

  def flaky(fd: int, size: int) -> bytes:
    """Raise an actual I/O error after reading the bad fixture's bytes."""
    data = os.read(fd, size)
    if data == b"y" * 10:
      raise OSError(errno.EIO, "simulated failed file read")
    return data

  # Replace only this module's os reference: other threads may also read files in the process.
  local_os = SimpleNamespace(**vars(os))
  local_os.read = flaky
  monkeypatch.setattr(prefetch, "os", local_os)
  deps = [wpbr.Dependency(path=str(tmp_path / n), mtime_ns=0, size=0) for n in ("ok", "bad", "missing")]
  p = prefetch.Prefetch(deps)
  # Wait for every read first: close() cancels reads not started yet, which would drop "ok".
  try:
    _, pending = concurrent.futures.wait(p._futures.values(), timeout=2.0)
    assert not pending, "prefetch did not finish its tiny file reads"
    done = p.done()
  finally:
    p.close()
  assert sorted(done) == [str(tmp_path / "missing"), str(tmp_path / "ok")]
  assert done[str(tmp_path / "missing")].size == -1 and done[str(tmp_path / "ok")].size == 1


def test_verify_source_checks_only_the_stage_file(tmp_path):
  """`verify="source"` (prepared caches checked otherwise) compares the stage file alone: an edited
  texture keeps the cache, an edited stage does not; the default compares every file."""
  stage, src, cache = _scene(tmp_path)
  write_png(src.parent / "base.png", np.full((4, 4, 3), 90, np.uint8))
  from pxr import Sdf, UsdShade

  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./base.png")
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI().CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
  bind(stage.GetPrimAtPath("/Q0"), mat)
  stage.Save()
  rw.formats.usd.load(src, cache=cache)
  write_png(src.parent / "base.png", np.full((8, 8, 3), 10, np.uint8))
  assert _warm(rw.formats.usd.load(src, cache=cache, verify="source"))
  assert not _warm(rw.formats.usd.load(src, cache=cache))
  with open(src, "a") as f:
    f.write("\n# edited\n")
  assert not _warm(rw.formats.usd.load(src, cache=cache, verify="source"))
