"""Packaged source files (`scene/images/files.py`): USD's package-relative paths read from zip
packages, and the image decoders reading through them."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from raywarp.scene.images import decode as imaging
from raywarp.scene.images import files

_HDR = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n" + bytes([128, 64, 32, 129]) * 4


def _zip(members: dict[str, bytes]) -> bytes:
  out = io.BytesIO()
  with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for name, data in members.items():
      z.writestr(name, data)
  return out.getvalue()


@pytest.mark.parametrize(
  ("path", "parts"),
  [
    ("/a/b.png", None),
    ("/a/p.usdz[t/c.png]", ("/a/p.usdz", "t/c.png")),
    ("/a/p.usdz[q.usdz[c.png]]", ("/a/p.usdz", "q.usdz[c.png]")),
    ("/a/x[1]", ("/a/x", "1")),
    ("[c.png]", None),
    ("/a/p.usdz[c.png", None),
  ],
)
def test_split(path, parts):
  assert files.split(path) == parts


def test_join_suffix_and_absolute():
  assert files.join("/a/p.usdz[m/d.mtlx]", "t/c.png") == "/a/p.usdz[m/t/c.png]"
  assert files.join("/a/p.usdz[m/d.mtlx]", "../c.png") == "/a/p.usdz[c.png]"
  assert files.join("/a/p.usdz[q.usdz[m/d.mtlx]]", "c.png") == "/a/p.usdz[q.usdz[m/c.png]]"
  assert files.join("/a/m/d.mtlx", "../t/c.png") == "/a/t/c.png"
  assert files.join("/a/m/d.mtlx", "/abs//c.png") == "/abs/c.png"
  assert files.suffix("/a/p.usdz[q.usdz[C.PNG]]") == ".png" and files.suffix("/a/b.HDR") == ".hdr"
  assert files.absolute("/a/../p.usdz[m/../c.png]") == "/p.usdz[m/../c.png]"
  assert files.container("/a/p.usdz[q.usdz[c.png]]") == "/a/p.usdz"


def test_read_size_and_missing_members(tmp_path):
  package = tmp_path / "p.usdz"
  package.write_bytes(_zip({"t/c.bin": b"0123456789", "q.usdz": _zip({"c.bin": b"abc"})}))
  member, nested = f"{package}[t/c.bin]", f"{package}[q.usdz[c.bin]]"
  assert files.read(member) == b"0123456789" and files.read(member, 4) == b"0123"
  assert files.read(nested) == b"abc" and files.size(member) == 10 and files.size(nested) == 3
  assert files.is_file(member) and files.is_file(nested) and not files.is_file(f"{package}[nope.png]")
  with pytest.raises(files.PackageError, match=r"no member 'nope\.png'"):
    files.read(f"{package}[nope.png]")
  (tmp_path / "bad.usdz").write_bytes(b"not a zip")
  with pytest.raises(files.PackageError, match="not a readable zip"):
    files.read(f"{tmp_path / 'bad.usdz'}[c.png]")
  assert not files.is_file(f"{tmp_path / 'missing.usdz'}[c.png]")


def test_a_packaged_hdr_decodes_like_the_file(tmp_path):
  """The float decoders (Radiance here) read packaged files too: the same texels as the file's."""
  (tmp_path / "e.hdr").write_bytes(_HDR)
  (tmp_path / "p.usdz").write_bytes(_zip({"env/e.hdr": _HDR}))
  plain = imaging.decode_linear(imaging.ImageSource(path=str(tmp_path / "e.hdr"), transfer="linear"), 2, 2)
  packaged = imaging.decode_linear(imaging.ImageSource(path=f"{tmp_path / 'p.usdz'}[env/e.hdr]", transfer="linear"), 2, 2)
  np.testing.assert_array_equal(plain, packaged)
  assert plain[0, 0, 0] > 0.0


def test_join_keeps_absolute_paths_and_past_root_members():
  """An absolute name is itself (normalised) next to a plain or a packaged document; a `..` past
  the package root stays in the member, where no package holds it."""
  for document in ("/a/m/d.mtlx", "/a/p.usdz[m/d.mtlx]", "/a/p.usdz[q.usdz[d.mtlx]]"):
    assert files.join(document, "/abs//x/../c.png") == "/abs/c.png"
  assert files.join("/a/p.usdz[m/d.mtlx]", "../../t.png") == "/a/p.usdz[../t.png]"


def test_join_anchors_as_ar_does(tmp_path):
  """Asset paths of a layer inside a `.usdz`, anchored by Ar and by `files.join`: the same path
  (inside, past the root, absolute); Ar resolves the past-root one to nothing: `is_file` False."""
  pytest.importorskip("pxr")
  from pxr import Ar, Sdf, Usd, UsdShade

  (tmp_path / "src" / "m").mkdir(parents=True)
  (tmp_path / "src" / "x.png").write_bytes(b"inside")
  (tmp_path / "x.png").write_bytes(b"outside")
  values = ("../x.png", "../../x.png", str(tmp_path / "x.png"), "./sub/../../x.png")
  stage = Usd.Stage.CreateNew(str(tmp_path / "src" / "m" / "s.usda"))
  shader = UsdShade.Shader.Define(stage, "/T")
  for i, value in enumerate(values):
    shader.CreateInput(f"a{i}", Sdf.ValueTypeNames.Asset).Set(value)
  stage.GetRootLayer().Save()
  writer = Sdf.ZipFileWriter.CreateNew(str(tmp_path / "p.usdz"))
  writer.AddFile(str(tmp_path / "src" / "m" / "s.usda"), "m/s.usda")
  writer.AddFile(str(tmp_path / "src" / "x.png"), "x.png")
  writer.Save()
  opened = Usd.Stage.Open(str(tmp_path / "p.usdz"))
  layer = opened.GetRootLayer()
  document = f"{tmp_path / 'p.usdz'}[m/s.usda]"
  for value in values:
    anchored = Sdf.ComputeAssetPathRelativeToLayer(layer, value)
    assert files.join(document, value) == anchored, value
    assert files.is_file(anchored) == bool(Ar.GetResolver().Resolve(anchored)), value


def test_archives_open_once_and_reopen_when_edited(tmp_path, monkeypatch):
  """Many reads of one package (and of a nested one) open each zip once; an edited package is
  read anew."""
  files.close_archives()
  package = tmp_path / "p.usdz"
  package.write_bytes(_zip({**{f"t{i}.bin": bytes([i]) * 4 for i in range(8)}, "q.usdz": _zip({"c.bin": b"abc"})}))
  opened = []
  real = zipfile.ZipFile

  def counting(*args, **kwargs):
    opened.append(args[0])
    return real(*args, **kwargs)

  monkeypatch.setattr(zipfile, "ZipFile", counting)
  for _ in range(3):
    for i in range(8):
      assert files.read(f"{package}[t{i}.bin]") == bytes([i]) * 4 and files.is_file(f"{package}[t{i}.bin]")
    assert files.read(f"{package}[q.usdz[c.bin]]") == b"abc" and files.size(f"{package}[q.usdz[c.bin]]") == 3
  assert len(opened) == 2  # the package, its nested package
  package.write_bytes(_zip({"t0.bin": b"edited!"}))
  assert files.read(f"{package}[t0.bin]") == b"edited!"
  files.close_archives()


def test_concurrent_reads_of_one_package(tmp_path):
  """Threads reading members of one open package concurrently each get their member's bytes."""
  import concurrent.futures

  files.close_archives()
  members = {f"m{i}.bin": np.random.default_rng(i).bytes(4096) for i in range(32)}
  (tmp_path / "p.usdz").write_bytes(_zip(members))
  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    got = list(pool.map(lambda name: files.read(f"{tmp_path / 'p.usdz'}[{name}]"), list(members) * 8))
  assert got == list(members.values()) * 8
  files.close_archives()


def _package(target: Path, src: Path, names: tuple[str, ...]) -> Path:
  """A `.usdz` of the files `names` under `src` (usd-core's writer: aligned, stored)."""
  from pxr import Sdf

  writer = Sdf.ZipFileWriter.CreateNew(str(target))
  for name in names:
    writer.AddFile(str(src / name), name)
  writer.Save()
  return target


def test_a_texture_in_a_nested_usdz_loads_end_to_end(tmp_path):
  """outer.usdz references inner.usdz, whose material reads `tex/c.png` inside it: the texture
  resolves to `outer.usdz[inner.usdz[tex/c.png]]` and decodes, its package the dependency."""
  pytest.importorskip("pxr")
  from pxr import Sdf, Usd, UsdShade
  from rw_helpers import bind, new_stage, preview_material, quad, write_png

  import raywarp as rw
  import raywarp.formats.usd

  inner = tmp_path / "inner"
  (inner / "tex").mkdir(parents=True)
  write_png(inner / "tex" / "c.png", np.full((2, 2, 3), (0, 0, 255), np.uint8))
  stage = new_stage(inner, "inner.usda")
  mat, shader = preview_material(stage, "/Asset/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Asset/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex/c.png")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Asset/Q"), mat)
  stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
  stage.GetRootLayer().Save()
  outer = tmp_path / "outer"
  outer.mkdir()
  _package(outer / "inner.usdz", inner, ("inner.usda", "tex/c.png"))
  stage = new_stage(outer, "scene.usda")
  stage.DefinePrim("/World").GetReferences().AddReference("./inner.usdz")
  stage.GetRootLayer().Save()
  package = _package(tmp_path / "outer.usdz", outer, ("scene.usda", "inner.usdz"))
  builder = rw.formats.usd.read(Usd.Stage.Open(str(package)))
  assert "texture.missing_file" not in builder.report.dropped
  images = builder.build().images
  assert list(images.paths) == [f"{package}[inner.usdz[tex/c.png]]"]
  assert images.data[int(images.offset[0]) : int(images.offset[0]) + 4].tolist() == [0, 0, 255, 255]
  assert str(package) in builder.dependencies


def test_more_packages_than_the_cache_read_concurrently(tmp_path):
  """Threads reading 12 packages (more than `ARCHIVE_CACHE`): evictions close an archive only
  once its readers are done, so every read succeeds (a reader could find it closed)."""
  import concurrent.futures

  files.close_archives()
  names = []
  for i in range(files.ARCHIVE_CACHE + 4):
    (tmp_path / f"p{i}.usdz").write_bytes(_zip({"m.bin": bytes([i]) * 65536}))
    names.append(f"{tmp_path / f'p{i}.usdz'}[m.bin]")

  def check(name: str) -> bool:
    i = int(name.split("[")[0].rsplit("p", 1)[1].split(".")[0])
    return files.read(name) == bytes([i]) * 65536 and files.is_file(name) and files.size(name) == 65536

  with concurrent.futures.ThreadPoolExecutor(8) as pool:
    assert all(pool.map(check, names * 20))
  files.close_archives()


def test_an_edited_package_does_not_keep_its_older_archives(tmp_path):
  """Each edit of a nested package opens it anew and drops the older states (they held the inner
  zip's bytes)."""
  files.close_archives()
  package = tmp_path / "p.usdz"
  for i in range(4):
    package.write_bytes(_zip({"q.usdz": _zip({"c.bin": bytes([i]) * (i + 1)})}))
    os.utime(package, ns=(i * 10**9, i * 10**9))
    assert files.read(f"{package}[q.usdz[c.bin]]") == bytes([i]) * (i + 1)
  held = [key for key in files._ARCHIVES._open if key[0].startswith(str(package))]
  assert len(held) == 2, held  # the package and its nested one, this state only
  files.close_archives()


def test_a_forked_child_drops_the_parent_s_archives(tmp_path):
  """`os.register_at_fork`: a forked child starts with no archive open (they are the parent's)."""
  if not hasattr(os, "fork"):
    pytest.skip("no fork")
  (tmp_path / "p.usdz").write_bytes(_zip({"m.bin": b"x"}))
  assert files.read(f"{tmp_path / 'p.usdz'}[m.bin]") == b"x"
  pid = os.fork()
  if pid == 0:  # the child: exit 0 when nothing is inherited and a read still works
    ok = not files._ARCHIVES._open and files.read(f"{tmp_path / 'p.usdz'}[m.bin]") == b"x"
    os._exit(0 if ok else 1)
  _, status = os.waitpid(pid, 0)
  assert os.waitstatus_to_exitcode(status) == 0
  files.close_archives()
