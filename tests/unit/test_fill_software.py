"""The bundle fill's software word modules (`tools/prebuilt/software.py`): grouping, capture of
Warp's cache entry, the LTO form, math flags, the one per-word compile (CPU, no GPU; the link
itself only with the NVRTC / nvJitLink wheels)."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import pytest
import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.trace.rays.primary import PINHOLE
from tools.prebuilt import fill, software

pytestmark = pytest.mark.usefixtures("_quiet")


@pytest.fixture
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(wp.config, "log_level", logging.WARNING)  # Warp's levels are logging's
  wp.init()


@wp.func_native(f"return ({toggles.WORD_MACRO} & (unsigned int)bit) != 0u;")
def _bit(bit: int) -> bool: ...


@functools.cache
def _kernel(word: int, fast_math: bool = True) -> wp.Kernel:
  """A unique-module kernel whose preamble defines word `word` (as `structs.module_options`)."""
  preamble = f"#define {toggles.WORD_MACRO} {word}u\n"
  options = {
    "fast_math": fast_math,
    "extra_build_options": wp.ModuleBuildOptions(extra_cuda_preamble=preamble, extra_cpu_preamble=preamble),
  }

  @wp.kernel(module="unique", enable_backward=False, module_options=options)
  def word_kernel(a: wp.array[float]):
    i = wp.tid()
    if _bit(1):
      a[i] = wp.sin(a[i])

  return word_kernel


def test_word_modules_group_by_their_kernels_not_their_word() -> None:
  """Two words of one kernel: distinct modules (ids), one group; the word read from the preamble;
  a module without a word has none."""
  a, b = _kernel(0).module, _kernel(1).module
  assert a.get_module_identifier() != b.get_module_identifier()
  assert software.group_key(a) == software.group_key(b)
  assert (software.words_of(a), software.words_of(b)) == ({toggles.WORD_MACRO: 0}, {toggles.WORD_MACRO: 1})
  plain = wp.Module("wpbr_test_plain", None)
  assert software.words_of(plain) == {}


def test_a_group_shards_together_and_a_large_module_alone() -> None:
  """The shard owner of a word module is its group (every word on one shard), else its name;
  owners spread by size, largest first into the smallest shard: one larger than a shard's share
  is alone; every owner on exactly one shard; a kernel's size counts the functions it reaches."""
  a, b = _kernel(0).module, _kernel(1).module
  assert fill._owner(a) == fill._owner(b) == software.group_key(a)
  sizes = {"big": 100, "a": 30, "b": 20, "c": 20, "d": 10}
  shards = fill._spread(sizes, 3)
  assert shards[0] == ["big"] and sorted(o for s in shards for o in s) == sorted(sizes)
  assert {sum(sizes[o] for o in s) for s in shards[1:]} == {40}
  assert fill._source_size(a) > len(_kernel(0).adj.source)


def test_spread_gives_each_relight_kernel_a_shard_of_its_own(tmp_path: Path) -> None:
  """`fill spread`: a relight kernel (a ~13 min compile alone) gets a Warp shard of its own however
  small its source (the package module of the same name does not), the other owners spread by size
  over the rest; each build key once."""
  p = {"scene": "s", "layout": "shared", "tier": "fast", "gi": "True", "set": "env", "outputs": "shaded"}
  relight = "make_relight__locals__relight_rays_x"
  modules = [
    [relight, relight, 1, p],
    ["big", "big", 100, p],
    ["a", "a", 30, p],
    ["raywarp.gi.relight", "raywarp.gi.relight", 20, p],
  ]
  (tmp_path / "part-0.json").write_text(json.dumps({"builds": [["k0", p], ["k1", p]], "modules": modules}))
  fill.spread(tmp_path, 2, 3)
  shards = [json.loads((tmp_path / f"warp-{i}.json").read_text())["modules"] for i in range(3)]
  assert shards[0] == [relight] and sorted(shards[1] + shards[2]) == ["a", "big", "raywarp.gi.relight"]
  assert [len(json.loads((tmp_path / f"ptx-{i}.json").read_text())["points"]) for i in range(2)] == [1, 1]


def test_capture_keeps_warp_s_cache_entry_and_the_lto_form(tmp_path: Path) -> None:
  """`capture`: Warp's AOT layout (cubin name, meta) with an empty binary, the source with the
  word's literal; `lto_source` swaps that literal for the extern word (exactly once); the math
  options become the link's flags."""
  c = software.capture(_kernel(1).module, 80, tmp_path)
  module_id = _kernel(1).module.get_module_identifier()
  assert c.cubin.name == f"{module_id}.sm80.cubin" and c.cubin.read_bytes() == b""
  assert (tmp_path / f"{module_id}.meta").is_file()
  assert f"#define {toggles.WORD_MACRO} 1u" in c.source and c.fast_math
  lto = software.lto_source(c)
  assert "wp_pbr_word()" in lto and f"#define {toggles.WORD_MACRO} 1u" not in lto
  assert software.link_options(c) == ["-lto", "-arch=sm_80", "-ftz=1", "-prec-div=0", "-prec-sqrt=0", "-fma=1"]
  assert "--use_fast_math" in software.nvrtc_options(c) and "--fmad=true" in software.nvrtc_options(c)
  precise = software.capture(_kernel(1, fast_math=False).module, 80, tmp_path / "precise")
  assert software.link_options(precise)[2:5] == ["-ftz=0", "-prec-div=1", "-prec-sqrt=1"]


def test_the_output_words_have_their_own_extern_functions(tmp_path: Path) -> None:
  """A software megakernel's preamble defines the build and output words (`toggles.preamble`):
  `words_of` reads all three, `lto_source` gives each its extern function (exactly once)."""
  preamble = f"#define {toggles.WORD_MACRO} 3u\n#define {toggles.OUT_MACRO} 5u\n#define {toggles.HALF_MACRO} 0u\n"
  options = {"extra_build_options": wp.ModuleBuildOptions(extra_cuda_preamble=preamble, extra_cpu_preamble=preamble)}

  @wp.kernel(module="unique", enable_backward=False, module_options=options)
  def out_kernel(a: wp.array[float]):
    i = wp.tid()
    if toggles.out_on(0, 1):
      a[i] = wp.sin(a[i])

  module = out_kernel.module
  assert software.words_of(module) == {toggles.WORD_MACRO: 3, toggles.OUT_MACRO: 5, toggles.HALF_MACRO: 0}
  lto = software.lto_source(software.capture(module, 80, tmp_path))
  assert all(f"{m.lower()}()" in lto for m in (toggles.WORD_MACRO, toggles.OUT_MACRO, toggles.HALF_MACRO))
  assert f"#define {toggles.OUT_MACRO} 5u" not in lto


def test_a_group_fills_one_entry_per_word_through_one_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`fill_group`: one capture (codegen) and one LTO-IR per group, every word's binary from
  `combination_cubin` (the only per-word compile: replacing it swaps the route), each in its
  module's cache entry with the group's meta; a held entry is not filled again."""
  calls: list[str] = []
  capture = software.capture
  monkeypatch.setattr(software, "capture", lambda *a: calls.append("capture") or capture(*a))
  monkeypatch.setattr(software, "group_ltoir", lambda c: calls.append("ltoir") or b"IR")
  monkeypatch.setattr(
    software, "combination_cubin", lambda c, words, ltoir: f"{ltoir.decode()}:{words[toggles.WORD_MACRO]}".encode()
  )
  modules = [_kernel(0).module, _kernel(1).module]
  assert software.fill_group(modules, 80, tmp_path) == 2
  assert calls == ["capture", "ltoir"]
  for word, m in enumerate(modules):
    entry = tmp_path / m.get_module_identifier()
    assert (entry / f"{entry.name}.sm80.cubin").read_bytes() == f"IR:{word}".encode()
    assert (entry / f"{entry.name}.meta").is_file() and not list(entry.glob("*.cu"))
  assert software.fill_group(modules, 80, tmp_path) == 0  # resumed: held


@pytest.mark.skipif(software.cuda_libs() is None, reason="needs the nvidia-cuda-nvrtc / nvidia-nvjitlink 13.4 wheels")
def test_the_link_folds_the_word(tmp_path: Path) -> None:
  """With the wheels: word 0 links to a smaller cubin than word 1 (the gated body folded away)."""
  c0 = software.capture(_kernel(0).module, 80, tmp_path / "0")
  c1 = software.capture(_kernel(1).module, 80, tmp_path / "1")
  ltoir = software.group_ltoir(c0)
  w0, w1 = ({toggles.WORD_MACRO: w} for w in (0, 1))
  assert len(software.combination_cubin(c0, w0, ltoir)) < len(software.combination_cubin(c1, w1, ltoir))


@functools.cache
def _plain() -> wp.Kernel:
  """A unique-module kernel without a word."""

  @wp.kernel(module="unique", enable_backward=False)
  def plain_kernel(a: wp.array[float]):
    a[wp.tid()] = 1.0

  return plain_kernel


def test_warp_shards_fill_each_module_once_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`fill._compile_warp` over listed shards (their module names): every module filled by the
  shard listing it (a word group whole on one), and a second pass over the same root fills
  nothing (a resumed shard)."""
  monkeypatch.setattr(software, "group_ltoir", lambda c: b"IR")
  monkeypatch.setattr(software, "combination_cubin", lambda c, words, ltoir: b"linked")
  modules = [_kernel(0).module, _kernel(1).module, _plain().module]
  monkeypatch.setattr(fill, "_MODULES", {m.name: m for m in modules})

  def fake_nvrtc(module: wp.Module, arch: int, cache: Path) -> int:
    """The plain modules' compile, without NVRTC: an entry with a binary."""
    entry = cache / module.get_module_identifier()
    entry.mkdir(parents=True)
    (entry / f"{entry.name}.sm{arch}.cubin").write_bytes(b"nvrtc")
    return 1

  monkeypatch.setattr(fill, "_compile_module", fake_nvrtc)
  shards = [frozenset({modules[0].name, modules[1].name}), frozenset({modules[2].name})]
  assert [fill._compile_warp(tmp_path, 80, 2, names, None) for names in shards] == [2, 1]
  cache = tmp_path / "warp" / wp.config.version
  assert {p.name for p in cache.iterdir()} == {m.get_module_identifier() for m in modules}
  assert sum(fill._compile_warp(tmp_path, 80, 2, names, None) for names in shards) == 0


def test_the_words_of_a_build_share_its_device_functions() -> None:
  """A software tier build's megakernel per word (`pixel.megakernel`: `structs.in_module_of`): one
  module per word (distinct ids, one group: the same kernels and symbols), every word's kernel
  calling the same device function objects (one function tree per build, not one per word), and
  each module the one a kernel built for that word's Features itself is (same id: bundles keep)."""
  import dataclasses

  import mujoco
  import mujoco_warp as mjw
  from fixtures.optix_codegen import PARITY_XML

  import raywarp as rw
  import raywarp.integrations.mujoco
  from raywarp.core.build.variants import tiers
  from raywarp.core.kernels import pixel
  from tools.prebuilt import reach

  mjm = mujoco.MjModel.from_xml_string(PARITY_XML)
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  settings = tiers.TIERS["fast"].replace(**{"performance.traversal": "software"})
  r = rw.Renderer(scene, nworld=1, resolution=(8, 8), settings=settings)
  f = dataclasses.replace(r.features, bound=toggles.DEFINED | toggles.CLASS_ALL)
  r.close()
  kernels = [pixel.megakernel(dataclasses.replace(f, word=w, out_word=f.out_word + w), PINHOLE) for w in (1, 2)]
  modules = [k.module for k in kernels]
  assert modules[0].get_module_identifier() != modules[1].get_module_identifier()
  assert software.group_key(modules[0]) == software.group_key(modules[1])
  trees = [{id(fn) for fn in reach.walk(m)[1]} for m in modules]
  assert trees[0] == trees[1] and trees[0]
  own = pixel._megakernel(dataclasses.replace(f, word=2, out_word=f.out_word + 2), PINHOLE, False)  # built for the word itself
  assert own.module.get_module_identifier() == modules[1].get_module_identifier()
