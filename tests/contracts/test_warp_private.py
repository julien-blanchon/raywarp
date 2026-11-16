"""The fence around raywarp's private uses of Warp: every one is listed here and checked
against the installed Warp, so a new Warp release (`VALIDATED_WARP`, the warp-lang cap) is
validated by running this file with it; a private use missing from the lists fails
`test_every_private_use_is_fenced`.
"""

from __future__ import annotations

import ctypes
import inspect
import re
import tomllib
from pathlib import Path
from typing import Final

import pytest
import warp as wp
import warp._src.context
import warp._src.texture
import warp._src.types

from raywarp.core.trace.accel import nodes

ROOT: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = ROOT / "raywarp"
NATIVE: Final = Path(wp.__file__).parent / "native"

# The C++ of Warp's native headers that raywarp's preambles use (trace/accel/nodes.py,
# trace/accel/packed.py: the BVH and mesh internals; the public math types and functions are not
# listed).
CPP: Final = (
  "BVHPackedNodeHalf",
  "bvh_load_node",
  "bvh_get",
  "mesh_get",
  "bvh_query_node_is_leaf",
  "bvh_query_node_lower_payload",
  "bvh_query_node_upper_payload",
  "bvh_query_node_pack",
  "mesh_query_ray_intersect_aabb",
  "mesh_query_ray_safe_dir",
  "mesh_query_ray_use_fast_aabb",
  "diff_product",
  "max_dim",
  "sign_mask",
  "xorf",
  "BVH_QUERY_STACK_SIZE",
  "WP_TILE_BLOCK_DIM",
)
# `wp::` names of the preambles that are Warp's public device API (not fenced).
CPP_PUBLIC: Final = frozenset({"vec2", "vec2i", "vec3", "vec4", "uint32", "uint64", "cross", "dot", "normalize", "atomic_add"})
# Fields of Warp's native BVH / Mesh / packed-node structs the preambles read.
CPP_FIELDS: Final = {
  "bvh.h": ("node_lowers", "node_uppers", "primitive_indices", "num_nodes", "max_nodes", "max_depth"),
  "mesh.h": ("points", "indices", "bvh"),
}
# Native runtime functions `textures.MipTexture2D` calls (Runtime.core, ctypes).
CORE: Final = (
  "wp_texture_get_mip_level_array_device",
  "wp_texture_copy_device",
  "wp_texture_object_destroy_device",
)
# Private Python of Warp: (name in raywarp's source, where it is used).
PYTHON: Final = (
  ("warp._src.texture", "textures (MemoryType)"),
  ("warp._src.types", "optix/params (array_t)"),
  ("warp._src.context", "viewer/watch (Module._compile: the compile events)"),
  ("_detach_references", "optix/launch (Module)"),
  ("_cls.vars", "bake_cache (struct instance fields)"),
  ("_upload_mipmap_chain", "textures.MipTexture2D (override)"),
  ("_generate_mipmap_chain", "textures.MipTexture2D (override)"),
)


def _sources() -> dict[Path, str]:
  return {p: p.read_text() for p in PACKAGE.rglob("*.py")}


def _headers() -> str:
  return "\n".join(p.read_text(errors="replace") for p in sorted(NATIVE.glob("*.h")))


def test_cpp_private_uses_exist_in_the_installed_warp() -> None:
  text = _headers()
  missing = [name for name in CPP if not re.search(rf"\b{name}\b", text)]
  for header, fields in CPP_FIELDS.items():
    body = (NATIVE / header).read_text()
    missing += [f"{header}:{f}" for f in fields if not re.search(rf"\b{f}\b\s*;", body) and f"{f};" not in body]
  assert not missing, f"Warp {wp.__version__} lacks: {missing}"


def test_python_private_uses_exist_in_the_installed_warp() -> None:
  wp.init()
  assert warp._src.texture.MemoryType.ARRAY and warp._src.texture.MemoryType.HOST
  assert hasattr(warp._src.context, "runtime")
  runtime = warp._src.context.runtime
  assert runtime is not None
  assert all(hasattr(runtime.core, name) for name in CORE)
  assert callable(getattr(wp.Module, "_detach_references", None))
  # viewer/watch wraps it: (module, device, ...) -> True when it compiled, False on a cache hit.
  compile_module = warp._src.context.Module._compile
  assert inspect.signature(compile_module).return_annotation in (bool, "bool")
  assert "compilation was performed" in " ".join((inspect.getdoc(compile_module) or "").split())
  for method in ("_upload_mipmap_chain", "_generate_mipmap_chain"):
    assert callable(getattr(wp.Texture2D, method, None)), method

  @wp.struct
  class _S:
    a: int

  assert list(type(_S())._cls.vars) == ["a"]  # pyrefly: ignore[missing-attribute]  # the private field table
  fields = {f[0]: f[1] for f in warp._src.types.array_t._fields_}
  for word in ("shape", "strides"):
    assert getattr(fields[word], "_type_", None) is ctypes.c_int32


def test_srgb_texture_object_swap_uses_fields_warp_owns() -> None:
  """`MipTexture2D(srgb8=True)` replaces Warp's texture object over Warp's CUDA array: the fields
  it reads and writes are ones Warp's Texture sets and its destructor frees (`_tex_handle`, a
  mipmapped `_array_handle` when `_is_mipmapped`)."""
  init = inspect.getsource(wp.Texture.__init__) + inspect.getsource(wp.Texture.__new__)
  free = inspect.getsource(wp.Texture.__del__)
  for field in ("_tex_handle", "_array_handle", "_is_mipmapped", "_num_mip_levels", "_address_mode_u", "_address_mode_v"):
    assert f"self._{field[1:]}" in init or f"instance.{field}" in init, field
  assert "wp_texture_object_destroy_device(self.device.context, self._tex_handle)" in free


# Patterns of private Python uses of Warp in raywarp's source (each match must be fenced).
_PRIVATE_PY: Final = re.compile(
  r"\b(?:warp|wp)\._src\.[a-z_]+|\._cls\.vars|_detach_references|_(?:upload|generate)_mipmap_chain|\bcore\.(wp_\w+)"
)


def test_every_private_use_is_fenced() -> None:
  """A new private use of Warp (a `warp._src` import, a private attribute, a native runtime call,
  a `wp::` internal) is in the lists above, and every listed one is still used."""
  sources = _sources()
  text = "\n".join(sources.values())
  found = {m.group(1) or m.group(0).replace("wp._src", "warp._src").lstrip(".") for m in _PRIVATE_PY.finditer(text)}
  fenced = {name for name, _ in PYTHON} | set(CORE)
  assert found <= fenced, sorted(found - fenced)
  assert fenced <= found, f"listed but no longer used: {sorted(fenced - found)}"
  cpp = {m for text in sources.values() for m in re.findall(r"\bwp::([A-Za-z_]\w*)", text)}
  cpp |= {m for text in sources.values() for m in re.findall(r"\b(BVH_[A-Z_]+|WP_TILE_[A-Z_]+)\b", text)}
  unfenced = cpp - set(CPP) - CPP_PUBLIC - {"BVH", "Mesh"}
  assert not unfenced, sorted(unfenced)


def test_warp_cap_matches_the_validated_families() -> None:
  """pyproject.toml allows exactly the validated Warp families (>= the first, < the next one)."""
  deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
  [spec] = [d for d in deps if d.startswith("warp-lang")]
  first, last = nodes.VALIDATED_WARP[0], nodes.VALIDATED_WARP[-1]
  major, minor = (int(x) for x in last.split("."))
  assert spec == f"warp-lang>={first},<{major}.{minor + 1}", spec


@pytest.mark.parametrize("version", ["1.17.9", "2.0.0", "unknown"])
def test_unvalidated_family_is_refused(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
  monkeypatch.setattr(wp, "__version__", version)
  with pytest.raises(RuntimeError, match="validated warp-lang"):
    nodes.require_compatible()
