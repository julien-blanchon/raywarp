"""One content hash per Warp function per process.

Warp names every module's kernel-cache entry by a deep content hash: each kernel's source plus,
recursively, every function it calls (`ModuleHasher`). Warp memoises a function's hash inside one
module's hasher only, so every module that reaches the same shading functions hashes the whole
tree again: on a warm start, where every module loads from the kernel cache, that hashing is most
of the time spent creating a Renderer. `install` keeps the function hashes for the process
(`ModuleHasher.hash_function`): the hash is the same value Warp computes (the function's own
overloads and their sources), keyed by the function object and its overload signatures, so a
function that gains an overload is hashed again. It assumes what Warp hashes besides the sources
(the module constants and `wp.static` values a device function reads) is fixed once a function was
first hashed: raywarp's device functions read only constants of their factories' closures (a new
function per value) and `Final` module constants.
"""

from __future__ import annotations

import threading
import weakref
from typing import Final

from warp._src import context as warp_context

# Function -> (its overload signatures, hash) of every function hashed in this process.
_HASHES: Final[weakref.WeakKeyDictionary[warp_context.Function, tuple[frozenset[str], bytes]]] = weakref.WeakKeyDictionary()
_LOCK: Final = threading.Lock()
_WARP_HASH_FUNCTION: Final = warp_context.ModuleHasher.hash_function


def _signatures(func: warp_context.Function) -> frozenset[str]:
  """The overload signatures Warp's function hash covers."""
  return frozenset(func.user_overloads) | frozenset(func.user_templates)


def _hash_function(self: warp_context.ModuleHasher, func: warp_context.Function) -> bytes:
  """`ModuleHasher.hash_function` (installed as the method: `self` is the hasher), memoised for
  the process (see the module doc)."""
  own = self.function_hashes.get(func)
  if own is not None:
    return own
  signatures = _signatures(func)
  with _LOCK:
    known = _HASHES.get(func)
  if known is not None and known[0] == signatures:
    self.function_hashes[func] = known[1]
    return known[1]
  h = _WARP_HASH_FUNCTION(self, func)
  with _LOCK:
    _HASHES[func] = (signatures, h)
  return h


def install() -> None:
  """Makes Warp's module hashing memoise function hashes for the process (idempotent)."""
  warp_context.ModuleHasher.hash_function = _hash_function
