"""Warp module loads, one thread per module at a time.

Warp (1.18) generates code under its codegen lock, but `Module.load` (the compile and load of a
module's binary, at a kernel's first launch or `wp.force_load`) takes no lock of its own:

- two threads loading one module both compile and load it, and the second executable replaces
  the first in the module's table;
- a thread that adds a kernel, an overload, a function or a struct to a module
  (`Module.mark_modified`, which clears the module's hashes) while another thread loads it makes
  that load read hashes that are gone (KeyError in `Module.load`).

Renderers built, warmed and rendered on several threads load the same modules (their kernel
factories add kernels to shared modules), and so does the caller's own Warp code. `install` gives
every Warp module a re-entrant lock held by its loads and its modifications. A modification made
while the thread generates code (Warp's codegen lock held: a `wp.static` expression that defines a
function) does not take it: a load holds its module's lock while it waits for the codegen lock, so
taking both in the other order could deadlock; such a modification is Warp's own case, as before.
Loads of different modules still run in parallel (the warm-up's compile threads).
"""

from __future__ import annotations

import threading
import weakref
from typing import Final

from warp._src import codegen as warp_codegen
from warp._src import context as warp_context

_LOCKS: Final = weakref.WeakKeyDictionary[warp_context.Module, threading.RLock]()
_GUARD: Final = threading.Lock()
_WARP_LOAD: Final = warp_context.Module.load
_WARP_MARK_MODIFIED: Final = warp_context.Module.mark_modified


def module_lock(module: warp_context.Module) -> threading.RLock:
  """The lock of `module`'s loads and modifications (made on first use)."""
  with _GUARD:
    lock = _LOCKS.get(module)
    if lock is None:
      lock = _LOCKS[module] = threading.RLock()
    return lock


def _load(self: warp_context.Module, *args: object, **kwargs: object) -> warp_context.ModuleExec | None:
  """`Module.load` under the module's lock."""
  with module_lock(self):
    return _WARP_LOAD(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]  # Warp's untyped signature


def _mark_modified(self: warp_context.Module) -> None:
  """`Module.mark_modified` under the module's lock (not while generating code: module doc)."""
  # Warp's codegen lock (module doc); `_is_owned` is RLock's (CPython's C RLock has it).
  if warp_codegen._codegen_lock._is_owned():  # noqa: SLF001  # pyrefly: ignore[missing-attribute]
    _WARP_MARK_MODIFIED(self)
    return
  with module_lock(self):
    _WARP_MARK_MODIFIED(self)


def install() -> None:
  """Locks Warp's module loads and modifications per module (idempotent)."""
  warp_context.Module.load = _load
  warp_context.Module.mark_modified = _mark_modified
