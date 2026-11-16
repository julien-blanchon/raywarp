"""USD prim path relations shared by the code that scopes a stage by prim subtrees (Newton's
`load_usd` world-local roots, Isaac Lab's clone roots, MaterialX look collections): pure string
rules, no USD import (pxr-free, unlike `formats.usd.util`: the Newton and Isaac Lab adapters import
it without the usd extra)."""


def under(path: str, root: str) -> bool:
  """Whether prim `path` is `root` or one of its descendants (whole path components: "/a/bc" is
  not under "/a/b"; a trailing "/" of `root` is ignored, so "/" holds every absolute path)."""
  root = root.rstrip("/")
  return path == root or path.startswith(root + "/")
