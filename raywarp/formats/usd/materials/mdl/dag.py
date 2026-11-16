"""A distilled material's expressions as `values.py` expressions (affine in texels where they are).

`Dag.value` reads what a texture slot stores without a bake: constants, lookups through affine
coordinates (mesh UVs or the world position), component selection, constructors, affine
arithmetic and the `base::` / distiller helpers around them. Other calls raise
`UnsupportedError` (the caller bakes), unless they do not vary over the surface: the SDK baker
then evaluates them once. The shading frame is the baker's (tangent space, +Z normal).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Final, final

import numpy as np

from raywarp.formats.usd.materials.mdl import ops, sdk
from raywarp.formats.usd.materials.mdl.values import (
  WORLD_SET,
  Lin,
  TexDecl,
  TexTerm,
  UnsupportedError,
  UvTerm,
  Value,
  binop,
  comps,
  const_value,
  is_const,
  lin_range,
  pack,
)
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model.types import Wrap

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.mdl import sdk_types as st

# tex::lookup_<type> -> components; a 2D lookup's arguments (tex, coord, wraps, crops, frame).
LOOKUPS: Final = {
  "tex::lookup_float": 1,
  "tex::lookup_float2": 2,
  "tex::lookup_float3": 3,
  "tex::lookup_float4": 4,
  "tex::lookup_color": 3,
}
_LOOKUP_ARGS: Final = 6
_NORMAL_MAP_ARGS: Final = 9
_WRAPS: Final = {
  "wrap_clamp": Wrap.CLAMP,
  "wrap_repeat": Wrap.REPEAT,
  "wrap_mirrored_repeat": Wrap.MIRROR,
  "wrap_clip": Wrap.BLACK,
}
# The standard modules' calls (their bodies are the SDK's: no hidden state read beyond `_VARYING`).
# Calls whose value varies over the surface (a subtree without them is a constant).
_VARYING: Final = ("state::texture_coordinate", "state::position", "tex::lookup_", "tex::texel_")
# Coordinate-space conversions (from, to, value): the identity between equal spaces; the
# internal space is world space (the compile's `internal_space`).
_SPACES: Final = ("state::transform_point", "state::transform_vector", "state::transform_normal")
_WORLD: Final = {"coordinate_internal": "coordinate_world"}
# The shading frame as the SDK's CPU baker evaluates it (tangent space).
_FRAME: Final = {
  "state::normal": (0.0, 0.0, 1.0),
  "state::texture_tangent_u": (1.0, 0.0, 0.0),
  "state::texture_tangent_v": (0.0, 1.0, 0.0),
}
_UNIT_TOL: Final = 1e-5
_SRGB_GAMMA: Final = 2.2
# Calls with a weight / condition that selects one argument: name -> its index.
_SELECTS: Final = {"math::lerp": 2, "operator?:": 0}
# Pixel types of 8-bit images (texel sums of them are bounded by [0, 1]; float images are not).
_BYTE_PIXELS: Final = frozenset({"Sint8", "Rgb", "Rgba"})
# Compound and resource value kinds (bake keys read through them).
COMPOUNDS: Final = ("VK_VECTOR", "VK_COLOR", "VK_MATRIX", "VK_ARRAY", "VK_STRUCT")
RESOURCES: Final = ("VK_TEXTURE", "VK_LIGHT_PROFILE", "VK_BSDF_MEASUREMENT")
# The calls a read or a walk visits at most (shared subexpressions that are not temporaries are
# visited each time).
MAX_NODES: Final = 20000


@final
class _VaryingError(UnsupportedError):
  """A call outside the subset whose subtree varies over the surface: its callers vary too (not
  walked again)."""


@final
class Dag:
  """The expressions of one distilled material."""

  def __init__(
    self,
    api: sdk.Api,
    transaction: st.ITransaction,
    material: st.ICompiled_material,
    inputs: Mapping[sdk.DbName, tuple[str, ColorSpace | None]],
    *,
    lane: int = 0,
    modules: frozenset[str] = frozenset(),
  ) -> None:
    """A reader of `material` (resources through `transaction`, of lane `lane`; `inputs`: the
    file and authored color space of the textures made for inputs, by DB name; else the image's
    file and the texture's effective gamma; `modules`: the files of its module and the modules
    it imports, which its non-inlined functions' bodies come from)."""
    self.api = api
    self.transaction = transaction
    self.material = material
    self.inputs = inputs
    self.lane = lane
    self.modules = modules
    self._textures: dict[str, tuple[TexDecl, int]] = {}
    self._tiles: dict[str, tuple[str, ...]] = {}  # each texture's files (UDIM: every tile)
    self.floats: set[TexDecl] = set()  # textures of float pixels (not bounded by [0, 1])
    self.notes: set[str] = set()  # approximations of a read, reported by the writer
    self._memo: dict[tuple[int, bool], Value] = {}
    self._values: dict[tuple[str, bool], Value | UnsupportedError] = {}
    # Reading a normal: math::normalize is the identity (the renderer normalizes its texel).
    self.direction = False
    self._count = 0
    self._uniforms: dict[str, bool] = {}  # `_opaque` by call name

  # Navigation ------------------------------------------------------------------------------

  def expr(self, path: str) -> st.IExpression | None:
    """The expression at `path` (temporaries resolved; None when there is none)."""
    e = self.material.lookup_sub_expression(path)
    return self.resolve(e) if e is not None and sdk.valid(e) else None

  def temporary(self, e: st.IExpression) -> int | None:
    """The index of the temporary `e` names (None: not a temporary)."""
    if e.get_kind().name != "EK_TEMPORARY":
      return None
    return e.get_interface(self.api.bindings.IExpression_temporary).get_index()

  def resolve(self, e: st.IExpression) -> st.IExpression:
    """`e`, or the expression of the temporary it names."""
    index = self.temporary(e)
    return e if index is None else self.material.get_temporary(index)

  def call(self, e: st.IExpression) -> tuple[str, st.IExpression_list] | None:
    """(call name: `tex::lookup_color`, arguments) of a direct call, None for another one."""
    if e.get_kind().name != "EK_DIRECT_CALL":
      return None
    c = e.get_interface(self.api.bindings.IExpression_direct_call)
    return c.get_definition().removeprefix("mdl::").split("(", 1)[0], c.get_arguments()

  def node(self, path: str) -> tuple[str | None, dict[str, str]]:
    """(call name, argument name -> path) at `path` (None, {} when it is not a call)."""
    e = self.expr(path)
    c = None if e is None else self.call(e)
    if c is None:
      return None, {}
    name, args = c
    return name, {args.get_name(i): f"{path}.{args.get_name(i)}" for i in range(args.get_size())}

  def walk(
    self, e: st.IExpression, path: str, descend: Callable[[str], bool] = lambda _: True
  ) -> Iterator[tuple[str, st.IExpression_list, str]]:
    """(name, arguments, path) of the calls below `e`, temporaries once, into the arguments of
    the calls `descend` accepts; UnsupportedError past `MAX_NODES` calls."""
    seen: set[int] = set()
    todo = [(e, path)]
    count = 0
    while todo:
      x, p = todo.pop()
      index = self.temporary(x)
      if index is not None:
        if index in seen:
          continue
        seen.add(index)
      c = self.call(self.resolve(x))
      if c is None:
        continue
      count += 1
      if count > MAX_NODES:
        raise UnsupportedError("expression too large")
      name, args = c
      yield name, args, p
      if descend(name):
        todo.extend((args.get_expression(i), f"{p}.{args.get_name(i)}") for i in range(args.get_size()))

  # Values ----------------------------------------------------------------------------------

  def value(self, path: str, *, direction: bool = False) -> Value:
    """The value at `path` (cached; UnsupportedError when a bake is needed). `direction`: a
    normal, read up to its length."""
    key = (path, direction)
    if key not in self._values:
      self._values[key] = self._read(path, direction=direction)
    result = self._values[key]
    if isinstance(result, UnsupportedError):
      raise result
    return result

  def _read(self, path: str, *, direction: bool) -> Value | UnsupportedError:
    """`value` uncached (its error as a value)."""
    e = self.expr(path)
    if e is None:
      return UnsupportedError(f"no expression at {path}")
    self.direction, self._count = direction, 0
    try:
      return self._value(e, path)
    except UnsupportedError as error:
      return error
    finally:
      self.direction = False

  def _value(self, e: st.IExpression, path: str) -> Value:
    """The value of expression `e` found at `path` (temporaries memoised)."""
    index = self.temporary(e)
    key = (-1 if index is None else index, self.direction)
    if key in self._memo:
      return self._memo[key]
    self._count += 1
    if self._count > MAX_NODES:
      raise UnsupportedError("expression too large")
    e = self.resolve(e)
    kind = e.get_kind().name
    if kind == "EK_CONSTANT":
      v = self.ivalue(e.get_interface(self.api.bindings.IExpression_constant).get_value())
    elif kind == "EK_DIRECT_CALL":
      v = self._call(e, path)
    else:
      raise UnsupportedError(f"expression kind {kind}")
    if index is not None:
      self._memo[key] = v
    return v

  def ivalue(self, v: st.IValue) -> Value:
    """A constant: numbers, vectors / colors / matrices (column-major), enums by name, textures."""
    b = self.api.bindings
    kind = v.get_kind().name
    numbers = {"VK_BOOL": b.IValue_bool, "VK_INT": b.IValue_int, "VK_FLOAT": b.IValue_float, "VK_DOUBLE": b.IValue_double}
    if kind in numbers:
      x = float(v.get_interface(numbers[kind]).get_value())
      if math.isnan(x):  # a folded 0 / 0 (glTF PBR's unset sheen color): no defined value, read as 0
        self.notes.add("nan_constant(read as 0)")
        return Lin(c=0.0)
      return Lin(c=x)
    if kind == "VK_ENUM":
      return v.get_interface(b.IValue_enumeration).get_name()
    if kind == "VK_STRING":
      return str(v.get_interface(b.IValue_string).get_value())
    if kind in COMPOUNDS[:3]:
      c = v.get_interface(b.IValue_compound)
      return tuple(x for i in range(c.get_size()) for x in comps(self.ivalue(c.get_value(i))))
    if kind == "VK_TEXTURE":
      return self.declaration(v.get_interface(b.IValue_texture).get_value())[0]
    raise UnsupportedError(f"constant of kind {kind}")

  def declaration(self, name: str) -> tuple[TexDecl, int]:
    """(declaration, largest side) of texture DB element `name` ((TexDecl(''), 0): invalid)."""
    if name not in self._textures:
      b = self.api.bindings
      tex = self.transaction.access_as(b.ITexture, name) if name else None
      image = self.transaction.access_as(b.IImage, tex.get_image()) if tex is not None and sdk.valid(tex) else None
      if tex is None or image is None or not sdk.valid(image):
        self._textures[name], self._tiles[name] = (TexDecl(""), 0), ()
      else:
        tiles = tuple(image.get_filename(0, i) for i in range(max(image.get_frame_length(0), 1)))
        authored, space = self.inputs.get(sdk.DbName(name), (tiles[0], None))
        srgb = abs(tex.get_effective_gamma(0, 0) - _SRGB_GAMMA) < _UNIT_TOL
        decl = TexDecl(authored, space or (ColorSpace.SRGB_REC709 if srgb else ColorSpace.LIN_REC709))
        if image.get_type(0, 0) not in _BYTE_PIXELS:
          self.floats.add(decl)
        side = max(image.resolution_x(0, 0, 0), image.resolution_y(0, 0, 0))
        self._textures[name], self._tiles[name] = (decl, side), (authored,) if sdk.DbName(name) in self.inputs else tiles
    return self._textures[name]

  def tiles(self, name: str) -> tuple[str, ...]:
    """The files of texture DB element `name` (UDIM: every tile; read by `declaration`)."""
    self.declaration(name)
    return self._tiles.get(name, ())

  def read(self) -> list[tuple[TexDecl, int]]:
    """The textures read so far (declaration, largest side)."""
    return list(self._textures.values())

  def _call(self, e: st.IExpression, path: str) -> Value:
    """The value of a direct call (a constant subtree outside the subset: the SDK's value)."""
    c = self.call(e)
    if c is None:
      raise UnsupportedError("not a call")
    name, args = c
    paths = [f"{path}.{args.get_name(i)}" for i in range(args.get_size())]
    exprs = [args.get_expression(i) for i in range(args.get_size())]
    try:
      return self._lookup(exprs, paths, LOOKUPS[name]) if name in LOOKUPS else self._intrinsic(name, exprs, paths)
    except _VaryingError:
      raise
    except UnsupportedError as error:
      if self._varies(e, path):
        raise _VaryingError(str(error)) from error
      values = self.bake_uniform(path)
      if values is None:
        raise
      return pack([Lin(c=x) for x in values])

  def _intrinsic(self, name: str, exprs: list[st.IExpression], paths: list[str]) -> Value:
    """A call that is not a lookup: state, arithmetic, constructors, selection, helpers."""
    if name == "state::texture_coordinate":
      i = int(const_value(comps(self._value(exprs[0], paths[0]))[0]))
      return (Lin({UvTerm(i, 0): 1.0}), Lin({UvTerm(i, 1): 1.0}), Lin())
    if name == "state::position":
      return tuple(Lin({UvTerm(WORLD_SET, k): 1.0}) for k in range(3))
    if name in _FRAME:
      return tuple(Lin(c=x) for x in _FRAME[name])
    if name in _SPACES:
      self._spaces(exprs, paths)
      return self._value(exprs[-1], paths[-1])
    if name == "math::normalize" and self.direction:
      return self._value(exprs[-1], paths[-1])
    if name.endswith("::tangent_space_normal_roughness_texture"):
      return self._normal_map(exprs, paths)
    if name.endswith("::NormalData"):  # OmniPBRBase's (normal, roughness) struct: 5 components
      return tuple(x for e, p in zip(exprs, paths, strict=True) for x in comps(self._value(e, p)))
    if name.endswith(("::NormalData.normal", "::NormalData.roughness")):
      cs = comps(self._value(exprs[0], paths[0]))
      return cs[:3] if name.endswith(".normal") else cs[3:5]
    if name == "base::coordinate_source" and self._value(exprs[0], paths[0]) == "texture_coordinate_object":
      self.notes.add("object_coordinates(as world)")
    if name == "math::sqrt" and self.direction:
      return ops.stored_z(self._value(exprs[0], paths[0]))
    if name == "nvidia::distilling_support::add_detail_normal":
      return self._detail(exprs, paths)
    if name == "nvidia::distilling_support::average" and len({self.hash(p) for p in paths[1::2]}) == 1:
      return self._value(exprs[1], paths[1])  # the distiller's merged tints: one value, any weights
    if name in _SELECTS:
      chosen = self._select(name, exprs, paths)
      if chosen is not None:
        return chosen
    return ops.apply(name, [self._value(x, p) for x, p in zip(exprs, paths, strict=True)], frozenset(self.floats))

  def _spaces(self, exprs: list[st.IExpression], paths: list[str]) -> None:
    """Notes a coordinate-space conversion between two spaces that differ (read as the
    identity: internal space is world space, the object's transform is not applied)."""
    ends = [self._value(e, p) for e, p in zip(exprs[:2], paths[:2], strict=True)]
    named = [_WORLD.get(str(x), str(x)) for x in ends]
    if named[0] != named[1]:
      self.notes.add(f"coordinate_space.{named[0]}_to_{named[1]}(as the identity)")

  def _detail(self, exprs: list[st.IExpression], paths: list[str]) -> Value:
    """add_detail_normal(nd, n): the other one when one is the shading normal; read as a
    direction, the reoriented normal mapping of both (`detail.rnm`'s arithmetic, unnormalised: a
    per-texel value, split into its two maps by `detail.split` or baked)."""
    values = [comps(self._value(e, p)) for e, p in zip(exprs[:2], paths[:2], strict=True)]
    for keep, other in ((1, 0), (0, 1)):
      v = values[other]
      if all(is_const(x) and abs(const_value(x) - u) < _UNIT_TOL for x, u in zip(v, _FRAME["state::normal"], strict=True)):
        return self._value(exprs[keep], paths[keep])
    if not self.direction or any(len(v) != len(_FRAME["state::normal"]) for v in values):
      raise UnsupportedError("a detail normal over a perturbed normal")
    return ops.rnm(values[0], values[1])

  def _select(self, name: str, exprs: list[st.IExpression], paths: list[str]) -> Value | None:
    """lerp(a, b, t) at a constant t of 0 / 1, `c ? a : b` at a constant c: the branch taken.
    `0 < a * x ? a : 0` (the distiller's metal weight) is a."""
    k = _SELECTS[name]
    if name == "operator?:" and self._guarded(paths):
      return self._value(exprs[1], paths[1])
    t = comps(self._value(exprs[k], paths[k]))
    if not all(is_const(x) for x in t) or len({const_value(x) for x in t}) != 1:
      return None
    c = const_value(t[0])
    i = (1 if c else 2) if name == "operator?:" else {0.0: 0, 1.0: 1}.get(c)
    return None if i is None else self._value(exprs[i], paths[i])

  def _guarded(self, paths: list[str]) -> bool:
    """Whether `cond ? a : b` is `0 < a * x ? a : 0` with x > 0 and a >= 0 (then equal to a)."""
    cond, args = self.node(paths[0])
    if cond != "operator<" or self.node(args.get("y", ""))[0] != "operator*":
      return False
    for p in (args["x"], paths[2]):
      e = self.expr(p)
      if e is None or e.get_kind().name != "EK_CONSTANT" or not ops.is_zero(self._value(e, p)):
        return False
    factors = list(self.node(args["y"])[1].values())
    others = [p for p in factors if self.hash(p) != self.hash(paths[1])]
    if len(others) != 1 or len(factors) != len(others) + 1:
      return False
    return self._positive(others[0], strict=True) and self._positive(paths[1], strict=False)

  def _positive(self, path: str, *, strict: bool) -> bool:
    """Whether every component of the value at `path` is > 0 (strict) or >= 0 everywhere."""
    try:
      cs = comps(self.value(path))
    except UnsupportedError:
      return False
    for c in cs:
      floats = isinstance(c, Lin) and any(isinstance(k, TexTerm) and k.tex in self.floats for k in c.terms)
      r = lin_range(c) if isinstance(c, Lin) and not floats else None
      if r is None or (r[0] <= 0.0 if strict else r[0] < 0.0):
        return False
    return True

  def _normal_map(self, exprs: list[st.IExpression], paths: list[str]) -> Value:
    """OmniPBRBase::tangent_space_normal_roughness_texture(texture, factor, flip_u, flip_v, uvw,
    crop_u, crop_v, wrap_u, wrap_v, ...) of a regular normal map (not OV's octahedron packing):
    (rgb x 2 - 1 scaled by the factor, flipped, in the baker's tangent frame; roughness 0)."""
    if len(exprs) < _NORMAL_MAP_ARGS:
      raise UnsupportedError("tangent_space_normal_roughness_texture arguments")
    lookup = (0, 4, 7, 8, 5, 6)  # as tex::lookup's (texture, coord, wraps, crops)
    rgb = comps(self._lookup([exprs[i] for i in lookup], [paths[i] for i in lookup], 3))
    factor, flip_u, flip_v = (const_value(comps(self._value(exprs[i], paths[i]))[0]) for i in (1, 2, 3))
    scales = (-factor if flip_u else factor, -factor if flip_v else factor, 1.0)
    unpacked = [binop("+", binop("*", c, Lin(c=2.0 * k)), Lin(c=-k)) for c, k in zip(rgb, scales, strict=True)]
    return (*[comps(v)[0] for v in unpacked], Lin(), Lin())

  def _lookup(self, exprs: list[st.IExpression], paths: list[str], n: int) -> Value:
    """tex::lookup_*(tex, coord, wrap_u, wrap_v, crop_u, crop_v, frame): n texel channels."""
    if len(exprs) < _LOOKUP_ARGS:  # cube / 3D lookups
      raise UnsupportedError("volume or cube lookup")
    tex, coord, *rest = (self._value(x, p) for x, p in zip(exprs[:6], paths[:6], strict=True))
    wraps, crops = rest[:2], [comps(c) for c in rest[2:]]
    if not isinstance(tex, TexDecl):
      raise UnsupportedError("a lookup of a non-texture")
    if not tex.path:
      self.notes.add("texture_unreadable(read as 0)")
      return pack([Lin()] * n)
    if not all(isinstance(w, str) and w in _WRAPS for w in wraps):
      raise UnsupportedError("texture wrap mode")
    if any(not all(is_const(x) for x in c) or [const_value(x) for x in c] != [0.0, 1.0] for c in crops):
      raise UnsupportedError("cropped lookup")
    wrap = (int(_WRAPS[str(wraps[0])]), int(_WRAPS[str(wraps[1])]))
    return pack([Lin({TexTerm(tex, k, ops.coord_map(coord), wrap): 1.0}) for k in range(n)])

  def _varies(self, e: st.IExpression, path: str = "") -> bool:
    """Whether `e` calls a varying state function or a lookup of a valid texture (or is too
    large to tell)."""
    try:
      return any(self._varying(name, args, p) for name, args, p in self.walk(e, path))
    except UnsupportedError:
      return True

  def _varying(self, name: str, args: st.IExpression_list, path: str) -> bool:
    """Whether a call varies over the surface: a varying state function, a lookup of a valid
    texture (an invalid one reads 0 everywhere), or a function that is not an SDK intrinsic
    (semantic DS_UNKNOWN: plain MDL, the vMaterials and `::nvidia::` helpers included) and not
    declared uniform (its body, not in the DAG, may read the state: a looped pattern). Calls
    without a module (`float3`, `operator*`) are the language's own."""
    if not name.startswith(_VARYING):
      return "::" in name and self._opaque(name, path)
    if not name.startswith(("tex::lookup_", "tex::texel_")) or not path or args.get_size() < 1:
      return True
    try:
      tex = self._value(args.get_expression(0), f"{path}.{args.get_name(0)}")
    except UnsupportedError:
      return True
    return not isinstance(tex, TexDecl) or bool(tex.path)

  def _opaque(self, name: str, path: str) -> bool:
    """Whether the call `name` at `path` is a plain MDL function (semantic DS_UNKNOWN: a body
    the DAG does not show) not declared uniform (`is_uniform`), memoised by name (overloads
    share their module and kind)."""
    if name not in self._uniforms:
      e = self.expr(path)
      c = None if e is None else e.get_interface(self.api.bindings.IExpression_direct_call)
      definition = c.get_definition() if c is not None else ""
      fn = self.transaction.access_as(self.api.bindings.IFunction_definition, definition) if definition else None
      user = fn is not None and sdk.valid(fn) and fn.get_semantic().name == "DS_UNKNOWN"
      self._uniforms[name] = user and fn is not None and not fn.is_uniform()
    return self._uniforms[name]

  def varies(self, path: str) -> bool:
    """Whether the expression at `path` varies over the surface (its structure: a varying state
    function or a valid texture's lookup below it)."""
    e = self.expr(path)
    return e is not None and self._varies(e, path)

  def bake_uniform(self, path: str) -> tuple[float, ...] | None:
    """The value at `path` evaluated once by the SDK's CPU baker (None when it cannot bake)."""
    baker = self.api.distiller.create_baker(self.material, path, self.api.bake_on_cpu)
    if baker is None or not sdk.valid(baker):
      return None
    canvas = self.api.images.create_canvas(baker.get_pixel_type(), 1, 1)
    if baker.bake_texture(canvas, 0.0, 1.0, 0.0, 1.0, 0.0, 1) != 0:
      return None
    return tuple(float(x) for x in np.asarray(canvas.get_tile().get_data_numpy(), np.float64).reshape(-1))

  # Bake inputs -----------------------------------------------------------------------------

  def hash(self, path: str) -> tuple[int, int, int, int]:
    """The SDK's hash of the expression at `path`: equal for equal expressions of this material
    (not across sessions: `key`)."""
    u = self.material.get_sub_expression_hash(path)
    return u.m_id1, u.m_id2, u.m_id3, u.m_id4
