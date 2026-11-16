"""MaterialX materials -> MDL modules, through the official MaterialX library (`materialx`) and its
MDL shader generator: the MDL SDK baker (`formats.usd.materials.mdl`) then converts them like any
MDL material, as Omniverse RTX renders MaterialX.

A MaterialX network authored in a stage (shader prims whose `info:id` is a MaterialX nodedef:
`from_shader`) is a `Network`: a MaterialX document over the standard data libraries and its
material. `generate` writes its MDL module (`MaterialXGenMdl`, complete interface) and the
arguments the MDL baker passes it:
- every file input becomes a `texture_2d` parameter, its file and color space passed as the
  argument (the MDL SDK reads it in that space; packaged files through raywarp's decoder);
- color constants authored in another color space are passed converted to linear Rec.709 (the
  generator applies no color management);
- `geompropvalue` reads of the UV primvars (`st`, `st1`) become `texcoord` reads of their UV set.

What the library rejects (an unknown nodedef, an invalid document, a generator error) raises
`MaterialXError`, which the baker reports and renders as its error material. Inputs a nodedef
does not have are reported (`mtlx.input:<node>.<input>`) and left out.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import os
from pathlib import Path
from typing import Final, final

import MaterialX as mx
from MaterialX import PyMaterialXGenMdl as genmdl
from MaterialX import PyMaterialXGenShader as genshader
from pxr import Sdf, Usd, UsdShade

from raywarp.cache.paths import remove_stale, tmp_name
from raywarp.formats.usd import util
from raywarp.formats.usd.materials.context import UV_PRIMVARS, MaterialXError, ShadeContext, set_ext
from raywarp.formats.usd.materials.mdl.load import Input
from raywarp.scene.images import colorspace, files
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model import normal_map
from raywarp.scene.model.material import Material, OpenPbrExtras
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_TEXTURE, Slot, TextureRef, Wrap

# The MDL name of the generated material.
MATERIAL: Final = "mtlx_material"
# The MaterialX output type of a surface shader nodedef.
SURFACE: Final = "surfaceshader"
# MaterialX color spaces the generator's output is already in (no conversion).
_LINEAR: Final = frozenset({"", "lin_rec709"})
_COLOR_TYPES: Final = frozenset({"color3", "color4"})
_RGB: Final = 3
_MODULE_PREFIX: Final = "mx_"
# raywarp's extension inputs of a surface shader prim (not MaterialX inputs: applied after the
# conversion, `extensions`): RaywarpExt constants and the detail normal map.
EXT_INPUTS: Final = ("ext:wrap", "ext:anisotropy_rotation")
DETAIL_NORMAL: Final = "geometry_detail_normal"
# The sheen color of a glTF PBR surface without sheen (`_gltf_sheen`).
_NO_SHEEN: Final = 1e-6
# MaterialX type names: the first token of a nodedef id starting with one ends its node category
# (ND_<category>_<type>[<variant>][_<version>]).
_TYPE_TOKENS: Final = (
  *("float", "integer", "boolean", "color3", "color4", "vector2", "vector3", "vector4", "matrix33", "matrix44"),
  *("string", "filename", "surfaceshader", "displacementshader", "volumeshader", "lightshader", "material"),
  *("BSDF", "EDF", "VDF"),
)
# Sdf value types of shader inputs / outputs -> MaterialX types.
_SDF_TYPES: Final = (
  dict.fromkeys(("float3", "vector3f", "normal3f", "point3f"), "vector3")
  | dict.fromkeys(("string", "token"), "string")
  | {"float": "float", "int": "integer", "bool": "boolean", "color3f": "color3", "color4f": "color4", "float2": "vector2"}
  | {"float4": "vector4", "asset": "filename", "matrix3d": "matrix33", "matrix4d": "matrix44"}
)


@functools.cache
def library() -> mx.Document:
  """The MaterialX standard data libraries (stdlib, pbrlib, bxdf, ...) of the `materialx`
  package."""
  lib = mx.createDocument()
  search = mx.FileSearchPath(str(Path(mx.__file__).parent))
  mx.loadLibraries(mx.getDefaultDataLibraryFolders(), search, lib)
  return lib


def is_surface(shader: UsdShade.Shader) -> bool:
  """Whether a shader prim is a MaterialX surface shader: its `info:id` is a surfaceshader nodedef
  of the standard libraries (OpenPBR, Standard Surface, glTF PBR, UsdPreviewSurface's MaterialX
  form, ...)."""
  node_id = str(shader.GetIdAttr().Get() or "")
  if not node_id.startswith("ND_"):
    return False
  nodedef = library().getNodeDef(node_id)
  return nodedef is not None and nodedef.getType() == SURFACE


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Network:
  """A MaterialX material: its document (over `library()`) and the name of its `surfacematerial`
  node (file values are resolved paths)."""

  document: mx.Document
  material: str


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Generated:
  """The MDL module of a network (material `MATERIAL`) and the arguments its conversion passes."""

  source: str
  inputs: tuple[Input, ...]


def new_document() -> mx.Document:
  """An empty document over the standard data libraries."""
  doc = mx.createDocument()
  doc.setDataLibrary(library())
  return doc


# USD shader networks -> documents --------------------------------------------------------------


def _value_string(value: util.UsdValue, kind: str) -> str:
  """A USD value as the MaterialX value string of type `kind`."""
  if kind == "boolean":
    return "true" if util.as_float(value) else "false"
  if kind == "integer":
    return str(int(util.as_float(value)))
  if kind in ("string", "filename"):
    return str(value)
  return ", ".join(f"{x:.9g}" for x in util.floats(value))


@final
@dataclasses.dataclass(slots=True)
class _UsdReader:
  """Shader prims of one network -> nodes of `doc`, each prim once."""

  doc: mx.Document
  report: LoadReport
  names: dict[Sdf.Path, str] = dataclasses.field(default_factory=dict)
  pending: list[Usd.Prim] = dataclasses.field(default_factory=list)

  def node(self, prim: Usd.Prim) -> str:
    """The name of the node of shader prim `prim` (queued for `read` on first use)."""
    path = prim.GetPath()
    if path not in self.names:
      self.names[path] = self.doc.createValidChildName(prim.GetName())
      self.pending.append(prim)
    return self.names[path]

  def read(self, prim: Usd.Prim) -> None:
    """Adds the node of shader prim `prim`, its inputs set and its sources queued.

    Raises:
      MaterialXError: a shader whose `info:id` is no nodedef of the standard libraries.
    """
    shader = UsdShade.Shader(prim)
    nodedef = _nodedef(shader, self.report)
    node = self.doc.addNodeInstance(nodedef, self.names[prim.GetPath()])
    for inp in shader.GetInputs():
      self._input(node, nodedef, inp)

  def _input(self, node: mx.Node, nodedef: mx.NodeDef, inp: UsdShade.Input) -> None:
    """Sets input `inp` of `node`: a connection, a file or a constant (with its color space)."""
    name = inp.GetBaseName()
    declared = nodedef.getActiveInput(name)
    if name in (*EXT_INPUTS, DETAIL_NORMAL) and nodedef.getType() == SURFACE:
      return  # raywarp's extensions (`extensions`)
    if declared is None:
      self.report.drop(f"mtlx.input:{nodedef.getNodeString()}.{name}", inp.GetAttr())
      return
    value, attr = util.producing(inp)
    if attr is not None and attr.GetName().startswith("outputs:"):
      target = node.addInputFromNodeDef(name)
      target.removeAttribute("value")
      source = attr.GetPrim()
      target.setNodeName(self.node(source))
      # Named unless it is the default "out" of a one-output node: exporters often author only
      # the connected output of a multi-output node (e.g. `outg` of separate3).
      if attr.GetBaseName() != "out" or len(UsdShade.Shader(source).GetOutputs()) > 1:
        target.setOutputString(attr.GetBaseName())
      return
    if value is None:
      return
    kind = declared.getType()
    if isinstance(value, Sdf.AssetPath):
      value = self._file(value, attr)
    target = node.setInputValue(name, _value_string(value, kind), kind)
    space = util.colorspace_name(attr)
    if space and (kind == "filename" or kind in _COLOR_TYPES):
      target.setColorSpace(space)

  def _file(self, value: Sdf.AssetPath, attr: Usd.Attribute | None) -> str:
    """The resolved path of a file value ("" when unresolved or a UDIM set: reported)."""
    if "<UDIM>" in value.path:
      self.report.drop("mtlx.image.udim", attr)
      return ""
    if not value.resolvedPath and value.path:
      self.report.drop("mtlx.image.missing", attr)
    return value.resolvedPath or ""


def _category(node_id: str) -> str:
  """The node category of a nodedef id ("ND_multiply_color3FA" -> "multiply")."""
  tokens = node_id.removeprefix("ND_").split("_")
  end = next((i for i, t in enumerate(tokens) if i > 0 and t.startswith(_TYPE_TOKENS)), len(tokens))
  return "_".join(tokens[:end])


def _sdf_type(prop: UsdShade.Input | UsdShade.Output) -> str:
  """The MaterialX type of a shader input or output ("": no MaterialX type)."""
  return _SDF_TYPES.get(str(prop.GetTypeName()), "")


def _fits(nodedef: mx.NodeDef, shader: UsdShade.Shader) -> bool:
  """Whether `nodedef` declares the shader's output types and every authored input with its
  type."""
  outputs = {o.getName(): o.getType() for o in nodedef.getActiveOutputs()}
  for out in shader.GetOutputs():
    kind = _sdf_type(out)
    if kind and outputs.get(out.GetBaseName(), kind) != kind and nodedef.getType() != kind:
      return False
  for inp in shader.GetInputs():
    declared = nodedef.getActiveInput(inp.GetBaseName())
    kind = _sdf_type(inp)
    if declared is not None and kind and declared.getType() not in (kind, "filename" if kind == "string" else kind):
      return False
  return True


def _nodedef(shader: UsdShade.Shader, report: LoadReport) -> mx.NodeDef:
  """The nodedef of a shader prim's `info:id`: the library's of that name, else (an id of an older
  MaterialX version, such as 1.38's `ND_normalmap`) the nodedef of its category whose types fit
  the shader's (reported `mtlx.nodedef:<id>-><nodedef>`).

  Raises:
    MaterialXError: no nodedef of the standard libraries fits.
  """
  node_id = str(shader.GetIdAttr().Get() or "")
  nodedef = library().getNodeDef(node_id) if node_id.startswith("ND_") else None
  if nodedef is not None:
    return nodedef
  candidates = library().getMatchingNodeDefs(_category(node_id)) if node_id.startswith("ND_") else []
  fitting = [nd for nd in candidates if _fits(nd, shader)]
  if not fitting:
    raise MaterialXError(f"unknown_node:{node_id or shader.GetPrim().GetTypeName()}")
  report.approx(f"mtlx.nodedef:{node_id}->{fitting[0].getName()}", shader.GetPrim())
  return fitting[0]


def from_shader(shader: UsdShade.Shader, report: LoadReport) -> Network:
  """The network of a MaterialX surface shader prim and every shader prim it reads (connections
  through node graphs followed: `util.producing`).

  Raises:
    MaterialXError: a shader of the network is no nodedef of the standard libraries.
  """
  doc = new_document()
  reader = _UsdReader(doc, report)
  surface = reader.node(shader.GetPrim())
  while reader.pending:
    reader.read(reader.pending.pop())
  material = doc.createValidChildName("material")
  node = doc.getNode(surface)
  doc.addMaterialNode(material, node)
  return Network(document=doc, material=material)


# Documents -> MDL --------------------------------------------------------------------------------


def _own_inputs(doc: mx.Document) -> list[mx.Input]:
  """The inputs of the document's own elements (nodes, node graphs and their interfaces; not the
  data library's)."""
  return [e for e in doc.traverseTree() if isinstance(e, mx.Input)]


def _linear_colors(doc: mx.Document, report: LoadReport, where: object) -> None:
  """Converts every color constant authored in another color space to linear Rec.709."""
  for inp in _own_inputs(doc):
    name = inp.getActiveColorSpace()
    if inp.getType() not in _COLOR_TYPES or not inp.getValueString() or name in _LINEAR:
      continue
    space = util.colorspace_named(name, report=report, where=where)
    xs = [float(x) for x in inp.getValueString().split(",")]
    if space is not None and len(xs) >= _RGB:
      xs[:_RGB] = colorspace.to_lin_rec709((xs[0], xs[1], xs[2]), space)
    inp.setValueString(", ".join(f"{x:.9g}" for x in xs))
    inp.setColorSpace("lin_rec709")


def _uv_reads(doc: mx.Document, report: LoadReport, where: object) -> None:
  """`geompropvalue` reads of a UV primvar -> `texcoord` reads of its UV set; other geometric
  properties are reported (the MDL bake reads them as their defaults)."""
  for node in [e for e in doc.traverseTree() if isinstance(e, mx.Node) and e.getCategory() == "geompropvalue"]:
    prop = node.getInputValue("geomprop")
    name = str(prop) if prop is not None else ""
    if node.getType() != "vector2" or name not in UV_PRIMVARS:
      report.drop(f"mtlx.geomprop:{name or '?'}", where)
      continue
    for inp in node.getInputs():
      node.removeInput(inp.getName())
    node.setCategory("texcoord")
    node.setNodeDefString("ND_texcoord_vector2")
    node.setInputValue("index", UV_PRIMVARS.index(name))


def _gltf_sheen(doc: mx.Document) -> None:
  """A glTF PBR surface without sheen (sheen_color 0, its default) gets sheen_color 1e-6: the
  library's `gltf_pbr` graph divides the color by its largest component, and the 0 / 0 makes the
  whole material NaN in the MDL distillation (a 1e-6 sheen renders as none)."""
  for node in [e for e in doc.traverseTree() if isinstance(e, mx.Node)]:  # node graphs' too
    if node.getType() != SURFACE or node.getCategory() != "gltf_pbr":
      continue
    inp = node.getInput("sheen_color")
    if inp is not None and (inp.getConnectedNode() is not None or inp.hasInterfaceName()):
      continue
    value = node.getInputValue("sheen_color")
    if value is None or max(value) <= 0.0:
      node.setInputValue("sheen_color", mx.Color3(_NO_SHEEN, _NO_SHEEN, _NO_SHEEN))


def _color_file(inp: mx.Input) -> bool:
  """Whether file input `inp` feeds a color image (`color3` / `color4` output: MaterialX
  color-manages only those); a node graph's interface input: whether any image reading it does."""
  parent = inp.getParent()
  if isinstance(parent, mx.Node):
    return parent.getType() in _COLOR_TYPES
  if isinstance(parent, mx.NodeGraph):
    readers = [n for n in parent.getNodes() if any(i.getInterfaceName() == inp.getName() for i in n.getInputs())]
    return any(n.getType() in _COLOR_TYPES for n in readers)
  return True


def _files(network: Network, report: LoadReport, where: object) -> dict[str, tuple[str, ColorSpace | None]]:
  """The file inputs (by element path: path and color space), their values cleared (the module
  takes them as arguments). A non-color image (a normal, roughness or metalness map) is data:
  linear, whatever color space the document or the input names; a UDIM set or a file that does
  not exist (after the document's `fileprefix`) is reported and left unbound."""
  out: dict[str, tuple[str, ColorSpace | None]] = {}
  for inp in _own_inputs(network.document):
    if inp.getType() != "filename" or not inp.getValueString():
      continue
    name = inp.getResolvedValueString()
    inp.setValueString("")
    if "<UDIM>" in name:
      report.drop("mtlx.image.udim", where)
      continue
    if not files.is_file(name):  # e.g. a `fileprefix` doubling the folder: the input's default
      report.drop("mtlx.image.missing", f"{util.describe(where)}: {name}")
      continue
    color = _color_file(inp)
    space = util.colorspace_named(inp.getActiveColorSpace(), report=report, where=where) if color else ColorSpace.LIN_REC709
    out[inp.getNamePath()] = (name, space)
  return out


@functools.cache
def _generator() -> tuple[genmdl.MdlShaderGenerator, genshader.GenContext]:
  """The MDL generator and its context (reused: the context caches the library's node
  implementations; the generator object is kept with it, the context alone loses its target)."""
  gen = genmdl.MdlShaderGenerator.create()
  context = genshader.GenContext(gen)
  context.registerSourceCodeSearchPath(mx.FileSearchPath(str(Path(mx.__file__).parent)))
  context.getOptions().shaderInterfaceType = genshader.ShaderInterfaceType.SHADER_INTERFACE_COMPLETE
  return gen, context


def generate(network: Network, report: LoadReport, where: object) -> Generated:
  """The MDL module of `network` and the arguments of its file inputs (see the module docstring;
  reports at `where`). Not thread-safe (one generator context).

  Raises:
    MaterialXError: an invalid document or a generator error.
  """
  doc = network.document
  material = doc.getChild(network.material)
  if material is None:
    raise MaterialXError(f"no_material:{network.material}")
  if not mx.isValidName(network.material):  # the MDL material is `MATERIAL` whatever its name
    material.setName(doc.createValidChildName(mx.createValidName(network.material)))
    report.approx("mtlx.material_name(invalid MaterialX name)", where)
  ok, message = material.validate()  # this material only: the document's other elements do not matter
  if not ok:
    raise MaterialXError(f"invalid_material({message.strip().splitlines()[0][:80] if message else '?'})")
  _uv_reads(doc, report, where)
  _gltf_sheen(doc)
  _linear_colors(doc, report, where)
  textures = _files(network, report, where)
  gen, context = _generator()
  try:
    shader = gen.generate(MATERIAL, material, context)
  except Exception as e:  # the generator raises its own exception types
    raise MaterialXError(f"generate({str(e).strip()[:80]})") from e
  if shader is None:
    raise MaterialXError("generate(no shader)")
  stage = shader.getStage(genshader.PIXEL_STAGE)
  inputs: list[Input] = []
  for block in stage.getInputBlocks().values():
    for i in range(block.size()):
      port = block[i]
      found = textures.get(port.getPath())
      if port.getType().getName() == "filename" and found is not None and found[0]:
        path, space = found
        inputs.append(Input(name=port.getVariable(), value=Sdf.AssetPath(path, path), space=space))
  return Generated(source=stage.getSourceCode(), inputs=tuple(inputs))


def module_file(directory: Path, source: str) -> Path:
  """The module file of generated `source` in `directory` (named by its content; written once)."""
  digest = hashlib.sha256(source.encode()).hexdigest()[:24]
  file = directory / f"{_MODULE_PREFIX}{digest}.mdl"
  if file.is_file():
    with contextlib.suppress(OSError):
      os.utime(file)  # in use: `memo.prune` removes the modules unused for a week
  else:
    directory.mkdir(parents=True, exist_ok=True)
    remove_stale(file)  # the temporary files of crashed writers
    tmp = file.with_name(tmp_name(file))  # this writer's own: concurrent readers see a whole module
    try:
      tmp.write_text(source)
      tmp.replace(file)
    finally:
      tmp.unlink(missing_ok=True)
  return file


# raywarp's extension inputs ------------------------------------------------------------------


def extensions(ctx: ShadeContext, shader: UsdShade.Shader, mat: Material) -> Material:
  """`mat` (a MaterialX surface's conversion) with raywarp's extension inputs of the surface
  prim: `ext:wrap` / `ext:anisotropy_rotation` constants (RaywarpExt), and `geometry_detail_normal`
  when it is a normal map of one image (`ND_normalmap*` over `ND_image_vector3` /
  `ND_tiledimage_vector3`: the core's detail normal, combined by RNM); another graph is reported."""
  inputs = {i.GetBaseName(): i for i in shader.GetInputs()}
  for key in EXT_INPUTS:
    value = util.producing(inputs[key])[0] if key in inputs else None
    if value is not None:
      field = key.removeprefix("ext:")
      x = util.as_float(value)
      mat = set_ext(mat, **{field: min(max(x, 0.0), 1.0) if field == "wrap" else x})
  if DETAIL_NORMAL in inputs:
    ref = _detail_normal(ctx, inputs[DETAIL_NORMAL])
    if ref is None:
      ctx.report.drop("mtlx.geometry_detail_normal(not a normal map of one image)", shader.GetPrim())
    else:
      extras = mat.extras or OpenPbrExtras()
      textures = (*(t for t in extras.textures if t[0] != DETAIL_NORMAL), (DETAIL_NORMAL, ref))
      mat = dataclasses.replace(mat, extras=dataclasses.replace(extras, textures=textures))
  return mat


def _report_unread(ctx: ShadeContext, prim: Usd.Prim, inputs: dict[str, UsdShade.Input], read: set[str]) -> None:
  """Reports the authored inputs of a detail-normal node that are not read (the map is sampled
  with REPEAT wrapping, UV set 0 and a tangent-space unpack): `mtlx.detail_normal.<node>.<name>`."""
  node = str(UsdShade.Shader(prim).GetIdAttr().Get() or prim.GetName())
  for name, inp in inputs.items():
    if name not in read and (inp.GetAttr().HasAuthoredValue() or inp.HasConnectedSource()):
      ctx.report.drop(f"mtlx.detail_normal.{node}.{name}", prim)


def _detail_normal(ctx: ShadeContext, inp: UsdShade.Input) -> TextureRef | None:
  """The detail normal map of a `normalmap` over one (tiled) image (None: another graph); the
  inputs it does not read are reported (`_report_unread`)."""
  attr = util.connected_output(inp)
  if attr is None or not str(UsdShade.Shader(attr.GetPrim()).GetIdAttr().Get() or "").startswith("ND_normalmap"):
    return None
  nm = util.shader_inputs(attr.GetPrim())
  image = util.connected_output(nm["in"]) if "in" in nm else None
  node_id = str(UsdShade.Shader(image.GetPrim()).GetIdAttr().Get() or "") if image is not None else ""
  if image is None or node_id not in ("ND_image_vector3", "ND_tiledimage_vector3"):
    return None
  ins = util.shader_inputs(image.GetPrim())
  path = util.asset_path(util.input_value(ins, "file"))
  texture = ctx.texture(path, Slot.GEOMETRY_NORMAL, ColorSpace.LIN_REC709, (Wrap.REPEAT, Wrap.REPEAT)) if path else NO_TEXTURE
  if texture == NO_TEXTURE:
    return None
  read = {"file", "uvtiling", "uvoffset"} if node_id.startswith("ND_tiled") else {"file"}
  _report_unread(ctx, image.GetPrim(), ins, read)
  _report_unread(ctx, attr.GetPrim(), nm, {"in", "scale"})
  scale = util.input_floats(nm, "scale", (1.0,))
  sx, sy = (scale[0], scale[0]) if len(scale) == 1 else (scale[0], scale[1])
  bias_scale = normal_map.unpack((sx, sy))
  tile = util.input_floats(ins, "uvtiling", (1.0, 1.0)) if node_id.startswith("ND_tiled") else (1.0, 1.0)
  offset = util.input_floats(ins, "uvoffset", (0.0, 0.0)) if node_id.startswith("ND_tiled") else (0.0, 0.0)
  xform = ((tile[0], 0.0, offset[0]), (0.0, tile[1], offset[1]))
  return TextureRef(texture, scale=bias_scale[0], bias=bias_scale[1], uv_xform=xform)
