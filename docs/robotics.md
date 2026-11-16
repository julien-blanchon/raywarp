# Robotics recipes

Patterns for training pipelines: observations into torch with no copies, domain randomisation
per world, one CUDA graph per step. The snippets share one small MuJoCo scene: a ball on a
textured table, two distractors and a light.

```python
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco

XML = """
<mujoco>
  <asset>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.6 0.4 0.2" rgb2="0.4 0.25 0.1" width="32" height="32"/>
    <material name="table" texture="wood" texrepeat="3 3"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.5 -0.5 2" dir="-0.2 0.2 -1" castshadow="true"/>
    <geom name="table" type="box" size="1 1 0.02" pos="0 0 -0.02" material="table"/>
    <geom name="distractor_cube" type="box" size="0.08 0.08 0.08" pos="0.5 0.4 0.08" rgba="0.2 0.7 0.3 1"/>
    <geom name="distractor_ball" type="sphere" size="0.08" pos="-0.5 0.4 0.08" rgba="0.8 0.8 0.2 1"/>
    <body name="object" pos="0 0 0.1">
      <freejoint/>
      <geom name="object" type="sphere" size="0.1" rgba="0.9 0.2 0.1 1"/>
    </body>
    <camera name="overhead" pos="0 -1.6 1.4" xyaxes="1 0 0 0 0.66 0.75" fovy="45"/>
  </worldbody>
</mujoco>
"""
NWORLD = 16
mjm = mujoco.MjModel.from_xml_string(XML)
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=NWORLD)
mjw.forward(m, d)
scene = rw.integrations.mujoco.scene(mjm, m)
```

## Observations into torch, with no copies

`RenderOutputs` holds device arrays written in place every frame; `out.torch()` returns
zero-copy torch views of them, created once (the same memory every frame).

```python
# docs: requires torch
r = rw.Renderer(
  scene,
  nworld=NWORLD,
  cameras=["overhead"],
  resolution=(64, 64),
  settings=rw.RenderSettings().replace(rw.FAST).replace(**{"outputs": rw.Outputs.of("rgb", "depth")}),
)
obs = r.render(d).torch()
rgb = obs["rgb"][..., :3]  # (nworld, ncam, H, W, 3) uint8 torch view of the RGBA bytes
depth = obs["depth"]  # (nworld, ncam, H, W) float32
```

Every output has one name, the one requested (`out.rgb`, `out.depth`, ...), and one axis order:
world-major, `(nworld, ncam, H, W)` with the per-pixel value as the element type (a trailing
channel axis in torch / NumPy). `ColorOutput` picks the color's storage, never its axis order:

| `ColorOutput(...)` | `out.rgb` (Warp) | `out.torch()["rgb"]` | bytes / pixel |
|---|---|---|---|
| `()` (default) | `(nworld, ncam, H, W)` vec4ub RGBA | `(nworld, ncam, H, W, 4)` uint8 | 4 |
| `(channels=3)` | `(nworld, ncam, H, W)` vec3ub RGB | `(nworld, ncam, H, W, 3)` uint8 | 3 |
| `(channels=3, planar=True)` | `(nworld * ncam, 3, H, W)` uint8 planes | `(nworld, ncam, 3, H, W)` uint8 | 3 |
| `(channels=3, planar=True, dtype="float16")` | `(nworld * ncam, 3, H, W)` float16 in [0, 1] | `(nworld, ncam, 3, H, W)` float16 | 6 |

The planar Warp view merges the world and camera axes (Warp arrays have at most four axes): row
`w * ncam + c`, still world-major. A policy that wants channels-first float input binds its own tensor
(`bind_outputs`: any contiguous array with the required element count, dtype and device) and
the kernel writes it directly:

```python
# docs: requires torch
import torch

planar = rw.settings.ColorOutput(channels=3, planar=True, dtype="float16")
r = rw.Renderer(
  scene,
  nworld=NWORLD,
  cameras=["overhead"],
  resolution=(64, 64),
  settings=rw.RenderSettings().replace(rw.FAST).replace(outputs=rw.Outputs(rgb=planar)),
)
policy_input = torch.zeros((NWORLD, 3, 64, 64), dtype=torch.float16, device=str(r.device))
r.bind_outputs(rgb=wp.from_torch(policy_input.view(-1)))  # the kernel writes the policy's tensor
r.render(d)
assert r.outputs.torch()["rgb"].shape == (NWORLD, 1, 3, 64, 64)
```

The mjlab backend (`raywarp.integrations.mjlab`) renders `ColorOutput(channels=3)`: the RGB bytes are
written straight into the world-major tensor mjlab reads, each sensor a `(nworld, H, W, 3)`
view of it. Configure its sensors with `RaywarpCameraSensorCfg`
(`raywarp_settings`, default `RenderSettings()`, must agree with the sensor's `use_shadows` /
`use_textures`; `raywarp_per_world`); sensors of one geom-group set share one scene.

JAX (MJX, MuJoCo Playground): `raywarp.integrations.mujoco.mjx.feed(d, mjx_data)` copies the poses a JAX
state holds into the `mjw.Data` the renderer is bound to (one device copy per field: JAX arrays
are new every step, `d`'s stay put, so a captured render graph keeps replaying), and
`raywarp.integrations.mujoco.mjx.to_jax(r.outputs)` views the outputs as JAX arrays. Both go through DLPack's
stream exchange: no host synchronisation.

Per-camera output views retain the underlying allocation. Rebinding validates all supplied
buffers before changing ownership; a rejected binding leaves the previous outputs intact.

## Outputs: what you can request and what it costs

Ask for exactly the images the policy reads: `rw.Outputs` has one field per output, its spec
when requested and `None` otherwise. Requested outputs decide what is allocated and written (whether
another output set compiles: [settings](settings.md#compiled-in-or-runtime)); the CHW color path also retains packed RGBA, so its memory cost includes both.
Automatic exposure (the default; `exposure.mode="manual"` turns it off) also writes an internal
HDR image it meters (float16 sqrt-encoded, 6 B per pixel and world; the `hdr` output when
requested). An output costs little more than its stores.

<!-- SHOWCASE: robotics/outputs.webp (docs/assets/SHOTS.md) -->
![Every output of one frame](assets/robotics/outputs.webp)

```python
outputs = rw.Outputs(
  rgb=rw.settings.ColorOutput(),  # display sRGB, (nworld, ncam, H, W) vec4ub
  depth=rw.settings.DepthOutput(convention="planar", miss="inf", dtype="float16"),
  seg=rw.settings.IdOutput(),  # instance ids
  semantic=rw.settings.IdOutput(),  # class ids (Renderer.set_semantics), in the same frame
  normal=rw.settings.NormalOutput(frame="geometric", dtype="float16"),
)
r = rw.Renderer(
  scene,
  nworld=NWORLD,
  cameras=["overhead"],
  resolution=(64, 48),
  settings=rw.RenderSettings().replace(rw.FAST).replace(outputs=outputs),
)
out = r.render(d)
assert out.depth.dtype == wp.float16 and out.albedo is None  # not requested: None
# The same request by name with default specs, or from plain data (a JSON config, a CLI flag):
assert rw.Outputs.of("rgb", "seg") == rw.Outputs.parse({"rgb": {}, "seg": {}}) == rw.Outputs.parse("rgb,seg")
```

| output | spec (options) | holds | dtype |
|---|---|---|---|
| `rgb` | `ColorOutput(channels, dtype, planar, alpha)` | display color: RGBA, RGB or channels-first planes (uint8 or float16); `alpha=True`: coverage in the alpha byte | vec4ub |
| `depth` | `DepthOutput(convention, miss, dtype)` | planar or radial depth; misses `"zero"`, `"inf"` or `"far"` | float32 / 16 |
| `seg` | `IdOutput()` | segment (instance) id, -1 on a miss | int32 |
| `semantic` | `IdOutput()` | class id (`set_semantics`; 1 unlabelled), 0 on a miss | int32 |
| `material` | `IdOutput()` | material id, -1 without one | int32 |
| `position` | `FloatOutput(dtype)` | world-space hit point [m] | vec3 / vec3h |
| `uv` | `FloatOutput(dtype)` | the texture coordinate the maps read (mesh / flex / primitive UVs with the material tiling, or its world projection) | vec2 / vec2h |
| `normal` | `NormalOutput(frame, dtype)` | unit normal facing the camera: `"shading"` (smooth, normal maps) or `"geometric"` | vec3 / vec3h |
| `albedo` | `FloatOutput(dtype)` | linear base color | vec3 / vec3h |
| `roughness`, `metallic` | `FloatOutput(dtype)` | as shaded (textured; roughness after specular AA) | float32 / 16 |
| `emission` | `FloatOutput(dtype)` | the emission term of the hit's radiance | vec3 / vec3h |
| `direct`, `indirect` | `FloatOutput(dtype)` | radiance from the lights / from everything else but emission (ambient, environment, GI, reflections) | vec3 / vec3h |
| `shadow` | `ShadowOutput(light, dtype)` | visible fraction of one light (an index, `"sun"`, `"strongest"`) | float32 / 16 |
| `hdr` | `FloatOutput(dtype)` | linear radiance before exposure (averaged like rgb) | vec3 / vec3h |
| `luminance` | `LuminanceOutput(dtype)` | Rec. 709 luminance of the display color, sRGB: a grey camera | uint8 / float16 |
| `motion` | `MotionOutput(status)` | screen-space motion of the first hit to the previous frame [px] | vec2 |
| `flow_status` | `MotionOutput(status=True)` | why each pixel's motion holds or not: `RenderOutputs.FlowStatus` (below) | uint8 |

What the values mean:

- **First hit**: the nearest surface of the pixel-center ray (translucent layers count;
  alpha-cut texels do not). Depth, labels, position, UVs, normals, material values, motion and
  radiance-split outputs refer to that same surface, independent of color multisampling.
- **Anti-aliasing** averages and resolves rgb, hdr and luminance only. The first-hit outputs are written every frame from one sample and never blended (a
  boundary pixel holds one surface's id, depth or normal, never a mix).
- **Misses** write 0, except seg and material (-1), shadow (1) and depth (its `miss` convention).
- **direct + indirect + emission = hdr** on the pixels that hit a surface, to float rounding,
  when the render is opaque (no translucent layer in front), unfogged and at `aa.spp=1`: with GI,
  environment light and glossy / mirror reflections too (the reflection counts in indirect). Fog
  (applied after the shading), translucent layers (composited) and supersampling (hdr averages
  the samples, the split is the pixel center's) break it, and misses hold the background
  in hdr only (`tests/render/test_outputs.py`). They need lit surfaces
  (`surfaces.shading="pbr"`).
- **shadow** traces the light's shadow rays once more at the first hit (one visibility query): 0
  where the surface faces away from the light, 1 for a light that casts no shadow or with shadows
  off.
- **Motion status** (`MotionOutput(status=True)` adds `flow_status`; the values are
  `rw.RenderOutputs.FlowStatus`): `VALID` the point was visible at the motion's target in the
  previous frame; `MISS` background; `OUT_OF_FRAME` outside the previous image or beyond its far
  plane (the motion still points there); `BEHIND` behind the previous camera's near plane (motion
  0); `OCCLUDED` hidden by a nearer surface in the previous frame (a disocclusion: its previous
  depth around the target, within one pixel, is nearer by more than that neighbourhood's depth
  spread plus the float16 precision); `UNKNOWN` no reprojection (the first frame after a reset,
  deformable (flex) hits, moving ray-bundle cameras). It costs 1 byte per pixel and two float16
  previous depths per pixel; nothing when not requested.
- **float16** stores the float32 value rounded to half precision (values above 65504 become inf):
  half the bytes and half the memory traffic, the same compute.

**Geometry-only renders.** When nothing shaded is requested (only depth, seg, semantic,
position, uv, material, a geometric normal and motion), the kernel traces one closest hit per
pixel (alpha cut-outs included) and writes them: no shading, light, environment or texture code
is built, and the kernel takes no material, light or environment table.

```python
geometry = rw.Outputs.of("depth", "seg", "position", normal=rw.settings.NormalOutput(frame="geometric"))
r = rw.Renderer(
  scene,
  nworld=NWORLD,
  cameras=["overhead"],
  resolution=(64, 48),
  settings=rw.RenderSettings().replace(rw.FAST).replace(outputs=geometry),
)
out = r.render(d)
assert not r.features.any_rgb  # the geometry-only kernel
```

**What they cost.** Each output added alone to `rgb`, RTX PRO 6000, OptiX, ms per step (every
camera of every world) and renderer memory; G1 = the Unitree G1 scene of the benchmarks, 32
worlds x 2 cameras at 512 x 512 (64 cameras); Office = the Isaac Office, 16 cameras at 1024 x 1024:

| output | G1 512², `FAST` (rgb 1.29 ms) | G1 512², default (rgb 4.05 ms) | Office 1024², `FAST` (rgb 8.87 ms) | memory (G1 / Office) |
|---|---|---|---|---|
| alpha | +0.01 ms | -0.01 ms | -0.06 ms | 0 |
| depth / depth16 | +0.05 / +0.03 ms (+4 / +3 %) | +0.02 / +0.00 ms | 0.00 / -0.03 ms | 64 / 32 MB |
| seg, semantic, material | +0.05 ms (+4 %) each | +0.01-0.04 ms | 0.0 ms | 64 MB each |
| seg + semantic | +0.09 ms (+7 %) | +0.05 ms (+1 %) | +0.05 ms (+1 %) | 128 MB |
| hdr / hdr16 | +0.06 / +0.09 ms (+4 / +7 %) | +0.05 / +0.07 ms | +0.01 / -0.01 ms | 192 / 96 MB |
| albedo, normal, position | +0.05 ms (+4 %) each | +0.02-0.04 ms | +0.02-0.09 ms (+1 %) | 192 MB each |
| normal16 / position16 | +0.07 / +0.08 ms (+6 / +7 %) | +0.07 / +0.05 ms | +0.05 / 0.00 ms | 96 MB each |
| uv | +0.20 ms (+15 %) | +0.12 ms (+3 %) | +0.24 ms (+3 %) | 128 MB |
| roughness, metallic | +0.04 ms (+3 %) | 0.00 ms | -0.02 ms | 64 MB each |
| emission, direct, indirect | +0.05 ms (+4 %) each | +0.02-0.04 ms | 0.00 ms | 192 MB each |
| shadow | +0.31 ms (+24 %) | +0.36 ms (+9 %) | +0.24 ms (+3 %) | 64 MB |
| luminance / luminance16 | +0.01 / +0.03 ms | +0.03 / +0.01 ms | -0.05 / -0.03 ms | 16 / 32 MB |
| motion | +0.33 ms (+26 %) | +0.35 ms (+9 %) | +0.52 ms (+6 %) | 128 MB |
| rgb planar uint8 / float16 (with RGBA) | +0.07 / +0.08 ms (+6 %) | +0.04 / +0.07 ms | -0.01 / +0.01 ms | 48 / 96 MB |

Reading it: an output that is a store of a value the shading already has costs its write, a few
percent of a light frame and nothing measurable on a heavy one (the Office); `uv` looks up the
mesh's texture coordinates again, `shadow` traces the light's shadow rays once more, and `motion`
switches on the temporal state (previous poses, reprojection). float16 halves the memory and the
bytes written; at these sizes the step is not bandwidth-bound, so it does not make the step
faster (a 6-byte vec3h store is no cheaper than a 12-byte one).

The geometry-only kernel (RTX PRO 6000, depth + seg, ms per step): G1 4096 worlds x 2 cameras at
64 x 64 2.56 -> 2.40 ms with the same register cap, **1.72 ms** with its own default
(geometry-only OptiX builds always use 64 registers, whatever cap the shaded builds
have: 1.49x faster than before);
Office 16 x 1080p 1.57 -> 1.49 ms; aloha_clutter 1024 x 128 x 128 0.82 -> 0.78 ms. At the
training sizes a geometry render costs a fraction of a color one: Office 16 x 1024² depth + seg
0.81 ms (`FAST` color: 8.87 ms), G1 64 x 512² 0.88 ms (1.29 ms).

## Point clouds and per-world labels

`r.point_cloud(npoints)` turns the last render's depth output into world-space points, one cloud
per world, every camera fused (or `cameras=[...]`): each hit pixel unprojected through its camera
ray (pose, pinhole or orthographic projection, intrinsics, lens distortion; planar or radial
depth, any miss value), on the device with no host synchronisation (`core/outputs/pointcloud.py`). With
the geometry renderer above:

```python
from raywarp.core.outputs.semantic_classes import body_instances

ball = body_instances(mjm, "object")  # the geoms of the body "object" and of its children
cloud = r.point_cloud(256, labels="seg", keep=ball, seed=0)
assert cloud.points.shape == (NWORLD, 256) and (cloud.count.numpy() > 0).all()  # (nworld, N) vec3 [m]
cloud = r.point_cloud(256, labels="seg", keep=ball, seed=0, out=cloud)  # written in place
```

<!-- SHOWCASE: robotics/point_cloud.webp (docs/assets/SHOTS.md) -->
![A fused point cloud of two cameras, coloured by segment](assets/robotics/point_cloud.webp)

`npoints=N` resamples deterministically (stratified over the kept pixels, one random offset per
world from `seed`; a world keeping fewer than N repeats its points, `count` says how many are
distinct); `npoints=None` returns every kept pixel, compacted, zeros past `count`.
`labels="seg"` / `"semantic"` returns each point's id (`cloud.ids`) and lets `keep` filter on it;
`colors=True` the display color's sRGB bytes (needs `rgb`). With `out=` (a cloud of the same
request) nothing is allocated, so a captured graph can run it after the render.

**Per-world semantic labels.** `r.set_semantics(rules, worlds=...)` (world indices or a
`(nworld,)` bool mask) labels the instances of those worlds only; the other worlds keep theirs and
one label list serves every world (an id means the same class everywhere; new labels get the next
ids; a call without `worlds` relabels every world and restarts the numbering). The first
per-world call replaces the shared class table by a per-world one (captured graphs become stale:
call it before capturing); later calls write it in place.

```python
r.set_semantics({"target": ball})  # every world
r.set_semantics({"target": "distractor_cube", "ball": ball}, worlds=np.arange(NWORLD) % 2 == 1)
assert r.semantic_labels == ("background", "unlabelled", "target", "ball")  # one list, every world
```

## Domain randomisation per world

The full guide (every randomisable quantity, each adapter, GI, costs, gaps) is
[domain_randomisation.md](domain_randomisation.md).

Declare at construction what varies per world (`rw.PerWorld`): the declaration decides what is
compiled in and sizes the tables once, so writing the tables later never recompiles and never
reconfigures. `r.per_world[field]` returns the `(nworld, n)` device table the kernels read,
filled with the current values: write it in place (`.assign(numpy or warp array)`, or through
`wp.to_torch`), between frames or inside a captured graph. An instance color
(`instance.color`) multiplies its material's color (host scenes: white shows the material as
is); MuJoCo scenes keep MuJoCo's rule: the default geom rgba (0.5, 0.5, 0.5, 1) shows the
material's color, any other replaces it. The material fields are OpenPBR's (docs/scenes.md, "The
material"): `specular_weight`, `specular_ior`, the coat inputs and `emission_luminance` are
derived into the device's reflectance, roughness, coat bytes and emission every frame. A
per-world emission on an emissive panel changes only the visible surface, not its fitted rect
light or the GI.

For prepared arrays shared with a simulator or several renderers, use
`r.bind_per_world(field, array)` after declaring the field. The array must match the declared
table's shape, dtype and device and be contiguous. The renderer retains it without copying or
reading values back. Changing this pointer invalidates captured graphs; ordinary in-place updates
do not. Reset affected histories after a material or lighting change, including a rebind.

```python
per_world = rw.PerWorld(
  material=["base_color", "specular_roughness", "tex_variation"],  # material tables (shared scenery stays shared)
  light=["strength", "color"],
  env=["exposure"],
  visible=["distractor_*"],  # shown / hidden per world, still shared
)
r = rw.Renderer(scene, nworld=NWORLD, cameras=["overhead"], resolution=(64, 64), per_world=per_world)
r.render(d)
features = r.features
rng = np.random.default_rng(0)

color = r.per_world["material.base_color"]  # (nworld, nmaterial) vec4: RGB tints
tint = color.numpy()
tint[..., :3] = rng.uniform(0.2, 1.0, tint[..., :3].shape)
color.assign(tint)
roughness = r.per_world["material.specular_roughness"]  # (nworld, nmaterial) float
roughness.assign(rng.uniform(0.2, 1.0, roughness.shape))
light = r.per_world["light.strength"]  # (nworld, nlight)
light.assign(light.numpy() * rng.uniform(0.5, 1.5, light.shape))
r.per_world["exposure"].assign(r.per_world["exposure"].numpy() * rng.uniform(0.7, 1.4, NWORLD))
shown = r.per_world.columns("instance.visible", "distractor_*")  # (nworld, 2) int columns: 0 hides
shown.assign(rng.integers(0, 2, (NWORLD, 2)).astype(np.int32))

tex = r.per_world["material.tex_variation"]  # (nworld, nmaterial) vec4: u, v offset, UV rotation, hue
tex.assign(rw.randomize.jitter(tex.numpy(), rng, uv_offset=1.0, uv_rotation=np.pi, hue=0.3))

r.render(d)
assert r.features == features  # nothing recompiled
```

What each kind of declaration costs (`core/pipeline/worlds/sharing.py`; 4096 worlds: every table at
once +10-13 % frame time on an RTX PRO 6000, +2-5 % on an L4):

| per world | declare | cost |
| --- | --- | --- |
| material (OpenPBR `base_color`, `base_metalness`, `specular_roughness`, `specular_weight`, `specular_ior`, coat inputs, `emission_luminance`; `mirror`) / instance color | `PerWorld(material=[...])` | one `(nworld, n)` table (derived inputs: plus their slots and one derive kernel per slot and frame); sharing unchanged |
| texture placement and hue (`tex_variation`), texel scale / bias (`tex_affine`) | `PerWorld(material=[...])` | 16 (320) bytes per world and material; no texture copy (93-97 % smaller than copies) |
| light intensity, color, position, direction, shape | `PerWorld(light=[...])` | one `(nworld, nlight)` table |
| camera fovy / intrinsics / lens distortion | `PerWorld(camera=[...])` | one `(nworld, ncam)` table |
| environment map choice, intensity, yaw; exposure; ambient; fog | `PerWorld(env=[...])` | one table each |
| visibility of an instance | `PerWorld(visible=[patterns])` | an int table; the instance stays shared (a mask tested where rays meet it) |
| pose / size / mesh of a static instance | `PerWorld(pose=..., size=..., mesh=...)` | the instance leaves the shared scenery: one tree entry and one pose per world |
| what the simulation moves | nothing to declare | per world, as the simulator says |

Rules worth knowing:

- **Colors are RGB tints.** A declared `base_color` / instance `color` varies RGB only; the
  alpha stays the scene's, so no translucency is compiled in. `PerWorld(..., alpha=True)` varies
  the alpha too and compiles the translucent layer loop in (about +6 % on the CPU). To hide an object, use `visible`, not alpha 0.
- **Patterns** are `fnmatch` globs on `scene.instance_names` (MuJoCo geom names, Newton shape
  labels, USD prim paths) or instance indices; a pattern that matches nothing raises.
- **Static props posed per world**: `PerWorld(pose=["crate*"])`, then
  `r.per_world.columns("instance.pos", "crate*").assign(...)` for host / Newton scenes; with MuJoCo the
  poses come from `mjw.Data`: write the declared geoms' `d.geom_xpos` / `d.geom_xmat` per world.
- **Mocap bodies posed alike are shared scenery.** A mocap body whose `d.mocap_pos` /
  `d.mocap_quat` rows are equal in every world when the scene is built
  (`rw.integrations.mujoco.scene(mjm, m, d)`; e.g. the room or table mjlab wraps in a mocap body) is traced
  once for all worlds, and seen by the GI probes and the light grid. Shared scenery
  is static: moving it (alike in every world or per world) raises at the next render with the
  declaration to add: `PerWorld(pose=["table/*"])` (mjlab:
  `RaywarpCameraSensorCfg(raywarp_per_world=...)`). Mocap bodies already posed per world in `d`
  are per world by themselves.
- **Shared bakes use static geometry only.** Moving objects, per-world visibility and deformables
  are excluded from GI tracing, relocation and automatic bounds. They receive the static
  indirect light but do not contribute their own indirect shadows or color bleeding yet.
- **Independent lighting.** The light grid checks live packed-light rows on the device every
  frame, including graph replay. A changed footprint or a shading point outside its cached
  bounds evaluates all lights. Simulator-driven lights keep their original bindings; no
  `PerWorld(light=["pos"])` override is needed to make them move correctly.
  When only a known subset moves, `PerWorld(dynamic_lights=[0, 7])` keeps those light indices
  in every shading point's candidate list and culls the remaining lights with the static grid.
  The hint does not allocate tables: combine it with `light=["pos"]` for renderer-owned
  overrides or retain simulator bindings. Undeclared light motion still triggers all-light
  fallback in that world. Without a hint, declared footprint fields retain all-light evaluation.
- **Per-world GI.** GI follows each world's own lighting and materials. The renderer hashes every
  world's radiance inputs (light, material and instance-appearance rows, environment) on the
  device and groups the worlds whose inputs are equal: one shared probe volume when the worlds
  agree (identical to an unrandomised scene), K stacked irradiance volumes for a few variants, and
  under continuous randomisation one shared volume tinted per world (`core/shade/gi/`). What
  each change costs, and how to avoid a re-bake, is in "GI: re-bakes and dynamic lights" below.
  Poses of per-world instances never re-bake (the probes see the static scenery only).
- **Safety checks.** Asking for a geometry field of a shared instance raises at once with the
  `PerWorld` fix; scenery that moves after the shared trees were built is detected on the device
  (one tiny kernel, read back a frame later without a sync) and raises with the
  `PerWorld(pose=...)` fix: shared scenery is static.
- **Episode resets.** Static bake caches are shared scene preparation, not a robot-state history.
  With the motion output, `r.reset_history(mask)` restarts the masked worlds (graph-capturable).
- **Environment maps per world**: pass a list to `rw.integrations.mujoco.scene(..., lighting=rw.Lighting(envmap=[a, b, c]))` and
  declare `PerWorld(env=["id"])`; `r.per_world["env.id"]` picks one map per world.

## GI: re-bakes and dynamic lights

The default and `FAST` bake probe GI on the first render (`gi.enabled`; cached on disk by content, so a warm
start loads it). After that, a change re-bakes only when the GI reads it.

**Declare lights that move as dynamic.** `PerWorld(dynamic_lights=[i, ...])` marks lights that
move, switch or change every step (a lamp on a robot, a flashlight): they get direct light and
shadows every frame, wherever they are, are left out of the GI bake and its change tracking, and
never re-light. A MuJoCo light on a moving body (or in a tracking mode) that is not declared
dynamic triggers a one-time warning at the bake, naming the indices to declare.

```python
pw = rw.PerWorld(light=["pos"], dynamic_lights=[1])  # light 1 moves per world, direct only
```

**What re-bakes, and what it costs** (`r.debug.gi_rebakes()` reports every bake with its reason, the
`update_gi` updates and, per world, the re-lights by rays):

| change | what the GI does | cost | to avoid it |
|---|---|---|---|
| first render | bake: placement, one traced probe update, then `gi.bake_updates` - 1 bounces of its hits (no ray) | ms to ~1 s cold (Office 0.68 s on an RTX PRO 6000), a disk load warm | `Renderer.warm` |
| moving or per-world instances (robot links, `PerWorld(pose=...)`), cameras, exposure | nothing (static scenery only) | 0 | |
| a dynamic light (any field) | nothing | 0 | |
| a `gi.*` setting tagged rebuild, `rebuild_gi()` | full re-bake | a bake | |
| a GI input every world shares (one-row table written in place: a light, a material, the environment) | full re-bake, a frame later | a bake | declare the field per world (`PerWorld(light=["color"])`, ...) |
| per-world inputs that take new values after the bake | one re-bake as a relit volume (shared maps + a per-world tint) | a bake, once | |
| relit volume: a world's light colour, intensity or on/off, its environment intensity | the tint recombined from per-light shares, no rays (`shade/gi/slots.py`; up to 7 lights get their own slot, the weaker ones share the 8th) | a few multiply-adds per cell | |
| relit volume: a world's materials, environment map or yaw, a light moved by more than half a probe spacing or changed in shape / kind | that world re-lit by rays | ~65k rays per world | keep such lights dynamic |
| `gi.update_every = N` | one probe update every N frames | an update | 0 (the default: frozen) |

Moving instances are not baked into the GI: they receive the static indirect
light but cast no indirect shadows or colour bleeding of their own.

## Sensor effects: noise, vignette, response, stereo depth, rolling shutter, motion blur

The `sensor` settings group adds what a real camera and depth sensor add to the images. Every
field is runtime (switching an effect on or off never recompiles) and everything is off by
default; with everything off no pass runs and no memory is allocated. The pass runs last, on the
final "rgb" / "luminance" and "depth" outputs; hdr, albedo and the other scene-referred outputs
stay clean.

```python
sim2real = (
  rw.RenderSettings()
  .replace(rw.FAST)
  .replace(
    outputs=rw.Outputs.of("rgb", "depth", "motion"),
    sensor={
      "noise": True,
      "full_well": 4000.0,
      "read_noise": 3.0,  # Poisson shot + Gaussian read noise [e-]
      "vignette": (-0.3, 0.0, 0.0),  # 1 + a1 r^2 + a2 r^4 + a3 r^6 <= 1 (Isaac PPISP)
      "rolling_shutter": 0.5,
      "motion_blur": 0.3,  # readout / exposure time over the frame interval
      "depth": "stereo",
      "baseline": 0.05,
      "disparity_noise": 0.08,  # RealSense D400-class active stereo
    },
  )
)
```

- **Noise** is seeded by the render seed, the world, the pixel and the world's sensor frame:
  per-world streams independent of the batch size, new noise every frame, restarted by
  `r.reset_history(mask)`. Display white is `full_well` electrons.
- **Response curve** (`response`: toe, shoulder, center, gamma) maps the sRGB-encoded value
  (PPISP's curve; (1, 1, 0.5, 1) is the identity).
- **Per world** (domain randomisation): `PerWorld(env=["sensor.noise", "sensor.vignette",
  "sensor.response"])`, then `r.per_world["sensor.noise"]` holds (full_well, read_noise) per world.
- **Stereo depth** treats the camera as the left imager of a pair `baseline` metres apart:
  disparity noise and 1/32 px quantisation give the depth error z^2 sigma / (f b); pixels the right
  imager cannot see (the band left of near objects), outside [min_depth, max_depth] or dropped
  (`dropout`) read the depth output's miss value. Undistorted perspective cameras only.
- **Rolling shutter and motion blur** resample the image along each pixel's motion vector (they
  need `outputs.motion`): row y shows the scene `rolling_shutter` x (1 - y / H) frames earlier, and
  the exposure averages `motion_blur` of a frame. An object's blur stays inside its silhouette
  (each pixel follows its own vector).
- **Cost**: one pass over the color (8 B per pixel of state) and/or one row pass over the depth (4
  B per pixel), only while the effect is on.

<!-- SHOWCASE: robotics/sensor_noise.webp (docs/assets/SHOTS.md) -->
![Sensor effects: clean, noise, vignette + response, stereo depth](assets/robotics/sensor_noise.webp)

<!-- SHOWCASE: robotics/rolling_shutter.webp (docs/assets/SHOTS.md) -->
[![Rolling shutter and motion blur on a fast-moving camera](assets/robotics/rolling_shutter.webp)](assets/robotics/rolling_shutter.mp4)

## Step and render in one CUDA graph

`r.render(sim)` already replays its own CUDA graph from the second frame with repeated inputs
([performance](performance.md#max-speed-checklist)); the simulator step stays eager.
`r.capture(step, sim)` runs the step and the render once, then captures both in one CUDA graph
(with OptiX, the graph is cut around the OptiX launches, transparently). Per-world tables,
the `active` world ids and the model's aliased arrays can be written in place between launches.

```python
# docs: requires cuda
with wp.ScopedDevice("cuda:0"):
  gm = mjw.put_model(mjm)
  gd = mjw.make_data(mjm, nworld=4096)
  gpu_scene = rw.integrations.mujoco.scene(mjm, gm)
  gr = rw.Renderer(
    gpu_scene, nworld=4096, cameras=["overhead"], resolution=(64, 64), settings=rw.RenderSettings().replace(rw.FAST)
  )
  graph = gr.capture(lambda: mjw.step(gm, gd), sim=gd)
  for _ in range(100):
    graph.launch(gd)  # one graph launch per step: physics + render, no host synchronisation
  obs = gr.outputs.rgb
```

A captured render reads its random seed from a device `int32` array of shape `(1,)`:

```python
# docs: requires cuda
seed = wp.zeros(1, dtype=wp.int32, device=gr.device)
graph = gr.capture(lambda: mjw.step(gm, gd), sim=gd, seed=seed)
# Your captured step can update seed on the device, or update it before replay.
seed.fill_(23)
graph.launch(gd)
```

The caller controls seed updates and their stream ordering. Integer seeds remain convenient:
`r.render(state, seed=23)` and `r.capture(step, sim=state, seed=23)`. Capture keeps its own
scalar, exposed as `graph.seed`; a later eager render cannot change it. The default seed stays
zero rather than advancing implicitly. Random keys mix seed, world index and camera-pixel address independently; changing a
seed does not shift the previous frame's random pattern across adjacent pixels.

Depth, segmentation, position and surface guides use the pixel center, including with
multisampling. Color integrates its subpixel samples. The extra center observation skips lighting
work unless direct/indirect radiance outputs require it, and RGB-only renders do not trace an
extra observation ray.

`graph.launch` raises `rw.StaleBinding` when an array differs from the captured one (another
`mjw.Data`, a re-bound output) or the renderer changed (`update_settings`, `bind_outputs`,
`set_camera_rays`): capture again. Newton swaps `state_0` / `state_1`: capture an even number of
substeps, or `SensorCamera.capture(lambda: ...)` for the drop-in.

## Selected worlds only

`r.render(d, active=ids)` renders only the worlds listed in `ids`, a `(k,)` int32 device array
of distinct world ids (`-1` entries are holes); `active=None` renders every world. Use it to skip
terminated episodes, or worlds whose observation the policy does not need this step:

```python
ids = wp.array(np.arange(NWORLD, dtype=np.int32))
ids.assign(np.where(np.arange(NWORLD) % 2 == 1, -1, np.arange(NWORLD)).astype(np.int32))
r.render(d, active=ids)  # even worlds rendered, odd worlds keep their last image
r.render(d, active=wp.array([3], dtype=wp.int32))  # world 3 alone: a one-world launch
r.render(d)  # every world again
```

<!-- SHOWCASE: robotics/batch_mask.webp (docs/assets/SHOTS.md) -->
[![render(active=ids): odd worlds paused, even worlds rendered](assets/robotics/batch_mask.webp)](assets/robotics/batch_mask.mp4)

- **No recompile, no host round trip.** The ids are read on the device. A fixed `(nworld,)` array
  written in place every step (`torch.where(done, -1, arange)`, also inside a captured graph:
  `r.capture(step, d, active=ids)`) changes the rendered worlds with no rebuild; a different
  length re-records the render's CUDA graph (no compile).
- **Cost follows the listed worlds.** The launch covers `k` units and packs the listed worlds
  first, so the frame time scales with the number of active worlds; a short list launches fewer
  threads. The per-world post passes (exposure, deferred display, sensor noise, supersampling)
  skip the other worlds; the scene structures are still refit for every world.
- **Outputs stay dense.** Outputs keep their `(nworld, ...)` shape: listed worlds are written,
  the rows of the others keep their values (the last image they rendered).
- **Temporal state.** A world not listed pauses its temporal state (motion, auto exposure). When it is listed again it restarts as on a first frame (`reset_history`), so a
  new episode never blends the previous one.
- An id out of `[0, nworld)` or listed twice is not rendered, and the next render raises.

## CPU workers

Warp runs a CPU launch on one core; to use every core give each worker thread (or pinned
process) its own `Data` and `Renderer`. Warp releases the GIL during launches. Render one
frame of the workers' configuration before starting the threads (or `Renderer.warm` it with the
workers' `nworld` and resolution): it fills the kernel and bake caches, so the workers start
without compiling or baking. The bakes that do run (environment prefilter, light grid, GI probes)
split their CPU launches over the process's cores on one shared set of threads, one per core of
its affinity: worker threads share those cores, and worker processes use the cores they are
pinned to (`taskset -c`, `os.sched_setaffinity`), so pin each one to its own cores.

```python
import threading

settings = rw.RenderSettings().replace(rw.FAST)
with wp.ScopedDevice("cpu"):
  d2 = mjw.make_data(mjm, nworld=2)
  mjw.forward(m, d2)
  rw.Renderer(scene, nworld=2, cameras=["overhead"], resolution=(32, 32), settings=settings).render(d2)
errors: list[BaseException] = []


def worker(seed: int) -> None:
  try:
    with wp.ScopedDevice("cpu"):
      dw = mjw.make_data(mjm, nworld=2)
      mjw.forward(m, dw)
      renderer = rw.Renderer(scene, nworld=2, cameras=["overhead"], resolution=(32, 32), settings=settings)
      renderer.render(dw)
  except BaseException as e:  # report a worker's failure in the main thread
    errors.append(e)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
for t in threads:
  t.start()
for t in threads:
  t.join()
assert not errors, errors
```

## Isaac Lab renderer lifecycle

The adapter targets the renderer API at
[Isaac Lab commit 647e39c5a108e6b217014e4f57ab86efeef7cdaa](https://github.com/isaac-sim/IsaacLab/tree/647e39c5a108e6b217014e4f57ab86efeef7cdaa),
verified against upstream `main` on 2026-10-29. This is a source/API pin, not a claim that an
installed Isaac runtime has passed integration tests. The former 3.0-only documentation and
historical integration results do not establish compatibility with a later checkout.

Isaac constructs `RaywarpRenderer(cfg, sim=owning_simulation)`. The adapter initializes
`BaseRenderer`, retains that simulation, requests its Newton render model and USD stage, and
reads body poses from its scene-data provider. It does not look up a global simulation singleton.
`render_batch` accepts already-prepared camera sessions and currently submits them serially;
camera grouping and a shared refit are still outstanding performance work. Captures are
synchronous, so the inherited `prepare_capture` hook needs no delayed observation metadata.

`reset(render_data, env_ids)` resets only selected environments' temporal history. A `None`
selection resets all environments; an empty selection does nothing. Isaac's tensor indices stay
on-device (converted to contiguous int32 when necessary on the Warp consumer stream). Device
indices must have been validated by Isaac as environment ids representable in int32; they are
not read back for host validation. Host sequences are bounds-checked. Per-environment calibration changes
compare the proposed matrices with the current intrinsics on-device and reset history only for
changed rows. Pose and calibration arrays must contain exactly one correctly typed entry per
environment on the renderer's device; malformed batches fail before camera poses are changed.

A warmed software session can participate in a caller-owned CUDA graph: it records its render
and output conversions into that graph rather than starting a nested capture. Standalone
sessions can still cache their own graph. This does not add support for arbitrary external
capture around native OptiX operations; use the core's supported segmented capture path there.

Scene preparation defaults to the authored USD stage, including its materials and lighting.
A private session-layer view reads one explicit clone prototype and retains global scenery in
its authored frame; the simulation's layers, resolver context, population mask, muted layers and
payload load rules are preserved. Native clones need only small root
transform records, rather than duplicated USD meshes. Canonical body aliases bind each world's
visuals to Newton body poses. Explicit root frames place static scenery and lights independently
of initial body randomisation. Generated clone origins use metres in the canonical Z-up frame,
with stage units and parent transforms accounted for; existing authored roots remain authoritative.
Model-only sources retain their explicit fallback lighting.

Worlds of different prototypes (multi-asset spawning: a different object per environment) each
show their own: every prototype's content is read once, from its first environment, bound to each
environment's own bodies and hidden in the environments of other prototypes (a per-world
visibility, set once: no recompile). The Newton adapter does the same for models whose worlds hold
different shapes (the distinct shapes pooled, absent ones hidden per world at alpha 0).

Current gaps are reported: body-attached and moving emissive-panel lights are
inactive until a live light binding exists. Clone-local environment domes are omitted and reported
for multi-world scenes until per-world environment placement is supported; global domes are retained.
USD-only visuals without a matching Newton shape
are unlabelled in Isaac's segmentation output, rather than borrowing another shape's label.
Their authored materials and color writes use USD provenance independently of collision shapes.
USD material names retain full prim paths (including referenced MaterialX bindings); instance
prototype addresses are resolved through USD instance metadata to each visual’s composed path,
so identical leaf names cannot merge independently addressable material namespaces.
Live color, roughness, metallic and OmniPBR specular-level writes are implemented; other supplied
channels (including emission, opacity, UV placement and IOR) are reported. Scalar input names
are checked explicitly. Roughness/metallic writes currently support untextured target rows only:
prepared roughness/metallic/ORM maps already contain authored blends, so those targets are omitted
and reported until source affine-map updates are available. Untextured roughness preserves
authored coupled-coat roughening and the glTF coat exception; OmniPBR `specular_level` maps to OpenPBR `specular_weight = 2 * specular_level`.
Isaac prepares these writers before camera sessions (`finalize_consumers`). The prepared writer
updates canonical material RGB, including channels initially equal to zero, and preserves authored
base weight, subsurface/transmission blends and opacity. The batch's exact authored input names
distinguish base color from OmniGlass `glass_color` transmission tint; unknown inputs are reported
and omitted. Distinct composed material addresses receive
separate parameter rows when necessary; geometry and textures stay shared. Every camera borrows the
same prepared device table for each enabled channel. No unused per-world instance-tint table
is allocated. A changed value resets only the affected worlds' camera histories; writing
the same values preserves history. Prepare a newly enabled material channel before creating cameras;
ordinary writes to a prepared channel never recompile or replace arrays.
Live material updates do not yet refresh cached shadow-transmission tables or fitted emitter-light
parameters. These dependent lighting effects need separate preparation/update support; material
surface updates alone do not establish complete lighting randomisation support.
The USD path always prepares authored textures (`load_textures=False` is reported unsupported).

Grouped cameras, remaining material channels, deformation updates, motion-vector output and
ISP remain followups. Distorted cameras are currently refused rather than silently approximated.
The USD path has focused host/fake and rendered-stage coverage. Real Torch CUDA reset/color
interop is checked separately; an installed pinned Isaac runtime remains required before claiming
full integration support.

Focused checks:

```sh
uv run pytest -q tests/adapters/isaaclab/test_isaaclab_api.py tests/adapters/isaaclab/test_isaaclab.py tests/adapters/isaaclab/test_isaaclab_usd_scope.py -n 2
```

The first file loads the adapter against faithful fakes of the pinned upstream constructor and
hooks; the second exercises its device tables, outputs and history on a small Newton model.
CUDA-only capture cases skip on CPU. `tests/adapters/isaaclab/test_isaaclab_integration.py` is the separately
coordinated real-runtime check; do not interpret a skip as a pass. Keep the tested upstream
commit and backend in the run provenance, and recheck the renderer interface before updating it.

Declaring `PerWorld(light=["type"])` prepares angular light shading even when the initial
lights are points. Runtime spot/cosine/tube values then use their angular profiles. Other light
properties remain separate: declare and update cone parameters, shape/radius and direction as
needed; changing a type does not automatically change an authored area-light shape. Undeclared
point-only scenes and unlit/depth-only builds retain their smaller specialization.
