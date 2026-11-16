# Domain randomisation

Domain randomisation (DR) renders every world of a batch with its own look: colours, textures,
lights, sky, camera, distractors, sensor. raywarp does it with **per-world tables**: declared once
at construction, written in place every step, read by the same compiled kernels. Nothing
recompiles, nothing is rebuilt, a captured CUDA graph keeps replaying.

This page lists every randomisable quantity, how each adapter reaches it, what it costs, and the
known gaps. The short version is in [robotics.md](robotics.md), "Domain randomisation per world".

## How it works

1. **Declare** what varies per world: `rw.PerWorld(...)` passed to the `Renderer`. The declaration
   decides what is compiled in and allocates each table once, `(nworld, n)`, filled with the
   scene's values.
2. **Write** the tables in place between frames: `r.per_world[field].assign(values)` (NumPy, a
   Warp array, or `wp.from_torch`), or `r.per_world.columns(field, pattern)` for some instances.
   A write from a device array is one copy or one scatter kernel, graph-capturable.
3. **Render**: world `w` reads row `w` of every declared table. Undeclared fields keep one value
   for every world.

Simulators can also drive appearance through their own tables: MuJoCo Warp's batched model tables
(`m.geom_rgba`, `m.mat_rgba`, `m.light_diffuse`, `m.cam_fovy`, ...) are read per world every frame,
and Isaac Lab's material writers fill the declared tables on the device.

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
    <texture name="wood" type="2d" builtin="checker" rgb1="0.6 0.4 0.2" rgb2="0.3 0.2 0.1" width="32" height="32"/>
    <material name="table" texture="wood" texrepeat="4 4"/>
    <material name="cube" rgba="0.8 0.1 0.1 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.5 -0.5 2" dir="-0.2 0.2 -1" type="spot" cutoff="60"/>
    <geom name="table" type="box" size="0.6 0.4 0.02" pos="0 0 0.5" material="table"/>
    <body name="cube" pos="0.1 0 0.56"><freejoint/><geom name="cube" type="box" size="0.04 0.04 0.04" material="cube"/></body>
    <geom name="distractor" type="sphere" size="0.05" pos="0.2 0.2 0.57" rgba="0.9 0.9 0.1 1"/>
    <camera name="cam" pos="0 -1.0 1.1" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
  </worldbody>
</mujoco>
"""
NWORLD = 4
mjm = mujoco.MjModel.from_xml_string(XML)
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=NWORLD)
mjw.forward(m, d)

per_world = rw.PerWorld(
  material=["base_color", "specular_roughness", "base_metalness", "tex_variation"],
  light=["strength", "color"],
  env=["exposure"],
  visible=["distractor"],
)
settings = rw.RenderSettings().replace(rw.FAST).replace(outputs=rw.Outputs.of("rgb"))
r = rw.Renderer(
  rw.integrations.mujoco.scene(mjm, m, d),
  nworld=NWORLD,
  cameras=["cam"],
  resolution=(32, 24),
  settings=settings,
  per_world=per_world,
)
r.render(d)
features = r.features
rng = np.random.default_rng(0)


def randomise(rng: np.random.Generator) -> None:
  """New random values in every world (call it on reset, or every step)."""
  color = r.per_world["material.base_color"]  # (nworld, nmaterial) RGBA: RGB tints
  tint = color.numpy()
  tint[..., :3] = rng.uniform(0.1, 1.0, tint[..., :3].shape)
  color.assign(tint)
  roughness = r.per_world["material.specular_roughness"]
  roughness.assign(rng.uniform(0.1, 1.0, roughness.shape).astype(np.float32))
  texture = r.per_world["material.tex_variation"]  # UV offset, UV rotation, hue per material
  texture.assign(rw.randomize.jitter(np.zeros((*texture.shape, 4), np.float32), rng, uv_offset=1.0, uv_rotation=np.pi, hue=0.3))
  strength = r.per_world["light.strength"]
  strength.assign(strength.numpy() * rng.uniform(0.5, 1.5, strength.shape).astype(np.float32))
  r.per_world["light.color"].assign(rng.uniform(0.6, 1.0, (*strength.shape, 3)).astype(np.float32))
  r.per_world.columns("instance.visible", "distractor").assign(rng.integers(0, 2, (NWORLD, 1)).astype(np.int32))


randomise(rng)
mjw.forward(m, d)
rgb = r.render(d).rgb.numpy()  # (nworld, 1, H, W, 4): each world its own look
assert r.features is features  # nothing recompiled
```

### Rules

- **Declared means compiled in.** The declaration (not the values) selects the kernel build:
  declaring `light=["type"]` selects one with the angular light profiles, `alpha=True` the
  translucent layer loop, `camera=["distortion"]` the lens model ("compiles ... in" below: a
  prebuilt build that holds that code, chosen at construction; lens cameras are the one
  [compile exception](settings.md#compiled-in-or-runtime)). Write anything the declaration
  allows; never redeclare.
- **One row or `nworld` rows.** An undeclared field is one row shared by every world. Writing a
  one-row scene table (a light every world shares, via the scene or the simulator) is a global
  change: it re-bakes the GI a frame later. Declare what varies per world.
- **Appearance never moves scenery out of the shared trees.** Colours, materials, textures, lights,
  cameras and the environment are indexed per world wherever they are read. Geometry fields (pose,
  size, mesh) of a static instance make it per world (one tree entry per world); `visible` keeps it
  shared (a per-world mask tested where rays meet it).
- **Device writes are free of host work.** `.assign(warp_array)` copies on the device; `.assign(numpy)`
  uploads first (a host-to-device copy and a temporary). In a training loop, draw the values on
  the device (Warp kernels, torch) and assign Warp arrays, or write `wp.to_torch(table)` in place.
- **Checked every frame.** Ids are clamped and reported (`env.id`, `material.texture`, ...);
  NaN, infinite, negative or out-of-range values of the float tables (colours, strengths,
  metalness, roughness, field of view, environment intensity and yaw, ...) raise at the next
  render, naming the field and its range. The check runs on the device and is read a frame
  later, without a synchronisation.
- **Episode resets.** With the motion output, call `r.reset_history(mask)` for the re-randomised
  worlds. Without it there is no history to reset.

## What can be randomised

Cost: what one update costs on the device beyond writing the table, and what the declaration adds
to every frame. "Free" means a table read where the value was already read. Every row is
zero-compile after construction.

### Materials and textures

| quantity | declare | table | update cost / per-frame cost |
| --- | --- | --- | --- |
| base colour (RGB tint) | `material=["base_color"]` | `(nworld, nmat)` vec4 | free |
| instance colour (per geom / shape tint) | `material=["color"]` | `(nworld, ninst)` vec4 | free |
| alpha (translucency) | `alpha=True` with a colour table | the colour tables' alpha | compiles the layer loop in (about +6 % frame time) |
| roughness, metalness | `material=["specular_roughness", "base_metalness"]` | `(nworld, nmat)` float | free |
| diffuse roughness (EON) | `material=["base_diffuse_roughness"]` | `(nworld, nmat)` float | compiles EON in |
| specular weight / IOR, coat weight / IOR / roughness, emission luminance | `material=[...]` (OpenPBR names) | `(nworld, nmat)` float each | one derive kernel per derived slot and frame |
| mirror reflectance (MuJoCo `reflectance`) | `material=["mirror"]` | `(nworld, nmat)` float | free |
| texture swap (among the scene's textures) | `material=["texture"]` | `(nworld, nmat, roles)` int texture ids | free; a new texture needs a new scene |
| texture placement and hue | `material=["tex_variation"]` | `(nworld, nmat)` vec4: u, v offset, UV rotation, hue | 16 B per world and material, no texture copy (`rw.randomize.jitter`) |
| texture contrast, per-role tints, normal-map strength | `material=["tex_affine"]` | `(nworld, nmat, roles)` scale / bias rows | 320 B per world and material (`rw.randomize.contrast`) |
| texture repeat, world-projected UVs | `material=["tex_repeat", "uv_world"]` | `(nworld, nmat)` | free |
| material assignment | `material=["instance.material"]` | `(nworld, ninst)` material ids | free (the shading bits follow the material) |

### Lights

The light count is fixed by the scene: add every light a world may need and switch them per world
(`active`). Declaring `type` compiles the angular profiles (spot, cosine, tube) in.

| quantity | declare | table | update cost |
| --- | --- | --- | --- |
| on / off | `light=["active"]` | `(nworld, nlight)` bool | free; GI: recombined from per-light shares (no rays) |
| intensity, colour | `light=["strength", "color"]` | float [cd or lux], normalised RGB | free; GI: recombined (no rays) |
| position, direction | `light=["pos", "dir"]` | vec3 | free; GI: a move over half a probe spacing re-lights that world by rays (~65k rays); the light grid evaluates that world's lights unculled |
| radius (soft shadows), range, attenuation, spot cone | `light=["radius", "range", "attenuation", "cutoff", "cone_inner"]` | float / vec3 | free |
| type, area shape and size | `light=["type", "shape", "axis", "size"]` | int / vec3 / vec2 | compiles the profiles / LTC area lights in |
| shadow casting | `light=["cast_shadow"]` | bool | free |
| lights that move every step | `dynamic_lights=[i, ...]` | (a hint, no table) | direct light only, never re-baked or re-lit |

MuJoCo lights follow `d.light_xpos` / `d.light_xdir`: move them through the model (`m.light_pos`,
`m.light_dir`, batched per world) rather than `light=["pos"]`, which MuJoCo scenes do not offer.

### Environment, background, atmosphere

| quantity | declare | table | update cost |
| --- | --- | --- | --- |
| environment map (one of a bank) | `env=["id"]`, maps `rw.Lighting(envmap=[a, b, c])` | `(nworld,)` int | free; GI: that world re-lit by rays |
| map rotation, intensity | `env=["yaw", "intensity"]` | `(nworld,)` float | free; GI: yaw re-lights by rays, intensity is recombined |
| constant ambient | `env=["ambient"]` | `(nworld,)` vec3 | free |
| exposure | `env=["exposure"]` | `(nworld,)` float | free |
| fog colour, range | `env=["fog.color", "fog.range"]` | per world | free (the fog pass runs when fog is on) |
| background | an environment map per world (`env.id`; a constant map for a plain colour), or a backdrop instance with a randomised material | | free |

The environment-map bank is fixed when the scene is created. The sun of every bank map that has a dominant one
(`rw.SceneOptions.env_sun`, on by default) is cut out once, at the upload, as a shadowed directional light: it is on
in the worlds that show its map and follows their `env.yaw` and `env.intensity`, so switching a world between a sunny
and an overcast map, turning or dimming it is run-time data (no compile).

### Camera and sensor

| quantity | declare | table | update cost |
| --- | --- | --- | --- |
| pose (free cameras) | none: `r.camera_pose` | `(nworld, ncam)` transforms | free |
| pose (scene cameras) | none: the simulator's (`m.cam_pos` / `m.cam_quat` batched, Newton / Isaac camera poses) | | free |
| field of view, intrinsics | `camera=["fovy", "intrinsics"]`; MuJoCo: `m.cam_fovy` / `m.cam_intrinsic` | `(nworld, ncam)` | free (MuJoCo: one gather per frame) |
| lens distortion | `camera=["distortion"]` | `(nworld, ncam)` coefficients | compiles the lens model in |
| colour grading, white balance, chromatic aberration | `env=["grading.*", "sensor.aberration"]` | per world | free when the look pass runs (aberration: the deferred display, automatic exposure) |
| sensor noise, vignette, response curve | `env=["sensor.noise", "sensor.vignette", "sensor.response"]` | per world | one pass over the colour while noise is on |

Resolution, the camera count, rolling shutter, motion blur and stereo depth are per renderer.

### Objects and distractors

| quantity | declare | update cost |
| --- | --- | --- |
| poses of simulated bodies | nothing: the simulator's state | free |
| poses of static props, per world | `pose=["crate*"]` | the props leave the shared scenery: one tree entry per world, a refit per frame |
| primitive sizes | `size=["block*"]` | per-world tree entries |
| mesh swap (among the scene's meshes) | `mesh=["object*"]` | per-world tree entries; every mesh gets its BLAS |
| visibility (distractors, clutter) | `visible=["distractor_*"]` | free: shared instances stay shared |
| deformables, height fields | the simulator's (MuJoCo flex, `m.hfield_data` + `refit_hfields`) | a refit |

Distractors are a pool placed in the scene and shown, hidden and posed per world. Shared bakes
(GI, the light grid) see the static scenery only: per-world instances receive
indirect light but cast no indirect shadows of their own.

## Per adapter

### MuJoCo and MuJoCo Warp

`rw.integrations.mujoco.scene(mjm, m, d)` aliases the device model: these `mjw.Model` tables are read every
frame, per world when batched (`(nworld, ...)` rows, or K rows read as world % K, as MuJoCo
Warp's own renderer reads them):

| model table | renders as |
| --- | --- |
| `geom_rgba`, `geom_matid`, `geom_size`, `geom_dataid` | instance colour, material, size, mesh |
| `mat_rgba`, `mat_emission`, `mat_specular`, `mat_texid`, `mat_texrepeat`, `mat_texuniform` | material colour, emission, reflectance, textures |
| `light_type`, `light_active`, `light_diffuse`, `light_castshadow`, `light_cutoff` (`light_attenuation` with classic lights) | the lights |
| `mat_shininess` (batched when the Renderer is created) | roughness 1 - shininess of the materials that leave `roughness` unset (a declared `specular_roughness` or coat field wins) |
| `cam_fovy`, `cam_intrinsic` | the scene cameras' field of view and intrinsics (cameras asked for by name; a `CameraSpec` of its own on a mounted pose keeps its lens) |
| `light_pos` / `light_dir`, `cam_pos` / `cam_quat`, `body_*`, `geom_pos` / `geom_quat` | poses, through `mjw.forward` (`d.*_xpos`) |

`cam_projection`, `cam_sensorsize` and `light_exponent` keep their values at scene creation: a
batched one warns once and names the per-world field to declare instead. A `mat_shininess` batched
only after the Renderer was created keeps its values at creation (create the Renderer after
batching it). A declared per-world field wins over the model table of the same quantity.

```python
mjm2 = mujoco.MjModel.from_xml_string(XML)
m2 = mjw.put_model(mjm2)
m2.geom_rgba = wp.array(np.repeat(m2.geom_rgba.numpy()[:1], NWORLD, axis=0), dtype=wp.vec4)  # batched: one row per world
m2.cam_fovy = wp.array(np.repeat(m2.cam_fovy.numpy()[:1], NWORLD, axis=0), dtype=float)
d2 = mjw.make_data(mjm2, nworld=NWORLD)
r2 = rw.Renderer(
  rw.integrations.mujoco.scene(mjm2, m2, d2), nworld=NWORLD, cameras=["cam"], resolution=(32, 24), settings=settings
)
rgba = m2.geom_rgba.numpy()
rgba[:, mjm2.geom("distractor").id, :3] = rng.uniform(0.1, 1.0, (NWORLD, 3))
m2.geom_rgba.assign(rgba)  # in place, every step if you like
m2.cam_fovy.assign(rng.uniform(40.0, 60.0, (NWORLD, 1)).astype(np.float32))
mjw.forward(m2, d2)
rgb = r2.render(d2).rgb.numpy()
```

The MuJoCo Warp drop-in (`raywarp.integrations.mujoco.create_render_context`) reads the same tables;
`rc.renderer` exposes `per_world` for the rest.

### mjlab

mjlab's events randomise the model tables above (`sim.expand_model_fields`, then writes per
environment): they reach raywarp's sensors as they reach MuJoCo Warp's renderer. Declare with
`RaywarpCameraSensorCfg(raywarp_per_world=rw.PerWorld(...))` the mocap entities that reset events
pose per world, and any per-world table mjlab has no model field for (textures, environment maps,
exposure, sensor noise); the renderer is `sensor`'s, its tables `per_world` as above.

### Newton

For DR, build the Renderer over `rw.integrations.newton.scene(model)` with a `PerWorld` and write
`r.per_world[...]` as above; the `SensorCamera` and viewer drop-ins make their own Renderer, with
no DR declaration. The shapes whose `model.shape_color` differs between worlds at load are
re-read from it on every render, per world (`rw.integrations.newton.Options(world_colors=True)`, the default;
False: world 0's colours, reported); for any shape, randomise colours with
`PerWorld(material=["color"])` and `r.per_world["instance.color"]`.

### Isaac Lab

Isaac's visual-material events (`randomize_visual_color` and the scalar channels) go through the
adapter's material writers: colour, roughness, metallic and OmniPBR specular level are written per
environment on the device, never recompile, and reset the changed environments' histories only.
Other material inputs (emission, opacity, UV placement, IOR, texture swaps) and light or dome
randomisation are reported, not applied. Live material writes do not refresh the shadow
transmission of translucent casters or fitted emissive-panel lights.

### USD and host scenes

Host scenes (`rw.formats.usd.scene`, `rw.SceneBuilder`) take every `PerWorld` field, including light
positions and directions, and `pose=` for static props (written with
`r.per_world.columns("instance.pos", pattern)`).

```python
b = rw.SceneBuilder()
wall = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=0.9))
b.add_instance("box", material=wall, size=(2.0, 0.05, 1.2), xform=np.c_[np.eye(3), [0.0, 2.0, 1.2]], path="wall")
b.add_instance("box", material=wall, size=(0.3, 0.3, 0.3), xform=np.c_[np.eye(3), [0.3, 0.5, 0.3]], path="crate")
b.add_light(rw.Light.point((0.0, 0.0, 2.2), intensity=80.0, radius=0.05))
b.add_light(rw.Light.point((1.2, 1.2, 1.5), intensity=40.0, radius=0.05))
room = rw.Renderer(
  rw.upload(b.build()),
  nworld=NWORLD,
  cameras=[rw.CameraSpec(fovy=70.0)],
  resolution=(32, 24),
  settings=settings,
  per_world=rw.PerWorld(material=["base_color"], light=["active", "color", "pos"], pose=["crate"]),
)
room.camera_pose.assign(np.tile(np.array([0.0, -1.9, 1.3, 0.45, 0.0, 0.0, 0.893], np.float32), (NWORLD, 1, 1)))
active = np.ones((NWORLD, 2), bool)
active[:, 1] = rng.integers(0, 2, NWORLD).astype(bool)  # the second lamp on or off
room.per_world["light.active"].assign(active)
lamps = room.per_world["light.pos"]
lamps.assign(lamps.numpy() + rng.uniform(-0.5, 0.5, (*lamps.shape, 3)).astype(np.float32))
crate = room.per_world.columns("instance.pos", "crate")  # (nworld, 1) positions
crate.assign(crate.numpy() + rng.uniform(-0.3, 0.3, (NWORLD, 1, 3)).astype(np.float32))
rgb = room.render().rgb.numpy()
```

## Cookbook

Short recipes, each a step of a training loop. They share one renderer over the room above that
declares every table they write (declare once, write every step):

```python
ck = rw.Renderer(
  rw.upload(b.build()),
  nworld=NWORLD,
  cameras=[rw.CameraSpec(fovy=70.0)],
  resolution=(32, 24),
  settings=settings.replace(**{"sensor.noise": True}),  # the noise switch; its level is per world
  per_world=rw.PerWorld(
    material=["base_color", "specular_roughness", "base_metalness"],
    light=["active", "strength", "color"],
    visible=["crate"],
    camera=["fovy"],
    env=["exposure", "sensor.noise"],
  ),
)
home = np.array([0.0, -1.9, 1.3, 0.45, 0.0, 0.0, 0.893], np.float32)  # position, quaternion (x, y, z, w)
ck.camera_pose.assign(np.tile(home, (NWORLD, 1, 1)))
```

**Colours and materials.** RGB tints keep each material's alpha; roughness and metalness are
plain floats per (world, material):

```python
color = ck.per_world["material.base_color"]
tint = color.numpy()
tint[..., :3] = rng.uniform(0.05, 0.95, tint[..., :3].shape)
color.assign(tint)
for field, low, high in (("material.specular_roughness", 0.1, 1.0), ("material.base_metalness", 0.0, 0.3)):
  table = ck.per_world[field]
  table.assign(rng.uniform(low, high, table.shape).astype(np.float32))
```

**Lights: count, intensity, colour.** The scene holds every lamp a world may use; `active`
switches them, so the light count varies per world:

```python
nlight = ck.per_world["light.active"].shape[1]
on = rng.random((NWORLD, nlight)) < 0.7
on[np.arange(NWORLD), rng.integers(0, nlight, NWORLD)] = True  # at least one lamp per world
ck.per_world["light.active"].assign(on)
strength = ck.per_world["light.strength"]
strength.assign(strength.numpy() * rng.uniform(0.5, 2.0, strength.shape).astype(np.float32))
ck.per_world["light.color"].assign(rng.uniform(0.7, 1.0, (NWORLD, nlight, 3)).astype(np.float32))
```

**Camera jitter.** Free cameras are posed by `camera_pose` (no declaration); the field of view is
a declared table:

```python
pose = np.tile(home, (NWORLD, 1, 1))
pose[..., :3] += rng.normal(0.0, 0.03, (NWORLD, 1, 3)).astype(np.float32)  # a few centimetres
ck.camera_pose.assign(pose)
fovy = ck.per_world["camera.fovy"]
fovy.assign(rng.uniform(60.0, 80.0, fovy.shape).astype(np.float32))
```

**Distractors.** Show or hide a pool of props per world; hidden instances stay in the shared
scenery (no tree work):

```python
crate = ck.per_world.columns("instance.visible", "crate")
crate.assign(rng.integers(0, 2, crate.shape).astype(np.int32))
```

**Exposure and sensor noise.** Exposure is a per-world factor; the noise rows are (full well,
read noise) in electrons:

```python
exposure = ck.per_world["exposure"]
exposure.assign(exposure.numpy() * rng.uniform(0.7, 1.4, exposure.shape).astype(np.float32))
noise = ck.per_world["sensor.noise"]
noise.assign(noise.numpy() * rng.uniform(0.5, 2.0, (NWORLD, 1)).astype(np.float32))
rgb = ck.render().rgb.numpy()  # every recipe above, in one frame
```

**Episode resets.** Re-randomise only the worlds that reset; with the motion output also restart
their history:

```python
done = rng.random(NWORLD) < 0.5
tint = color.numpy()
tint[done, :, :3] = rng.uniform(0.05, 0.95, tint[done, :, :3].shape)
color.assign(tint)
ck.reset_history(wp.array(done, dtype=bool))
```

**On the device.** In a training loop draw the values where the policy runs: a table is a Warp
array, so torch writes it in place (no host copy, graph-capturable):

```python
# docs: requires torch
import torch

roughness = wp.to_torch(ck.per_world["material.specular_roughness"])
roughness.uniform_(0.1, 1.0)  # in place: the next render reads it
```

## GI under randomisation

The probe GI follows each world's lights and materials. The renderer hashes every world's
radiance inputs on the device and groups worlds:

- all worlds equal: one volume, as without DR;
- a few variants (at most half the worlds distinct): one exactly baked volume each;
- continuous randomisation: one shared volume tinted per world. A world's light colours,
  intensities, on / off and environment intensity are recombined from per-light shares without
  rays; its materials, environment map or yaw, and lights moved over half a probe spacing re-light
  that world by rays (~65k rays per world, in the frame of the change).

**The first frame after a new variant appears shows the previous GI**: the variant is detected on
the device and read a frame later (no synchronisation), then the volume re-bakes once as a tinted
volume. To avoid even that frame, randomise before the first render (the first bake sees the
variants) or render once more after the first randomisation. After that, every re-randomisation
is applied in the frame it is written. `r.debug.gi_rebakes()` lists the bakes and the per-world
re-lights. Turn GI off (`gi={"enabled": False}`) for open outdoor scenes.

Not followed by the GI: a per-world emission on an emissive panel changes the visible surface but
not its fitted rect light or the GI; per-world instances (poses, visibility) are not in the bake.

## What it costs

Measured on the CPU at 32 x 24 px, 4 worlds (`tests/render/test_dr_scenarios.py`, `pytest -s`);
per-frame GPU costs are in [performance.md](performance.md).

| | cost |
| --- | --- |
| a declared appearance table | the table, `nworld x n x element` bytes; read where the shared value was read |
| writing every tabletop table (11 tables, device arrays) | about 1 ms on the CPU, no allocation, no host round trip |
| the per-frame validation | one tiny kernel per declared float table, on the frames it checks |
| GI | re-lights by rays only for materials, environment map / yaw and moved lights; nothing for intensities and colours |
| per-world geometry | the instance leaves the shared trees (one entry and one refit per world) |
| compile | none after construction; `alpha=True`, `light=["type"]`, `camera=["distortion"]`, EON and LTC area lights compile at construction |

Measured on the GPU (4096 worlds, every table at once): +10-13 % frame time on an RTX PRO 6000,
+2-5 % on an L4.

## Recipes from the literature

| recipe | what it randomises | here |
| --- | --- | --- |
| Tobin et al. 2017 (object detection, sim-to-real) | object / table / floor textures (random colours, checkers, gradients, noise), distractor count and shape, lights (count, position, orientation, specular), camera pose and fov, noise | `base_color`, `texture` over a bank of procedural textures, `tex_variation`, `visible` + `pose` distractor pools, `light` tables (switch `active` for the count), `camera_pose` / `fovy`, `sensor.noise` |
| OpenAI 2018-2019 (dexterous hand, Rubik's cube) | colours, textures, materials, lights, camera pose and intrinsics, post-processing (blur, colour, noise) | the material, light and camera tables, `grading.*`, `lens.*`, `sensor.*`; blur is per renderer (`sensor.motion_blur`) |
| Isaac Lab (replicator events) | visual colour, texture from a list, dome / light intensity and colour, camera | colour and scalar channels through the adapter; textures and lights: declare and write the tables directly |
| MuJoCo Playground / mjlab | `geom_rgba`, `mat_rgba`, light diffuse / position, camera pose / fovy, via model fields | read per world from the batched model (table above) |

## Known gaps

- New textures, meshes or lights at run time: a scene holds a fixed bank (swap among them per
  world; add every light a world may need and switch it).
- Per-world non-uniform mesh scale. A uniform scale s is pose data: declare the instance's pose
  (`pose=[...]`) and write `instance.rot` as s R (no recompile, no mesh copy). A non-uniform,
  sheared or mirrored scale is a different mesh: author it, or swap meshes with `mesh=[...]`.
  Primitives scale with `size`.
- Glass per world costs speed: `instance.material` in a scene with a see-through material or colour
  compiles the per-world optics (`Features.world_optics`) and takes every instance out of the
  shared scenery (each world traverses its own copy: slower in large scenes). A per-world
  `specular_ior` alone takes out only the instances of solid (refracting) materials, and nothing
  without a solid or transmissive material. Choosing among opaque materials keeps full sharing.
- Per-world environment tint (use `grading.white_balance` per world), per-world resolution,
  per-world shadow settings.
- Newton `shape_color` written at run time for shapes whose colour was equal in every world at
  load, Isaac Lab texture / light / dome randomisation: declare the matching `PerWorld` field
  instead.
- Image-space augmentations (JPEG artefacts, random erasing, crops) belong to the training code.
