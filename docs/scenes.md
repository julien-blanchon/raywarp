# Scenes: lights, environments, materials, cameras

Load a scene, light it, check its materials and set up its cameras. USD stages and Newton models
are the main path; MuJoCo models and Python-built scenes work the same way.

## Quick start

A USD stage, lit by the studio rig, rendered from its own camera:

```python
# docs: requires pxr
from pathlib import Path

import raywarp as rw
import raywarp.formats.usd

Path("quick.usda").write_text("""#usda 1.0
(
  upAxis = "Z"
  metersPerUnit = 1
)
def Xform "World"
{
  def Sphere "ball"
  {
    double radius = 0.3
    double3 xformOp:translate = (0, 0, 0.3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    color3f[] primvars:displayColor = [(0.8, 0.3, 0.2)]
  }
  def Camera "camera"
  {
    double3 xformOp:translate = (0, -3, 1.2)
    float3 xformOp:rotateXYZ = (70, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
  }
}
""")
quick = rw.formats.usd.scene("quick.usda", cache=None, lighting=rw.Lighting.studio())  # the stage has no lights
print(quick.report)  # what was dropped or approximated
quick_out = rw.Renderer(quick, nworld=1, cameras=None, resolution=(64, 48)).render()  # None: the stage's cameras
```

The other snippets of this page build on one small MuJoCo scene, which runs anywhere. It is
photometric (a dim 3 lux sun), so it renders with a matching manual exposure:

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
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.2 0.2 0.2" width="64" height="64"/>
    <material name="floor" texture="grid" texrepeat="4 4" roughness="0.8"/>
    <material name="steel" rgba="0.8 0.8 0.85 1" metallic="1" roughness="0.25"/>
  </asset>
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" intensity="3" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="ball" type="sphere" pos="0 0 0.3" size="0.3" material="steel"/>
    <geom name="crate" type="box" pos="0.7 0.3 0.2" size="0.2 0.2 0.2" rgba="0.9 0.4 0.1 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
  </worldbody>
</mujoco>
"""
mjm = mujoco.MjModel.from_xml_string(XML)
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)
m = mjw.put_model(mjm)
d = mjw.put_data(mjm, mjd, nworld=2)

scene = rw.integrations.mujoco.scene(mjm, m)
dim = rw.RenderSettings().replace(**{"exposure.mode": "manual", "exposure.ev100": 0.0})  # 3 lux: a dim sun
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48), settings=dim)
assert r.render(d).rgb.numpy()[..., :3].mean() > 30
```

## Lighting a scene

Every adapter takes the same `lighting=rw.Lighting(...)`, applied before upload. Use it when an
imported scene lacks light, or to debug one. It holds:

| field | what |
| --- | --- |
| `lights` | extra `rw.Light`s ([light types](#light-types)) |
| `ambient` | a constant fill, used only where no environment map lights the scene |
| `envmap`, `envmap_intensity`, `envmap_rotation` | an environment map ([below](#environment-maps-hdri)) |
| `headlight` | a camera light, `rw.Headlight(ambient=..., diffuse=...)` |

Display-relative quantities (a light without an intensity, the ambient, the headlight, an envmap
without an intensity) show a white surface at the given color at `lighting.ev100` (default
14.97). Under automatic exposure that needs nothing; with a manual exposure, use the same `ev100`.

```python
sky = np.ones((32, 64, 3), np.float32)  # (height, width, 3) linear radiance; row 0 is the zenith
sky[16:] *= 0.3  # a darker ground half
fill = rw.Lighting(
  ambient=(0.2, 0.2, 0.2),  # display-relative radiance: a white surface shows 0.2
  envmap=sky,  # a path (.hdr / .exr linear, 8-bit images sRGB) or a linear radiance array
  envmap_intensity=None,  # None: display-relative, a texel of 1 shows white
  headlight=rw.Headlight(ambient=(0.0, 0.0, 0.0), diffuse=(0.4, 0.4, 0.4)),
)
scene = rw.integrations.mujoco.scene(mjm, m, lighting=fill)
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48))
assert r.render(d).rgb.numpy()[..., :3].mean() > 30
```

### The studio rig

`rw.Lighting.studio()` is the quickest fix for a dark scene. It makes any scene readable at the
default exposure:

- a sky dome: bluish zenith, white horizon, dim warm ground;
- a warm shadowed key light 50 deg above the horizon;
- a weak cool fill light (`fill=False` drops it).

It adds to the scene's own lights; `replace=True` drops them (and the headlight, ambient and
image lighting). Use `up="y"` for a Y-up Newton model (`model.up_axis`). MuJoCo and USD scenes are
Z-up after loading. Its values are display-relative, so it exposes the same in photometric scenes
and MuJoCo's classic ones.

### Per adapter

USD and Newton take `lighting=` at load. USD adds it after the `.rayw` cache, so the cache key does
not change. Newton's front ends (`SensorCamera`, `ViewerRaywarpHeadless`) and Isaac Lab's
`RaywarpRendererCfg(lighting=...)` take it too.

```python
# docs: requires pxr
from pxr import Usd, UsdGeom

import raywarp.formats.usd

stage = Usd.Stage.CreateNew("ball.usda")  # any stage without lights
UsdGeom.Sphere.Define(stage, "/ball")
stage.Save()
host = rw.formats.usd.load("ball.usda", cache=".", lighting=rw.Lighting(envmap=sky))
assert host.nenv == 1
```

```python
# docs: requires newton
import newton

import raywarp.integrations.newton

builder = newton.ModelBuilder()
builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5)
model = builder.finalize(device="cpu")
lights = rw.Lighting(rw.integrations.newton.default_lights(), ambient=(0.05, 0.05, 0.05))
host = rw.integrations.newton.load(model, lighting=lights)
gpu = rw.upload(host, model.device, binding=rw.integrations.newton.binding(host, model))
```

A scene that is already built (a `.rayw` file, a USD or Newton host scene) gets the same with
`Scene.with_lighting`. It returns a new scene. Point and spot lights without an intensity are
display-relative: their intensity follows their distance to the scene's centre.

```python
b = rw.SceneBuilder()
grey = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8)))
b.add_instance("box", size=(3.0, 3.0, 0.05), material=grey, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
host = b.build()  # or rw.Scene.load("scene.rayw")
lit = host.with_lighting(rw.Lighting(lights=[rw.Light.sun((0.3, 0.2, -1.0))], ambient=(0.1, 0.1, 0.1)))
gpu = rw.upload(lit)
studio = rw.upload(host.with_lighting(rw.Lighting.studio(replace=True)))  # the rig alone
```

MuJoCo's lights are the model's own: MuJoCo Warp's light tables are used live, not copied. So
`rw.integrations.mujoco.add_lights(spec, lighting)` adds the lights to the `MjSpec` before
compiling, in the model's light model. It returns the rest (dome, ambient, headlight) for
`rw.integrations.mujoco.scene`. Passing lights to `scene` directly raises.

```python
spec = mujoco.MjSpec.from_string(XML)
rest = rw.integrations.mujoco.add_lights(spec, rw.Lighting.studio())  # the key and fill, as MJCF lights
mjm2 = spec.compile()
scene = rw.integrations.mujoco.scene(mjm2, mjw.put_model(mjm2), lighting=rest)  # the dome
r = rw.Renderer(scene, nworld=1, cameras=["front"], resolution=(64, 48))
mjd2 = mujoco.MjData(mjm2)
mujoco.mj_forward(mjm2, mjd2)
assert r.render(mjw.put_data(mjm2, mjd2)).rgb.numpy()[..., :3].mean() > 60
```

### Light types

`rw.Light` is the one light description, in physical units (UsdLux inputs):

| constructor | intensity | notes |
| --- | --- | --- |
| `Light.sun(direction, illuminance=lux)` | lux | a directional light |
| `Light.point(position, intensity=cd \| power=lm, radius=0)` | candela or lumens | takes an IES profile, `ies=rw.scene.Ies(path)` |
| `Light.spot(position, direction, cone_angle=45, ...)` | candela or lumens | takes an IES profile |
| `Light.sphere`, `Light.rect`, `Light.disk`, `Light.tube` | radiance [nits] | area lights; `Light.rect(..., visible=True)` shows its surface |

A sun, point or spot made without an intensity is display-relative. `Light.point` / `Light.spot`
also take `spot_mode`, and the sun, point and spot a colour `temperature` and `illuminant`
([photometric units](#photometric-authoring-units)).

<!-- SHOWCASE: scenes/lights.webp (docs/assets/SHOTS.md) -->
![Point, spot, rect area and IES lights](assets/scenes/lights.webp)

Under `lighting.area_lights="ltc"` (the default) lights with a size are true area lights: soft
highlights, and soft shadows from `shadows.samples` points on the light. `"point"` (as in `FAST`)
shades them as points.

## Environment maps (HDRI)

An equirectangular HDR image lights the scene and fills the background. Its prefiltered map is
baked once and cached on disk.

<!-- SHOWCASE: scenes/envmaps.webp (docs/assets/SHOTS.md) -->
![One object under three environment maps](assets/scenes/envmaps.webp)

```python
scene = rw.integrations.mujoco.scene(mjm, m, lighting=rw.Lighting(envmap=sky, envmap_rotation=90.0))  # or "studio.hdr"
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48))
out = r.render(d)
```

| field | what |
| --- | --- |
| `envmap` | a path (`.hdr`, `.exr` linear; 8-bit images sRGB), a float array of linear radiance, or a list of those |
| `envmap_intensity` | a scale. None: display-relative, a texel of 1 shows white at the default exposure |
| `envmap_rotation` | turns it about +z [degrees] |
| per world | `PerWorld(env=["id"])` picks one map of the list per world (`r.per_world["env.id"]`); also `"env.intensity"`, `"env.yaw"` ([robotics](robotics.md)) |

- USD DomeLights and host scenes (`SceneBuilder.add_environment(rw.scene.EnvironmentLight(texture=...))`)
  work the same way.
- `environment.background=False` shows `environment.background_color` instead of the map.
- GI keeps the sky out of enclosed spaces ([a closed room](#a-closed-room-lit-by-an-environment-map)).
- **Sun extraction.** A sunny sky's sun is cut out and lit as a directional light with shadows and
  its disc's size. A "sun" is a compact region of texels far above the sky, holding at least 30 %
  of the irradiance of a surface facing it. The camera still sees the disc. The sun follows each
  world's `env.id`, `env.yaw` and `env.intensity`. Overcast skies and studios have none.
  `rw.SceneOptions(env_sun=False)` keeps the sun in the map.
- Sharp environment reflections filter texels by the ray cone and the map's resolution. Rough
  reflections use the prefiltered GGX stack.

## Indirect light (GI)

Light that reaches a surface only after a bounce (a room lit through a window, a ceiling above
downlights) needs probe GI. It is on by default and in `FAST`, off under unlit shading. Turn it off
with `gi.enabled=False`; its grid fields are in `rw.settings.GI`:

```python
fast_with_gi = (
  rw.RenderSettings().replace(rw.FAST).replace(**{"gi.enabled": True, "gi.spacing": 1.0, "gi.rays": 64, "gi.bake_updates": 8})
)
r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=["front"], resolution=(64, 48), settings=fast_with_gi)
```

- GI is read from a texture resampled from the probes after each update: cheap per pixel.
- Its filtering never reaches across a wall, whatever its thickness. Closed rooms stay dark;
  corners do not pick up light from outside.
- Per-world randomised materials, lights or environments: a few variants get one baked volume
  each. Mostly distinct worlds share one volume, tinted per world and recomputed when their
  inputs change. Details: [settings](settings.md#what-each-group-does).

### A closed room lit by an environment map

An environment map lights what sees the sky. The GI probes keep it out of enclosed spaces: a closed
room gets almost nothing, a window lets in only what it should. Keep GI on, and give the room the
light it lacks:

- lights inside the room (`Lighting(lights=[rw.Light.point(...)])`), or `Lighting.studio()`
  (its fill light is unshadowed, so it reaches a closed room);
- if no window light is wanted: `environment.enabled=False` and `Lighting(ambient=...)`. The
  ambient is never occluded, so it lights the room evenly. With an environment lighting the scene,
  `ambient` is not added;
- a render-time fill on top of everything: `lighting.fill`, the headlight, `rw.Lighting.fill_lights`
  ([light a dark interior](faq.md#dark-interiors)).

Turning GI off indoors makes the sky light every room as if it had no roof (the Isaac Office: +2.6
EV, washed out). It only suits open scenes.

```python
def closed_room() -> rw.Scene:
  """A 4 x 4 x 3 m room with no opening: floor, ceiling and four walls."""
  b = rw.SceneBuilder()
  white = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  slabs = [((2.0, 2.0, 0.05), (0.0, 0.0, -0.05)), ((2.0, 2.0, 0.05), (0.0, 0.0, 3.05))]
  slabs += [((0.05, 2.0, 1.5), (x, 0.0, 1.5)) for x in (-2.05, 2.05)]
  slabs += [((2.0, 0.05, 1.5), (0.0, y, 1.5)) for y in (-2.05, 2.05)]
  for size, pos in slabs:
    b.add_instance("box", size=size, material=white, xform=np.c_[np.eye(3), pos])
  return b.build()


def floor_radiance(lighting: rw.Lighting, **settings: object) -> float:
  """Mean HDR radiance of the floor seen from the middle of the room."""
  s = rw.RenderSettings().replace(rw.FAST).replace({"outputs": rw.Outputs.of("hdr"), "gi.enabled": True, **settings})
  scene = rw.upload(closed_room().with_lighting(lighting))
  r = rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(16, 12), settings=s)
  r.camera_pose.assign(np.array([[[0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0]]], np.float32))  # looking down
  return float(r.render().hdr.numpy().mean())


occluded = floor_radiance(rw.Lighting(envmap=sky))  # GI: the roof keeps the sky out
no_roof = floor_radiance(rw.Lighting(envmap=sky), **{"gi.enabled": False})  # wrong
lamp = floor_radiance(rw.Lighting([rw.Light.point((0.0, 0.0, 2.8))], envmap=sky))
fill = floor_radiance(rw.Lighting(ambient=(0.3, 0.3, 0.3)), **{"environment.enabled": False, "gi.enabled": False})
assert occluded < 0.01 * no_roof and min(lamp, fill) > 0.1 * no_roof
```

## How each format's lights are read

**USD.** UsdLux `DistantLight` (illuminance [lux]), `SphereLight`, `RectLight`, `DiskLight`,
`CylinderLight` (surface radiance [nits]; `ShapingAPI` cones) and `DomeLight` (the environment).

- As in Omniverse RTX, camera rays do not see a light's surface unless it authors
  `visibleInPrimaryRay = true`. A dome is the background either way.
- Emissive materials glow. Flat rectangular emissive panels also become rect lights at upload
  (`lighting.emissive_lights`).
- Dropped inputs (light textures, ...) are counted in the [load report](#the-load-report).
- Omniverse's photometric APIs and IES profiles are read ([below](#photometric-authoring-units)).

**Host scenes.** `SceneBuilder.add_light(rw.Light(kind=..., intensity=..., xform=...))` takes a
UsdLux record with its inputs, names and defaults. Kinds: `LightKind.DISTANT`, `SPHERE` (a point
light: `radius=0`, `treat_as_point=True`; a spot: add `shaping=rw.scene.Shaping(cone_angle=<deg>)`),
`RECT`, `DISK`, `CYLINDER`. Sizes are in metres; the light emits along its local -Z. The radiance is
`intensity x 2^exposure x color` (x the blackbody colour when enabled).

**MuJoCo.** Only native MJCF fields are read, so existing models render unchanged. The adapter
follows MuJoCo's own rule for its two light models:

| MJCF | rendered as |
| --- | --- |
| any light with `intensity` > 0 (photometric model) | every light physical: `diffuse` x `intensity` (lux for directional, candela otherwise), inverse-square falloff windowed at `range`, spot inner angle 1 - `softness`, soft shadows from `bulbradius`; intensity-0 lights dark, headlight off |
| every `intensity` 0 (classic model) | no distance falloff; a white surface shows the light's `diffuse` color at `Lighting.ev100` (14.97); `visual/headlight` is a camera light (`lighting.camera_light`); its `ambient` plus the lights' `ambient` is a constant ambient |
| classic point / spot `attenuation` | live constant / linear / quadratic coefficients; the default (1, 0, 0) has no falloff |
| classic `exponent` | the spot's inner angle (its falloff crosses one half where cos^exponent does) |
| classic `specular` | not rendered (one color drives both lobes) |
| `type="image"` with a `texture` | image-based lighting: the texture's radiance x `intensity` [nits], also the background |
| skybox texture | background only (MuJoCo's skybox lights nothing), when no envmap or image light is given |

Lights with a `bulbradius` are sphere lights. MJCF has no IES profiles.

## Light units and exposure

The `hdr` output is scene-linear radiance before exposure and tone mapping. A Lambertian surface
of reflectance `rho` under illuminance `E` shows `rho * E / pi`.

<!-- SHOWCASE: scenes/exposure.webp (docs/assets/SHOTS.md) -->
![Automatic exposure vs manual ev100 8 / 12 / 15](assets/scenes/exposure.webp)

Exposure is automatic by default (`exposure.mode="auto"`). Each camera image is metered on a
histogram of its luminance: the 10-90 % band lands at middle grey 0.18, darkened by at most 1 stop
when a large area would clip.

| `exposure.adaptation` | behaviour |
| --- | --- |
| `"locked"` (default) | metered on the first frame, then locked per world and camera until `reset_history` (episode reset, re-randomisation) |
| `"continuous"` | follows the scene smoothly |
| `"per_frame"` | meters every frame on its own |

`exposure.mode="manual"` exposes at `exposure.ev100`. Every tone mapper receives the same scale,
`2 ** -ev100`. Display-relative emission, headlight and ambient follow the exposure (under
automatic exposure, the reference 14.97). Authored USD camera exposure and the full lens-exposure
path are not calibrated yet.

**Calibration status.** The checks cover directional-light units, the EV scale and the RTX curve
(`tests/calibration/test_units.py`). Not yet covered: finite-area lights and attenuation, shaping
and IES, environment transport, authored camera exposure, scene-unit / up-axis invariance, a
ColorChecker, independent RTX HDR evidence.

### USD light dialects

The USD adapter reads lights in one of two dialects ([table](#usd-light-dialects-reference)):

- **UsdLux spec** (the default).
- **RTX legacy** (`omni:rtx:usdluxVersion` < 2505). It applies to a light whose authored version
  says so. Without a version, it applies to a light whose strongest opinion comes from a layer
  saved by Omniverse Kit (`customLayerData` with `omni_layer`, `renderSettings` or `cameraSettings`).
- Kit's attribute naming alone (`omni:` inputs, or pre-21.02 names) is ambiguous: it is reported
  as `UsdLux.dialect.ambiguous(...)` and warned.
- `usd.Options(lux_dialect=rw.formats.usd.Dialect.RTX_LEGACY)` sets the dialect of unversioned
  lights for an unmarked stage (one built live in a Kit app).

A DomeLight's map pole is its local +Y with OpenEXR's latlong layout in the spec (DomeLight_1
`poleAxis`), and its local +Z with RTX's layout in RTX legacy. Its texture's `colorSpace` decides
the decoding (unauthored: by file type). `texture:format` `automatic` and `latlong` read latlong;
`angular`, `mirroredBall` and `cubeMapVerticalCross` are resampled to latlong once at load. An
unknown format, or an image of the wrong shape, raises a `ValueError` naming the dome.

Sized suns (angular diameters up to 180 degrees) soften shadows and widen highlights: a sized sun
is a soft highlight, not a dot. The diffuse response uses the disc's central direction. Sources:
[UsdLux normalization](https://openusd.org/release/user_guides/schemas/usdLux/LightAPI.html#inputs-normalize),
[RTX lighting-version selection](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html#controlling-the-usdlux-behavior).

### Photometric authoring units

`raywarp.scene.lighting.photometry` converts authoring units once into the canonical UsdLux
record:

| helper | converts |
| --- | --- |
| `with_power` | luminous flux [lm]; spot cones `"reflector"` (Filament's focused spot) or `"absorber"` |
| `with_illuminance` | illuminance [lux] at a distance on the light's axis (exact for spheres, disks, rects) |
| `watts_to_lumens` | watts x efficacy or efficiency |
| `cct_color` | colour temperature (UsdLux's blackbody table, the Planckian locus, the CIE daylight series) |
| `spectrum_color` | a sampled spectrum |

The USD adapter reads Omniverse's `PhotometricAreaLightAPI` (power, illuminance at distance),
`PhotometricDistantLightAPI` (lux), `PhysicalLightIlluminantAPI` (`physical:illuminant`) and
`PhotometricDomeLightAPI` (the dome scaled to its illuminance on an upward surface).

### IES profiles

UsdLux ShapingAPI `ies:file` (LM-63 1986-2019, types C, B and A, any symmetry) shapes a light's
direct lighting. It is compiled in only when a light has a profile.

| mode | intensity |
| --- | --- |
| `ies:normalize` | the profile over its mean: the unshaped light's flux |
| `"mask"` | the profile over its peak, times the light's intensity (plain UsdLux) |
| `"measured"` | the file's candela values x intensity x 2^exposure (`PhysicalLightIlluminantAPI`, as Omniverse) |

Orientation: type C nadir along the light's -Z, horizontal 0 along +X, 90 along +Y; types A / B
axis along -Z, up +Y. `ies:angleScale` follows UsdLux. A TILT=INCLUDE factor follows the light's
orientation per frame. Area lights with a profile keep their area form, times the profile's mean.

### Emission

Physical emission (USD, host scenes) is radiance [nits] at any exposure. Display-relative emission
(MuJoCo) is `emission x color / exposure`: its exposed value is fixed (Filament's
exposure-compensated emissive, 3 + log2(1.2) EV).

## Materials and textures

Every adapter converts its materials into one record, `rw.Material`: an
[OpenPBR Surface 1.1.1](https://academysoftwarefoundation.github.io/OpenPBR/) surface with the
spec's input names and defaults (base color 0.8, `specular_roughness` 0.3, `specular_ior` 1.5,
`coat_ior` 1.6).

```python
b = rw.SceneBuilder()
y, x = np.mgrid[0:64, 0:64]
checker = np.where((x // 8 + y // 8) % 2 == 0, 220, 40).astype(np.uint8)
tex = b.add_texture(np.stack([checker] * 3 + [np.full_like(checker, 255)], -1))  # RGBA8, sRGB
tiles = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=0.5).with_texture(
  rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(tex)
)
gold = rw.Material(base_color=(1.0, 0.77, 0.34), base_metalness=1.0, specular_roughness=0.3)
b.add_instance("box", size=(3.0, 3.0, 0.05), material=b.add_material(tiles), xform=np.c_[np.eye(3), [0, 0, -0.05]])
b.add_instance("sphere", size=(0.3, 0.0, 0.0), material=b.add_material(gold), xform=np.c_[np.eye(3), [0, 0, 0.3]])
b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0))  # emits along its local -z: down
b.add_light(
  rw.Light(kind=rw.scene.LightKind.RECT, intensity=2000.0, width=0.5, height=0.5, xform=np.c_[np.eye(3), [0.5, 0.5, 1.5]])
)
b.add_camera("cam", rw.scene.Camera(xform=np.c_[[[1, 0, 0], [0, 0.34, 0.94], [0, -0.94, 0.34]], [0, -3, 1.2]]))
host = b.build()  # a rw.Scene (b.build("scene.rayw") also writes the cache file)
r = rw.Renderer(rw.upload(host), nworld=1, cameras=None, resolution=(64, 48))
out = r.render()
```

<!-- SHOWCASE: scenes/materials.webp (docs/assets/SHOTS.md) -->
![OpenPBR: metalness, roughness, coat, sheen, transmission](assets/scenes/materials.webp)

### The material: OpenPBR Surface

The core renders OpenPBR's real-time subset, as the spec allows: one GGX specular lobe (alpha =
roughness², Schlick Fresnel) over a Lambert base, an optional GGX coat, emission and opacity.
Device values are derived from the record once, at upload. A `rw.Material` holds:

| part | what |
| --- | --- |
| 16 fields | the inputs the core renders, plus `specular_color` and `geometry_thin_walled` (carried, not rendered) |
| `extras=rw.scene.OpenPbrExtras(...)` | the other OpenPBR inputs: carried (`.rayw` included) and reported `gpu.material.<input>` when authored |
| `ext=rw.scene.RaywarpExt(...)` | raywarp's extensions ([below](#extensions)) |
| textures per slot | `rw.scene.Slot` (named after the input it drives), `rw.scene.TextureRef` (channel, scale / bias, UV transform, UV set). A bound slot's value is `field x (scale x texel + bias)`. |

What each OpenPBR input renders as: [the coverage table](#openpbr-inputs).

<!-- SHOWCASE: scenes/glass.webp (docs/assets/SHOTS.md) -->
![Glass: a refracting solid, a thin-walled pane, tinted absorption](assets/scenes/glass.webp)

**Coats.** OpenPBR, Standard Surface and other coupled sources use OpenPBR's coat: the base
reflects at `specular_ior` / `coat_ior` under the coat, the coat roughens the base, and the coat
weight scales by F0(`coat_ior`) / 0.04. glTF, UsdPreviewSurface, OmniPBR and OmniSurface coats
carry `MaterialFlags.COAT_GLTF` (at `coat_ior` 1.5): the base keeps its F0 and is neither
roughened nor darkened.

**MDL layers.** Distilled MDL layers keep their metal lobe when the dielectric layer's weight is
zero. Non-Unreal layers carry `DIELECTRIC_SPECULAR_LEVEL`: the weight changes dielectric F0 only,
grazing reflectance stays 1. Where the authored layer differs, the report records
`mdl.specular_grazing(fixed at 1)`. A tinted metal reports `mdl.metal_fresnel(Schlick instead of
constant tint)`.

**Diffuse roughness.** `OpenPbrExtras.textures` with the input `"base_diffuse_roughness"` binds a
diffuse roughness map. Set the scalar `base_diffuse_roughness=1.0` to use the texture alone
(USD / MaterialX connections do this). The scalar multiplies the texture channel.

### Normal maps

A `Slot.GEOMETRY_NORMAL` binding's value (`scale` x texel + `bias`) is the tangent-space normal.
The convention is OpenGL / glTF / Blender (+Y up), with UVs from the image's bottom-left.

| map | binding |
| --- | --- |
| plain 8-bit, +Y up | scale (2, 2, 2, 1), bias (-1, -1, -1, 0) |
| +Y down (DirectX / Unreal) | `normal_map.scale_xy(ref, normal_map.Y_DOWN)` |
| strength k | `normal_map.unpack((k, k))` |

(`normal_map` is `rw.scene.model.normal_map`.) Every importer derives these from the authored data:
UsdUVTexture scale / bias, MaterialX `normalmap` scale, glTF `normalTexture.scale`, MDL strengths.
A strength above 1 decodes beyond [-1, 1]. A binding whose decoded z is never positive is rejected.

### Extensions

`rw.scene.RaywarpExt`, outside OpenPBR:

| field | meaning |
| --- | --- |
| `alpha_mode`, `alpha_cutoff` | glTF `AlphaMode`: OPAQUE, BLEND (needed for `geometry_opacity` < 1 or an opacity texture), MASK (a cut-out at the cutoff) |
| `occlusion_strength` | blend of the `Slot.OCCLUSION` map (glTF AO) |
| `mirror` | MuJoCo's mirror reflectance (`mat_reflectance`: one extra ray) |
| `flags` | `MaterialFlags`: `TWO_SIDED`; `UNLIT` (glTF KHR_materials_unlit, MaterialX `surface_unlit`: base color plus emission, display-relative); `UV_WORLD` (texture repeat per metre, MuJoCo `texuniform`); `SPECULAR_CURVE` (Unreal / OmniUe4 specular); `VERTEX_COLOR`; `COAT_GLTF`; `DIELECTRIC_SPECULAR_LEVEL` (weight changes dielectric F0 only) |

### Per-world materials

Declare `rw.PerWorld(material=[...])`, then write `Renderer.per_world["material.<name>"]`, an
`(nworld, nmaterial)` table, in place.

| kind | names |
| --- | --- |
| OpenPBR inputs | `base_color` (RGB unless `PerWorld(alpha=True)`), `base_metalness`, `specular_roughness`, `specular_weight`, `specular_ior`, `coat_weight`, `coat_ior`, `coat_roughness`, `emission_luminance`, `base_diffuse_roughness` |
| extensions | `mirror`, `texture`, `tex_affine`, `tex_variation`, `tex_repeat`, `alpha_cutoff`, `uv_world` |
| per instance | `color`, `shading` (packed material flags and coat), `instance.material` |

- The tables start at the scene's values (MuJoCo: the live `m.mat_specular` / `m.mat_emission`).
  Updates equal a fresh upload within 1 ulp and never re-bake images.
- Do not declare a coat field together with the `specular_roughness` / `shading` it derives.
- `tex_affine` changes a texture's scale / bias per world; `tex_variation` its UV offset, rotation
  and hue, without copying the texture (`rw.randomize.jitter` / `contrast` draw such tables).
- `instance.material` swaps materials per world, each with its own UV set, transform, cut-out and
  shading flags, without per-world mesh copies. A declared `instance.shading` owns those flags
  instead.
- Limits: one UV set and transform per material (conflicts are reported; a missing second UV set
  falls back to UV0). On an emissive panel turned into a rect light, a per-world emission changes
  only the visible surface, not its light or the GI.

### Displacement

`Material(displacement=rw.scene.Displacement(amount=...))` takes a constant object-space distance
or a scalar-channel `TextureRef` (its scale, bias, UV transform and addressing apply; no implicit
midpoint). `SceneBuilder` refines the mesh into ordinary triangles, shared by instances and saved
in `.rayw`. Material swaps change shading, not geometry.

- UsdPreviewSurface's scalar `displacement` input is read (reported).
  `rw.formats.usd.Options(displacement=False)` omits it.
- Refinement is linear, bounded by `displacement_max_level` (default 6) and
  `displacement_max_triangles` (16 million); the builder's `DisplacementOptions` set them.
- Normals are smooth over welded positions, rebuilt after displacement. Hard edges and authored
  corner normals are not kept (reported). UV-seam disagreements are averaged (reported).
- Not supported: subdivision limit surfaces, primitive displacement, MaterialX / MDL
  displacement, vector displacement, bump-only mode, per-world geometry changes.

### Sources

**USD.** UsdPreviewSurface (with UsdUVTexture, UsdTransform2d), MaterialX surfaces and MDL
materials (Omniverse `OmniPBR`, `OmniGlass`, `OmniSurface`, `OmniUe4*`, ...). Procedural parts are
baked into textures at source resolution and cached. Also `primvars:displayColor` and the
`raywarp:vertexColor` attribute. What could not be translated is counted in the load report.

USD geometry and scene features:

| support | features |
| --- | --- |
| read | meshes (GeomSubsets of `materialBind`, two UV sets, normals and vertex colors of any interpolation), UsdGeom primitives (a Cone is tessellated), PointInstancers, instanceable prims, every composition arc, purposes (`default`, `render`), visibility, metersPerUnit and upAxis (scenes become metres, Z-up), cameras, UsdLux lights |
| approximated (reported) | subdivision surfaces (the cage, smooth normals, no creases), n-gons (fan-triangulated), skinned meshes and blend shapes (bind pose), time samples (one time, `Options.time`; poses come from the simulator), light linking (every light lights everything) |
| not supported (reported `prim.<Type>`) | Points, BasisCurves / NurbsCurves, NurbsPatch, TetMesh, volumes, GeometryLight, PortalLight, PluginLight |
| skipped silently | physics data, UsdRender prims (`rw.formats.usd.settings(stage)` reads those) |

**MaterialX.** The official MaterialX library writes each material as an MDL module, which the
bundled MDL SDK converts like any MDL material. This is how Omniverse RTX renders MaterialX. Every
surface shader of the standard libraries converts (`open_pbr_surface`, `standard_surface`,
`gltf_pbr`, `surface_unlit`, ...).

- Render contexts are tried in order before the universal one:
  `rw.formats.usd.load(path, adapter=rw.formats.usd.Options(material_contexts=...))`. Default
  `("mtlx", "mdl")`; `("mdl", "mtlx")` keeps MDL; `("mtlx",)` never reads MDL.
- Referenced `.mtlx` documents are not composed: they are reported as dropped.
- File inputs keep their authored color space; `geompropvalue` of `st` / `st1` reads UV set 0 / 1.
- Rejected networks are reported (`mtlx.*`) and render as the error material.

**glTF 2.0.** `rw.formats.gltf.scene(path)` / `rw.formats.gltf.load(path)` (`import
raywarp.formats.gltf`; the `gltf` extra) convert the asset to a USD stage and read it as USD.

- Covered: metallic-roughness and specular-glossiness with their maps, `KHR_texture_transform`,
  `KHR_materials_emissive_strength`, `_ior`, `_clearcoat` factors, `_transmission` (as opacity),
  vertex colors, `KHR_lights_punctual`, cameras, shared meshes as instancers.
- Emission is in the environment's units, as in glTF viewers: an emission of 1 is as bright as an
  environment texel of 1. Under `Lighting(envmap=hdr, envmap_intensity=k)` it becomes emission x
  k nits.
- The conversion is cached by the content of the asset and every file it references (`cache=True`,
  the default). `raywarp convert scene.gltf scene.usdc` writes the same stage ([CLI](cli.md)).

**MuJoCo.** The material tables of `m` are used live, re-read every frame. Each MuJoCo material
maps to an OpenPBR material ([table](#mujoco-materials)).

- A flex `material` applies the same way, except the mirror `reflectance` and the alpha cutoff.
  Its color follows MuJoCo's rule; a flex without texture coordinates shows no texture.
- A custom numeric `raywarp:alpha_cutoff:<material>` makes a material a cut-out instead of
  blended. `raywarp convert IN.glb OUT.xml` writes these fields.

Texture memory: USD textures are capped at 1024 px per side by default
([texture storage](settings.md#store-textures-in-less-gpu-memory)).

## Cameras

A `rw.CameraSpec` is one camera of a Renderer. A scene camera name (MJCF `camera`, USD `Camera`
prim path) is short for `CameraSpec(mount=name)` with that camera's lens.

```python
cams = [
  "front",  # the scene camera, its lens and pose
  rw.CameraSpec(mount="front", fovy=30.0, local=(np.array([0.1, 0.0, 0.0]), np.eye(3))),  # offset from it
  rw.CameraSpec(fovy=60.0, znear=0.05, zfar=10.0),  # free: posed by r.camera_pose
  rw.CameraSpec(intrinsics=(40.0, 40.0, 0.0, 0.0), sensor_size=(36.0, 27.0), mount="front"),  # MuJoCo intrinsics
]
scene = rw.integrations.mujoco.scene(mjm, m)
r = rw.Renderer(scene, nworld=2, cameras=cams, resolution=(64, 48))
eye = wp.transformf((0.0, -3.0, 1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.25))
r.camera_pose.fill_(eye)  # (nworld, ncam) world poses of the free cameras: -z forward, +y up
r.set_clip(znear=0.02, camera=2)  # runtime: no recompilation
out = r.render(d)
```

<!-- SHOWCASE: scenes/camera_models.webp (docs/assets/SHOTS.md) -->
![Pinhole, orthographic, fisheye 180 deg, OpenCV distortion](assets/scenes/camera_models.webp)

### Camera models

| model | how | depth | limits |
| --- | --- | --- | --- |
| pinhole | `CameraSpec(fovy=...)`, or `intrinsics=` + `sensor_size=` | planar or radial (`DepthOutput`) | none; never compiles |
| orthographic | `CameraSpec(projection="orthographic", fovy=view_height_in_m)` | planar | |
| OpenCV distortion | `CameraSpec.from_opencv(K, (width, height), distortion=...)` | planar | lens camera: compiles once on first use ([note](#use-fisheye-and-opencv-cameras)) |
| fisheye | `CameraSpec(projection="fisheye", fovy=220.0)` or `CameraSpec.from_opencv_fisheye(...)` | along the ray | lens camera; no ray bundle on a fisheye |
| ray bundle | `CameraSpec(rays=...)`: one camera-frame ray per pixel | along the ray | lens camera; all-or-none per Renderer, one resolution, no motion vectors |

Pinhole, orthographic and fisheye cameras mix in one Renderer. There is no thin lens (depth of
field): defocus at 1 sample per pixel is noise.

### Use fisheye and OpenCV cameras

**What it is.** Lens cameras reproduce a calibrated real camera: OpenCV distortion, fisheye lenses
and measured ray bundles. Distortion is applied per pixel in the kernel.

**How to use it.**

- **OpenCV calibrations:** `CameraSpec.from_opencv(K, (width, height), distortion=(k1, k2, p1, p2[,
  k3[, k4, k5, k6[, s1, s2, s3, s4]]]))`. `K` is the 3 x 3 matrix in pixels (OpenCV's pixel centres
  are converted). The distortion vector has 4, 5, 8 or 12 entries (the rational model's k4-k6,
  the thin prism's s1-s4; not the tilted sensor). `K` must be finite and zero-skew with bottom row
  `(0, 0, 1)`; anything else raises `ValueError`.
- **Fisheye:** `CameraSpec(projection="fisheye", fovy=220.0)` is an equidistant (f-theta) lens
  whose image height spans `fovy` degrees (up to 360). `CameraSpec.from_opencv_fisheye(K, (width,
  height), (k1, k2, k3, k4))` takes an OpenCV `cv::fisheye` (Kannala-Brandt) calibration.
- **Per-world distortion:** `PerWorld(camera=["distortion"])`, then
  `r.per_world["camera.distortion"]` is an (nworld, ncam) table of 12 floats per camera (OpenCV's
  order; a fisheye's 4 then zeros). Lens randomisation without recompiling; motion vectors use each
  frame's own coefficients.

**Outputs.** A fisheye's depth and clip planes are distances along the ray. Pixels outside the
image circle, or past the model's fold, are misses (depth / seg miss values, the background color).

**Limits.** Each lens model's program compiles once per kernel build on first use (1-9 min, then
cached, plus seconds to create its module). `FAST`'s OpenCV and fisheye programs are prebuilt.
`Renderer.warm(scene, settings)` with the lens camera does it ahead
([zero compiles](performance.md#zero-compiles)).

<!-- SHOWCASE: scenes/lens_distortion.webp (docs/assets/SHOTS.md) -->
[![Per-world lens distortion, k1 from -0.4 to 0.4, no recompile](assets/scenes/lens_distortion.webp)](assets/scenes/lens_distortion.mp4)

**Ray bundles** (measured lens models, LiDAR-like patterns) give each pixel its own camera-frame
ray. Prefer `projection="fisheye"` for fisheye lenses.

```python
w, h = 32, 24
u, v = np.meshgrid((np.arange(w) + 0.5) / w * 2 - 1, (np.arange(h) + 0.5) / h * 2 - 1)
theta, phi = np.hypot(u, v) * np.radians(90.0), np.arctan2(-v, u)  # equidistant fisheye, 180 deg
dirs = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], -1)
rays = np.stack([np.zeros_like(dirs), dirs], axis=2).astype(np.float32)  # (h, w, 2, 3): origin, direction
fisheye = rw.CameraSpec(mount="front", rays=rays)
r = rw.Renderer(scene, nworld=2, cameras=[fisheye], resolution=(w, h))
out = r.render(d)
r.set_camera_rays([rays * np.array([1.0, 1.0, 1.0], np.float32)])  # new rays, no recompilation
```

The ray layout is Newton's `camera_rays`: Newton's `compute_camera_rays_*` helpers (pinhole,
OpenCV, fisheye) produce it directly.

### Camera options

| option | what |
| --- | --- |
| `CameraSpec(outputs=("depth",))` | the camera writes only some outputs, e.g. a depth-only wrist camera next to an RGB head camera. A camera without shaded outputs costs a geometry-only render. Groups: every shaded output, depth, seg with semantic; position, uv, material, a geometric normal and motion must be listed. Buffers stay `(nworld, ncam, H, W)`: no memory is saved. |
| `CameraSpec(resolution=(w, h))` | per-camera resolution; the outputs are then per camera (`out.cameras[c]`). Motion needs one size for every camera. |
| `CameraSpec(exposure=stops)` | brightens one camera's image (USD camera exposure) |
| `znear` / `zfar` | clip the camera rays (default: the scene's; MuJoCo `vis.map.znear / zfar x extent`, USD `clippingRange`). Nearer surfaces are not seen; depth still counts from the eye; shadow, reflection and GI rays are not clipped. MuJoCo Warp's ray caster has no near clip. |
| `PerWorld(camera=["fovy"])` or `["intrinsics"]` | per-world intrinsics: `r.per_world["camera.fovy"]` is an (nworld, ncam) table |
| `CameraSpec.conform="fit"` | USD and glTF cameras frame like Hydra when the image's aspect differs: the whole aperture shows. MuJoCo cameras keep MuJoCo's rule. |

## Segmentation

`seg` holds the instance's segment id (MuJoCo: the geom id; -1 where nothing is hit). `semantic`
holds class ids from name patterns, set at run time:

```python
settings = rw.RenderSettings(outputs=rw.Outputs.of("rgb", "semantic"))
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48), settings=settings)
labels = r.set_semantics({"ground": ["floor"], "object": ["ball", "crate"]})
assert labels == ("background", "unlabelled", "ground", "object")  # class ids 0, 1, 2, 3
semantic = r.render(d).semantic.numpy()
```

<!-- SHOWCASE: scenes/segmentation.webp (docs/assets/SHOTS.md) -->
![Instance and semantic segmentation](assets/scenes/segmentation.webp)

USD `UsdSemantics` labels are kept in the host scene (`Instances.label`).

## The load report

Every host-scene load counts what it read, dropped and approximated. Read it first when something
is missing: an unsupported shader input, a texture that failed to decode, a light with no
counterpart.

```python
report = host.report
print(report)  # counts, dropped / approximated attributes with an example path, timings
assert report.counts["images"] == 1 and not report.dropped  # a USD load also counts prims: usd.Mesh, ...
```

`host.report` is a `rw.scene.LoadReport`: `counts`, `dropped`, `approximated` (attribute ->
occurrences), `examples` (the first prim path of each) and `timings`. `rw.upload` adds `gpu.*`
entries (textures resized, materials approximated, emissive panels turned into lights). The
MuJoCo adapter maps everything it can and warns about the rest.

## Scene scale

Lengths are metres. Scenes from 1 cm to 10 km across render the same image as the scene scaled to
metres: ray offsets and the lights' falloff floor scale with the scene. Beyond that:

- one instance spanning more than 10 km counts as unbounded (sky, terrain) and stays out of the GI
  probe box;
- light-grid cells stay at least `lighting.light_grid_cell` (1 m) wide: a millimetre scene is one
  cell, correct but without culling.

## Reference

### OpenPBR inputs

Every OpenPBR input, where it lives, its default and how the core renders it:

<!-- openpbr-inputs:begin (generated by tools/settings_table.py; do not edit) -->
| input | in | default | status | rendered as |
| --- | --- | --- | --- | --- |
| `base_weight` | `Material` | `1.0` | supported | multiplies the base color |
| `base_color` | `Material` | `(0.8, 0.8, 0.8)` | supported | the Lambert base color |
| `base_diffuse_roughness` | `extras` | `0.0` | approximated | constant/textured EON with sampled area/environment; existing specular/coat coupling and probe GI; subsurface/transmission combinations unsupported; shared material UV transform |
| `base_metalness` | `Material` | `0.0` | supported | a linear mix of metal and dielectric |
| `specular_weight` | `Material` | `1.0` | approximated | weighted dielectric F0 with index-match zero; whole metal lobe scaled; Schlick and mixed-metal compensation approximate |
| `specular_color` | `Material` | `(1.0, 1.0, 1.0)` | missing | carried: the dielectric tint and the metal edge tint are not rendered |
| `specular_roughness` | `Material` | `0.3` | supported | GGX alpha = r^2, roughened under an OpenPBR coat |
| `specular_ior` | `Material` | `1.5` | supported | F0 (the ratio to `coat_ior` under an OpenPBR coat), Schlick Fresnel |
| `specular_roughness_anisotropy` | `extras` | `0.0` | supported | anisotropic GGX D / Smith V (alpha_t = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_b = (1 - a) alpha_t) along the mesh's UV tangent (generated per vertex) or the primitive's, turned by `ext.anisotropy_rotation`; bent-normal image-based lighting, anisotropic VNDF reflection rays; area lights approximate it with the bent normal |
| `transmission_weight` | `Material` | `0.0` | approximated | a specular-only see-through layer; a solid (weight 1, not thin-walled) bends camera rays (Snell, total internal reflection), shadow and reflection rays pass it straight |
| `transmission_color` | `Material` | `(1.0, 1.0, 1.0)` | approximated | a closed solid's Beer-Lambert extinction (reached at `transmission_depth`, else at its instances' median mean chord), else blended into the base color (the tint at each surface crossed) |
| `transmission_depth` | `extras` | `0.0` | supported | the depth at which a solid's absorption reaches `transmission_color` |
| `transmission_scatter` | `extras` | `(0.0, 0.0, 0.0)` | missing | no volume scattering |
| `transmission_scatter_anisotropy` | `extras` | `0.0` | missing | no volume scattering |
| `transmission_dispersion_scale` | `extras` | `0.0` | missing | no dispersion |
| `transmission_dispersion_abbe_number` | `extras` | `20.0` | missing | no dispersion |
| `subsurface_weight` | `extras` | `0.0` | approximated | `subsurface_color` blended into the base color (the spec's LOD rule) |
| `subsurface_color` | `extras` | `(0.8, 0.8, 0.8)` | approximated | the diffuse color under `subsurface_weight` |
| `subsurface_radius` | `extras` | `1.0` | missing | no subsurface transport |
| `subsurface_radius_scale` | `extras` | `(1.0, 0.5, 0.25)` | missing | no subsurface transport |
| `subsurface_scatter_anisotropy` | `extras` | `0.0` | missing | no subsurface transport |
| `fuzz_weight` | `extras` | `0.0` | approximated | a Charlie sheen lobe (Neubelt visibility) over the coated base, albedo-scaled 1 - F E_fuzz (E from the cloth table, at most 1); image-based light from the prefiltered environment at the fuzz roughness |
| `fuzz_color` | `extras` | `(1.0, 1.0, 1.0)` | supported | the fuzz lobe's tint (constant or textured) |
| `fuzz_roughness` | `extras` | `0.5` | supported | the Charlie roughness alpha (glTF sheenRoughness^2) |
| `coat_weight` | `Material` | `0.0` | supported | the coat's coverage C: C f_coat + lerp(1, T (1 - E_coat), C) f_base |
| `coat_color` | `extras` | `(1.0, 1.0, 1.0)` | supported | the coat's transmittance T^(1 / mu_i + 1 / mu_o) (coat_color = T^2 at normal incidence) |
| `coat_roughness` | `Material` | `0.0` | supported | the coat's GGX roughness |
| `coat_roughness_anisotropy` | `extras` | `0.0` | missing | isotropic coat |
| `coat_ior` | `Material` | `1.6` | supported | the coat's own Schlick Fresnel F0((n - 1) / (n + 1))^2 and the base IOR ratio |
| `coat_darkening` | `extras` | `1.0` | supported | OpenPBR's lerp(1, (1 - K) / (1 - E_b K), C d) on the base under an OpenPBR coat |
| `thin_film_weight` | `extras` | `0.0` | missing | no thin-film iridescence |
| `thin_film_thickness` | `extras` | `0.5` | missing | no thin-film iridescence |
| `thin_film_ior` | `extras` | `1.4` | missing | no thin-film iridescence |
| `emission_luminance` | `Material` | `0.0` | supported | radiance [nits] |
| `emission_color` | `Material` | `(1.0, 1.0, 1.0)` | approximated | not attenuated by an OpenPBR coat (reported) |
| `geometry_opacity` | `Material` | `1.0` | supported | presence, with `ext.alpha_mode` BLEND |
| `geometry_thin_walled` | `Material` | `False` | supported | the core's surfaces are thin; a transmissive thin wall reflects its two interfaces' 2 F / (1 + F) (Schlick on that F0) |
| `geometry_normal` | `Slot.GEOMETRY_NORMAL` |  | supported | the normal map (`Slot.GEOMETRY_NORMAL`; strengths / flips: its scale / bias) |
| `geometry_coat_normal` | `extras.textures` |  | supported | the coat's own normal map (unbound: the unmapped normal; COAT_BASE_NORMAL: the base's) |
| `geometry_tangent` | `extras.textures` |  | missing | the mesh's UV tangent (per vertex) or the primitive's |
| `geometry_coat_tangent` | `extras.textures` |  | missing | isotropic coat: no tangent needed |
| `geometry_detail_normal` | `extras.textures` |  | supported | raywarp extension: a detail normal map combined with geometry_normal by RNM, own UV transform |
<!-- openpbr-inputs:end -->

### MuJoCo materials

How each MuJoCo material field maps to `rw.Material`:

<!-- mujoco-materials:begin (generated by tools/settings_table.py from raywarp.integrations.mujoco.materials; do not edit) -->
| MuJoCo | OpenPBR `rw.Material` |
| --- | --- |
| `mat_rgba rgb, a` | base_color, geometry_opacity (ext.alpha_mode BLEND when a < 1) |
| `mat_specular s` | specular_weight = 4 s^2 at specular_ior 1.5 (F0 = 0.16 s^2, exact): the device reflectance r is s. MuJoCo metals ignore s, and s = 0 keeps F90: ext.flags DIELECTRIC_SPECULAR_LEVEL preserves these native semantics |
| `mat_shininess` | specular_roughness = 1 - shininess, clipped, when roughness is unset (a batched live m.mat_shininess: per world, `shininess_inputs`) |
| `mat_roughness / metallic` | specular_roughness / base_metalness (-1, unset: `unset_*`, else 1 - shininess / 0, glTF) |
| `mat_reflectance` | ext.mirror (MuJoCo's mirror reflection) |
| `mat_emission e` | emission_luminance = e in MuJoCo's display-relative unit (`Capabilities.physical_emission` False); emission_color = rgb with the base texture (the core scales the textured base color), or the role="emissive" texture (white constant) |
| `mat_texid[role]` | rgb -> BASE_COLOR (wins over rgba); rgba -> BASE_COLOR + GEOMETRY_OPACITY (alpha); opacity -> GEOMETRY_OPACITY (red; with an rgba layer the device multiplies both: reported); orm -> OCCLUSION / SPECULAR_ROUGHNESS / BASE_METALNESS (red / green / blue; wins over the separate layers, read red); normal -> GEOMETRY_NORMAL (the plain unpack 2 t - 1, +Y up: the device's own decode of MuJoCo's top-left UVs); emissive -> EMISSION_COLOR |
| `mat_texrepeat` | each TextureRef's uv_xform scale |
| `mat_texuniform` | ext.flags UV_WORLD |
| `<numeric raywarp:alpha_cutoff:<mat>> c > 0` | ext.alpha_mode MASK, ext.alpha_cutoff c. The device tests the opacity layer's texel against c and draws what passes opaque: geometry_opacity 1, GEOMETRY_OPACITY from the opacity layer only (the rgba alpha is ignored; without an opacity layer the surface is opaque: both reported) |
<!-- mujoco-materials:end -->

### USD light dialects (reference)

| Authored DistantLight | Illuminance [lux] from `intensity * 2 ** exposure` |
| --- | --- |
| RTX legacy | already lux, irrespective of `normalize` |
| UsdLux, `normalize = true` | already lux |
| UsdLux, `normalize = false`, nonzero angular diameter | multiply by `pi * sin(angle / 2) ** 2` (angle in radians) |
| zero angular diameter | size factor 1, as UsdLux specifies |

## See also

- [Quickstart](quickstart.md): the first USD, Newton and MuJoCo renders
- [Settings](settings.md): exposure, GI, shadows, every field
- [Robotics](robotics.md): per-world tables, outputs
- [Troubleshooting](faq.md): dark images, missing textures
