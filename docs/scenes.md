# Scenes: lights, environments, materials, cameras

What a scene is made of, per adapter, and how to check what was kept. The snippets build on
one small MuJoCo scene; the host-scene sections (`rw.SceneBuilder`) are what the USD and
Newton adapters produce under the hood.

**Scene scale.** Lengths are metres. Scenes from 1 cm to 10 km across render the same image as
the scene scaled to metres: the secondary rays' origin offset and the lights' inverse-square
floor scale with the scene's extent below 10 cm (ground planes left out), and the offset with
each instance's largest vertex coordinate (a 20 km ground quad shows no shadow acne). Beyond that range, a single
instance spanning more than 10 km counts as unbounded (sky, terrain) and stays out of the GI
probe box, and the light grid's cells stay at least `lighting.light_grid_cell` (1 m) wide (a
millimetre scene is one cell: correct, without the grid's culling).

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
```

## Lights

**MuJoCo.** Only native MJCF fields are read, so existing models render unchanged. MuJoCo has
two light models and the adapter follows MuJoCo's own (Filament renderer) rule:

| MJCF | rendered as |
| --- | --- |
| any light with `intensity` > 0 (photometric model) | every light physical: color `diffuse` x `intensity` (lux for directional lights, candela otherwise), inverse-square falloff windowed at `range`, spot cone with inner angle 1 - `softness`, soft shadows from `bulbradius`; intensity-0 lights dark, headlight off |
| every `intensity` 0 (classic model, MuJoCo's GL lights) | mapped exactly: no distance falloff, a white diffuse surface shows the light's `diffuse` color at `Lighting.ev100` (default 14.97), also for moving lights; `visual/headlight` is a camera-attached light (`lighting.camera_light`), its `ambient` plus the lights' `ambient` a constant ambient |
| Classic point/spot `attenuation` | rendered from live constant/linear/quadratic coefficients; default (1, 0, 0) has no distance falloff |
| Classic `exponent` | the spot cone's inner angle (its falloff crosses one half where cos^exponent does) |
| Classic `specular` | not rendered (one color drives both lobes) |
| `type="image"` with a `texture` | image-based lighting: the texture's radiance x `intensity` [nits] (also the background) |
| skybox texture | background only (MuJoCo's skybox lights nothing) when no envmap (`lighting=`) or image light is given |

Lights with a `bulbradius` are sphere lights; under `lighting.area_lights="ltc"` (the default;
"point", as in `FAST`, shades them as points) they are shaded as true area lights (LTC highlights,
soft shadows from `shadows.samples` points on the light). The scene above is photometric (`intensity="3"`): a dim 3 lux sun, so render it
with a matching manual exposure, `exposure.ev100` 0 (each +1 halves the brightness; 14.97 is
Filament's sunny-day camera, for tens of thousands of lux; the default, `exposure.mode="auto"`,
meters each image automatically).

```python
scene = rw.integrations.mujoco.scene(mjm, m)
dim = rw.RenderSettings().replace(**{"exposure.mode": "manual", "exposure.ev100": 0.0})
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48), settings=dim)
assert r.render(d).rgb.numpy()[..., :3].mean() > 30
```

**USD.** UsdLux `DistantLight` (illuminance [lux]), `SphereLight`, `RectLight`, `DiskLight`,
`CylinderLight` (surface radiance [nits]; `ShapingAPI` cones) and `DomeLight` (the
environment). As in Omniverse RTX, camera rays do not see a light's surface unless it authors
`visibleInPrimaryRay = true` (`Light.visible_in_primary_ray`; `Light.rect(..., visible=True)` from
Python); a dome is the background either way. Emissive materials glow; flat rectangular emissive
panels also become rect lights at upload (`lighting.emissive_lights`). Dropped inputs (IES profiles, light textures) are
counted in the load report.

**Host scenes.** `SceneBuilder.add_light(rw.Light(kind=..., intensity=..., xform=...))`: a UsdLux
record (`raywarp.scene.model.lights.Light`: UsdLux's inputs, names and defaults) with kinds
`LightKind.DISTANT`, `SPHERE` (a point light: `radius=0`, `treat_as_point=True`; a spot: add
`shaping=rw.scene.Shaping(cone_angle=<deg>)`), `RECT`, `DISK`, `CYLINDER`; sizes in world metres, the
light emits along its local -Z; the radiance is `intensity x 2^exposure x color`
(x blackbody(`color_temperature`) when enabled), derived at upload.

## Environment maps (HDRI)

An equirectangular HDR image lights the scene (split-sum image-based lighting, a prefiltered
map baked once and cached on disk) and fills the background.

```python
sky = np.ones((32, 64, 3), np.float32)  # (height, width, 3) linear radiance; row 0 is the zenith
sky[16:] *= 0.3  # a darker ground half
scene = rw.integrations.mujoco.scene(mjm, m, lighting=rw.Lighting(envmap=sky, envmap_rotation=90.0))  # or "studio.hdr"
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48))
out = r.render(d)
```

- `Lighting.envmap` takes a path (`.hdr`, `.exr` linear; 8-bit images sRGB), a float array
  of linear radiance, or a list of those: `Renderer.per_world["env.id"]` then picks one per world
  (declared with `PerWorld(env=["id"])`, [robotics recipes](robotics.md)).
- `envmap_intensity` scales it (None: display-relative, a texel of 1 shows as white at the
  default exposure); `envmap_rotation` turns it about +z [degrees]; per world: `"env.intensity"`,
  `"env.yaw"`.
- GI (on by default) keeps the sky out of enclosed spaces ([a closed
  room](#a-closed-room-lit-by-an-environment-map));
  `environment.background=False` shows `environment.background_color` instead of the map.
- USD domes and host scenes (`SceneBuilder.add_environment(rw.scene.EnvironmentLight(texture=...))`) work the same way.
- A map with a dominant sun (a sunny sky: a compact region of texels far above the sky holding at least 30 % of the
  irradiance of a surface facing it) has the sun cut out and lit as a directional light with shadow rays and its
  disc's size; the map keeps the rest of the light, and the camera sees the disc drawn back over its sky (a mirror shows
  it as the light's highlight). Overcast skies and studios with several softboxes have none.
  The sun follows each world's `env.id`, `env.yaw` and `env.intensity`; it is no light of the light tables (a USD
  DomeLight stays one light). `rw.SceneOptions(env_sun=False)` keeps the sun in the map.

Sharp environment reflections filter unresolved texels using the existing ray cone and the
selected map's resolution. Rough reflections still use the cached GGX stack. This footprint
approximation does not account for surface curvature or anisotropic latitude distortion; it
does not replace antialiasing. Polar GGX prefilter accuracy remains an open validation item.

## Lighting a scene

Every adapter takes the same `lighting=rw.Lighting(...)`, applied before the scene is
uploaded: extra lights (`rw.Light`), a constant `ambient` fill, environment maps
(`envmap`, `envmap_intensity`, `envmap_rotation`) and a camera `headlight`
(`rw.Headlight`). Use it when an imported scene is missing light, or to debug one. Its
display-relative quantities (a light made without an intensity, the ambient, the headlight, an
envmap without an intensity) show a white surface at the given color at `lighting.ev100`
(default 14.97): render with the same manual `exposure.ev100`.

`rw.Light` is the one light description, in physical units (UsdLux inputs):
`Light.sun(direction, illuminance=lux)`, `Light.point(position, intensity=cd | power=lm,
radius=0)`, `Light.spot(position, direction, cone_angle=45, ...)` (both take an IES profile,
`ies=rw.scene.Ies(path)`), `Light.sphere`, `Light.rect`, `Light.disk`, `Light.tube` (radiance [nits]).
A sun, point or spot made without an intensity is display-relative.

```python
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

**The studio rig: the quickest fix for a dark scene.** `rw.Lighting.studio()` is a
neutral rig that makes any scene readable at the default exposure: a sky dome (bluish zenith,
white horizon, dim warm ground), a warm shadowed key light 50 deg above the horizon and a weak
cool fill (`fill=False` drops it). It adds to the scene's own lights; `replace=True` drops them
(and the headlight, the ambient and image-based lighting) instead. `up="y"` for a Y-up Newton
model (`model.up_axis`); MuJoCo and USD scenes are Z-up after ingest. Its values are
display-relative (calibrated for `ev100`), so it exposes the same in photometric and in MuJoCo's
classic scenes.

MuJoCo's lights are the model's own (MuJoCo Warp's light tables are aliased, not copied):
`rw.integrations.mujoco.add_lights(spec, lighting)` adds `lighting`'s lights to the `MjSpec` before
compiling (in the model's light model: classic or photometric) and returns the rest for
`rw.integrations.mujoco.scene` (passing lights to `rw.integrations.mujoco.scene` raises).

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

A host scene that is already built (a `.wpbr` file, a USD or Newton scene) gets the same with
`Scene.with_lighting`, which returns a new scene (point and spot lights made without an
intensity are display-relative: their intensity follows their distance to the scene's centre):

```python
b = rw.SceneBuilder()
grey = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8)))
b.add_instance("box", size=(3.0, 3.0, 0.05), material=grey, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
host = b.build()  # or rw.Scene.load("scene.wpbr")
lit = host.with_lighting(rw.Lighting(lights=[rw.Light.sun((0.3, 0.2, -1.0))], ambient=(0.1, 0.1, 0.1)))
gpu = rw.upload(lit)
studio = rw.upload(host.with_lighting(rw.Lighting.studio(replace=True)))  # the rig alone
```

USD and Newton take it at ingest (USD adds it after the `.wpbr` cache, so the cache key does not
change); Newton's front ends (`SensorCamera`, `ViewerRaywarpHeadless`) take it as `lighting=` too, and Isaac
Lab as `RaywarpRendererCfg(lighting=...)`:

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

**Indirect light (GI).** Light that reaches a surface only after a bounce (a room lit through a
window, a ceiling above downlights) needs probe GI: on by default and in `FAST` (off under unlit shading). Turn it off with
`gi.enabled=False`; its grid fields (spacing, rays, bake updates) are in `settings.GI`:

```python
fast_with_gi = (
  rw.RenderSettings().replace(rw.FAST).replace(**{"gi.enabled": True, "gi.spacing": 1.0, "gi.rays": 64, "gi.bake_updates": 8})
)
r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=["front"], resolution=(64, 48), settings=fast_with_gi)
```

GI is read from an ambient-cube texture resampled from the probes after each update: one fetch
of its static wall data and three filtered reads. Its texels are checked against the probes in
sight of them (texels inside geometry or seen by no probe take their neighbours' values), and the
filtering never reaches across a wall, whatever its thickness: closed rooms stay dark and corners
do not pick up light from outside. Per-world randomised materials, lights or environments: a few
variants get one exactly baked volume each; mostly distinct worlds (continuous randomisation)
share one volume tinted per world (a coarse directional ratio field from probe rays shaded in
that world, recomputed in the frame its inputs change). The developer tool `internal/tools/dev/gi_debug.py`
(`view(renderer, kind)`) draws the GI lookups, the probes and their sky visibility over a rendered frame (outputs `position` and `normal`).

### A closed room lit by an environment map

An environment map lights what sees the sky. The GI probes (`gi.enabled`, on by
default) keep it out of enclosed spaces: inside a closed room it brings almost nothing, and through
a window only what the window lets in, which is what makes window light right. Keep GI on, and
give the room the light it lacks:

- lights inside the room (`Lighting(lights=[rw.Light.point(...)])`), or
  `Lighting.studio()` (its fill light is unshadowed, so it reaches a closed room);
- when the sky does not matter (no window light wanted): no environment lighting
  (`environment.enabled=False`) and an ambient fill, `Lighting(ambient=...)`. The ambient is
  never occluded (by design), so it lights the room evenly. It is a fill for scenes that the
  environment does not light: with an environment map lighting the scene, the environment is
  the ambient term and `ambient` is not added;
- a render-time fill on top of everything (GI, environment, lights): `lighting.fill`
  (runtime, display-relative, diffuse only), the headlight and `rw.Lighting.fill_lights`
  ([dark interiors](faq.md#dark-interiors)).

Turning GI off indoors makes the sky light every room as if it had no roof (the Isaac Office:
+2.6 EV, washed out). It only suits open scenes, where it changes little.

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

## Light units and exposure

The renderer's `hdr` output is scene-linear radiance before display exposure and tone mapping.
For a Lambertian surface with reflectance `rho`, normal-incidence illuminance `E` produces
`rho * E / pi` radiance. Calibration checks use that equation directly, without comparing
8-bit tone-mapped images as though they were radiance.

Exposure is automatic by default (`exposure.mode = "auto"`): each camera image is metered on a
histogram of its scene luminance (the 10-90 % band shown at middle grey 0.18, darkened by at most
1 stop when a large area would clip; settings `exposure.*`). By default the first frame is metered
and the exposure locked per world and camera until `reset_history` (episode reset,
re-randomisation); `exposure.adaptation="continuous"` follows the scene smoothly and `"per_frame"`
meters every frame on its own; `exposure.mode = "manual"` exposes at
`exposure.ev100`. Every tone mapper receives the same exposure scale,
`2 ** -EV100`; `aces_rtx` supplies RTX's ACES approximation with no separate exposure multiplier.
Display-relative emission, headlight and ambient terms are derived from the exposure (under
automatic exposure from the reference EV100 14.97) and are checked separately. Authored USD camera exposure
and the complete lens-exposure path are not calibrated yet.

The USD adapter reads lights in one of two dialects. The default is the UsdLux spec. Omniverse
RTX's legacy dialect (`omni:rtx:usdluxVersion` < 2505, which Kit assumes when the attribute is
absent) applies to a light whose authored version says so or, without a version, to a light whose
strongest opinion comes from a layer saved by Omniverse Kit (`customLayerData` with `omni_layer`,
`renderSettings` or `cameraSettings`). Kit's attribute naming alone (`omni:` inputs, or pre-21.02
names such as `intensity` next to `inputs:intensity`) also reads as RTX legacy, and pre-21.02
names alone read as the spec; both are ambiguous, reported as `UsdLux.dialect.ambiguous(...)` and
warned. `usd.Options(lux_dialect=rw.formats.usd.Dialect.RTX_LEGACY)` sets the dialect of unversioned
lights for a stage without marks (one built live in a Kit app).

| Authored DistantLight | Illuminance [lux] from `intensity * 2 ** exposure` |
| --- | --- |
| RTX legacy | Already lux, irrespective of `normalize` |
| UsdLux, `normalize = true` | Already lux |
| UsdLux, `normalize = false`, nonzero angular diameter | Multiply by `pi * sin(angle / 2) ** 2`, with angle in radians |
| Zero angular diameter | Size factor 1, as specified by UsdLux |

A DomeLight's map pole is its local +Y with OpenEXR's latlong layout in the spec (DomeLight_1:
`poleAxis`, "scene" = the stage's up axis), and its local +Z with RTX's layout in RTX legacy. Its
texture's authored `colorSpace` decides the decoding (unauthored: by file type). `texture:format`
`automatic` and `latlong` read the file as latlong (as Hydra Storm); `angular`, `mirroredBall` and
`cubeMapVerticalCross` are resampled to latlong once at load, centred on the latlong's centre
column (conventions in `raywarp/formats/usd/stage/dome_formats.py`). An unknown format or an image whose shape
cannot hold it (a non-square ball, a cross that is not 3:4) raises a ValueError naming the
dome: no mapping of it would be right.

This follows [UsdLux LightAPI's normalization rule](https://openusd.org/release/user_guides/schemas/usdLux/LightAPI.html#inputs-normalize)
and [RTX's documented lighting-version selection](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html#controlling-the-usdlux-behavior).
The current core supports angular diameters through 180 degrees; angular size controls its
shadow sampling and widens its specular lobes (`light_evaluators.source_widening`: a sized sun
is a soft highlight, not a dot). The diffuse response uses the disc's central direction.

Run the focused checks with:

```sh
uv run pytest -q tests/calibration/test_units.py -n 2
```

To retain the tested HDR arrays and annotated, enlarged images during a coordinated evidence
run, set `RAYWARP_CALIBRATION_OUTPUT` to the output directory. Each image names its analytic
expected radiance and its display scale; different scales make very dim and bright test cases
visible, so compare the saved HDR values rather than their presentation brightness. Record the
commit, dependencies, backend and hardware alongside the artifacts.

The checks cover directional-light units and the common EV scale and RTX curve. Not yet covered:
finite-area lights and attenuation, shaping / IES, environment transport, full authored camera
exposure, scene-unit / up-axis invariance, a ColorChecker and independent RTX HDR evidence.

### Photometric authoring units

`scene/lighting/photometry.py` converts authoring units once into the canonical UsdLux record (radiance,
`normalize`, shaping), from the record as the renderer shades it: luminous flux [lm]
(`with_power`; spot cones "reflector", Filament's focused spot, or "absorber"), illuminance [lux]
at a distance on the light's axis (`with_illuminance`: exact for spheres, disks and rects),
watts x efficacy or efficiency (`watts_to_lumens`), and colour temperatures (`cct_color`:
UsdLux's blackbody table, Krystek's Planckian locus, the CIE daylight series) or a sampled
spectrum (`spectrum_color`). `Light.point` / `Light.spot` take `power` (`watts_to_lumens` for
watts) and `spot_mode`, the sun, point and spot `temperature` + `illuminant`; the USD adapter reads Omniverse's
`PhotometricAreaLightAPI` (power, illuminance at distance), `PhotometricDistantLightAPI` (lux) and
`PhysicalLightIlluminantAPI` (`physical:illuminant`: white, blackbody, illuminant D, custom)
(`formats/usd/stage/physical.py`); `PhotometricDomeLightAPI` scales the dome to its illuminance on an upward
surface at upload. MuJoCo lights keep their
own model (`raywarp.integrations.mujoco.lights`); MJCF has no photometric profiles (IES).

### IES profiles

UsdLux ShapingAPI `ies:file` (LM-63 1986-2019, photometric types C, B and A, any symmetry) is
evaluated per light in direct lighting (`core/shade/lights/ies.py`, compiled in only when a light has a
profile). Three modes: `ies:normalize` (the profile over its mean intensity: the unshaped light's
flux), "mask" (the profile over its peak times the light's intensity: plain UsdLux) and "measured"
(the file's candela values times intensity x 2^exposure: `PhysicalLightIlluminantAPI`, as
Omniverse). Orientation: type C nadir along the light's -Z, horizontal angle 0 along +X, 90 along
+Y; types A / B photometric axis along -Z, up +Y. `ies:angleScale` follows the UsdLux formula. A
TILT=INCLUDE factor follows the light's orientation per frame (+Z up). Rect / disk / sphere lights
with a profile alone keep their LTC area form, times the profile's mean over the emitter.

### Emission

Physical emission (host scenes) is radiance [nits] at any exposure. Display-relative emission
(MuJoCo) is `emission x color / exposure`, so its exposed value is fixed: Filament's
exposure-compensated emissive with a compensation of 3 + log2(1.2) EV. A per-material EV
compensation (Filament's `emissive.w`) is not added: no source format carries one, and it would
make a third mode between the two calibrated ones (`tests/calibration/test_emission.py`).

## Materials and textures

### The material: OpenPBR Surface

Every adapter converts its materials into one record, `rw.Material`: an
[OpenPBR Surface 1.1.1](https://academysoftwarefoundation.github.io/OpenPBR/) surface with the
spec's input names and defaults (base color 0.8, `specular_roughness` 0.3, `specular_ior` 1.5,
`coat_ior` 1.6). The device values are derived from it once, at upload (`core/pipeline/derive/materials.py`:
the one place where values are converted). The core renders OpenPBR's real-time subset, as the
spec's "Flexibility of implementation" clause allows: one GGX specular lobe (alpha = roughness²,
Schlick Fresnel) over a Lambert base, an optional GGX coat, emission and opacity.

A `rw.Material` holds:
- 16 inputs as fields: the ones the core renders or derives from, and two it carries without
  rendering them (`specular_color`, `geometry_thin_walled`);
- `extras=rw.scene.OpenPbrExtras(...)`, the other OpenPBR inputs. They are carried (`.wpbr`
  included) and reported `gpu.material.<input>` when authored, but not rendered;
- `ext=rw.scene.RaywarpExt(...)`, the raywarp extensions (below);
- textures per slot (`rw.scene.Slot`, named after the input each drives; `rw.scene.TextureRef`: channel,
  scale / bias, UV transform, UV set). A bound slot's value is `field x (scale x texel + bias)`.

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
host = b.build()  # a rw.Scene (b.build("scene.wpbr") also writes the cache file)
r = rw.Renderer(rw.upload(host), nworld=1, cameras=None, resolution=(64, 48))
out = r.render()
```

Host (USD, glTF) cameras frame like Hydra when the image's aspect differs from the apertures'
(`CameraSpec.conform="fit"`, OpenUSD's CameraUtilFit): the whole aperture shows, extended along its
shorter side; aperture offsets keep the window's centre. MuJoCo cameras keep MuJoCo's rule (a fovy
camera keeps its vertical field of view; intrinsics crop the sensor to the image).

Diffuse roughness textures use `OpenPbrExtras.textures` with the input name
`"base_diffuse_roughness"`. Set the scalar `base_diffuse_roughness=1.0` for the texture alone;
USD/MaterialX connected inputs do this automatically. The scalar multiplies the selected,
affine-adjusted texture channel. The image and mip chain are shared across worlds.

For a prepared diffuse map, declare `PerWorld(material=["tex_affine"])` to change its scale/bias
per world through the existing texture-affine table. Declare `"base_diffuse_roughness"` to change
the scalar factor. Declaring that factor together with `"texture"` or `"tex_affine"` reserves the
optional role once even when initially unbound; ids select from the scene's prepared textures.
Ordinary updates reuse the storage and do not rebake images. The optional role is
`raywarp._base.types.TexRole.DIFFUSE_ROUGHNESS`; scenes without it retain the ten-role layout.

The renderer currently uses one UV set/transform per material, reporting conflicting bindings.
Declaring `PerWorld(material=["instance.material"])` reserves shared mapping metadata when
prepared materials use different mesh UV sets or transforms. Selecting a material then selects
its UV frame for color, normal maps, observations and cutout rays without per-world mesh copies.
A missing second UV set falls back to UV0 and is reported. Declared swaps activate the exact
prepared texture-role, alpha-cutout and world-projection superset; undeclared scenes keep their
initial-material specialization. World-projected cutouts use the same projected coordinates
for camera, shadow and reflected rays. Prepared mirror, specular-curve, coat, transmission and
vertex-color capabilities are also reserved. A device update selects the current material's
shading flags and vertex-color eligibility, composing declared coat inputs in the same table;
mesh/color data stays shared. An explicit `instance.shading` declaration instead owns those bits
(including `ShadingFlag.VERTEX_COLOR`), and is never overwritten by material selection.
Fitted emitter/shadow-transmission coupling remains separately tracked. EON retains the documented
specular/coat and probe-GI approximations, and mixed SSS/transmission remains unsupported.

Every OpenPBR input, where it lives, its default and how the core renders it (generated from
`raywarp.scene.model.openpbr.COVERAGE`, which also drives the upload's reports):

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

**Coats.** OpenPBR's coat semantics apply to OpenPBR, Standard Surface and other coupled sources:
the base reflects at the IOR ratio `specular_ior` / `coat_ior` under the coat, the coat roughens
the base, and the coat weight is scaled by F0(`coat_ior`) / 0.04. glTF, UsdPreviewSurface,
OmniPBR and OmniSurface coats carry `MaterialFlags.COAT_GLTF` (at `coat_ior` 1.5): the base keeps
its F0 against air and is neither roughened nor darkened.

Distilled MDL layers keep their independent metal lobe when the dielectric layer's weight is
zero. Non-Unreal layers carry `DIELECTRIC_SPECULAR_LEVEL`: the existing approximation changes
dielectric F0 while keeping dielectric grazing reflectance at 1. Where the authored layer's
weight or curve changes that endpoint, the load report records
`mdl.specular_grazing(fixed at 1)`. A standalone tinted metal also reports its Schlick
approximation (`mdl.metal_fresnel(Schlick instead of constant tint)`). These source-layer
semantics differ from canonical OpenPBR's weight over both metal and dielectric response.
A genuinely diffuse-only MDL material has zero canonical specular weight and no native flag.

**Normal maps** (`rw.scene.model.normal_map`). A `Slot.GEOMETRY_NORMAL` binding's value, `scale` x
texel + `bias` per component, is the tangent-space normal: x along the tangent dp/du, y along
+dp/dv (the bitangent follows the UV handedness), z along the shading normal, with UVs whose origin
is the image's bottom-left (USD `st`, MaterialX). That is the OpenGL / glTF / Blender convention
(+Y up); the plain 8-bit unpack is scale (2, 2, 2, 1), bias (-1, -1, -1, 0). Strengths and flips
are per-axis and live in the same scale / bias: a +Y-down (DirectX / Unreal) map is
`normal_map.scale_xy(ref, normal_map.Y_DOWN)` (UsdUVTexture scale (2, -2, 2, 1), bias
(-1, 1, -1, 0)), a strength k is `normal_map.unpack((k, k))`. Every importer derives them from the
authored data: UsdUVTexture scale / bias as authored, MaterialX `normalmap` scale (float or
vector2), glTF `normalTexture.scale`, the MDL families' strengths. A strength or offset compiles a
scaled decode in (`Features.scaled_normal_map`: specular AA reads the texture's own filtered
length, and the texel map is unclamped, so a strength above 1 decodes beyond [-1, 1]); flips and
the plain unpack keep their kernels. A per-world `material.tex_affine` declaration compiles the
scaled decode in. A binding whose decoded z is never positive is rejected.

**Extensions** (`rw.scene.RaywarpExt`, outside OpenPBR):

| field | meaning |
| --- | --- |
| `alpha_mode`, `alpha_cutoff` | glTF `AlphaMode`: OPAQUE, BLEND (`geometry_opacity` < 1 or an opacity texture needs it), MASK (a cut-out at the cutoff) |
| `occlusion_strength` | blend of the `Slot.OCCLUSION` map (glTF AO) |
| `mirror` | MuJoCo's mirror reflectance (`mat_reflectance`: one extra ray) |
| `flags` | `MaterialFlags`: `TWO_SIDED`, `UNLIT` (glTF KHR_materials_unlit, MaterialX `surface_unlit`: no light, base color plus emission shown display-relative, divided by the world's exposure, where other emission is physical nits), `UV_WORLD` (texture repeat per metre, MuJoCo `texuniform`), `SPECULAR_CURVE` (Unreal / OmniUe4 specular), `VERTEX_COLOR`, `COAT_GLTF`, `DIELECTRIC_SPECULAR_LEVEL` (weight changes dielectric F0 only; metal reflection and dielectric grazing reflectance stay unchanged) |

**Per world** (`rw.PerWorld(material=[...])`, then `Renderer.per_world["material.<name>"]`, an
`(nworld, nmaterial)` table written in place): the OpenPBR inputs `base_color` (the derived base
color; RGB unless `PerWorld(alpha=True)`), `base_metalness`, `specular_roughness` (under an
OpenPBR coat, the final coat-roughened value), `specular_weight`, `specular_ior`, `coat_weight`,
`coat_ior`, `coat_roughness` and `emission_luminance`, and the extensions `mirror`, `texture`,
`tex_affine`, `tex_variation`, `tex_repeat`, `alpha_cutoff`, `uv_world`; per instance `color`
and `shading` (packed material flags and coat). The derived inputs reach the device by one small kernel per
derived slot and frame (captured in CUDA graphs; `core/pipeline/derive/fields.py`): `specular_weight`,
`specular_ior` and the coat the reflectance, the coat also the coat-roughened roughness and the
instances' coat bytes, `emission_luminance` the emission. They equal a fresh upload at float32
values (the roughening within 1 ulp), and float64-authored values (0.9) within 1 ulp. The tables
start at the scene's values (MuJoCo: at the live `m.mat_specular` / `m.mat_emission`); the
emission color stays the scene's. A coat field and the `specular_roughness` / `shading` it
derives are not declared together. A material that emits nothing at upload gets its emission
color as a tint map with a declared `emission_luminance`, as a fresh upload. Caveat: on an
emissive panel made into a rect light (`lighting.emissive_lights`), a per-world emission changes
only the visible surface, not its fitted light or the GI.

### Prepared scalar displacement

`Material(displacement=rw.scene.Displacement(amount=...))` accepts a constant object-space distance
or a scalar-channel `TextureRef`. The texture binding supplies scale, bias, UV transform and
addressing. There is no implicit midpoint. `SceneBuilder` prepares the authored height into
ordinary triangle meshes shared by instances and saved in `.wpbr`; runtime material swaps
change shading, not geometry. No new shader or device material field is needed.

UsdPreviewSurface's scalar `displacement` input is read even without a material displacement
terminal (reported). Distances use object-space scene units, with instance/stage transforms
applied afterwards. Linear refinement is bounded by `displacement_max_level` (default 6) and
`displacement_max_triangles` (16 million across new variants). The same controls are available
to the host builder through `DisplacementOptions`. `rw.formats.usd.Options(displacement=False)` reports and omits it.
Height is filtered to the prepared edge spacing; original mesh IDs remain available.
When several stages are imported into a supplied builder, that builder's preparation policy
applies to all of them. Explicit conflicting displacement options fail before importing.

Limits are explicit: this is linear refinement, not subdivision limit-surface evaluation.
Displacement uses position-welded smooth normals (positions closer than a millionth of the mesh's
size weld, so seam copies that differ by rounding stay closed; zero-area triangles, such as a UV
sphere's pole fans, are dropped), then reconstructs normals on the prepared
geometry. Authored corner normals that differ from those smooth directions are reported as
an approximation; hard edges and artist-modified normals are not preserved by this path.
UV-seam height disagreements are averaged and reported; separately refined material subsets
are reported because their boundaries are not stitched. Analytic primitive displacement is
reported as unsupported. MaterialX/MDL displacement, vector displacement, residual bump,
bump-only mode and per-world geometry changes remain unimplemented. The first scalar checkpoint
passed both-backend GPU depth tests and downloaded-tile depth parity; 1024px keyframes were
inspected. See `internal/research/plan/reviews/scalar_displacement_preparation_2026_10_30.md` for exact
source, preparation measurements, images and remaining reference/motion/performance checks.

### Sources

**MuJoCo** (live: the material tables of `m` are aliased and re-read every frame). Each MuJoCo
material is an OpenPBR material (`rw.integrations.mujoco.materials.openpbr_materials`; generated from its
table):

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

A flex `material` applies the same way (its color when the flex rgba is the default grey, the
flex rgba otherwise: MuJoCo's color rule; under the `TINT` color rule of host scenes the shape's
color multiplies the material's instead, so white shows the material's color; the
material parameters, the texture layers through the flex's texture coordinates, scaled by
`texrepeat`; a flex without texture coordinates shows no texture, as in MuJoCo), except the mirror
`reflectance` and the alpha cutoff (a flex's material id is read once, at scene creation; the material tables stay live).

Adapter conventions beyond MJCF: a custom numeric `raywarp:alpha_cutoff:<material>` makes a
material alpha-tested (cut-out) instead of blended. `raywarp/integrations/mujoco/from_gltf.py` converts glTF 2.0
assets to MJCF with these fields.

**USD**: UsdPreviewSurface (with UsdUVTexture, UsdTransform2d), MaterialX surfaces and MDL
materials (Omniverse `OmniPBR`, `OmniGlass`, `OmniSurface`, `OmniUe4*`, ...: translated,
procedural parts baked into textures at source resolution and cached), each converted to
`rw.Material`; `primvars:displayColor`; the `raywarp:vertexColor` attribute for vertex colors.
What could not be translated is counted in the load report.

USD geometry and scene features: meshes (GeomSubsets of the `materialBind` family, two UV sets,
normals and vertex colors of any interpolation), the UsdGeom primitives (a Cone is tessellated),
PointInstancers, instanceable prims, every composition arc, purposes (`default` and `render` by
default), visibility, metersPerUnit and upAxis (scenes are metres, Z-up), cameras and UsdLux
lights. Read approximately, and reported: subdivision surfaces (the control cage with smooth
normals; creases dropped), n-gons (fan-triangulated), skinned meshes and blend shapes (the bind
pose), time samples (one time, `Options.time`: animation is not played back; poses come from the
simulator), light linking (dropped: every light lights everything). **Not supported** and
reported as `prim.<Type>` (the rest of the stage loads): Points, BasisCurves / NurbsCurves (hair),
NurbsPatch, TetMesh, volumes (Volume, OpenVDBAsset), GeometryLight, PortalLight and PluginLight
(`light.<Type>`). Physics data (joints, physics scenes and materials) and the UsdRender prims
(`rw.formats.usd.settings(stage)` reads those) are skipped without a report entry.

**glTF 2.0**: `rw.formats.gltf.scene(path)` / `rw.formats.gltf.load(path)` (`import raywarp.formats.gltf`; needs
`pygltflib` and the `usd` extra) convert the asset to a USD stage (`raywarp.formats.gltf.to_usd`: the
metallic-roughness and specular-glossiness PBR models with their maps, `KHR_texture_transform`,
`KHR_materials_emissive_strength`, `_ior`, `_clearcoat` factors, `_transmission` approximated as
opacity, vertex colors, `KHR_lights_punctual`, cameras, meshes used by several nodes as
instancers) and read it as USD, so a glTF asset renders as its USD conversion does everywhere.
glTF emission (`emissiveFactor` x texture x `emissiveStrength`) is in the environment's units, as in
glTF viewers and path tracers: an emission of 1 is as bright as an environment texel of 1. Under
a display-relative environment (`envmap_intensity=None`, or none) it stays display-relative (1
shows white at the exposure); under an absolute one (`Lighting(envmap=hdr, envmap_intensity=k)`)
it becomes a radiance of emission x k nits, so a raw HDR sky and the asset's emissive maps keep
their authored ratio under automatic exposure.
The conversion is kept (`cache=True`, the default; `cache=None`: a temporary conversion) by the
content of the asset and every file it references
(plus the converter's code), with the host scene's `.wpbr` next to it (`verify=` as
`rw.formats.usd.load`'s: "files" checks every file of the conversion, "source" the converted stage only);
`raywarp convert scene.gltf scene.usdc` writes the same stage for other renderers ([CLI](cli.md)).

**MaterialX.** MaterialX materials go through the official MaterialX library (the `materialx`
package): its MDL shader generator writes each material as an MDL module, which the MDL SDK then
converts like any MDL material (distilled, procedural parts baked into textures, cached). This is
how Omniverse RTX renders MaterialX, and every surface shader of the standard libraries converts
(`open_pbr_surface`, `standard_surface`, `gltf_pbr`, `surface_unlit`, ...): MaterialX scenes need
the MDL SDK (bundled with raywarp), like MDL ones. They are USD materials with a MaterialX network on
the `mtlx` or the universal render context. As USD specifies, the render contexts of
`rw.formats.usd.load(path, adapter=rw.formats.usd.Options(material_contexts=...))` are tried in order before the
universal one (where a UsdPreviewSurface is usually an approximation): `("mtlx", "mdl")` by
default (SimReady's target is OpenPBR), `("mdl", "mtlx")` to keep the MDL material, `("mtlx",)` to
never read MDL. `.mtlx` documents a stage references or sublayers are not composed: they are
reported as dropped in the load report (convert them into the stage's own MaterialX networks).

File inputs become the MDL module's texture arguments (read in their authored color space), color
constants in another color space are converted to linear Rec.709, `geompropvalue` reads of `st` /
`st1` read UV set 0 / 1. A network the library rejects is reported (`mtlx.*`) and renders as the
error material; inputs a nodedef does not have are reported `mtlx.input:<node>.<input>`; what the
distillation approximates is reported `mdl.*`.

Texture memory: USD textures are box-filtered to at most 1024 px per side at decode
(`rw.formats.usd.load(path, options=rw.SceneOptions(textures=texture_formats.TextureStorage(max_size=...)))`); the per-world
appearance tables (`PerWorld(material=[...])`) vary colors, roughness or a texture's UV offset
/ rotation and hue (`"tex_variation"`) per world without copying a texture
(`rw.randomize.jitter` / `contrast` draw such tables).

## Cameras

A `rw.CameraSpec` is one camera of a Renderer; a scene camera name (MJCF `camera`, USD
`Camera` prim path) is shorthand for `CameraSpec(mount=name)` with the scene camera's lens.

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

Lens cameras (OpenCV distortion, fisheye, ray bundles `CameraSpec(rays=...)`) are the
one documented exception to the prebuilt kernels: each lens model's render program compiles once
per kernel build on its first use (NVRTC, 1-9 min measured, then cached on disk, plus seconds to
create its module); `Renderer.warm(scene, settings)` with the lens camera does it ahead of the first
frame. Pinhole cameras never compile ([performance](performance.md#zero-compiles)).

- **OpenCV calibrations**: `CameraSpec.from_opencv(K, (width, height), distortion=(k1, k2, p1,
  p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]))` takes the 3 x 3 matrix in pixels (it converts
  OpenCV's integer pixel centres) and OpenCV's distortion vector of 4, 5, 8 or 12 entries
  (the rational model's k4-k6, the thin prism's s1-s4; not the tilted sensor's tau).
  The matrix must be finite and canonical zero-skew, with bottom row `(0, 0, 1)`;
  unrepresentable entries raise `ValueError` rather than being silently ignored.
  Distortion is applied in the kernel, per pixel (compiled in only when a camera has it).
- **Fisheye lenses**: `CameraSpec(projection="fisheye", fovy=220.0)` is an equidistant
  (f-theta) lens whose image height spans `fovy` degrees (up to 360); `CameraSpec.from_opencv_fisheye(K,
  (width, height), (k1, k2, k3, k4))` takes an OpenCV `cv::fisheye` (Kannala-Brandt) calibration.
  Fields wider than 180 degrees are exact; pixels outside the image circle or past the model's
  fold are misses (depth / seg miss values, the background color along the axis). A fisheye's
  depth is the distance along the ray, and its clip planes are distances along the ray. Fisheye,
  pinhole and orthographic cameras mix in one Renderer; the lens code is compiled in only when a
  camera needs it (a Renderer without one renders bit-identical pixels). No ray bundle on a
  fisheye. There is no thin lens (depth of field): defocus at 1 sample per pixel is noise.
- **Per-world intrinsics**: `PerWorld(camera=["fovy"])` or `["intrinsics"]`, then
  `r.per_world["camera.fovy"]` is an (nworld, ncam) table to write.
- **Per-world lens distortion**: `PerWorld(camera=["distortion"])`, then
  `r.per_world["camera.distortion"]` is an (nworld, ncam) table of 12 floats per camera (OpenCV's
  `(k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4)`, a fisheye's `(k1, k2, k3, k4)` then zeros),
  starting as the CameraSpecs'
  coefficients: lens randomization without recompiling (the declaration compiles the lens code
  in; motion vectors reproject through each frame's own coefficients).
- **Per-camera outputs**: `CameraSpec(outputs=("depth",))` makes a camera write only some of the
  Renderer's outputs, e.g. a depth-only wrist camera next to an RGB head camera: a camera without
  the shaded outputs is not shaded (it costs a geometry-only render). Cameras select whole groups:
  every shaded output (rgb, hdr, albedo, ...), depth, and seg with semantic; position, uv,
  material, a geometric normal and motion are written for every camera and must be listed.
  The outputs a camera leaves out keep their buffers' contents (zeros as allocated, or what a
  bound array holds): the buffers stay `(nworld, ncam, H, W)`, so no memory is saved.
- **Per-camera resolution**: `CameraSpec(resolution=(w, h))`; the outputs are then per camera
  (`out.cameras[c]`), and the motion output needs one size for every camera.
- **Exposure**: `CameraSpec(exposure=stops)` brightens one camera's display image (USD camera
  exposure); a manual `exposure.ev100` and the per-world `"exposure"` table set the world's.
- **Orthographic**: `CameraSpec(projection="orthographic", fovy=view_height_in_m)`.
- **Clip planes**: `znear` / `zfar` (default: the scene's, MuJoCo `vis.map.znear / zfar x extent`,
  the USD camera's `clippingRange`) clip the camera rays, planar (along the ray for ray bundles):
  surfaces nearer than `znear` are not seen in RGB, depth or segmentation (a camera inside or
  touching an object sees past it), and depth still counts from the eye. Shadow, reflection and GI
  rays are not clipped. MuJoCo Warp's ray caster has no near clip (MuJoCo's native renderer has).

**Ray bundles** (measured lens models, LiDAR-like patterns): give each pixel its own
camera-frame ray; depth is then the distance along the ray. Prefer `projection="fisheye"` for
fisheye lenses: bundles are all-or-none per Renderer, one resolution, and have no motion vectors.

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
OpenCV, fisheye) produce it directly. Every camera of a Renderer has rays or none.

## Segmentation

`seg` holds the instance's segment id (MuJoCo: the geom id; -1 where nothing is hit);
`semantic` (alone or with `seg`) holds class ids from name patterns, written at runtime:

```python
settings = rw.RenderSettings(outputs=rw.Outputs.of("rgb", "semantic"))
r = rw.Renderer(scene, nworld=2, cameras=["front"], resolution=(64, 48), settings=settings)
labels = r.set_semantics({"ground": ["floor"], "object": ["ball", "crate"]})
assert labels == ("background", "unlabelled", "ground", "object")  # class ids 0, 1, 2, 3
semantic = r.render(d).semantic.numpy()
```

USD `UsdSemantics` labels are kept in the host scene (`Instances.label`).

## The load report

Every host-scene load counts what it read, dropped and approximated, so coverage is measurable
on real assets: `host.report` (a `rw.scene.LoadReport`) with `counts`, `dropped`, `approximated`
(attribute name -> occurrences), `examples` (the first prim path of each) and `timings`.
`rw.upload` adds its own `gpu.*` entries (textures resized, materials approximated, emissive
panels turned into lights).

```python
report = host.report
print(report)  # counts, dropped / approximated attributes with an example path, timings
assert report.counts["images"] == 1 and not report.dropped  # a USD load also counts prims: usd.Mesh, ...
```

Read it first when something is missing: an unsupported shader input, a texture that failed to
decode, a light type with no counterpart. The MuJoCo adapter maps everything it can by
construction (the tables above) and warns about the rest.
