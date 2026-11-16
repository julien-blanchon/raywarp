# Feature-verification scenes

Small, fast scenes in which every renderer feature has a visible, checkable effect: to check that
each feature works and to compare with reference renderers (Omniverse RTX through the same USD
files, Mitsuba 3 and MuJoCo's renderer through the MuJoCo models).

- USD stages (`*.usda`) use only UsdGeom, UsdPreviewSurface (MaterialX `open_pbr_surface` /
  `standard_surface` / `gltf_pbr` networks in the `layered_*` stages) and UsdLux, in metres, Z up, with
  physical light units: area lights and emission in nits, distant lights in lux, `normalize` off.
  Each stage has its views as cameras `/World/View0`, `/World/View1`, ... so RTX renders the same
  images. Variants are thin layers over a base stage (`cornell_rect.usda` sublayers
  `cornell_box.usda`, `environment_rotated.usda` sublayers `environment.usda`, ...).
- MuJoCo models (`*.xml`) cover what only MJCF expresses: the geom types, height fields, the
  `reflectance` mirror, simulated motion and batched worlds. Lights are physical
  (`intensity` in lux for directional lights, candela for point and spot lights); camera `cam` is
  the first view.
- Textures (`textures/`) are tiny procedural PNGs and three RGBE environment maps (`sky.hdr`: a
  clear sky with a 90 klx sun; `studio.hdr`: three softboxes; `white.hdr`: a uniform 1000 nit furnace), stored in Omniverse's latlong
  layout.

Everything here is written by the maintainers' scene generator (development repository only; a
test there fails when the committed files are out of date): edit the scenes through it, not by
hand. The same repository holds the catalogue (views, camera paths, the features each scene
verifies and its exposure, a "quality" preset that turns every feature on), registers the scenes
in its benchmark registry and renders them as contact sheets and videos.

USD scenes upload with the catalogue's `rw.upload` options (the default: a sky map's sun is cut
out as a shadowed directional light). For Omniverse RTX, render each stage's `View<i>` cameras with RealTimePathTracing
or PathTracing; the expected differences are listed under "Known approximations".

## Feature checklist

| scene | file | verifies | expected result | settings |
|---|---|---|---|---|
| `cornell_rect` | [`cornell_rect.usda`](cornell_rect.usda) | lighting.area_lights=ltc (rect)<br>shadows.samples<br>gi.enabled (diffuse bounce, color bleeding) | Classic Cornell box: soft block shadows under the ceiling panel; with GI the white blocks' faces turned to the walls pick up red (left) and green (right), the ceiling is lit by the bounce. Without GI the ceiling and the shadowed faces are black. | `exposure.ev100=7.0` |
| `cornell_sphere` | [`cornell_sphere.usda`](cornell_sphere.usda) | lighting.area_lights=ltc (sphere)<br>soft shadows from a sphere light<br>gi.enabled | The Cornell box under a 12 cm sphere light of the rect light's power: sharper shadows than the rect version, the ceiling lit directly (the sphere emits upwards too). | `exposure.ev100=8.0` |
| `materials_hdri` | [`materials_hdri.usda`](materials_hdri.usda) | split-sum IBL<br>multiscatter energy compensation<br>Fresnel<br>specular AA<br>ior / specular workflow / clearcoat (variants row) | 5 x 5 copper spheres: metallic rows (bottom dielectric -> top metal), roughness columns (left mirror-like -> right matte); the softboxes' reflections blur left to right; no row darkens at high roughness (energy compensation). Front row: plastic, ior 1.33 (weaker highlight), ior 2.4 (stronger), gold specular workflow, clearcoat (a sharp coat highlight), emissive. | `exposure.ev100=9.5` |
| `materials_sun` | [`materials_sun.usda`](materials_sun.usda) | directional light BRDF (GGX + Lambert)<br>specular AA on small highlights<br>shadows | The same grid under one 30 klx distant light on black: a single highlight per sphere, a tiny bright dot at roughness 0 (stable under motion with specular AA), wide and dim at roughness 1; metals show no diffuse term. | `exposure.ev100=13.0`, `environment.enabled=False` |
| `textures` | [`textures.usda`](textures.usda) | textures (base color, normal, ORM, emissive, opacity maps)<br>texture repeat (UsdTransform2d)<br>sRGB vs raw color space<br>ray-cone mip LOD (checker to the horizon)<br>alpha cutout (opacityThreshold) | Top row: UV grid (yellow corner top-left, red to the right, green up), bumps lit from the upper left by the normal map, ORM (dark grooves, roughness steps left to right, metal lower half), emissive stripes. Bottom: a leaf cutout (the 0.3-alpha bands above and below the leaf must vanish), the grid repeated 3 x 3, the grey ramp as sRGB (perceptually even steps) and as raw (brighter). View 1: the checker floor fades to uniform grey towards the horizon without moire. | `exposure.ev100=14.0` |
| `lights_area` | [`lights_area.usda`](lights_area.usda) | lighting.area_lights=ltc: rect, disk, sphere, cylinder<br>shadows.samples (soft)<br>UsdLux normalize off, radiance units | Four cells, each light 540 cd straight down: the floor under each light is equally lit; the glossy floor shows each emitter's shape in its highlight (a wide rectangle, an ellipse, a round blob, a line); shadows soften with the emitter size; the sphere and the tube also light the ceiling. | `exposure.ev100=7.5` |
| `lights_punctual` | [`lights_punctual.usda`](lights_punctual.usda) | point light (treatAsPoint)<br>spot cone angle / softness (ShapingAPI)<br>color temperature (blackbody)<br>shadows | Six cells, 120 cd each: a point light (hard shadow); a 25 degree spot with a hard edge; the same spot with a soft edge; 2700 K (orange), 4000 K (warm white) and 6500 K (neutral / slightly blue) lights on white walls. | `exposure.ev100=6.0` |
| `many_lights` | [`many_lights.usda`](many_lights.usda) | lighting.light_grid (culling)<br>shadows.max_casters (budget remainder)<br>256 lights | A hall under 256 warm and cool ceiling lights: overlapping soft multi-colored shadows around the pillars. The light grid image matches full shading; the 8-caster budget keeps the shadows' layout. | `exposure.ev100=6.5`, `lighting.light_grid=True`, `shadows.max_casters=8`, `gi.spacing=0.75` |
| `shadows` | [`shadows.usda`](shadows.usda) | shadows (sun, 0.53 degree)<br>contact shadows<br>thin geometry<br>no acne / peter-panning at a 15 degree sun | Long shadows to the left: the box's and the ball's start at the contact (no gap), the five poles (2 mm - 3 cm) each cast a continuous line, the 2 mm sheet hovering 1 cm up casts a shadow starting right under its edge, the lit floor has no acne stripes. | `exposure.ev100=14.0` |
| `shadows_area` | [`shadows_area.usda`](shadows_area.usda) | lighting.area_lights=ltc<br>shadows.samples (fixed per frame / stochastic over frames)<br>shadows.samples | Under a 1 m panel: the pole's shadow is sharp at its foot and widens into a soft penumbra; the slab 0.6 m up casts a very soft shadow; the ball has a dark contact core. | `exposure.ev100=7.5` |
| `sun_disc` | [`sun_disc.usda`](sun_disc.usda) | sun disc: DistantLight angle (0.53 degrees) -> soft sun shadows<br>shadows.samples | The pergola's slat stripes on the floor have soft edges (5 m x 0.0093 rad = 4.6 cm penumbrae, a third of the 15 cm gaps); the flagpole's shadow is sharp at its foot and fades out towards its tip (the 6 cm pole is narrower than the penumbra beyond 6.5 m); the crate's shadow is sharp. With one fixed shadow sample (no temporal accumulation) every edge is hard. | `exposure.ev100=14.5` |
| `glass` | [`glass.usda`](glass.usda) | transparency (alpha blend, transparency.max_layers)<br>shadows.translucent (tinted)<br>alpha cutout (opacityThreshold) | The glass roof's shadow is light (80 % transmittance) next to the board's dark shadow; the red, green and blue panes darken and tint the checker where they overlap, and their shadows are colored (base color x (1 - opacity)); the leaves are cut out (no faint quad outline) and cast leaf-shaped shadows. | `exposure.ev100=14.5`, `transparency.enabled=True` |
| `reflections` | [`reflections.usda`](reflections.usda) | reflections.enabled<br>reflections.max_roughness fade<br>emissive panels in reflections<br>reflections.shadows | The mirror shows the ball, the floor and the three glowing panels (red, green, blue) from the front; the dark glossy floor reflects them slightly blurred; the four metal strips blur progressively (roughness 0.05 - 0.5), the last one matching the environment term. | `exposure.ev100=9.5` |
| `environment` | [`environment.usda`](environment.usda) | environment (DomeLight latlong)<br>env_sun (sun extracted as a shadowed directional light)<br>gi sky visibility | Lit by the sky only: with env_sun (the default), sharp shadows pointing away from the sun (back-left from the camera); without it, only soft shading under the objects; the chrome ball mirrors the sky and the sun. | `exposure.ev100=14.0`; upload `env_sun` |
| `environment_rotated` | [`environment_rotated.usda`](environment_rotated.usda) | DomeLight rotation (yaw)<br>env_sun follows the dome | As environment, with the dome turned 90 degrees about +Z: the shadows turn by 90 degrees (counter-clockwise seen from above). | `exposure.ev100=14.0`; upload `env_sun` |
| `environment_tilted` | [`environment_tilted.usda`](environment_tilted.usda) | DomeLight tilt (resampled latlong) | As environment, with the dome tilted 20 degrees about +X: the horizon in the background and the chrome ball is slanted, the sun higher or lower. | `exposure.ev100=14.0`; upload `env_sun` |
| `gi_window` | [`gi_window.usda`](gi_window.usda) | gi.enabled (sky and sun through a window)<br>env_sun<br>light leaking through thin walls (closet: view 2) | Sunlight through the window draws a bright patch on the floor; with GI the room fills with warm bounce light and the red sofa tints the floor near it; view 2, inside the sealed closet behind 5 cm walls, must stay black. | `exposure.ev100=10.0`, `gi.spacing=0.4`; upload `env_sun` |
| `gi_emissive` | [`gi_emissive.usda`](gi_emissive.usda) | emissive geometry as rect lights (flat panels)<br>gi.emission (curved emitters through probes)<br>gi.enabled | Lit only by emission: the flat ceiling panel (converted to a rect light) lights the room directly; the glowing orange sphere lights its surroundings orange only through the probes (no GI: its neighbourhood stays as dark as the rest). | `exposure.ev100=7.0`, `gi.spacing=0.4` |
| `meshes` | [`meshes.usda`](meshes.usda) | vertex normals (flat / smooth / none)<br>meshes with and without UVs<br>n-gon triangulation<br>mirrored (negative scale) instances<br>surfaces.backface_culling / doubleSided | Front row: a faceted icosphere (faceVarying normals), a smooth one, one without normals (faceted), a UV-mapped sphere (grid texture), an icosphere without UVs (one uniform texel color). Back: a pentagon prism and its mirror image (both lit the same, no inside-out faces), an open half pipe whose inside is visible (double sided). | `exposure.ev100=14.0` |
| `instancing` | [`instancing.usda`](instancing.usda) | PointInstancer (1024 copies, 2 prototypes)<br>very large (2 km ground, 300 m tower) and very small (5 mm bead) scales<br>texture LOD on a huge plane | A field of 1024 rocks and bricks (random yaw and scale) with sun shadows, a tower 400 m away; view 1: a 5 mm red bead at the post's foot, shaded and shadowed without precision artefacts. | `exposure.ev100=14.0` |
| `layered_anisotropy` | [`layered_anisotropy.usda`](layered_anisotropy.usda) | specular_anisotropy (OpenPBR specular_roughness_anisotropy, anisotropic GGX)<br>standard_surface specular_rotation (ext.anisotropy_rotation)<br>primitive and mesh (UV) tangents | Under the small sphere light: the a = 0 sphere has a round highlight (isotropic), a = 0.5 an elongated one, a = 0.9 a long streak along the tangent (around the sphere's azimuth); the sphere turned 90 degrees streaks the other way (pole to pole); the cylinder's streak runs around its circumference, the disc shows a brushed radial / circular sheen, the UV-sphere mesh matches the a = 0.9 primitive sphere's look. | `exposure.ev100=6.5` |
| `layered_anisotropy_hdri` | [`layered_anisotropy_hdri.usda`](layered_anisotropy_hdri.usda) | specular_anisotropy under an environment (anisotropic IBL) | The brushed-metal scene under the full studio HDRI: the softboxes' reflections are smeared along the tangent on the anisotropic spheres (round on a = 0), the other way on the turned one. | `exposure.ev100=9.5` |
| `layered_fuzz` | [`layered_fuzz.usda`](layered_fuzz.usda) | fuzz (OpenPBR fuzz_weight / fuzz_color / fuzz_roughness: Charlie sheen) | Dark red cloth balls back-lit by a strip light: without fuzz only a dim red ball; with fuzz a bright sheen rim at the silhouettes (grazing), white, pink or blue after fuzz_color, broader and softer at fuzz roughness 0.6 than 0.3; the lying capsule has a white sheen along its edges. | `exposure.ev100=8.0` |
| `layered_coat` | [`layered_coat.usda`](layered_coat.usda) | coat_layer (OpenPBR coat_weight / coat_roughness / coat_ior / coat_color / coat_darkening)<br>coat_normal_map (geometry_coat_normal) vs the unmapped coat normal<br>glTF clearcoat (gltf_pbr) | Back row: the dimpled red ball (no coat) has a broad, bumpy highlight; with a coat the softboxes reflect sharp and undistorted on top (smooth coat over the bumpy base); the wavy coat normal bends the sharp reflections into wavy lines; the glTF clearcoat looks like the smooth coat (view 1: the back row from above the front one). Front: a blue coat darkens the red base towards purple (ior 2.0: stronger coat reflection than 1.5); coat_darkening 1 makes the base darker and more saturated than coat_darkening 0. | `exposure.ev100=9.5` |
| `layered_detail` | [`layered_detail.usda`](layered_detail.usda) | detail_normal (geometry_detail_normal, its own tiling, RNM over geometry_normal)<br>horizon_fade (normal-mapped specular at grazing light) | Under a 10 degree sun both planes and spheres show the 4 x 4 coarse bumps; the left ones (detail on) add a fine dimpled grain 8 x denser across every bump (views 1 / 2 side by side), the right ones are smooth between the bumps; no sparkling or black facets at grazing angles. | `exposure.ev100=13.0` |
| `layered_glass` | [`layered_glass.usda`](layered_glass.usda) | transmissive (OpenPBR transmission_weight: clear glass keeps its reflection)<br>geometry_opacity coverage (fades the reflection too)<br>transmission_color | Three panes before the checker wall: the clear transmissive pane shows the checker almost unchanged plus a full-strength reflection of the bright strip light across its upper part; the 50 % coverage pane shows the checker through a white veil and a strip reflection at half strength; the blue transmissive pane tints the checker blue and keeps the white reflection. | `exposure.ev100=11.5`, `transparency.enabled=True` |
| `layered_furnace` | [`layered_furnace.usda`](layered_furnace.usda) | energy conservation of the layered lobes (white furnace)<br>coat_layer, fuzz, specular_anisotropy | Five white spheres (rough dielectric, metal, coated, fuzz, anisotropic metal) in a uniform white environment with no lights: each nearly vanishes into the background; none is brighter than the environment, slight darkening (energy loss) at most, strongest at the rims. | `exposure.ev100=12.0` |
| `geoms` | [`geoms.xml`](geoms.xml) | MuJoCo geoms: sphere, capsule, ellipsoid, cylinder, box, plane, hfield, mesh<br>mesh normals (given / computed)<br>mesh texcoords<br>MuJoCo reflectance (mirror)<br>physical MJCF lights (lux / candela) | Every MuJoCo primitive in PBR materials with sun and spot shadows; the chrome box mirrors its neighbours (reflectance); the two sand icospheres (given / computed normals) look the same; the UV sphere shows the grid; the height field is a smooth wavy patch. | `exposure.ev100=13.5` |
| `motion` | [`motion.xml`](motion.xml) | motion vectors<br>camera motion | A spinning paddle (0.5 rev/s) and a sliding block over a high-contrast checker: the thin poles stay solid while the camera pans; the motion output is zero on the static floor (static camera) and follows the paddle's rotation. | `exposure.ev100=13.5` |
| `randomize` | [`randomize.xml`](randomize.xml) | PerWorld: pose, visible, material base_color / roughness, light intensity / color<br>shared scenery<br>batched worlds | World 0: the authored table top (six crates in a row, four distractors, the arm). Other worlds (`randomize`): crates scattered, some distractors hidden, recolored materials, dimmer or tinted lights, the arm posed differently; each world changes only what was randomised. | `exposure.ev100=12.5` |

## Automated checks

The development repository's feature-scene test (CPU, about 10 s with a warm Warp kernel cache)
loads every file and renders a few pixels of the scenes where a feature has a physically expected value:

| test | scene | asserts |
|---|---|---|
| `test_scene_files_are_generated` | all | the committed files are the generator's output |
| `test_catalogue_and_registration` | all | views, camera paths, settings valid; registered in `scenes.SCENES` |
| `test_usd_stages_load_cleanly` | USD | only the documented load-report entries; instance / light / dome counts |
| `test_dome_rotation_moves_the_sun` | `environment*` | the sun cut out of `sky.hdr` is at the authored direction and turns with the DomeLight |
| `test_cornell_box_color_bleeding` | `cornell_rect` | with probe GI the floor next to the red wall is redder than next to the green wall; the ceiling is lit |
| `test_glass_roof_shadow_transmittance` | `glass` | the sun's share under the glass box is 0.8² (two faces of opacity 0.2) |
| `test_colored_pane_tints_its_shadow` | `glass` | the red pane's sun shadow is red (RGB transmittance); `"clear"` passes the sun untinted |
| `test_sun_disc_softens_the_pergola_stripes` | `sun_disc` | 16 shadow samples of the 0.53 degree disc soften the slat stripes; one fixed sample keeps them hard |
| `test_mirror_reflects_the_scene` | `reflections` | the mirror shows the red panel and the orange ball where their images are |
| `test_light_grid_matches_full_shading` | `many_lights` | 256 lights: the light grid matches full shading within 2 % |
| `test_accumulation_converges_and_motion_follows_the_paddle` | `motion` | 16 accumulated frames approach a 4 x 4 spp reference; motion is non-zero on the turning paddle, zero on the floor |
| `test_per_world_randomisation_changes_only_its_worlds` | `randomize` | randomising world 1 leaves worlds 0 and 2 bit-identical |
| `test_closet_behind_thin_walls_stays_dark` | `gi_window` | the probes leak no light through 5 cm walls into the sealed closet |
| `test_alpha_cutout_drops_texels_below_the_threshold` | `textures` | the leaf's 0.3-alpha band (`opacityThreshold` 0.5) shows the sky behind it (alpha test, no blending) |
| `test_ior_sets_the_dielectric_reflectance` | `materials_sun` | UsdPreviewSurface `ior` sets the dielectric F0 = ((ior - 1) / (ior + 1))^2 |

## Known approximations (expected differences from RTX / Mitsuba)

Each one appears in the scene's load report (`rw.formats.usd.load(...).report`):

- The specular workflow is converted to metallic-roughness (constant colors; a textured
  `specularColor` is dropped) (`materials_*` variant row). The clearcoat layer reflects the
  environment (or the probes) even where `reflections` trace the base's reflection, and uses the
  base's normal map.
- Sun discs (DistantLight `angle`, an environment sun's disc) soften shadows only with several shadow
  samples or temporal accumulation (one fixed sample: the centre's hard shadow); the disc's
  highlight is a point's.
- ShapingAPI `cone:softness` s: the UsdLux falloff, 1 - smoothstep of the off-axis angle from
  (1 - s) x the cone angle out to it (B54); `shaping:focus` is dropped.
- Cylinder lights in point form (`lighting.area_lights="point"`) emit I0 sin(theta) from their axis (no
  end caps); their length is ignored up close (`lighting.area_lights="ltc"` integrates it).
- Host (USD) scenes have no `material.mirror`: mirrors need `reflections.enabled=True` (the default; off in `FAST`).
- N-gons are fan-triangulated; primitives (Cube, Sphere, ...) carry no UVs.
