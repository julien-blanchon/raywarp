# Realistic scenes: sources, licences, attribution

The complete scenes added for realism checks (interiors, a warehouse, exteriors). None of their
files are in this repository: `internal/tools/fetch_assets.py` (groups `showcase`, `isaac`, `hdri`) and
`internal/benchmarks/corpus_refs.py` (the Bitterli scenes) download them into `$MJPBR_ASSET_DIR`, the
bucket's `assets/`; `python -m internal.tools.hf prepare` builds the prepared `.wpbr` files. Downloads are
untrusted data: they are read with `python -I` on HF jobs and never executed.

| Registry name | Kind | Source | Licence | Attribution |
| --- | --- | --- | --- | --- |
| `moore_lane` | furnished house, interior + garden | Intel, *4004 Moore Lane* v1.2.0, [dpel.aswf.io/4004-moore-lane](https://dpel.aswf.io/4004-moore-lane/) (zip sha256 `00075649b9b8b77f…`) | ASWF Digital Assets License v1.1: education, research, software / hardware development and benchmarking; redistribution keeps the copyright notice, conditions and disclaimer, modified copies say how they differ; images in publications carry the copyright notice; no endorsement or trademark use | "4004 Moore Lane" © Intel Corporation, ASWF Digital Assets License v1.1 |
| `polyhaven_apartment` | living / dining room (our composition) | `internal/tools/polyhaven_apartment.py` from Poly Haven models, textures and the `kloofendal_48d_partly_cloudy_puresky` HDRI | CC0-1.0 ([polyhaven.com/license](https://polyhaven.com/license)) | none required; credit: Poly Haven and its artists |
| `isaac_warehouse` | warehouse | NVIDIA Omniverse sample `Samples/Showcases/2023_2_1/IsaacWarehouse` (public S3 content bucket) | NVIDIA Omniverse sample content, no licence file: local benchmarking only, never committed or redistributed (as `old_attic`) | NVIDIA |
| `rivermark` | small town, streets, exterior | NVIDIA Isaac Sim 5.1 `Isaac/Environments/Outdoor/Rivermark` | NVIDIA Isaac Sim environment supplement licence: "Limited Use Content: for your use only, without modifications"; local benchmarking only, never committed or redistributed | NVIDIA |
| `bitterli_modern_living_room` | living room | Wig42, *The Modern Living Room* ([blendswap 75692](http://www.blendswap.com/blends/view/75692)), Mitsuba version from [Bitterli's rendering resources](https://benedikt-bitterli.me/resources/) | CC-BY-3.0 | "The Modern Living Room" by Wig42, CC BY 3.0 |
| `bitterli_breakfast_room` | breakfast room | Wig42, *The Breakfast Room* ([blendswap 75431](http://www.blendswap.com/blends/view/75431)), Bitterli's Mitsuba version | CC-BY-3.0 | "The Breakfast Room" by Wig42, CC BY 3.0 |
| `bitterli_victorian_house` | house exterior | MrChimp2313, *Victorian Style House*, Bitterli's Mitsuba version | CC0-1.0 | none required |
| `intel_sponza` | New Sponza (2022), palace courtyard | Intel Graphics Research samples, [intel.com/.../graphics-research/samples.html](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html), *Main Sponza* (`getContent/830833`, zip MD5 `8c9c6c6f7d8b…`) | CC-BY-4.0 (the archive's `credits_license.txt`) | "Sponza 2022" commissioned by Frank Meinl, sponsored by Anton Kaplanyan (Intel Sample Library); cite `@misc{sponza22, Author = {Frank Meinl and Anton Kaplanyan}, Year = {2022}, Title = {Intel Sample Library}}` |
| `intel_sponza_full`, `intel_sponza_candles` | New Sponza + add-ons: curtains, ivy, cypress tree, 10k candles | the same page: *Colorful Curtains* (`726650`), *Ivy* (`726656`), *Trees* (`726662`), *Emissive Candles* (`726676`) | CC-BY-4.0 | as above, plus the Sponza Addon Package Crew: Katica Putica, Cristiano Siqueira, Timothy Heath, Justin Prazen, Sebastian Herholz, Bruce Cherniak, Anton Kaplanyan; reference photos Katica Putica, Princino.photo (Dubrovnik) |
| `intel_jungle_ruins` | jungle pyramid, exterior with dense vegetation | the same page: *Jungle Ruins* 1.0.1b (`844047`) | CC-BY-4.0 | "Jungle Ruins" by Cristiano Siqueira, artistic support Charlene Teets (Intel); cite `@misc{jungleruins, Author = {Cristiano Siqueira and Charlene Teets and Sebastian Herholz and Anton Sochenov and Anton Kaplanyan}, Year = {2024}}` |
| `intel_knight` | armoured knight (static: the animation's first frame) | the same page: *Animated Knight* (`763175`) | CC-BY-4.0 | Intel Sample Library (the knight package's credits) |

The Intel samples' stages (`intel/*.usda`) are ours: `internal/tools/intel_samples.py` composes the
authors' USD exports (sublayered / referenced as published, with a sun and sky rig, a floor under
the knight); the archives' DCC files (3ds Max, FBX, Blender, Maya, Alembic) are not extracted.
The archives' licence text says "for personal use and educational use, limited commercial use for
marketing and print purposes" above the CC-BY-4.0 terms: renders shown outside the project carry
the attribution above.

Pins: the Intel zips by size and their CDN MD5 ETag (`internal/tools/intel_samples.py`), the Moore Lane zip by sha256 (`ARCHIVE_SCENES`), Omniverse / Isaac files by their S3 MD5
ETags, Poly Haven files by the md5 its API publishes, the Bitterli zips by sha256
(`internal/benchmarks/scenes/corpus/reference_corpus.json`).
