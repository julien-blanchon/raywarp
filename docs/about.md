# About: citation, licence, credits

How to cite Raywarp, its licence, and the projects and assets it builds on.

## Quick start

If Raywarp helps your research, please cite it:

```bibtex
@software{raywarp,
  author = {Blanchon, Julien},
  title = {Raywarp: batched physically based ray tracing for robot learning},
  url = {https://github.com/julien-blanchon/raywarp},
  version = {0.4.0},
  year = {2026},
}
```

## Licence

Apache-2.0 ([LICENSE](../LICENSE)). The wheel bundles the NVIDIA OptiX SDK headers and the NVIDIA MDL
SDK under their own licences.

## Acknowledgements

Raywarp builds on [NVIDIA Warp](https://github.com/NVIDIA/warp), [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix),
the [NVIDIA MDL SDK](https://github.com/NVIDIA/MDL-SDK), [MuJoCo](https://github.com/google-deepmind/mujoco)
and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [Newton](https://github.com/newton-physics/newton),
[OpenUSD](https://openusd.org) and [MaterialX](https://materialx.org). Its shading follows
[Filament](https://google.github.io/filament/Filament.html) and the
[OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/) specification.

Scenes and assets used in the documentation, its images and the benchmarks:

- New Sponza, [Intel Graphics Research Samples](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html),
  "Sponza" by Intel Corporation, CC BY 4.0.
- HDR environments and models from [Poly Haven](https://polyhaven.com), CC0.
- Robots from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache-2.0.

NVIDIA Omniverse and Isaac Sim sample scenes are used only in the maintainers' internal tests;
they are not redistributed and do not appear in the published images.

## See also

- [Changelog](../CHANGELOG.md): what changed in each release.
- [Documentation index](README.md).
