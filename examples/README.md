# Examples

Small, complete scripts; each runs on a GPU (Warp's default device) or with `--device cpu`, and
`tests/tooling/test_examples.py` runs them at a tiny size.

| script | shows | needs |
| --- | --- | --- |
| [`mujoco_render.py`](mujoco_render.py) | a MuJoCo Warp simulation, 8 worlds x 2 cameras of RGB, depth, segmentation | `raywarp[mujoco]` |
| [`outputs.py`](outputs.py) | output specs: channels-first float16 color, radial depth, normals, albedo | `raywarp[mujoco]` |
| [`domain_randomisation.py`](domain_randomisation.py) | per-world colours, roughness, lights, field of view, distractors | `raywarp[mujoco]` |
| [`usd_scene.py`](usd_scene.py) | a USD stage, read once into a cached `.rayw` | `raywarp[usd]` |
| [`newton_sensor.py`](newton_sensor.py) | Newton's `SensorCamera`, PBR shaded | `raywarp[newton]` |

The guides: [quickstart](../docs/quickstart.md), [domain randomisation](../docs/domain_randomisation.md),
[robotics recipes](../docs/robotics.md).
