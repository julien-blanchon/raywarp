# Raywarp

Batched, physically based ray tracing for robot learning: thousands of simulated worlds per launch, on one GPU.

<!-- SHOWCASE: docs/assets/readme/hero.webp, see docs/assets/SHOTS.md -->
![Raywarp: many randomised worlds of one scene, rendered in one launch](docs/assets/readme/hero.webp)

```bash
pip install "raywarp[newton,usd]"    # or [mujoco], [gltf], [all]
```

**Newton, a USD stage:**

```python
import newton
import newton.examples
import warp as wp

import raywarp as rw
import raywarp.integrations.newton

robot = newton.ModelBuilder()
robot.add_usd(newton.examples.get_asset("ant.usda"))  # any USD stage
builder = newton.ModelBuilder()
builder.add_ground_plane()
builder.replicate(robot, 16)  # 16 worlds
model = builder.finalize()
scene = rw.integrations.newton.scene(model, lighting=rw.Lighting.studio())  # Newton models carry no lights
r = rw.Renderer(scene, nworld=16, cameras=[rw.CameraSpec(fovy=60)], resolution=(512, 512))
r.camera_pose.fill_(wp.transformf((0.0, -3.0, 1.2), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.35)))
rgb = r.render(model.state()).rgb  # (16, 1, 512, 512)
```

**MuJoCo Warp, an MJCF model:**

```python
import mujoco
import mujoco_warp as mjw

import raywarp as rw
import raywarp.integrations.mujoco

mjm = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
  <light pos="0 0 2" dir="0 0 -1"/><geom type="plane" size="2 2 0.1"/>
  <body pos="0 0 0.3"><freejoint/><geom type="sphere" size="0.2" rgba="0.8 0.3 0.2 1"/></body>
  <camera name="front" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
</worldbody></mujoco>""")  # or MjModel.from_xml_path("robot.xml")
m, d = mjw.put_model(mjm), mjw.make_data(mjm, nworld=64)
r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=64, cameras=["front"], resolution=(512, 512))
mjw.step(m, d)
rgb = r.render(d).rgb  # (64 worlds, 1 camera, 512, 512) RGBA, on the GPU
```

Read the [documentation](docs/README.md) for features, integrations (mjlab, Isaac Lab, glTF), supported GPUs and performance.

Apache-2.0. Citation, credits and third-party licences: [docs/about.md](docs/about.md).
