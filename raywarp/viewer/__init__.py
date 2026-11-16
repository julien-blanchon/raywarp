"""A web viewer for raywarp scenes: `raywarp view` on a GPU machine, then the printed URL in a
browser (over an SSH port forward from another machine).

* pick a scene: the built-in demo, scene files and directories given on the command line, or any
  `.rayw` / USD / glTF / MJCF path typed in the page;
* change any setting (every `RenderSettings` field, grouped, from the default or `rw.FAST`): the
  renderer switches between frames, on the prebuilt kernels (`performance.kernels="prebuilt"`,
  downloaded on first use);
* fly through the scene (mouse and keyboard);
* watch frame times and every kernel compile, PTX compile or OptiX module creation, live (also
  while the renderer is blocked by one): a setting that compiles is flagged;
* frames pushed over one WebSocket, paced by a credit window (the link's bandwidth, not its round
  trip, bounds the frame rate); the views and the settings one change away warmed in the
  background after a scene opens.

Needs the `viewer` extra (Pillow, for the JPEG frames); the server is the standard library's.
Scene formats need their adapters' extras (`usd`, `mujoco`). The modules: `server` (HTTP),
`stream` (the frame stream), `websocket` (its protocol), `session` (scene, settings, renderer),
`warm` (the background warm), `scenes`, `camera`, `controls` (the settings schema), `display`
(output views), `watch` (compile events).
"""
