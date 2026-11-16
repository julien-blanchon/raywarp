"""Host scene -> GpuScene upload: copies a host `Scene` to the device as the tables the kernels
read (`gpu_scene.upload`, re-exported as `rw.upload`).

One module per part: shapes and meshes (`meshes`), material UV mappings (`uv_plans`), materials
(`derive`, `materials`, `optics`) and textures (`textures`), lights (`lights`), environments
and cameras (`environment`), and what the content uses (`capabilities`). Modules are imported by
path."""
