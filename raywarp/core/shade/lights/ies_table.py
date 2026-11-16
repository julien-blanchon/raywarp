"""The device layout of the IES profile table (`IesTable`, a field of
`shade.lights.punctual.Lights`); filled and read by `shade/lights/ies.py`."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class IesProfile:
  """One profile instance (a grid with a light's UsdLux angle scale): `kind`
  (ies_profile.KIND_*: the TILT table's angle); `coarse` the offset in `IesTable.data` of its
  coarse profile (`ies_profile.coarse`: COARSE_KNOTS (bound, mean) pairs in cos(theta)); `tilt` /
  `ntilt` the offset and size of its TILT table (ntilt angles [deg], then ntilt factors; 0: none).
  Its relative intensity over every emission direction is strip `instance` of `IesTable.maps`."""

  kind: wp.int32
  coarse: wp.int32
  tilt: wp.int32
  ntilt: wp.int32


@wp.struct
class IesTable:
  """The profiles of a scene (without any, one placeholder entry: `ies.ies_table`): their headers
  and coarse / TILT `data`; `maps` their OCTA x OCTA octahedral maps (`shade/lights/ies.py`, the
  angle scale baked in) stacked as rows of strips of one bilinear texture, `inv_rows` 1 / its strip
  count (one fetch a light)."""

  profiles: wp.array[IesProfile]
  data: wp.array[wp.float32]
  maps: wp.Texture2D
  inv_rows: wp.float32
