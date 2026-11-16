"""IES LM-63 photometric files (1986, 1991, 1995, 2002 and 2019 formats): `parse` -> `IesData`.

A file is a header line (none in the 1986 format), `[KEYWORD] text` lines, a `TILT=` line
(`NONE`, `INCLUDE` followed by the tilt block, or a tilt file's name: `inline_tilt` copies that
file's block into the text so a stored profile is self-contained), then numbers separated by
blanks, commas or line breaks:

  lamps, lumens per lamp (-1: absolute photometry), candela multiplier, vertical angle count,
  horizontal angle count, photometric type (1 C, 2 B, 3 A), units (1 feet, 2 metres),
  luminous opening width, length, height,
  ballast factor, ballast-lamp photometric factor (2002 and newer: "future use" / the 2019 file
  generation type, not a factor), input watts,
  the vertical angles [deg], the horizontal angles [deg], then the candela values, one row of
  vertical angles per horizontal angle.

The angular conventions per photometric type, and how a profile is gridded and evaluated, are in
`scene/lighting/ies_profile.py`. Host only (numpy).
"""

from __future__ import annotations

import dataclasses
import enum
import math
import os
import re
from typing import Final

import numpy as np

from raywarp.scene.model.types import Column, Rows

FEET: Final = 0.3048  # [m]
# Header line of each format with one (the 1986 format has none).
_FORMATS: Final = {
  "IESNA:LM-63-1995": "LM-63-1995",
  "IESNA:LM-63-2002": "LM-63-2002",
  "IES:LM-63-2019": "LM-63-2019",
  "IESNA91": "LM-63-1991",
}
# Formats whose second ballast value multiplies the candela values (later ones: future use).
_BALLAST_LAMP_FACTOR: Final = ("LM-63-1986", "LM-63-1991", "LM-63-1995")
_NUMBER: Final = re.compile(r"[^\s,]+")
_KEYWORD: Final = re.compile(r"^\[([^\]]+)\]\s?(.*)$")
_TILT_PREFIX: Final = "TILT="
# Fields before the angles: 10 numbers, then the three ballast / watt numbers.
_NHEADER: Final = 13
# Angles this close [deg] are the same angle.
ANGLE_TOL: Final = 1.0e-4


class PhotometricType(enum.IntEnum):
  """LM-63 photometric type (goniometer coordinate system)."""

  C = 1  # polar axis vertical through the luminaire: architectural luminaires
  B = 2  # polar axis horizontal: floodlights
  A = 3  # polar axis vertical, angles from the horizontal photometric axis: automotive


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Tilt:
  """A TILT=INCLUDE block: candela factors of the lamp by luminaire tilt angle.

  Attributes:
    geometry: lamp-to-luminaire geometry (1 vertical lamp, 2 / 3 horizontal along the 0 / 90 degree
      plane).
    angles: (n,) tilt angles [deg], increasing.
    factors: (n,) candela multipliers.
  """

  geometry: int
  angles: Column[np.float64]
  factors: Column[np.float64]

  def factor(self, angle_deg: float) -> float:
    """The multiplier at tilt `angle_deg` (linear between the angles, clamped at the ends)."""
    return float(np.interp(np.array([angle_deg]), self.angles, self.factors)[0])


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IesData:
  """A parsed LM-63 file (see the module doc for the fields' meaning).

  Attributes:
    format: "LM-63-1986", "-1991", "-1995", "-2002" or "-2019".
    keywords: the `[KEYWORD]` lines ([MORE] lines appended to the previous keyword).
    lamps, lumens_per_lamp, multiplier: as in the file (lumens_per_lamp -1: absolute photometry).
    photometric_type: PhotometricType.
    opening: luminous opening (width, length, height) [m] (negative: round openings, LM-63-2002).
    ballast_factor, ballast_lamp_factor: the factors the candela values are multiplied by
      (ballast_lamp_factor 1 for 2002 and newer files).
    input_watts: [W].
    tilt: the TILT=INCLUDE block (None: TILT=NONE).
    vertical: (nv,) vertical angles [deg], increasing.
    horizontal: (nh,) horizontal angles [deg], increasing.
    values: (nh, nv) candela values as listed (before the multipliers).
  """

  format: str
  keywords: dict[str, str]
  lamps: int
  lumens_per_lamp: float
  multiplier: float
  photometric_type: PhotometricType
  opening: tuple[float, float, float]
  ballast_factor: float
  ballast_lamp_factor: float
  input_watts: float
  tilt: Tilt | None
  vertical: Column[np.float64]
  horizontal: Column[np.float64]
  values: Rows[np.float64, int]

  @property
  def candela(self) -> Rows[np.float64, int]:
    """(nh, nv) luminous intensity [cd]: the values x multiplier x ballast factors (before tilt)."""
    return self.values * (self.multiplier * self.ballast_factor * self.ballast_lamp_factor)


class IesError(ValueError):
  """A malformed LM-63 file."""


def _lines(text: str) -> list[str]:
  """The lines of `text` (CR LF, CR or LF line ends; only those: latin-1 text keeps a NEL byte)."""
  return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _split_header(lines: list[str]) -> tuple[str, dict[str, str], int]:
  """(format, keywords, index of the TILT= line) of a file's lines."""
  first = lines[0].strip() if lines else ""
  fmt = _FORMATS.get(first.upper().replace(" ", ""), "LM-63-1986")
  keywords: dict[str, str] = {}
  last = ""
  for i, raw in enumerate(lines[0 if fmt == "LM-63-1986" else 1 :], start=0 if fmt == "LM-63-1986" else 1):
    line = raw.strip()
    if line.upper().startswith(_TILT_PREFIX):
      return fmt, keywords, i
    m = _KEYWORD.match(line)
    if m is None:
      continue  # 1986 free text, blank lines
    key, text = m.group(1).strip().upper(), m.group(2).strip()
    if key == "MORE" and last:
      keywords[last] = f"{keywords[last]}\n{text}"
    else:
      keywords[key], last = text, key
  raise IesError("no TILT= line")


def _numbers(text: str, where: str) -> Column[np.float64]:
  """The numbers of a block of text (blanks, commas or line breaks between them)."""
  tokens = _NUMBER.findall(text)
  try:
    return np.array([float(t) for t in tokens], np.float64)
  except ValueError as e:
    raise IesError(f"{where}: {e}") from None


def _tilt(numbers: Column[np.float64]) -> tuple[Tilt, int]:
  """(the TILT=INCLUDE block at the start of `numbers`, how many numbers it takes)."""
  if len(numbers) < 2:
    raise IesError("truncated TILT=INCLUDE block")
  declared = float(numbers[1])
  if not (math.isfinite(declared) and declared >= 1.0 and declared.is_integer()):
    raise IesError(f"TILT=INCLUDE angle count must be an integer >= 1, got {declared}")
  count = int(declared)
  if len(numbers) < 2 + 2 * count:
    raise IesError(f"TILT=INCLUDE block of {count} angles is truncated")
  angles, factors = numbers[2 : 2 + count], numbers[2 + count : 2 + 2 * count]
  if not np.all(np.isfinite(numbers[: 2 + 2 * count])):
    raise IesError("TILT=INCLUDE values must be finite")
  if np.any(np.diff(angles) < 0.0) or np.any(factors < 0.0):
    raise IesError("TILT angles must increase and factors be >= 0")
  return Tilt(geometry=int(numbers[0]), angles=angles.copy(), factors=factors.copy()), 2 + 2 * count


def _angles(name: str, angles: Column[np.float64]) -> Column[np.float64]:
  """Checked angles: finite, strictly increasing, within [-360, 360] degrees."""
  if not np.all(np.isfinite(angles)) or np.any(np.abs(angles) > 360.0 + ANGLE_TOL):
    raise IesError(f"{name} angles must be finite degrees in [-360, 360], got {angles.tolist()[:8]}")
  if np.any(np.diff(angles) <= 0.0):
    raise IesError(f"{name} angles must strictly increase, got {angles.tolist()[:8]}")
  return angles


def parse(text: str) -> IesData:
  """The LM-63 file `text` (a TILT=<file> must have been inlined: `inline_tilt`).

  Raises:
    IesError: for a malformed file (missing TILT line, truncated, non-numeric or non-finite
      data, angles that do not increase, negative candela values, multipliers or ballast
      factors, an unknown photometric type).
  """
  lines = _lines(text)
  fmt, keywords, at = _split_header(lines)
  tilt_mode = lines[at].strip()[len(_TILT_PREFIX) :].strip()
  numbers = _numbers("\n".join(lines[at + 1 :]), "photometric data")
  tilt = None
  if tilt_mode.upper() == "INCLUDE":
    tilt, used = _tilt(numbers)
    numbers = numbers[used:]
  elif tilt_mode.upper() != "NONE":
    raise IesError(f"TILT={tilt_mode}: a tilt file must be inlined first (inline_tilt)")
  if len(numbers) < _NHEADER:
    raise IesError("truncated photometric header")
  h = numbers[:_NHEADER]
  if not np.all(np.isfinite(h)):
    raise IesError(f"photometric header values must be finite, got {h.tolist()}")
  nv, nh = int(h[3]), int(h[4])
  if nv < 1 or nh < 1 or h[3] != nv or h[4] != nh:
    raise IesError(f"bad angle counts {h[3]} x {h[4]}")
  if int(h[5]) not in tuple(PhotometricType) or h[5] != int(h[5]):
    raise IesError(f"unknown photometric type {h[5]}")
  if int(h[6]) not in (1, 2):
    raise IesError(f"unknown units type {h[6]} (1 feet, 2 metres)")
  ballast_lamp = float(h[11]) if fmt in _BALLAST_LAMP_FACTOR else 1.0
  if min(h[2], h[10], ballast_lamp) < 0.0:
    raise IesError(f"candela multiplier and ballast factors must be >= 0, got {h[2]}, {h[10]}, {ballast_lamp}")
  data = numbers[_NHEADER:]
  if len(data) < nv + nh + nv * nh:
    raise IesError(f"{len(data)} numbers for {nv} vertical x {nh} horizontal angles")
  values = data[nv + nh : nv + nh + nv * nh].reshape(nh, nv)
  if not np.all(np.isfinite(values)) or np.any(values < 0.0):
    raise IesError("candela values must be finite and >= 0")
  unit = FEET if int(h[6]) == 1 else 1.0
  return IesData(
    format=fmt,
    keywords=keywords,
    lamps=int(h[0]),
    lumens_per_lamp=float(h[1]),
    multiplier=float(h[2]),
    photometric_type=PhotometricType(int(h[5])),
    opening=(float(h[7]) * unit, float(h[8]) * unit, float(h[9]) * unit),
    ballast_factor=float(h[10]),
    ballast_lamp_factor=ballast_lamp,
    input_watts=float(h[12]),
    tilt=tilt,
    vertical=_angles("vertical", data[:nv].copy()),
    horizontal=_angles("horizontal", data[nv : nv + nh].copy()),
    values=values.copy(),
  )


def inline_tilt(text: str, directory: str) -> str:
  """`text` with a TILT=<file> line replaced by TILT=INCLUDE and that file's block (the file is
  looked up in `directory`, the profile's own; other files are returned unchanged).

  Raises:
    IesError: when the tilt file cannot be read.
  """
  lines = _lines(text)
  _fmt, _keywords, at = _split_header(lines)
  mode = lines[at].strip()[len(_TILT_PREFIX) :].strip()
  if mode.upper() in ("NONE", "INCLUDE"):
    return text
  path = os.path.join(directory, mode)
  try:
    with open(path, encoding="latin-1") as f:
      block = f.read()
  except OSError as e:
    raise IesError(f"TILT file {path}: {e}") from None
  return "\n".join([*lines[:at], "TILT=INCLUDE", block.strip(), *lines[at + 1 :]])


def read(path: str) -> str:
  """The text of LM-63 file `path` with its tilt file inlined (`inline_tilt`), checked (`parse`).

  Raises:
    IesError: for a file that cannot be read or parsed.
  """
  try:
    with open(path, encoding="latin-1") as f:
      text = inline_tilt(f.read(), os.path.dirname(path))
  except OSError as e:
    raise IesError(f"{path}: {e}") from None
  parse(text)
  return text
