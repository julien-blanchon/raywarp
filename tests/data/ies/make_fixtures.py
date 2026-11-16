"""Writes the synthetic LM-63 fixtures of `tests/test_ies*.py` (run from the repo root:
`python tests/data/ies/make_fixtures.py`). Synthetic on purpose: public manufacturer files carry
redistribution terms. Each one exercises a format, a photometric type and a symmetry:

* downlight_1995.ies: LM-63-1995, type C, axially symmetric (1D), 0-90 deg, a 30 deg beam with a
  batwing shoulder; 1000 lm rated, multiplier 1.
* wallwash_2002.ies: LM-63-2002, type C, quadrant symmetric (0-90 horizontal), 0-180 vertical.
* asym_2002.ies: LM-63-2002, type C, full 0-360 horizontal (a lobe toward +X), absolute
  photometry (lumens -1), multiplier 2, ballast factor 0.5.
* bilateral_2019.ies: LM-63-2019 header, type C, 90-270 horizontal (bilateral about that plane).
* flood_b_1995.ies: type B, horizontal 0-90 (lateral symmetry), vertical -90-90.
* auto_a_2002.ies: type A, horizontal -90-90, vertical 0-90 (symmetric about the horizon).
* tilt_1986.ies: 1986 format (no header line), TILT=INCLUDE, feet units, uneven vertical steps.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def _steps(start: float, stop: float, step: float) -> np.ndarray:
  """`start` to `stop` (both included) by `step`."""
  return np.linspace(start, stop, round((stop - start) / step) + 1)


def _rows(values: np.ndarray, per_line: int = 8) -> str:
  """Numbers `per_line` to a line."""
  flat = [f"{v:.4f}" for v in np.ravel(values)]
  return "\n".join(" ".join(flat[i : i + per_line]) for i in range(0, len(flat), per_line))


def _file(
  header: str, keywords: dict[str, str], tilt: str, fields: list[float], v: np.ndarray, h: np.ndarray, cd: np.ndarray
) -> str:
  """An LM-63 text."""
  lines = [header] if header else []
  lines += [f"[{k}] {t}" for k, t in keywords.items()]
  lines.append(tilt)
  lines.append(" ".join(f"{x:g}" for x in fields[:10]))
  lines.append(" ".join(f"{x:g}" for x in fields[10:]))
  lines += [_rows(v, 13), _rows(h, 13), _rows(cd)]
  return "\n".join(lines) + "\n"


def downlight() -> str:
  """Type C, axial, 0-90 by 2.5 deg."""
  v = _steps(0.0, 90.0, 2.5)
  cd = 800.0 * np.exp(-((v / 25.0) ** 2)) + 250.0 * np.exp(-(((v - 40.0) / 8.0) ** 2))
  cd[-1] = 0.0
  fields = [1, 1000, 1, len(v), 1, 1, 2, 0.1, 0.1, 0.0, 1.0, 1.0, 12.0]
  return _file(
    "IESNA:LM-63-1995", {"TEST": "downlight", "MANUFAC": "synthetic"}, "TILT=NONE", fields, v, np.array([0.0]), cd[None]
  )


def wallwash() -> str:
  """Type C, quadrant, horizontal 0-90 by 22.5, vertical 0-180 by 5."""
  v = _steps(0.0, 180.0, 5.0)
  h = _steps(0.0, 90.0, 22.5)
  cd = np.array(
    [[500.0 * max(math.cos(math.radians(t)), 0.0) * (1.0 + 0.8 * math.cos(math.radians(p)) ** 2) for t in v] for p in h]
  )
  fields = [1, 2000, 1, len(v), len(h), 1, 2, 0.6, 0.2, 0.1, 1.0, 1.0, 20.0]
  return _file("IESNA:LM-63-2002", {"TEST": "wallwash"}, "TILT=NONE", fields, v, h, cd)


def asymmetric() -> str:
  """Type C, full 0-360 by 30, a lobe toward +X (phi 0), absolute photometry."""
  v = _steps(0.0, 180.0, 10.0)
  h = _steps(0.0, 360.0, 30.0)
  cd = np.array([[100.0 * math.sin(math.radians(t)) ** 2 * (1.2 + math.cos(math.radians(p))) + 5.0 for t in v] for p in h])
  fields = [1, -1, 2.0, len(v), len(h), 1, 2, 0.3, 0.3, 0.0, 0.5, 1.0, 9.0]
  return _file("IESNA:LM-63-2002", {"TEST": "asym"}, "TILT=NONE", fields, v, h, cd)


def bilateral() -> str:
  """Type C, horizontal 90-270 by 45 (bilateral about the 90-270 plane)."""
  v = _steps(0.0, 90.0, 15.0)
  h = _steps(90.0, 270.0, 45.0)
  cd = np.array([[300.0 * math.cos(math.radians(t)) * (1.5 + math.sin(math.radians(p))) for t in v] for p in h])
  fields = [1, 1500, 1, len(v), len(h), 1, 2, 0.5, 0.5, 0.0, 1.0, 1.00001, 15.0]
  return _file("IES:LM-63-2019", {"TEST": "bilateral"}, "TILT=NONE", fields, v, h, cd)


def flood_b() -> str:
  """Type B, horizontal 0-90 by 10 (lateral symmetry), vertical -90-90 by 10."""
  v = _steps(-90.0, 90.0, 10.0)
  h = _steps(0.0, 90.0, 10.0)
  cd = np.array([[4000.0 * math.exp(-((x / 20.0) ** 2) - ((y / 35.0) ** 2)) for y in v] for x in h])
  fields = [1, 10000, 1, len(v), len(h), 2, 2, 0.4, 0.3, 0.1, 1.0, 1.0, 100.0]
  return _file("IESNA:LM-63-1995", {"TEST": "flood"}, "TILT=NONE", fields, v, h, cd)


def auto_a() -> str:
  """Type A, horizontal -90-90 by 15, vertical 0-90 by 15 (symmetric about the horizon)."""
  v = _steps(0.0, 90.0, 15.0)
  h = _steps(-90.0, 90.0, 15.0)
  cd = np.array([[2000.0 * math.exp(-((y / 10.0) ** 2) - (((x - 15.0) / 30.0) ** 2)) for y in v] for x in h])
  fields = [1, 1200, 1, len(v), len(h), 3, 2, 0.2, 0.1, 0.1, 1.0, 1.0, 55.0]
  return _file("IESNA:LM-63-2002", {"TEST": "auto"}, "TILT=NONE", fields, v, h, cd)


def tilt_1986() -> str:
  """1986 format with TILT=INCLUDE, feet, vertical steps 0 5 10 20 35 50 70 90."""
  v = np.array([0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 90.0])
  cd = 600.0 * np.cos(np.radians(v)) ** 2
  tilt = "TILT=INCLUDE\n1\n3\n0 45 90\n1.0 0.9 0.7"
  fields = [2, 800, 1, len(v), 1, 1, 1, 1.0, 1.0, 0.0, 1.0, 0.95, 30.0]
  text = _file("", {"TEST": "tilt"}, tilt, fields, v, np.array([0.0]), cd[None])
  return "Synthetic 1986 test file\n" + text


FIXTURES = {
  "downlight_1995.ies": downlight,
  "wallwash_2002.ies": wallwash,
  "asym_2002.ies": asymmetric,
  "bilateral_2019.ies": bilateral,
  "flood_b_1995.ies": flood_b,
  "auto_a_2002.ies": auto_a,
  "tilt_1986.ies": tilt_1986,
}

if __name__ == "__main__":
  for name, make in FIXTURES.items():
    (HERE / name).write_text(make())
