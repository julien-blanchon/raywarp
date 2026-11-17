"""B37: the published extras resolve together (uv, PyPI + NVIDIA index; needs the network)."""

import shutil
import subprocess
from importlib import metadata

import pytest
from conftest import REPO
from packaging.requirements import Requirement

ROOT = REPO


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
@pytest.mark.parametrize("extras", ["mujoco,newton", "newton,usd"])
def test_extras_resolve(tmp_path, extras):
  """B37: the `newton` extra holds Newton's `[sim]` (its MuJoCo / MuJoCo Warp pin) beside the
  `mujoco` extra."""
  requirements = tmp_path / "requirements.in"
  requirements.write_text(f"raywarp[{extras}] @ {ROOT.as_uri()}\n")
  cmd = ["uv", "pip", "compile", str(requirements), "--python", "3.13", "--no-header"]
  cmd += ["--extra-index-url", "https://pypi.nvidia.com/", "--index-strategy", "unsafe-best-match"]
  result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
  if result.returncode != 0 and "network" in result.stderr.lower():
    pytest.skip("no network")
  assert result.returncode == 0, result.stderr[-2000:]
  pins = dict(line.split("==", 1) for line in result.stdout.splitlines() if "==" in line and not line.startswith(" "))
  assert "newton" in pins and "pyglet" in pins and "mujoco-warp" in pins
  if "mujoco" in extras:
    assert "mujoco" in pins and "mujoco-warp" in pins


def locked(*selection: str) -> dict[str, str]:
  """name -> version of the lock's packages for an `uv export` selection (offline, frozen)."""
  cmd = ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", *selection]
  out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120, check=True).stdout
  return dict(line.split(" ")[0].split("==", 1) for line in out.splitlines() if "==" in line and not line.startswith(" "))


def sim_pin_mismatches(requires: list[str], installed: dict[str, str]) -> list[str]:
  """The MuJoCo packages Newton's `[sim]` extra pins (`requires`: Newton's Requires-Dist lines)
  whose `installed` version (name -> version) falls outside the pin, as "name version (pin)"."""
  found = []
  for line in requires:
    req = Requirement(line)
    marker = req.marker
    in_sim = marker is not None and marker.evaluate({"extra": "sim"}) and not marker.evaluate({"extra": ""})
    if in_sim and req.name in ("mujoco", "mujoco-warp") and not req.specifier.contains(installed[req.name], prereleases=True):
      found.append(f"{req.name} {installed[req.name]} ({req.specifier})")
  return found


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_one_lock_holds_newtons_mujoco_pins():
  """The lock has one MuJoCo / MuJoCo Warp pair, inside Newton's `[sim]` pins (Newton's
  SolverMuJoCo), whichever extras and groups a selection holds."""
  pairs = [
    locked(*selection)
    for selection in (("--group", "bench", "--extra", "newton"), ("--extra", "mujoco"), ("--all-extras", "--group", "dev"))
  ]
  assert {(p["mujoco"], p["mujoco-warp"]) for p in pairs} == {(pairs[0]["mujoco"], pairs[0]["mujoco-warp"])}
  assert pairs[0]["newton"] == metadata.version("newton")  # Newton's own Requires-Dist below
  assert sim_pin_mismatches(metadata.requires("newton") or [], pairs[0]) == []


def test_only_the_library_ships() -> None:
  """The wheel's packages are raywarp's (benchmarks/, tools/, tests/, internal/ never), the sdist
  prunes them too, the public export (`git archive`) leaves internal/ out, and the one console
  script is `raywarp`."""
  import fnmatch
  import tomllib

  cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
  find = cfg["tool"]["setuptools"]["packages"]["find"]
  tops = [p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".") and not p.name.endswith(".egg-info")]
  shipped = [
    t
    for t in tops
    if any(fnmatch.fnmatch(t, i) for i in find["include"]) and not any(fnmatch.fnmatch(t, e) for e in find["exclude"])
  ]
  assert shipped == ["raywarp"], shipped
  assert cfg["project"]["scripts"] == {"raywarp": "raywarp.cli:main"}
  manifest = (ROOT / "MANIFEST.in").read_text().splitlines()
  assert {"prune internal", "prune benchmarks", "prune tests"} <= set(manifest)
  assert "internal/ export-ignore" in (ROOT / ".gitattributes").read_text().splitlines()
  if shutil.which("git") is None or not (ROOT / ".git").exists():
    pytest.skip("not a git checkout")
  import io
  import tarfile

  listed = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
  names = tarfile.open(fileobj=io.BytesIO(listed)).getnames()
  assert "raywarp/__init__.py" in names and "benchmarks/run.py" in names
  assert not any(n.startswith("internal") for n in names)
