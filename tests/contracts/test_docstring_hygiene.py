"""The library's comments and docstrings state behaviour, not development history: no stale file
paths, no paths into the git-ignored results / research trees, no bug, review or lane ids, no
scene names (a constant is justified by its measurement, not by the scene it was tuned on)."""

from __future__ import annotations

import functools
import re

import pytest
from conftest import REPO

ROOT = REPO / "raywarp"

# Repository paths a comment may cite: they must exist (a `*` is a glob).
_REPO_PATH = re.compile(r"\b(?:tests|benchmarks|docs|tools|webui|third_party)/[\w./*-]*[\w*/]")
# Markdown files cited by name: they must exist next to the module, in docs/ or at the repo root.
_MARKDOWN = re.compile(r"(?<![\w/.-])[\w-]+\.md\b")
# Development notes are git-ignored or move: never cited from the library.
_NOTES = re.compile(r"\b(?:results|research)/")
# Bug / review / perf-item / lane ids: BUG-12, (B58), P13, W1-6, "lane L1", "[O F1]".
_IDS = re.compile(r"\bBUG-\d+|\b[BP]\d{1,3}\b|\bW\d+-\d+\b|\bW\d{1,3}\b|\blane L\w+|\[[A-Z]\d? (?:[A-Z]+ ?\d+|§ ?\d)")
# Tokens that look like ids but are names: Display P3, OpenEXR's B44 compression, NVENC's quoted
# presets ("P1" ... "P7"), ruff codes.
_NOT_IDS = re.compile(r'\bP3\b|\bB44\b|"P[1-7]"|noqa: [A-Z0-9, ]+')
# Scene names (R1: no per-scene justification or routing).
_SCENES = re.compile(r"\bOffice\b")


@functools.cache
def _lines() -> list[tuple[str, int, str]]:
  """(path relative to the repo, line number, text) of every line of every library module."""
  out = []
  for path in sorted(ROOT.rglob("*.py")):
    rel = path.relative_to(REPO).as_posix()
    out.extend((rel, i, line) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1))
  return out


def _exists(token: str) -> bool:
  """Whether a repository path (glob) exists."""
  token = token.rstrip(".")
  return any(REPO.glob(token)) if "*" in token else (REPO / token).exists()


def _markdown_exists(module: str, name: str) -> bool:
  """Whether a bare markdown name resolves next to `module`, in docs/ or at the repo root."""
  return any((d / name).exists() for d in ((REPO / module).parent, REPO / "docs", REPO))


def _hits(rule: str) -> list[str]:
  """`file:line: token` of every violation of `rule`."""
  hits = []
  for rel, i, line in _lines():
    if rule == "paths":
      bad = [t for t in _REPO_PATH.findall(line) if not _exists(t)]
      bad += [t for t in _MARKDOWN.findall(line) if not _markdown_exists(rel, t)]
    elif rule == "notes":
      bad = _NOTES.findall(line)
    elif rule == "ids":
      bad = _IDS.findall(_NOT_IDS.sub("", line))
    else:
      bad = _SCENES.findall(line)
    hits.extend(f"{rel}:{i}: {t}" for t in bad)
  return hits


@pytest.mark.parametrize(
  ("rule", "why"),
  [
    ("paths", "cites a repository path that does not exist"),
    ("notes", "cites internal/results/ or internal/research/ (development notes, not library documentation)"),
    ("ids", "cites a bug / review / lane id (state the behaviour instead)"),
    ("scenes", "names a benchmark scene (state the measurement's conditions instead)"),
  ],
)
def test_library_text_has_no(rule: str, why: str) -> None:
  """Every comment and docstring of `raywarp` passes the rule."""
  hits = _hits(rule)
  assert not hits, f"{len(hits)} line(s) {why}:\n" + "\n".join(hits)


def test_rules_catch_rot() -> None:
  """The patterns match the rot they target and pass the look-alike names."""
  assert _IDS.findall(_NOT_IDS.sub("", "fixed (B58); P13: stretch; BUG-14; W1-6; lane L1; [O F1]")) == [
    "B58",
    "P13",
    "BUG-14",
    "W1-6",
    "lane L1",
    "[O F1",
  ]
  assert not _IDS.findall(_NOT_IDS.sub("", "Display P3 primaries, B44 / PIZ, BC7, # noqa: B008"))
  assert _REPO_PATH.findall("see tests/unit/test_x.py.") == ["tests/unit/test_x.py"]
  assert not _exists("tests/unit/no_such_test_file.py") and _exists("tests/contracts/*.py")
  assert _MARKDOWN.findall("(canonical_design.md §3) docs/settings.md") == ["canonical_design.md"]
  assert not _markdown_exists("raywarp/core/x.py", "canonical_design.md") and _markdown_exists("raywarp/x.py", "settings.md")
  assert _NOTES.findall("internal/results/hf/x.json") and _SCENES.findall("the Office's glass")
