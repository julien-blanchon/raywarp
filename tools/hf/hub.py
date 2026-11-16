"""The Hugging Face side: the `hf` CLI (jobs, sandboxes, uploads, listings) and the bucket HTTP API
(range reads, batch deletes), standard library only.

Every remote call has a timeout and raises `HubError` on failure.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

from tools.hf.config import BUCKET

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None

ENDPOINT: Final = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
# Server-side operations per batch request.
_BATCH: Final = 500


class HubError(RuntimeError):
  """A failed `hf` command or bucket request."""


def obj(value: Json) -> dict[str, Json]:
  """`value` as a JSON object (HubError otherwise)."""
  if not isinstance(value, dict):
    raise HubError(f"expected a JSON object, got {type(value).__name__}")
  return value


def items(value: Json) -> list[Json]:
  """`value` as a JSON array (HubError otherwise)."""
  if not isinstance(value, list):
    raise HubError(f"expected a JSON array, got {type(value).__name__}")
  return value


def text(value: Json, default: str = "") -> str:
  """`value` as a string (`default` for null / missing)."""
  return value if isinstance(value, str) else default


def number(value: Json) -> float:
  """`value` as a number (0 for null / missing)."""
  return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def timestamp(value: Json) -> dt.datetime | None:
  """An ISO-8601 time of the Hub (UTC), or None."""
  return dt.datetime.fromisoformat(value) if isinstance(value, str) and value else None


def hf(*args: str, timeout: float = 120, stdin: bytes | None = None) -> str:
  """Runs `hf <args>` and returns its stdout."""
  try:
    done = subprocess.run(["hf", *args], input=stdin, capture_output=True, timeout=timeout, check=False)
  except subprocess.TimeoutExpired as e:
    raise HubError(f"hf {' '.join(args[:3])}: no answer in {timeout:.0f} s") from e
  if done.returncode != 0:
    message = (done.stderr or done.stdout).decode(errors="replace").strip()[-2000:]
    raise HubError(f"hf {' '.join(args[:4])} failed ({done.returncode}): {message}")
  return done.stdout.decode(errors="replace")


def hf_json(*args: str, timeout: float = 120) -> Json:
  """`hf <args> --format json` parsed."""
  out = hf(*args, "--format", "json", timeout=timeout)
  # The CLI may print a "Hint: ..." line before the JSON document.
  start = min((i for i in (out.find("["), out.find("{")) if i >= 0), default=-1)
  return json.loads(out[start:]) if start >= 0 else None


def token() -> str:
  """The user's Hub token ($HF_TOKEN or the CLI's saved token)."""
  token = os.environ.get("HF_TOKEN")
  if token:
    return token
  saved = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
  if not saved.exists():
    raise HubError("no Hub token: run `hf auth login` or set HF_TOKEN")
  return saved.read_text().strip()


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 60) -> bytes:
  """An authenticated HTTP request (redirects followed), retried on 5xx / 429. A `Range` request
  must be answered with 206: a server ignoring it would send the whole (multi-GB) file."""
  for attempt in range(4):
    request = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {token()}", **(headers or {})})
    try:
      with urllib.request.urlopen(request, timeout=timeout) as response:
        if headers and "Range" in headers and response.status != 206:
          raise HubError(f"{url}: range request answered with HTTP {response.status}")
        return response.read()
    except urllib.error.HTTPError as e:
      if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
        raise HubError(f"{url}: HTTP {e.code} {e.read()[:500].decode(errors='replace')}") from e
    except (urllib.error.URLError, TimeoutError) as e:
      if attempt == 3:
        raise HubError(f"{url}: {e}") from e
    time.sleep(2**attempt)
  raise AssertionError("unreachable")


def _resolve_url(path: str) -> str:
  return f"{ENDPOINT}/buckets/{BUCKET}/resolve/{urllib.parse.quote(path, safe='')}"


def read_range(path: str, start: int, length: int) -> bytes:
  """`length` bytes of bucket file `path` from byte `start` (an HTTP range read: no download)."""
  data = _request(_resolve_url(path), headers={"Range": f"bytes={start}-{start + length - 1}"})
  if len(data) != length:
    raise HubError(f"{path}: range read returned {len(data)} of {length} bytes")
  return data


def read_bytes(path: str) -> bytes:
  """The whole (small) bucket file `path`."""
  return _request(_resolve_url(path))


def exists(path: str) -> bool:
  """Whether bucket file `path` exists."""
  try:
    _request(_resolve_url(path), headers={"Range": "bytes=0-0"})
  except HubError as e:
    if "HTTP 404" in str(e):
      return False
    raise
  return True


@dataclasses.dataclass(frozen=True, slots=True)
class BucketFile:
  """A file of the bucket listing."""

  path: str
  size: int
  mtime: dt.datetime
  xet_hash: str


def list_files(prefix: str, *, timeout: float = 900) -> list[BucketFile]:
  """Every file under `prefix` (recursive; [] when the prefix does not exist)."""
  try:
    found = hf_json("buckets", "list", f"{BUCKET}/{prefix}".rstrip("/"), "-R", timeout=timeout)
  except HubError as e:
    if "not found" in str(e).lower() or "404" in str(e):
      return []
    raise
  out: list[BucketFile] = []
  for entry in items(found or []):
    row = obj(entry)
    if row.get("type") == "file":
      when = timestamp(row.get("mtime")) or timestamp(row.get("uploaded_at"))
      if when is None:
        raise HubError(f"bucket listing row without a time: {row}")
      out.append(BucketFile(text(row.get("path")), int(number(row.get("size"))), when, text(row.get("xet_hash"))))
  return out


def write(remote: str, data: bytes) -> None:
  """Writes `data` to bucket path `remote` (replacing it)."""
  hf("buckets", "cp", "-", f"hf://buckets/{BUCKET}/{remote}", stdin=data, timeout=300)


def sync_down(remote: str, local: Path, *excludes: str) -> None:
  """Mirrors bucket directory `remote` into `local` (only missing / changed files)."""
  local.mkdir(parents=True, exist_ok=True)
  args = [arg for pattern in excludes for arg in ("--exclude", pattern)]
  hf("buckets", "sync", f"hf://buckets/{BUCKET}/{remote}", str(local), *args, timeout=1800)


def sync_up(local: Path, remote: str) -> None:
  """Uploads directory `local` to bucket directory `remote`."""
  hf("buckets", "sync", str(local), f"hf://buckets/{BUCKET}/{remote}", timeout=1800)


def _batch(lines: list[dict[str, Json]]) -> None:
  for i in range(0, len(lines), _BATCH):
    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines[i : i + _BATCH])
    answer = _request(
      f"{ENDPOINT}/api/buckets/{BUCKET}/batch", data=body, headers={"Content-Type": "application/x-ndjson"}, timeout=300
    )
    failed = obj(json.loads(answer or b"{}")).get("failed") if answer.strip() else None
    if failed:
      raise HubError(f"bucket batch: {len(items(failed))} operations failed, first: {items(failed)[0]}")


def delete(paths: list[str]) -> None:
  """Deletes bucket files."""
  _batch([{"type": "deleteFile", "path": path} for path in paths])
