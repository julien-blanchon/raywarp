"""The kernel store: where the prebuilt bundles' files are found, downloaded and checked
(`prebuilt`).

A bundle directory `sm<arch>/<content>/` exists twice, laid out alike: on the remote (`remote`:
`RAYWARP_KERNELS_URL`, else the public storage bucket the release's pins name, else `BUCKET`) and
in the local
store (`store_root`: `RAYWARP_KERNELS`, else `<cache dir>/kernels`). Its `files.json` (`Listing`)
names every file with its sha256 and size; the remote holds each file xz-compressed
(`<path>.xz`), except those already compressed (`.xz`). A file the store lacks is downloaded
(`fetch`: in parallel, streamed, decompressed, its sha256 checked, then renamed into place: a
reader never sees a partial file); a store filled ahead is read as it is, nothing fetched.

The bucket's files are mutable; the versioning is the layout: a content key names one bundle's
bytes, the package pins its file list's sha256, and a published path is never written again
(`tools/release/kernels.py publish` refuses an existing content key or pack).

Plain HTTPS (urllib), not huggingface_hub: the files are small and content-addressed, a mirror
needs nothing but a static file server, and raywarp's core keeps its two dependencies.
Rate limits (Hugging Face: 3 000 file requests per 5 minutes per IP without a token) are honoured:
a 429 or 503 waits the time the server names, then retries. HF_TOKEN, when set, is sent to
huggingface.co (a higher limit).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import lzma
import os
import re
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Final, final

from raywarp.cache.paths import cache_dir
from raywarp.core.build.kernel_cache import policy

# The releases' public Hugging Face storage bucket of the bundles (OWNER/NAME), where a release
# publishes: its pins name it (`pinned_bucket`; an older release's pins name the bucket it published
# to). Anonymous reads of `<bucket>/resolve/<path>` redirect (302) to the Xet CDN; a missing path is
# a 404.
BUCKET: Final = "blanchon/raywarp-kernels"


def bucket_url(bucket: str) -> str:
  """The download base URL of storage bucket `bucket` (OWNER/NAME)."""
  return f"https://huggingface.co/buckets/{bucket}/resolve"


KERNELS_URL: Final = bucket_url(BUCKET)
URL_ENV: Final = "RAYWARP_KERNELS_URL"
STORE_ENV: Final = "RAYWARP_KERNELS"
# A bundle's (or pack's) file list.
FILES: Final = "files.json"
# A bundle's archive of its common Warp modules (`prebuilt.common_modules`): one request for them.
COMMON: Final = "warp/common.tar"
# The release's pins (beside this module): per architecture, its bundle's content and the sha256
# of that bundle's `files.json`.
PIN: Final = Path(__file__).with_name("kernels.json")
# Parallel downloads of one batch, the seconds a request may stall, the attempts of a file.
WORKERS: Final = 16
TIMEOUT_S: Final = 30.0
ATTEMPTS: Final = 4
# The longest wait a rate limit may ask for before the download gives up [s].
MAX_WAIT_S: Final = 120.0
CHUNK: Final = 1 << 20
_RATE_RESET: Final = re.compile(r"t=(\d+)")


class FetchError(OSError):
  """A file could not be fetched: absent from the remote, unreachable, or not what its listing
  names (damaged)."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Listing:
  """A bundle's or pack's `files.json`: path -> (sha256, size) of every file, as stored locally."""

  files: dict[str, tuple[str, int]]

  @staticmethod
  def parse(data: bytes) -> Listing:
    """The listing of `files.json`'s bytes (ValueError, KeyError, TypeError: damaged)."""
    raw = json.loads(data)["files"]
    return Listing({str(path): (str(sha), int(size)) for path, (sha, size) in raw.items()})

  @staticmethod
  def under(root: Path) -> dict[str, Path]:
    """Every file under directory `root` by its path there (its own `files.json` excepted)."""
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file() and p.name != FILES}

  @staticmethod
  def encode(files: Mapping[str, Path]) -> bytes:
    """The `files.json` bytes listing `files` (path in the bundle -> the file it holds): the one
    encoder of the store's and the release's listings."""
    listed = {rel: [sha256_of(p), p.stat().st_size] for rel, p in sorted(files.items())}
    return json.dumps({"files": listed}, indent=0, sort_keys=True).encode()

  @staticmethod
  def of(root: Path) -> bytes:
    """The `files.json` bytes of every file under directory `root` (its own excepted)."""
    return Listing.encode(Listing.under(root))


def sha256_of(path: Path) -> str:
  """A file's sha256 (hex)."""
  digest = hashlib.sha256()
  with path.open("rb") as f:
    while block := f.read(CHUNK):
      digest.update(block)
  return digest.hexdigest()


def store_root() -> Path | None:
  """The local store: RAYWARP_KERNELS, else `<cache dir>/kernels` (None with caching off)."""
  if own := os.environ.get(STORE_ENV):
    return Path(own).expanduser()
  user = cache_dir()
  return None if user is None else user / "kernels"


def remote() -> str:
  """The remote's base URL (no trailing slash): URL_ENV, else the pinned bucket's, else
  `BUCKET`'s."""
  return (os.environ.get(URL_ENV) or bucket_url(pinned_bucket() or BUCKET)).rstrip("/")


def _pin_file() -> dict[str, object]:
  """The release's pins file (`PIN`), {} without one (a source tree without pins, damaged)."""
  try:
    raw = json.loads(PIN.read_text())
  except (OSError, ValueError):
    return {}
  return raw if isinstance(raw, dict) else {}


def pinned_bucket() -> str | None:
  """The storage bucket the release's bundles were published to (the pins' "bucket"), if named."""
  bucket = _pin_file().get("bucket")
  return bucket if isinstance(bucket, str) and bucket else None


def pins() -> dict[str, tuple[str, str]]:
  """The release's pinned bundles (`PIN`): SM version (e.g. "120") -> (content, sha256 of its
  files.json); empty without pins (a source tree without them, damaged)."""
  raw = _pin_file().get("bundles")
  if not isinstance(raw, dict):
    return {}
  try:
    return {str(arch): (str(entry["content"]), str(entry["files"])) for arch, entry in raw.items()}
  except (KeyError, TypeError):
    return {}


def pin(arch: int) -> tuple[str, str] | None:
  """The pinned bundle of SM version `arch`: (content, sha256 of its files.json); None when the
  release has none (a source tree without pins, an architecture without a bundle)."""
  return pins().get(str(arch))


def remote_name(path: str) -> str:
  """The remote file of store path `path` (xz-compressed unless it already is)."""
  return path if path.endswith(".xz") else f"{path}.xz"


def listing(local: Path, url: str, sha256: str | None) -> Listing:
  """The listing of the bundle or pack directory `local` / `url`: the store's `files.json`, else
  downloaded into it (checked against `sha256` when pinned).

  Raises:
    FetchError: it is absent from the remote, unreachable or damaged.
  """
  path = local / FILES
  if not path.is_file():
    _download(f"{url}/{FILES}", path, sha256, compressed=False)
  data = path.read_bytes()
  if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
    path.unlink(missing_ok=True)  # a stale or damaged copy: fetched again next time
    raise FetchError(f"{path} is not the pinned file list (sha256 {sha256})")
  try:
    return Listing.parse(data)
  except (ValueError, KeyError, TypeError) as e:
    raise FetchError(f"{path} is damaged ({e})") from e


def fetch(local: Path, url: str, files: Listing, paths: Iterable[str]) -> dict[str, str]:
  """Downloads the listed `paths` the store directory `local` lacks, `WORKERS` at a time, into it;
  {path: error} of those that failed (an unlisted path is an error)."""
  todo = [p for p in dict.fromkeys(paths) if not (local / p).is_file()]
  if not todo:
    return {}
  failed: dict[str, str] = {}

  def one(path: str) -> None:
    if path not in files.files:
      failed[path] = "not in the bundle's file list"
      return
    sha, _ = files.files[path]
    try:
      _download(f"{url}/{remote_name(path)}", local / path, sha, compressed=not path.endswith(".xz"))
    except FetchError as e:
      failed[path] = str(e)

  with concurrent.futures.ThreadPoolExecutor(min(WORKERS, len(todo)), thread_name_prefix="rw_fetch") as pool:
    list(pool.map(one, todo))
  return failed


def unpack(local: Path, url: str, files: Listing, archive: str) -> dict[str, str]:
  """Fetches the listed tar `archive` (many small files in one request) into a file of this call's
  own and writes its members the store directory `local` lacks, each checked against the listing
  (a member it does not list is an error); then deletes that file. {path: error} of what failed.

  Concurrent unpacks of one archive (processes starting at once) each read their own download:
  none deletes the archive another is about to open, and each member lands by an atomic rename."""
  if archive not in files.files:
    return {archive: "not in the bundle's file list"}
  dest = local / archive
  dest.parent.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".own", dir=dest.parent)
  os.close(fd)
  own, failed = Path(name), {}
  try:
    _download(f"{url}/{remote_name(archive)}", own, files.files[archive][0], compressed=not archive.endswith(".xz"))
    with tarfile.open(own) as tar:
      for member in tar:
        if not member.isfile() or member.name not in files.files:
          failed[member.name] = "not in the bundle's file list"
        elif not (local / member.name).is_file():
          _extract(tar, member, local / member.name, files.files[member.name][0])
  except (OSError, tarfile.TarError) as e:  # FetchError included
    failed[archive] = str(e)
  finally:
    own.unlink(missing_ok=True)
  return failed


def _extract(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path, sha256: str) -> None:
  """Archive member `member` into `dest`, checked against `sha256`, renamed into place.

  Raises:
    FetchError: its content is not what `sha256` names.
  """
  data = tar.extractfile(member)
  if data is None:
    raise FetchError(f"{member.name}: no content")
  dest.parent.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".partial", dir=dest.parent)
  partial = Path(name)
  try:
    digest = hashlib.sha256()
    with os.fdopen(fd, "wb") as out:
      while block := data.read(CHUNK):
        digest.update(block)
        out.write(block)
    if digest.hexdigest() != sha256:
      raise FetchError(f"{member.name}: sha256 {digest.hexdigest()}, expected {sha256}")
    partial.replace(dest)
  finally:
    partial.unlink(missing_ok=True)


def _download(url: str, dest: Path, sha256: str | None, *, compressed: bool) -> None:
  """`url` into `dest` (decompressed when `compressed`), checked against `sha256`, renamed into
  place.

  Raises:
    FetchError: absent, unreachable, or another file than `sha256` names.
  """
  dest.parent.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".partial", dir=dest.parent)
  partial = Path(name)
  try:
    with os.fdopen(fd, "wb") as out:
      digest = _copy(url, out, compressed=compressed)
    if sha256 is not None and digest != sha256:
      raise FetchError(f"{url}: sha256 {digest}, expected {sha256} (damaged or not this release's)")
    partial.replace(dest)
  except FetchError:
    raise
  except OSError as e:
    raise FetchError(f"{url}: {e}") from e
  except lzma.LZMAError as e:
    raise FetchError(f"{url}: damaged ({e})") from e
  finally:
    partial.unlink(missing_ok=True)


def _copy(url: str, out: IO[bytes], *, compressed: bool) -> str:
  """Streams `url` into `out` (decompressed when `compressed`); the sha256 of what was written."""
  digest = hashlib.sha256()
  decompressor = lzma.LZMADecompressor() if compressed else None
  with _open(url) as response:
    while block := response.read(CHUNK):
      data = decompressor.decompress(block) if decompressor is not None else block
      digest.update(data)
      out.write(data)
  if decompressor is not None and not decompressor.eof:
    raise FetchError(f"{url}: truncated")
  return digest.hexdigest()


def _open(url: str) -> IO[bytes]:
  """The response of a GET of `url`, retried on a network error or a rate limit (429 / 503: after
  the wait the server names).

  Raises:
    FetchError: absent (404), or still failing after `ATTEMPTS`.
  """
  if not policy.current().download:
    raise FetchError(f"{url}: downloads are off ({policy.DOWNLOAD_ENV} / configure(download=False)): the local store only")
  request = urllib.request.Request(url, headers=_headers(url))
  error: Exception | None = None
  for attempt in range(ATTEMPTS):
    try:
      return urllib.request.urlopen(request, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as e:
      if e.code not in (429, 500, 502, 503, 504):
        raise FetchError(f"{url}: HTTP {e.code}") from e
      error, wait = e, _wait(e, attempt)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
      if url.startswith("file:"):
        raise FetchError(f"{url}: {getattr(e, 'reason', e)}") from e
      error, wait = e, 2.0**attempt
    if attempt + 1 < ATTEMPTS:
      time.sleep(wait)
  raise FetchError(f"{url}: {error}")


def _wait(error: urllib.error.HTTPError, attempt: int) -> float:
  """Seconds to wait before retrying a rate-limited or failed request: the server's Retry-After or
  RateLimit reset (`t=`), else a backoff; at most `MAX_WAIT_S`."""
  named = error.headers.get("Retry-After") or ""
  reset = _RATE_RESET.search(error.headers.get("RateLimit") or "")
  seconds = float(named) if named.isdigit() else float(reset.group(1)) if reset else 2.0**attempt
  return min(seconds + 0.5, MAX_WAIT_S)


def _headers(url: str) -> dict[str, str]:
  """The request headers: a user agent, and HF_TOKEN for huggingface.co (a higher rate limit)."""
  headers = {"User-Agent": "raywarp kernel store"}
  token = os.environ.get("HF_TOKEN")
  if token and urllib.parse.urlsplit(url).hostname == "huggingface.co":
    headers["Authorization"] = f"Bearer {token}"
  return headers
