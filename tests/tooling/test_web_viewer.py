"""The web viewer (`raywarp.viewer`): settings controls, camera, catalogue, compile watch, the
frame stream (window, back-pressure, latest camera, statuses, a long round trip) and the HTTP API
driven end to end with real CPU renders."""

import io
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

import numpy as np
import pytest
import warp as wp

pytest.importorskip("PIL", reason="the viewer extra (Pillow) encodes the frames")

from conftest import recipe
from fixtures.delay_proxy import DelayProxy
from fixtures.viewer_client import Json, StreamClient
from PIL import Image

import raywarp as rw
from raywarp.viewer import display as dsp
from raywarp.viewer.camera import Camera, resolution, view_camera
from raywarp.viewer.controls import change_kind, changed, patch_settings, schema
from raywarp.viewer.scenes import DEMO, catalogue, file_entry
from raywarp.viewer.server import TOKEN_HEADER, RenderThread, ViewerServer
from raywarp.viewer.session import Options, Session
from raywarp.viewer.stream import Window
from raywarp.viewer.watch import WATCH, CompileWatch, OptixMisses

TOKEN = "test-token"
SIZE = (32, 18)

# Settings controls -------------------------------------------------------------------------------


def test_schema_covers_every_field_with_its_change_kind():
  settings = recipe("default")
  rows = {row["path"]: row for row in schema(settings)}
  assert rows["gi.enabled"]["type"] == "boolean" and rows["post.tone_mapper"]["type"] == "enum"
  assert rows["exposure.ev100"]["change"] == "runtime" and rows["gi.spacing"]["change"] == "rebake"
  assert rows["gi.enabled"]["change"] == "recompile" and rows["outputs.rgb.alpha"]["change"] == "recompile"
  for field in settings.fields_table():
    expected = "recompile" if not field.runtime else "rebake" if field.rebuild else "runtime"
    assert change_kind(field.name) == expected
    assert field.name in rows or any(str(p).startswith(field.name + ".") for p in rows)  # every field has a control
  assert all(r["group"] == str(r["path"]).split(".")[0] for r in rows.values())


def test_patch_validates_atomically():
  settings = recipe("default")
  updated = patch_settings(settings, {"gi.enabled": False, "aa.spp": 2})
  assert not updated.gi.enabled and updated.aa.spp == 2 and settings.gi.enabled
  assert changed(settings, updated).keys() == {"gi.enabled", "aa.spp"}
  for patch in [{"aa.spp": True}, {"aa.spp": -1}, {"gi.spacing": float("nan")}, {"__class__": "bad"}, {}, "x"]:
    with pytest.raises((TypeError, ValueError)):
      patch_settings(settings, patch)
  assert patch_settings(settings, {"gi.bounds": "[[0,0,0],[1,1,1]]"}).gi.bounds == ((0, 0, 0), (1, 1, 1))
  assert patch_settings(settings, {"exposure.ev100": 12}).exposure.ev100 == 12.0


# Camera, views, catalogue ------------------------------------------------------------------------


def rotate(q, v):
  """Vector v rotated by quaternion q (x, y, z, w)."""
  u, w = np.asarray(q[:3], float), float(q[3])
  return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def test_camera_pose_convention():
  for yaw, pitch, expected in [(0, 0, [0, 1, 0]), (np.pi / 2, 0, [1, 0, 0]), (0, np.pi / 4, [0, 2**-0.5, 2**-0.5])]:
    pose = Camera((1, 2, 3), yaw, pitch, 60).pose()
    np.testing.assert_allclose(pose[:3], [1, 2, 3])
    np.testing.assert_allclose(rotate(pose[3:], np.array([0, 0, -1.0])), expected, atol=1e-6)  # looks along -z
    assert rotate(pose[3:], np.array([0, 1.0, 0]))[2] > 0  # up stays up
  camera = view_camera(((0, -2, 1), (0, 0, 1), 60))
  assert camera.yaw == camera.pitch == 0 and camera.fov == 60
  back = view_camera(Camera((0, 0, 0), 0.3, -0.2, 40).view())
  assert back.yaw == pytest.approx(0.3) and back.pitch == pytest.approx(-0.2)
  for size in [(True, 24), (4000, 20), (32, -1)]:
    with pytest.raises(ValueError):
      resolution(*size)
  for bad in [{"position": [0, 0], "yaw": 0, "pitch": 0, "fov": 50}, {"position": [0, 0, 0], "yaw": 0, "pitch": 2, "fov": 50}]:
    with pytest.raises(ValueError):
      Camera.parse(bad)


def test_with_outputs_reads_exactly_the_view():
  s = dsp.with_outputs(recipe("default"), ["rgb"])
  assert s.outputs.rgb is not None and s.outputs.rgb.channels == 4 and s.outputs.hdr is None
  d = dsp.with_outputs(rw.RenderSettings(outputs=rw.Outputs.of("depth", "seg")), ["albedo"])
  assert dsp.written(d) == ["rgb", "depth", "seg", "albedo"]
  assert dsp.written(dsp.with_outputs(recipe("default"), ["ev"])) == ["rgb", "hdr"]
  with pytest.raises(ValueError):
    dsp.with_outputs(recipe("default"), ["bogus"])


def test_catalogue_files_and_directories(tmp_path):
  (tmp_path / "a.wpbr").touch()
  (tmp_path / "sub").mkdir()
  (tmp_path / "sub" / "b.usda").touch()
  (tmp_path / "sub" / "notes.txt").touch()
  entries = catalogue([], [tmp_path])
  assert entries[DEMO].path is None and entries[DEMO].format == "demo"
  names = {e.name: e for e in entries.values()}
  assert names["a.wpbr"].format == "wpbr" and names["b.usda"].format == "usd" and "notes.txt" not in names
  assert names["b.usda"].group == tmp_path.name
  with pytest.raises(ValueError, match="no scene file"):
    file_entry(tmp_path / "missing.usd")
  with pytest.raises(ValueError, match="not a scene file"):
    file_entry(tmp_path / "sub" / "notes.txt")
  with pytest.raises(ValueError, match="no scene directory"):
    catalogue([], [tmp_path / "missing"])


# Compile watch -----------------------------------------------------------------------------------


def test_watch_counts_real_compiles_only():
  wp.init()
  WATCH.install()
  value = random.random()  # a module no cache holds

  @wp.kernel(module="unique")
  def fresh(out: wp.array[float]):
    out[0] = float(value)

  out = wp.zeros(1, dtype=float, device="cpu")
  before = WATCH.total()
  wp.launch(fresh, dim=1, inputs=[out], device="cpu")
  events = WATCH.since(before)
  assert [e.kind for e in events] == ["other"]  # this test's module: not the renderer's
  assert events[0].seconds > 0 and out.numpy()[0] == pytest.approx(value, rel=1e-6)
  wp.launch(fresh, dim=1, inputs=[out], device="cpu")  # loaded: no compile
  assert WATCH.total() == before + 1


def test_watch_records_optix_module_creations():
  watch = CompileWatch()
  log = logging.getLogger("raywarp.optix")
  handler = OptixMisses(watch)
  log.addHandler(handler)
  level = log.level
  log.setLevel(logging.INFO)
  try:
    log.info("[DISKCACHE] Cache miss for key ptx-123-key")
    log.info("[DISKCACHE] Cache hit for key ptx-456-key")
  finally:
    log.removeHandler(handler)
    log.setLevel(level)
  assert [(e.kind, e.name) for e in watch.since(0)] == [("optix_module", "ptx-123-key")]
  assert watch.counts() == {"kernel": 0, "ptx": 0, "optix_module": 1, "other": 0}


# The HTTP API with real CPU renders --------------------------------------------------------------


@pytest.fixture(scope="module")
def viewer():
  wp.init()
  WATCH.install()
  session = Session(catalogue([], []), Options(device="cpu", size=SIZE, base="fast"))
  srv = ViewerServer(("127.0.0.1", 0), RenderThread(session), TOKEN)
  thread = threading.Thread(target=srv.serve_forever, daemon=True)
  thread.start()
  yield srv
  srv.stopping.set()
  srv.shutdown()
  srv.server_close()
  srv.render.close()


@pytest.fixture(scope="module")
def server(viewer):
  return f"http://127.0.0.1:{viewer.server_address[1]}"


def call(base, path, body=None, token=TOKEN):
  """(status, headers, body bytes) of one request."""
  data = None if body is None else json.dumps(body).encode()
  request = urllib.request.Request(base + path, data=data, headers={TOKEN_HEADER: token, "Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(request, timeout=900) as r:
      return r.status, r.headers, r.read()
  except urllib.error.HTTPError as e:
    return e.code, e.headers, e.read()


def frame(base, camera):
  """(info, JPEG) of one frame of `camera` through a new stream."""
  with StreamClient.connect(base, TOKEN) as s:
    s.send(camera=camera, seq=1, refresh=True)
    return s.frame(900)


def field[T](message: Json, key: str, kind: type[T]) -> T:
  """`message[key]`, which must be a `kind`."""
  value = message[key]
  assert isinstance(value, kind), (key, value)
  return value


def number(message: Json, key: str) -> float:
  """`message[key]`, which must be a JSON number."""
  value = message[key]
  assert isinstance(value, int | float), (key, value)
  return value


def jpeg_image(jpeg):
  """The decoded frame."""
  return np.asarray(Image.open(io.BytesIO(jpeg)))


def test_page_and_token(server):
  status, headers, body = call(server, "/", token="")
  assert status == 200 and b"raywarp viewer" in body and headers["Content-Type"].startswith("text/html")
  assert call(server, "/app.js", token="")[0] == 200 and call(server, "/style.css", token="")[0] == 200
  assert call(server, "/api/state", token="wrong")[0] == 403
  assert call(server, "/nope")[0] == 404 and call(server, "/api/nope")[0] == 404


def test_frames_settings_views_and_errors(server):
  status, _, body = call(server, "/api/command", {"type": "scene", "id": DEMO})
  state = json.loads(body)
  assert status == 200 and state["scene"] == DEMO and state["loaded"]["extent"] > 0
  assert not any(r["path"].startswith("outputs.") for r in state["settings"])  # the view owns them
  info, jpeg = frame(server, state["camera"])
  image = jpeg_image(jpeg)
  assert image.shape == (SIZE[1], SIZE[0], 3) and image.mean() > 5 and info["seq"] == 1
  assert field(info, "kernels", str).startswith("tier") and set(field(info, "compiles", dict)) == {
    "kernel",
    "ptx",
    "optix_module",
    "other",
  }

  # A run-time setting: applied to the live renderer, reported as changed.
  status, _, body = call(
    server, "/api/command", {"type": "settings", "patch": {"exposure.mode": "manual", "exposure.ev100": 9.0}}
  )
  assert status == 200 and json.loads(body)["changed"] == ["exposure.ev100", "exposure.mode"]
  brighter = jpeg_image(frame(server, state["camera"])[1])
  assert brighter.mean() > image.mean()

  # Refused changes leave the state as it was.
  for bad in [{"exposure.ev100": "oops"}, {"aa.spp": -1}, {"no.such": 1}]:
    status, _, body = call(server, "/api/command", {"type": "settings", "patch": bad})
    assert status == 400 and "error" in json.loads(body)
  assert json.loads(call(server, "/api/state")[2])["changed"] == ["exposure.ev100", "exposure.mode"]

  # A bad scene is an error message; the demo stays.
  status, _, body = call(server, "/api/command", {"type": "path", "path": "/nonexistent/room.usd"})
  assert status == 400 and "no scene file" in json.loads(body)["error"]
  assert json.loads(call(server, "/api/state")[2])["scene"] == DEMO
  status, _, body = call(server, "/api/command", {"type": "settings", "patch": {"outputs.depth": "{}"}})
  assert status == 400 and "view menu" in json.loads(body)["error"]

  # Views, size, camera validation, base reset.
  status, _, body = call(server, "/api/command", {"type": "view", "name": "depth"})
  assert status == 200 and json.loads(body)["view"] == "depth"
  assert frame(server, state["camera"])[0]["view"] == "depth"
  call(server, "/api/command", {"type": "view", "name": "rgb"})
  assert json.loads(call(server, "/api/command", {"type": "size", "width": 24, "height": 16})[2])["size"] == [24, 16]
  assert jpeg_image(frame(server, state["camera"])[1]).shape == (16, 24, 3)
  with pytest.raises(RuntimeError, match="frame failed"):
    frame(server, {"position": [0, 0, 0]})
  assert call(server, "/api/command", {"type": "warp"})[0] == 400
  status, _, body = call(server, "/api/command", {"type": "base", "name": "fast"})
  assert json.loads(body)["changed"] == []
  events = json.loads(call(server, "/api/events?since=0")[2])
  assert set(events["compiles"]) == {"kernel", "ptx", "optix_module", "other"} and isinstance(events["events"], list)


def test_scene_file_opens(server, tmp_path):
  from raywarp.viewer.scenes import demo

  path = tmp_path / "demo_copy.wpbr"
  demo().save(path)
  status, _, body = call(server, "/api/command", {"type": "path", "path": str(path)})
  state = json.loads(body)
  assert status == 200 and state["scene"] == str(path.resolve()) and any(s["name"] == path.name for s in state["scenes"])
  assert frame(server, state["camera"])[1]


def test_watch_owner_tells_renderer_modules_from_others():
  from types import SimpleNamespace

  from raywarp.viewer.watch import owner

  def module(name, *owners):
    kernels = {f"k{i}": SimpleNamespace(func=SimpleNamespace(__module__=o)) for i, o in enumerate(owners)}
    return SimpleNamespace(name=name, kernels=kernels)

  assert owner(module("m", "raywarp.core.kernels.megakernel")) == "kernel"
  assert owner(module("m", "warp._src.builtins")) == "kernel"  # Warp's mesh / BVH kernels the renderer launches
  assert owner(module("m", "mujoco_warp._src.forward")) == "other"
  assert owner(module("raywarp.core.pipeline.platform.cadence")) == "kernel"  # no kernels: by its name


def test_gltf_file_opens_and_renders(tmp_path, monkeypatch):
  pytest.importorskip("pygltflib", reason="the glTF adapter needs pygltflib (the gltf extra)")
  pytest.importorskip("pxr", reason="the glTF adapter reads its USD conversion (usd extra)")
  from fixtures.gltf import quad_glb

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "cache"))  # the conversion's cache
  path = tmp_path / "quad.glb"
  quad_glb(path)
  scenes = catalogue([path], [])
  entry = file_entry(path)
  assert entry.format == "gltf" and entry.id in scenes
  wp.init()
  WATCH.install()
  session = Session(scenes, Options(device="cpu", size=SIZE, base="fast", scene=entry.id))
  try:
    state = session.command({"type": "scene", "id": entry.id})
    assert state["scene"] == entry.id and "loaded quad.glb" in str(state["message"])
    image, info = session.render(None)
    assert image.shape == (SIZE[1], SIZE[0], 3) and image.dtype == np.uint8 and info["frame"] == 1
  finally:
    session.close()


def test_a_file_that_fails_to_load_is_not_listed(server, tmp_path):
  path = tmp_path / "broken.wpbr"
  path.write_bytes(b"not a scene")
  status, _, body = call(server, "/api/command", {"type": "path", "path": str(path)})
  assert status == 400 and "could not load" in json.loads(body)["error"]
  state = json.loads(call(server, "/api/state")[2])
  assert all(s["id"] != str(path.resolve()) for s in state["scenes"]) and state["scene"] != str(path.resolve())


# The frame stream --------------------------------------------------------------------------------


def opened(server):
  """The demo's state, once it is open."""
  return json.loads(call(server, "/api/command", {"type": "scene", "id": DEMO})[2])


def test_stream_needs_the_token_and_an_upgrade(server):
  for token in ["", "wrong"]:
    with pytest.raises(ConnectionError, match="403"):
      StreamClient.connect(server, token)
  assert call(server, "/api/stream")[0] == 400  # a plain GET, with the token: not a WebSocket


def test_stream_back_pressure_and_the_latest_camera(server):
  camera = opened(server)["camera"]
  with StreamClient.connect(server, TOKEN, timeout=60) as s:
    s.send(camera=camera, seq=1)
    first, _ = s.frame()
    assert first["seq"] == 1 and first["window"] == Window.START and first["inflight"] == 1
    for seq in range(2, 12):  # cameras faster than the frames: never queued
      s.send(camera={**camera, "yaw": camera["yaw"] + 0.01 * seq}, seq=seq)
      time.sleep(0.02)
    second, _ = s.frame()  # the window's second frame (a camera newer than 1), then nothing
    assert number(second, "seq") > 1 and second["inflight"] == 2
    while not s.quiet(1.0):  # statuses may come; a frame may not
      kind, *_ = s.next()
      assert kind == "status"
    s.send(ack=second["id"])  # acknowledges both
    third, _ = s.frame()
    assert third["seq"] == 11  # the newest camera, not the next one in line


def test_stream_reports_what_a_busy_render_thread_waits_for(viewer, server):
  opened(server)
  key = WATCH.begin("optix_module", "test module")  # a creation in progress (this thread's)
  try:
    with StreamClient.connect(server, TOKEN, timeout=60) as s:
      job = viewer.render.submit(lambda _: time.sleep(1.2), "scene")
      found = None
      while found is None:
        kind, message, _ = s.next()
        if kind == "status" and message["busy"]:
          found = message
      job.result()
  finally:
    WATCH.end(key)
  busy = field(found, "busy", dict)
  assert busy["what"] == "scene" and busy["seconds"] > 0.25
  running = [r for r in field(found, "running", list) if r["name"] == "test module"]
  assert running and running[0]["kind"] == "optix_module" and running[0]["estimate"] > 0 and not running[0]["background"]


def warmed(server):
  """The background warm's state once it is done (`/api/state`'s "warm"): on the CPU it loads the
  software path's Warp modules of the settings one change away, which would share the render's
  cores."""
  deadline = time.monotonic() + 300
  while (warm := json.loads(call(server, "/api/state")[2])["warm"]).get("running") and time.monotonic() < deadline:
    time.sleep(0.2)
  return warm


def test_stream_rate_is_not_bound_by_the_round_trip(server):
  """Over a link with a 200 ms round trip the stream keeps more than one frame in flight: it
  sends the next frame before the previous one's ack is back (one frame per round trip would
  wait for it). Counted, not timed: the frames arrive with no ack sent, whatever the CPU's
  render speed (a frame rate over a fixed time was flaky on a loaded machine)."""
  camera = opened(server)["camera"]
  warmed(server)
  port = int(server.rsplit(":", 1)[1])
  proxy = DelayProxy(("127.0.0.1", port), 0.1)  # 100 ms each way: 200 ms round trip
  try:
    with StreamClient.connect(f"http://127.0.0.1:{proxy.port}", TOKEN, timeout=60) as s:
      s.send(camera=camera, seq=1, live=True)
      info, _ = s.frame()
      s.send(ack=info["id"])  # the first frame may compile (this worker's renderer)
      second, _ = s.frame()
      third, _ = s.frame()  # not waiting for `second`'s ack: two frames in flight
      s.send(ack=third["id"])
      later, _ = s.frame()
      s.send(ack=later["id"])
      last, _ = s.frame()
  finally:
    proxy.close()
  assert number(info, "id") < number(second, "id") < number(third, "id") < number(later, "id")
  assert number(third, "window") >= 2
  assert number(last, "rtt_ms") / 1e3 >= 0.19  # the proxy's delay, at least


def test_window_follows_the_link():
  """Without a queue in the link the window grows while it is full; a standing queue of more
  than `HIGH` frames shrinks it."""
  w, t, frame = Window(), 0.0, 0
  for _ in range(400):  # 100 ms round trip, nothing queued, the window always full
    while w.open():
      frame += 1
      w.sent(frame, t)
    t += 0.1
    w.acked(frame, t)
  assert w.size == Window.MAX
  for _ in range(100):  # the link slows: each round trip 0.5 s more than the smallest, queued
    while w.open():
      frame += 1
      w.sent(frame, t)
    t += 0.6
    first = frame - w.inflight + 1
    for f in range(first, frame + 1):  # one by one, 10 ms apart
      w.acked(f, t + 0.01 * (f - first))
  assert w.size == Window.MIN


def test_background_warm_status(server):
  opened(server)
  warm = warmed(server)
  # The CPU has no OptiX module; its nearby warm loads the software path's Warp modules.
  assert warm["scene"] and "error" not in warm, warm


def test_watch_times_an_optix_creation_and_shows_it_running(monkeypatch):
  from raywarp.core.trace.optix.context import DeviceContext
  from raywarp.viewer import watch as watch_lib

  watch = CompileWatch()
  log = logging.getLogger("raywarp.optix")
  handler = OptixMisses(watch)
  log.addHandler(handler)
  level = log.level
  log.setLevel(logging.INFO)
  seen = []

  def create(ctx, *args):
    seen.append(watch.running())
    log.info("[DISKCACHE] Cache miss for key ptx-789-key")
    time.sleep(0.05)
    return "module"

  monkeypatch.setattr(DeviceContext, "module", create)
  try:
    watch._install_optix()
    module: Callable[..., object] = DeviceContext.module  # the watched `create`
    assert module(object()) == "module"
  finally:
    log.removeHandler(handler)
    log.setLevel(level)
  assert seen[0][0]["kind"] == "optix_module" and seen[0][0]["estimate"] == watch_lib.nearby.ESTIMATE_S["full"]
  (event,) = watch.since(0)
  assert (event.kind, event.name) == ("optix_module", "ptx-789-key") and event.seconds >= 0.05
  assert watch.running() == [] and watch.estimate("optix_module") == event.seconds
