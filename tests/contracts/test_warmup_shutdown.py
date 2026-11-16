"""An unfinished renderer warmup must complete before interpreter resource teardown."""

import os
import subprocess
import sys
from pathlib import Path


def test_process_exit_waits_for_warmup(tmp_path: Path) -> None:
  """A started compiler finishes before atexit, even if no first frame was rendered."""
  script = r"""
import atexit
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from raywarp.core.pipeline import warmup

finished = threading.Event()
started = threading.Event()
output = Path(sys.argv[1])

def load(device, mods, block_dim):
    started.set()
    time.sleep(0.2)
    finished.set()

warmup.first_frame = lambda ctx, backend: SimpleNamespace(modules=lambda: ([object()], []))
warmup.Warmup._run = staticmethod(load)
warmup._fetch = lambda device, mods, block_dim: None  # no bundle fetch for the fake modules
atexit.register(lambda: output.write_text(str(finished.is_set())))
ctx = SimpleNamespace(scene=SimpleNamespace(device="cpu"), features=SimpleNamespace(block_dim=64))
warmup.start(ctx, None)
assert started.wait(5)
"""
  root = Path(__file__).resolve().parents[2]
  result = tmp_path / "finished.txt"
  process = subprocess.run(
    [sys.executable, "-c", script, str(result)],
    cwd=root,
    env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
  )
  assert process.returncode == 0, process.stdout + process.stderr
  assert result.read_text() == "True", "Interpreter teardown ran while the compiler thread was active"
