"""Static checks that each adapter module satisfies the conformance protocol it is tested
through (`fixtures.conformance.HostAdapter` / `DirectAdapter`), checked by pyrefly.

A type assertion, not a test (pytest does not collect it): CI runs
`pyrefly check raywarp tests/fixtures/adapter_protocols.py`, so a signature change that breaks an
adapter's protocol fails the type check (tests/contracts/test_conformance.py runs the same protocols
at runtime). It lives in tests/fixtures/ because it imports every adapter's scene format.
"""

from typing import Final

import mujoco
import mujoco_warp as mjw
import newton
from fixtures.conformance import DirectAdapter, HostAdapter

import raywarp.formats.usd
import raywarp.integrations.mujoco
import raywarp.integrations.newton
from raywarp.formats.usd.api import StageSource
from raywarp.integrations.newton.types import NewtonState

USD: Final[HostAdapter[StageSource, None]] = raywarp.formats.usd
NEWTON: Final[HostAdapter[newton.Model, NewtonState]] = raywarp.integrations.newton
MUJOCO: Final[DirectAdapter[mujoco.MjModel, mjw.Model, mjw.Data]] = raywarp.integrations.mujoco
