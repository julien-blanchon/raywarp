"""USD adapter: reads a Usd.Stage into a `raywarp.Scene` (needs usd-core and pillow, `[usd]`).

```python
import raywarp as rw
import raywarp.formats.usd

scene = rw.formats.usd.scene("office.usd", cache="~/.cache/raywarp")  # parse once, then mmap
rw.formats.usd.bake("office.usd", "office.wpbr")  # explicit offline bake
print(scene.model.report)  # what was dropped / approximated, per phase timings
settings = rw.formats.usd.settings("office.usd")  # the stage's Omniverse render settings
```

pxr is imported on the first stage read, not by this package.
"""

from raywarp.formats.usd.api import bake as bake
from raywarp.formats.usd.api import cache_key as cache_key
from raywarp.formats.usd.api import cache_path as cache_path
from raywarp.formats.usd.api import load as load
from raywarp.formats.usd.api import read as read
from raywarp.formats.usd.api import scene as scene
from raywarp.formats.usd.binding import binding as binding
from raywarp.formats.usd.options import Dialect as Dialect
from raywarp.formats.usd.options import Options as Options
from raywarp.formats.usd.render_settings import settings as settings
