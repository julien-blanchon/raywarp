"""`python -m raywarp.cli`: the `raywarp` command without its console script."""

import sys

from raywarp.cli import main

sys.exit(main())
