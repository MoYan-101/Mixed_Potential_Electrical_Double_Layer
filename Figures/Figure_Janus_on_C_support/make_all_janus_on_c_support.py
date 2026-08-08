#!/usr/bin/env python3
"""Generate the canonical C(5)|Au(4)|Pd(4)|C(5) result bundle."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from janus_on_c_support.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
