#!/usr/bin/env python3
"""Assemble a validated, sanitized VideoScope Phase-0 evidence bundle."""


import os
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_SOURCE_ROOT = os.fspath(_PROJECT_ROOT / "backend" / "src")
sys.path[:] = [
    _BACKEND_SOURCE_ROOT,
    *(entry for entry in sys.path if entry != _BACKEND_SOURCE_ROOT),
]

from videoscope.benchmark.phase0_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
