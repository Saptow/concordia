"""Compatibility package wrapper for the local Concordia layout.

This repo keeps the upstream Concordia package under `concordia/concordia/`
while also storing project-specific modules such as `hdb_simulation/` beside
it under `concordia/`. Adding this wrapper restores the import behavior that
used to work when the upstream repo lived as a separate submodule.
"""

from __future__ import annotations

from pathlib import Path
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent
_INNER_PACKAGE_DIR = _PACKAGE_DIR / 'concordia'
_REPO_ROOT = _PACKAGE_DIR.parent

# Make the repo root importable so package code can still resolve the shared
# top-level `configs.py` module during local runs and tests.
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

# Expose the upstream Concordia core package as part of this outer package.
if _INNER_PACKAGE_DIR.is_dir():
  __path__.append(str(_INNER_PACKAGE_DIR))

