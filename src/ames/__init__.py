"""AMES - atomistic molecular evolution simulator.

The modules in here import each other by bare name (`from seqtools import ...`)
so that `python src/ames/ames.py` keeps working. That only resolves when this
directory is on sys.path, so importing the package puts it there. Drop this
shim once the modules use relative imports.
"""

import sys
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
