#!/usr/bin/env python
"""
Atajo: ejecuta la Parte 2.

Uso:
    python run_part2.py --bootstrap 300 --sobol-n 256

"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# NumPy 2.x: compatibilidad para src/model.py (np.trapz)
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "run_part2.py"), run_name="__main__")
