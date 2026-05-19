#!/usr/bin/env python
"""
Atajo: ejecuta la Parte 1.

Uso::
    python run_part1.py --monte-carlo 100 --jobs 4

El codigo principal esta en: scripts/run_part1.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# NumPy 2.x renombro trapz -> trapezoid; model.py usa np.trapz
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "run_part1.py"), run_name="__main__")
