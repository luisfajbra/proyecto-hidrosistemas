#!/usr/bin/env python
"""
Atajo: ejecuta la Parte 2.

Uso:
    python run_part2.py --bootstrap 300 --sobol-n 256

Codigo principal: scripts/run_part2.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import numpy as np
except ModuleNotFoundError:
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    print("ERROR: No esta instalado numpy en el Python que uso ahora.")
    print(f"  Python actual: {sys.executable}")
    print("  Ejecute fix_entorno.bat y luego use .venv\\Scripts\\python.exe run_part2.py")
    raise SystemExit(1) from None

# NumPy 2.x: compatibilidad para src/model.py (np.trapz)
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "run_part2.py"), run_name="__main__")
