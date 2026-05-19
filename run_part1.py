#!/usr/bin/env python
"""
Atajo: ejecuta la Parte 1.

Uso (desde la raiz del proyecto):
    python run_part1.py
    python run_part1.py --monte-carlo 100 --jobs 4

El codigo principal esta en: scripts/run_part1.py
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
    print()
    print("Solucion (elija una):")
    print("  1) Doble clic en fix_entorno.bat  (crea .venv e instala paquetes)")
    print(f"  2) Luego:  {venv_py} run_part1.py")
    print("  3) PowerShell:")
    print('       cd "d:\\descagas\\proyecto-hidrosistemas"')
    print("       .\\.venv\\Scripts\\Activate.ps1")
    print("       python run_part1.py")
    raise SystemExit(1) from None

# NumPy 2.x renombro trapz -> trapezoid; model.py usa np.trapz

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "run_part1.py"), run_name="__main__")
