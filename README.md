# proyecto-hidrosistemas

Proyecto final **ICYA 4715** — Saint-Venant 1D (canal abierto, datos sintéticos).

Hoja de ruta: [`roadmap_ICYA4715.md`](roadmap_ICYA4715.md)

**Documentación:**
- [`COMO_EJECUTAR.txt`](COMO_EJECUTAR.txt) — instalación, PowerShell, qué hace cada `.bat`
- [`EXPLICACION_FUNCIONES.txt`](EXPLICACION_FUNCIONES.txt) — qué hace cada función (Parte 1 y 2)
- [`GUIA_FIGURAS_PARTE2.txt`](GUIA_FIGURAS_PARTE2.txt) — cómo interpretar cada gráfica de sensibilidad

## Instalación (una vez)

```powershell
cd "d:\descagas\proyecto-hidrosistemas"
.\fix_entorno.bat
```

En Cursor: **Ctrl+Shift+P** → *Python: Select Interpreter* → `.venv\Scripts\python.exe`

Si quedó la carpeta rota `hidrosistemas\` dentro del repo: ejecute `limpiar_proyecto.bat`.

## Ejecutar

El código real está en **`scripts/run_part1.py`** y **`scripts/run_part2.py`**.

Los `.bat` son solo atajos opcionales en Windows (ver `COMO_EJECUTAR.txt`).

```powershell
cd "d:\descagas\proyecto-hidrosistemas"
.\.venv\Scripts\Activate.ps1
python scripts\run_part1.py
python scripts\run_part2.py
```

Atajos (doble clic): `fix_entorno.bat` (1ª vez) → `ejecutar_part1.bat` → `ejecutar_part2.bat`

Parte 2 opciones: `--bootstrap 300 --sobol-n 256` o `--skip-sobol`

## Estructura (código principal)

```
proyecto-hidrosistemas/
├── src/
│   ├── model.py           # Solver MacCormack (nucleo)
│   ├── config.py          # Parametros verdaderos
│   ├── synthetic_data.py  # Datos sinteticos + ruido
│   ├── verify.py          # Verificacion Parte 1
│   ├── monte_carlo.py     # Monte Carlo (joblib)
│   ├── sensibilidad.py    # SSC + suposiciones + IC
│   └── sobol.py           # Indices Sobol (SALib)
├── scripts/
│   ├── run_part1.py
│   └── run_part2.py
├── notebooks/             # 01-04
├── data/synthetic/        # Salidas tablas
├── figures/               # Salidas figuras
├── environment.yml      # Conda (opcional)
├── requirements.txt       # pip + .venv
├── fix_entorno.bat
├── ejecutar_part1.bat
└── ejecutar_part2.bat
```

## API del modelo

```python
from src.model import saint_venant_1d

params = [0.035, 0.001, 50.0, 100.0, 50.0]  # n, S0, Q0, A_hyd, B_w
Q_out = saint_venant_1d(params)
```

## Repo

https://github.com/luisfajbra/proyecto-hidrosistemas
