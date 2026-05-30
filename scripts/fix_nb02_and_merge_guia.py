"""Remove MC/GLUE from nb02, clear stale errors, merge GUIA txt files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB02 = ROOT / "notebooks" / "02_sensitivity.ipynb"
GUIA_OLD = ROOT / "GUIA_SENSIBILIDAD.txt"
GUIA_NEW = ROOT / "GUIA_PROYECTO.txt"
GUIA_OUT = ROOT / "GUIA_PROYECTO.txt"


def fix_nb02(nb):
    nb["cells"] = [
        c
        for c in nb["cells"]
        if "Visualizacion Monte Carlo / GLUE" not in "".join(c.get("source", []))
        and "scatter_param_vs_Y_todas" not in "".join(c.get("source", []))
        and "GLUE_MAX_TRAJ" not in "".join(c.get("source", []))
        and "**Figuras MC/GLUE" not in "".join(c.get("source", []))
    ]
    for cell in nb["cells"]:
        src = cell.get("source", [])
        if cell.get("cell_type") == "code" and "texto = " in "".join(src):
            cell["outputs"] = []
            cell["execution_count"] = None
            new_src = []
            for line in src:
                if line.strip() == '".join(lineas)':
                    continue
                if line == 'texto = "\n':
                    new_src.append('texto = "\\n".join(lineas)\n')
                else:
                    new_src.append(line)
            cell["source"] = new_src
        if "fig, axes = plt.subplots(2, 2, figsize=(10, 7))" in "".join(src):
            cell["source"] = [
                l.replace(
                    "fig, axes = plt.subplots(2, 2, figsize=(10, 7))",
                    "npar = len(PARAM_NAMES)\n    fig, axes = plt.subplots(2, int(np.ceil(npar / 2)), figsize=(10, 7))",
                )
                if "fig, axes = plt.subplots(2, 2" in l
                else l
                for l in src
            ]
            # fix duplicate npar if already exists
            text = "".join(cell["source"])
            if text.count("npar = len(PARAM_NAMES)") > 1:
                lines = cell["source"]
                seen = False
                fixed = []
                for line in lines:
                    if "npar = len(PARAM_NAMES)" in line:
                        if seen:
                            continue
                        seen = True
                    fixed.append(line)
                cell["source"] = fixed


def merge_guia():
    base = GUIA_NEW.read_text(encoding="utf-8") if GUIA_NEW.exists() else ""
    extra = GUIA_OLD.read_text(encoding="utf-8") if GUIA_OLD.exists() else ""
    # Append unique sections from old guia (celda a celda, FAQ tecnico)
    appendix = """
################################################################################
# APENDICE A — DETALLE CELDA A CELDA (02_sensitivity.ipynb)
################################################################################
(Integrado desde GUIA_SENSIBILIDAD.txt)

""" + "\n".join(
        line
        for line in extra.splitlines()
        if not line.startswith("====") or "GUIA —" in line
    )
    # Extract sections 3-7 from old file more cleanly
    parts = extra.split("################################################################################")
    detail_blocks = []
    for p in parts:
        if any(
            x in p
            for x in (
                "CELDA A CELDA",
                "FUNCIONES OBJETIVO",
                "POR QUE HAY VALORES NEGATIVOS",
                "SALIDAS PRINCIPALES",
                "ORDEN DE EJECUCION",
            )
        ):
            detail_blocks.append("################################################################################" + p)

    appendix_clean = "\n".join(detail_blocks)

    troubleshooting = """
################################################################################
# APENDICE B — ERRORES FRECUENTES AL CORRER NOTEBOOK 02
################################################################################

1) SyntaxError: unterminated string literal en celda "Conclusiones"
   CAUSA: version vieja tenia texto = "\\n" partido en dos lineas.
   SOLUCION: actualizar repo; la celda debe tener una sola linea:
       texto = "\\n".join(lineas)
   Accion: Kernel -> Restart, Run All desde configuracion.

2) NameError: GLUE_MAX_TRAJ, BEHAVIORAL_Y_QUANTILE o labels
   CAUSA: celdas Monte Carlo/GLUE en Parte 2 (ya no deben estar).
   SOLUCION: usar notebook 02 actualizado; MC/GLUE solo en 03_calibration.ipynb.

3) ValueError: shapes mismatch en q0 OLS (4 parametros)
   CAUSA: q0 con solo 3 factores [1.05, 0.98, 1.02].
   SOLUCION: q0 = np.array(PARAMS_TRUE) * np.array([1.05, 0.98, 1.02, 1.02])

4) Sobol S1/ST = NaN en tabla
   CAUSA: muchas simulaciones inestables (parametros extremos) o Y constante.
   SOLUCION: revisar warnings del solver; usar series_corta_balance.csv;
   ejecutar 01 antes; con INFORME=True y SOBOL_N grande.

5) ModuleNotFoundError: SALib
   SOLUCION: pip install -r requirements.txt en .venv

6) Q_sim negativos
   El notebook aplica np.maximum(q,0) en salida; si el ajuste es malo, revisar
   parametros y estabilidad (ver APENDICE valores negativos en guia antigua).

7) correlacion_parametros.png solo con INFORME=True
   Con INFORME=False el complemento OLS no corre (es intencional para prueba rapida).

"""

    if "APENDICE A" not in base:
        out = base.rstrip() + "\n\n" + appendix_clean + "\n" + troubleshooting
    else:
        out = base + troubleshooting

    out = out.replace(
        "Fin GUIA_PROYECTO.txt",
        "Fin GUIA_PROYECTO.txt (fusion con GUIA_SENSIBILIDAD.txt — eliminar ese archivo si queda duplicado)",
    )
    GUIA_OUT.write_text(out, encoding="utf-8")
    if GUIA_OLD.exists():
        GUIA_OLD.unlink()


def main():
    nb = json.loads(NB02.read_text(encoding="utf-8"))
    fix_nb02(nb)
    NB02.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    merge_guia()
    print("Fixed nb02, merged GUIA -> GUIA_PROYECTO.txt, removed GUIA_SENSIBILIDAD.txt")


if __name__ == "__main__":
    main()
