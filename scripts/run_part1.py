#!/usr/bin/env python
"""Parte 1 — verificacion, datos sinteticos, figuras y tablas (modo batch)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NumPy 2.x: model.py usa integracion trapezoidal compatible
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "data" / "synthetic"


def _check_deps() -> None:
    for pkg in ("numpy", "scipy", "matplotlib", "pandas", "joblib"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"Falta {pkg}. Cree el entorno: conda env create -f environment.yml")
            raise SystemExit(1)


def main() -> int:
    _check_deps()
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.config import RANDOM_SEED, default_true_parameters
    from src.monte_carlo import run_monte_carlo
    from src.synthetic_data import generate_synthetic_data, save_synthetic_dataset
    from src.verify import run_verification

    parser = argparse.ArgumentParser()
    parser.add_argument("--monte-carlo", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("proyecto-hidrosistemas — Parte 1")
    print("Repo: https://github.com/luisfajbra/proyecto-hidrosistemas")
    print("=" * 60)

    print("\n[1/4] Verificacion...")
    report = run_verification()
    print(f"  RMSE Manning:     {report.steady_manning_rmse:.4f} m")
    print(f"  Error pico kin.:  {report.kinematic_peak_error_s:.1f} s")
    print(f"  Residual masa:    {report.mass_balance_residual:.4f}")
    print(f"  Estado:           {'OK' if report.passed else 'REVISAR'}")

    print("\n[2/4] Tabla de parametros verdaderos...")
    tp = default_true_parameters()
    table = pd.DataFrame(
        [
            ("n (Manning)", "s/m^(1/3)", tp.n, "Calibrable"),
            ("S0", "-", tp.s0, "Calibrable"),
            ("Q0", "m3/s", tp.q0, "Calibrable / BC aguas arriba"),
            ("A_hyd", "m3/s", tp.a_hyd, "Calibrable / hidrograma"),
            ("B_w", "m", tp.b_w, "Calibrable"),
            ("L", "m", tp.L, "Dominio"),
            ("nx", "-", tp.nx, "Malla"),
            ("nt", "-", tp.nt, "Pasos de tiempo"),
            ("dt", "s", tp.dt, "Paso de tiempo"),
            ("Semilla", "-", tp.seed, "Reproducibilidad"),
            ("sigma ruido", "m3/s", "5% * Q_max", "Datos sinteticos"),
        ],
        columns=["Parametro", "Unidad", "Valor verdadero", "Tipo"],
    )
    table_path = TABLE_DIR / "parametros_verdaderos.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"  -> {table_path}")

    print("\n[3/4] Datos sinteticos e hidrograma...")
    dataset = generate_synthetic_data(seed=args.seed)
    save_synthetic_dataset(dataset, TABLE_DIR)

    t_h = dataset.t / 3600.0
    n_sta = dataset.n_stations
    fig, axes = plt.subplots(n_sta, 1, figsize=(9, 3.8 * n_sta), sharex=True)
    if n_sta == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t_h, dataset.q_true[i], "b-", lw=2, label="Simulado (verdad)")
        ax.plot(
            t_h,
            dataset.q_obs[i],
            "o",
            color="crimson",
            ms=3,
            alpha=0.7,
            label=f"Observado (sigma={dataset.sigma:.2f} m3/s)",
        )
        ax.set_ylabel("Q (m3/s)")
        ax.set_title(f"Estacion: {dataset.station_labels[i]}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Tiempo (h)")
    fig.suptitle(
        "Datos sinteticos multi-estacion — Saint-Venant 1D (MacCormack)",
        y=1.01,
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "hidrograma_simulado_vs_observado.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  sigma = {dataset.sigma:.2f} m3/s")
    print(f"  Estaciones: {', '.join(dataset.station_labels)}")
    print(f"  -> {fig_path}")
    print(f"  -> {TABLE_DIR / 'hidrograma_sintetico.npz'}")
    print("  Vuelva a ejecutar run_part2.py tras regenerar el NPZ.")

    if args.monte_carlo > 0:
        print(f"\n[4/4] Monte Carlo n={args.monte_carlo}...")
        mc = run_monte_carlo(n_samples=args.monte_carlo, n_jobs=args.jobs)
        mc_path = TABLE_DIR / "monte_carlo_resumen.csv"
        pd.DataFrame(mc).to_csv(mc_path, index=False)
        print(f"  -> {mc_path}")
    else:
        print("\n[4/4] Monte Carlo omitido (--monte-carlo N)")

    print("\nListo. Siguiente paso: notebooks/02_sensibilidad.ipynb (Sobol).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
