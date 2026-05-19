#!/usr/bin/env python
"""Parte 2 — SSC, Sobol, suposiciones e intervalos de confianza."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "data" / "synthetic"


def _load_dataset(seed: int):
    from src.synthetic_data import generate_synthetic_data, load_synthetic_dataset

    npz = TABLE_DIR / "hidrograma_sintetico.npz"
    if npz.exists():
        print(f"  Datos cargados desde {npz}")
        dataset = load_synthetic_dataset(npz)
        if dataset.n_stations > 1:
            print(
                f"  Multi-estacion ({dataset.n_stations}): "
                f"{', '.join(dataset.station_labels)}"
            )
        return dataset
    return generate_synthetic_data(seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parte 2: sensibilidad")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Repeticiones bootstrap para IC de parametros (0=omitir)",
    )
    parser.add_argument(
        "--sobol-n",
        type=int,
        default=256,
        help="N base Saltelli (0=omitir Sobol). Total sims ~ N*(2+p)",
    )
    parser.add_argument("--sobol-jobs", type=int, default=-1)
    parser.add_argument("--skip-sobol", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from src.config import RANDOM_SEED
    from src.sensibilidad import run_sensitivity_analysis, save_results
    from src.sobol import run_sobol_analysis

    seed = args.seed or RANDOM_SEED
    print("=" * 60)
    print("Parte 2 — Sensibilidad (SSC + Sobol + IC + suposiciones)")
    print("=" * 60)

    dataset = _load_dataset(seed)

    nboot = 0 if args.bootstrap <= 0 else args.bootstrap
    print(f"\n[1/4] OLS + SSC + IC (bootstrap n={nboot})...")
    result = run_sensitivity_analysis(dataset=dataset, seed=seed, nboot=nboot)

    print("\nParametros (verdadero vs calibrado):")
    for i, name in enumerate(result.param_names):
        line = f"  {name:6s}  true={result.q_true[i]:10.4f}  ols={result.q_ols[i]:10.4f}"
        if result.ols_ci_reliable and np.all(np.isfinite(result.ci_params[i])):
            lo, hi = result.ci_params[i]
            line += f"  IC_t=[{lo:.4f}, {hi:.4f}]"
        if result.ci_params_boot is not None:
            blo, bhi = result.ci_params_boot[i]
            line += f"  IC_boot=[{blo:.4f}, {bhi:.4f}]"
        print(line)

    print(f"\nRMSE del ajuste: {result.sigma:.3f} m3/s")
    print(f"RMSE relativo (vs std Q_obs): {result.rmse_rel*100:.2f}%")
    print("Variable analizada: Q(t) en x=L (caudal aguas abajo). Ver GUIA_FIGURAS_PARTE2.txt")
    print("\nSuposiciones:")
    for msg in result.assumptions.messages:
        print(f"  {msg}")

    sobol_result = None
    if not args.skip_sobol and args.sobol_n > 0:
        total = args.sobol_n * (2 + 5)
        print(f"\n[2/4] Sobol Saltelli N={args.sobol_n} (~{total} simulaciones)...")
        print("  (puede tardar varios minutos)")
        sobol_result = run_sobol_analysis(
            n_samples=args.sobol_n,
            metric="nrmse",
            q_obs=dataset.y_obs_vector(),
            x_obs=dataset.x_obs,
            n_jobs=args.sobol_jobs,
            seed=seed,
        )
        for i, name in enumerate(result.param_names):
            print(
                f"  {name:6s}  S1={sobol_result.s1[i]:.3f}+/-{sobol_result.s1_conf[i]:.3f}  "
                f"ST={sobol_result.st[i]:.3f}+/-{sobol_result.st_conf[i]:.3f}"
            )
    else:
        print("\n[2/4] Sobol omitido")

    print("\n[3/4] Guardando figuras y tablas...")
    save_results(result, FIG_DIR, TABLE_DIR, sobol_result=sobol_result)

    print("\n[4/4] Archivos en figures/:")
    for name in sorted(p.name for p in FIG_DIR.glob("*.png")):
        print(f"  {name}")
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
