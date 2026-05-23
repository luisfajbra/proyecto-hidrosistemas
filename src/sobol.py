"""Analisis de sensibilidad global de Sobol (SALib)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import RANDOM_SEED, default_true_parameters, param_bounds
from .model import saint_venant_1d
from .synthetic_data import simulate_stations

PARAM_NAMES = ["n", "S0", "Q0", "A_hyd", "B_w"]


@dataclass
class SobolResult:
    problem: dict
    param_values: np.ndarray
    y: np.ndarray
    s1: np.ndarray
    s1_conf: np.ndarray
    st: np.ndarray
    st_conf: np.ndarray
    n_samples: int
    metric: str


def _sobol_problem() -> dict:
    b = param_bounds()
    return {
        "num_vars": 5,
        "names": PARAM_NAMES,
        "bounds": [
            b["n"],
            b["S0"],
            b["Q0"],
            b["A_hyd"],
            b["B_w"],
        ],
    }


def _scalar_response(
    params_row: np.ndarray,
    metric: str,
    q_ref: np.ndarray | None = None,
    x_obs: np.ndarray | None = None,
) -> float:
    """
    Metrica escalar Y para Sobol.

    - nrmse : RMSE vs q_ref (vector apilado multi-estacion o solo x=L)
    - q_peak: max(Q) en x=L
    - q_mean: media(Q) en x=L
    """
    params = list(map(float, params_row))
    if metric == "nrmse" and q_ref is not None:
        q_ref = np.asarray(q_ref, dtype=float)
        if x_obs is not None and len(x_obs) > 0:
            tp = default_true_parameters()
            sim = simulate_stations(
                params, np.asarray(x_obs, dtype=float), tp.L, tp.nx, tp.nt, tp.dt
            ).ravel()
        else:
            sim = np.asarray(saint_venant_1d(params), dtype=float)
        return float(np.sqrt(np.mean((sim - q_ref) ** 2)))
    q = np.asarray(saint_venant_1d(params), dtype=float)
    if metric == "q_mean":
        return float(np.mean(q))
    return float(np.max(q))


def run_sobol_analysis(
    n_samples: int = 512,
    metric: str = "nrmse",
    q_obs: np.ndarray | None = None,
    x_obs: np.ndarray | None = None,
    n_jobs: int = -1,
    seed: int = RANDOM_SEED,
) -> SobolResult:
    """
    Indices de Sobol (S1, ST) con intervalos de confianza (SALib).

    metric='nrmse' requiere q_obs (vector apilado multi-estacion o solo x=L).
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli

    if metric == "nrmse" and q_obs is None:
        raise ValueError("Para metric='nrmse' debe pasar el vector q_obs.")

    problem = _sobol_problem()
    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

    def _eval(row: np.ndarray) -> float:
        return _scalar_response(row, metric, q_obs, x_obs=x_obs)

    if n_jobs == 1:
        y = np.array([_eval(row) for row in param_values])
    else:
        y = np.array(
            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_eval)(row) for row in param_values
            )
        )

    si = sobol_analyze.analyze(
        problem,
        y,
        calc_second_order=False,
        conf_level=0.95,
        print_to_console=False,
    )

    return SobolResult(
        problem=problem,
        param_values=param_values,
        y=y,
        s1=si["S1"],
        s1_conf=si["S1_conf"],
        st=si["ST"],
        st_conf=si["ST_conf"],
        n_samples=n_samples,
        metric=metric,
    )
