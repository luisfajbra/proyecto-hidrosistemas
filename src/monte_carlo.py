"""Monte Carlo masivo con joblib (Parte 1 y base para GLUE)."""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .config import RANDOM_SEED, default_true_parameters
from .model import saint_venant_1d


def _run_one(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tp = default_true_parameters()
    n = float(rng.uniform(0.022, 0.048))
    s0 = float(rng.uniform(0.0003, 0.004))
    q0 = float(rng.uniform(30.0, 70.0))
    a_hyd = float(rng.uniform(60.0, 140.0))
    b_w = float(rng.uniform(30.0, 70.0))
    params = [n, s0, q0, a_hyd, b_w]
    full = saint_venant_1d(
        params, L=tp.L, nx=tp.nx, nt=tp.nt, dt=tp.dt, return_full=True
    )
    return {
        "seed": seed,
        "n": n,
        "S0": s0,
        "Q0": q0,
        "A_hyd": a_hyd,
        "B_w": b_w,
        "Q_max": float(np.max(full["Q_out"])),
        "mass_residual": full["mass_balance_residual"],
    }


def run_monte_carlo(n_samples: int = 1000, n_jobs: int = -1) -> list[dict[str, Any]]:
    seeds = list(range(RANDOM_SEED, RANDOM_SEED + n_samples))
    return Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_run_one)(s) for s in seeds
    )
