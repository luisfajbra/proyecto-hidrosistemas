"""Verificacion del solver MacCormack (Parte 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (
    kinematic_wave_speed,
    manning_discharge,
    normal_depth,
    saint_venant_1d,
)


@dataclass
class VerificationReport:
    steady_manning_rmse: float
    kinematic_peak_error_s: float
    mass_balance_residual: float
    passed: bool


def verify_steady_uniform(
    n=0.035, S0=0.001, Q0=50.0, B_w=50.0, L=5000.0, nx=100, nt=400, dt=1.0
) -> float:
    """Flujo permanente (A_hyd=0): h simulado vs tirante normal de Manning."""
    params = [n, S0, Q0, 0.0, B_w]
    full = saint_venant_1d(params, L=L, nx=nx, nt=nt, dt=dt, return_full=True)
    h_anal = normal_depth(Q0, B_w, n, S0)
    h_sim = full["A"][-1, 10:-10] / B_w
    return float(np.sqrt(np.mean((h_sim - h_anal) ** 2)))


def verify_kinematic_peak_delay(
    n=0.035, S0=0.001, Q0=50.0, A_hyd=100.0, B_w=50.0, L=5000.0, nx=100, nt=300, dt=1.0
) -> float:
    """Retardo del pico en x=L vs L/ck (aproximacion cinematica)."""
    params = [n, S0, Q0, A_hyd, B_w]
    full = saint_venant_1d(params, L=L, nx=nx, nt=nt, dt=dt, return_full=True)
    h0 = normal_depth(Q0, B_w, n, S0)
    ck = kinematic_wave_speed(Q0, B_w, h0)
    travel = L / ck
    total_time = (nt - 1) * dt
    t_peak_in = 0.5 * total_time
    t_peak_out = full["t"][np.argmax(full["Q_out"])]
    return abs(t_peak_out - (t_peak_in + travel))


def run_verification() -> VerificationReport:
    rmse = verify_steady_uniform()
    kin_err = verify_kinematic_peak_delay()
    full = saint_venant_1d(
        [0.035, 0.001, 50.0, 100.0, 50.0], return_full=True
    )
    mass = full["mass_balance_residual"]
    passed = rmse < 0.2 and kin_err < 600.0 and mass < 0.25
    return VerificationReport(
        steady_manning_rmse=rmse,
        kinematic_peak_error_s=kin_err,
        mass_balance_residual=mass,
        passed=passed,
    )
