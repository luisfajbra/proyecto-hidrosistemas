"""
Parte 2 — Sensibilidad local (metodo de clase: nonlinear_v1.ipynb).

- Coeficientes de sensibilidad escalados (SSC) y Jacobiano por diferencias finitas
- Ajuste por minimos cuadrados no lineales (scipy.optimize.least_squares)
- Validacion de las 5 suposiciones del modelo de errores probabilistico
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm, t as student_t

from .config import RANDOM_SEED, default_true_parameters, param_bounds
from .model import saint_venant_1d
from .synthetic_data import (
    SyntheticDataset,
    generate_synthetic_data,
    simulate_stations,
)

PARAM_NAMES = ["n", "S0", "Q0", "A_hyd", "B_w"]
ALPHA = 0.05
DH = 1e-4


@dataclass
class AssumptionChecks:
    """Resultados de las 5 suposiciones sobre los residuales."""

    mean_residual: float
    mean_near_zero: bool
    cond_number: float
    cond_ok: bool
    max_correlation: float
    correlation_ok: bool
    uncorrelated_runs: int
    min_runs_required: float
    uncorrelated_ok: bool
    messages: list[str] = field(default_factory=list)


@dataclass
class IntervalBands:
    """Intervalos de confianza (95%) y prediccion para la respuesta Q(t)."""

    ci_inf: np.ndarray
    ci_sup: np.ndarray
    pi_inf: np.ndarray
    pi_sup: np.ndarray


@dataclass
class SensitivityResult:
    q_true: np.ndarray
    q_ols: np.ndarray
    se: np.ndarray
    se_rel: np.ndarray
    ci_params: np.ndarray
    ci_params_boot: np.ndarray | None
    cov: np.ndarray
    corr: np.ndarray
    sigma: float
    rmse_rel: float
    ssc: np.ndarray
    jacobian: np.ndarray
    jac_ols: np.ndarray
    t: np.ndarray
    t_plot: np.ndarray
    y_obs: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    intervals: IntervalBands
    assumptions: AssumptionChecks
    param_names: list[str]
    ols_ci_reliable: bool
    cond_jtj: float
    n_stations: int = 1
    station_labels: list[str] = field(default_factory=lambda: ["x=L"])


def _sim_config():
    tp = default_true_parameters()
    return tp.L, tp.nx, tp.nt, tp.dt


def model_hydrograph(q: np.ndarray, nt: int | None = None, dt: float | None = None) -> np.ndarray:
    """Hidrograma Q(t) en x=L (ultima estacion por defecto)."""
    L, nx, nt_def, dt_def = _sim_config()
    nt_use = nt_def if nt is None else nt
    dt_use = dt_def if dt is None else dt
    return np.asarray(
        saint_venant_1d(q, L=L, nx=nx, nt=nt_use, dt=dt_use),
        dtype=float,
    )


def model_obs_vector(
    q: np.ndarray,
    x_obs: np.ndarray,
    nt: int,
    dt: float,
) -> np.ndarray:
    """Q apilado en todas las estaciones (para calibracion multi-seccion)."""
    L, nx, _, _ = _sim_config()
    mat = simulate_stations(q, x_obs, L, nx, nt, dt)
    return mat.ravel()


def make_model_callable(
    t: np.ndarray,
    x_obs: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Wrapper f(q) -> vector respuesta.
    x_obs=None: solo Q(t) en x=L. Con x_obs: apila todas las estaciones.
    """
    nt = len(t)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    x_obs_arr = None if x_obs is None else np.asarray(x_obs, dtype=float)

    def model(x: np.ndarray, q: np.ndarray) -> np.ndarray:
        del x
        qv = np.asarray(q, dtype=float)
        if x_obs_arr is None:
            return model_hydrograph(qv, nt=nt, dt=dt)
        return model_obs_vector(qv, x_obs_arr, nt, dt)

    return model


def SSC(
    q: np.ndarray,
    x: np.ndarray,
    fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dh: float = DH,
) -> np.ndarray:
    """
    Matriz de coeficientes de sensibilidad escalados X' (n x p).
    X'[:,i] = q_i * df/dq_i  (aprox. diferencias finitas, como en clase).
    """
    q = np.asarray(q, dtype=float)
    y_old = fun(x, q)
    xp = np.zeros((len(x), len(q)))
    for i in range(len(q)):
        q_in = q.copy()
        q_in[i] = (1.0 + dh) * q[i]
        y_new = fun(x, q_in)
        xp[:, i] = (y_new - y_old) / dh
    return xp


def jacobian(
    q: np.ndarray,
    x: np.ndarray,
    fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dh: float = DH,
) -> np.ndarray:
    """Jacobiano df/dq (n x p) por diferencias finitas."""
    q = np.asarray(q, dtype=float)
    y_old = fun(x, q)
    jac = np.zeros((len(x), len(q)))
    for i in range(len(q)):
        q_in = q.copy()
        q_in[i] = (1.0 + dh) * q[i]
        y_new = fun(x, q_in)
        jac[:, i] = (y_new - y_old) / (dh * q[i])
    return jac


def model_residuals(
    q: np.ndarray,
    x: np.ndarray,
    y_obs: np.ndarray,
    fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    return y_obs - fun(x, q)


def _bounds_vectors() -> tuple[list[float], list[float]]:
    b = param_bounds()
    lo = [b["n"][0], b["S0"][0], b["Q0"][0], b["A_hyd"][0], b["B_w"][0]]
    hi = [b["n"][1], b["S0"][1], b["Q0"][1], b["A_hyd"][1], b["B_w"][1]]
    return lo, hi


def _cond_jtj(jac: np.ndarray) -> float:
    jtj = jac.T @ jac
    if not np.all(np.isfinite(jtj)):
        return float("inf")
    s = np.linalg.svd(jtj, compute_uv=False)
    if s[-1] <= 0:
        return float("inf")
    return float(s[0] / s[-1])


def _ols_ci_is_reliable(jac: np.ndarray, threshold: float = 1e8) -> bool:
    return _cond_jtj(jac) < threshold


def _covariance_from_jacobian(jac: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Covarianza de parametros OLS: sigma^2 * (J'J)^+.
    Usa pseudoinversa + regularizacion (J singular si hay equifinalidad).
    """
    jtj = jac.T @ jac
    p = jtj.shape[0]
    scale = float(np.trace(jtj) / max(p, 1))
    reg = (1e-10 * scale if scale > 0 else 1e-10) * np.eye(p)
    return sigma2 * np.linalg.pinv(jtj + reg)


def _xt_x_inv_from_jacobian(jac: np.ndarray) -> np.ndarray:
    """(J'J)^+ para intervalos de confianza de la respuesta."""
    jtj = jac.T @ jac
    p = jtj.shape[0]
    scale = float(np.trace(jtj) / max(p, 1))
    reg = (1e-10 * scale if scale > 0 else 1e-10) * np.eye(p)
    return np.linalg.pinv(jtj + reg)


def fit_parameters(
    y_obs: np.ndarray,
    t: np.ndarray,
    q0: np.ndarray | None = None,
    x_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Minimos cuadrados no lineales.
    Returns: q_ols, residuals, jacobian at solution, rmse sigma.
    """
    fun = make_model_callable(t, x_obs=x_obs)
    x = t
    if q0 is None:
        q0 = np.array(default_true_parameters().as_vector()) * np.array(
            [1.05, 0.95, 1.02, 0.98, 1.01]
        )
    lo, hi = _bounds_vectors()
    result = least_squares(
        model_residuals,
        q0,
        bounds=(lo, hi),
        args=(x, y_obs, fun),
        method="trf",
    )
    q_ols = result.x
    resid = result.fun
    jac = result.jac
    n, p = len(y_obs), len(q0)
    ss = resid @ resid
    sigma2 = ss / max(n - p, 1)
    sigma = float(np.sqrt(sigma2))
    return q_ols, resid, jac, sigma


def response_confidence_and_prediction_intervals(
    q_ols: np.ndarray,
    jac: np.ndarray,
    sigma: float,
    y_pred: np.ndarray,
) -> IntervalBands:
    """
    IC y IP del 95% para Q(t) (formulas de clase con matriz de sensibilidad).
    """
    n, p = len(y_pred), len(q_ols)
    dof = max(n - p, 1)
    tcrit = float(student_t.ppf(1 - ALPHA / 2, dof))
    xt_x_inv = _xt_x_inv_from_jacobian(jac)
    ci_inf = np.zeros(n)
    ci_sup = np.zeros(n)
    pi_inf = np.zeros(n)
    pi_sup = np.zeros(n)
    for i in range(n):
        h = jac[i, :] @ xt_x_inv @ jac[i, :]
        se_ci = sigma * np.sqrt(max(h, 0.0))
        se_pi = sigma * np.sqrt(max(1.0 + h, 0.0))
        ci_inf[i] = y_pred[i] - tcrit * se_ci
        ci_sup[i] = y_pred[i] + tcrit * se_ci
        pi_inf[i] = y_pred[i] - tcrit * se_pi
        pi_sup[i] = y_pred[i] + tcrit * se_pi
    return IntervalBands(ci_inf=ci_inf, ci_sup=ci_sup, pi_inf=pi_inf, pi_sup=pi_sup)


def ols_diagnostics(
    q_ols: np.ndarray,
    residuals: np.ndarray,
    jac: np.ndarray,
    y_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n, p = len(y_obs), len(q_ols)
    ss = residuals @ residuals
    sigma2 = ss / max(n - p, 1)
    sigma = float(np.sqrt(sigma2))
    reliable = _ols_ci_is_reliable(jac)
    if reliable:
        cov = _covariance_from_jacobian(jac, sigma2)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        dinv = np.diag(1.0 / np.maximum(np.sqrt(np.diag(cov)), 1e-30))
        corr = dinv @ cov @ dinv
        tcrit = student_t.ppf(1 - ALPHA / 2, n - p)
        ci_inf = q_ols - se * tcrit
        ci_sup = q_ols + se * tcrit
        ci_params = np.column_stack((ci_inf, ci_sup))
    else:
        cov = np.full((p, p), np.nan)
        se = np.full(p, np.nan)
        corr = np.full((p, p), np.nan)
        ci_params = np.full((p, 2), np.nan)
    se_rel = se / np.maximum(np.abs(q_ols), 1e-12)
    return se, se_rel, cov, corr, ci_params


def check_error_assumptions(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    t: np.ndarray,
    jac: np.ndarray,
    sigma: float,
) -> AssumptionChecks:
    """Valida las 5 suposiciones del modelo de errores (clase)."""
    msgs: list[str] = []
    mean_r = float(np.mean(residuals))
    mean_ok = abs(mean_r) < 0.05 * max(np.std(residuals), 1e-9)
    msgs.append(
        f"Sup.2 Media~0: mean(R)={mean_r:.3e} -> {'OK' if mean_ok else 'REVISAR'}"
    )

    jtj = jac.T @ jac
    cond = float(np.linalg.cond(jtj))
    cond_ok = cond < 1e12
    msgs.append(f"Sup. identificabilidad: cond(J'J)={cond:.2e} -> {'OK' if cond_ok else 'REVISAR'}")

    cov = _covariance_from_jacobian(jac, sigma**2)
    diag_cov = np.maximum(np.diag(cov), 1e-30)
    dinv = np.diag(1.0 / np.sqrt(diag_cov))
    corr = dinv @ cov @ dinv
    off_diag = corr - np.diag(np.diag(corr))
    max_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
    corr_ok = max_corr < 0.98
    msgs.append(f"Correlacion parametros max={max_corr:.3f} -> {'OK' if corr_ok else 'REVISAR'}")

    rescross = residuals[1:] * residuals[:-1]
    count = int(np.sum(np.sign(rescross) < 0))
    minrun = (len(residuals) + 1) / 2
    uncorr_ok = count >= minrun
    msgs.append(
        f"Sup.4 No correlacion: {count} cruces (min {minrun:.0f}) -> "
        f"{'OK' if uncorr_ok else 'REVISAR'}"
    )
    msgs.append("Sup.1 Aditivos: revisar grafico R vs Ypred")
    msgs.append("Sup.3 Var constante: revisar grafico R vs t")
    msgs.append("Sup.5 Normalidad: revisar histograma de R")

    return AssumptionChecks(
        mean_residual=mean_r,
        mean_near_zero=mean_ok,
        cond_number=cond,
        cond_ok=cond_ok,
        max_correlation=max_corr,
        correlation_ok=corr_ok,
        uncorrelated_runs=count,
        min_runs_required=minrun,
        uncorrelated_ok=uncorr_ok,
        messages=msgs,
    )


def bootstrap_parameters(
    y_obs: np.ndarray,
    t: np.ndarray,
    q0: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    nboot: int = 500,
    seed: int = RANDOM_SEED,
    x_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remuestreo bootstrap de parametros (como nonlinear_v1.ipynb).
    Returns: q_boot (nboot x p), CI percentiles (p x 2).
    """
    fun = make_model_callable(t, x_obs=x_obs)
    x = t
    lo, hi = _bounds_vectors()
    n = len(y_obs)
    rng = np.random.default_rng(seed)
    q_boot = np.zeros((nboot, len(q0)))
    for i in range(nboot):
        idx = rng.integers(0, n, size=n)
        y_b = y_pred + residuals[idx]
        res = least_squares(
            model_residuals,
            q0,
            bounds=(lo, hi),
            args=(x, y_b, fun),
            method="trf",
        )
        q_boot[i, :] = res.x
    q_sorted = np.sort(q_boot, axis=0)
    L = int(np.round((ALPHA / 2) * nboot))
    U = int(np.round((1 - ALPHA / 2) * nboot))
    ci = np.column_stack((q_sorted[L, :], q_sorted[U, :]))
    return q_boot, ci


def run_sensitivity_analysis(
    dataset: SyntheticDataset | None = None,
    seed: int = RANDOM_SEED,
    nboot: int = 0,
) -> SensitivityResult:
    """Pipeline completo Parte 2."""
    if dataset is None:
        dataset = generate_synthetic_data(seed=seed)

    t = dataset.t
    nt = len(t)
    x_obs = np.asarray(dataset.x_obs, dtype=float)
    y_obs_fit = dataset.y_obs_vector()
    y_obs_plot = dataset.q_obs_downstream()
    q_true = np.array(dataset.params_true, dtype=float)
    fun_fit = make_model_callable(t, x_obs=x_obs)
    fun_L = make_model_callable(t, x_obs=None)

    n_sta = dataset.n_stations
    if n_sta > 1:
        print(
            f"  Calibracion multi-estacion ({n_sta} secciones: "
            f"{', '.join(dataset.station_labels)}); "
            f"{len(y_obs_fit)} observaciones."
        )

    q_ols, resid, jac_ols, sigma = fit_parameters(y_obs_fit, t, x_obs=x_obs)
    cond_j = _cond_jtj(jac_ols)
    ols_reliable = _ols_ci_is_reliable(jac_ols)
    if not ols_reliable:
        print(
            f"  AVISO: equifinalidad (cond(J'J)={cond_j:.2e}). "
            "IC por OLS no se reportan; use IC bootstrap en figuras/tablas."
        )
    else:
        print(f"  Identificabilidad: cond(J'J)={cond_j:.2e} (IC OLS mas fiables).")
    se, se_rel, cov, corr, ci_params = ols_diagnostics(
        q_ols, resid, jac_ols, y_obs_fit
    )
    y_pred = fun_L(t, q_ols)
    jac_L = jac_ols[(n_sta - 1) * nt : n_sta * nt, :]
    resid_L = resid[(n_sta - 1) * nt : n_sta * nt]

    t_plot = t.copy()
    ssc = SSC(q_ols, t_plot, fun_L)
    jac = jacobian(q_ols, t, fun_L)

    assumptions = check_error_assumptions(resid_L, y_pred, t, jac_L, sigma)
    rmse_rel = sigma / max(float(np.std(y_obs_fit)), 1e-9)
    if ols_reliable:
        intervals = response_confidence_and_prediction_intervals(
            q_ols, jac_L, sigma, y_pred
        )
    else:
        dof = max(len(y_obs_fit) - len(q_ols), 1)
        tcrit = float(student_t.ppf(1 - ALPHA / 2, dof))
        band = tcrit * sigma
        intervals = IntervalBands(
            ci_inf=y_pred - band,
            ci_sup=y_pred + band,
            pi_inf=y_pred - band,
            pi_sup=y_pred + band,
        )

    ci_boot = None
    if nboot > 0:
        _, ci_boot = bootstrap_parameters(
            y_obs_fit,
            t,
            q_ols,
            fun_fit(t, q_ols),
            resid,
            nboot=nboot,
            seed=seed,
            x_obs=x_obs,
        )

    return SensitivityResult(
        q_true=q_true,
        q_ols=q_ols,
        se=se,
        se_rel=se_rel,
        ci_params=ci_params,
        ci_params_boot=ci_boot,
        cov=cov,
        corr=corr,
        sigma=sigma,
        rmse_rel=rmse_rel,
        ssc=ssc,
        jacobian=jac,
        jac_ols=jac_ols,
        t=t,
        t_plot=t_plot,
        y_obs=y_obs_plot,
        y_pred=y_pred,
        residuals=resid_L,
        intervals=intervals,
        assumptions=assumptions,
        param_names=PARAM_NAMES,
        ols_ci_reliable=ols_reliable,
        cond_jtj=cond_j,
        n_stations=n_sta,
        station_labels=list(dataset.station_labels),
    )


def _plot_ssc_hidrograma(result: SensitivityResult, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    t_h = result.t_plot / 3600
    colors = ["c", "r", "g", "m", "orange"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_h, result.y_pred, "k-", lw=2)
    axes[0].set_ylabel("Q (m3/s)")
    loc = result.station_labels[-1] if result.station_labels else "x=L"
    axes[0].set_title(
        f"Panel A: Hidrograma calibrado en {loc} "
        f"(calibracion con {result.n_stations} estacion/es)"
    )
    axes[0].grid(True, alpha=0.3)
    for i, name in enumerate(result.param_names):
        axes[1].plot(
            t_h, result.ssc[:, i], "--", color=colors[i % len(colors)], label=name
        )
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Tiempo (h)")
    axes[1].set_ylabel("SSC (m3/s)")
    ssc_max = np.max(np.abs(result.ssc), axis=0)
    active = [result.param_names[i] for i in range(len(result.param_names)) if ssc_max[i] > 1e-6]
    inactive = [n for n in result.param_names if n not in active]
    subtitle = (
        f"Sensibles aqui: {', '.join(active) if active else 'ninguno'}"
        + (f" | ~0: {', '.join(inactive)}" if inactive else "")
    )
    axes[1].set_title(
        "Panel B: Sensibilidad local (SSC) — cambio en Q al perturbar cada parametro\n"
        + subtitle
    )
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "ssc_hidrograma.png", dpi=150)
    plt.close(fig)


def _plot_sup02_mean_zero(result: SensitivityResult, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mean_r = result.assumptions.mean_residual
    std_r = float(np.std(result.residuals))
    n = len(result.residuals)

    axes[0].bar(["Media(R)"], [mean_r], color="steelblue", edgecolor="k")
    axes[0].axhline(0, color="r", ls="--", lw=2)
    axes[0].set_ylabel("m3/s")
    axes[0].set_title(
        f"Sup.2 Media del error = 0\nmean(R)={mean_r:.3e} (debe estar cerca de cero)"
    )

    axes[1].boxplot(result.residuals, vert=True, widths=0.5)
    axes[1].axhline(0, color="r", ls="--", lw=2)
    axes[1].set_ylabel("Residuales (m3/s)")
    axes[1].set_title(f"Distribucion centrada (std={std_r:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "sup02_media_error_cero.png", dpi=150)
    plt.close(fig)


def _plot_sup04_uncorrelated(result: SensitivityResult, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    r = result.residuals
    cross = r[1:] * r[:-1]
    signs = np.sign(cross)
    is_cross = signs < 0
    count = int(np.sum(is_cross))
    minrun = result.assumptions.min_runs_required

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(result.t / 3600, r, "b-", lw=1)
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_ylabel("R (m3/s)")
    axes[0].set_title("Sup.4 Errores no correlacionados — serie de residuales")

    colors = np.where(is_cross, "green", "gray")
    axes[1].scatter(result.t[1:] / 3600, cross, c=colors, s=18, alpha=0.8)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("Tiempo (h)")
    axes[1].set_ylabel("R[i]*R[i+1]")
    axes[1].set_title(
        f"Cruces (prod. < 0): {count} / minimo {minrun:.0f} — "
        f"{'OK' if result.assumptions.uncorrelated_ok else 'REVISAR'}"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "sup04_errores_no_correlacionados.png", dpi=150)
    plt.close(fig)


def _plot_intervals(result: SensitivityResult, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    t_h = result.t / 3600
    iv = result.intervals
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_h, result.y_obs, "rs", ms=3, alpha=0.6, label="Q observado (sintetico)")
    ax.plot(t_h, result.y_pred, "k-", lw=2, label="Q simulado (parametros calibrados)")
    if result.ols_ci_reliable:
        ax.fill_between(
            t_h, iv.ci_inf, iv.ci_sup, color="green", alpha=0.25, label="IC 95% media (OLS)"
        )
        ax.fill_between(
            t_h, iv.pi_inf, iv.pi_sup, color="magenta", alpha=0.15, label="IP 95% (OLS)"
        )
        subtitle = "Bandas sobre la serie Q(t) en x=L (clase nonlinear_v1)"
    else:
        ax.fill_between(
            t_h, iv.pi_inf, iv.pi_sup, color="orange", alpha=0.2,
            label="Banda +/- 1.96*sigma (equifinalidad)",
        )
        subtitle = "OLS singular: banda simple por sigma del ajuste (ver bootstrap en parametros)"
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Caudal Q en x=L (m3/s)")
    ax.set_title(f"Incertidumbre del hidrograma\n{subtitle}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "intervalos_confianza_prediccion.png", dpi=150)
    plt.close(fig)


def _plot_param_ci(result: SensitivityResult, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = result.param_names
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, result.q_true, width, label="Verdadero (sintetico)", color="0.7")
    if result.ols_ci_reliable and np.all(np.isfinite(result.ci_params)):
        yerr = np.row_stack(
            (
                result.q_ols - result.ci_params[:, 0],
                result.ci_params[:, 1] - result.q_ols,
            )
        )
        ax.errorbar(
            x + width / 2,
            result.q_ols,
            yerr=yerr,
            fmt="o",
            capsize=5,
            color="crimson",
            label="Calibrado +/- IC 95% (t)",
        )
    else:
        ax.plot(x + width / 2, result.q_ols, "o", color="crimson", label="Calibrado (OLS)")
    if result.ci_params_boot is not None:
        boot_err = np.row_stack(
            (
                result.q_ols - result.ci_params_boot[:, 0],
                result.ci_params_boot[:, 1] - result.q_ols,
            )
        )
        ax.errorbar(
            x + width / 2,
            result.q_ols,
            yerr=boot_err,
            fmt="none",
            ecolor="navy",
            elinewidth=1.5,
            capsize=3,
            label="IC 95% bootstrap",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(
        "Incertidumbre de PARAMETROS (no del hidrograma)\n"
        "Una barra por variable; IC bootstrap es el mas confiable si hay equifinalidad"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "parametros_intervalos_confianza.png", dpi=150)
    plt.close(fig)


def save_sobol_results(sobol_result, fig_dir: Path, table_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from .sobol import SobolResult

    if not isinstance(sobol_result, SobolResult):
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    names = sobol_result.problem["names"]
    df = pd.DataFrame(
        {
            "parametro": names,
            "S1": sobol_result.s1,
            "S1_conf": sobol_result.s1_conf,
            "ST": sobol_result.st,
            "ST_conf": sobol_result.st_conf,
        }
    )
    df.to_csv(table_dir / "sobol_indices.csv", index=False)

    # Etiquetas claras: Q0 es PARAMETRO (caudal afluente), no la serie Q(t)
    xlabels = [
        "n",
        "S0",
        "Q0\n(caudal\nentrada)",
        "A_hyd",
        "B_w",
    ]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, sobol_result.s1, w, yerr=sobol_result.s1_conf, capsize=4, label="S1")
    ax.bar(x + w / 2, sobol_result.st, w, yerr=sobol_result.st_conf, capsize=4, label="ST")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Indice de Sobol (0 = no influye en Y)")
    ax.set_xlabel("Los 5 PARAMETROS del modelo (eje X); no es solo uno")
    metric_txt = {
        "nrmse": "Y = RMSE(Q_sim, Q_obs) en todas las estaciones",
        "q_peak": "Y = max(Q) en x=L",
        "q_mean": "Y = mean(Q) en x=L",
    }.get(sobol_result.metric, sobol_result.metric)
    ax.set_title(
        f"Sensibilidad global Sobol de cada PARAMETRO\n{metric_txt}\n"
        f"S1=directo, ST=total (Saltelli N={sobol_result.n_samples})"
    )
    for i, name in enumerate(names):
        val = max(sobol_result.s1[i], sobol_result.st[i])
        if val > 0.05:
            ax.text(
                i, val + sobol_result.st_conf[i] + 0.03,
                f"{name}\nS1={sobol_result.s1[i]:.2f}",
                ha="center", fontsize=8,
            )
        else:
            ax.text(i, 0.02, "~0", ha="center", fontsize=8, color="gray")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "sobol_indices.png", dpi=150)
    plt.close(fig)


def save_results(
    result: SensitivityResult,
    fig_dir: Path,
    table_dir: Path,
    sobol_result=None,
) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # --- Tabla parametros estimados ---
    rows = []
    for i, name in enumerate(result.param_names):
        row = {
            "parametro": name,
            "verdadero": result.q_true[i],
            "estimado_ols": result.q_ols[i],
            "SE": result.se[i],
            "SE_rel": result.se_rel[i],
            "CI_t_inf": result.ci_params[i, 0],
            "CI_t_sup": result.ci_params[i, 1],
        }
        if result.ci_params_boot is not None:
            row["CI_boot_inf"] = result.ci_params_boot[i, 0]
            row["CI_boot_sup"] = result.ci_params_boot[i, 1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(table_dir / "parametros_ols_sensibilidad.csv", index=False)

    # --- IC/IP respuesta (CSV) ---
    pd.DataFrame(
        {
            "t_s": result.t,
            "Q_pred": result.y_pred,
            "Q_obs": result.y_obs,
            "CI_inf": result.intervals.ci_inf,
            "CI_sup": result.intervals.ci_sup,
            "PI_inf": result.intervals.pi_inf,
            "PI_sup": result.intervals.pi_sup,
        }
    ).to_csv(table_dir / "intervalos_hidrograma.csv", index=False)

    # --- Resumen SSC (max |SSC| en el tiempo) ---
    ssc_rows = []
    for i, name in enumerate(result.param_names):
        col = result.ssc[:, i]
        ssc_rows.append(
            {
                "parametro": name,
                "SSC_max_abs": float(np.max(np.abs(col))),
                "SSC_mean_abs": float(np.mean(np.abs(col))),
            }
        )
    pd.DataFrame(ssc_rows).to_csv(table_dir / "ssc_resumen.csv", index=False)

    # --- Suposiciones ---
    with open(table_dir / "suposiciones_errores.txt", "w", encoding="utf-8") as f:
        for msg in result.assumptions.messages:
            f.write(msg + "\n")

    _plot_ssc_hidrograma(result, fig_dir)

    # --- Sup.1: R vs Ypred ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.y_pred, result.residuals, "bs", ms=4)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Q simulado (m3/s)")
    ax.set_ylabel("Residual R = Q_obs - Q_sim (m3/s)")
    ax.set_title(
        "Sup.1 Errores aditivos\n"
        "Patron aleatorio sin forma = OK; curva sistematica = revisar modelo"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "sup01_residuales_vs_prediccion.png", dpi=150)
    plt.close(fig)

    # --- Sup.3: R vs t ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.t / 3600, result.residuals, "bs", ms=4)
    ax.axhline(0, color="r", lw=1)
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Residuales (m3/s)")
    ax.set_title("Sup.3 Varianza constante (visual)")
    fig.tight_layout()
    fig.savefig(fig_dir / "sup03_residuales_vs_tiempo.png", dpi=150)
    plt.close(fig)

    # --- Sup.5: histograma + normal ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(result.residuals, density=True, bins=12, alpha=0.5, edgecolor="black")
    xmin, xmax = ax.get_xlim()
    xnorm = np.linspace(xmin, xmax, 100)
    ax.plot(xnorm, norm.pdf(xnorm, 0, result.sigma), "k-", lw=2)
    ax.set_xlabel("Residuales (m3/s)")
    ax.set_ylabel("Densidad")
    ax.set_title("Sup.5 Normalidad de errores")
    fig.tight_layout()
    fig.savefig(fig_dir / "sup05_histograma_residuales.png", dpi=150)
    plt.close(fig)

    # --- Correlacion parametros ---
    fig, ax = plt.subplots(figsize=(5, 4))
    corr_plot = result.corr
    corr_title = "Correlacion entre parametros (OLS)"
    if not np.all(np.isfinite(corr_plot)):
        cov_j = _covariance_from_jacobian(result.jac_ols, result.sigma**2)
        diag = np.maximum(np.diag(cov_j), 1e-30)
        dinv = np.diag(1.0 / np.sqrt(diag))
        corr_plot = dinv @ cov_j @ dinv
        corr_title += "\n(equifinalidad: aprox. pinv J'J; puede ser poco fiable)"
    im = ax.imshow(corr_plot, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(len(result.param_names)):
        for j in range(len(result.param_names)):
            val = corr_plot[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.5 else "black",
                )
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(result.param_names, rotation=45, ha="right")
    ax.set_yticklabels(result.param_names)
    fig.colorbar(im, ax=ax)
    ax.set_title(corr_title)
    fig.tight_layout()
    fig.savefig(fig_dir / "correlacion_parametros.png", dpi=150)
    plt.close(fig)

    _plot_sup02_mean_zero(result, fig_dir)
    _plot_sup04_uncorrelated(result, fig_dir)
    _plot_intervals(result, fig_dir)
    _plot_param_ci(result, fig_dir)

    if sobol_result is not None:
        save_sobol_results(sobol_result, fig_dir, table_dir)
