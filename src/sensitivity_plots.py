"""Figuras del complemento de sensibilidad (OLS, IC/IP, SSC). Parte 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_hydrograph_ci_pi(
    t_hours: np.ndarray,
    q_obs: np.ndarray,
    q_sim: np.ndarray,
    sigma: float,
    *,
    mask_plot: np.ndarray,
    warmup_hours: float = 0.0,
    z_pi: float = 1.96,
    z_ci: float | None = None,
    title: str = "Ajuste con intervalos de confianza y prediccion",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """
    Curva simulada, IC estrecho, IP ancho, puntos observados (complemento OLS).
    IC ~ media +/- z_ci * sigma/sqrt(n); IP ~ sim +/- z_pi * sigma.
    """
    m = mask_plot
    th = t_hours[m]
    obs = q_obs[m]
    sim = q_sim[m]
    n_cal = int(m.sum())
    if z_ci is None:
        z_ci = 1.96 / max(np.sqrt(n_cal), 1.0)

    ci_half = z_ci * sigma
    pi_half = z_pi * sigma

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if warmup_hours > 0:
        ax.axvspan(0, warmup_hours, color="gray", alpha=0.1)
    ax.fill_between(
        th,
        sim - pi_half,
        sim + pi_half,
        color="mediumpurple",
        alpha=0.2,
        label="Intervalo de prediccion",
    )
    ax.plot(th, sim - ci_half, color="green", ls="--", lw=1.2, label="IC confianza (inferior)")
    ax.plot(th, sim + ci_half, color="green", ls="--", lw=1.2, label="IC confianza (superior)")
    ax.plot(th, sim, "k-", lw=2, label="Simulacion")
    step = max(1, len(th) // 100)
    ax.plot(th[::step], obs[::step], "rs", ms=4, mfc="r", mew=0, label="Datos observados")
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Q (m3/s)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scaled_sensitivity_time(
    t_hours: np.ndarray,
    q_base: np.ndarray,
    ssc: np.ndarray,
    param_names: list[str],
    *,
    mask_plot: np.ndarray | None = None,
    title: str = "Coeficientes de sensibilidad escalados (SSC)",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Respuesta base + curvas SSC por parametro (sensibilidad local en el tiempo)."""
    m = mask_plot if mask_plot is not None else np.ones(len(t_hours), dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t_hours[m], q_base[m], "k-", lw=2, label="Respuesta (Q_sim)")
    axes[0].set_ylabel("Q (m3/s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)

    for j, nm in enumerate(param_names):
        axes[1].plot(t_hours[m], ssc[m, j], "--", lw=1.2, label=f"SSC_{nm}")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Tiempo (h)")
    axes[1].set_ylabel("dQ/dtheta (aprox.)")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
