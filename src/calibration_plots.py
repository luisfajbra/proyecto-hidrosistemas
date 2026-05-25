"""Figuras de calibracion e incertidumbre (Monte Carlo, GLUE). Parte 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_param_vs_objective(
    samples: np.ndarray,
    objective: np.ndarray,
    param_names: list[str],
    *,
    y_label: str = "Y",
    title: str = "Parametro vs metrica",
    threshold: float | None = None,
    ncols: int = 2,
    figsize_per_ax: tuple[float, float] = (4.2, 3.5),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Dispersion theta_i vs objetivo (Monte Carlo / filtro GLUE)."""
    p = len(param_names)
    nrows = int(np.ceil(p / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    obj = np.asarray(objective, dtype=float)
    if threshold is not None:
        mask = obj <= threshold
        samples_plot = samples[mask]
        obj_plot = obj[mask]
        subtitle = f"conductuales Y <= {threshold:.4g} (n={mask.sum()})"
    else:
        samples_plot = samples
        obj_plot = obj
        subtitle = f"todas las corridas (n={len(obj)})"

    for i, nm in enumerate(param_names):
        ax = axes.ravel()[i]
        ax.scatter(samples_plot[:, i], obj_plot, c="k", s=8, alpha=0.45, edgecolors="none")
        ax.set_xlabel(nm)
        if i % ncols == 0:
            ax.set_ylabel(y_label)
    for j in range(p, nrows * ncols):
        axes.ravel()[j].set_visible(False)

    fig.suptitle(f"{title} — {subtitle}", y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_obs_vs_sim(
    q_obs: np.ndarray,
    q_sim: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    title: str = "Simulado vs observado",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Dispersion Q_sim vs Q_obs con linea 1:1."""
    if mask is not None:
        o = q_obs[mask]
        s = q_sim[mask]
    else:
        o = q_obs
        s = q_sim
    lo = float(np.nanmin([o.min(), s.min()]))
    hi = float(np.nanmax([o.max(), s.max()]))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(s, o, c="b", s=14, alpha=0.5, edgecolors="none", label="Datos")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="1:1")
    ax.set_xlabel("Q simulado (m3/s)")
    ax.set_ylabel("Q observado (m3/s)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def glue_bands(Q_stack: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bandas p05–p50–p95 en cada instante (filas = simulaciones)."""
    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)
    return (
        np.percentile(Q_stack, lo, axis=0),
        np.percentile(Q_stack, 50, axis=0),
        np.percentile(Q_stack, hi, axis=0),
    )


def plot_hydrograph_glue(
    t_hours: np.ndarray,
    q_obs: np.ndarray,
    Q_stack: np.ndarray,
    q_ref: np.ndarray,
    *,
    mask_plot: np.ndarray,
    warmup_hours: float = 0.0,
    alpha: float = 0.05,
    title: str = "Hidrograma y bandas GLUE (5–95 %)",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Envolvente de credibilidad GLUE + referencia + observaciones."""
    p05, p50, p95 = glue_bands(Q_stack, alpha=alpha)
    m = mask_plot
    th = t_hours[m]
    obs = q_obs[m]

    fig, ax = plt.subplots(figsize=(10, 4))
    if warmup_hours > 0:
        ax.axvspan(0, warmup_hours, color="gray", alpha=0.12, label="Warm-up")
    ax.fill_between(th, p05[m], p95[m], color="gray", alpha=0.45, label=f"Banda {100*(1-alpha):.0f}%")
    ax.plot(th, p50[m], color="0.35", lw=1, ls="--", label="Mediana ensemble")
    ax.plot(th, q_ref[m], "k-", lw=1.8, label="Simulacion referencia")
    step = max(1, len(th) // 120)
    ax.plot(th[::step], obs[::step], "r.", ms=5, label="Observado")
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Q en x = L (m3/s)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
