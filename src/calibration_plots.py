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


def plot_rsa_behavioral_cdfs(
    samples: np.ndarray,
    metric: np.ndarray,
    param_names: list[str],
    *,
    metric_higher_is_better: bool = True,
    n_levels: int = 9,
    cmap: str = "cool",
    title: str = "RSA — CDF empirica por umbral de desempeno",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """
    Analisis regional tipo GLUE/Hornberger-Spear: por parametro, curvas ECDF acumulada
    al variar el umbral de desempeno (likelihood proxy). Cuanto mas todas las curvas
    se pegan a la diagonal y=x (parametro normalizado), menos identificable suele ser.
    Inspiracion: rejilla 3x3 con gradiente cyan–magenta vs linea discontinua diagonal.

    metric: vector largo N (ej. NSE o KGE). Si es perdida minimizar, poner metric_higher_is_better=False.
    """
    samp = np.asarray(samples, dtype=float)
    met = np.asarray(metric, dtype=float)
    assert samp.shape[0] == met.shape[0]
    if not metric_higher_is_better:
        met = -met

    p = len(param_names)
    nrows = int(np.ceil(np.sqrt(p)))
    ncols = int(np.ceil(p / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.5 * nrows), squeeze=False)
    qs = np.linspace(0.1, 0.95, n_levels)

    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(0.0, 1.0)

    ax_flat = axes.ravel()
    for j, nm in enumerate(param_names):
        ax = ax_flat[j]
        v = samp[:, j]
        lo, hi = float(np.min(v)), float(np.max(v))
        span = hi - lo
        span = span if span > 1e-12 else 1.0

        xx = np.linspace(0.0, 1.0, 200)
        for k, q in enumerate(qs):
            thr = np.quantile(met, q)
            mask = met >= thr
            if mask.sum() < 5:
                continue
            u = (v[mask] - lo) / span
            u = np.clip(np.sort(u), 0.0, 1.0)
            y = np.linspace(0.0, 1.0, len(u))
            color = cmap_obj(norm(float(k) / max(n_levels - 1, 1)))
            ax.plot(u, y, color=color, lw=1.35, alpha=0.85)

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="no preferencia")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(nm)
        ax.set_xlabel("param normalizado ~[0,1]")
        ax.set_ylabel("F acumulada (dentro del subconjunto)")
        ax.grid(True, alpha=0.25)

    for j in range(p, len(ax_flat)):
        ax_flat[j].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Umbral (quantil sobre desempeno)")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dotty_mc(
    samples: np.ndarray,
    metric: np.ndarray,
    param_names: list[str],
    *,
    y_label: str = "NSE (o metrica objetivo)",
    title: str = "Dotty plot — dispersion Monte Carlo",
    ncols: int = 3,
    color: str = "0.2",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """
    Graficas dispersion parametro vs desempeno (blanco y negro / monocromo),
    tipo dotty plots usados en GLUE para juicio rapido de identificabilidad.
    """
    p = len(param_names)
    nrows = int(np.ceil(p / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.9 * ncols, 3.3 * nrows),
        squeeze=False,
    )
    obj = np.asarray(metric, dtype=float)
    samp = np.asarray(samples, dtype=float)

    for i, nm in enumerate(param_names):
        ax = axes.ravel()[i]
        ax.scatter(samp[:, i], obj, c=color, s=10, alpha=0.38, edgecolors="none")
        ax.set_xlabel(nm)
        if i % ncols == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)

    for j in range(p, nrows * ncols):
        axes.ravel()[j].set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


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
