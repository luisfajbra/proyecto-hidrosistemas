"""
Ejemplo de parcial — Monte Carlo con distribuciones NO uniformes + GLUE.

Modelo didáctico (caudal en salida de un tramo):
    Q_out(t) = eta_Q * Q_in(t) + S0 + eps

Parámetros muestreados:
    eta_Q ~ LogNormal   (incertidumbre multiplicativa del aforo)
    S0    ~ Normal      (offset / almacenamiento base)
    n     ~ Beta        (factor de atenuación en [0, 1] sobre Q_in)

Incluye:
  - Muestreo con scipy.stats.rvs
  - Transformada inversa manual (ppf) para comparar con el método teórico
  - NSE, KGE, función objetivo para minimización
  - Pesos GLUE y cuantiles ponderados del caudal simulado

Ejecutar desde la raíz del proyecto:
    python scripts/ejemplo_parcial_monte_carlo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Permitir importar métricas del proyecto si se ejecuta desde cualquier cwd
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sensitivity import kge, nse  # noqa: E402


# ── Configuración ─────────────────────────────────────────────────────────────

N_SIM = 5_000
RNG_SEED = 42
FIG_PATH = _ROOT / "figures" / "ejemplo_parcial_mc_distribuciones.png"


# ── Datos sintéticos (caudal aguas arriba + “verdad” en salida) ───────────────

def generar_serie(n: int = 120, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q_in sintético + observaciones con ruido en parámetros verdaderos."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    q_in = 50.0 + 30.0 * np.sin(2 * np.pi * t / 30.0) + 5.0 * rng.standard_normal(n)
    q_in = np.clip(q_in, 5.0, None)

    eta_true, s0_true, n_true = 1.05, 8.0, 0.15
    q_obs = eta_true * q_in + s0_true + n_true * np.sqrt(q_in)
    q_obs += 2.0 * rng.standard_normal(n)
    return t, q_in, q_obs


# ── Modelo hidrológico sencillo ───────────────────────────────────────────────

def modelo_caudal(q_in: np.ndarray, eta_q: float, s0: float, n_atten: float) -> np.ndarray:
    return eta_q * q_in + s0 + n_atten * np.sqrt(np.clip(q_in, 0.0, None))


# ── Muestreo: distribuciones no uniformes ─────────────────────────────────────

def muestrear_parametros_rvs(n_sim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Método habitual: .rvs() usa transformada inversa internamente."""
    return {
        "eta_Q": stats.lognorm.rvs(s=0.25, scale=np.exp(0.0), size=n_sim, random_state=rng),
        "S0": stats.norm.rvs(loc=5.0, scale=4.0, size=n_sim, random_state=rng),
        "n": stats.beta.rvs(a=2.0, b=8.0, size=n_sim, random_state=rng) * 0.5,
    }


def muestrear_parametros_ppf(n_sim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Transformada inversa explícita (teoría del parcial):
        u ~ U(0,1)  →  x = F^{-1}(u) = dist.ppf(u)
    """
    u = rng.uniform(0.0, 1.0, size=n_sim)
    return {
        "eta_Q": stats.lognorm.ppf(u, s=0.25, scale=1.0),
        "S0": stats.norm.ppf(u, loc=5.0, scale=4.0),
        "n": stats.beta.ppf(u, a=2.0, b=8.0) * 0.5,
    }


# ── Función objetivo y GLUE ───────────────────────────────────────────────────

def objetivo_minimizar(q_obs: np.ndarray, q_sim: np.ndarray) -> float:
    """OF = 1 - NSE (menor es mejor; 0 = ajuste perfecto)."""
    return 1.0 - nse(q_obs, q_sim, np.ones(len(q_obs), dtype=bool))


def pesos_glue(of: np.ndarray) -> np.ndarray:
    """
    GLUE (Clase 11):
      L = 1 - OF
      si min(L) < 0 → L = L - min(L)
      normalizar L / sum(L)
    """
    lhood = 1.0 - of
    lhood = lhood - float(np.min(lhood))
    lhood = np.maximum(lhood, 0.0)
    total = float(lhood.sum())
    if total <= 0.0:
        return np.ones(len(lhood)) / len(lhood)
    return lhood / total


def cuantil_ponderado(x: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Cuantil empírico con pesos GLUE (sin depender de numpy >= 1.22 weighted quantile)."""
    order = np.argsort(x)
    xs, ws = x[order], weights[order]
    cw = np.cumsum(ws)
    idx = int(np.searchsorted(cw, q, side="left"))
    idx = min(idx, len(xs) - 1)
    return float(xs[idx])


# ── Pipeline Monte Carlo ───────────────────────────────────────────────────────

def ejecutar_monte_carlo(
    q_in: np.ndarray,
    q_obs: np.ndarray,
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    n_sim = len(params["eta_Q"])
    q_mean = np.empty(n_sim)
    of = np.empty(n_sim)
    kge_vals = np.empty(n_sim)
    mask = np.ones(len(q_obs), dtype=bool)

    for i in range(n_sim):
        q_sim = modelo_caudal(q_in, params["eta_Q"][i], params["S0"][i], params["n"][i])
        q_mean[i] = float(np.mean(q_sim))
        of[i] = objetivo_minimizar(q_obs, q_sim)
        kge_vals[i] = kge(q_obs, q_sim, mask)

    pesos = pesos_glue(of)
    conductual = kge_vals >= 0.5

    return {
        "q_mean": q_mean,
        "of": of,
        "kge": kge_vals,
        "pesos": pesos,
        "conductual": conductual,
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    _, q_in, q_obs = generar_serie()

    # Comparar dos formas de muestreo (deben ser estadísticamente equivalentes)
    p_rvs = muestrear_parametros_rvs(N_SIM, rng)
    p_ppf = muestrear_parametros_ppf(N_SIM, rng)

    res_rvs = ejecutar_monte_carlo(q_in, q_obs, p_rvs)
    res_ppf = ejecutar_monte_carlo(q_in, q_obs, p_ppf)

    w = res_rvs["pesos"]
    qm = res_rvs["q_mean"]
    p5 = cuantil_ponderado(qm, w, 0.05)
    p50 = cuantil_ponderado(qm, w, 0.50)
    p95 = cuantil_ponderado(qm, w, 0.95)
    n_cond = int(res_rvs["conductual"].sum())

    print("=== Monte Carlo — distribuciones no uniformes ===")
    print(f"Simulaciones: {N_SIM}")
    print(f"OF mínima (mejor): {res_rvs['of'].min():.4f}  |  NSE máx: {1 - res_rvs['of'].min():.4f}")
    print(f"KGE máximo: {np.nanmax(res_rvs['kge']):.4f}")
    print(f"Conjuntos conductuales (KGE >= 0.5): {n_cond} ({100 * n_cond / N_SIM:.1f} %)")
    print(f"Caudal medio simulado — cuantiles GLUE 5-50-95 %: {p5:.2f}, {p50:.2f}, {p95:.2f} m³/s")
    print(
        "Media eta_Q (muestreo rvs vs ppf): "
        f"{p_rvs['eta_Q'].mean():.3f} vs {p_ppf['eta_Q'].mean():.3f}"
    )

    if _HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.hist(res_rvs["of"], bins=40, color="steelblue", edgecolor="k", alpha=0.8)
        ax.axvline(1.0 - 0.5, color="crimson", ls="--", label="OF si NSE=0.5")
        ax.set_xlabel("OF = 1 − NSE")
        ax.set_ylabel("frecuencia")
        ax.set_title("Distribución de la función objetivo")
        ax.legend()

        ax = axes[1]
        sc = ax.scatter(
            p_rvs["eta_Q"],
            res_rvs["kge"],
            c=res_rvs["of"],
            s=8,
            alpha=0.35,
            cmap="viridis_r",
        )
        ax.axhline(0.5, color="crimson", ls="--", label="umbral KGE (GLUE)")
        ax.set_xlabel("eta_Q (LogNormal)")
        ax.set_ylabel("KGE")
        ax.set_title("Parámetro vs desempeño")
        fig.colorbar(sc, ax=ax, label="OF")
        ax.legend()

        fig.tight_layout()
        FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_PATH, dpi=150)
        plt.close(fig)
        print(f"Figura guardada: {FIG_PATH}")
    else:
        print("(matplotlib no instalado: se omitió la figura)")


if __name__ == "__main__":
    main()
