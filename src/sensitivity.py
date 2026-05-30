"""
Parte 2 — sensibilidad global (Sobol), métricas, complemento OLS y figuras.

Toda la lógica de programación vive aquí; el notebook solo configura y llama.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from scipy.optimize import least_squares
from scipy.stats import norm, t as student_t


# ── Raíz de proyecto e import de sinteticos ───────────────────────────────────


def project_root_from_cwd(cwd: Path | None = None) -> Path:
    root = Path(cwd or Path.cwd()).resolve()
    if not (root / "src").is_dir():
        root = root.parent
    return root


def load_sinteticos_module(project_root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "_sinteticos_part2", project_root / "data" / "sinteticos.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_csv_name(*, use_long_series: bool, use_noisy_csv: bool) -> str:
    if use_long_series:
        return "series_larga_ruido.csv" if use_noisy_csv else "series_larga_shift.csv"
    return "series_corta_ruido.csv" if use_noisy_csv else "series_corta_shift.csv"


# ── Métricas ────────────────────────────────────────────────────────────────────


def nse(obs: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> float:
    o, s = obs[mask], sim[mask]
    den = float(np.sum((o - np.mean(o)) ** 2))
    return 1.0 if den < 1e-12 else float(1.0 - np.sum((o - s) ** 2) / den)


def kge(obs: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> float:
    o, s = obs[mask], sim[mask]
    if np.std(o) < 1e-12 or np.std(s) < 1e-12:
        return float("-inf")
    r = float(np.corrcoef(o, s)[0, 1])
    return float(
        1.0
        - np.sqrt(
            (r - 1) ** 2
            + (np.std(s) / np.std(o) - 1) ** 2
            + (np.mean(s) / np.mean(o) - 1) ** 2
        )
    )


def rmse_m(obs: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def metrics_table(
    obs: np.ndarray,
    sim: np.ndarray,
    *,
    mask_cal: np.ndarray,
    mask_val: np.ndarray,
    mask_sobol: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for label, m in (
        ("calibracion", mask_cal),
        ("validacion", mask_val),
        ("post_warmup", mask_sobol),
    ):
        if m.sum() < 2:
            continue
        rows.append(
            {
                "periodo": label,
                "NSE": nse(obs, sim, m),
                "KGE": kge(obs, sim, m),
                "RMSE_m3s": rmse_m(obs, sim, m),
            }
        )
    return pd.DataFrame(rows)


def sobol_scalar_y(
    q_obs: np.ndarray,
    q_sim: np.ndarray,
    mask_sobol: np.ndarray,
    metric: str,
) -> float:
    metric = metric.lower().strip()
    if metric == "rmse":
        return rmse_m(q_obs, q_sim, mask_sobol)
    if metric == "kge":
        return 1.0 - kge(q_obs, q_sim, mask_sobol)
    return 1.0 - nse(q_obs, q_sim, mask_sobol)


def sobol_objective_two_stations(
    q_obs_up: np.ndarray,
    q_obs_dn: np.ndarray,
    q_sim_up: np.ndarray,
    q_sim_dn: np.ndarray,
    mask: np.ndarray,
    metric: str,
    *,
    weight_up: float = 0.5,
    weight_dn: float = 0.5,
) -> float:
    """Objetivo Sobol: promedio ponderado de desempeño en aguas arriba y salida."""
    metric = metric.lower().strip()
    wu = float(weight_up)
    wd = float(weight_dn)
    s = wu + wd
    if s <= 0:
        wu, wd = 0.5, 0.5
    else:
        wu, wd = wu / s, wd / s
    if metric == "downstream":
        return sobol_scalar_y(q_obs_dn, q_sim_dn, mask, "nse")
    y_up = sobol_scalar_y(q_obs_up, q_sim_up, mask, metric)
    y_dn = sobol_scalar_y(q_obs_dn, q_sim_dn, mask, metric)
    return wu * y_up + wd * y_dn


SOBOL_LABELS = {
    "nse": "Y=0.5*(1-NSE_up)+0.5*(1-NSE_dn)",
    "rmse": "Y=RMSE mixto (up+dn)",
    "kge": "Y=1-KGE mixto (up+dn)",
    "downstream": "Y=1-NSE solo salida",
}


# ── Simulación dos estaciones (x=0 y x=L) ─────────────────────────────────────


def simulate_discharge_upstream_downstream(
    saint_venant_1d: Callable[..., Any],
    params_sequence: np.ndarray | Sequence[float],
    *,
    q_upstream: np.ndarray,
    time_seconds: np.ndarray,
    nt: int,
    L: float,
    nx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Caudales en borde aguas arriba (i=0) y salida (i=nx-1) [m³/s], nt pasos.

    Forzante aguas arriba + Saint-Venant 1D (sin término lateral en Parte 2/3).
    """
    full = saint_venant_1d(
        params_sequence,
        q_upstream=q_upstream,
        time_seconds=time_seconds,
        nt=nt,
        L=L,
        nx=nx,
        return_full=True,
    )
    Q = np.asarray(full["Q"], dtype=float)
    return np.maximum(Q[:, 0], 0.0), np.maximum(Q[:, -1], 0.0)


# ── Simulador dos estaciones (picklable para joblib) ───────────────────────────


class TwoStationSimulator:
    """θ = [n, S0, B_W, eta_Q]. Solver 1D sin aporte lateral en el análisis."""

    __slots__ = ("_solve", "_q_up", "_t_sec", "_nt", "_L", "_nx")

    def __init__(
        self,
        saint_venant_1d: Callable[..., Any],
        q_upstream_csv: np.ndarray,
        time_seconds: np.ndarray,
        nt: int,
        *,
        canal_L: float,
        nx: int,
    ) -> None:
        self._solve = saint_venant_1d
        self._q_up = np.asarray(q_upstream_csv, dtype=float)
        self._t_sec = np.asarray(time_seconds, dtype=float)
        self._nt = int(nt)
        self._L = float(canal_L)
        self._nx = int(nx)

    def both(self, params: np.ndarray | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        n, s0, bw, eta = map(float, params)
        return simulate_discharge_upstream_downstream(
            self._solve,
            [n, s0, bw],
            q_upstream=eta * self._q_up,
            time_seconds=self._t_sec,
            nt=self._nt,
            L=self._L,
            nx=self._nx,
        )

    def __call__(self, params: np.ndarray | Sequence[float]) -> np.ndarray:
        return self.both(params)[1]


# ── Contexto Parte 2 ──────────────────────────────────────────────────────────


@dataclass
class SensitivityPart2Context:
    root: Path
    fig_dir: Path
    data_dir: Path
    reports_dir: Path
    sinteticos: Any
    df: pd.DataFrame
    csv_name: str
    t_sec: np.ndarray
    q_up: np.ndarray
    q_obs: np.ndarray
    nt: int
    mask_warm: np.ndarray
    mask_cal: np.ndarray
    mask_val: np.ndarray
    mask_fit: np.ndarray
    mask_sobol: np.ndarray
    param_names: list[str]
    params_true: list[float]
    bounds_lo: list[float]
    bounds_hi: list[float]
    problem: dict[str, Any]
    simulator: TwoStationSimulator
    weight_up: float
    weight_dn: float
    l_canal: float
    nx: int
    warmup_seconds: float
    nt_raw: int
    subsample_stride: int


def _eta_q_bounds(
    q_up: np.ndarray,
    mask_cal: np.ndarray,
    *,
    k_sigma: float = 2.0,
    eta_min: float = 0.01,
) -> tuple[float, float, float]:
    """
    Rango uniforme de eta_Q desde la dispersión de Q_upstream en calibración.

    Intervalo [1 - k*sigma/mu, 1 + k*sigma/mu] con sigma y mu del CSV en calibracion
    (equivalente a 1 +/- k*CV). Sin recorte fijo del 30 %.
    """
    q = np.asarray(q_up, dtype=float)[mask_cal]
    if q.size < 2:
        return 1.0, max(eta_min, 0.5), 1.5
    mu = float(np.mean(q))
    sigma = float(np.std(q, ddof=1))
    if mu <= 1e-12:
        return 1.0, eta_min, 2.0
    rel = k_sigma * sigma / mu
    lo = max(eta_min, 1.0 - rel)
    hi = 1.0 + rel
    if lo >= hi:
        hi = lo + max(rel, 0.05)
    return 1.0, lo, hi


def eta_q_dispersion_stats(
    q_up: np.ndarray,
    mask_cal: np.ndarray,
    *,
    k_sigma: float = 2.0,
) -> dict[str, float]:
    """Estadísticas usadas para documentar el rango de eta_Q (informe / frontend)."""
    q = np.asarray(q_up, dtype=float)[mask_cal]
    if q.size < 2:
        return {"mu": float("nan"), "sigma": float("nan"), "cv": float("nan"), "k_sigma": k_sigma}
    mu = float(np.mean(q))
    sigma = float(np.std(q, ddof=1))
    cv = sigma / mu if abs(mu) > 1e-12 else float("nan")
    return {"mu": mu, "sigma": sigma, "cv": cv, "k_sigma": k_sigma}


def _bw_bounds(
    bw_ref: float,
    *,
    rel: float = 0.4,
    abs_clip: tuple[float, float] = (20.0, 100.0),
) -> tuple[float, float, float]:
    lo = max(abs_clip[0], bw_ref * (1.0 - rel))
    hi = min(abs_clip[1], bw_ref * (1.0 + rel))
    return float(bw_ref), lo, hi


def subsample_series_dataframe(
    df: pd.DataFrame,
    *,
    stride: int = 1,
    max_nt: int | None = None,
) -> pd.DataFrame:
    """
    Reduce filas para acelerar Sobol/OLS (serie larga).

    stride: conserva 1 de cada `stride` filas (p. ej. 12 → cada 3 h si dt=15 min).
    max_nt: si quedan más filas, remuestrea índices uniformes hasta max_nt.
    """
    stride = max(1, int(stride))
    out = df.iloc[::stride].copy()
    if max_nt is not None and len(out) > int(max_nt):
        idx = np.linspace(0, len(out) - 1, int(max_nt), dtype=int)
        out = out.iloc[idx].copy()
    return out.reset_index(drop=True)


def default_runtime_plan(
    *,
    use_long_series: bool,
    informe: bool,
) -> dict[str, int | bool]:
    """
    Presets de tiempo de ejecución (notebook 02).

    Serie corta + informe: Sobol moderado. Serie larga: stride + tope de filas
    y menos muestras Sobol para evitar días de cómputo.
    """
    if not use_long_series:
        if informe:
            return {
                "subsample_stride": 1,
                "max_nt": None,
                "sobol_n": 512,
                "nboot": 50,
                "compute_ssc": True,
                "run_ols": True,
            }
        return {
            "subsample_stride": 1,
            "max_nt": None,
            "sobol_n": 128,
            "nboot": 0,
            "compute_ssc": False,
            "run_ols": False,
        }
    if informe:
        return {
            "subsample_stride": 12,
            "max_nt": 6000,
            "sobol_n": 256,
            "nboot": 30,
            "compute_ssc": False,
            "run_ols": True,
        }
    return {
        "subsample_stride": 24,
        "max_nt": 3000,
        "sobol_n": 64,
        "nboot": 0,
        "compute_ssc": False,
        "run_ols": False,
    }


def build_sensitivity_context(
    *,
    project_root: Path,
    saint_venant_fn: Callable[..., Any],
    sinteticos_mod: Any,
    bw_ref: float,
    l_canal: float,
    nx: int,
    warmup_seconds: float,
    param_names: list[str],
    params_true_n_s0: Sequence[float],
    bounds_lo_n_s0: Sequence[float],
    bounds_hi_n_s0: Sequence[float],
    eta_k_sigma: float = 2.0,
    bw_rel: float = 0.4,
    weight_up: float = 0.5,
    weight_dn: float = 0.5,
    cal_frac: float,
    val_frac: float,
    use_long_series: bool,
    use_noisy_csv: bool,
    subsample_stride: int = 1,
    max_nt: int | None = None,
) -> SensitivityPart2Context:
    fig_dir = project_root / "figures"
    data_dir = project_root / "data" / "synthetic"
    reports_dir = project_root / "reports"
    for d in (fig_dir, data_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_name = synthetic_csv_name(use_long_series=use_long_series, use_noisy_csv=use_noisy_csv)
    csv_path = data_dir / csv_name
    if not csv_path.is_file():
        sinteticos_mod.generate(str(data_dir))

    df_raw = pd.read_csv(csv_path, parse_dates=["datetime"])
    nt_raw = len(df_raw)
    df = subsample_series_dataframe(df_raw, stride=subsample_stride, max_nt=max_nt)
    t_sec = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    q_up = df["Q_upstream_m3s"].to_numpy(dtype=float)
    if "Q_downstream_m3s" not in df.columns:
        raise ValueError("El CSV debe incluir la columna Q_downstream_m3s.")
    q_obs = df["Q_downstream_m3s"].to_numpy(dtype=float)
    nt = len(df)

    mask_warm = t_sec < warmup_seconds
    idx_post = np.where(~mask_warm)[0]
    n_post = len(idx_post)
    n_cal = min(max(1, int(cal_frac * n_post)), n_post - max(1, int(val_frac * n_post)))
    n_val = max(1, int(val_frac * n_post))
    i_cal = idx_post[:n_cal]
    i_val = idx_post[n_cal : n_cal + n_val]

    mask_cal = np.zeros(nt, dtype=bool)
    mask_val = np.zeros(nt, dtype=bool)
    mask_cal[i_cal] = True
    mask_val[i_val] = True
    mask_fit = mask_cal
    mask_sobol = ~mask_warm

    eta_true, eta_lo, eta_hi = _eta_q_bounds(q_up, mask_cal, k_sigma=eta_k_sigma)
    bw_true, bw_lo, bw_hi = _bw_bounds(bw_ref, rel=bw_rel)
    bounds_lo = list(bounds_lo_n_s0) + [bw_lo, eta_lo]
    bounds_hi = list(bounds_hi_n_s0) + [bw_hi, eta_hi]
    problem: dict[str, Any] = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": list(zip(bounds_lo, bounds_hi)),
    }

    sim = TwoStationSimulator(
        saint_venant_fn,
        q_up,
        t_sec,
        nt,
        canal_L=l_canal,
        nx=nx,
    )
    params_true = list(params_true_n_s0) + [bw_true, eta_true]

    return SensitivityPart2Context(
        root=project_root,
        fig_dir=fig_dir,
        data_dir=data_dir,
        reports_dir=reports_dir,
        sinteticos=sinteticos_mod,
        df=df,
        csv_name=csv_name,
        t_sec=t_sec,
        q_up=q_up,
        q_obs=q_obs,
        nt=nt,
        mask_warm=mask_warm,
        mask_cal=mask_cal,
        mask_val=mask_val,
        mask_fit=mask_fit,
        mask_sobol=mask_sobol,
        param_names=list(param_names),
        params_true=params_true,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        problem=problem,
        simulator=sim,
        weight_up=float(weight_up),
        weight_dn=float(weight_dn),
        l_canal=l_canal,
        nx=nx,
        warmup_seconds=warmup_seconds,
        nt_raw=nt_raw,
        subsample_stride=max(1, int(subsample_stride)),
    )


def metrics_table_two_stations(
    ctx: SensitivityPart2Context,
    q_sim_up: np.ndarray,
    q_sim_dn: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for estacion, obs, sim in (
        ("aguas_arriba", ctx.q_up, q_sim_up),
        ("salida", ctx.q_obs, q_sim_dn),
    ):
        m = metrics_table(
            obs,
            sim,
            mask_cal=ctx.mask_cal,
            mask_val=ctx.mask_val,
            mask_sobol=ctx.mask_sobol,
        )
        m.insert(0, "estacion", estacion)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def print_data_summary(ctx: SensitivityPart2Context, *, n_sobol_eval: int) -> None:
    if ctx.nt_raw != ctx.nt:
        print(
            f"CSV: {ctx.csv_name}  nt={ctx.nt}  (original {ctx.nt_raw}, "
            f"stride={ctx.subsample_stride})"
        )
    else:
        print(f"CSV: {ctx.csv_name}  nt={ctx.nt}")
    print(f"warm-up={ctx.mask_warm.sum()}  cal={ctx.mask_cal.sum()}  val={ctx.mask_val.sum()}")
    bw_t, eta_t = ctx.params_true[2], ctx.params_true[3]
    print(f"B_W        verdadero={bw_t:.2f} m  rango Sobol=[{ctx.bounds_lo[2]:.2f}, {ctx.bounds_hi[2]:.2f}]")
    stats = eta_q_dispersion_stats(ctx.q_up, ctx.mask_cal)
    print(
        f"eta_Q      verdadero={eta_t:.4f}  rango Sobol=[{ctx.bounds_lo[3]:.4f}, {ctx.bounds_hi[3]:.4f}]  "
        f"(1 +/- {stats['k_sigma']:.1f}*sigma/mu, sigma={stats['sigma']:.3f} mu={stats['mu']:.3f} m3/s)"
    )
    print(
        f"Obs. salida: Q_downstream_m3s del CSV | "
        f"Y Sobol: w_up={ctx.weight_up:.2f}  w_dn={ctx.weight_dn:.2f}"
    )
    print(f"Evaluaciones Sobol: {n_sobol_eval}")


def run_sobol_pipeline(
    ctx: SensitivityPart2Context,
    *,
    sobol_n: int,
    sobol_metric: str,
    sobol_conf: float,
    n_jobs: int,
    rng_seed: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    np.random.seed(rng_seed)
    n_eval = sobol_n * (ctx.problem["num_vars"] + 2)
    label = SOBOL_LABELS.get(sobol_metric, sobol_metric)
    print(f"{label} | evaluaciones: {n_eval} | N_JOBS: {n_jobs}")
    samples = sobol_sample.sample(ctx.problem, sobol_n, calc_second_order=False)

    sim = ctx.simulator
    ms = ctx.mask_sobol

    def eval_one(row: np.ndarray) -> float:
        try:
            q_up_s, q_dn_s = sim.both(row)
            y = sobol_objective_two_stations(
                ctx.q_up,
                ctx.q_obs,
                q_up_s,
                q_dn_s,
                ms,
                sobol_metric,
                weight_up=ctx.weight_up,
                weight_dn=ctx.weight_dn,
            )
            return float(y) if np.isfinite(y) else 1e6
        except Exception:
            return 1e6

    print(f"Corriendo {len(samples)} simulaciones en {n_jobs} núcleos ...")
    if n_jobs == 1:
        y = np.array([eval_one(row) for row in samples])
    else:
        y = np.array(
            Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
                delayed(eval_one)(row) for row in samples
            )
        )
    print(f"Listo. Y: min={np.nanmin(y):.4f}  median={np.nanmedian(y):.4f}  max={np.nanmax(y):.4f}")

    Si = sobol_analyze.analyze(
        ctx.problem, y, calc_second_order=False, conf_level=sobol_conf, print_to_console=False
    )
    sobol_df = pd.DataFrame(
        {
            "parametro": ctx.param_names,
            "S1": Si["S1"],
            "S1_conf": Si["S1_conf"],
            "ST": Si["ST"],
            "ST_conf": Si["ST_conf"],
        }
    )
    sobol_df.to_csv(ctx.data_dir / "sobol_indices.csv", index=False)

    xv = np.arange(len(ctx.param_names))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        xv - bw / 2,
        Si["S1"],
        bw,
        yerr=Si["S1_conf"],
        capsize=4,
        label=f"S1 (IC {sobol_conf:.0%})",
        color="steelblue",
    )
    ax.bar(
        xv + bw / 2,
        Si["ST"],
        bw,
        yerr=Si["ST_conf"],
        capsize=4,
        label=f"ST (IC {sobol_conf:.0%})",
        color="coral",
    )
    ax.set_xticks(xv)
    ax.set_xticklabels(ctx.param_names)
    ax.set_ylabel("Índice de Sobol")
    ax.set_title(
        f"Sensibilidad global — {label} | {ctx.csv_name}\n"
        f"({n_eval} eval., {n_jobs} núcleos, SOBOL_N={sobol_n})"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ctx.fig_dir / "sobol_indices.png", dpi=150)
    plt.show()

    pd.DataFrame(
        {
            "item": ["n_evaluaciones", "SOBOL_N", "SOBOL_METRIC", "conf_level", "N_JOBS"],
            "valor": [len(y), sobol_n, sobol_metric, sobol_conf, n_jobs],
        }
    ).to_csv(ctx.data_dir / "sobol_ejecucion.csv", index=False)

    return y, Si, sobol_df


def write_sobol_conclusions(
    ctx: SensitivityPart2Context,
    *,
    Si: dict[str, Any],
    Y: np.ndarray,
    sobol_metric: str,
    n_jobs: int,
    st_umbral_rel: float,
) -> str:
    st = Si["ST"].copy()
    st_max = float(np.max(st)) if st.size else 0.0
    umbral = st_umbral_rel * st_max if st_max > 0 else 0.0
    rank = np.argsort(-st)
    metric_label = SOBOL_LABELS.get(sobol_metric, sobol_metric)
    lineas = [
        "=== Sensibilidad global (Sobol) ===",
        f"Metrica: {metric_label} | evaluaciones: {len(Y)} | N_JOBS: {n_jobs}",
        f"Umbral poco sensible: ST < {st_umbral_rel:.0%} de max(ST) = {umbral:.4f}",
        "",
    ]
    for r in rank:
        nm = ctx.param_names[r]
        infl = "INFLUYENTE" if st[r] >= umbral else "poco sensible"
        lineas.append(
            f"  {nm}: ST={st[r]:.4f} +/- {Si['ST_conf'][r]:.4f}  "
            f"S1={Si['S1'][r]:.4f}  -> {infl}"
        )
    lineas += [
        "",
        "Para calibracion (Parte 3): priorizar parametros INFLUYENTES;",
        "fijar o acotar fuerte los poco sensibles.",
        "",
        "Nota eta_Q: ST alto indica que la incertidumbre del aforo aguas arriba",
        "condiciona el ajuste; si ST es bajo, fijar eta_Q=1 en calibracion (Parte 3).",
        "Salida: Q_downstream del CSV frente a simulacion en x = L.",
    ]
    texto = "\n".join(lineas)
    print(texto)
    (ctx.data_dir / "sensibilidad_conclusiones.txt").write_text(texto, encoding="utf-8")
    return texto


@dataclass
class OLSBundle:
    params_ols: np.ndarray
    res_fit: np.ndarray
    jac: np.ndarray
    sigma: float
    q_pred_up: np.ndarray
    q_pred: np.ndarray
    res_all: np.ndarray
    cond_j: float
    ols_ok: bool
    n_fit: int
    ci_ols: np.ndarray | None = None
    ci_boot: np.ndarray | None = None
    se: np.ndarray | None = None
    corr: np.ndarray | None = None
    ssc: np.ndarray | None = None


def run_ols_bundle(
    ctx: SensitivityPart2Context,
    *,
    nboot: int,
    n_jobs: int,
    alpha_sig: float,
    dh_ssc: float,
    compute_ssc: bool,
) -> tuple[OLSBundle, pd.DataFrame, pd.DataFrame]:
    p = len(ctx.param_names)
    sim = ctx.simulator
    mask_fit = ctx.mask_fit
    bounds = (np.asarray(ctx.bounds_lo, dtype=float), np.asarray(ctx.bounds_hi, dtype=float))

    wu, wd = ctx.weight_up, ctx.weight_dn
    swu, swd = np.sqrt(wu), np.sqrt(wd)

    def ols_res(pp: np.ndarray) -> np.ndarray:
        q_up_s, q_dn_s = sim.both(pp)
        r_up = ctx.q_up[mask_fit] - q_up_s[mask_fit]
        r_dn = ctx.q_obs[mask_fit] - q_dn_s[mask_fit]
        return np.concatenate([swu * r_up, swd * r_dn])

    q0 = np.asarray(ctx.params_true) * np.array([1.05, 0.98, 1.02, 1.02])
    ols = least_squares(ols_res, q0, bounds=bounds, method="trf")
    q_ols = ols.x
    res_fit = ols.fun
    jac = ols.jac
    n_fit = int(mask_fit.sum())
    sigma = float(np.sqrt((res_fit @ res_fit) / max(n_fit - p, 1)))
    q_pred_up, q_pred = sim.both(q_ols)
    res_all = ctx.q_obs - q_pred
    cond_j = float(np.linalg.cond(jac.T @ jac))
    ols_ok = cond_j < 1e8

    print(f"OLS | cond(J'J)={cond_j:.2e} | sigma={sigma:.4f} | IC OLS fiables={ols_ok}")
    print(dict(zip(ctx.param_names, np.round(q_ols, 5))))

    m_ols = metrics_table_two_stations(ctx, q_pred_up, q_pred)
    m_ols.to_csv(ctx.data_dir / "metricas_ols_cal_val.csv", index=False)

    if ols_ok:
        cov = sigma**2 * np.linalg.pinv(jac.T @ jac + 1e-10 * np.eye(p))
        se_arr = np.sqrt(np.maximum(np.diag(cov), 0))
        tcrit = float(student_t.ppf(1 - alpha_sig / 2, n_fit - p))
        ci_ols = np.column_stack((q_ols - tcrit * se_arr, q_ols + tcrit * se_arr))
        inv_se = np.diag(1 / np.maximum(se_arr, 1e-30))
        corr_mat = inv_se @ cov @ inv_se
    else:
        se_arr = np.full(p, np.nan)
        ci_ols = np.full((p, 2), np.nan)
        corr_mat = np.full((p, p), np.nan)

    ci_boot: np.ndarray | None = None
    if nboot > 0:
        print(f"Bootstrap {nboot} iteraciones (paralelo con {n_jobs} núcleos)...")
        rng = np.random.default_rng(42)
        yhat_up = q_pred_up[mask_fit]
        yhat_dn = q_pred[mask_fit]
        r_up = res_fit[:n_fit] / swu
        r_dn = res_fit[n_fit:] / swd

        def boot_one(seed: int) -> np.ndarray:
            rng_b = np.random.default_rng(seed)
            idx = rng_b.integers(0, n_fit, n_fit)
            y_up_b = yhat_up + swu * r_up[idx]
            y_dn_b = yhat_dn + swd * r_dn[idx]

            def rb(pp: np.ndarray) -> np.ndarray:
                qu, qd = sim.both(pp)
                return np.concatenate(
                    [y_up_b - swu * qu[mask_fit], y_dn_b - swd * qd[mask_fit]]
                )

            return least_squares(rb, q_ols, bounds=bounds, method="trf").x

        seeds = rng.integers(0, 2**31, nboot)
        q_boot = np.array(
            Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(boot_one)(int(s)) for s in seeds
            )
        )
        qs = np.sort(q_boot, 0)
        lb_idx = max(0, int(alpha_sig / 2 * nboot))
        ub_idx = min(nboot - 1, int((1 - alpha_sig / 2) * nboot))
        ci_boot = np.column_stack((qs[lb_idx], qs[ub_idx]))

    tab = []
    for i, nm in enumerate(ctx.param_names):
        row = {
            "parametro": nm,
            "verdadero": ctx.params_true[i],
            "ols": float(q_ols[i]),
            "SE": float(se_arr[i]),
        }
        if ols_ok:
            row["CI_ols_inf"], row["CI_ols_sup"] = float(ci_ols[i, 0]), float(ci_ols[i, 1])
        if ci_boot is not None:
            row["CI_boot_inf"], row["CI_boot_sup"] = float(ci_boot[i, 0]), float(ci_boot[i, 1])
        tab.append(row)
    params_df = pd.DataFrame(tab)
    params_df.to_csv(ctx.data_dir / "parametros_ols_sensibilidad.csv", index=False)

    ssc_mat: np.ndarray | None = None
    if compute_ssc:
        ssc_mat = np.zeros((ctx.nt, p))
        y0 = q_pred.copy()
        for i in range(p):
            pt = q_ols.copy()
            pt[i] *= 1 + dh_ssc
            _, q_pt = sim.both(pt)
            ssc_mat[:, i] = (q_pt - y0) / (dh_ssc * q_ols[i])
        pd.DataFrame(
            {
                "parametro": ctx.param_names,
                "SSC_max_abs": [float(np.max(np.abs(ssc_mat[:, j]))) for j in range(p)],
            }
        ).to_csv(ctx.data_dir / "ssc_resumen.csv", index=False)
    else:
        print("SSC omitido (COMPUTE_SSC=False).")

    R = res_all[mask_fit]
    cross = R[1:] * R[:-1]
    n_cross = int(np.sum(np.sign(cross) < 0))
    min_cross = (len(R) + 1) / 2
    mean_r = float(np.mean(R))
    std_r = float(np.std(R))
    mean_ok = abs(mean_r) < 0.05 * max(std_r, 1e-9)
    uncorr_ok = n_cross >= min_cross

    msgs = [
        "=== 5 suposiciones (periodo calibracion) ===",
        "Sup.1 Aditivos: revisar figura sup01_residuales_vs_prediccion.png",
        f"Sup.2 Media cero: mean(R)={mean_r:.3e} -> {'OK' if mean_ok else 'REVISAR'}",
        "Sup.3 Var constante: revisar sup03_residuales_vs_tiempo.png",
        f"Sup.4 No correlacion: {n_cross} cruces (min {min_cross:.0f}) -> {'OK' if uncorr_ok else 'REVISAR'}",
        "Sup.5 Normalidad: revisar sup05_histograma_residuales.png",
        f"Identificabilidad cond(J'J)={cond_j:.2e} -> {'OK' if ols_ok else 'REVISAR (usar IC bootstrap)'}",
    ]
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    (ctx.data_dir / "suposiciones_errores.txt").write_text("\n".join(msgs), encoding="utf-8")
    print("\n".join(msgs))

    bundle = OLSBundle(
        params_ols=q_ols,
        res_fit=np.asarray(res_fit),
        jac=np.asarray(jac),
        sigma=sigma,
        q_pred_up=q_pred_up,
        q_pred=q_pred,
        res_all=res_all,
        cond_j=cond_j,
        ols_ok=ols_ok,
        n_fit=n_fit,
        ci_ols=ci_ols,
        ci_boot=ci_boot,
        se=se_arr,
        corr=corr_mat,
        ssc=ssc_mat,
    )

    plot_ols_figures(
        ctx,
        bundle,
        cross=cross,
        R=R,
        mean_ok=mean_ok,
        uncorr_ok=uncorr_ok,
        n_cross=n_cross,
        min_cross=min_cross,
    )

    return bundle, m_ols, params_df


def plot_ols_figures(
    ctx: SensitivityPart2Context,
    b: OLSBundle,
    *,
    cross: np.ndarray,
    R: np.ndarray,
    mean_ok: bool,
    uncorr_ok: bool,
    n_cross: int,
    min_cross: float,
) -> None:
    p = len(ctx.param_names)
    ci_ols = b.ci_ols
    ci_boot = b.ci_boot
    corr = b.corr
    ssc_mat = b.ssc

    q_ols = b.params_ols
    q_pred_up = b.q_pred_up
    q_pred = b.q_pred
    t_h = ctx.t_sec / 3600.0
    Yp = q_pred[ctx.mask_fit]

    fig, axes = plt.subplots(2, int(np.ceil(p / 2)), figsize=(11, 7))
    axes = axes.ravel()
    for i, (ax, nm) in enumerate(zip(axes, ctx.param_names)):
        vt, vo = ctx.params_true[i], q_ols[i]
        ax.bar([0], [vt], width=0.35, color="0.75", label="Verdadero")
        ax.plot([1], [vo], "o", color="crimson", ms=10, label="OLS")
        if b.ols_ok and ci_ols is not None and np.all(np.isfinite(ci_ols)):
            ax.errorbar(
                [1],
                [vo],
                yerr=[[vo - ci_ols[i, 0]], [ci_ols[i, 1] - vo]],
                fmt="none",
                color="crimson",
                capsize=4,
            )
        if ci_boot is not None:
            ax.errorbar(
                [1],
                [vo],
                yerr=[[vo - ci_boot[i, 0]], [ci_boot[i, 1] - vo]],
                fmt="none",
                ecolor="navy",
                capsize=3,
                label="IC bootstrap",
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["verd.", "OLS"])
        ax.set_title(nm)
        ax.legend(fontsize=7)
    fig.suptitle("Parámetros: valor sintético vs OLS (periodo calibración)", y=1.02)
    fig.tight_layout()
    fig.savefig(ctx.fig_dir / "parametros_intervalos_confianza.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_h, ctx.q_up, "b.", ms=2, label="Q_obs aguas arriba")
    axes[0].plot(t_h, q_pred_up, "k-", lw=1.2, label="Q_sim OLS (x=0)")
    axes[0].set_ylabel("Q (m³/s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Hidrograma aguas arriba")
    ax = axes[1]
    ax.plot(t_h, ctx.q_obs, "b.", ms=2, label="Q_obs salida")
    ax.plot(t_h, q_pred, "k-", lw=1.2, label="Q_sim OLS (x=L)")
    ax.axvspan(0, ctx.warmup_seconds / 3600, color="gray", alpha=0.12, label="Warm-up")
    ax.axvspan(t_h[ctx.mask_cal][0], t_h[ctx.mask_cal][-1], color="orange", alpha=0.08, label="Calibración")
    ax.legend(fontsize=8)
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Q (m³/s)")
    ax.set_title("Hidrograma salida (OLS)")
    fig.suptitle("Dos estaciones — periodo completo", y=1.01)
    fig.savefig(ctx.fig_dir / "intervalos_confianza_prediccion.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Yp, R, s=12, alpha=0.5, c="steelblue", edgecolors="none")
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Q_sim en calibración (m³/s)")
    ax.set_ylabel("R = Q_obs − Q_sim")
    ax.set_title("Sup.1: nube de residuales (un punto por instante)")
    fig.savefig(ctx.fig_dir / "sup01_residuales_vs_prediccion.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["mean(R)"], [float(np.mean(R))], color="steelblue")
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_title(f"Sup.2 Media cero ({'OK' if mean_ok else 'REVISAR'})")
    axes[1].boxplot(R, vert=True)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_title("Distribución de R")
    fig.savefig(ctx.fig_dir / "sup02_media_error_cero.png", dpi=150)
    plt.show()

    t_cal = ctx.t_sec[ctx.mask_fit]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_cal / 3600, R, s=12, alpha=0.5, c="steelblue", edgecolors="none")
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Tiempo (h) — solo calibración")
    ax.set_ylabel("R (m³/s)")
    ax.set_title("Sup.3: R vs tiempo (buscar embudo / patrón sistemático)")
    fig.savefig(ctx.fig_dir / "sup03_residuales_vs_tiempo.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_cal / 3600, R, "b-")
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_ylabel("R")
    axes[0].set_title("Sup.4 Serie de residuales")
    is_cross = np.sign(cross) < 0
    axes[1].scatter(t_cal[1:] / 3600, cross, c=np.where(is_cross, "green", "gray"), s=14)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("Tiempo (h)")
    axes[1].set_ylabel("R[i]*R[i+1]")
    axes[1].set_title(f"Cruces: {n_cross} / min {min_cross:.0f} ({'OK' if uncorr_ok else 'REVISAR'})")
    fig.savefig(ctx.fig_dir / "sup04_errores_no_correlacionados.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(R, bins=12, density=True, alpha=0.5, edgecolor="k")
    xnorm = np.linspace(float(R.min()), float(R.max()), 100)
    ax.plot(xnorm, norm.pdf(xnorm, 0, b.sigma), "k-", lw=2)
    ax.set_xlabel("R (m³/s)")
    ax.set_title("Sup.5 Normalidad (sigma OLS)")
    fig.savefig(ctx.fig_dir / "sup05_histograma_residuales.png", dpi=150)
    plt.show()

    if corr is not None and np.all(np.isfinite(corr)):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels(ctx.param_names, rotation=45)
        ax.set_yticklabels(ctx.param_names)
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlación parámetros OLS")
        fig.savefig(ctx.fig_dir / "correlacion_parametros.png", dpi=150)
        plt.show()

    if ssc_mat is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_h, q_pred, "k-")
        axes[0].set_ylabel("Q (m³/s)")
        for j, nm in enumerate(ctx.param_names):
            axes[1].plot(t_h, ssc_mat[:, j], "--", label=nm)
        axes[1].legend(fontsize=8)
        axes[1].set_xlabel("Tiempo (h)")
        axes[1].set_title("SSC: sensibilidad local de Q(t) a cada parámetro")
        fig.savefig(ctx.fig_dir / "ssc_hidrograma.png", dpi=150)
        plt.show()


def optional_ydata_profiling(
    ctx: SensitivityPart2Context,
    *,
    q_pred: np.ndarray,
    enabled: bool,
) -> Path | None:
    if not enabled:
        print("Profiling omitido. Activar RUN_PROFILING=True para informe final.")
        return None
    from ydata_profiling import ProfileReport

    cols = ["datetime", "Q_upstream_m3s", "Q_downstream_m3s", "h_outlet_m"]
    cols = [c for c in cols if c in ctx.df.columns]
    prof = ctx.df[cols].copy()
    prof["Q_sim_ols"] = q_pred
    prof["residual"] = ctx.q_obs - prof["Q_sim_ols"]
    prof["periodo"] = np.where(
        ctx.mask_warm,
        "warmup",
        np.where(ctx.mask_cal, "calibracion", "validacion"),
    )
    out = ctx.reports_dir / "profile_latest.html"
    ProfileReport(prof, title="Sensibilidad — serie sintética", minimal=True).to_file(out)
    print("Reporte:", out)
    return out
