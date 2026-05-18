"""Modelo de Saint-Venant 1D para un canal rectangular.

Se usa la forma conservativa vista en clase:

    dA/dt + dQ/dx = 0

    dQ/dt + d/dx(beta Q^2 / A) + d/dx(g h_c A)
        - g A (S0 - Sf) = 0

Para un canal rectangular:

    A = B h
    h_c = h / 2
    Sf = n^2 Q |Q| / (A^2 R^(4/3))
    R = A / P,  P = B + 2h

El solver procesa un hidrograma aguas arriba externo. Ese hidrograma puede
venir de datos sinteticos o reales, siempre que tenga una columna de tiempo y
una columna de caudal.
"""

from __future__ import annotations

import numpy as np


G = 9.81
MIN_DEPTH = 1e-4
MIN_AREA = 1e-6


def hydraulic_geometry(A: np.ndarray, B_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna h, h_c y R para canal rectangular."""
    A = np.maximum(A, MIN_AREA)
    h = np.maximum(A / B_w, MIN_DEPTH)
    h_c = 0.5 * h
    per_mojado = B_w + 2.0 * h
    r_hidra = A / per_mojado
    return h, h_c, r_hidra


def friction_slope(Q: np.ndarray, A: np.ndarray, B_w: float, n: float) -> np.ndarray:
    """Pendiente de friccion de Manning en funcion de Q y A."""
    _, _, R = hydraulic_geometry(A, B_w)
    A_safe = np.maximum(A, MIN_AREA)
    return n**2 * Q * np.abs(Q) / (A_safe**2 * R ** (4.0 / 3.0))


def momentum_flux(Q: np.ndarray, A: np.ndarray, B_w: float, beta: float) -> np.ndarray:
    """Flujo conservativo beta Q^2/A + g h_c A."""
    _, h_c, _ = hydraulic_geometry(A, B_w)
    A_safe = np.maximum(A, MIN_AREA)
    return beta * Q**2 / A_safe + G * h_c * A_safe


def manning_discharge(h: float, B_w: float, n: float, S0: float) -> float:
    """Caudal de Manning para un canal rectangular con tirante h."""
    A = B_w * h
    R = A / (B_w + 2.0 * h)
    return (1.0 / n) * A * R ** (2.0 / 3.0) * np.sqrt(S0)


def normal_depth(Q: float, B_w: float, n: float, S0: float) -> float:
    """Calcula el tirante normal con Manning por biseccion."""
    Q_abs = abs(Q)
    S0 = max(S0, 1e-8)

    lo = MIN_DEPTH
    hi = 1.0
    while manning_discharge(hi, B_w, n, S0) < Q_abs:
        hi *= 2.0
        if hi > 1000.0:
            break

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if manning_discharge(mid, B_w, n, S0) < Q_abs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _parse_params(params: list[float] | tuple[float, ...] | np.ndarray) -> tuple[float, float, float]:
    """Desempaca [n, S0, B_w]."""
    values = list(map(float, params))
    if len(values) != 3:
        raise ValueError("params debe ser [n, S0, B_w].")
    n, S0, B_w = values
    return n, S0, B_w


def _apply_boundaries(
    A: np.ndarray,
    Q: np.ndarray,
    q_up: float,
    B_w: float,
    n: float,
    S0: float,
) -> None:
    """Actualiza condiciones de frontera aguas arriba y aguas abajo."""
    A[0] = B_w * normal_depth(q_up, B_w, n, S0)
    Q[0] = q_up
    A[-1] = B_w * normal_depth(Q[-2], B_w, n, S0)
    Q[-1] = Q[-2]


def _advance_one_step(
    A: np.ndarray,
    Q: np.ndarray,
    dt_step: float,
    dx: float,
    t_next: float,
    time_seconds: np.ndarray,
    q_upstream: np.ndarray,
    B_w: float,
    n: float,
    S0: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Avanza un subpaso MacCormack usando el caudal externo interpolado."""
    F_A = Q
    F_Q = momentum_flux(Q, A, B_w, beta)
    Sf = friction_slope(Q, A, B_w, n)
    source_Q = G * A * (S0 - Sf)

    A_pred = A.copy()
    Q_pred = Q.copy()
    A_pred[:-1] = A[:-1] - dt_step / dx * (F_A[1:] - F_A[:-1])
    Q_pred[:-1] = Q[:-1] - dt_step / dx * (F_Q[1:] - F_Q[:-1]) + dt_step * source_Q[:-1]

    q_up_next = float(np.interp(t_next, time_seconds, q_upstream))
    _apply_boundaries(A_pred, Q_pred, q_up_next, B_w, n, S0)

    F_A_pred = Q_pred
    F_Q_pred = momentum_flux(Q_pred, A_pred, B_w, beta)
    Sf_pred = friction_slope(Q_pred, A_pred, B_w, n)
    source_Q_pred = G * A_pred * (S0 - Sf_pred)

    A_new = A.copy()
    Q_new = Q.copy()
    A_new[1:] = 0.5 * (
        A[1:]
        + A_pred[1:]
        - dt_step / dx * (F_A_pred[1:] - F_A_pred[:-1])
    )
    Q_new[1:] = 0.5 * (
        Q[1:]
        + Q_pred[1:]
        - dt_step / dx * (F_Q_pred[1:] - F_Q_pred[:-1])
        + dt_step * source_Q_pred[1:]
    )

    A_new = np.maximum(A_new, B_w * MIN_DEPTH)
    _apply_boundaries(A_new, Q_new, q_up_next, B_w, n, S0)
    return A_new, Q_new


def _advance_to(
    A: np.ndarray,
    Q: np.ndarray,
    t_current: float,
    t_target: float,
    dx: float,
    time_seconds: np.ndarray,
    q_upstream: np.ndarray,
    B_w: float,
    n: float,
    S0: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Avanza el estado (A, Q) desde t_current hasta t_target con CFL adaptivo."""
    while t_current < t_target:
        h, _, _ = hydraulic_geometry(A, B_w)
        celerity = np.sqrt(G * h)
        velocity = Q / np.maximum(A, MIN_AREA)
        max_wave_speed = np.max(np.abs(velocity) + celerity)
        dt_cfl = 0.9 * dx / max(max_wave_speed, 1e-12)
        dt_step = min(t_target - t_current, dt_cfl)
        t_current += dt_step
        A, Q = _advance_one_step(
            A, Q, dt_step, dx, t_current,
            time_seconds, q_upstream, B_w, n, S0, beta,
        )
    return A, Q


def run_batch(
    csv_path: str,
    params,
    time_col: str = "datetime",
    q_up_col: str = "Q_upstream_m3s",
    q_down_col: str = "Q_downstream_m3s",
    warmup_seconds: float = 3600.0,
    L: float = 5000,
    nx: int = 100,
    beta: float = 1.0,
):
    """Carga un CSV, corre el solver y devuelve un DataFrame con resultados.

    El DataFrame incluye Q_upstream, Q_sim_m3s, t_seconds e is_warmup.
    Si q_down_col existe en el CSV, agrega Q_downstream_m3s y residual_m3s.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df[time_col])
    t_sec = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    q_up = df[q_up_col].to_numpy(dtype=float)

    Q_sim = saint_venant_1d(
        params, q_up, t_sec,
        warmup_seconds=warmup_seconds, L=L, nx=nx, beta=beta,
    )

    result = {
        time_col:   ts.values,
        q_up_col:   q_up,
        "Q_sim_m3s":  Q_sim,
        "t_seconds":  t_sec,
        "is_warmup":  t_sec < (t_sec[0] + warmup_seconds),
    }
    if q_down_col and q_down_col in df.columns:
        q_down = df[q_down_col].to_numpy(dtype=float)
        result[q_down_col]    = q_down
        result["residual_m3s"] = q_down - Q_sim

    return pd.DataFrame(result)


def saint_venant_1d(
    params,
    q_upstream,
    time_seconds=None,
    warmup_seconds=0.0,
    L=5000,
    nx=100,
    beta=1.0,
):
    """Resuelve Saint-Venant 1D con MacCormack predictor-corrector.

    Parameters
    ----------
    params : sequence
        [n, S0, B_w].
    q_upstream : array-like
        Hidrograma observado o sintetico aguas arriba [m3/s].
    time_seconds : array-like, optional
        Tiempos de cada observacion en segundos desde el inicio. Si se omite,
        se asume un paso uniforme de 1 segundo.
    warmup_seconds : float, optional
        Tiempo de calentamiento antes del primer dato. El solver avanza con el
        primer caudal disponible y no devuelve esos pasos. Por defecto es 0.

    Returns
    -------
    numpy.ndarray
        Hidrograma simulado en la frontera aguas abajo, con la misma longitud
        que `q_upstream`.
    """
    n, S0, B_w = _parse_params(params)
    q_upstream = np.asarray(q_upstream, dtype=float)
    if q_upstream.ndim != 1 or len(q_upstream) < 2:
        raise ValueError("q_upstream debe ser una serie 1D con al menos 2 datos.")
    if np.any(~np.isfinite(q_upstream)):
        raise ValueError("q_upstream contiene valores no finitos.")
    q_upstream = np.maximum(q_upstream, 0.0)

    if time_seconds is None:
        time_seconds = np.arange(len(q_upstream), dtype=float)
    else:
        time_seconds = np.asarray(time_seconds, dtype=float)

    if time_seconds.shape != q_upstream.shape:
        raise ValueError("time_seconds y q_upstream deben tener la misma longitud.")
    if np.any(np.diff(time_seconds) <= 0.0):
        raise ValueError("time_seconds debe ser estrictamente creciente.")
    warmup_seconds = float(warmup_seconds)
    if warmup_seconds < 0.0:
        raise ValueError("warmup_seconds debe ser mayor o igual a 0.")

    dx = L / nx
    nt = len(q_upstream)

    h0 = normal_depth(float(q_upstream[0]), B_w, n, S0)
    A = np.full(nx, B_w * h0, dtype=float)
    Q = np.full(nx, float(q_upstream[0]), dtype=float)
    Q_out = np.zeros(nt, dtype=float)
    _apply_boundaries(A, Q, float(q_upstream[0]), B_w, n, S0)

    if warmup_seconds > 0.0:
        A, Q = _advance_to(
            A, Q,
            float(time_seconds[0]) - warmup_seconds,
            float(time_seconds[0]),
            dx, time_seconds, q_upstream, B_w, n, S0, beta,
        )

    for k in range(nt):
        _apply_boundaries(A, Q, float(q_upstream[k]), B_w, n, S0)
        Q_out[k] = Q[-1]

        if k == nt - 1:
            break

        A, Q = _advance_to(
            A, Q,
            float(time_seconds[k]),
            float(time_seconds[k + 1]),
            dx, time_seconds, q_upstream, B_w, n, S0, beta,
        )

    return Q_out


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/series_corta.csv"
    resultado = run_batch(csv, [0.035, 0.001, 50.0])

    eval_df = resultado[~resultado["is_warmup"]]
    obs = eval_df["Q_downstream_m3s"].to_numpy()
    sim = eval_df["Q_sim_m3s"].to_numpy()

    mae  = float(np.mean(np.abs(obs - sim)))
    rmse = float(np.sqrt(np.mean((obs - sim) ** 2)))
    bias = float(np.mean(sim - obs))
    r    = float(np.corrcoef(obs, sim)[0, 1])
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    nse  = float(1.0 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else float("nan")

    print(f"Archivo : {csv}")
    print(f"Filas   : {len(resultado):,}  (warmup: {resultado['is_warmup'].sum()})")
    print(f"MAE     : {mae:.3f} m³/s")
    print(f"RMSE    : {rmse:.3f} m³/s")
    print(f"Bias    : {bias:.3f} m³/s")
    print(f"r       : {r:.4f}")
    print(f"NSE     : {nse:.4f}")
