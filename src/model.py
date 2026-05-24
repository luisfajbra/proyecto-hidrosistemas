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


def hydraulic_geometry(A: np.ndarray, B_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # Importante porque convierte el area A en las variables hidraulicas h, h_c y R que usa Saint-Venant.
    """Retorna h, h_c y R para canal rectangular."""
    A = np.maximum(A, MIN_AREA)
    h = np.maximum(A / B_w, MIN_DEPTH)
    h_c = 0.5 * h
    per_mojado = B_w + 2.0 * h
    r_hidra = A / per_mojado
    return h, h_c, r_hidra


def friction_slope(Q: np.ndarray, A: np.ndarray, B_w: float, n: float) -> np.ndarray:  # Importante porque calcula Sf, que representa la perdida de energia por friccion de Manning.
    """Pendiente de friccion de Manning en funcion de Q y A."""
    h, hc, R = hydraulic_geometry(A, B_w)
    return n**2 * Q * np.abs(Q) / (np.maximum(A, MIN_AREA) ** 2 * R ** (4.0 / 3.0))


def momentum_flux(Q: np.ndarray, A: np.ndarray, B_w: float, beta: float) -> np.ndarray:  # Importante porque agrupa los terminos de transporte de momentum: inercia beta Q^2/A y presion g h_c A, continuando con el tema de los términso de la ecuacion de SV
    """Flujo conservativo beta Q^2/A + g h_c A."""
    h, h_c, R = hydraulic_geometry(A, B_w)
    A_safe = np.maximum(A, MIN_AREA)
    return beta * Q**2 / A_safe + G * h_c * A_safe


def manning_discharge(h: float, B_w: float, n: float, S0: float) -> float:  # Importante porque calcula el caudal uniforme asociado a un tirante h usando la ecuacion de Manning.
    """Caudal de Manning para un canal rectangular con tirante h."""
    A = B_w * h
    R = A / (B_w + 2.0 * h)
    return (1.0 / n) * A * R ** (2.0 / 3.0) * np.sqrt(S0)


def normal_depth(Q: float, B_w: float, n: float, S0: float) -> float:  # Importante porque da el tirante normal para iniciar el canal y fijar condiciones de frontera estables.
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


def _upstream_hydrograph(t: float, total_time: float, Q0: float, A_hyd: float) -> float:  # Importante porque define el caudal de entrada que genera la onda que viaja por el canal.
    """Hidrograma triangular sintetico: Q0 -> Q0 + A_hyd -> Q0."""
    if total_time <= 0.0:
        return Q0

    tau = t / total_time
    if tau <= 0.5:
        return Q0 + 2.0 * A_hyd * tau
    return Q0 + 2.0 * A_hyd * (1.0 - tau)


def saint_venant_1d(
    params,
    q_upstream=None,
    time_seconds=None,
    L=5000,
    nx=100,
    nt=200,
    dt=1,
    beta=1.0,
    q_lat=None,
    warmup_seconds=0.0,
    return_full=False,
):
    """Resuelve Saint-Venant 1D con MacCormack predictor-corrector.

    Parameters
    ----------
    params : sequence
        [n, S0, Q0, A_hyd, B_w] o [n, S0, B_w]. Con q_upstream externo,
        [n, S0, B_w] usa el primer caudal de entrada como Q0.
    q_upstream : None, float o array-like
        Hidrograma aguas arriba [m3/s]. Si se entrega, el modelo usa esta
        entrada y no calcula el hidrograma triangular interno.
    time_seconds : None o array-like
        Tiempos asociados a q_upstream [s]. Si se entrega, define nt y dt.
    q_lat : None, float o array-like
        Aporte lateral total del tramo [m3/s]. Si es array, debe tener nt valores.
        Internamente se reparte uniformemente como q_lat/L [m2/s].
    warmup_seconds : float
        Se conserva por compatibilidad con notebooks antiguos; run_batch marca
        el warmup en la tabla de salida.
    return_full : bool
        Si es True, devuelve un dict con t, x, A, Q y Q_out (para figuras y verificacion).

    Returns
    -------
    numpy.ndarray o dict
        Por defecto: hidrograma Q(t) en x=L. Con return_full=True: estado completo.
    """
    del warmup_seconds
    q_upstream_series, t_values, nt, dt = _prepare_upstream_series(
        q_upstream, time_seconds, nt, dt
    )
    params_full = _normalize_params(params, q_upstream_series)
    result = _integrate_maccormack(
        params_full,
        L=L,
        nx=nx,
        nt=nt,
        dt=dt,
        beta=beta,
        q_lat=q_lat,
        q_upstream=q_upstream_series,
        t_values=t_values,
    )
    if return_full:
        return result
    return result["Q_out"]


def _normalize_params(params, q_upstream=None) -> list[float]:
    """Acepta [n, S0, Q0, A_hyd, B_w] o la firma antigua [n, S0, B_w]."""
    p = list(map(float, params))
    if len(p) == 5:
        return p
    if len(p) == 3:
        n, S0, B_w = p
        if q_upstream is None:
            Q0 = 50.0
            A_hyd = 100.0
        else:
            q_arr = np.asarray(q_upstream, dtype=float)
            Q0 = float(q_arr[0]) if q_arr.size else 50.0
            A_hyd = 0.0
        return [n, S0, Q0, A_hyd, B_w]
    raise ValueError("params debe ser [n, S0, Q0, A_hyd, B_w] o [n, S0, B_w].")


def _prepare_upstream_series(q_upstream, time_seconds, nt: int, dt: float):
    """Prepara Q aguas arriba externo y la grilla temporal de salida."""
    if q_upstream is None:
        t_values = np.arange(nt, dtype=float) * float(dt)
        return None, t_values, int(nt), float(dt)

    if np.isscalar(q_upstream):
        q_arr = np.full(nt, float(q_upstream), dtype=float)
    else:
        q_arr = np.asarray(q_upstream, dtype=float)
        if q_arr.ndim != 1:
            raise ValueError("q_upstream debe ser None, escalar o un arreglo 1D.")
        nt = len(q_arr)

    if time_seconds is None:
        dt = float(dt)
        t_values = np.arange(nt, dtype=float) * dt
    else:
        t_values = np.asarray(time_seconds, dtype=float)
        if t_values.ndim != 1:
            raise ValueError("time_seconds debe ser None o un arreglo 1D.")
        if len(t_values) != nt:
            raise ValueError(
                f"time_seconds debe tener longitud nt={nt}; recibido {len(t_values)}."
            )
        if nt > 1:
            dt = float(np.median(np.diff(t_values)))
        else:
            dt = float(dt)

    return q_arr, t_values, int(nt), float(dt)


def _upstream_at_time(
    t: float,
    total_time: float,
    Q0: float,
    A_hyd: float,
    q_upstream,
    t_values: np.ndarray,
) -> float:
    if q_upstream is None:
        return _upstream_hydrograph(t, total_time, Q0, A_hyd)
    return float(np.interp(t, t_values, q_upstream, left=q_upstream[0], right=q_upstream[-1]))


def _prepare_lateral_inflow(q_lat, nt: int) -> np.ndarray:
    """Serie temporal del aporte lateral total del tramo [m3/s]."""
    if q_lat is None:
        return np.zeros(nt, dtype=float)

    if np.isscalar(q_lat):
        return np.full(nt, float(q_lat), dtype=float)

    q_lat_arr = np.asarray(q_lat, dtype=float)
    if q_lat_arr.ndim != 1:
        raise ValueError("q_lat debe ser None, escalar o un arreglo 1D.")
    if len(q_lat_arr) != nt:
        raise ValueError(f"q_lat debe tener longitud nt={nt}; recibido {len(q_lat_arr)}.")
    return q_lat_arr


def _integrate_maccormack(
    params,
    L=5000,
    nx=100,
    nt=200,
    dt=1,
    beta=1.0,
    q_lat=None,
    q_upstream=None,
    t_values=None,
):
    """Integracion interna; usada por saint_venant_1d y por scripts de verificacion."""
    n, S0, Q0, A_hyd, B_w = map(float, params)
    dx = L / nx
    x = np.linspace(0.0, L, nx)
    if t_values is None:
        t_values = np.arange(nt, dtype=float) * float(dt)
    else:
        t_values = np.asarray(t_values, dtype=float)
    total_time = max(float(t_values[-1] - t_values[0]) if nt > 1 else 0.0, dt)
    q_lat_total = _prepare_lateral_inflow(q_lat, nt)
    q_lat_unit = q_lat_total / max(float(L), 1e-12)

    h0 = normal_depth(Q0, B_w, n, S0)
    A = np.full(nx, B_w * h0, dtype=float)
    Q = np.full(nx, Q0, dtype=float)
    Q_out = np.zeros(nt, dtype=float)
    t_hist = np.zeros(nt, dtype=float)
    A_hist = np.zeros((nt, nx), dtype=float)
    Q_hist = np.zeros((nt, nx), dtype=float)

    for k in range(nt):
        t = float(t_values[k])
        q_up = _upstream_at_time(t, total_time, Q0, A_hyd, q_upstream, t_values)
        h_up = normal_depth(q_up, B_w, n, S0)
        A[0] = B_w * h_up
        Q[0] = q_up

        h_down = normal_depth(Q[-2], B_w, n, S0)
        A[-1] = B_w * h_down
        Q[-1] = Q[-2]
        Q_out[k] = Q[-1]
        t_hist[k] = t
        A_hist[k] = A
        Q_hist[k] = Q

        if k == nt - 1:
            break

        h, _, _ = hydraulic_geometry(A, B_w)
        celerity = np.sqrt(G * h)
        velocity = Q / np.maximum(A, MIN_AREA)
        max_wave_speed = np.max(np.abs(velocity) + celerity)
        dt_out = max(float(t_values[k + 1] - t_values[k]), 1e-12)
        dt_step = min(dt_out, 0.9 * dx / max(max_wave_speed, 1e-12))

        F_A = Q
        F_Q = momentum_flux(Q, A, B_w, beta)
        Sf = friction_slope(Q, A, B_w, n)
        source_Q = G * A * (S0 - Sf)
        source_A = q_lat_unit[k]

        A_pred = A.copy()
        Q_pred = Q.copy()
        A_pred[:-1] = A[:-1] - dt_step / dx * (F_A[1:] - F_A[:-1]) + dt_step * source_A
        Q_pred[:-1] = Q[:-1] - dt_step / dx * (F_Q[1:] - F_Q[:-1]) + dt_step * source_Q[:-1]

        q_up_next = _upstream_at_time(
            t + dt_step, total_time, Q0, A_hyd, q_upstream, t_values
        )
        A_pred[0] = B_w * normal_depth(q_up_next, B_w, n, S0)
        Q_pred[0] = q_up_next
        A_pred[-1] = B_w * normal_depth(Q_pred[-2], B_w, n, S0)
        Q_pred[-1] = Q_pred[-2]

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
            + dt_step * source_A
        )
        Q_new[1:] = 0.5 * (
            Q[1:]
            + Q_pred[1:]
            - dt_step / dx * (F_Q_pred[1:] - F_Q_pred[:-1])
            + dt_step * source_Q_pred[1:]
        )

        A_new = np.maximum(A_new, B_w * MIN_DEPTH)
        A_new[0] = B_w * normal_depth(q_up_next, B_w, n, S0)
        Q_new[0] = q_up_next
        A_new[-1] = B_w * normal_depth(Q_new[-2], B_w, n, S0)
        Q_new[-1] = Q_new[-2]

        A, Q = A_new, Q_new

    mass_residual = _mass_balance_residual(t_hist, A_hist, Q_hist, x, q_lat_total)

    return {
        "t": t_hist,
        "x": x,
        "A": A_hist,
        "Q": Q_hist,
        "Q_out": Q_out,
        "params": np.array([n, S0, Q0, A_hyd, B_w], dtype=float),
        "mass_balance_residual": mass_residual,
        "q_upstream": np.asarray(q_upstream, dtype=float) if q_upstream is not None else None,
        "q_lat": q_lat_total,
        "q_lat_unit": q_lat_unit,
        "L": L,
        "nx": nx,
        "nt": nt,
        "dt": dt,
        "B_w": B_w,
    }


def _mass_balance_residual(t, A_hist, Q_hist, x, q_lat_total=None):
    """Error relativo mediano entre dV/dt y (Q_entrada - Q_salida)."""
    if hasattr(np, "trapezoid"):
        volume = np.trapezoid(A_hist, x, axis=1)
    else:
        volume = np.trapz(A_hist, x, axis=1)
    q_in = Q_hist[:, 0]
    q_out = Q_hist[:, -1]
    q_lat = _prepare_lateral_inflow(q_lat_total, len(t))
    dvol_dt = np.gradient(volume, t)
    flux_net = q_in - q_out + q_lat
    scale = np.max(np.abs(flux_net)) + 1e-9
    return float(np.median(np.abs(dvol_dt - flux_net) / scale))


def kinematic_wave_speed(q: float, B_w: float, h: float) -> float:
    """Celeridad cinematica ck = (5/3) u para verificacion."""
    a = max(B_w * h, MIN_AREA)
    return (5.0 / 3.0) * (q / a)


def run_batch(
    csv_path,
    params,
    L=5000,
    nx=100,
    beta=1.0,
    warmup_seconds=3600.0,
):
    """
    Ejecuta el solver con un CSV tabular.

    Requiere `datetime` y `Q_upstream_m3s`. Usa `q_lat_m3s` si existe.
    Devuelve un DataFrame con Q_sim_m3s, t_seconds, is_warmup y residuales si
    el CSV trae Q_downstream_m3s.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    if "Q_upstream_m3s" not in df.columns:
        raise ValueError("El CSV debe tener la columna Q_upstream_m3s.")

    if "datetime" in df.columns:
        t_seconds = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy()
    else:
        t_seconds = np.arange(len(df), dtype=float)

    q_upstream = df["Q_upstream_m3s"].to_numpy(dtype=float)
    q_lat = df["q_lat_m3s"].to_numpy(dtype=float) if "q_lat_m3s" in df.columns else None

    full = saint_venant_1d(
        params,
        q_upstream=q_upstream,
        time_seconds=t_seconds,
        L=L,
        nx=nx,
        beta=beta,
        q_lat=q_lat,
        return_full=True,
    )

    out = df.copy()
    out["Q_sim_m3s"] = full["Q_out"]
    out["t_seconds"] = full["t"]
    out["is_warmup"] = out["t_seconds"] < warmup_seconds
    if "Q_downstream_m3s" in out.columns:
        out["residual_m3s"] = out["Q_downstream_m3s"] - out["Q_sim_m3s"]
    out.attrs["model_full"] = full
    out.attrs["mass_balance_residual"] = full["mass_balance_residual"]
    return out
