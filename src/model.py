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

La funcion `saint_venant_1d` devuelve el hidrograma de salida Q(t, x=L).
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


def saint_venant_1d(params, L=5000, nx=100, nt=200, dt=1, beta=1.0):  # Importante porque es el solver principal que integra continuidad y momentum para obtener el hidrograma de salida.
    """Resuelve Saint-Venant 1D con MacCormack predictor-corrector.

    Parameters
    ----------
    params : sequence
        [n, S0, Q0, A_hyd, B_w], donde n es Manning, S0 la pendiente de
        fondo, Q0 el caudal base, A_hyd la amplitud del hidrograma y B_w
        el ancho rectangular.

    Returns
    -------
    numpy.ndarray
        Hidrograma simulado en la frontera aguas abajo, con longitud `nt`.
    """
    n, S0, Q0, A_hyd, B_w = map(float, params)
    dx = L / nx
    total_time = max((nt - 1) * dt, dt)

    h0 = normal_depth(Q0, B_w, n, S0)
    A = np.full(nx, B_w * h0, dtype=float)
    Q = np.full(nx, Q0, dtype=float)
    Q_out = np.zeros(nt, dtype=float)

    for k in range(nt):
        t = k * dt
        q_up = _upstream_hydrograph(t, total_time, Q0, A_hyd)
        h_up = normal_depth(q_up, B_w, n, S0)
        A[0] = B_w * h_up
        Q[0] = q_up

        h_down = normal_depth(Q[-2], B_w, n, S0)
        A[-1] = B_w * h_down
        Q[-1] = Q[-2]
        Q_out[k] = Q[-1]

        if k == nt - 1:
            break

        h, _, _ = hydraulic_geometry(A, B_w)
        celerity = np.sqrt(G * h)
        velocity = Q / np.maximum(A, MIN_AREA)
        max_wave_speed = np.max(np.abs(velocity) + celerity)
        dt_step = min(dt, 0.9 * dx / max(max_wave_speed, 1e-12))

        F_A = Q
        F_Q = momentum_flux(Q, A, B_w, beta)
        Sf = friction_slope(Q, A, B_w, n)
        source_Q = G * A * (S0 - Sf)

        A_pred = A.copy()
        Q_pred = Q.copy()
        A_pred[:-1] = A[:-1] - dt_step / dx * (F_A[1:] - F_A[:-1])
        Q_pred[:-1] = Q[:-1] - dt_step / dx * (F_Q[1:] - F_Q[:-1]) + dt_step * source_Q[:-1]

        q_up_next = _upstream_hydrograph(t + dt_step, total_time, Q0, A_hyd)
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
            A[1:] + A_pred[1:] - dt_step / dx * (F_A_pred[1:] - F_A_pred[:-1])
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

    return Q_out
