"""Parametros verdaderos y semillas (roadmap ICYA 4715)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

RANDOM_SEED = 42
NOISE_SIGMA_FRACTION = 0.05

# Valores "verdaderos" del roadmap
TRUE_N = 0.035
TRUE_S0 = 0.001
TRUE_Q0 = 50.0
TRUE_A_HYD = 100.0
TRUE_B_W = 50.0

# Discretizacion por defecto (roadmap: L=5000, nx=100, dx=50)
DEFAULT_L = 5000.0
DEFAULT_NX = 100
DEFAULT_NT = 200
DEFAULT_DT = 1.0

# Fracciones de L donde se generan Q_obs sinteticos (multi-estacion)
# Mas estaciones => mejor identificabilidad en calibracion (menos equifinalidad)
OBS_X_FRACTIONS: tuple[float, ...] = (0.5, 1.0)


@dataclass(frozen=True)
class TrueParameters:
    n: float
    s0: float
    q0: float
    a_hyd: float
    b_w: float
    L: float
    nx: int
    nt: int
    dt: float
    seed: int

    def as_vector(self) -> list[float]:
        return [self.n, self.s0, self.q0, self.a_hyd, self.b_w]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_true_parameters() -> TrueParameters:
    return TrueParameters(
        n=TRUE_N,
        s0=TRUE_S0,
        q0=TRUE_Q0,
        a_hyd=TRUE_A_HYD,
        b_w=TRUE_B_W,
        L=DEFAULT_L,
        nx=DEFAULT_NX,
        nt=DEFAULT_NT,
        dt=DEFAULT_DT,
        seed=RANDOM_SEED,
    )


def observation_stations(L: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Posiciones x (m) y etiquetas para datos sinteticos / calibracion."""
    L = DEFAULT_L if L is None else L
    xs = np.array([float(f) * L for f in OBS_X_FRACTIONS], dtype=float)
    labels: list[str] = []
    for f in OBS_X_FRACTIONS:
        if f >= 0.999:
            labels.append("x=L")
        elif abs(f - 0.5) < 0.01:
            labels.append("x=L/2")
        else:
            labels.append(f"x={int(round(f * L))} m")
    return xs, labels


def param_bounds() -> dict[str, list[float]]:
    """Rangos del roadmap para Sobol / calibracion."""
    return {
        "n": [0.01, 0.06],
        "S0": [0.0001, 0.005],
        "Q0": [10.0, 100.0],
        "A_hyd": [20.0, 200.0],
        "B_w": [20.0, 80.0],
    }
