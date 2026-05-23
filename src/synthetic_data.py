"""Datos sinteticos: simulacion con verdad conocida + ruido gaussiano."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import NOISE_SIGMA_FRACTION, RANDOM_SEED, default_true_parameters, observation_stations
from .model import saint_venant_1d


@dataclass
class SyntheticDataset:
    """Hidrogramas observados en una o mas secciones a lo largo del canal."""

    t: np.ndarray
    x_obs: np.ndarray
    station_labels: list[str]
    q_true: np.ndarray
    q_obs: np.ndarray
    sigma: float
    params_true: list[float]
    seed: int

    @property
    def n_stations(self) -> int:
        return int(self.q_obs.shape[0])

    @property
    def nt(self) -> int:
        return int(self.q_obs.shape[1])

    def y_obs_vector(self) -> np.ndarray:
        """Vector apilado [estacion0 @ todos t, estacion1 @ todos t, ...] para OLS."""
        return np.asarray(self.q_obs, dtype=float).ravel()

    def q_true_downstream(self) -> np.ndarray:
        return np.asarray(self.q_true[-1], dtype=float)

    def q_obs_downstream(self) -> np.ndarray:
        return np.asarray(self.q_obs[-1], dtype=float)


def extract_q_at_x(
    Q_hist: np.ndarray,
    x_grid: np.ndarray,
    x_target: float,
) -> np.ndarray:
    """Q(t) en la celda de malla mas cercana a x_target. Q_hist shape (nt, nx)."""
    ix = int(np.argmin(np.abs(x_grid - x_target)))
    return np.asarray(Q_hist[:, ix], dtype=float)


def simulate_stations(
    params: list[float] | np.ndarray,
    x_obs: np.ndarray,
    L: float,
    nx: int,
    nt: int,
    dt: float,
) -> np.ndarray:
    """
    Matriz (n_stations, nt) de caudales simulados en las secciones x_obs.
    """
    full = saint_venant_1d(
        params, L=L, nx=nx, nt=nt, dt=dt, return_full=True
    )
    Q_hist = full["Q"]
    x_grid = full["x"]
    rows = [extract_q_at_x(Q_hist, x_grid, float(xo)) for xo in x_obs]
    return np.vstack(rows)


def generate_synthetic_data(seed: int = RANDOM_SEED) -> SyntheticDataset:
    """
    Simula con parametros verdaderos, extrae Q(t) en varias x (config OBS_X_FRACTIONS)
    y anade ruido gaussiano sigma = 5% * max(Q) sobre todas las estaciones.
    """
    tp = default_true_parameters()
    params = tp.as_vector()
    x_obs, labels = observation_stations(tp.L)
    q_true = simulate_stations(params, x_obs, tp.L, tp.nx, tp.nt, tp.dt)
    t = np.arange(tp.nt) * tp.dt
    q_max = float(np.max(q_true))
    sigma = NOISE_SIGMA_FRACTION * q_max
    rng = np.random.default_rng(seed)
    q_obs = q_true + rng.normal(0.0, sigma, size=q_true.shape)

    return SyntheticDataset(
        t=t,
        x_obs=x_obs,
        station_labels=labels,
        q_true=q_true,
        q_obs=q_obs,
        sigma=sigma,
        params_true=params,
        seed=seed,
    )


def save_synthetic_dataset(dataset: SyntheticDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "hidrograma_sintetico.npz",
        t=dataset.t,
        x_obs=dataset.x_obs,
        station_labels=np.array(dataset.station_labels, dtype=object),
        q_obs_stations=dataset.q_obs,
        q_true_stations=dataset.q_true,
        q_obs=dataset.q_obs_downstream(),
        q_true=dataset.q_true_downstream(),
        sigma=dataset.sigma,
        params_true=np.array(dataset.params_true),
        seed=dataset.seed,
        n_stations=dataset.n_stations,
    )


def load_synthetic_dataset(path: Path) -> SyntheticDataset:
    """Carga NPZ multi-estacion o formato antiguo (solo x=L)."""
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d["t"], dtype=float)
    sigma = float(d["sigma"])
    params_true = list(np.asarray(d["params_true"], dtype=float))
    seed = int(d["seed"])

    if "q_obs_stations" in d.files:
        q_obs = np.asarray(d["q_obs_stations"], dtype=float)
        q_true = np.asarray(d["q_true_stations"], dtype=float)
        x_obs = np.asarray(d["x_obs"], dtype=float)
        labels = [str(x) for x in d["station_labels"].tolist()]
        return SyntheticDataset(
            t=t,
            x_obs=x_obs,
            station_labels=labels,
            q_true=q_true,
            q_obs=q_obs,
            sigma=sigma,
            params_true=params_true,
            seed=seed,
        )

    q_obs_1d = np.asarray(d["q_obs"], dtype=float)
    q_true_1d = np.asarray(d["q_true"], dtype=float)
    L = default_true_parameters().L
    x_obs = np.array([L], dtype=float)
    labels = ["x=L"]
    return SyntheticDataset(
        t=t,
        x_obs=x_obs,
        station_labels=labels,
        q_true=q_true_1d.reshape(1, -1),
        q_obs=q_obs_1d.reshape(1, -1),
        sigma=sigma,
        params_true=params_true,
        seed=seed,
    )
