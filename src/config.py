"""Canonical configuration for the Saint-Venant 1D project.

This module concentrates user/developer-defined problem parameters so notebooks
and modules do not invent values independently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalParams:
    """Physical channel parameters and base synthetic hydrograph values."""

    n_manning: float = 0.035
    bed_slope: float = 0.001
    base_flow: float = 50.0
    hydrograph_amplitude: float = 100.0
    channel_width: float = 50.0
    channel_length: float = 2000.0
    upstream_elevation: float = 200.0


@dataclass(frozen=True)
class BathymetryParams:
    """Geometric parameters used to generate synthetic bathymetry."""

    x_step_m: int = 250
    y_step_m: int = 5
    bank_height: float = 5.0
    parabola_dz: float = 0.5
    riffle_amp: float = 0.10
    riffle_length: float = 500.0
    meander_amplitude: float = 8.0


@dataclass(frozen=True)
class SyntheticHydrologyParams:
    """Defaults for synthetic discharge and noise generation."""

    events_per_year: int = 8
    noise_frac: float = 0.05
    seed: int = 42
    short_series_steps: int = 500
    dt_min: float = 15.0
    muskingum_x: float = 0.1


@dataclass(frozen=True)
class NumericalDefaults:
    """Numerical defaults shared by the solver and tabular runs."""

    nx: int = 100
    nt: int = 200
    dt: float = 1.0
    beta: float = 1.0
    warmup_seconds: float = 3600.0


@dataclass(frozen=True)
class PinnDefaults:
    """Architecture and training defaults used by the PINN."""

    hidden_size: int = 64
    n_layers: int = 4
    beta: float = 1.0
    n_init: float = 0.030
    bw_init: float = 45.0
    lambda_data: float = 1.0
    lambda_pde: float = 0.05
    n_epochs_adam: int = 6000
    n_epochs_warmup: int = 2000
    n_epochs_ramp: int = 1000
    n_iter_lbfgs: int = 500
    n_colloc: int = 2000
    resample_every: int = 1000
    gradient_clip_max_norm: float = 1.0
    # Nuevos: términos de pérdida adicionales
    lambda_h: float = 1.0  # peso L_h (profundidad observada en x=L)
    lambda_steady: float = 0.1  # peso L_steady (residuos PDE en t=0)
    n_colloc_steady: int = 200  # puntos de colocación en t=0


DEFAULT_PHYSICAL = PhysicalParams()
DEFAULT_BATHYMETRY = BathymetryParams()
DEFAULT_SYNTHETIC_HYDROLOGY = SyntheticHydrologyParams()
DEFAULT_NUMERICAL = NumericalDefaults()
DEFAULT_PINN = PinnDefaults()
