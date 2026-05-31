"""PINN para estimación de parámetros del modelo Saint-Venant 1D.

Red MLP f(x_norm, t_norm) -> (A, Q) con activación sigmoid en capas ocultas
y softplus en la capa de salida (garantiza A > 0, Q > 0).
Los parámetros físicos n y B_w se almacenan en log-espacio como nn.Parameter.

Para cambiar qué parámetros se estiman, modificar ESTIMATE_PARAMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

G = 9.81
MIN_AREA = 1e-4

# ── Configuración de estimación ───────────────────────────────────────────────
# Cambiar True/False para incluir o excluir parámetros del entrenamiento.
ESTIMATE_PARAMS: dict[str, bool] = {
    "n": True,
    "Bw": True,
}


# ── Modelo ────────────────────────────────────────────────────────────────────

class SVPINN(nn.Module):
    """MLP espacio-temporal + parámetros físicos estimables.

    Entradas: (x_norm, t_norm) ∈ [0,1]².
    Salidas:  (A [m²], Q [m³/s]) en cada punto.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        n_layers: int = 4,
        S0: float = 0.001,
        beta: float = 1.0,
        n_init: float = 0.030,
        Bw_init: float = 45.0,
        estimate_params: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        ep = ESTIMATE_PARAMS if estimate_params is None else estimate_params

        # MLP: 2 → [hidden_size]*n_layers → 2
        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.Sigmoid()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

        # Parámetros estimables en log-espacio (garantiza positividad)
        self._log_n = nn.Parameter(
            torch.log(torch.tensor(float(n_init))),
            requires_grad=bool(ep.get("n", False)),
        )
        self._log_Bw = nn.Parameter(
            torch.log(torch.tensor(float(Bw_init))),
            requires_grad=bool(ep.get("Bw", False)),
        )

        # Parámetros fijos (sin gradiente)
        self.register_buffer("S0",   torch.tensor(float(S0)))
        self.register_buffer("beta", torch.tensor(float(beta)))

    @property
    def n(self) -> torch.Tensor:
        return torch.exp(self._log_n)

    @property
    def Bw(self) -> torch.Tensor:
        return torch.exp(self._log_Bw)

    def forward(
        self, x_norm: torch.Tensor, t_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.stack([x_norm, t_norm], dim=-1)
        out = self.net(inp)
        # softplus garantiza A > MIN_AREA y Q > MIN_AREA en todo momento
        A = nn.functional.softplus(out[..., 0]) + MIN_AREA
        Q = nn.functional.softplus(out[..., 1]) + MIN_AREA
        return A, Q
