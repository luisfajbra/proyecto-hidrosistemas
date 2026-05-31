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


# ── Residuos de la PDE ────────────────────────────────────────────────────────

def pde_residuals(
    model: SVPINN,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    L: float,
    T: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuos de Saint-Venant en puntos de colocación (x_col [m], t_col [s]).

    Usa torch.autograd.grad con create_graph=True para que los gradientes
    fluyan hasta n y Bw durante el entrenamiento.

    Returns: (R_mass, R_mom) — shape (N,) cada uno.
    """
    # Normalizar y crear hojas de autograd para derivadas espaciales/temporales
    x_norm = (x_col / L).clone().detach().requires_grad_(True)
    t_norm = (t_col / T).clone().detach().requires_grad_(True)

    A, Q = model(x_norm, t_norm)

    Bw   = model.Bw
    n    = model.n
    S0   = model.S0
    beta = model.beta

    # Geometría hidráulica
    h     = torch.clamp(A / Bw, min=1e-4)
    h_c   = 0.5 * h
    per   = Bw + 2.0 * h
    R_hyd = A / per                          # radio hidráulico [m]

    # Flujo de momentum: F_Q = β Q²/A + g h_c A
    F_Q = beta * Q**2 / (A + MIN_AREA) + G * h_c * A

    # Pendiente de fricción de Manning: Sf = n² Q|Q| / (A² R^(4/3))
    Sf = n**2 * Q * torch.abs(Q) / ((A + MIN_AREA)**2 * R_hyd**(4.0 / 3.0))

    # Derivadas por autograd (regla de la cadena: ∂A/∂t = (∂A/∂t_norm)/T)
    ones     = torch.ones_like(A)
    dA_dtn   = torch.autograd.grad(A,   t_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dxn   = torch.autograd.grad(Q,   x_norm, grad_outputs=ones, create_graph=True)[0]
    dFQ_dxn  = torch.autograd.grad(F_Q, x_norm, grad_outputs=ones, create_graph=True)[0]
    dQ_dtn   = torch.autograd.grad(Q,   t_norm, grad_outputs=ones, create_graph=True)[0]

    # Residuos de continuidad y momentum
    R_mass = dA_dtn / T + dQ_dxn / L
    R_mom  = dQ_dtn / T + dFQ_dxn / L - G * A * (S0 - Sf)

    return R_mass, R_mom


# ── Entrenamiento ─────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model: SVPINN
    loss_history: list[dict[str, Any]]
    n_estimate: float
    Bw_estimate: float


def train(
    model: SVPINN,
    *,
    x0_data: torch.Tensor,
    xL_data: torch.Tensor,
    t_data: torch.Tensor,
    L: float,
    T: float,
    lambda_data: float = 1.0,
    lambda_pde: float = 0.05,
    n_epochs_adam: int = 5000,
    n_iter_lbfgs: int = 500,
    n_colloc: int = 2000,
    resample_every: int = 1000,
    t_warmup: float = 3600.0,
    verbose_every: int = 500,
) -> TrainResult:
    """Entrena la PINN con Adam (exploración) y L-BFGS (convergencia fina).

    x0_data: Q observado en x=0 [m³/s], shape (nt,)
    xL_data: Q observado en x=L [m³/s], shape (nt,)
    t_data:  tiempos [s],              shape (nt,)
    L, T:    longitud del canal [m] y duración [s]
    """
    x0_data = x0_data.float()
    xL_data = xL_data.float()
    t_data  = t_data.float()

    post_warm  = t_data >= t_warmup
    t_post     = t_data[post_warm]
    T_total    = float(t_data[-1]) if T <= 0 else float(T)

    loss_history: list[dict[str, Any]] = []

    def _sample_colloc(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        x_c = torch.rand(n_colloc) * L
        t_c = torch.rand(n_colloc) * (T_total - max(t_warmup, 0.0)) + max(t_warmup, 0.0)
        return x_c, t_c

    def _compute_loss(
        x_c: torch.Tensor, t_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pérdida de datos en los dos bordes (x=0 y x=L)
        tn = t_post / T_total
        _, Q0_pred = model(torch.zeros_like(tn), tn)
        _, QL_pred = model(torch.ones_like(tn),  tn)
        Q0_obs = x0_data[post_warm]
        QL_obs = xL_data[post_warm]
        l_data = (
            torch.mean((Q0_pred - Q0_obs) ** 2)
            + torch.mean((QL_pred - QL_obs) ** 2)
        )

        # Pérdida de física en puntos de colocación interiores
        R_mass, R_mom = pde_residuals(model, x_c, t_c, L=L, T=T_total)
        l_pde = torch.mean(R_mass ** 2) + torch.mean(R_mom ** 2)

        l_total = lambda_data * l_data + lambda_pde * l_pde
        return l_total, l_data, l_pde

    # ── Fase 1: Adam ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_c, t_c = _sample_colloc(seed=0)

    for epoch in range(n_epochs_adam):
        if epoch > 0 and epoch % resample_every == 0:
            x_c, t_c = _sample_colloc(seed=epoch)
        optimizer.zero_grad()
        l_total, l_data, l_pde = _compute_loss(x_c, t_c)
        l_total.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == n_epochs_adam - 1:
            entry: dict[str, Any] = {
                "epoch": epoch,
                "total": float(l_total),
                "data":  float(l_data),
                "pde":   float(l_pde),
                "n":     float(model.n),
                "Bw":    float(model.Bw),
            }
            loss_history.append(entry)
            print(
                f"[Adam {epoch:5d}] "
                f"total={l_total:.4e}  data={l_data:.4e}  pde={l_pde:.4e}  "
                f"n={model.n:.5f}  Bw={model.Bw:.3f}"
            )

    # ── Fase 2: L-BFGS ───────────────────────────────────────────────────────
    x_c, t_c = _sample_colloc(seed=9999)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=n_iter_lbfgs, history_size=50
    )

    def _closure() -> torch.Tensor:
        lbfgs.zero_grad()
        l_total, _, _ = _compute_loss(x_c, t_c)
        l_total.backward()
        return l_total

    l_final = lbfgs.step(_closure)
    loss_history.append({
        "epoch": n_epochs_adam,
        "total": float(l_final) if l_final is not None else float("nan"),
        "n":  float(model.n),
        "Bw": float(model.Bw),
        "phase": "lbfgs",
    })
    print(f"[L-BFGS] n={model.n:.5f}  Bw={model.Bw:.3f}")

    return TrainResult(
        model=model,
        loss_history=loss_history,
        n_estimate=float(model.n),
        Bw_estimate=float(model.Bw),
    )
