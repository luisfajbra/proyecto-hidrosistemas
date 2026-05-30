"""
Generación de datos sintéticos — Proyecto Saint-Venant 1D (ICYA 4715)

Produce cinco archivos en data/synthetic/:
  batimetria.csv                — perfiles transversales del lecho
  series_corta_balance.csv      — 500 filas, balance exacto de caudales
  series_corta_ruido.csv        — 500 filas, mediciones con ruido gaussiano
  series_larga_balance.csv      — 20 años, balance exacto de caudales
  series_larga_ruido.csv        — 20 años, mediciones con ruido gaussiano

Usar las series cortas para desarrollar y validar el pipeline; escalar a las
series largas para el análisis final. Todas tienen las mismas columnas que los
datos reales esperados → el reemplazo por datos reales es directo (drop-in).

Ejecutar:  python data/sinteticos.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Parámetros verdaderos del canal ────────────────────────────────────────────
N_MANN = 0.035   # coeficiente de Manning [-]
S0     = 0.001   # pendiente de fondo    [-]
Q0     = 50.0    # caudal base           [m³/s]
A_HYD  = 100.0   # amplitud hidrograma   [m³/s]
B_W    = 50.0    # ancho del canal       [m]
L      = 5000.0  # longitud del canal    [m]
Z_REF  = 200.0   # cota upstream         [m.s.n.m.]

# ── Configuración batimétrica ──────────────────────────────────────────────────
X_STEP_M   = 250   # espaciado entre estaciones longitudinales [m]
Y_STEP_M   = 5     # espaciado entre puntos laterales          [m]
H_BANCO    = 5.0   # altura del coronamiento sobre el talweg   [m]
DZ_PARAB   = 0.5   # variación lateral máxima del lecho        [m]
RIFFLE_AMP = 0.10  # amplitud pool-riffle                      [m]
RIFFLE_L   = 500.0 # espaciado pool-riffle                     [m]
MEANDER_A  = 8.0   # amplitud meandering del talweg            [m]

# ── Hidrología ─────────────────────────────────────────────────────────────────
EVENTS_PER_YEAR = 8    # valor esperado de crecidas/año (Poisson)
NOISE_FRAC      = 0.05 # σ = 5 % del máximo
SEED            = 42
SHORT_SERIES_STEPS = 500


# ── Física auxiliar ────────────────────────────────────────────────────────────

def normal_depth(Q: np.ndarray) -> np.ndarray:
    """Tirante normal de Manning — canal rectangular ancho [m]."""
    return (Q * N_MANN / (B_W * np.sqrt(S0))) ** 0.6


def _compute_travel_time(dt_min: float) -> tuple[int, float]:
    """Tiempo de viaje cinemático: n_shift (pasos) y tau_s (segundos)."""
    h0 = float(normal_depth(np.asarray([Q0]))[0])
    A0 = B_W * h0
    c_k = (5.0 / 3.0) * Q0 / A0   # celeridad cinemática [m/s]
    tau_s = L / c_k                 # tiempo de viaje [s]
    n_shift = max(1, round(tau_s / (dt_min * 60.0)))
    return n_shift, tau_s


def _shift_routing(Q_up: np.ndarray, n_shift: int) -> np.ndarray:
    """Desfase puro: Q_down[t] = Q_up[t - n_shift]."""
    Q_down = np.empty_like(Q_up)
    Q_down[:n_shift] = Q_up[0]
    Q_down[n_shift:] = Q_up[:-n_shift]
    return Q_down


def _muskingum_routing(Q_up: np.ndarray, tau_s: float, X: float, dt_s: float) -> np.ndarray:
    """Tránsito Muskingum lineal con K=tau_s, parámetro de atenuación X, paso dt_s [s]."""
    denom = 2.0 * tau_s * (1.0 - X) + dt_s
    C0 = (dt_s - 2.0 * tau_s * X) / denom
    C1 = (dt_s + 2.0 * tau_s * X) / denom
    C2 = (2.0 * tau_s * (1.0 - X) - dt_s) / denom
    Q_down = np.empty_like(Q_up)
    Q_down[0] = Q_up[0]
    for t in range(1, len(Q_up)):
        Q_down[t] = C0 * Q_up[t] + C1 * Q_up[t - 1] + C2 * Q_down[t - 1]
    return Q_down


def _seasonal_factor(t_days: np.ndarray) -> np.ndarray: #Revisar
    """
    Ciclo estacional bimodal (Andes colombianos).
    Picos ≈ abril (día 100) y octubre (día 283); valles en enero y julio.
    Rango: 0.6–1.4 × Q0.
    """
    doy = t_days % 365.25
    return 1.0 + 0.4 * np.cos(4.0 * np.pi * (doy - 100.0) / 365.25)


def _flood_pulse(n_steps: int, Q_peak: float, Q_base: float) -> np.ndarray:
    """
    Pulso de crecida con forma asimétrica:
    subida rápida en el primer 25 %, recesión exponencial en el 75 % restante.
    """
    t = np.linspace(0.0, 1.0, n_steps)
    t_rise = 0.25
    shape = np.where(
        t <= t_rise,
        t / t_rise,
        np.exp(-4.0 * (t - t_rise) / (1.0 - t_rise)),
    )
    return Q_base + (Q_peak - Q_base) * shape


# ── Batimetría ─────────────────────────────────────────────────────────────────

def generate_bathymetry(rng: np.random.Generator, output_dir: Path) -> pd.DataFrame:
    """
    Genera perfiles transversales del lecho en estaciones cada X_STEP_M metros.

    Características:
    - Pendiente longitudinal S0 con pool-riffle superpuesto
    - Talweg que serpentea lateralmente (meandering)
    - Sección parabólica: más profunda en el talweg
    - Micro-rugosidad aleatoria (~3 cm σ)
    - Coronamiento de berma y lámina de agua de referencia (Q0, Manning)

    Columnas: estacion_x_m, pos_y_m, z_lecho_m, z_banco_m, z_superficie_m
    """
    x_stations = np.arange(0.0, L + 1.0, X_STEP_M)
    y_points   = np.arange(0.0, B_W + 1.0, Y_STEP_M)

    meander_phase = rng.uniform(0.0, 2.0 * np.pi)
    riffle_phase  = rng.uniform(0.0, 2.0 * np.pi)
    h_normal      = float(normal_depth(np.asarray([Q0]))[0])

    rows = []
    for x in x_stations:
        # Talweg longitudinal con pool-riffle
        z_talweg = (
            Z_REF
            - S0 * x
            + RIFFLE_AMP * np.sin(2.0 * np.pi * x / RIFFLE_L + riffle_phase)
        )
        # Posición lateral del talweg (meandering suavizado)
        y_talweg = B_W / 2.0 + MEANDER_A * np.sin(
            2.0 * np.pi * x / (2.0 * L / 3.0) + meander_phase
        )
        y_talweg = float(np.clip(y_talweg, 5.0, B_W - 5.0))

        # Lámina de agua de referencia (condición Q0, Manning)
        z_sup = z_talweg + h_normal

        for y in y_points:
            # Perfil parabólico: profundidad proporcional a distancia al talweg
            dz_lateral = DZ_PARAB * ((y - y_talweg) / (B_W / 2.0)) ** 2
            z_lecho    = z_talweg + dz_lateral + rng.normal(0.0, 0.03)

            rows.append({
                "estacion_x_m":   int(x),
                "pos_y_m":        int(y),
                "z_lecho_m":      round(float(z_lecho), 3),
                "z_banco_m":      round(z_talweg + H_BANCO, 3),
                "z_superficie_m": round(z_sup, 3),
            })

    df = pd.DataFrame(rows)
    path = output_dir / "batimetria.csv"
    df.to_csv(path, index=False)
    print(f"  Guardado: {path}  ({len(df)} filas × {len(df.columns)} columnas)")
    return df


# ── Serie temporal ─────────────────────────────────────────────────────────────

def _build_upstream(n_steps: int, dt_min: float, rng: np.random.Generator) -> np.ndarray:
    """
    Construye Q_upstream en n_steps pasos de dt_min minutos.

    Modelo:  Q = Q0 × estacional(t) × factor_anual(año) + crecidas
    - Ciclo estacional: bimodal (picos abril y octubre)
    - Variabilidad interanual: log-normal (σ = 18 %, rango 0.55–1.55)
    - Crecidas: Poisson(λ=8/año), duración 0.5–5 días, picos log-normales
    """
    dt_days = dt_min / 1440.0
    t_days  = np.arange(n_steps) * dt_days

    n_years     = max(1, int(np.ceil(t_days[-1] / 365.25)))
    year_factor = np.clip(rng.lognormal(0.0, 0.18, n_years), 0.55, 1.55)
    year_idx    = np.clip((t_days / 365.25).astype(int), 0, n_years - 1)

    Q_base = Q0 * _seasonal_factor(t_days) * year_factor[year_idx]
    Q      = Q_base.copy()

    for yr in range(n_years):
        i0 = int(yr * 365.25 / dt_days)
        i1 = min(int((yr + 1) * 365.25 / dt_days), n_steps)
        if i1 - i0 < 2:
            continue

        n_events = int(rng.poisson(EVENTS_PER_YEAR))
        for _ in range(n_events):
            ev_start  = int(rng.integers(i0, i1 - 1))
            dur_steps = max(2, int(rng.uniform(0.5, 5.0) * 1440.0 / dt_min))
            ev_end    = min(ev_start + dur_steps, n_steps)
            n_ev      = ev_end - ev_start
            if n_ev < 2:
                continue

            peak_mult = float(rng.lognormal(np.log(1.5 * year_factor[yr]), 0.4))
            Q_peak    = min(float(Q_base[ev_start]) + A_HYD * peak_mult,
                           Q0 + 3.5 * A_HYD)

            pulse = _flood_pulse(n_ev, Q_peak, float(Q_base[ev_start]))
            Q[ev_start:ev_end] = np.maximum(Q[ev_start:ev_end], pulse)

    return Q



def generate_timeseries(
    n_years: int,
    start_date: str,
    rng: np.random.Generator,
    output_dir: Path,
    filename: str,
    dt_min: float = 15.0,
    n_steps: int | None = None,
    add_noise: bool = False,
) -> pd.DataFrame:
    """
    Genera una serie de n_years años a resolución dt_min minutos.
    Si n_steps se entrega, genera exactamente ese número de filas.

    Flujo de generación:
      Q_upstream = Q_downstream → Manning normal depth → h_outlet

    Si add_noise=True, las señales reciben ruido gaussiano σ = 5 % del máximo
    respectivo. Si add_noise=False, Q_downstream = Q_upstream después del
    redondeo del CSV.

    Columnas: datetime, Q_upstream_m3s, Q_downstream_m3s, h_outlet_m
    """
    freq  = f"{int(dt_min)}min"
    start = pd.Timestamp(start_date)
    if n_steps is None:
        end = start + pd.DateOffset(years=n_years)
        ts = pd.date_range(start, end, freq=freq, inclusive="left")
    else:
        ts = pd.date_range(start, periods=n_steps, freq=freq)
    N     = len(ts)

    # Upstream
    Q_up_true = _build_upstream(N, dt_min, rng)
    Q_down_true = Q_up_true.copy()

    # Tirante en el outlet (Manning)
    h_true = normal_depth(Q_down_true)

    if add_noise:
        sigma_Q = NOISE_FRAC * float(Q_up_true.max())
        sigma_h = NOISE_FRAC * float(h_true.max())

        Q_up_csv = np.maximum(Q_up_true + rng.normal(0.0, sigma_Q, N), 0.0).round(3)
        Q_down_csv = np.maximum(Q_down_true + rng.normal(0.0, sigma_Q, N), 0.0).round(3)
        h_out = np.maximum(h_true + rng.normal(0.0, sigma_h, N), 0.01)
    else:
        Q_up_csv = Q_up_true.round(3)
        Q_down_csv = Q_up_csv.copy()
        h_out = h_true

    df = pd.DataFrame({
        "datetime":           ts,
        "Q_upstream_m3s":     Q_up_csv,
        "Q_downstream_m3s":   Q_down_csv,
        "h_outlet_m":         h_out.round(4),
    })

    path = output_dir / filename
    df.to_csv(path, index=False)
    print(f"  Guardado: {path}  ({len(df):,} filas × {len(df.columns)} columnas)")
    return df


# ── Orquestador ────────────────────────────────────────────────────────────────

def generate(output_dir: str = "data/synthetic") -> None:
    """
    Genera todos los archivos de datos sintéticos.

    Cada componente usa su propia secuencia de semilla derivada de SEED=42,
    por lo que son independientes entre sí (cambiar la batimetría no afecta
    las series temporales y viceversa).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Semillas hijas independientes y reproducibles
    master = np.random.SeedSequence(SEED)
    s_bathy, s_corta_balance, s_corta_ruido, s_larga_balance, s_larga_ruido = master.spawn(5)

    print("--- Batimetria (secciones transversales cada 250 m) ---")
    generate_bathymetry(np.random.default_rng(s_bathy), out)

    print(f"--- Serie corta balance ({SHORT_SERIES_STEPS} filas - 15 min - 2022-01-01) ---")
    generate_timeseries(
        n_years=1, start_date="2022-01-01",
        rng=np.random.default_rng(s_corta_balance),
        output_dir=out, filename="series_corta_balance.csv",
        n_steps=SHORT_SERIES_STEPS,
        add_noise=False,
    )

    print(f"--- Serie corta ruido ({SHORT_SERIES_STEPS} filas - 15 min - 2022-01-01) ---")
    generate_timeseries(
        n_years=1, start_date="2022-01-01",
        rng=np.random.default_rng(s_corta_ruido),
        output_dir=out, filename="series_corta_ruido.csv",
        n_steps=SHORT_SERIES_STEPS,
        add_noise=True,
    )

    print("--- Serie larga balance (20 anios - 15 min - 2000-01-01) ---")
    generate_timeseries(
        n_years=20, start_date="2000-01-01",
        rng=np.random.default_rng(s_larga_balance),
        output_dir=out, filename="series_larga_balance.csv",
        add_noise=False,
    )

    print("--- Serie larga ruido (20 anios - 15 min - 2000-01-01) ---")
    generate_timeseries(
        n_years=20, start_date="2000-01-01",
        rng=np.random.default_rng(s_larga_ruido),
        output_dir=out, filename="series_larga_ruido.csv",
        add_noise=True,
    )

    print(f"\nListo. Archivos en: {out.resolve()}")


if __name__ == "__main__":
    generate()
