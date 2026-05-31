# Proyecto Saint-Venant 1D — ICYA 4715

Modelación hidráulica 1D del canal rectangular usando las ecuaciones de Saint-Venant,
implementada para el curso ICYA 4715 (Hidrosistemas). El proyecto cubre verificación
del solver, análisis de sensibilidad, calibración y estimación de parámetros con PINN.

## Estructura

| Notebook | Descripción |
|----------|-------------|
| `01_model_verification.ipynb` | Verificación del solver MacCormack: propagación de hidrograma, conservación de masa, demo de paralelismo |
| `02_sensitivity.ipynb` | Sensibilidad global (Sobol) y OLS para los parámetros n, S0, B_w, η_Q |
| `03_calibration.ipynb` | Calibración Monte Carlo / GLUE del modelo Saint-Venant |
| `04_pinn.ipynb` | Estimación de parámetros con PINN (Physics-Informed Neural Network) |

## Datos sintéticos

Los datos de entrada se generan con `data/sinteticos.py`. Parámetros verdaderos del canal:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| n | 0.035 | Coeficiente de Manning |
| S₀ | 0.001 | Pendiente de fondo |
| Q₀ | 50.0 m³/s | Caudal base |
| B_w | 50.0 m | Ancho del canal |
| L | 5 000 m | Longitud del canal |

## Solver

`src/model.py` implementa Saint-Venant 1D conservativo con esquema MacCormack
predictor-corrector. El sub-stepping CFL garantiza estabilidad numérica: cada
intervalo de salida (dt=15 min) se avanza en múltiples pasos internos con
dt_CFL ≈ dx / (|u| + c).

## PINN — Decisiones de diseño y lecciones aprendidas

### Primera implementación (fallida)

La primera versión usaba:
- Activación **sigmoid** en capas ocultas
- Función de pérdida: L_total = L_data + 0.05 · L_pde
- Sin normalización de Q
- Sin curriculum (datos y física desde el inicio)

**Resultados:** La PINN convergió a flujo uniforme constante (Q ≈ 100 m³/s,
línea horizontal), estimando n = 0.011 y B_w = 18 m (valores verdaderos: 0.035 y 50 m).

### Diagnóstico

Tres causas apiladas:

1. **Desbalance de escala en la pérdida.** Q_obs ∈ [40, 370] m³/s → L_data inicial ≈ 10⁴.
   L_pde inicial ≈ 1. El gradiente de L_pde minimizaba trivialmente: Q = constante
   satisface ∂Q/∂x = 0 y ∂A/∂t = 0 (flujo uniforme), haciendo L_pde → 0 sin que la
   red aprenda la dinámica temporal. L_data nunca bajaba porque la solución constante
   no ajusta los bordes.

2. **Saturación por sigmoid.** Con 4 capas ocultas y activación sigmoid, las capas
   intermedias saturaban hacia valores constantes antes de aprender variaciones
   espacio-temporales. Tanh es zero-centered y tiene gradiente más robusto para PINNs.

3. **Sin curriculum.** Introducir la restricción física desde el primer epoch empuja
   al optimizador hacia la solución trivial (flujo uniforme) antes de que la red haya
   aprendido la señal temporal de los datos.

### Fixes aplicados

| Fix | Cambio | Razón |
|-----|--------|-------|
| Activación | sigmoid → tanh en capas ocultas | Evita saturación; gradiente más robusto |
| Normalización | MSE((Q_pred - Q_obs) / Q_max)² | Escala L_data de 10⁴ a ~1 |
| Curriculum | n_epochs_warmup=2000: primeras 2000 épocas sin L_pde | Red aprende señal temporal antes de aplicar física |

Configuración final del notebook: N_EPOCHS_ADAM=6000, N_EPOCHS_WARMUP=2000,
N_ITER_LBFGS=500, LAMBDA_PDE=0.05.

## Instalación

```bash
pip install -r requirements.txt   # incluye torch>=2.0
```
