"""---

## ⚙️ El Sistema: Saint-Venant 1D

### Ecuaciones gobernantes

Canal rectangular, flujo 1D no permanente:

```
∂h/∂t  +  ∂(hu)/∂x  =  0                              (continuidad)

∂(hu)/∂t  +  ∂(hu² + gh²/2)/∂x  =  -gh(∂B/∂x + Sf)  (momentum)
```

Donde la fricción de Manning es:

```
Sf = n² · u|u| / h^(4/3)
```

### Parámetros calibrables

| Parámetro | Símbolo | Rango de búsqueda | Valor "verdadero" |
|---|---|---|---|
| Coeficiente de Manning | `n` | 0.010 – 0.060 | **0.035** |
| Pendiente de fondo | `S₀` | 0.0001 – 0.005 | **0.001** |
| Caudal base upstream | `Q₀` | 10 – 100 m³/s | **50 m³/s** |
| Amplitud del hidrograma | `A_hyd` | 20 – 200 m³/s | **100 m³/s** |
| Ancho del canal | `B_w` | 20 – 80 m | **50 m** |

### Esquema numérico: MacCormack (predictor-corrector)

- 2do orden en espacio y tiempo
- Condición de estabilidad CFL: `dt ≤ 0.9 · dx / (|u| + √(gh))`
- Dominio: L = 5000 m, nx = 100 celdas, dx = 50 m
- Condición upstream: hidrograma triangular sintético
- Condición downstream: tirante normal (Manning)

### Verificación del modelo ⚠️

Antes de calibrar, verificar con:
- [ ] Solución analítica de onda cinemática en régimen subcrítico
- [ ] Conservación de masa en cada paso de tiempo
- [ ] Caso límite: flujo uniforme permanente → `h` constante

> **Nota:** Es muy común calibrar un modelo mal implementado y no enterarse. ¡La verificación no es opcional!

---
"""
"""

## 📊 Parte 1 — Modelo Numérico *(15 pts)*

**Responsable principal:** Persona 1 + Persona 2  
**Duración estimada:** Semana 1

### Checklist

- [ ] Implementar solver `saint_venant_1d(params)` en `src/model.py`
- [ ] Generación de datos sintéticos con ruido gaussiano (`σ = 5% · Q_max`)
- [ ] Modo batch funcional (sin GUI, ejecutable desde terminal)
- [ ] Paralelización con `joblib` para Monte Carlo masivo
- [ ] Verificación contra solución analítica y conservación de masa
- [ ] Figura: hidrograma simulado vs. "observado" sintético
- [ ] Tabla: parámetros verdaderos y condiciones iniciales/frontera

```python
# Llamada mínima esperada
Q_sim = saint_venant_1d(params=[n, S0, Q0, A_hyd, B_w])
```

---"""

#Modelo numérico de Saint-Venant 1D para flujo en canal rectangular usando la FORMA CONSERVATIDA


"""Canal rectangular, flujo 1D no permanente:

∂h/∂t  +  ∂(hu)/∂x  =  0 (continuidad)

∂(hu)/∂t  +  ∂(hu² + gh²/2)/∂x  =  -gh(∂B/∂x + Sf)  (momentum)"""
import numpy as np
def saint_venant_1d(params, L=5000, nx=100, nt=200, dt=1):
    n, S0, Q0, A_hyd, B_w = params
    dx = L / nx
    g = 9.81

    # Inicialización de variables
    h = np.zeros(nx)  # Tirante
    u = np.zeros(nx)  # Velocidad
    Q = np.zeros((nt, nx))  # Caudal simulado

    # Condición inicial: flujo uniforme permanente
    h[:] = (Q0 / (B_w * np.sqrt(g * h))) ** (3/5)  # Tirante inicial
    u[:] = Q0 / (B_w * h)  # Velocidad inicial

    for t in range(nt):
        # Predictor (MacCormack)
        h_pred = np.copy(h)
        u_pred = np.copy(u)

        for i in range(1, nx-1):
            # Continuidad
            h_pred[i] = h[i] - dt/dx * (h[i]*u[i] - h[i-1]*u[i-1])
            # Momentum
            u_pred[i] = u[i] - dt/dx * (u[i]**2 + g*h[i]/2 - u[i-1]**2 - g*h[i-1]/2)