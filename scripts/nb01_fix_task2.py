import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Actualizar markdown seccion hidrograma (celda 8 -> pasara a posicion 6)
cells[8]['source'] = [
    "## Verificacion — Propagacion del hidrograma\n",
    "\n",
    "El modelo recibe Q(t) en x=0 (aguas arriba) y propaga la onda hasta x=L mediante el esquema MacCormack. "
    "Las tres curvas muestran la entrada al canal, la salida simulada y la referencia sintetica generada "
    "con los mismos parametros verdaderos.",
]

# Actualizar markdown seccion metricas (celda 6 -> pasara a posicion 8)
cells[6]['source'] = [
    "## Metricas cuantitativas de ajuste\n",
    "\n",
    "NSE, RMSE, sesgo y correlacion calculados sobre el periodo post-calentamiento "
    "(excluye el warm-up inicial de 3600 s). Un NSE > 0.9 indica ajuste excelente para datos sinteticos.",
]

# Reordenar: [0..5] + [celda8, celda9] + [celda6, celda7] + [10..]
nuevo_orden = cells[:6] + [cells[8], cells[9]] + [cells[6], cells[7]] + cells[10:]
nb['cells'] = nuevo_orden

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: celdas reordenadas y markdown actualizado.")
