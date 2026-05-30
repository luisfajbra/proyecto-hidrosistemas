import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Celda 6: markdown de la seccion hidrograma
nb['cells'][6]['source'] = [
    "## Verificación — Propagación del hidrograma\n",
    "\n",
    "El modelo recibe Q(t) en x=0 (aguas arriba) y propaga la onda hasta x=L mediante el esquema MacCormack. "
    "Las tres curvas muestran la entrada al canal, la salida simulada en x=L y la referencia sintética "
    "generada con los parámetros verdaderos. Se espera que la onda llegue atenuada y desfasada.",
]

# Celda 8: markdown de la seccion metricas
nb['cells'][8]['source'] = [
    "## Métricas cuantitativas de ajuste\n",
    "\n",
    "NSE, RMSE, sesgo y correlación calculados sobre el período post-calentamiento (columna `is_warmup == False`). "
    "Para datos sintéticos sin ruido se espera NSE ≈ 1 y RMSE ≈ 0.",
]

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: markdown corregido con tildes y texto mejorado.")
