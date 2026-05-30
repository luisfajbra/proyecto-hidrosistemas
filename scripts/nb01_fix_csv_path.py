import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src_actual = ''.join(nb['cells'][5]['source'])
assert 'series_corta_balance.csv' in src_actual, 'No se encontro series_corta_balance.csv en celda 5'

nb['cells'][5]['source'] = [
    line.replace('series_corta_balance.csv', 'series_corta_shift.csv')
    for line in nb['cells'][5]['source']
]
nb['cells'][5]['outputs'] = []
nb['cells'][5]['execution_count'] = None

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('OK: CSV_PATH actualizado a series_corta_shift.csv')
