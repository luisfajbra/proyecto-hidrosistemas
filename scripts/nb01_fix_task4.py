import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nuevo_codigo = [
    "t_arr  = resultado['t_seconds'].to_numpy()\n",
    "dt_arr = np.diff(t_arr, prepend=t_arr[0])\n",
    "dt_arr[0] = dt_arr[1]\n",
    "\n",
    "V_in  = np.cumsum(resultado['Q_upstream_m3s'].to_numpy() * dt_arr)\n",
    "V_out = np.cumsum(resultado['Q_sim_m3s'].to_numpy()       * dt_arr)\n",
    "err_rel = (V_in[-1] - V_out[-1]) / V_in[-1] * 100\n",
    "print(f'Error de masa relativo (total): {err_rel:.4f} %')\n",
    "\n",
    "fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)\n",
    "\n",
    "axes[0].plot(resultado['datetime'], resultado['Q_upstream_m3s'],\n",
    "             label='Q upstream', alpha=0.8)\n",
    "axes[0].plot(resultado['datetime'], resultado['Q_sim_m3s'],\n",
    "             label='Q sim salida', alpha=0.8, ls='--')\n",
    "axes[0].set_ylabel('Q [m³/s]')\n",
    "axes[0].legend()\n",
    "axes[0].set_title('Test B — Conservación de masa (hidrograma)')\n",
    "\n",
    "axes[1].plot(resultado['datetime'], V_in  / 1e6, label='Vol. acumulado entrada [Mm³]')\n",
    "axes[1].plot(resultado['datetime'], V_out / 1e6, label='Vol. acumulado salida [Mm³]', ls='--')\n",
    "axes[1].set_ylabel('Volumen acumulado [Mm³]')\n",
    "axes[1].set_xlabel('Fecha')\n",
    "axes[1].legend()\n",
    "axes[1].set_title(f'Volumen acumulado — Error relativo: {err_rel:.4f} %')\n",
    "\n",
    "plt.tight_layout()\n",
    "fig.savefig(FIGURES_DIR / '01b_mass_conservation.png', dpi=200)\n",
    "plt.show()\n",
]

nb['cells'][11]['source'] = nuevo_codigo
nb['cells'][11]['outputs'] = []
nb['cells'][11]['execution_count'] = None

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: Test B mejorado.")
