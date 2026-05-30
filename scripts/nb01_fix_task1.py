import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nuevo_codigo = [
    "# Usar resultados ya calculados por run_batch\n",
    "t_min      = resultado['t_seconds'].to_numpy() / 60\n",
    "q_upstream = resultado['Q_upstream_m3s'].to_numpy()\n",
    "q_sim_dn   = resultado['Q_sim_m3s'].to_numpy()\n",
    "q_dn_obs   = resultado['Q_downstream_m3s'].to_numpy()\n",
    "is_warmup  = resultado['is_warmup'].to_numpy()\n",
    "\n",
    "t_wu_end = t_min[(~is_warmup).argmax()] if (~is_warmup).any() else t_min[-1]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(11, 4))\n",
    "ax.axvspan(t_min[0], t_wu_end, color='gray', alpha=0.12, label='Warm-up')\n",
    "ax.plot(t_min, q_upstream, color='0.55', lw=1.0, ls='--', label='Q entrada (x=0)')\n",
    "ax.plot(t_min, q_sim_dn,   color='steelblue', lw=1.6, label='Q simulado (x=L)')\n",
    "ax.plot(t_min, q_dn_obs,   color='crimson',   lw=1.0, ls=':',  label='Q referencia sintetica (x=L)')\n",
    "ax.set_xlabel('Tiempo [min]')\n",
    "ax.set_ylabel('Q [m3/s]')\n",
    "ax.set_title('Propagacion del hidrograma — entrada, salida simulada y referencia sintetica')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "fig.savefig(FIGURES_DIR / '01a_variable_upstream.png', dpi=200)\n",
    "plt.show()\n",
    "\n",
    "print(f'Max desviacion |Q_sim - Q_ref|: {np.max(np.abs(q_sim_dn - q_dn_obs)):.4f} m3/s')\n",
]

nb['cells'][9]['source'] = nuevo_codigo
nb['cells'][9]['outputs'] = []
nb['cells'][9]['execution_count'] = None

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: celda 9 corregida.")
