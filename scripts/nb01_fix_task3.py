import json

with open('notebooks/01_model_verification.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nuevo_codigo = [
    "eval_df = resultado[~resultado['is_warmup']]\n",
    "obs = eval_df['Q_downstream_m3s'].to_numpy()\n",
    "sim = eval_df['Q_sim_m3s'].to_numpy()\n",
    "\n",
    "mae   = np.mean(np.abs(obs - sim))\n",
    "rmse  = np.sqrt(np.mean((obs - sim) ** 2))\n",
    "bias  = np.mean(sim - obs)\n",
    "r     = np.corrcoef(obs, sim)[0, 1]\n",
    "denom = np.sum((obs - np.mean(obs)) ** 2)\n",
    "nse   = float(1.0 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else float('nan')\n",
    "\n",
    "metricas = pd.Series({\n",
    "    'NSE':          nse,\n",
    "    'MAE [m³/s]':   mae,\n",
    "    'RMSE [m³/s]':  rmse,\n",
    "    'Bias [m³/s]':  bias,\n",
    "    'r':            r,\n",
    "}).to_frame('valor').round(4)\n",
    "display(metricas)\n",
    "\n",
    "# Scatter obs vs sim (figura de verificacion estandar)\n",
    "lim = [min(obs.min(), sim.min()) * 0.97, max(obs.max(), sim.max()) * 1.03]\n",
    "fig, ax = plt.subplots(figsize=(5, 5))\n",
    "ax.scatter(obs, sim, s=10, alpha=0.4, color='steelblue', edgecolors='none')\n",
    "ax.plot(lim, lim, 'k--', lw=1.2, label='1:1')\n",
    "ax.set_xlim(lim); ax.set_ylim(lim)\n",
    "ax.set_xlabel('Q observado [m³/s]')\n",
    "ax.set_ylabel('Q simulado [m³/s]')\n",
    "ax.set_title(f'Dispersión obs vs sim — NSE={nse:.4f}  r={r:.4f}')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]

nb['cells'][9]['source'] = nuevo_codigo
nb['cells'][9]['outputs'] = []
nb['cells'][9]['execution_count'] = None

with open('notebooks/01_model_verification.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: celda de metricas mejorada con scatter.")
