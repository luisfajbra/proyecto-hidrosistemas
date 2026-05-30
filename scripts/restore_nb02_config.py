"""Restore config cell in 02 and fix nt= in simulate_outlet (02 and 03)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "config02",
    "metadata": {},
    "outputs": [],
    "source": [
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import importlib\n",
        "import importlib.util\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from joblib import Parallel, delayed\n",
        "from SALib.analyze import sobol as sobol_analyze\n",
        "from SALib.sample import sobol as sobol_sample\n",
        "from scipy.optimize import least_squares\n",
        "\n",
        "ROOT = Path.cwd().resolve()\n",
        "if not (ROOT / \"src\").exists():\n",
        "    ROOT = ROOT.parent\n",
        "sys.path.insert(0, str(ROOT))\n",
        "\n",
        "import src.model as _model_mod\n",
        "\n",
        "importlib.reload(_model_mod)\n",
        "from src.model import saint_venant_1d\n",
        "\n",
        "_spec = importlib.util.spec_from_file_location(\"sinteticos\", ROOT / \"data\" / \"sinteticos.py\")\n",
        "sinteticos = importlib.util.module_from_spec(_spec)\n",
        "_spec.loader.exec_module(sinteticos)\n",
        "\n",
        "B_W_FIXED = sinteticos.B_W\n",
        "PARAM_NAMES = [\"n\", \"S0\", \"Q_up\", \"q_lat\"]\n",
        "PARAMS_TRUE_N_S0 = [sinteticos.N_MANN, sinteticos.S0]\n",
        "PARAMS_TRUE = None\n",
        "L_CANAL = sinteticos.L\n",
        "NX = 100\n",
        "WARMUP_SECONDS = 3600.0\n",
        "BOUNDS_LO_N_S0 = [0.01, 0.0001]\n",
        "BOUNDS_HI_N_S0 = [0.06, 0.005]\n",
        "BOUNDS_LO = None\n",
        "BOUNDS_HI = None\n",
        "CAL_FRAC, VAL_FRAC = 0.60, 0.30\n",
        "SOBOL_METRIC = \"nse\"\n",
        "SOBOL_CONF = 0.95\n",
        "\n",
        "FIG = ROOT / \"figures\"\n",
        "DATA = ROOT / \"data\" / \"synthetic\"\n",
        "for d in (FIG, DATA):\n",
        "    d.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "INFORME = False\n",
        "FAST = not INFORME\n",
        "SOBOL_N = 128 if FAST else 2048\n",
        "NBOOT = 0 if FAST else 200\n",
        "RUN_COMPLEMENTO = INFORME\n",
        "COMPUTE_SSC = INFORME\n",
        "RUN_PROFILING = False\n",
        "N_JOBS = 1\n",
        "USE_NOISY_CSV = False\n",
        "USE_LONG_SERIES = False\n",
        "RNG_SEED = 42\n",
        "\n",
        "print(\"ROOT:\", ROOT, \"| INFORME:\", INFORME, \"| SOBOL_N:\", SOBOL_N)\n",
    ],
}


def patch_simulate(nb):
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "def simulate_outlet" not in src:
            continue
        if "nt=nt" in src:
            continue
        new = []
        for line in cell["source"]:
            if "time_seconds=t_sec," in line:
                new.append(line)
                new.append("            nt=nt,\n")
            else:
                new.append(line)
        cell["source"] = new
        cell["outputs"] = []
        cell["execution_count"] = None


def main():
    nb02_path = ROOT / "notebooks" / "02_sensitivity.ipynb"
    nb02 = json.loads(nb02_path.read_text(encoding="utf-8"))
    if not any("INFORME =" in "".join(c.get("source", [])) for c in nb02["cells"] if c["cell_type"] == "code"):
        nb02["cells"].insert(1, CONFIG_CELL)
    patch_simulate(nb02)
    nb02_path.write_text(json.dumps(nb02, ensure_ascii=False, indent=1), encoding="utf-8")

    nb03_path = ROOT / "notebooks" / "03_calibration.ipynb"
    nb03 = json.loads(nb03_path.read_text(encoding="utf-8"))
    patch_simulate(nb03)
    nb03_path.write_text(json.dumps(nb03, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK: config restored in 02, nt=nt in 02 and 03")


if __name__ == "__main__":
    main()
