
# Ejecutar corrida de 5 puntos desde Colab


import sys
import time
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Rutas base del repo en Colab

REPO_PATH = Path("/content/VQE-native")
SRC_PATH = REPO_PATH / "src"
DATA_PATH = REPO_PATH / "data"

# Si one_EP está fuera de src pero dentro del repo, agrega también REPO_PATH
sys.path.append(str(SRC_PATH))
sys.path.append(str(REPO_PATH))


# Imports de tu proyecto

from vqe_native.chemistry import pyscf_backend
from vqe_native.mapping import jordan_wigner
from vqe_native.circuits import Cluster_Operator
from vqe_native.circuits import uccsd
from vqe_native.estimation import measurement_NA
from vqe_native.optim import cobyla
from vqe_native.vqe import vqe
from qiskit_data import qiskit_pipeline
from one_EP import one_Energy_Point

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer


# Parámetros de corrida

nombre_corrida = "5_puntos_ENERGIA_SIMULADOS_RUIDOSOS 2"

FO = cobyla.funcion_objetivo_Agrupada
n_shots=30000 

measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0

print("Resultados con Agrupamiento")

t0 = time.time()

R_AB = np.loadtxt(
    DATA_PATH / "5_puntos_hardware_sugeridos.csv",
    delimiter=",",
    skiprows=1
)

datos_thetas = np.loadtxt(
    DATA_PATH / "Thetas_iniciales_sim.csv",
    delimiter=",",
    skiprows=1
)

R_AB_thetas = datos_thetas[:, 0]
thetas = datos_thetas[:, 1:]

if not np.allclose(R_AB, R_AB_thetas):
    raise ValueError("R_AB y R_AB_thetas no coinciden")


backend = measurement_NA.Obtener_Backend("hardware", "ibm_kingston")

print("Número de puntos:", len(R_AB))
print("Dimensión theta:", thetas.shape[1])


E_0 = []
E_HF = []
E_qiskit_tot = []
E_exacta = []

resultados_puntos = []

for i, r_AB in enumerate(R_AB):
    theta_ini = thetas[i]

    E, E_TOT, Dif_E_TOT, E_total = one_Energy_Point.one_EP(
        r_AB,
        "hardware",
        backend,
        n_shots,
        theta_ini,
        FO
    )

    E_qiskit_VQE_tot, E_fci = qiskit_pipeline.HeHplus_Qiskit_Exact(r_AB)

    E_0.append(E_TOT)
    E_HF.append(E_total)
    E_qiskit_tot.append(E_qiskit_VQE_tot)
    E_exacta.append(E_fci)

    resultado_i = {
        "indice_punto": int(i),
        "R_AB": float(r_AB),
        "shots_por_circuito": int(n_shots),
        "energia_electronica_vqe": float(E),
        "energia_total_vqe": float(E_TOT),
        "energia_total_referencia": float(E_total),
        "error_absoluto_total": float(Dif_E_TOT),
        "energia_total_qiskit": float(E_qiskit_VQE_tot),
        "energia_exacta_fci": float(E_fci),
    }
    resultados_puntos.append(resultado_i)

    print(f"Punto {i+1}/{len(R_AB)} | R_AB = {r_AB:.6f}")

t1 = time.time()


print()
print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)
print("Tiempo total (s):", t1 - t0)
print()

# ------------------------------------------------------------
# Guardar CSV principal de curva
# ------------------------------------------------------------
datos = np.column_stack((R_AB, E_0, E_HF, E_qiskit_tot, E_exacta))

csv_curva_path = Path(f"/content/{nombre_corrida}.csv")

np.savetxt(
    csv_curva_path,
    datos,
    delimiter=",",
    header="R_AB,E_VQE,E_HF,E_qiskit_tot,E_exacta",
    comments=""
)

print("Archivo CSV de curva generado en:")
print(csv_curva_path)

# ------------------------------------------------------------
# Guardar JSON de resumen
# ------------------------------------------------------------
resumen = {
    "nombre_corrida": nombre_corrida,
    "numero_puntos": int(len(R_AB)),
    "shots_por_circuito": int(n_shots),
    "agrupamiento": True,
    "circuitos_totales": int(getattr(measurement_NA, "TOTAL_CIRCUITOS", -1)),
    "shots_totales": int(getattr(measurement_NA, "TOTAL_SHOTS", -1)),
    "tiempo_total_segundos": float(t1 - t0),
    "resultados_punto_a_punto": resultados_puntos,
}

json_path = Path(f"/content/{nombre_corrida}_resumen.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("Archivo JSON generado en:")
print(json_path)

# ------------------------------------------------------------
# Descargar automáticamente
# ------------------------------------------------------------
files.download(str(csv_curva_path))
files.download(str(json_path))