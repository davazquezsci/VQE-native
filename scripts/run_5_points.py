 #Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))



data_path = Path(__file__).resolve().parent.parent / "data" 
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
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
 
nombre_corrida = "5_puntos_ENERGIA_SIMULADOS"


'''
=============================================================================================
OBTENCIÓN CURVA de energias de  Energia de estado base de HeH^+ HARDWARE
=============================================================================================
'''
print("Resultados con Agrupamiento")
n_shots=100000

measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0 

R_AB = np.loadtxt(data_path / "5_puntos_hardware_sugeridos.csv",
                  delimiter=",", skiprows=1)

datos_thetas = np.loadtxt(data_path / "Thetas_iniciales_sim.csv",
                          delimiter=",", skiprows=1)


R_AB_thetas = datos_thetas[:, 0]
thetas = datos_thetas[:, 1:]  

if not np.allclose(R_AB, R_AB_thetas):
    raise ValueError("R_AB y R_AB_thetas no coinciden")

#backend = measurement_NA.Obtener_Backend("hardware", "ibm_kingston")
backend = None


#print("Backend usado:", backend.name)
print("Número de puntos:", len(R_AB))
print("Dimensión theta:", thetas.shape[1])

E_0=[]
E_HF=[]
E_qiskit_tot=[]
E_exacta=[]


for i, r_AB in enumerate(R_AB):
    theta_ini = thetas[i]

    E, E_TOT, Dif_E_TOT, E_total=one_Energy_Point.one_EP(r_AB,"simulador", backend,n_shots,theta_ini, cobyla.funcion_objetivo_Agrupada) 
    E_qiskit_VQE_tot, E_fci= qiskit_pipeline.HeHplus_Qiskit_Exact(r_AB)

    E_0.append(E_TOT) 
    E_HF.append(E_total)
    E_qiskit_tot.append(E_qiskit_VQE_tot) 
    E_exacta.append(E_fci)

    i+=1
    print("punto:",i)





print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)
print() 

datos = np.column_stack((R_AB, E_0, E_HF,E_qiskit_tot,E_exacta))

np.savetxt(
    data_path / f"{nombre_corrida}.csv",
    datos,
    delimiter=",",
    header="R_AB,E_VQE,E_HF,E_qiskit_tot,E_exacta",
    comments=""
)