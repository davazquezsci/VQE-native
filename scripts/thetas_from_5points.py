  #Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.chemistry import pyscf_backend 
from vqe_native.mapping import jordan_wigner
from vqe_native.circuits import Cluster_Operator
from vqe_native.circuits import uccsd
from vqe_native.estimation import measurement_NA
from vqe_native.optim import cobyla 
from vqe_native.vqe import vqe
from qiskit_data import qiskit_pipeline
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
 
nombre_corrida = "Thetas_iniciales_sim_har_2"

print("Resultados con Agrupamiento")
n_shots=100000

measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0 

R_AB = np.loadtxt("datos/5_puntos_hardware_sugeridos.csv", delimiter=",", skiprows=1)



Thetas_min=[]
evaluaciones_totales = 0
j=0

for r_AB in R_AB: 


    cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(r_AB,"simulador", None,n_shots,cobyla.funcion_objetivo_Agrupada,None) 
    E_qiskit_VQE_tot, E_fci= qiskit_pipeline.HeHplus_Qiskit_Exact(r_AB)

    Thetas_min.append(cob.x) 


    evaluaciones_totales += cob.nfev 
    j+=1
    print("punto:",j)





print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)
print("Evaluaciones COBYLA:", evaluaciones_totales)
print() 

Thetas_min = np.array(Thetas_min)

datos = np.column_stack((R_AB, Thetas_min))

n_thetas = Thetas_min.shape[1]
header = "R_AB," + ",".join([f"theta_{i}" for i in range(n_thetas)])

np.savetxt(
    f"datos/{nombre_corrida}.csv",
    datos,
    delimiter=",",
    header=header,
    comments=""
)