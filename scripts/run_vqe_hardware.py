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
 
nombre_corrida = "HeHplus_5 pts hardware 10000 v1"


'''
=============================================================================================
OBTENCIÓN CURVA de energias de  Energia de estado base de HeH^+ HARDWARE
=============================================================================================
'''
print("Resultados con Agrupamiento")
n_shots=2000

measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0 

R_AB = np.loadtxt("datos/5_puntos_hardware_sugeridos.csv", delimiter=",", skiprows=1) 

datos_thetas = np.loadtxt("datos/Thetas_iniciales_sim.csv", delimiter=",", skiprows=1)

R_AB_thetas = datos_thetas[:, 0]
thetas = datos_thetas[:, 1:]  

if not np.allclose(R_AB, R_AB_thetas):
    raise ValueError("R_AB y R_AB_thetas no coinciden")

backend = measurement_NA.Obtener_Backend("hardware", "ibm_kingston")


print("Backend usado:", backend.name)
print("Número de puntos:", len(R_AB))
print("Dimensión theta:", thetas.shape[1])

E_0=[]
E_HF=[]
E_qiskit_tot=[]
E_exacta=[]
evaluaciones_totales = 0
j=0

for i, r_AB in enumerate(R_AB):
    theta_ini = thetas[i]


    cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(r_AB,"hardware", backend,n_shots,cobyla.funcion_objetivo_Agrupada,theta_ini) 
    E_qiskit_VQE_tot, E_fci= qiskit_pipeline.HeHplus_Qiskit_Exact(r_AB)

    E_0.append(Min_E_TOT) 
    E_HF.append(E_total)
    E_qiskit_tot.append(E_qiskit_VQE_tot) 
    E_exacta.append(E_fci)

    evaluaciones_totales += cob.nfev 
    j+=1
    print("punto:",j)





    







plt.figure(figsize=(6,6))

plt.plot(R_AB, E_HF, 'o-', 
         color='#4C6A92',  
         label="Hartree-Fock")

plt.plot(R_AB, E_qiskit_tot, 'o-', 
         color='#6A8F3A',  
         label="VQE (Qiskit)")

plt.plot(R_AB, E_exacta, 'o-', 
         color='#C2A878',  
         label="Full Configuration Interaction") 


plt.plot(R_AB, E_0, 'o-', 
         color='#6E2F33',  
         label="VQE")

plt.xlabel(r"Distancia internuclear $R_{AB}$ ($\AA$)")
plt.ylabel("Energía total (Hartree)")
plt.title(r"Curva de energía de HeH$^+$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0.8, 1.1)  

plt.ylim(-2.87, -2.83)
plt.savefig(f"graficas/{nombre_corrida}.png", dpi=300)
plt.close()
plt.show() 

print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)
print("Evaluaciones COBYLA:", evaluaciones_totales)
print() 

datos = np.column_stack((R_AB, E_0, E_HF,E_qiskit_tot,E_exacta))

np.savetxt(
    f"datos/{nombre_corrida}.csv",
    datos,
    delimiter=",",
    header="R_AB,E_VQE,E_HF,E_qiskit_tot,E_exacta",
    comments=""
)