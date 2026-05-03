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
 
nombre_corrida = "HeHplus_100pts sim har e id 30000 range 80-1.1 100 lim"


'''
=============================================================================================
OBTENCIÓN CURVA de energias de  Energia de estado base de HeH^+ 
=============================================================================================
'''
print("Resultados con Agrupamiento")


measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0 
measurement_NA.DIAGNOSTICO_HECHO = False

n_shots=30000
R_AB=np.linspace(0.8,1.1,100)

E_0_sim_id=[]
E_0_sim_hard=[]
E_HF=[]
E_qiskit_tot=[]
E_exacta=[]
evaluaciones_totales = 0
j=0
for r_AB in R_AB: 


    cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(r_AB,"simulador_ruidoso",None,n_shots,cobyla.funcion_objetivo_Agrupada,None) 

    cob2,Min_E2,Min_E_TOT2,Dif_E_TOT2,E_total2=vqe.VQE_HeH_plus(r_AB,"simulador",None,n_shots,cobyla.funcion_objetivo_Agrupada,None) 

    E_qiskit_VQE_tot, E_fci= qiskit_pipeline.HeHplus_Qiskit_Exact(r_AB)


    E_0_sim_id.append(Min_E_TOT2) 
    E_0_sim_hard.append(Min_E_TOT) 
    E_HF.append(E_total)
    E_qiskit_tot.append(E_qiskit_VQE_tot) 
    E_exacta.append(E_fci)

    evaluaciones_totales += cob.nfev 
    j+=1
    print("punto:",j)





    







plt.figure(figsize=(6,6))

plt.plot(R_AB, E_HF, '-', 
         color="#9E21C4",  
         label="Hartree-Fock")

plt.plot(R_AB, E_qiskit_tot, '-', 
         color='#6A8F3A',  
         label="VQE (Qiskit)")

plt.plot(R_AB, E_exacta, '-', 
         color='#C2A878',  
         label="Full Configuration Interaction") 


plt.plot(R_AB, E_0_sim_hard, 'o-', 
         color="#208B9E",  
         label="VQE simulado hardware") 


plt.plot(R_AB, E_0_sim_id, 'o-', 
         color='#6E2F33',  
         label="VQE simulado ideal")

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

datos = np.column_stack((R_AB, E_0_sim_id, E_0_sim_hard, E_HF,E_qiskit_tot,E_exacta))

np.savetxt(
    f"datos/{nombre_corrida}.csv",
    datos,
    delimiter=",",
    header="R_AB,E_VQE_sim_id,E_VQE_sim_hard,E_HF,E_qiskit_tot,E_exacta",
    comments=""
)