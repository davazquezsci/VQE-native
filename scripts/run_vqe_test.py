#Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.chemistry import pyscf_backend 
from vqe_native.mapping import jordan_wigner
from vqe_native.circuits import Cluster_Operator
from vqe_native.circuits import uccsd
from vqe_native.estimation import measurement
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

'''
#------------------------------------------------
Obtener Hamiltoniano 
#------------------------------------------------
'''

dataMol=pyscf_backend .hamiltoniano_heh(0.774) 
fermionic_op=dataMol["fermionic_op"]
#print(fermionic_op)

num_spin_orbitals=2*dataMol["num_spatial_orbitals"] 
num_electrones=dataMol["num_electrones"]

HamOPJW=jordan_wigner.JWdic(fermionic_op,num_spin_orbitals) 
print(HamOPJW)
'''

for pauli, coef in HamOPJW.items():
    print(f"{coef: .6f} * {''.join(pauli)}")

'''

'''
#------------------------------------------------
Energias HF 
#------------------------------------------------
'''

E_Repulsion=dataMol["nuclear_repulsion_energy"]
E_total=dataMol["reference_energy"]
E_electron_HF=E_total-E_Repulsion

'''
#------------------------------------------------
Obtener Estado inicial de Hartree Fock 
#------------------------------------------------
'''
HF_state= uccsd.Estado_HF(num_spin_orbitals,num_electrones)
print(HF_state)

'''
#------------------------------------------------
Obtener ANZATS UCCSD
#------------------------------------------------
'''

Theta_0=Cluster_Operator.Theta_Inicial(num_spin_orbitals, num_electrones)
print(Theta_0)


S_operator=Cluster_Operator.Operador_Cluster(num_spin_orbitals, num_electrones, Theta_0)
#print(S_operator) 


jw_S_operator=jordan_wigner.JWdic(S_operator,num_spin_orbitals)
#print(jw_S_operator)


#Qxp = uccsd.QuantumExp(num_spin_orbitals, jw_S_operator)
Qxp=QuantumCircuit(num_spin_orbitals)

print(Qxp)

'''
#IMPRIMIR  CIRCUITO CUANTICO 

print("size =", Qxp.size())
print("depth =", Qxp.depth())

fig = Qxp.draw(output="mpl")
fig.savefig("/mnt/c/Users/dangv/Desktop/circuito.png", dpi=300, bbox_inches="tight")
print("Guardado:", Path("circuito.png").resolve())

'''



'''
#------------------------------------------------
Obtener MEDICION DE PUNTO 
#------------------------------------------------
'''

Circuitos_NT,Coeficientes_NT,Cadenas_Pauli_NT,Coeficientes_T=measurement.Generar_Circuitos_Medicion_Hamiltoniano(HamOPJW, HF_state, Qxp, num_spin_orbitals)

backend=measurement.Obtener_Backend("simulador")

Count_list=measurement.Ejecutar_QCircuit_1point(Circuitos_NT,backend,10000)

E=measurement.PostProcesado_1point(Count_list, Coeficientes_NT,Coeficientes_T)

print(E) 

print(abs(E-E_electron_HF))





