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


Qxp = uccsd.QuantumExp(num_spin_orbitals, jw_S_operator)
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
Backend de Medicion
#------------------------------------------------
'''

Shots=10000
backend=measurement_NA.Obtener_Backend("simulador")


import os
print("Directorio actual:", os.getcwd())
'''
========================================================
=========================================================
Obtener MEDICION DE PUNTO  sin Agrupamiento
========================================================
=========================================================
'''
measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0



E=cobyla.funcion_objetivo(Theta_0,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots)

print("Energia en el Circuito:",E) 
print("Energia de Hartree Fock", E_electron_HF)
print()
print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)


'''
========================================================
=========================================================
Obtener MEDICION DE PUNTO  CON Agrupamiento
========================================================
=========================================================
'''
measurement_NA.TOTAL_CIRCUITOS = 0
measurement_NA.TOTAL_SHOTS = 0



E=cobyla.funcion_objetivo_Agrupada(Theta_0,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots)

print("Energia en el Circuito:",E) 
print("Energia de Hartree Fock", E_electron_HF)
print()
print("Circuitos totales:", measurement_NA.TOTAL_CIRCUITOS)
print("Shots totales:", measurement_NA.TOTAL_SHOTS)



