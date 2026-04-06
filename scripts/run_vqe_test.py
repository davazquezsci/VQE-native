#Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.chemistry import pyscf_backend 
from vqe_native.mapping import jordan_wigner
from vqe_native.circuits import Cluster_Operator
from vqe_native.circuits import uccsd
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

dataMol=pyscf_backend .hamiltoniano_heh(0.74) 
fermionic_op=dataMol["fermionic_op"]
#print(fermionic_op)

num_spin_orbitals=2*dataMol["num_spatial_orbitals"] 
num_electrones=dataMol["num_electrones"]

HamOPJW=jordan_wigner.JWdic(fermionic_op,num_spin_orbitals) 
'''
for pauli, coef in HamOPJW.items():
    print(f"{coef: .6f} * {''.join(pauli)}")

'''
Theta_0=Cluster_Operator.Theta_Inicial(num_spin_orbitals, num_electrones)
#print(Theta_0)


S_operator=Cluster_Operator.Operador_Cluster(num_spin_orbitals, num_electrones, Theta_0)
#print(S_operator) 


jw_S_operator=jordan_wigner.JWdic(S_operator,num_spin_orbitals)
print(jw_S_operator)





Qxp = uccsd.QuantumExp(num_spin_orbitals, jw_S_operator)

print(Qxp)
print("size =", Qxp.size())
print("depth =", Qxp.depth())

fig = Qxp.draw(output="mpl")
fig.savefig("/mnt/c/Users/dangv/Desktop/circuito.png", dpi=300, bbox_inches="tight")
print("Guardado:", Path("circuito.png").resolve())