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
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  


'''
for i in [1000,10000,50000,100000,500000]:
    cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(0.774,"simulador",i) 

    print("Shots:", i)
    print("VQE:", Min_E_TOT)
    print("HF :", E_total)
    print("Error HF:", abs(E_total - Min_E_TOT))
    print()

'''
n_shots=100000
'''
==================================================================
==================================================================
Primera aproximación: Sin agrupamiento de mediciones. 
==================================================================
==================================================================
'''
print("Resultados sin Agrupamiento")


cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(0.774,"simulador",n_shots,cobyla.funcion_objetivo) 

print("Shots:", n_shots)
print("VQE:", Min_E_TOT)
print("HF :", E_total)
print("Error HF:", abs(E_total - Min_E_TOT))
print()

'''
==================================================================
==================================================================
Segunda aproximación: CON agrupamiento de mediciones. 
==================================================================
==================================================================
'''

print("Resultados con Agrupamiento")

cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total=vqe.VQE_HeH_plus(0.774,"simulador",n_shots,cobyla.funcion_objetivo_Agrupada) 

print("Shots:", n_shots)
print("VQE:", Min_E_TOT)
print("HF :", E_total)
print("Error HF:", abs(E_total - Min_E_TOT))
print()