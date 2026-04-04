#Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.chemistry import pyscf_backend 
from vqe_native.mapping import jordan_wigner

'''
=============================================================================================
OBTENCIÓN DEL HAMITONIANO DE HeH^+ EN SEGUNDA CUANTIZACIÓN EXPRESADO EN OPERADORES DE PAULI
=============================================================================================
'''
'''
No=2 #Número de puntos de muestreo. 
MaxR_AB=0.74
R_AB=np.linspace(1e-3,MaxR_AB,No) #Definimos las distancias interatomicas. 

for xi in R_AB:
    dataMol=pyscf_backend .hamiltoniano_heh(xi) 
    fermionic_op=dataMol["fermionic_op"]
    print(fermionic_op)

'''
