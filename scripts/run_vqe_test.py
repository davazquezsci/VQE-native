#Importar bibliotecas.
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.chemistry import pyscf_backend 
from vqe_native.mapping import jordan_wigner


dataMol=pyscf_backend .hamiltoniano_heh(0.74) 
fermionic_op=dataMol["fermionic_op"]
print(fermionic_op)
num_spin_orbitals=2*dataMol["num_spatial_orbitals"] 
print(num_spin_orbitals)
HamOPJW=jordan_wigner.JWdic(fermionic_op,num_spin_orbitals) 
for pauli, coef in HamOPJW.items():
    print(f"{coef: .6f} * {''.join(pauli)}")