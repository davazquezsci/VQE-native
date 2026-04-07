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
from vqe_native.optim import cobyla 
from typing import Literal



def VQE_HeH_plus( distancia_atomica,tipo: Literal["simulador", "hardware"], Shots):
    '''
    #------------------------------------------------
    Obtener Hamiltoniano 
    #------------------------------------------------
    '''

    dataMol=pyscf_backend .hamiltoniano_heh(distancia_atomica) 
    fermionic_op=dataMol["fermionic_op"]
    num_spin_orbitals=2*dataMol["num_spatial_orbitals"] 
    num_electrones=dataMol["num_electrones"]
    E_Repulsion=dataMol["nuclear_repulsion_energy"]
    E_total=dataMol["reference_energy"]
    #E_electron_HF=E_total-E_Repulsion

    HamOPJW=jordan_wigner.JWdic(fermionic_op,num_spin_orbitals) 

    '''
    #------------------------------------------------
    Obtener Estado inicial de Hartree Fock 
    #------------------------------------------------
    '''
    HF_state= uccsd.Estado_HF(num_spin_orbitals,num_electrones) 
    '''
    #------------------------------------------------
    Obtener Theta Inicial
    #------------------------------------------------
    '''

    Theta_0=Cluster_Operator.Theta_Inicial(num_spin_orbitals, num_electrones)

    '''
    #------------------------------------------------
    Inicializamos Backend
    #------------------------------------------------
    '''

    backend=measurement.Obtener_Backend(tipo) 

    '''
    #------------------------------------------------
    Buscamos el theta óptimo y la energía mínima mediante COBYLA.
    #------------------------------------------------
    '''

    cob=cobyla.cobyla(Theta_0,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots)

    Min_E=cob.fun
    Min_E_TOT=Min_E+E_Repulsion  

    Dif_E_TOT=abs(Min_E_TOT-E_total)

    return cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total
