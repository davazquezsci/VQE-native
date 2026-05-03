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
from vqe_native.estimation import measurement_A 
from vqe_native.optim import cobyla 
from typing import Literal



def VQE_HeH_plus( R_AB,tipo: Literal["simulador", "hardware"],backend_o_nombre=None,Shots=10000,FO=None,theta_inicial=None):
    '''
    #------------------------------------------------
    Obtener Hamiltoniano 
    #------------------------------------------------
    '''

    dataMol=pyscf_backend .hamiltoniano_heh(R_AB) 
    fermionic_op=dataMol["fermionic_op"]
    num_spin_orbitals=2*dataMol["num_spatial_orbitals"] 
    num_electrones=dataMol["num_electrones"]
    E_Repulsion=dataMol["nuclear_repulsion_energy"]
    E_total=dataMol["reference_energy"]
    #E_electron_HF=E_total-E_Repulsion

    HamOPJW=jordan_wigner.JWdic(fermionic_op,num_spin_orbitals) 
    HamOPJW=measurement_A.Preprocesamiento_hamiltoniano(HamOPJW)

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

    if theta_inicial is None:
        Theta_0 = Cluster_Operator.Theta_Inicial(
            num_spin_orbitals,
            num_electrones
        )
    else:
        Theta_0 = np.array(theta_inicial)

    '''
    #------------------------------------------------
    Inicializamos Backend
    #------------------------------------------------
    '''

    if hasattr(backend_o_nombre, "run"):
        backend = backend_o_nombre
    else:
        backend = measurement_NA.obtener_backend(tipo, backend_o_nombre)

    '''
    #------------------------------------------------
    Buscamos el theta óptimo y la energía mínima mediante COBYLA.
    #------------------------------------------------
    '''

    cob=cobyla.cobyla(Theta_0,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots,FO)

    Min_E=cob.fun
    Min_E_TOT=Min_E+E_Repulsion  

    Dif_E_TOT=abs(Min_E_TOT-E_total)

    return cob,Min_E,Min_E_TOT,Dif_E_TOT,E_total






