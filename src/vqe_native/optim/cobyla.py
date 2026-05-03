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
from scipy.optimize import minimize




def funcion_objetivo(theta,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots):
    
    '''
    #------------------------------------------------
    Obtener ANZATS UCCSD
    #------------------------------------------------
    '''

    
    S_operator=Cluster_Operator.Operador_Cluster(num_spin_orbitals, num_electrones, theta)
    jw_S_operator=jordan_wigner.JWdic(S_operator,num_spin_orbitals)

    Qxp = uccsd.QuantumExp(num_spin_orbitals, jw_S_operator)



    Circuitos_NT,Coeficientes_NT,Cadenas_Pauli_NT,Coeficientes_T=measurement_NA.Generar_Circuitos_Medicion_Hamiltoniano(HamOPJW, HF_state, Qxp, num_spin_orbitals)

    

    Count_list=measurement_NA.Ejecutar_QCircuit_1point(Circuitos_NT,backend,shots=Shots)

    E=measurement_NA.PostProcesado_1point(Count_list, Coeficientes_NT,Coeficientes_T)

    return float(np.real(E)) 


def funcion_objetivo_Agrupada(theta,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots):
    
    '''
    #------------------------------------------------
    Obtener ANZATS UCCSD
    #------------------------------------------------
    '''

    
    S_operator=Cluster_Operator.Operador_Cluster(num_spin_orbitals, num_electrones, theta)
    jw_S_operator=jordan_wigner.JWdic(S_operator,num_spin_orbitals)

    Qxp = uccsd.QuantumExp(num_spin_orbitals, jw_S_operator)

    grupos, Coeficientes_T=measurement_A.Agrupar_Cadenas_Pauli(HamOPJW, num_spin_orbitals)

    Circuitos_NT,Soportes, Pauli_and_Coef_NT=measurement_A.Generar_Circuitos_Medicion_Hamiltoniano_A(grupos, HF_state, Qxp, num_spin_orbitals)

    

    Count_list=measurement_NA.Ejecutar_QCircuit_1point(Circuitos_NT,backend,shots=Shots)

    E=measurement_A.PostProcesado_A(Count_list, Soportes, Pauli_and_Coef_NT,Coeficientes_T)

    return float(np.real(E))
        



def cobyla(Theta_0,HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots,FO):
    
    resultado = minimize(
        fun=FO,
        x0=Theta_0,
        method="COBYLA",
        args=(HamOPJW,HF_state,num_spin_orbitals,num_electrones,backend,Shots),
        options={
            "maxiter": 100,
            "rhobeg": 0.08,
            "tol": 1e-3,
            "disp": False,
        }
        )
    

    return resultado
