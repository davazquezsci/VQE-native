import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import math 



def Theta_Inicial(num_spin_orbitals,num_electrons):
    """
    Función para calcular el valor inicial de theta para el ansatz UCCSD.
    
    Parámetros:
    num_spin_orbitals (int): El número de orbitales de espín en el sistema.
    
    Retorna:
    float: El valor inicial de theta.
    """

    #Obtenemos tamaño del vector de parámetros para el ansatz UCCSD.
    num_parameters = 0

    num_virtual_orbitals=num_spin_orbitals-num_electrons
    for i in range (1,2+1): #Exitaciones para UCCSD, solo Sigles and Doubles.
        num_parameters += math.comb(num_virtual_orbitals, i) * math.comb(num_electrons, i) 


    theta0= np.random.uniform(-0.01, 0.01, size=num_parameters)

    return theta0


def Operador_Cluster(num_spin_orbitals, num_electrons, theta):

    Op_C = {}

    # orbitales ocupados (referencia HF)
    occupied = list(range(num_electrons))

    # orbitales virtuales
    virtual = list(range(num_electrons, num_spin_orbitals))

    param_index=0 #Indice depara guiarnos con el vector theta 
    # -----------------
    # Singles
    # -----------------
    for i in occupied:
        for a in virtual:

            # operador de excitación
            term = f"+_{a} -_{i}"

            # conjugado
            term_dag = f"+_{i} -_{a}"

            Op_C[term]=theta[param_index]
            Op_C[term_dag]=-theta[param_index]


            param_index += 1    

    # -----------------
    # Dobles
    # -----------------
    for i in range(len(occupied)):
        for j in range(i+1, len(occupied)):

            for a in range(len(virtual)):
                for b in range(a+1, len(virtual)):

                    occ_i = occupied[i]
                    occ_j = occupied[j]

                    vir_a = virtual[a]
                    vir_b = virtual[b]

                    term = f"+_{vir_a} +_{vir_b} -_{occ_j} -_{occ_i}"
                    term_dag = f"+_{occ_i} +_{occ_j} -_{vir_b} -_{vir_a}"

                    Op_C[term]=theta[param_index]
                    Op_C[term_dag]=-theta[param_index]

                    param_index += 1

    # Verificacion de consistencia
    if param_index != len(theta):
        raise ValueError(
            f"Numero de parametros inconsistente: "
            f"se generaron {param_index} excitaciones pero theta tiene {len(theta)} parametros."
        )

    return  Op_C


