
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import Literal
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.estimation import measurement_NA
from vqe_native.circuits import uccsd
import itertools





'''
==================================================================
==================================================================
Segunda aproximación: CON agrupamiento de mediciones. 
==================================================================
==================================================================
'''

'''
Agrupamiento
'''

def Son_Compatibles(op1,op2):
     for i in range(len(op1)):
        if op1[i] != op2[i] and op1[i] != 'I' and op2[i] != 'I':
            return False
     return True
    
     
def Fusionar_Bases(op1, op2):
    if not Son_Compatibles(op1, op2):
        raise ValueError("Las cadenas de Pauli no son compatibles.")

    OpPauli = []

    for a, b in zip(op1, op2):
        if a != 'I':
            OpPauli.append(a)
        else:
            OpPauli.append(b)

    return tuple(OpPauli) 

#Para mejorar los agrupamientos , primero odenamos las cadenas de menos a más flexibles. 

def num_trivial(op):
    n=0
    for i in range(len(op)):
        if op[i]=='I':
            n+=1
        else:
            continue
    return n 

def Preprocesamiento_hamiltoniano(Hamiltoniano: dict):
    ham_ordenado = dict(
        sorted(Hamiltoniano.items(), key=lambda x: num_trivial(x[0]))
    )
    return ham_ordenado


def Agrupar_Cadenas_Pauli(Hamiltoniano: dict, num_spin_orbitals: int):

    grupos = []
    Trivial_Chain = tuple('I' for _ in range(num_spin_orbitals))
    Coeficientes_T = []

    for cadena, coef in Hamiltoniano.items():

        if cadena == Trivial_Chain:
            Coeficientes_T.append(coef)
            continue

        colocada = False

        for grupo in grupos:
            if Son_Compatibles(cadena, grupo["base"]):
                grupo["terms"].append((cadena, coef))
                grupo["base"] = Fusionar_Bases(cadena, grupo["base"])
                colocada = True
                break

        if not colocada:
            grupos.append({
                "base": cadena,
                "terms": [(cadena, coef)]
            })

    return grupos, Coeficientes_T

'''
Generar lista de circuitos cuanticos.
'''

#Preparar circuito de medicion en cierta base es equivalente a NA. 

def Generar_Circuitos_Medicion_Hamiltoniano_A(
    Hamiltoniano_agrupado: list,
    HF_state:QuantumCircuit,
    anzats: QuantumCircuit,
    num_spin_orbitals: int
):
        
        '''
        Generaremos una lista de Circuirtos cuantos para  cada theta especifico 
        '''

        Circuitos_NT=[]
        Soportes=[]
        Pauli_and_Coef_NT=[]

        for grupo in Hamiltoniano_agrupado:
            qc, soporte = measurement_NA.Circuito_Medicion_PauliChainH(
                grupo["base"], HF_state, anzats, num_spin_orbitals
            )
            Circuitos_NT.append(qc)
            Soportes.append(soporte)
            Pauli_and_Coef_NT.append(grupo["terms"])

        return Circuitos_NT,Soportes, Pauli_and_Coef_NT

#Backend de medicion es equivalente a NA. 

#Ejecición de medicion es equivalente a NA. 





def PostProcesado_A(counts_list, Soportes, Pauli_and_Coef_NT,Coeficientes_T): 
        E=0

        for j in range(len(Coeficientes_T)):
            E+=Coeficientes_T[j]

        '''
        Calcular número total de shots realmente obtenidos
        Nesesario por que, en algunos casos , simuladores o Hardware real cuantico
        descarta algunos shots
        
        '''
        n=0 #Cunta el soporte y Pauli_and_coef  empleado
        for counts in counts_list: 
            
            total_shots =  sum(counts.values())

            
            soporte_bloque=Soportes[n]   
            #Para cada elemento de la lista de tuplas
            for i in Pauli_and_Coef_NT[n]:
                #Para cada Pauli String elemento de tupla:  
                cadena_individual = i[0]
                soporte_individual = uccsd.Soporte(cadena_individual, len(cadena_individual))
                expectation=0
                    
                Nuevo_soporte=[]


                for k in range(len(soporte_bloque)):
                    if soporte_bloque[k] in soporte_individual:
                        Nuevo_soporte.append(k)
                
                

                for bitstring, veces in counts.items():
                    #Corregimos el orden del bitstring para ajustarse a nuestra convencion.
                    bits =  bitstring[::-1] 
                    numero_unos =0

                    for k in Nuevo_soporte:
                        if bits[k]=='1':
                            numero_unos +=1

                    Lambda= (-1)**(numero_unos)
                    Probabilidad=(veces)/total_shots

                    expectation += Probabilidad*Lambda

        
                contribucion = i[1]* expectation
                E += contribucion 
            n+=1
        

        return E

