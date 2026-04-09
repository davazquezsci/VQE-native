from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import Literal
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from vqe_native.circuits import uccsd



def pre_measurement(PauliChain,num_spin_orbitals):
    qc=QuantumCircuit(num_spin_orbitals)
    for i in range(num_spin_orbitals):
        if PauliChain[i]=='X':
            qc.h(i)
        elif PauliChain[i]=='Y':
            qc.sdg(i)
            qc.h(i)
    return qc

'''
==================================================================
==================================================================
Primera aproximación: Sin agrupamiento de mediciones. 
==================================================================
==================================================================
'''

def Circuito_Medicion_PauliChainH(PauliChain,
    HF_state:QuantumCircuit,
    anzats: QuantumCircuit,
    num_spin_orbitals: int): 
        '''
        Contrucción de Circuito de Medición correspondiente a una sola cadena de Pauli no trivial..
        '''

        soporte=uccsd.Soporte(PauliChain,num_spin_orbitals) 
        qc=QuantumCircuit(num_spin_orbitals,len(soporte))
    

        qc.compose(HF_state,inplace=True)
        qc.barrier()   


        qc.compose(anzats,inplace=True)
        qc.barrier()  

        qc.compose(pre_measurement(PauliChain, num_spin_orbitals), inplace=True) 

        for i in range(len(soporte)):
            qc.measure(soporte[i],i)

        return qc, soporte





def Generar_Circuitos_Medicion_Hamiltoniano(
    Hamiltoniano: dict,
    HF_state:QuantumCircuit,
    anzats: QuantumCircuit,
    num_spin_orbitals: int
):
        
        '''
        Generaremos una lista de Circuirtos cuantos para  cada theta especifico 
        '''

        Circuitos_NT=[]
        Coeficientes_NT=[]
        Cadenas_Pauli_NT=[]
        Coeficientes_T=[] 

        Trivial_Chain=tuple('I' for _ in range(num_spin_orbitals))

        for PauliCHain, coef in Hamiltoniano.items(): 
            if PauliCHain != Trivial_Chain:
                Circuitos_NT.append( Circuito_Medicion_PauliChainH(PauliCHain,HF_state,anzats,num_spin_orbitals)[0])
                Coeficientes_NT.append(coef)
                Cadenas_Pauli_NT.append(PauliCHain)
            else:
                 Coeficientes_T.append(coef)

        return Circuitos_NT,Coeficientes_NT,Cadenas_Pauli_NT,Coeficientes_T
             
            
def Obtener_Backend(
    tipo: Literal["simulador", "hardware"],
    backend_name: str | None = None
):
    if tipo not in ("simulador", "hardware"):
        raise ValueError("Tipo de medición del circuito invalida.")

    if tipo == "simulador":
        return AerSimulator()

    # Hardware real
    service = QiskitRuntimeService()

    if backend_name is not None:
        return service.backend(backend_name)

    return service.least_busy(operational=True, simulator=False)
     


def Ejecutar_QCircuit_1point(circuitos,backend,shots:int):

    qc_t = transpile(circuitos, backend) #circuito adaptado al backend 

    # Ejecutar el circuito con cierto número de shots.
    job = backend.run(qc_t, shots=shots)
    result = job.result()
    counts_list = result.get_counts()

    return counts_list



def PostProcesado_1point(counts_list, Coeficientes_NT,Coeficientes_T): 
        E=0

        for j in range(len(Coeficientes_T)):
            E+=Coeficientes_T[j]

        '''
        Calcular número total de shots realmente obtenidos
        Nesesario por que, en algunos casos , simuladores o Hardware real cuantico
        descarta algunos shots
        
        '''
        n=0
        for counts in counts_list: 
            expectation=0
            total_shots =  sum(counts.values())

            for bitstring, veces in counts.items():
                #Corregimos el orden del bitstring para ajustarse a nuestra convencion.
                bits =  bitstring[::-1] 
                numero_unos =0

                for i in range(len(bits)):
                    if bits[i]=='1':
                        numero_unos +=1

                Lambda= (-1)**(numero_unos)
                Probabilidad=(veces)/total_shots

                expectation += Probabilidad*Lambda

            
            contribucion = Coeficientes_NT[n]* expectation
            n+=1
            E += contribucion 

        return E



    


def EvaluarPuntos_VQE():
    ...
        



















