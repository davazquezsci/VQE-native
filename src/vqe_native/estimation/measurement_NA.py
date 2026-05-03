from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
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
        Contrucción de Circuito de Medición correspondiente a una sola cadena de Pauli no trivial...
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
             
            
     
def obtener_backend(modo: str, backend_name: str = "ibm_kingston"):
    """
    Devuelve el backend adecuado para:
    - simulador: ideal
    - simulador_ruidoso: AerSimulator con noise model del backend IBM
    - hardware: backend real de IBM
    """
    modo = modo.lower().strip()

    if modo == "simulador":
        return AerSimulator()

    if modo == "hardware":
        service = QiskitRuntimeService()
        return service.backend(backend_name)

    if modo == "simulador_ruidoso":
        try:
            service = QiskitRuntimeService()
            backend_real = service.backend(backend_name)

            # Esto configura noise model + basis gates + coupling map
            backend_ruidoso = AerSimulator.from_backend(backend_real)
            backend_ruidoso.set_options(method="density_matrix")
            return backend_ruidoso

        except Exception:
            # Respaldo si no tienes acceso al servicio o falla la conexión
            if FakeSherbrooke is None:
                raise RuntimeError(
                    "No se pudo acceder al backend real y tampoco hay fake backend disponible."
                )

            fake_backend = FakeSherbrooke()
            return AerSimulator.from_backend(fake_backend)

    raise ValueError(f"Modo no reconocido: {modo}")


def Ejecutar_QCircuit_1point(circuitos, backend, shots: int):
    global TOTAL_CIRCUITOS
    global TOTAL_SHOTS
    global DIAGNOSTICO_HECHO 

    num_circuitos = len(circuitos)

    TOTAL_CIRCUITOS += num_circuitos
    TOTAL_SHOTS += num_circuitos * shots

    qc_t = transpile(
        circuitos,
        backend=backend,
        optimization_level=1,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=1234
    )

    if not DIAGNOSTICO_HECHO:
        diagnostico_circuitos(qc_t, etapa="después de transpilar")
        DIAGNOSTICO_HECHO = True
    '''
    # Diagnóstico útil
    if isinstance(qc_t, list):
        print("\n--- Diagnóstico de circuitos transpileados ---")
        for i, qc in enumerate(qc_t[:3]):  # muestra solo los primeros 3
            print(f"Circuito {i}: depth={qc.depth()}, ops={qc.count_ops()}")
    else:
        print("\n--- Diagnóstico de circuito transpileado ---")
        print(f"depth={qc_t.depth()}, ops={qc_t.count_ops()}")
    '''
    # Caso simulador local
    if isinstance(backend, AerSimulator):
        job = backend.run(qc_t, shots=shots)
        result = job.result()
        counts_list = result.get_counts()
        if isinstance(counts_list, dict):
            counts_list = [counts_list]
        return counts_list

    # Caso hardware IBM Runtime
    sampler = Sampler(mode=backend)
    job = sampler.run(qc_t, shots=shots)
    result = job.result()

    counts_list = []
    for pub_result in result:
        data_bin = pub_result.data
        reg_name = list(data_bin.keys())[0]
        counts = getattr(data_bin, reg_name).get_counts()
        counts_list.append(counts)

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


def diagnostico_circuitos(circuitos, etapa=""):
    """
    Imprime métricas básicas de una lista de circuitos cuánticos.
    """

    if not isinstance(circuitos, list):
        circuitos = [circuitos]

    depths = [qc.depth() for qc in circuitos]
    sizes = [qc.size() for qc in circuitos]

    print(f"\n--- Diagnóstico de circuitos: {etapa} ---")
    print(f"Número de circuitos: {len(circuitos)}")
    print(f"Depth promedio: {sum(depths) / len(depths):.2f}")
    print(f"Depth máximo: {max(depths)}")
    print(f"Size promedio: {sum(sizes) / len(sizes):.2f}")
    print(f"Size máximo: {max(sizes)}")

    print("\nOperaciones de los primeros circuitos:")
    for i, qc in enumerate(circuitos[:3]):
        print(f"Circuito {i}: depth={qc.depth()}, size={qc.size()}, ops={qc.count_ops()}")
    


        



















