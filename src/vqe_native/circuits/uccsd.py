#Importar bibliotecas.
from qiskit import QuantumCircuit
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


def Estado_HF(num_spin_orbitals,num_electrones):
    qc=QuantumCircuit(num_spin_orbitals)
    for i in range(num_electrones):
        qc.x(i)
    return qc


def Soporte(op,num_spin_orbitals):
    soporte=[]
    for i in range(num_spin_orbitals):
        if op[i]!='I':
            soporte.append(i)
    return soporte


def B(op,num_spin_orbitals):
    qc=QuantumCircuit(num_spin_orbitals)
    for i in range(num_spin_orbitals):
        if op[i]=='X':
            qc.h(i)
        elif op[i]=='Y':
            qc.sdg(i)
            qc.h(i)


    return qc 

def B_dag(op,num_spin_orbitals):
    qc=QuantumCircuit(num_spin_orbitals)
    for i in range(num_spin_orbitals):
        if op[i]=='X':
            qc.h(i)
        elif op[i]=='Y':
            qc.h(i)
            qc.s(i)

    return qc 

def CnotChain(op,num_spin_orbitals):
     qc=QuantumCircuit(num_spin_orbitals)
     soporte=Soporte(op,num_spin_orbitals)
     for i in range(len(soporte)-1):
        qc.cx(soporte[i],soporte[i+1] )
     return qc 
                     
def CnotChain_dag(op,num_spin_orbitals):
     qc=QuantumCircuit(num_spin_orbitals)
     soporte=Soporte(op,num_spin_orbitals)
     for i in range(len(soporte)-1):
         qc.cx(soporte[len(soporte)-i-2],soporte[len(soporte)-i-1] )
     return qc 


def ExtraerTheta(coef, tol=1e-12):
    if abs(coef.real) > tol:
        raise ValueError(
            f"El coeficiente {coef} no es puramente imaginario."
        )

    theta = 1j * coef

    if abs(theta.imag) > tol:
        raise ValueError(
            f"No se pudo extraer un theta real a partir de {coef}."
        )

    return float(theta.real)

def D(op,theta:float,num_spin_orbitals): 
    qc=QuantumCircuit(num_spin_orbitals)
    soporte=Soporte(op,num_spin_orbitals)
    qc.rz(2*theta,soporte[len(soporte)-1])      
    return qc
             




def QuantumExp(num_spin_orbitals: int, Op_Cluster_PAULI: dict):
    qc=QuantumCircuit(num_spin_orbitals)



    for op, coef in Op_Cluster_PAULI.items():
        theta = ExtraerTheta(coef)

        qc.compose(B(op, num_spin_orbitals), inplace=True)
        qc.barrier()
        qc.compose(CnotChain(op, num_spin_orbitals), inplace=True)
        qc.barrier()
        qc.compose(D(op, theta, num_spin_orbitals), inplace=True)
        qc.barrier()
        qc.compose(CnotChain_dag(op, num_spin_orbitals), inplace=True)
        qc.barrier()
        qc.compose(B_dag(op, num_spin_orbitals), inplace=True)
        qc.barrier()

    return qc