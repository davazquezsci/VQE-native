from qiskit import QuantumCircuit
import numpy as np


def pre_mesurment(op,num_spin_orbitals):
    qc=QuantumCircuit(num_spin_orbitals)
    for i in range(num_spin_orbitals):
        if op[i]=='X':
            qc.h(i)
        elif op[i]=='Y':
            qc.sdg(i)
            qc.h(i)
    return qc


def Medicion_Hamiltoniano(Hamiltoniano,num_spin_orbitals):
    qc=QuantumCircuit(num_spin_orbitals)















    return E 