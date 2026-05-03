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
from vqe_native.optim import cobyla 
from vqe_native.vqe import vqe
from qiskit_data import qiskit_pipeline
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
 
nombre_corrida = "GFK sim har e id 100000 range 88-94"