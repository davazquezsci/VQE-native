from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import StatevectorEstimator

import numpy as np

import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def HeHplus_Qiskit_Exact(R_AB: float):
    # =========================================================
    # 1. Problema molecular
    # =========================================================

    driver = PySCFDriver(
        atom=f"He 0 0 0; H 0 0 {R_AB}",
        unit=DistanceUnit.ANGSTROM,
        charge=1,
        spin=0,
        basis="sto-3g",
    )

    problem = driver.run()


    # =========================================================
    # 2. Energías Hartree-Fock
    # =========================================================

    E_HF_total = problem.reference_energy
    E_nuc = problem.hamiltonian.nuclear_repulsion_energy
    E_HF_electronica = E_HF_total - E_nuc



    # =========================================================
    # 3. Operadores del problema
    # =========================================================

    mapper = JordanWignerMapper()

    main_op, aux_ops = problem.second_q_ops()

    qubit_op = mapper.map(main_op)
    qubit_aux_ops = {name: mapper.map(op) for name, op in aux_ops.items()}


    # =========================================================
    # 4. Ansatz UCCSD + estado inicial Hartree-Fock
    # =========================================================

    initial_state = HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
    )


    # =========================================================
    # 5. VQE
    # =========================================================

    estimator = StatevectorEstimator()
    optimizer = SLSQP(maxiter=1000)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    vqe.initial_point = np.zeros(ansatz.num_parameters)

    vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)

    E_VQE_electronica = vqe_result.eigenvalue.real
    E_VQE_total = E_VQE_electronica + E_nuc


    # =========================================================
    # 6. Valor exacto de referencia CORRECTO
    # =========================================================

    numpy_solver = NumPyMinimumEigensolver(
        filter_criterion=problem.get_default_filter_criterion()
    )

    exact_result = numpy_solver.compute_minimum_eigenvalue(
        qubit_op,
        aux_operators=qubit_aux_ops,
    )

    E_exacta_electronica = exact_result.eigenvalue.real
    E_exacta_total = E_exacta_electronica + E_nuc


    # =========================================================
    # 7. Resultados
    # =========================================================
    '''

    print("\n================ RESULTADOS ================\n")
    print("HF total:                ", E_HF_total)
    print("VQE total:               ", E_VQE_total)
    print("Exacta total:            ", E_exacta_total)
    print("Error VQE:               ", E_VQE_total - E_exacta_total)   
    print("Corr. electrónica HF:    ", E_HF_total - E_exacta_total)
    print("Parámetros del ansatz:   ", ansatz.num_parameters)
    '''

    return E_VQE_total,E_exacta_total