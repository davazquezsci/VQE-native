# Importar bibliotecas
import sys
import numpy as np
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from vqe_native.chemistry import pyscf_backend
from vqe_native.mapping import jordan_wigner
from vqe_native.circuits import Cluster_Operator
from vqe_native.circuits import uccsd
from vqe_native.estimation import measurement_NA
from vqe_native.estimation import measurement_A
from vqe_native.optim import cobyla


def initial(distancia_atomica: float):
    """
    ------------------------------------------------
    Obtener Hamiltoniano y estado inicial
    ------------------------------------------------
    """

    dataMol = pyscf_backend.hamiltoniano_heh(distancia_atomica)

    fermionic_op = dataMol["fermionic_op"]
    num_spin_orbitals = 2 * dataMol["num_spatial_orbitals"]
    num_electrones = dataMol["num_electrones"]
    E_Repulsion = dataMol["nuclear_repulsion_energy"]
    E_total = dataMol["reference_energy"]

    HamOPJW = jordan_wigner.JWdic(fermionic_op, num_spin_orbitals)
    HamOPJW = measurement_A.Preprocesamiento_hamiltoniano(HamOPJW)

    HF_state = uccsd.Estado_HF(num_spin_orbitals, num_electrones)

    return {
        "HamOPJW": HamOPJW,
        "HF_state": HF_state,
        "num_spin_orbitals": num_spin_orbitals,
        "num_electrones": num_electrones,
        "E_Repulsion": E_Repulsion,
        "E_total": E_total,
    }


def one_EP(
    distancia_atomica: float,
    tipo: Literal["simulador", "hardware"],
    backend_o_nombre=None,
    Shots: int = 10000,
    theta_inicial=None,
    FO=cobyla.funcion_objetivo_Agrupada
):
    """
    ------------------------------------------------
    Ejecutar una evaluación de energía para una
    distancia internuclear dada
    ------------------------------------------------
    """

    # ------------------------------------------------
    # Obtener datos moleculares e iniciales
    # ------------------------------------------------
    datos = initial(distancia_atomica)

    HamOPJW = datos["HamOPJW"]
    HF_state = datos["HF_state"]
    num_spin_orbitals = datos["num_spin_orbitals"]
    num_electrones = datos["num_electrones"]
    E_Repulsion = datos["E_Repulsion"]
    E_total = datos["E_total"]

    # ------------------------------------------------
    # Obtener theta inicial
    # ------------------------------------------------
    if theta_inicial is None:
        Theta_0 = Cluster_Operator.Theta_Inicial(
            num_spin_orbitals,
            num_electrones
        )
    else:
        Theta_0 = np.array(theta_inicial, dtype=float)

    # ------------------------------------------------
    # Inicializar backend
    # ------------------------------------------------
    if hasattr(backend_o_nombre, "run"):
        backend = backend_o_nombre
    else:
        backend = measurement_NA.Obtener_Backend(tipo, backend_o_nombre)

    # ------------------------------------------------
    # Evaluar función objetivo
    # ------------------------------------------------
    E = FO(
        Theta_0,
        HamOPJW,
        HF_state,
        num_spin_orbitals,
        num_electrones,
        backend,
        Shots
    )

    # ------------------------------------------------
    # Energía total y diferencia
    # ------------------------------------------------
    E_TOT = E + E_Repulsion
    Dif_E_TOT = abs(E_TOT - E_total)

    return E, E_TOT, Dif_E_TOT, E_total