# comparar_diccionarios_pauli.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from vqe_native.chemistry import pyscf_backend
from vqe_native.mapping import jordan_wigner
from vqe_native.estimation import measurement_A


def contar_no_triviales(Hamiltoniano: dict, num_spin_orbitals: int):
    """
    Cuenta las cadenas de Pauli no triviales del Hamiltoniano.
    La cadena identidad no requiere medición, por eso se excluye.
    """

    cadena_identidad = tuple("I" for _ in range(num_spin_orbitals))

    return sum(
        1 for cadena in Hamiltoniano.keys()
        if cadena != cadena_identidad
    )


def comparar_agrupamiento(R_AB):
    """
    Para una distancia R_AB:
    - construye el Hamiltoniano de HeH+
    - lo transforma a cadenas de Pauli
    - cuenta cadenas no agrupadas
    - agrupa cadenas compatibles
    - cuenta grupos
    """

    # Construcción del Hamiltoniano molecular
    dataMol = pyscf_backend.hamiltoniano_heh(R_AB)

    fermionic_op = dataMol["fermionic_op"]
    num_spin_orbitals = 2 * dataMol["num_spatial_orbitals"]

    # Transformación Jordan-Wigner
    HamOPJW = jordan_wigner.JWdic(
        fermionic_op,
        num_spin_orbitals
    )

    # Ordenamiento previo usado por tu agrupamiento
    HamOPJW = measurement_A.Preprocesamiento_hamiltoniano(HamOPJW)

    # Número de cadenas no triviales originales
    n_original = contar_no_triviales(
        HamOPJW,
        num_spin_orbitals
    )

    # Hamiltoniano agrupado
    grupos, coeficientes_triviales = measurement_A.Agrupar_Cadenas_Pauli(
        HamOPJW,
        num_spin_orbitals
    )

    # Número de grupos
    n_agrupado = len(grupos)

    reduccion = n_original - n_agrupado

    reduccion_porcentual = 100 * reduccion / n_original

    return {
        "R_AB": R_AB,
        "num_cadenas_original": n_original,
        "num_grupos_agrupados": n_agrupado,
        "reduccion_absoluta": reduccion,
        "reduccion_porcentual": reduccion_porcentual
    }


if __name__ == "__main__":

    # Puedes cambiar estos valores
    R_values = np.linspace(0.6, 1.6, 100)

    resultados = []

    for R_AB in R_values:
        resultado = comparar_agrupamiento(R_AB)
        resultados.append(resultado)

        print(
            f"R_AB = {R_AB:.4f} Å | "
            f"original = {resultado['num_cadenas_original']} | "
            f"agrupado = {resultado['num_grupos_agrupados']} | "
            f"reducción = {resultado['reduccion_porcentual']:.2f}%"
        )

    df = pd.DataFrame(resultados)

    df.to_csv("comparacion_numero_cadenas_pauli.csv", index=False)

    print("\nArchivo guardado:")
    print("comparacion_numero_cadenas_pauli.csv")

    print("\nResumen promedio:")
    print(f"Cadenas originales promedio: {df['num_cadenas_original'].mean():.2f}")
    print(f"Grupos agrupados promedio: {df['num_grupos_agrupados'].mean():.2f}")
    print(f"Reducción porcentual promedio: {df['reduccion_porcentual'].mean():.2f}%")