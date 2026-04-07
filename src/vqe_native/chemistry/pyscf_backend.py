"""
    Genera el operador Hamiltoniano en segunda cuantización para la molécula HeH+
    a una distancia interatómica específica.

    Args:
        R_0 (float): Distancia entre el átomo de He y H en Angstroms.

    Returns:
        FermionicOp: El hamiltoniano en segunda cuantización.
        
"""
    
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit 
import re


def block_to_interleaved_index(p: int, num_spatial_orbitals: int) -> int:
    """
    Convierte un índice de spin-orbital en orden block:
        alfa0, alfa1, ..., alfa_{n-1}, beta0, beta1, ..., beta_{n-1}
    al orden intercalado:
        alfa0, beta0, alfa1, beta1, ..., alfa_{n-1}, beta_{n-1}
    """
    n = num_spatial_orbitals

    if p < 0 or p >= 2 * n:
        raise ValueError(
            f"Índice fuera de rango: p={p}, pero se esperaban índices entre 0 y {2*n - 1}."
        )

    if p < n:
        # alpha_k -> 2k
        return 2 * p
    else:
        # beta_k -> 2k+1
        k = p - n
        return 2 * k + 1


def reindex_fermionic_term(term: str, num_spatial_orbitals: int) -> str:
    """
    Reindexa un término fermiónico, por ejemplo:
        '+_1 -_0'  -> '+_2 -_0'   (si n=2)
    conservando el orden de operadores.
    """
    tokens = term.split()
    new_tokens = []

    for tok in tokens:
        m = re.fullmatch(r'([+-])_(\d+)', tok)
        if not m:
            raise ValueError(f"Término inválido en operador fermiónico: '{tok}'")

        sign = m.group(1)
        old_idx = int(m.group(2))
        new_idx = block_to_interleaved_index(old_idx, num_spatial_orbitals)
        new_tokens.append(f"{sign}_{new_idx}")

    return " ".join(new_tokens)


def reindex_fermionic_dict_block_to_interleaved(
    fermionic_dict: dict,
    num_spatial_orbitals: int,
    tol: float = 1e-12
) -> dict:
    """
    Reindexa todas las claves de un diccionario de términos fermiónicos.
    Si dos términos distintos colapsan en la misma clave reindexada,
    sus coeficientes se suman.
    """
    out = {}

    for term, coef in fermionic_dict.items():
        new_term = reindex_fermionic_term(term, num_spatial_orbitals)

        if new_term in out:
            out[new_term] += coef
        else:
            out[new_term] = coef

    return {k: v for k, v in out.items() if abs(v) > tol}

def hamiltoniano_heh(R_AB: float):

    driver = PySCFDriver(
        atom=f"He 0 0 0; H 0 0 {R_AB}", 
        unit=DistanceUnit.ANGSTROM,
        charge=1,  # Carga neta +1
        spin=0,    # 0 electrones desapareados (Singlete)
        basis="sto-3g"
    ) 
    problem = driver.run()

    fermionic_op_block = problem.hamiltonian.second_q_op()

    fermionic_dict_interleaved = reindex_fermionic_dict_block_to_interleaved(
        fermionic_op_block,
        problem.num_spatial_orbitals
    )

    data = {
        "problem": problem,
        "fermionic_op": fermionic_dict_interleaved,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "num_particles": problem.num_particles,
        "num_electrones": sum(problem.num_particles),
        "nuclear_repulsion_energy": problem.nuclear_repulsion_energy,
        "reference_energy": problem.reference_energy,
    }

    return data