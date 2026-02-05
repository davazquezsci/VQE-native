from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit 



# ión HeH+

def hamiltoniano_heh(distancia: float):
    """
    Genera el operador Hamiltoniano en segunda cuantización para la molécula HeH+
    a una distancia interatómica específica.

    Args:
        distancia (float): Distancia entre el átomo de He y H en Angstroms.

    Returns:
        FermionicOp: El hamiltoniano en segunda cuantización.
    """
    
    # Usamos un f-string para insertar la variable 'distancia' dinámicamente
    # La estructura es: "Átomo X Y Z; Átomo X Y Z"
    atom_config = f"He 0 0 0; H 0 0 {distancia}"

    driver = PySCFDriver(
        atom=atom_config, 
        unit=DistanceUnit.ANGSTROM,
        charge=1,  # Carga neta +1
        spin=0,    # 0 electrones desapareados (Singlete)
        basis="sto-3g"
    )

    problem = driver.run()
    
    # Extraemos el Hamiltoniano fermiónico
    fermionic_op = problem.hamiltonian.second_q_op()
    
    return fermionic_op