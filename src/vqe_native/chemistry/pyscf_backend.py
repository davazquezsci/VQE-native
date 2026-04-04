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

def hamiltoniano_heh(R_AB: float):

    driver = PySCFDriver(
        atom=f"He 0 0 0; H 0 0 {R_AB}", 
        unit=DistanceUnit.ANGSTROM,
        charge=1,  # Carga neta +1
        spin=0,    # 0 electrones desapareados (Singlete)
        basis="sto-3g"
    ) 
    problem = driver.run()
    
    #Extrahemos toda la informacion nesesaria de PySCF
    data = {
        "problem": problem,
        "fermionic_op": problem.hamiltonian.second_q_op(),
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "num_particles": problem.num_particles,
        "nuclear_repulsion_energy": problem.nuclear_repulsion_energy,
        "reference_energy": problem.reference_energy,
    }
    return data