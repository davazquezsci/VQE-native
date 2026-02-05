from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit 

def MolGeoMtric(r):
    
    # ión HeH+
    driver = PySCFDriver(
        atom="He 0 0 0; H 0 0 r",  # Coordenadas atómicas en Angstroms
        unit=DistanceUnit.ANGSTROM,    # Define la unidad de distancia

        charge=1,  # Carga total del sistema (HeH+ tiene carga +1)
        spin=0,    # REQUEIERE MAS ARGUMENTACION !!!!!
        basis="sto-3g"  # Base de funciones de onda
        
    )


    problem = driver.run()  


    fermionic_op = problem.hamiltonian.second_q_op() # Obtiene hamiltoniano Fermionico de segunda Cuantizacion 

    # Imprimir el Hamiltoniano
    print(fermionic_op)