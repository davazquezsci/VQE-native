from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

print("Backends reales accesibles:\n")
for backend in service.backends(simulator=False):
    try:
        status = backend.status()
        print(f"{backend.name:20} operational={status.operational} pending_jobs={status.pending_jobs}")
    except Exception:
        print(backend.name)