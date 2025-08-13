"""
Quantum backend providers (Qiskit, Braket, Sim/local), cloud config, retry/backoff, device logic.
"""

class SimProvider:
    def run_ae(self, oracles, shots, epsilon, mode):
        # Classical MC estimation of means (simulate quantum query)
        return [oracle() for oracle in oracles]


class BraketProvider:
    def __init__(self):
        self._initialized = False
        # Lazy import braket SDK
    def run_ae(self, oracles, shots, epsilon, mode):
        # Placeholder: batch AE circuits, submit jobs, handle polling
        raise NotImplementedError("AWS Braket provider integration required.")

import os
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_provider import IBMProvider as QiskitIBMProvider
from qiskit_ibm_provider import least_busy
from qiskit_aer import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import AmplitudeEstimation, EstimationProblem

class IBMProvider:
    def __init__(self):
        self._initialized = False
        self.provider = None
        self.backend = None
        self._authenticate()

    def _authenticate(self):
        token = os.environ.get('IBM_QUANTUM_TOKEN')
        if token is None:
            raise RuntimeError("IBM_QUANTUM_TOKEN environment variable not set. Please provide your IBM Quantum API token.")
        self.provider = QiskitIBMProvider(token=token)
        # Choose a real device (preferring open/free devices)
        backends = self.provider.backends(filters=lambda b: b.configuration().n_qubits >= 2 and not b.configuration().simulator)
        if not backends:
            raise RuntimeError("No suitable IBM Quantum hardware backend found.")
        self.backend = least_busy(backends)
        self._initialized = True

    def run_ae(self, oracles, shots, epsilon, mode):
        # For demonstration: each oracle is a function returning a Qiskit circuit and measurement, or just amplitude
        results = []
        for oracle in oracles:
            qc, objective_qubit = oracle()  # Each oracle should build a QuantumCircuit and specify measurement qubit
            problem = EstimationProblem(
                state_preparation=qc,
                objective_qubits=[objective_qubit]
            )
            ae = AmplitudeEstimation(epsilon, QuantumInstance(self.backend, shots=shots))
            result = ae.estimate(problem)
            results.append(result.estimation)
        return results

_provider_map = {
    'sim': SimProvider,
    'ibm': IBMProvider,
    'braket': BraketProvider,
}

def get_provider(backend, strict_local=False):
    if strict_local or backend == 'sim':
        return SimProvider()
    if backend == 'ibm':
        return IBMProvider()
    if backend == 'braket':
        return BraketProvider()
    raise ValueError(f"Unknown backend {backend}")
